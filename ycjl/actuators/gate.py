"""
闸门执行器模型
==============

弧形闸门液压启闭机仿真:
- 动作速率限制
- 两阶段关闭规律
- 避振穿越逻辑
- 故障模式
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto

from ..config.settings import Config


class GateMode(Enum):
    """闸门运行模式"""
    NORMAL = auto()        # 正常调节
    EMERGENCY_CLOSE = auto()  # 紧急关闭
    MAINTENANCE = auto()   # 检修模式
    FAULT = auto()         # 故障


class GateStatus(Enum):
    """闸门状态"""
    STOPPED = auto()       # 停止
    OPENING = auto()       # 开启中
    CLOSING = auto()       # 关闭中
    FULLY_OPEN = auto()    # 全开
    FULLY_CLOSED = auto()  # 全关


@dataclass
class GateState:
    """闸门状态"""
    position: float        # 当前位置 (0~1)
    target: float          # 目标位置
    velocity: float        # 当前速度 (%/s)
    status: GateStatus     # 状态
    mode: GateMode         # 模式
    motor_current: float   # 电机电流 (A)
    is_fault: bool         # 故障标志


class GateActuator:
    """
    闸门执行器仿真

    特性:
    - 液压启闭机动力学
    - 两阶段关闭规律 (快关-慢关)
    - 避振区穿越
    - 力矩限制
    """

    def __init__(self, name: str, max_opening: float = 6.0):
        self.name = name
        self.max_opening = max_opening

        # 状态
        self.position = 0.0            # 当前位置 (0~1)
        self.target = 0.0              # 目标位置
        self.velocity = 0.0            # 当前速度

        # 速率参数
        self.max_rate_normal = 0.01    # 正常最大速率 (%/s) = 1%/s
        self.max_rate_emergency = 0.03 # 紧急速率 (%/s) = 3%/s

        # 两阶段关闭参数
        self.fast_close_ratio = 0.7    # 快关阶段比例
        self.slow_close_ratio = 0.3    # 慢关阶段比例
        self.fast_close_rate = 0.02    # 快关速率
        self.slow_close_rate = 0.005   # 慢关速率

        # 避振区
        self.vibration_zone = (0.05, 0.15)  # 避振开度区间
        self.rapid_traverse_rate = 0.05     # 快速穿越速率

        # 死区
        self.dead_band = 0.005         # 位置死区

        # 模式
        self.mode = GateMode.NORMAL
        self.status = GateStatus.STOPPED

        # 故障
        self.is_fault = False
        self.fault_type = None

        # 电机参数
        self.motor_current = 0.0
        self.motor_max_current = 100.0
        self.motor_stall_current = 120.0

        # 历史
        self.history = []

    def _get_current_rate(self) -> float:
        """获取当前允许速率"""
        if self.mode == GateMode.EMERGENCY_CLOSE:
            return self.max_rate_emergency
        elif self.mode == GateMode.NORMAL:
            return self.max_rate_normal
        return 0.0

    def _is_in_vibration_zone(self) -> bool:
        """判断是否在避振区"""
        return self.vibration_zone[0] <= self.position <= self.vibration_zone[1]

    def _compute_close_rate(self) -> float:
        """计算关闭速率（两阶段规律）"""
        if self.mode != GateMode.EMERGENCY_CLOSE:
            return self.max_rate_normal

        # 两阶段关闭
        remaining = self.position
        threshold = self.fast_close_ratio

        if remaining > threshold:
            return self.fast_close_rate
        else:
            return self.slow_close_rate

    def _compute_motor_current(self, rate: float) -> float:
        """估算电机电流"""
        base_current = 20.0  # 空载电流
        load_factor = abs(rate) / self.max_rate_normal

        # 下吸力影响 (关闭时增加)
        if self.status == GateStatus.CLOSING:
            load_factor *= 1.5

        return base_current + load_factor * (self.motor_max_current - base_current)

    def set_target(self, target: float):
        """设置目标位置"""
        self.target = np.clip(target, 0.0, 1.0)

    def set_mode(self, mode: GateMode):
        """设置运行模式"""
        self.mode = mode
        if mode == GateMode.EMERGENCY_CLOSE:
            self.target = 0.0

    def step(self, dt: float) -> GateState:
        """
        推进一个时间步

        Parameters:
            dt: 时间步长 (s)

        Returns:
            GateState: 当前状态
        """
        # 故障检查
        if self.is_fault or self.mode == GateMode.FAULT:
            self.velocity = 0.0
            self.status = GateStatus.STOPPED
            return self._get_state()

        # 计算误差
        error = self.target - self.position

        # 死区判断
        if abs(error) < self.dead_band:
            self.velocity = 0.0
            if self.position > 0.99:
                self.status = GateStatus.FULLY_OPEN
            elif self.position < 0.01:
                self.status = GateStatus.FULLY_CLOSED
            else:
                self.status = GateStatus.STOPPED
            return self._get_state()

        # 确定方向和速率
        if error > 0:
            # 开启
            self.status = GateStatus.OPENING
            max_rate = self._get_current_rate()

            # 避振区快速穿越
            if self._is_in_vibration_zone():
                max_rate = self.rapid_traverse_rate
        else:
            # 关闭
            self.status = GateStatus.CLOSING
            max_rate = self._compute_close_rate()

            # 避振区快速穿越
            if self._is_in_vibration_zone():
                max_rate = self.rapid_traverse_rate

        # 计算实际速率
        desired_rate = error / dt
        self.velocity = np.clip(desired_rate, -max_rate, max_rate)

        # 更新位置
        self.position += self.velocity * dt
        self.position = np.clip(self.position, 0.0, 1.0)

        # 计算电机电流
        self.motor_current = self._compute_motor_current(self.velocity)

        # 堵转检测
        if self.motor_current > self.motor_stall_current:
            self.inject_fault('stall')

        state = self._get_state()
        self.history.append(state)
        return state

    def _get_state(self) -> GateState:
        """获取当前状态"""
        return GateState(
            position=self.position,
            target=self.target,
            velocity=self.velocity,
            status=self.status,
            mode=self.mode,
            motor_current=self.motor_current,
            is_fault=self.is_fault
        )

    def get_opening_meters(self) -> float:
        """获取实际开度（米）"""
        return self.position * self.max_opening

    def emergency_close(self):
        """紧急关闭"""
        self.mode = GateMode.EMERGENCY_CLOSE
        self.target = 0.0

    def emergency_stop(self):
        """紧急停止"""
        self.target = self.position

    def inject_fault(self, fault_type: str):
        """注入故障"""
        self.is_fault = True
        self.fault_type = fault_type
        self.mode = GateMode.FAULT

    def clear_fault(self):
        """清除故障"""
        self.is_fault = False
        self.fault_type = None
        self.mode = GateMode.NORMAL

    def reset(self):
        """重置执行器"""
        self.position = 0.0
        self.target = 0.0
        self.velocity = 0.0
        self.mode = GateMode.NORMAL
        self.status = GateStatus.FULLY_CLOSED
        self.is_fault = False
        self.motor_current = 0.0
        self.history.clear()
