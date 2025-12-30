"""
阀门执行器模型
==============

调流调压阀电动执行器仿真:
- 线性行程特性
- 空蚀保护
- 无扰切换
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
from enum import Enum, auto

from ..config.settings import Config


class ValveStatus(Enum):
    """阀门状态"""
    STOPPED = auto()
    OPENING = auto()
    CLOSING = auto()
    FULLY_OPEN = auto()
    FULLY_CLOSED = auto()
    REGULATING = auto()


@dataclass
class ValveState:
    """阀门执行器状态"""
    position: float        # 当前位置 (0~1)
    target: float          # 目标位置
    velocity: float        # 速度 (%/s)
    status: ValveStatus    # 状态
    torque: float          # 力矩 (Nm)
    is_cavitating: bool    # 空蚀状态
    is_fault: bool         # 故障标志


class ValveActuator:
    """
    阀门电动执行器仿真

    特性:
    - 电动推杆特性
    - 行程时间控制
    - 力矩限制
    - 空蚀保护
    """

    def __init__(self, name: str, stroke_time: float = 60.0):
        self.name = name
        self.stroke_time = stroke_time  # 全行程时间 (s)

        # 状态
        self.position = 0.5            # 当前位置 (0~1)
        self.target = 0.5              # 目标位置
        self.velocity = 0.0            # 当前速度

        # 速率参数
        self.max_rate = 1.0 / stroke_time  # 最大速率 (%/s)
        self.min_rate = 0.001              # 最小调节速率

        # 死区
        self.dead_band = 0.002         # 位置死区

        # 力矩
        self.max_torque = 1000.0       # 最大力矩 (Nm)
        self.current_torque = 0.0

        # 空蚀保护
        self.cavitation_limit = 0.1    # 空蚀保护开度下限
        self.is_cavitating = False

        # 故障
        self.is_fault = False
        self.fault_type = None

        # 状态
        self.status = ValveStatus.STOPPED

        # 历史
        self.history = []

    def set_target(self, target: float, rate: float = None):
        """
        设置目标位置

        Parameters:
            target: 目标位置 (0~1)
            rate: 动作速率，如果指定则覆盖默认速率
        """
        # 空蚀保护
        if target < self.cavitation_limit and self.is_cavitating:
            target = self.cavitation_limit

        self.target = np.clip(target, 0.0, 1.0)

        if rate is not None:
            self._override_rate = min(rate, self.max_rate)
        else:
            self._override_rate = None

    def set_cavitation_state(self, is_cavitating: bool):
        """设置空蚀状态"""
        self.is_cavitating = is_cavitating

        # 如果检测到空蚀，强制限制开度
        if is_cavitating and self.target < self.cavitation_limit:
            self.target = self.cavitation_limit

    def step(self, dt: float, differential_pressure: float = 0.0) -> ValveState:
        """
        推进一个时间步

        Parameters:
            dt: 时间步长 (s)
            differential_pressure: 阀门前后压差 (m)

        Returns:
            ValveState: 当前状态
        """
        # 故障检查
        if self.is_fault:
            self.velocity = 0.0
            self.status = ValveStatus.STOPPED
            return self._get_state()

        # 计算误差
        error = self.target - self.position

        # 死区判断
        if abs(error) < self.dead_band:
            self.velocity = 0.0
            if self.position > 0.99:
                self.status = ValveStatus.FULLY_OPEN
            elif self.position < 0.01:
                self.status = ValveStatus.FULLY_CLOSED
            else:
                self.status = ValveStatus.STOPPED
            return self._get_state()

        # 确定速率
        rate = self._override_rate if hasattr(self, '_override_rate') and self._override_rate else self.max_rate

        # 计算力矩 (简化模型)
        # 力矩与压差成正比
        base_torque = 100.0
        pressure_torque = differential_pressure * 5.0
        self.current_torque = base_torque + pressure_torque

        # 力矩限制
        if self.current_torque > self.max_torque:
            rate *= 0.5  # 高负载时降速

        # 计算实际速率
        if error > 0:
            self.velocity = min(rate, error / dt)
            self.status = ValveStatus.OPENING
        else:
            self.velocity = max(-rate, error / dt)
            self.status = ValveStatus.CLOSING

        # 更新位置
        self.position += self.velocity * dt
        self.position = np.clip(self.position, 0.0, 1.0)

        state = self._get_state()
        self.history.append(state)
        return state

    def _get_state(self) -> ValveState:
        """获取当前状态"""
        return ValveState(
            position=self.position,
            target=self.target,
            velocity=self.velocity,
            status=self.status,
            torque=self.current_torque,
            is_cavitating=self.is_cavitating,
            is_fault=self.is_fault
        )

    def emergency_close(self, fast: bool = True):
        """紧急关闭"""
        self.target = 0.0
        if fast:
            self._override_rate = self.max_rate * 2  # 加速

    def hold_position(self):
        """保持当前位置"""
        self.target = self.position

    def inject_fault(self, fault_type: str):
        """注入故障"""
        self.is_fault = True
        self.fault_type = fault_type

    def clear_fault(self):
        """清除故障"""
        self.is_fault = False
        self.fault_type = None

    def reset(self, position: float = 0.5):
        """重置执行器"""
        self.position = position
        self.target = position
        self.velocity = 0.0
        self.status = ValveStatus.STOPPED
        self.is_fault = False
        self.is_cavitating = False
        self.current_torque = 0.0
        self.history.clear()

# 向后兼容别名
Valve = ValveActuator
