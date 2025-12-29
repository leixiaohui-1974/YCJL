"""
水泵执行器模型
==============

水轮发电机/水泵仿真:
- 启停动力学
- 流量特性曲线
- 转速控制
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto

from ..config.settings import Config


class PumpStatus(Enum):
    """水泵状态"""
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    FAULT = auto()


@dataclass
class PumpState:
    """水泵状态"""
    status: PumpStatus     # 运行状态
    speed: float           # 转速 (rpm)
    flow: float            # 流量 (m³/s)
    head: float            # 扬程 (m)
    power: float           # 功率 (kW)
    efficiency: float      # 效率
    is_fault: bool         # 故障标志


class PumpActuator:
    """
    水泵/水轮机执行器仿真

    特性:
    - 启动/停止惯性
    - H-Q特性曲线
    - 效率曲线
    - 甩负荷响应
    """

    def __init__(self, name: str, rated_flow: float = 15.0,
                 rated_head: float = 50.0):
        self.name = name

        # 额定参数
        self.rated_flow = rated_flow       # 额定流量 (m³/s)
        self.rated_head = rated_head       # 额定扬程 (m)
        self.rated_speed = 300.0           # 额定转速 (rpm)
        self.rated_power = 500.0           # 额定功率 (kW)

        # 状态
        self.status = PumpStatus.STOPPED
        self.speed = 0.0
        self.target_speed = 0.0
        self.flow = 0.0
        self.head = 0.0
        self.power = 0.0

        # 惯性
        self.inertia_time = 30.0           # 机组惯性时间常数 (s)
        self.start_time = 60.0             # 启动时间 (s)
        self.stop_time = 90.0              # 停机时间 (s)

        # 效率
        self.peak_efficiency = 0.88
        self.efficiency = 0.0

        # 故障
        self.is_fault = False

        # 历史
        self.history = []

    def _compute_head(self, flow: float, speed: float) -> float:
        """
        计算扬程 (相似定律)

        H = H_rated * (n/n_rated)² * (1 - (Q/Q_rated)²)
        """
        if speed < 1.0:
            return 0.0

        speed_ratio = speed / self.rated_speed
        flow_ratio = flow / max(self.rated_flow, 0.1)

        # 简化H-Q曲线
        H = self.rated_head * (speed_ratio ** 2) * (1 - 0.3 * (flow_ratio ** 2))
        return max(H, 0)

    def _compute_flow(self, speed: float, system_head: float) -> float:
        """
        计算流量 (泵-管特性交点)

        简化：假设流量与转速成正比
        """
        if speed < 10.0:
            return 0.0

        speed_ratio = speed / self.rated_speed
        Q = self.rated_flow * speed_ratio

        # 系统扬程限制
        H_pump = self._compute_head(Q, speed)
        if H_pump < system_head:
            Q *= np.sqrt(max(H_pump / system_head, 0))

        return Q

    def _compute_efficiency(self, flow: float) -> float:
        """
        计算效率

        效率曲线：在额定点附近最高
        """
        if flow < 0.1 or self.speed < 10:
            return 0.0

        flow_ratio = flow / self.rated_flow

        # 抛物线效率曲线
        eta = self.peak_efficiency * (1 - 0.5 * (flow_ratio - 1) ** 2)
        return max(eta, 0.1)

    def _compute_power(self, flow: float, head: float, efficiency: float) -> float:
        """计算功率"""
        if efficiency < 0.01:
            return 0.0

        # P = ρgQH / η
        rho = Config.physics.RHO
        g = Config.physics.G
        return rho * g * flow * head / (efficiency * 1000)  # kW

    def start(self):
        """启动"""
        if self.status == PumpStatus.STOPPED:
            self.status = PumpStatus.STARTING
            self.target_speed = self.rated_speed

    def stop(self):
        """停机"""
        if self.status == PumpStatus.RUNNING:
            self.status = PumpStatus.STOPPING
            self.target_speed = 0.0

    def set_speed(self, speed: float):
        """设置转速"""
        self.target_speed = np.clip(speed, 0, self.rated_speed * 1.1)

    def step(self, dt: float, system_head: float = 0.0) -> PumpState:
        """
        推进一个时间步

        Parameters:
            dt: 时间步长 (s)
            system_head: 系统扬程需求 (m)

        Returns:
            PumpState: 当前状态
        """
        # 故障检查
        if self.is_fault:
            self.status = PumpStatus.FAULT
            return self._get_state()

        # 转速动力学
        if self.status == PumpStatus.STARTING:
            tau = self.start_time / 3
            alpha = dt / (tau + dt)
            self.speed = (1 - alpha) * self.speed + alpha * self.target_speed

            if self.speed >= self.target_speed * 0.95:
                self.status = PumpStatus.RUNNING

        elif self.status == PumpStatus.STOPPING:
            tau = self.stop_time / 3
            alpha = dt / (tau + dt)
            self.speed = self.speed * (1 - alpha)

            if self.speed < 5.0:
                self.speed = 0.0
                self.status = PumpStatus.STOPPED

        elif self.status == PumpStatus.RUNNING:
            tau = self.inertia_time
            alpha = dt / (tau + dt)
            self.speed = (1 - alpha) * self.speed + alpha * self.target_speed

        # 计算流量、扬程
        self.flow = self._compute_flow(self.speed, system_head)
        self.head = self._compute_head(self.flow, self.speed)
        self.efficiency = self._compute_efficiency(self.flow)
        self.power = self._compute_power(self.flow, self.head, self.efficiency)

        state = self._get_state()
        self.history.append(state)
        return state

    def _get_state(self) -> PumpState:
        """获取当前状态"""
        return PumpState(
            status=self.status,
            speed=self.speed,
            flow=self.flow,
            head=self.head,
            power=self.power,
            efficiency=self.efficiency,
            is_fault=self.is_fault
        )

    def emergency_stop(self):
        """紧急停机"""
        self.target_speed = 0.0
        self.status = PumpStatus.STOPPING

    def inject_fault(self):
        """注入故障"""
        self.is_fault = True
        self.status = PumpStatus.FAULT

    def clear_fault(self):
        """清除故障"""
        self.is_fault = False
        if self.speed > 0:
            self.status = PumpStatus.RUNNING
        else:
            self.status = PumpStatus.STOPPED

    def reset(self):
        """重置"""
        self.status = PumpStatus.STOPPED
        self.speed = 0.0
        self.target_speed = 0.0
        self.flow = 0.0
        self.head = 0.0
        self.power = 0.0
        self.efficiency = 0.0
        self.is_fault = False
        self.history.clear()
