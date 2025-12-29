"""
调压塔模型 - 双向阻抗式
=======================

物理特性:
- 质量振荡方程
- 阻抗孔口非对称特性
- 水锤波能量耗散
- 涌浪高度计算
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from ..config.settings import Config


@dataclass
class SurgeTankState:
    """调压塔状态"""
    level: float           # 塔内水位 (m)
    flow_in: float         # 入塔流量 (m³/s)
    impedance_head: float  # 阻抗水头损失 (m)
    boundary_head: float   # 塔底管道边界水头 (m)
    oscillation_period: float  # 振荡周期估计 (s)
    is_rising: bool        # 水位上升中


class SurgeTank:
    """
    双向阻抗式调压塔仿真模型

    工作原理:
    - 入流工况（下游关阀）：水流入塔，阻抗孔产生较大水头损失
    - 出流工况（下游开阀）：水流出塔，阻抗孔产生较小水头损失
    - 形成非对称阻尼，快速平息涌浪
    """

    def __init__(self, config: Optional[Config] = None):
        self.cfg = config.surge_tank if config else Config.surge_tank
        self.physics = Config.physics

        # 几何参数
        self.diameter = self.cfg.diameter
        self.area = self.cfg.area
        self.base_elevation = self.cfg.base_elevation

        # 阻抗参数
        self.R_in = self.cfg.r_inflow    # 入流阻抗系数
        self.R_out = self.cfg.r_outflow  # 出流阻抗系数
        self.impedance_area = self.cfg.impedance_area

        # 状态变量
        self.level = 45.0  # 初始水位 (相对塔底)
        self.flow_in = 0.0  # 入塔流量 (正为入，负为出)
        self.velocity = 0.0  # 塔内水流速度

        # 连接管道参数
        self.pipe_area = Config.pipeline.area
        self.pipe_length = 500.0  # 连接管长度 (m)

        # 振荡参数
        self.prev_level = self.level
        self.oscillation_count = 0

        # 历史记录
        self.history = []

    def compute_impedance_head(self, flow: float) -> float:
        """
        计算阻抗孔水头损失

        h = R * Q * |Q|

        R = (1/(2*g*A^2)) * (K_entry + K_exit + K_friction)
        """
        if abs(flow) < 0.001:
            return 0.0

        # 选择阻抗系数
        R = self.R_in if flow > 0 else self.R_out

        # 转换为水头损失
        # R 定义为无量纲系数，需要转换
        velocity = flow / self.impedance_area
        head_loss = R * (velocity ** 2) / (2 * self.physics.G)

        # 带符号返回
        return head_loss * np.sign(flow)

    def compute_boundary_head(self) -> float:
        """
        计算塔底管道边界水头

        H_boundary = Z_tank + base_elevation + h_impedance
        """
        h_impedance = self.compute_impedance_head(self.flow_in)
        return self.level + self.base_elevation + h_impedance

    def estimate_oscillation_period(self) -> float:
        """
        估算振荡周期

        T = 2π * sqrt(L*A_tank / (g*A_pipe))
        """
        T = 2 * np.pi * np.sqrt(
            self.pipe_length * self.area /
            (self.physics.G * self.pipe_area)
        )
        return T

    def step(self, dt: float, Q_pool_out: float, Q_pipe_in: float) -> SurgeTankState:
        """
        推进一个时间步

        质量振荡方程:
        A_tank * dZ/dt = Q_pool_out - Q_pipe_in

        动量方程 (简化):
        L/gA * dQ/dt = H_pool - Z - h_impedance - H_pipe

        Parameters:
            dt: 时间步长 (s)
            Q_pool_out: 稳流池出流 (m³/s)
            Q_pipe_in: 管道入口流量 (m³/s)

        Returns:
            SurgeTankState: 当前状态
        """
        # 净入塔流量
        net_flow = Q_pool_out - Q_pipe_in
        self.flow_in = net_flow

        # 水位变化
        self.prev_level = self.level
        dZ = net_flow / self.area * dt
        self.level += dZ

        # 物理约束
        if self.level > self.cfg.max_surge_level:
            self.level = self.cfg.max_surge_level
            # 实际应触发溢流，这里简化
        elif self.level < self.cfg.min_surge_level:
            self.level = self.cfg.min_surge_level
            # 实际应触发吸气，这里简化

        # 计算阻抗水头
        h_impedance = self.compute_impedance_head(self.flow_in)

        # 边界水头
        h_boundary = self.compute_boundary_head()

        # 振荡周期
        T_oscillation = self.estimate_oscillation_period()

        # 判断趋势
        is_rising = self.level > self.prev_level

        state = SurgeTankState(
            level=self.level,
            flow_in=self.flow_in,
            impedance_head=h_impedance,
            boundary_head=h_boundary,
            oscillation_period=T_oscillation,
            is_rising=is_rising
        )

        self.history.append(state)
        return state

    def get_damping_ratio(self) -> float:
        """
        计算阻尼比

        高阻尼比表示涌浪快速衰减
        """
        # 简化计算：基于阻抗系数比值
        ratio = (self.R_in - self.R_out) / (self.R_in + self.R_out)
        return abs(ratio)

    def predict_max_surge(self, flow_change: float) -> float:
        """
        预测最大涌浪高度

        简化公式 (忽略阻尼):
        Z_max = Q_change * sqrt(L / (g * A_tank * A_pipe))
        """
        Z_max = abs(flow_change) * np.sqrt(
            self.pipe_length /
            (self.physics.G * self.area * self.pipe_area)
        )
        return Z_max

    def is_stable(self, threshold: float = 0.1) -> bool:
        """判断是否稳定"""
        if len(self.history) < 10:
            return False

        recent_levels = [s.level for s in self.history[-10:]]
        std = np.std(recent_levels)
        return std < threshold

    def reset(self, level: float = 45.0):
        """重置状态"""
        self.level = level
        self.flow_in = 0.0
        self.velocity = 0.0
        self.prev_level = level
        self.oscillation_count = 0
        self.history.clear()

    def get_state_dict(self) -> dict:
        """获取状态字典"""
        return {
            'level': self.level,
            'flow_in': self.flow_in,
            'impedance_head': self.compute_impedance_head(self.flow_in),
            'boundary_head': self.compute_boundary_head(),
            'is_stable': self.is_stable()
        }
