"""
倒虹吸模型
==========

洮儿河、归流河倒虹吸枢纽仿真:
- 进口前池动力学
- 管身有压流
- 出口分水控制
- 防吸气逻辑
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from ..config.settings import Config


@dataclass
class SiphonState:
    """倒虹吸状态"""
    inlet_level: float      # 进口水位 (m)
    outlet_level: float     # 出口水位 (m)
    main_flow: float        # 主线流量 (m³/s)
    branch_flow: float      # 分水流量 (m³/s)
    submergence: float      # 进口淹没深度 (m)
    is_air_risk: bool       # 吸气风险
    pressure_drop: float    # 压力损失 (m)


class Siphon:
    """
    倒虹吸仿真模型

    物理特性:
    - 进口前池积分环节
    - 管身惯性与储能
    - 出口节制闸与分水闸耦合
    """

    def __init__(self, name: str = "TaoerRiver"):
        self.name = name
        self.physics = Config.physics

        # 几何参数
        self.pipe_length = 3000.0       # 管长 (m)
        self.pipe_diameter = 3.0        # 管径 (m)
        self.pipe_area = np.pi * (self.pipe_diameter / 2) ** 2

        # 进口前池
        self.forebay_area = 500.0       # 前池面积 (m²)
        self.crown_elevation = 5.0      # 管顶高程 (m)
        self.min_submergence = 1.5      # 最小淹没深度 (m)

        # 出口参数
        self.outlet_weir_elevation = 0.0  # 出口堰顶高程

        # 分水口 (乌兰浩特/科右前旗)
        self.branch_diameter = 0.8      # 分水管径 (m)
        self.branch_area = np.pi * (self.branch_diameter / 2) ** 2

        # 状态变量
        self.inlet_level = 7.0          # 进口水位
        self.outlet_level = 3.0         # 出口水位
        self.main_flow = 10.0           # 主线流量
        self.branch_flow = 1.0          # 分水流量

        # 闸门状态
        self.inlet_gate_opening = 1.0   # 进口闸开度
        self.outlet_gate_opening = 1.0  # 出口节制闸开度
        self.branch_gate_opening = 0.5  # 分水闸开度

        # 水力参数
        self.pipe_resistance = 0.02     # 管道阻力系数
        self.inertia_time = self.pipe_length / (self.physics.G * 10)  # 惯性时间常数

        # 历史记录
        self.history = []

    def compute_submergence(self) -> float:
        """计算进口淹没深度"""
        return max(self.inlet_level - self.crown_elevation, 0)

    def is_air_entrainment_risk(self) -> bool:
        """判断吸气风险"""
        return self.compute_submergence() < self.min_submergence

    def compute_main_flow(self) -> float:
        """
        计算主线过流能力

        Q = A * sqrt(2g * ΔH / (K + f*L/D))
        """
        delta_h = max(self.inlet_level - self.outlet_level, 0)

        # 总阻力系数
        K_inlet = 0.5  # 进口损失
        K_outlet = 1.0  # 出口损失
        K_friction = self.pipe_resistance * self.pipe_length / self.pipe_diameter
        K_total = K_inlet + K_outlet + K_friction

        # 考虑闸门开度
        K_valve = 0.5 * ((1 / max(self.outlet_gate_opening, 0.01)) ** 2 - 1)
        K_total += K_valve

        if K_total < 0.1:
            K_total = 0.1

        Q = self.pipe_area * np.sqrt(2 * self.physics.G * delta_h / K_total)
        return Q

    def compute_branch_flow(self) -> float:
        """
        计算分水流量

        受主线水头和分水闸开度控制
        """
        # 分水点水头 (近似为出口水位)
        H_branch = self.outlet_level

        # 分水闸流量
        Cd = 0.6
        e = self.branch_gate_opening * 0.8  # 最大开度0.8m
        B = 1.0  # 闸宽

        if H_branch > 0.1:
            Q = Cd * B * e * np.sqrt(2 * self.physics.G * H_branch)
        else:
            Q = 0.0

        return min(Q, 5.0)  # 限制最大分水量

    def compute_coupling_effect(self) -> float:
        """
        计算干支线耦合效应

        分水闸调节对主线流量的影响系数
        """
        # 耦合系数：分水占比越大，耦合越强
        if self.main_flow > 0.1:
            ratio = self.branch_flow / self.main_flow
            return ratio * 0.2  # 经验系数
        return 0.0

    def step(self, dt: float, Q_upstream: float,
             outlet_gate: float = None,
             branch_gate: float = None) -> SiphonState:
        """
        推进一个时间步

        Parameters:
            dt: 时间步长 (s)
            Q_upstream: 上游来水流量 (m³/s)
            outlet_gate: 出口闸开度
            branch_gate: 分水闸开度
        """
        if outlet_gate is not None:
            self.outlet_gate_opening = np.clip(outlet_gate, 0.0, 1.0)
        if branch_gate is not None:
            self.branch_gate_opening = np.clip(branch_gate, 0.0, 1.0)

        # 计算主线和分水流量
        Q_main_capacity = self.compute_main_flow()

        # 惯性修正 (一阶滞后)
        tau = self.inertia_time
        alpha = dt / (tau + dt)
        self.main_flow = (1 - alpha) * self.main_flow + alpha * Q_main_capacity

        self.branch_flow = self.compute_branch_flow()

        # 前池水位动力学
        Q_out = self.main_flow + self.branch_flow
        dH = (Q_upstream - Q_out) / self.forebay_area * dt
        self.inlet_level += dH

        # 物理约束
        self.inlet_level = np.clip(self.inlet_level, 1.0, 15.0)

        # 计算压力损失
        v = self.main_flow / self.pipe_area
        pressure_drop = self.pipe_resistance * self.pipe_length / self.pipe_diameter * \
                        (v ** 2) / (2 * self.physics.G)

        state = SiphonState(
            inlet_level=self.inlet_level,
            outlet_level=self.outlet_level,
            main_flow=self.main_flow,
            branch_flow=self.branch_flow,
            submergence=self.compute_submergence(),
            is_air_risk=self.is_air_entrainment_risk(),
            pressure_drop=pressure_drop
        )

        self.history.append(state)
        return state

    def set_outlet_level(self, level: float):
        """设置出口水位（由下游边界决定）"""
        self.outlet_level = level

    def get_decoupling_matrix(self) -> np.ndarray:
        """
        获取解耦矩阵

        用于干支线协调控制
        [ΔQ_main  ]   [K11  K12] [Δe_outlet]
        [ΔQ_branch] = [K21  K22] [Δe_branch]
        """
        # 数值微分估计增益矩阵
        eps = 0.01

        # 保存状态
        e_out_0 = self.outlet_gate_opening
        e_branch_0 = self.branch_gate_opening
        Q_main_0 = self.main_flow
        Q_branch_0 = self.branch_flow

        # K11: ∂Q_main/∂e_outlet
        self.outlet_gate_opening = e_out_0 + eps
        Q_main_1 = self.compute_main_flow()
        K11 = (Q_main_1 - Q_main_0) / eps

        # K12: ∂Q_main/∂e_branch
        self.outlet_gate_opening = e_out_0
        self.branch_gate_opening = e_branch_0 + eps
        Q_main_2 = self.compute_main_flow()
        K12 = (Q_main_2 - Q_main_0) / eps

        # K21: ∂Q_branch/∂e_outlet
        self.branch_gate_opening = e_branch_0
        self.outlet_gate_opening = e_out_0 + eps
        Q_branch_1 = self.compute_branch_flow()
        K21 = (Q_branch_1 - Q_branch_0) / eps

        # K22: ∂Q_branch/∂e_branch
        self.outlet_gate_opening = e_out_0
        self.branch_gate_opening = e_branch_0 + eps
        Q_branch_2 = self.compute_branch_flow()
        K22 = (Q_branch_2 - Q_branch_0) / eps

        # 恢复状态
        self.outlet_gate_opening = e_out_0
        self.branch_gate_opening = e_branch_0
        self.main_flow = Q_main_0
        self.branch_flow = Q_branch_0

        return np.array([[K11, K12], [K21, K22]])

    def reset(self, inlet_level: float = 7.0):
        """重置状态"""
        self.inlet_level = inlet_level
        self.outlet_level = 3.0
        self.main_flow = 10.0
        self.branch_flow = 1.0
        self.inlet_gate_opening = 1.0
        self.outlet_gate_opening = 1.0
        self.branch_gate_opening = 0.5
        self.history.clear()

    def get_state_dict(self) -> dict:
        """获取状态字典"""
        return {
            'inlet_level': self.inlet_level,
            'outlet_level': self.outlet_level,
            'main_flow': self.main_flow,
            'branch_flow': self.branch_flow,
            'submergence': self.compute_submergence(),
            'is_air_risk': self.is_air_entrainment_risk(),
            'outlet_gate': self.outlet_gate_opening,
            'branch_gate': self.branch_gate_opening
        }
