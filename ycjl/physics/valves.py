"""
阀门模型集合
============

包含各类阀门的物理特性仿真:
- RadialGate: 弧形闸门
- PlungerValve: 活塞阀
- ButterflyValve: 蝶阀
- ReliefValve: 泄压阀
- AirValve: 空气阀
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..config.settings import Config


class BaseValve(ABC):
    """阀门基类"""

    def __init__(self, name: str):
        self.name = name
        self.physics = Config.physics

        # 状态
        self.opening = 0.0           # 开度 (0~1)
        self.target_opening = 0.0    # 目标开度
        self.is_enabled = True       # 是否可用
        self.is_fault = False        # 是否故障

        # 动作参数
        self.max_rate = 0.02         # 最大动作速率 (%/s)
        self.dead_band = 0.01        # 死区

    @abstractmethod
    def compute_flow(self, H_up: float, H_down: float = 0.0) -> float:
        """计算流量"""
        pass

    @abstractmethod
    def compute_head_loss(self, Q: float) -> float:
        """计算水头损失"""
        pass

    def step(self, dt: float, target: float = None):
        """更新阀门状态"""
        if target is not None:
            self.target_opening = np.clip(target, 0.0, 1.0)

        if self.is_fault or not self.is_enabled:
            return

        # 按速率限制调节
        error = self.target_opening - self.opening
        if abs(error) > self.dead_band:
            delta = np.clip(error, -self.max_rate * dt, self.max_rate * dt)
            self.opening += delta
            self.opening = np.clip(self.opening, 0.0, 1.0)

    def set_fault(self, fault: bool):
        """设置故障状态"""
        self.is_fault = fault


@dataclass
class GateFlowResult:
    """闸门流量计算结果"""
    flow: float
    is_submerged: bool
    flow_coefficient: float
    head_loss: float


class RadialGate(BaseValve):
    """
    弧形闸门模型

    用于:
    - 溢洪道闸门
    - 进水口闸门
    - 节制闸
    """

    def __init__(self, name: str, width: float = 6.0, max_opening: float = 6.0):
        super().__init__(name)

        self.width = width             # 闸孔宽度 (m)
        self.max_opening = max_opening # 最大开度 (m)
        self.sill_elevation = 0.0      # 堰槛高程

    def compute_flow_coefficient(self, relative_opening: float) -> float:
        """
        计算流量系数 Cd

        Vuskovic经验公式
        """
        sigma = np.clip(relative_opening, 0.01, 1.0)
        return 0.611 * np.sqrt(1 + 0.045 * sigma)

    def compute_flow(self, H_up: float, H_down: float = 0.0) -> float:
        """
        计算过闸流量

        Parameters:
            H_up: 上游水位 (相对堰槛)
            H_down: 下游水位 (相对堰槛)

        Returns:
            流量 (m³/s)
        """
        if H_up <= 0.001 or self.opening < 0.001:
            return 0.0

        e = self.opening * self.max_opening  # 实际开度
        sigma = e / max(H_up, 0.1)           # 相对开度

        Cd = self.compute_flow_coefficient(sigma)

        # 淹没判定
        contraction_depth = e * 0.62
        is_submerged = H_down > contraction_depth

        if is_submerged:
            # 淹没出流
            delta_h = max(H_up - H_down, 0)
            Cs = 0.8  # 淹没系数
            Q = Cs * Cd * self.width * e * np.sqrt(2 * self.physics.G * delta_h)
        else:
            # 自由出流
            if e >= H_up * 0.9:
                # 堰流
                Q = 0.42 * self.width * (H_up ** 1.5) * np.sqrt(2 * self.physics.G)
            else:
                # 孔流
                Q = Cd * self.width * e * np.sqrt(2 * self.physics.G * H_up)

        return Q

    def compute_head_loss(self, Q: float) -> float:
        """计算水头损失"""
        e = max(self.opening * self.max_opening, 0.01)
        A = e * self.width
        v = Q / A
        K = 0.5 * ((1 / max(self.opening, 0.01)) ** 2 - 1)
        return K * (v ** 2) / (2 * self.physics.G)

    def get_flow_result(self, H_up: float, H_down: float = 0.0) -> GateFlowResult:
        """获取详细流量计算结果"""
        Q = self.compute_flow(H_up, H_down)
        e = self.opening * self.max_opening
        sigma = e / max(H_up, 0.1)

        return GateFlowResult(
            flow=Q,
            is_submerged=H_down > e * 0.62,
            flow_coefficient=self.compute_flow_coefficient(sigma),
            head_loss=self.compute_head_loss(Q)
        )


class PlungerValve(BaseValve):
    """
    活塞阀模型

    用于:
    - T212调流调压阀
    - 末端调流阀
    """

    def __init__(self, name: str, diameter: float = 1.6, cv_max: float = 25.0):
        super().__init__(name)

        self.diameter = diameter
        self.area = np.pi * (diameter / 2) ** 2
        self.cv_max = cv_max           # 最大流量系数
        self.k_full_open = 0.15        # 全开阻力系数

    def compute_k(self) -> float:
        """
        计算阻力系数 K

        活塞阀: K = K0 + 0.5 * (1/s² - 1)
        """
        s = np.clip(self.opening, 0.01, 1.0)
        return self.k_full_open + 0.5 * ((1 / s) ** 2 - 1)

    def compute_cv(self) -> float:
        """计算当前Cv值"""
        return self.opening * self.cv_max

    def compute_flow(self, H_up: float, H_down: float = 0.0) -> float:
        """
        计算流量

        Q = Cv * sqrt(ΔP)
        """
        delta_h = max(H_up - H_down, 0)
        Cv = self.compute_cv()

        if Cv < 0.01:
            return 0.0

        return Cv * np.sqrt(delta_h)

    def compute_head_loss(self, Q: float) -> float:
        """计算水头损失"""
        K = self.compute_k()
        v = Q / self.area
        return K * (v ** 2) / (2 * self.physics.G)

    def check_cavitation(self, H_up: float, H_down: float) -> bool:
        """
        检查空蚀风险

        空蚀系数 σ = (H_down - H_vapor) / (H_up - H_down)
        """
        delta_h = H_up - H_down
        if delta_h < 0.1:
            return False

        H_vapor = self.physics.VAPOR_PRESSURE
        sigma = (H_down - H_vapor) / delta_h

        return sigma < 0.25  # 临界值


class ButterflyValve(BaseValve):
    """
    蝶阀模型

    用于:
    - 联通阀
    - 检修阀
    """

    def __init__(self, name: str, diameter: float = 2.8):
        super().__init__(name)

        self.diameter = diameter
        self.area = np.pi * (diameter / 2) ** 2

    def compute_k(self) -> float:
        """
        计算阻力系数

        蝶阀: 即使全开也有一定阻力 (蝶板在流道中)
        """
        if self.opening < 0.01:
            return 1e6  # 接近无穷大

        # 经验公式
        theta = (1 - self.opening) * 90  # 角度 (0~90°)
        K = 0.5 + 2.0 * np.tan(np.radians(theta)) ** 2
        return K

    def compute_flow(self, H_up: float, H_down: float = 0.0) -> float:
        """计算流量"""
        delta_h = max(H_up - H_down, 0)
        K = self.compute_k()

        if K > 1e5:
            return 0.0

        return self.area * np.sqrt(2 * self.physics.G * delta_h / K)

    def compute_head_loss(self, Q: float) -> float:
        """计算水头损失"""
        K = self.compute_k()
        v = Q / self.area
        return K * (v ** 2) / (2 * self.physics.G)

    def get_torque(self, delta_p: float) -> float:
        """
        估算操作力矩

        用于防止动水误操作
        """
        # 简化模型: T = C * D² * ΔP
        C = 0.1  # 经验系数
        return C * (self.diameter ** 2) * delta_p


class ReliefValve(BaseValve):
    """
    泄压阀模型

    用于:
    - 超压泄压阀 (被动式)
    """

    def __init__(self, name: str, set_pressure: float = 115.0, cv: float = 30.0):
        super().__init__(name)

        self.set_pressure = set_pressure   # 开启压力 (m)
        self.close_ratio = 0.9             # 关闭压力比
        self.cv = cv                       # 流量系数
        self.damping_time = 30.0           # 液压阻尼时间 (s)

        # 状态
        self.is_tripped = False            # 是否触发过

    def compute_flow(self, H_up: float, H_down: float = 0.0) -> float:
        """
        计算泄流量

        被动式: 压力超过设定值自动开启
        """
        if H_up < self.set_pressure:
            return 0.0

        # 超过设定压力的部分
        delta_h = H_up - self.set_pressure

        # 开度与超压成正比 (简化)
        equiv_opening = np.clip(delta_h / 10.0, 0.0, 1.0)

        return self.cv * equiv_opening * np.sqrt(delta_h)

    def compute_head_loss(self, Q: float) -> float:
        """泄压阀不计入管道损失"""
        return 0.0

    def step(self, dt: float, pressure: float):
        """
        更新状态

        考虑液压阻尼的开闭特性
        """
        if pressure >= self.set_pressure:
            # 快开
            target = np.clip((pressure - self.set_pressure) / 10.0, 0.0, 1.0)
            self.opening = target  # 快速响应
            self.is_tripped = True
        else:
            # 慢关 (液压阻尼)
            close_pressure = self.set_pressure * self.close_ratio
            if pressure < close_pressure:
                # 按时间常数缓慢关闭
                tau = self.damping_time
                alpha = dt / (tau + dt)
                self.opening = self.opening * (1 - alpha)

    def check_status(self, pressure: float) -> str:
        """检查状态"""
        if pressure >= self.set_pressure:
            return "OPEN"
        elif self.opening > 0.01:
            return "CLOSING"
        elif self.is_tripped:
            return "TRIPPED"
        else:
            return "STANDBY"


class AirValve(BaseValve):
    """
    空气阀模型

    用于:
    - 高点排气
    - 负压补气
    - 防水锤
    """

    def __init__(self, name: str, intake_capacity: float = 0.5,
                 exhaust_capacity: float = 0.2):
        super().__init__(name)

        self.intake_capacity = intake_capacity    # 进气能力 (m³/s)
        self.exhaust_capacity = exhaust_capacity  # 排气能力 (m³/s)

        # 双孔设计
        self.large_orifice = 0.1                  # 大孔直径 (m)
        self.small_orifice = 0.02                 # 小孔直径 (m)

        # 状态
        self.air_volume = 0.0                     # 管内气体体积
        self.is_venting = False                   # 正在排气

    def compute_flow(self, H_up: float, H_down: float = 0.0) -> float:
        """
        空气阀流量 (正为进气，负为排气)
        """
        P_atm = self.physics.PATM

        if H_up < P_atm:
            # 负压，进气
            delta_p = P_atm - H_up
            # 大孔进气
            Q_air = self.intake_capacity * np.sqrt(delta_p / P_atm)
            return Q_air
        elif H_up > P_atm and self.air_volume > 0:
            # 正压且有积气，排气
            delta_p = H_up - P_atm

            if H_up > P_atm * 1.5:
                # 高压下小孔缓慢排气 (防水锤)
                Q_air = -self.exhaust_capacity * 0.3 * np.sqrt(delta_p / P_atm)
            else:
                # 正常排气
                Q_air = -self.exhaust_capacity * np.sqrt(delta_p / P_atm)
            return Q_air

        return 0.0

    def compute_head_loss(self, Q: float) -> float:
        """空气阀损失忽略"""
        return 0.0

    def step(self, dt: float, local_pressure: float, pipe_flow: float = 0.0):
        """更新状态"""
        Q_air = self.compute_flow(local_pressure)

        if Q_air > 0:
            # 进气
            self.air_volume += Q_air * dt
            self.is_venting = False
        elif Q_air < 0:
            # 排气
            self.air_volume -= abs(Q_air) * dt
            self.air_volume = max(self.air_volume, 0)
            self.is_venting = True
        else:
            self.is_venting = False

        # 更新开度状态（指示）
        if abs(Q_air) > 0.001:
            self.opening = min(abs(Q_air) / self.intake_capacity, 1.0)
        else:
            self.opening = 0.0

    def get_cushion_effect(self) -> float:
        """
        获取气垫缓冲效果

        返回等效弹簧刚度
        """
        if self.air_volume <= 0:
            return 0.0

        # P * V^k = const, k = 1.4 (绝热)
        # 等效刚度 K = k * P / V
        P = self.physics.PATM
        K = 1.4 * P / max(self.air_volume, 0.001)
        return K
