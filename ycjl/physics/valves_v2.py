"""
阀门模型 V2 - 增强版
===================

基于配置数据库 v3.2 数字化流阻特性实现:
- 在线调流调压阀 (活塞式)
- 末端调流阀
- 检修蝶阀
- 泄压阀
- 空气阀

特点:
- 使用实测流阻曲线
- 空化校核
- 动作时间约束
- 故障模式仿真
"""

import math
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod

from ..config.config_database import (
    ProjectParams, CurveDatabase, ValveConfig, SafetyConfig, GlobalConfig
)


class ValveState(Enum):
    """阀门状态"""
    CLOSED = auto()         # 全关
    OPENING = auto()        # 开启中
    OPEN = auto()           # 开启
    CLOSING = auto()        # 关闭中
    FAULT = auto()          # 故障
    MAINTENANCE = auto()    # 检修


class ValveFaultType(Enum):
    """阀门故障类型"""
    NONE = auto()           # 无故障
    STUCK_OPEN = auto()     # 卡在开位
    STUCK_CLOSED = auto()   # 卡在关位
    STUCK_POSITION = auto() # 卡在中间位置
    SLOW_RESPONSE = auto()  # 响应慢
    LEAKAGE = auto()        # 内漏


@dataclass
class ValveOperatingPoint:
    """阀门运行工况点"""
    opening: float              # 开度 (0-1)
    flow: float                 # 流量 (m³/s)
    head_loss: float            # 水头损失 (m)
    zeta: float                 # 阻力系数
    velocity: float             # 阀门处流速 (m/s)
    cavitation_index: float     # 空化指数
    is_cavitating: bool         # 是否发生空化


class BaseValveV2(ABC):
    """阀门基类 V2"""

    def __init__(self, name: str, diameter: float, location: str = ""):
        """
        初始化阀门

        Args:
            name: 阀门名称
            diameter: 公称直径 (m)
            location: 位置 (桩号)
        """
        self.name = name
        self.diameter = diameter
        self.location = location
        self.physics = GlobalConfig

        # 几何参数
        self.area = math.pi * (diameter / 2) ** 2

        # 状态变量
        self.opening = 0.0              # 当前开度 (0-1)
        self.target_opening = 0.0       # 目标开度
        self.state = ValveState.CLOSED
        self.fault_type = ValveFaultType.NONE

        # 动作参数
        self.max_rate = 0.02            # 最大动作速率 (%/s)
        self.dead_band = 0.005          # 死区

        # 流阻特性 (子类设置)
        self.zeta_curve: List[Tuple[float, float]] = []

        # 空化参数
        self.sigma_incipient = 0.25     # 初生空化系数
        self.sigma_critical = 0.15      # 临界空化系数

        # 历史
        self.history: List[ValveOperatingPoint] = []

    def get_zeta(self, opening_pct: float) -> float:
        """
        获取阻力系数

        Args:
            opening_pct: 开度百分比 (0-100)

        Returns:
            阻力系数 zeta
        """
        if not self.zeta_curve:
            return 1.0

        x = [p[0] for p in self.zeta_curve]
        y = [p[1] for p in self.zeta_curve]
        return float(np.interp(opening_pct, x, y))

    def compute_flow(self, H_up: float, H_down: float) -> float:
        """
        计算流量

        Q = A * sqrt(2g * dH / zeta)

        Args:
            H_up: 上游水头 (m)
            H_down: 下游水头 (m)

        Returns:
            流量 (m³/s)
        """
        dH = max(H_up - H_down, 0)
        if dH < 0.001 or self.opening < 0.001:
            return 0.0

        zeta = self.get_zeta(self.opening * 100)
        if zeta > 1e6:
            return 0.0

        g = self.physics.G
        Q = self.area * math.sqrt(2 * g * dH / zeta)

        return Q

    def compute_head_loss(self, Q: float) -> float:
        """
        计算水头损失

        dH = zeta * v² / 2g

        Args:
            Q: 流量 (m³/s)

        Returns:
            水头损失 (m)
        """
        if Q < 1e-6:
            return 0.0

        v = Q / self.area
        zeta = self.get_zeta(self.opening * 100)
        g = self.physics.G

        return zeta * v**2 / (2 * g)

    def compute_cavitation_index(self, H_up: float, H_down: float) -> float:
        """
        计算空化指数

        sigma = (H_down - H_vapor) / (H_up - H_down)

        Args:
            H_up: 上游水头 (m)
            H_down: 下游水头 (m)

        Returns:
            空化指数
        """
        dH = H_up - H_down
        if dH < 0.1:
            return 1.0

        H_atm = self.physics.PATM_HEAD
        H_vapor = abs(self.physics.VAPOR_PRESSURE_HEAD)

        sigma = (H_down + H_atm - H_vapor) / dH
        return sigma

    def check_cavitation(self, sigma: float) -> bool:
        """检查是否发生空化"""
        return sigma < self.sigma_incipient

    def update_state(self):
        """更新阀门状态"""
        if self.fault_type != ValveFaultType.NONE:
            self.state = ValveState.FAULT
        elif self.opening < 0.01:
            self.state = ValveState.CLOSED
        elif self.opening > 0.99:
            self.state = ValveState.OPEN
        elif self.target_opening > self.opening:
            self.state = ValveState.OPENING
        elif self.target_opening < self.opening:
            self.state = ValveState.CLOSING
        else:
            self.state = ValveState.OPEN

    def step(self, dt: float, H_up: float = 0.0, H_down: float = 0.0,
             target: float = None) -> ValveOperatingPoint:
        """
        推进一个时间步

        Args:
            dt: 时间步长 (s)
            H_up: 上游水头 (m)
            H_down: 下游水头 (m)
            target: 目标开度

        Returns:
            运行工况点
        """
        if target is not None:
            self.target_opening = np.clip(target, 0.0, 1.0)

        # 故障处理
        if self.fault_type == ValveFaultType.STUCK_OPEN:
            self.opening = 1.0
        elif self.fault_type == ValveFaultType.STUCK_CLOSED:
            self.opening = 0.0
        elif self.fault_type == ValveFaultType.STUCK_POSITION:
            pass  # 保持当前位置
        elif self.fault_type == ValveFaultType.SLOW_RESPONSE:
            # 响应减慢
            rate = self.max_rate * 0.3
            error = self.target_opening - self.opening
            if abs(error) > self.dead_band:
                delta = np.clip(error, -rate * dt, rate * dt)
                self.opening += delta
        elif self.fault_type == ValveFaultType.NONE:
            # 正常动作
            error = self.target_opening - self.opening
            if abs(error) > self.dead_band:
                delta = np.clip(error, -self.max_rate * dt, self.max_rate * dt)
                self.opening += delta

        self.opening = np.clip(self.opening, 0.0, 1.0)
        self.update_state()

        # 计算水力参数
        Q = self.compute_flow(H_up, H_down)
        dH = self.compute_head_loss(Q)
        zeta = self.get_zeta(self.opening * 100)
        v = Q / self.area if self.area > 0 else 0.0
        sigma = self.compute_cavitation_index(H_up, H_down)
        is_cavitating = self.check_cavitation(sigma)

        point = ValveOperatingPoint(
            opening=self.opening,
            flow=Q,
            head_loss=dH,
            zeta=zeta,
            velocity=v,
            cavitation_index=sigma,
            is_cavitating=is_cavitating
        )

        self.history.append(point)
        return point

    def set_fault(self, fault_type: ValveFaultType):
        """设置故障类型"""
        self.fault_type = fault_type
        self.update_state()

    def clear_fault(self):
        """清除故障"""
        self.fault_type = ValveFaultType.NONE
        self.update_state()

    def emergency_close(self, close_time: float, dt: float):
        """
        紧急关闭

        Args:
            close_time: 关闭时间 (s)
            dt: 时间步长 (s)
        """
        self.target_opening = 0.0
        # 临时提高动作速率
        self.max_rate = 1.0 / close_time

    def reset(self):
        """重置阀门"""
        self.opening = 0.0
        self.target_opening = 0.0
        self.state = ValveState.CLOSED
        self.fault_type = ValveFaultType.NONE
        self.history.clear()


class InlineRegulatingValve(BaseValveV2):
    """
    在线调流调压阀

    活塞式，位于T212调流调压阀室
    """

    def __init__(self, valve_id: int):
        """
        初始化在线调流阀

        Args:
            valve_id: 阀门编号 (1-3)
        """
        super().__init__(
            name=f"T212-{valve_id}",
            diameter=ValveConfig.INLINE_VALVE_DN,
            location=ValveConfig.INLINE_VALVE_LOCATION
        )

        self.valve_id = valve_id

        # 加载流阻曲线
        self.zeta_curve = CurveDatabase.INLINE_REGULATING_VALVE_ZETA

        # 动作约束
        self.min_close_time = ValveConfig.VALVE_CLOSE_TIMES.get('Main_Inline', 2500.0)
        self.max_rate = 1.0 / self.min_close_time  # 基于关闭时间

        # 活塞阀特性
        self.stroke_length = 0.8  # 行程 (m)

    def get_cv(self) -> float:
        """获取当前Cv值"""
        # Cv与开度近似线性
        cv_max = 25.0  # 全开Cv值
        return self.opening * cv_max

    def compute_flow_cv(self, dP: float) -> float:
        """
        基于Cv值计算流量

        Q = Cv * sqrt(dP)  (dP in bar, Q in m³/h)

        Args:
            dP: 压差 (m水头)

        Returns:
            流量 (m³/s)
        """
        Cv = self.get_cv()
        if Cv < 0.01:
            return 0.0

        # 转换单位
        dP_bar = dP * 0.098  # m -> bar
        Q_m3h = Cv * math.sqrt(max(dP_bar, 0))
        return Q_m3h / 3600  # m³/h -> m³/s


class EndRegulatingValve(BaseValveV2):
    """
    末端调流阀

    位于管道末端，特性与在线阀略有不同
    """

    def __init__(self, valve_id: int = 1):
        """初始化末端调流阀"""
        super().__init__(
            name=f"END-{valve_id}",
            diameter=ValveConfig.END_VALVE_DN,
            location=ValveConfig.END_VALVE_LOCATION
        )

        self.valve_id = valve_id

        # 加载流阻曲线
        self.zeta_curve = CurveDatabase.END_REGULATING_VALVE_ZETA

        # 动作约束
        self.min_close_time = ValveConfig.VALVE_CLOSE_TIMES.get('Main_End', 2500.0)
        self.max_rate = 1.0 / self.min_close_time


class ButterflyValveV2(BaseValveV2):
    """
    检修蝶阀 V2

    使用数字化流阻曲线
    """

    def __init__(self, name: str, diameter: float = 2.8):
        """初始化蝶阀"""
        super().__init__(
            name=name,
            diameter=diameter
        )

        # 加载流阻曲线
        self.zeta_curve = CurveDatabase.BUTTERFLY_VALVE_ZETA

        # 蝶阀特性
        self.is_normally_open = True  # 常开

    def get_torque(self, dP: float) -> float:
        """
        估算操作力矩

        蝶阀动水操作力矩较大

        Args:
            dP: 阀门前后压差 (m)

        Returns:
            力矩 (kN·m)
        """
        # 经验公式: T = C * D² * dP
        C = 0.12  # 蝶阀系数
        return C * self.diameter**2 * dP


class ReliefValveV2(BaseValveV2):
    """
    泄压阀 V2

    被动式超压保护
    """

    def __init__(self, valve_id: int, set_pressure: float):
        """
        初始化泄压阀

        Args:
            valve_id: 阀门编号
            set_pressure: 开启压力 (m水头)
        """
        super().__init__(
            name=f"RELIEF-{valve_id}",
            diameter=0.3  # 泄压阀口径较小
        )

        self.valve_id = valve_id
        self.set_pressure = set_pressure
        self.close_ratio = SafetyConfig.RELIEF_VALVE_CLOSE_RATIO
        self.cv = SafetyConfig.RELIEF_VALVE_CV
        self.damping_time = SafetyConfig.RELIEF_VALVE_DAMPING_TIME

        # 泄压阀不使用常规流阻曲线
        self.zeta_curve = []

        # 状态
        self.is_tripped = False

    def compute_flow(self, pressure: float, H_down: float = 0.0) -> float:
        """
        计算泄流量

        只有超压时才泄流

        Args:
            pressure: 上游压力 (m水头)
            H_down: 下游水头

        Returns:
            泄流量 (m³/s)
        """
        if pressure < self.set_pressure:
            return 0.0

        # 超压部分
        delta_p = pressure - self.set_pressure

        # 等效开度 (与超压成正比)
        equiv_opening = np.clip(delta_p / 10.0, 0.0, 1.0)

        return self.cv * equiv_opening * math.sqrt(delta_p)

    def step(self, dt: float, pressure: float,
             H_down: float = 0.0, target: float = None) -> ValveOperatingPoint:
        """
        更新泄压阀状态

        被动响应，不接受目标开度
        """
        if pressure >= self.set_pressure:
            # 快开
            target_opening = np.clip((pressure - self.set_pressure) / 10.0, 0.0, 1.0)
            self.opening = target_opening  # 快速响应
            self.is_tripped = True
            self.state = ValveState.OPEN
        else:
            # 慢关 (液压阻尼)
            close_pressure = self.set_pressure * self.close_ratio
            if pressure < close_pressure:
                tau = self.damping_time
                alpha = dt / (tau + dt)
                self.opening = self.opening * (1 - alpha)
                if self.opening < 0.01:
                    self.opening = 0.0
                    self.state = ValveState.CLOSED

        Q = self.compute_flow(pressure, H_down)

        return ValveOperatingPoint(
            opening=self.opening,
            flow=Q,
            head_loss=0.0,
            zeta=0.0,
            velocity=Q / self.area if Q > 0 else 0.0,
            cavitation_index=1.0,
            is_cavitating=False
        )

    def get_status(self) -> str:
        """获取状态字符串"""
        if self.state == ValveState.OPEN:
            return "RELEASING"
        elif self.is_tripped:
            return "TRIPPED"
        else:
            return "STANDBY"


class AirValveV2:
    """
    空气阀 V2

    双孔进排气阀
    """

    def __init__(self, location: str, valve_id: int):
        """
        初始化空气阀

        Args:
            location: 位置 (桩号)
            valve_id: 阀门编号
        """
        self.name = f"AIR-{valve_id}"
        self.location = location
        self.physics = GlobalConfig

        # 规格
        self.large_orifice = SafetyConfig.AIR_VALVE_LARGE_ORIFICE
        self.small_orifice = SafetyConfig.AIR_VALVE_SMALL_ORIFICE
        self.intake_capacity = SafetyConfig.AIR_VALVE_INTAKE_CAPACITY
        self.exhaust_capacity = SafetyConfig.AIR_VALVE_EXHAUST_CAPACITY

        # 状态
        self.air_volume = 0.0   # 管内积气量 (m³)
        self.is_venting = False
        self.is_intaking = False

    def compute_air_flow(self, pipe_pressure: float) -> float:
        """
        计算空气流量

        正值为进气，负值为排气

        Args:
            pipe_pressure: 管内压力 (m水头)

        Returns:
            空气流量 (m³/s)
        """
        P_atm = self.physics.PATM_HEAD

        if pipe_pressure < P_atm:
            # 负压，进气 (大孔)
            delta_p = P_atm - pipe_pressure
            Q_air = self.intake_capacity * math.sqrt(delta_p / P_atm)
            self.is_intaking = True
            self.is_venting = False
            return Q_air

        elif pipe_pressure > P_atm and self.air_volume > 0:
            # 正压且有积气，排气
            delta_p = pipe_pressure - P_atm

            if pipe_pressure > P_atm * 1.5:
                # 高压下小孔缓慢排气 (防水锤)
                Q_air = -self.exhaust_capacity * 0.3 * math.sqrt(delta_p / P_atm)
            else:
                # 正常排气
                Q_air = -self.exhaust_capacity * math.sqrt(delta_p / P_atm)

            self.is_venting = True
            self.is_intaking = False
            return Q_air

        self.is_venting = False
        self.is_intaking = False
        return 0.0

    def step(self, dt: float, pipe_pressure: float) -> Dict:
        """
        更新空气阀状态

        Args:
            dt: 时间步长 (s)
            pipe_pressure: 管内压力 (m水头)

        Returns:
            状态字典
        """
        Q_air = self.compute_air_flow(pipe_pressure)

        if Q_air > 0:
            self.air_volume += Q_air * dt
        elif Q_air < 0:
            self.air_volume = max(0, self.air_volume + Q_air * dt)

        return {
            'air_flow': Q_air,
            'air_volume': self.air_volume,
            'is_intaking': self.is_intaking,
            'is_venting': self.is_venting
        }

    def get_cushion_stiffness(self) -> float:
        """
        获取气垫刚度

        用于水锤缓冲计算
        """
        if self.air_volume <= 0:
            return 0.0

        P = self.physics.PATM_HEAD
        k = 1.4  # 绝热指数
        return k * P / max(self.air_volume, 0.001)


class ValveSystem:
    """
    阀门系统

    管理全线所有阀门的协调运行
    """

    def __init__(self):
        """初始化阀门系统"""
        # 在线调流阀 (3台)
        self.inline_valves: List[InlineRegulatingValve] = [
            InlineRegulatingValve(i) for i in range(1, 4)
        ]

        # 末端调流阀 (2台)
        self.end_valves: List[EndRegulatingValve] = [
            EndRegulatingValve(i) for i in range(1, 3)
        ]

        # 泄压阀
        self.relief_valves: List[ReliefValveV2] = [
            ReliefValveV2(1, SafetyConfig.RELIEF_VALVE_SET_PRESSURE_1),
            ReliefValveV2(2, SafetyConfig.RELIEF_VALVE_SET_PRESSURE_2)
        ]

        # 运行参数
        self.active_inline_count = 2  # 在线阀运行数量 (2用1备)

    def get_inline_total_flow(self, H_up: float, H_down: float) -> float:
        """获取在线阀总流量"""
        total = 0.0
        for i, valve in enumerate(self.inline_valves[:self.active_inline_count]):
            total += valve.compute_flow(H_up, H_down)
        return total

    def set_inline_openings(self, opening: float):
        """设置所有在线阀开度"""
        for valve in self.inline_valves[:self.active_inline_count]:
            valve.target_opening = opening

    def step(self, dt: float, pressures: Dict[str, float]) -> Dict:
        """
        更新阀门系统

        Args:
            dt: 时间步长 (s)
            pressures: 各位置压力 {location: pressure}

        Returns:
            状态汇总
        """
        status = {
            'inline': [],
            'end': [],
            'relief': []
        }

        # 更新在线阀
        H_inline = pressures.get('inline', 80.0)
        for valve in self.inline_valves:
            point = valve.step(dt, H_inline, H_inline - 5)
            status['inline'].append({
                'id': valve.valve_id,
                'opening': point.opening,
                'flow': point.flow,
                'state': valve.state.name
            })

        # 更新末端阀
        H_end = pressures.get('end', 50.0)
        for valve in self.end_valves:
            point = valve.step(dt, H_end, 0)
            status['end'].append({
                'id': valve.valve_id,
                'opening': point.opening,
                'flow': point.flow,
                'state': valve.state.name
            })

        # 更新泄压阀
        for valve in self.relief_valves:
            P = pressures.get(f'relief_{valve.valve_id}', H_inline)
            point = valve.step(dt, P)
            status['relief'].append({
                'id': valve.valve_id,
                'status': valve.get_status(),
                'flow': point.flow
            })

        return status

    def emergency_shutdown(self):
        """紧急停机"""
        for valve in self.inline_valves:
            valve.emergency_close(60.0, 0.5)
        for valve in self.end_valves:
            valve.emergency_close(60.0, 0.5)


# 导出
__all__ = [
    'ValveState',
    'ValveFaultType',
    'ValveOperatingPoint',
    'BaseValveV2',
    'InlineRegulatingValve',
    'EndRegulatingValve',
    'ButterflyValveV2',
    'ReliefValveV2',
    'AirValveV2',
    'ValveSystem'
]
