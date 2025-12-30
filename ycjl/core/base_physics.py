"""
物理模型基类 (Base Physics Models)
==================================

提供水利工程物理模型的抽象基类，包括：
- 水库模型
- 管道模型
- 泵站模型
- 阀门模型
- 渠道模型

设计原则：
- 使用组合模式支持复杂系统
- 统一的状态管理接口
- 支持瞬态和稳态计算
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum, auto
import numpy as np


class ComponentType(Enum):
    """水力组件类型"""
    RESERVOIR = auto()          # 水库
    PIPELINE = auto()           # 有压管道
    CHANNEL = auto()            # 明渠
    TUNNEL = auto()             # 隧洞
    PUMP_STATION = auto()       # 泵站
    VALVE = auto()              # 阀门
    GATE = auto()               # 闸门
    SURGE_TANK = auto()         # 调压井
    POOL = auto()               # 蓄水池/稳流池
    TURBINE = auto()            # 水轮机
    SPILLWAY = auto()           # 溢洪道
    JUNCTION = auto()           # 节点


class ComponentStatus(Enum):
    """组件状态"""
    NORMAL = auto()             # 正常
    WARNING = auto()            # 警告
    ALARM = auto()              # 报警
    FAULT = auto()              # 故障
    OFFLINE = auto()            # 离线
    MAINTENANCE = auto()        # 维护中


@dataclass
class HydraulicState:
    """
    水力状态数据类

    存储组件的瞬时水力状态
    """
    time: float = 0.0                           # 时间 (s)
    flow: float = 0.0                           # 流量 (m³/s)
    pressure_head: float = 0.0                  # 压力水头 (m)
    velocity: float = 0.0                       # 流速 (m/s)
    level: float = 0.0                          # 水位 (m)
    volume: float = 0.0                         # 体积 (m³)
    head_loss: float = 0.0                      # 水头损失 (m)
    power: float = 0.0                          # 功率 (kW)
    efficiency: float = 0.0                     # 效率
    status: ComponentStatus = ComponentStatus.NORMAL


class BaseHydraulicComponent(ABC):
    """
    水力组件抽象基类

    所有水力设施的基类，定义通用接口
    """

    def __init__(self, name: str, component_type: ComponentType):
        self.name = name
        self.component_type = component_type
        self.status = ComponentStatus.NORMAL
        self._state = HydraulicState()
        self._history: List[HydraulicState] = []

    @property
    def state(self) -> HydraulicState:
        """获取当前状态"""
        return self._state

    @abstractmethod
    def update(self, dt: float, **kwargs) -> HydraulicState:
        """
        更新组件状态

        Args:
            dt: 时间步长 (s)
            **kwargs: 边界条件

        Returns:
            更新后的状态
        """
        pass

    @abstractmethod
    def get_head_loss(self, flow: float) -> float:
        """
        计算水头损失

        Args:
            flow: 流量 (m³/s)

        Returns:
            水头损失 (m)
        """
        pass

    def record_state(self):
        """记录当前状态到历史"""
        self._history.append(HydraulicState(
            time=self._state.time,
            flow=self._state.flow,
            pressure_head=self._state.pressure_head,
            velocity=self._state.velocity,
            level=self._state.level,
            volume=self._state.volume,
            head_loss=self._state.head_loss,
            power=self._state.power,
            efficiency=self._state.efficiency,
            status=self._state.status
        ))

    def get_history(self) -> List[HydraulicState]:
        """获取历史记录"""
        return self._history

    def clear_history(self):
        """清除历史记录"""
        self._history.clear()


class BaseReservoir(BaseHydraulicComponent):
    """
    水库模型基类

    实现水库的基本水量平衡和水位-库容关系
    """

    def __init__(self, name: str,
                 normal_level: float,
                 dead_level: float,
                 max_level: float,
                 zv_curve: Optional[Callable[[float], float]] = None,
                 za_curve: Optional[Callable[[float], float]] = None):
        """
        初始化水库模型

        Args:
            name: 水库名称
            normal_level: 正常蓄水位 (m)
            dead_level: 死水位 (m)
            max_level: 最高水位 (m)
            zv_curve: 水位-库容曲线函数 level -> volume
            za_curve: 水位-面积曲线函数 level -> area
        """
        super().__init__(name, ComponentType.RESERVOIR)
        self.normal_level = normal_level
        self.dead_level = dead_level
        self.max_level = max_level
        self._zv_curve = zv_curve
        self._za_curve = za_curve
        self._state.level = normal_level

    def get_volume(self, level: float) -> float:
        """水位->库容"""
        if self._zv_curve:
            return self._zv_curve(level)
        # 默认线性关系
        return (level - self.dead_level) * 1e6  # 简化假设

    def get_area(self, level: float) -> float:
        """水位->面积"""
        if self._za_curve:
            return self._za_curve(level)
        return 1e6  # 默认1km²

    def get_level(self, volume: float) -> float:
        """库容->水位（反查）"""
        # 二分法查找
        low, high = self.dead_level, self.max_level
        for _ in range(50):
            mid = (low + high) / 2
            v_mid = self.get_volume(mid)
            if abs(v_mid - volume) < 1e3:
                return mid
            if v_mid < volume:
                low = mid
            else:
                high = mid
        return mid

    def update(self, dt: float, inflow: float = 0.0, outflow: float = 0.0) -> HydraulicState:
        """
        更新水库状态

        Args:
            dt: 时间步长 (s)
            inflow: 入库流量 (m³/s)
            outflow: 出库流量 (m³/s)

        Returns:
            更新后的状态
        """
        # 当前库容
        current_volume = self.get_volume(self._state.level)

        # 水量平衡
        delta_v = (inflow - outflow) * dt
        new_volume = current_volume + delta_v

        # 约束库容
        new_volume = max(self.get_volume(self.dead_level), new_volume)
        new_volume = min(self.get_volume(self.max_level), new_volume)

        # 更新水位
        new_level = self.get_level(new_volume)

        self._state.time += dt
        self._state.level = new_level
        self._state.volume = new_volume
        self._state.flow = outflow

        # 检查状态
        if new_level >= self.max_level * 0.98:
            self.status = ComponentStatus.ALARM
        elif new_level <= self.dead_level * 1.02:
            self.status = ComponentStatus.WARNING
        else:
            self.status = ComponentStatus.NORMAL

        self._state.status = self.status
        return self._state

    def get_head_loss(self, flow: float) -> float:
        """水库无水头损失"""
        return 0.0


class BasePipeline(BaseHydraulicComponent):
    """
    管道模型基类

    实现有压管道的水力计算
    """

    def __init__(self, name: str,
                 length: float,
                 diameter: float,
                 roughness: float = 0.012,
                 wave_speed: float = 1000.0):
        """
        初始化管道模型

        Args:
            name: 管道名称
            length: 长度 (m)
            diameter: 内径 (m)
            roughness: 达西摩阻系数
            wave_speed: 压力波速 (m/s)
        """
        super().__init__(name, ComponentType.PIPELINE)
        self.length = length
        self.diameter = diameter
        self.roughness = roughness
        self.wave_speed = wave_speed

    @property
    def area(self) -> float:
        """断面积"""
        return math.pi * (self.diameter / 2) ** 2

    def get_head_loss(self, flow: float) -> float:
        """
        计算沿程水头损失 (Darcy-Weisbach)

        hf = f * (L/D) * (V²/2g)
        """
        if abs(flow) < 1e-9:
            return 0.0

        velocity = abs(flow) / self.area
        g = 9.80665
        hf = self.roughness * (self.length / self.diameter) * (velocity ** 2) / (2 * g)
        return hf if flow >= 0 else -hf

    def get_velocity(self, flow: float) -> float:
        """计算流速"""
        return flow / self.area if self.area > 0 else 0.0

    def update(self, dt: float, upstream_head: float = 0.0,
               downstream_head: float = 0.0) -> HydraulicState:
        """
        更新管道状态（稳态计算）

        Args:
            dt: 时间步长
            upstream_head: 上游水头 (m)
            downstream_head: 下游水头 (m)

        Returns:
            更新后的状态
        """
        # 稳态流量计算 (迭代)
        delta_h = upstream_head - downstream_head
        if abs(delta_h) < 0.01:
            flow = 0.0
        else:
            # 初始估计
            g = 9.80665
            flow_sign = 1.0 if delta_h > 0 else -1.0
            delta_h = abs(delta_h)

            # 迭代求解
            flow = flow_sign * math.sqrt(2 * g * delta_h * self.diameter /
                                          (self.roughness * self.length)) * self.area
            for _ in range(10):
                hf = abs(self.get_head_loss(flow))
                if abs(hf - delta_h) < 0.01:
                    break
                flow = flow_sign * math.sqrt(delta_h / max(hf, 0.01)) * abs(flow)

        self._state.time += dt
        self._state.flow = flow
        self._state.velocity = self.get_velocity(flow)
        self._state.head_loss = self.get_head_loss(flow)
        self._state.pressure_head = upstream_head - self._state.head_loss / 2

        return self._state


class BasePumpStation(BaseHydraulicComponent):
    """
    泵站模型基类

    实现泵站的扬程-流量特性
    """

    def __init__(self, name: str,
                 pump_count: int,
                 design_flow: float,
                 design_head: float,
                 power_rating: float,
                 peak_efficiency: float = 0.85,
                 hq_curve: Optional[Callable[[float], float]] = None,
                 efficiency_curve: Optional[Callable[[float], float]] = None):
        """
        初始化泵站模型

        Args:
            name: 泵站名称
            pump_count: 机组数量
            design_flow: 单机设计流量 (m³/s)
            design_head: 设计扬程 (m)
            power_rating: 单机功率 (kW)
            peak_efficiency: 峰值效率
            hq_curve: 扬程-流量曲线 Q -> H
            efficiency_curve: 效率曲线 Q_ratio -> eta
        """
        super().__init__(name, ComponentType.PUMP_STATION)
        self.pump_count = pump_count
        self.design_flow = design_flow
        self.design_head = design_head
        self.power_rating = power_rating
        self.peak_efficiency = peak_efficiency
        self._hq_curve = hq_curve
        self._efficiency_curve = efficiency_curve
        self.running_pumps = 0

    def get_head(self, flow: float, running_pumps: Optional[int] = None) -> float:
        """
        计算泵扬程

        Args:
            flow: 总流量 (m³/s)
            running_pumps: 运行机组数

        Returns:
            扬程 (m)
        """
        n = running_pumps if running_pumps is not None else self.running_pumps
        if n <= 0:
            return 0.0

        flow_per_pump = flow / n

        if self._hq_curve:
            return self._hq_curve(flow_per_pump)

        # 默认抛物线特性: H = H_des * (1.2 - 0.2*(Q/Q_des)²)
        q_ratio = flow_per_pump / self.design_flow
        q_ratio = max(0, min(q_ratio, 1.5))  # 限制范围
        return self.design_head * (1.2 - 0.2 * q_ratio ** 2)

    def get_efficiency(self, flow: float, running_pumps: Optional[int] = None) -> float:
        """
        计算效率

        Args:
            flow: 总流量 (m³/s)
            running_pumps: 运行机组数

        Returns:
            效率 (0-1)
        """
        n = running_pumps if running_pumps is not None else self.running_pumps
        if n <= 0:
            return 0.0

        q_ratio = (flow / n) / self.design_flow

        if self._efficiency_curve:
            return self._efficiency_curve(q_ratio)

        # 默认效率曲线（抛物线）
        # 在设计点附近效率最高
        eta = self.peak_efficiency * (1.0 - 0.5 * (q_ratio - 1.0) ** 2)
        return max(0.5, min(eta, self.peak_efficiency))

    def get_power(self, flow: float, head: float, efficiency: float) -> float:
        """
        计算功率

        P = ρ * g * Q * H / η
        """
        if efficiency <= 0:
            return 0.0
        rho = 998.2
        g = 9.80665
        power_w = rho * g * flow * head / efficiency
        return power_w / 1000.0  # kW

    def get_head_loss(self, flow: float) -> float:
        """泵站提供正水头（负损失）"""
        return -self.get_head(flow)

    def update(self, dt: float, flow: float = 0.0,
               running_pumps: Optional[int] = None) -> HydraulicState:
        """
        更新泵站状态

        Args:
            dt: 时间步长
            flow: 流量 (m³/s)
            running_pumps: 运行机组数

        Returns:
            更新后的状态
        """
        if running_pumps is not None:
            self.running_pumps = running_pumps

        head = self.get_head(flow)
        efficiency = self.get_efficiency(flow)
        power = self.get_power(flow, head, efficiency)

        self._state.time += dt
        self._state.flow = flow
        self._state.pressure_head = head
        self._state.power = power
        self._state.efficiency = efficiency

        # 检查状态
        if efficiency < 0.6:
            self.status = ComponentStatus.WARNING
        elif self.running_pumps > self.pump_count:
            self.status = ComponentStatus.FAULT
        else:
            self.status = ComponentStatus.NORMAL

        self._state.status = self.status
        return self._state


class BaseValve(BaseHydraulicComponent):
    """
    阀门模型基类

    实现阀门的流阻特性
    """

    def __init__(self, name: str,
                 diameter: float,
                 cv_coefficient: float = 1.0,
                 zeta_curve: Optional[Callable[[float], float]] = None):
        """
        初始化阀门模型

        Args:
            name: 阀门名称
            diameter: 直径 (m)
            cv_coefficient: 流量系数
            zeta_curve: 阻力系数曲线 opening% -> zeta
        """
        super().__init__(name, ComponentType.VALVE)
        self.diameter = diameter
        self.cv_coefficient = cv_coefficient
        self._zeta_curve = zeta_curve
        self.opening = 100.0  # 开度 (%)

    @property
    def area(self) -> float:
        """阀门面积"""
        return math.pi * (self.diameter / 2) ** 2

    def get_zeta(self, opening: Optional[float] = None) -> float:
        """
        获取阻力系数

        Args:
            opening: 开度 (%)

        Returns:
            阻力系数
        """
        op = opening if opening is not None else self.opening

        if self._zeta_curve:
            return self._zeta_curve(op)

        # 默认阀门特性（近似蝶阀）
        if op <= 0:
            return 1e8
        elif op >= 100:
            return 0.3
        else:
            # 指数关系
            return 0.3 + 100.0 * math.exp(-0.05 * op)

    def get_head_loss(self, flow: float, opening: Optional[float] = None) -> float:
        """
        计算水头损失

        hf = ζ * V² / (2g)
        """
        if abs(flow) < 1e-9:
            return 0.0

        zeta = self.get_zeta(opening)
        velocity = abs(flow) / self.area
        g = 9.80665
        hf = zeta * velocity ** 2 / (2 * g)
        return hf if flow >= 0 else -hf

    def update(self, dt: float, flow: float = 0.0,
               target_opening: Optional[float] = None,
               rate_limit: float = 1.0) -> HydraulicState:
        """
        更新阀门状态

        Args:
            dt: 时间步长
            flow: 流量
            target_opening: 目标开度
            rate_limit: 开度变化速率限制 (%/s)

        Returns:
            更新后的状态
        """
        # 开度变化（带速率限制）
        if target_opening is not None:
            delta = target_opening - self.opening
            max_delta = rate_limit * dt
            if abs(delta) > max_delta:
                delta = max_delta if delta > 0 else -max_delta
            self.opening = max(0, min(100, self.opening + delta))

        self._state.time += dt
        self._state.flow = flow
        self._state.head_loss = self.get_head_loss(flow)
        self._state.velocity = flow / self.area if self.area > 0 else 0

        return self._state


class BaseChannel(BaseHydraulicComponent):
    """
    明渠模型基类

    实现明渠的均匀流计算（曼宁公式）
    """

    def __init__(self, name: str,
                 length: float,
                 bottom_width: float,
                 side_slope: float,
                 bed_slope: float,
                 manning_n: float = 0.025):
        """
        初始化明渠模型（梯形断面）

        Args:
            name: 渠道名称
            length: 长度 (m)
            bottom_width: 底宽 (m)
            side_slope: 边坡 (水平:垂直)
            bed_slope: 纵坡
            manning_n: 曼宁糙率系数
        """
        super().__init__(name, ComponentType.CHANNEL)
        self.length = length
        self.bottom_width = bottom_width
        self.side_slope = side_slope
        self.bed_slope = bed_slope
        self.manning_n = manning_n

    def get_area(self, depth: float) -> float:
        """计算过水断面积"""
        return (self.bottom_width + self.side_slope * depth) * depth

    def get_wetted_perimeter(self, depth: float) -> float:
        """计算湿周"""
        return self.bottom_width + 2 * depth * math.sqrt(1 + self.side_slope ** 2)

    def get_hydraulic_radius(self, depth: float) -> float:
        """计算水力半径"""
        P = self.get_wetted_perimeter(depth)
        if P <= 0:
            return 0
        return self.get_area(depth) / P

    def get_normal_depth(self, flow: float) -> float:
        """
        计算正常水深（曼宁公式反算）

        Q = (1/n) * A * R^(2/3) * S^(1/2)
        """
        if flow <= 0 or self.bed_slope <= 0:
            return 0.0

        # 二分法求解
        y_low, y_high = 0.01, 20.0
        for _ in range(50):
            y = (y_low + y_high) / 2
            A = self.get_area(y)
            R = self.get_hydraulic_radius(y)
            Q_calc = (1.0 / self.manning_n) * A * R ** (2/3) * self.bed_slope ** 0.5
            if abs(Q_calc - flow) / max(flow, 1e-6) < 0.001:
                return y
            if Q_calc < flow:
                y_low = y
            else:
                y_high = y
        return y

    def get_head_loss(self, flow: float) -> float:
        """计算水头损失（等于高程差）"""
        return self.length * self.bed_slope

    def update(self, dt: float, flow: float = 0.0) -> HydraulicState:
        """更新渠道状态"""
        depth = self.get_normal_depth(flow)
        area = self.get_area(depth)
        velocity = flow / area if area > 0 else 0

        self._state.time += dt
        self._state.flow = flow
        self._state.level = depth
        self._state.velocity = velocity
        self._state.head_loss = self.get_head_loss(flow)

        return self._state


__all__ = [
    'ComponentType',
    'ComponentStatus',
    'HydraulicState',
    'BaseHydraulicComponent',
    'BaseReservoir',
    'BasePipeline',
    'BasePumpStation',
    'BaseValve',
    'BaseChannel'
]
