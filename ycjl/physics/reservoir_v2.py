"""
水库模型 V2 - 文得根水利枢纽
============================

基于配置数据库 v3.2 实现:
- 精确水位-库容-面积曲线
- 溢洪道弧形闸门精确流量计算
- 进水口多工况流态
- 鱼道运行逻辑
- 水轮机组协调
"""

import math
import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass, field
from datetime import datetime

from ..config.config_database import (
    ProjectParams, CurveDatabase, SourceConfig, GlobalConfig
)
from .turbine import PowerStation, TurbineState


@dataclass
class ReservoirStateV2:
    """水库状态 (增强版)"""
    timestamp: Optional[datetime] = None
    level: float = 0.0              # 水位 (m)
    storage: float = 0.0            # 蓄水量 (m³)
    surface_area: float = 0.0       # 水面面积 (m²)
    inflow: float = 0.0             # 入库流量 (m³/s)
    spillway_flow: float = 0.0      # 溢洪流量 (m³/s)
    intake_flow: float = 0.0        # 引水流量 (m³/s)
    power_flow: float = 0.0         # 发电流量 (m³/s)
    fishway_flow: float = 0.0       # 鱼道流量 (m³/s)
    total_outflow: float = 0.0      # 总出库流量 (m³/s)
    power_output: float = 0.0       # 发电功率 (MW)
    head: float = 0.0               # 发电水头 (m)
    tailwater_level: float = 0.0    # 尾水位 (m)
    fishway_outlet: int = 0         # 鱼道出口编号


@dataclass
class SpillwayGateState:
    """溢洪道闸门状态"""
    gate_id: int
    opening: float          # 开度 (0-1)
    target_opening: float   # 目标开度
    flow: float             # 流量 (m³/s)
    is_fault: bool = False


class WendegenReservoir:
    """
    文得根水库增强模型

    特点:
    - 使用配置数据库 v3.2 精确曲线
    - 支持多闸门独立控制
    - 集成水轮机电站模型
    - 鱼道智能运行
    """

    def __init__(self):
        """初始化水库模型"""
        # 加载配置
        self.cfg = SourceConfig
        self.curves = CurveDatabase
        self.physics = GlobalConfig

        # 状态变量
        self.level = self.cfg.NORMAL_LEVEL
        self.storage = self.curves.get_wendegen_volume(self.level)
        self.surface_area = self.curves.get_wendegen_area(self.level)

        # 入库流量
        self.inflow = 0.0

        # 溢洪道闸门 (5孔)
        self.spillway_gates: List[SpillwayGateState] = [
            SpillwayGateState(gate_id=i, opening=0.0, target_opening=0.0, flow=0.0)
            for i in range(self.cfg.SPILLWAY_GATE_COUNT)
        ]
        self.gate_max_rate = 0.01  # 闸门最大动作速率 (1%/s)

        # 进水口
        self.intake_opening = 0.0
        self.intake_target = 0.0

        # 水电站
        self.power_station = PowerStation()

        # 鱼道
        self.fishway_enabled = False
        self.fishway_outlet = 0

        # 历史记录
        self.history: List[ReservoirStateV2] = []
        self.max_history_length = 10000

    def get_volume(self, level: float) -> float:
        """水位->库容"""
        return self.curves.get_wendegen_volume(level)

    def get_level(self, volume: float) -> float:
        """库容->水位"""
        return self.curves.get_wendegen_level(volume)

    def get_area(self, level: float) -> float:
        """水位->面积"""
        return self.curves.get_wendegen_area(level)

    def get_tailwater_level(self, total_discharge: float) -> float:
        """
        计算尾水位

        Args:
            total_discharge: 总下泄流量 (m³/s)

        Returns:
            尾水位 (m)
        """
        return self.curves.get_dam_tailwater_level(total_discharge)

    def compute_spillway_flow_single(self, gate: SpillwayGateState,
                                      H_up: float) -> float:
        """
        计算单孔溢洪流量

        弧形闸门考虑:
        - 自由出流 / 堰流切换
        - 开度-流量系数关系
        """
        h = H_up - self.cfg.SPILLWAY_WEIR_EL
        if h <= 0 or gate.opening < 0.001:
            return 0.0

        B = self.cfg.SPILLWAY_GATE_WIDTH
        e = gate.opening * self.cfg.SPILLWAY_GATE_HEIGHT  # 实际开度

        # 相对开度
        sigma = min(e / max(h, 0.1), 1.0)

        # 流量系数 (Vuskovic公式)
        Cd = 0.611 * math.sqrt(1 + 0.045 * sigma)

        g = self.physics.G

        if sigma >= 0.9:
            # 堰流
            Q = 0.42 * B * (h ** 1.5) * math.sqrt(2 * g)
        else:
            # 孔流
            Q = Cd * B * e * math.sqrt(2 * g * h)

        return Q

    def compute_total_spillway_flow(self) -> float:
        """计算溢洪道总流量"""
        total = 0.0
        for gate in self.spillway_gates:
            gate.flow = self.compute_spillway_flow_single(gate, self.level)
            total += gate.flow
        return total

    def compute_intake_flow(self, downstream_head: float = 0.0) -> float:
        """
        计算进水口流量

        Args:
            downstream_head: 下游水头 (隧洞入口压力)

        Returns:
            引水流量 (m³/s)
        """
        if self.intake_opening < 0.001:
            return 0.0

        # 水头
        H_up = self.level - self.cfg.INTAKE_SILL_EL
        if H_up <= 0:
            return 0.0

        # 进水口尺寸
        B = self.cfg.INTAKE_GATE_WIDTH
        H_gate = self.cfg.INTAKE_GATE_HEIGHT
        e = self.intake_opening * H_gate

        # 进口损失
        k_gate = self.curves.INTAKE_GATE_LOSS_COEF
        k_rack = self.curves.INTAKE_TRASH_RACK_LOSS_COEF
        k_total = k_gate / max(self.intake_opening, 0.1) + k_rack

        # 有效水头
        H_eff = H_up - downstream_head
        if H_eff <= 0:
            return 0.0

        # 流量
        A = B * e
        g = self.physics.G
        Q = A * math.sqrt(2 * g * H_eff / (1 + k_total))

        # 限制在设计流量内
        return min(Q, self.cfg.INTAKE_DESIGN_FLOW)

    def compute_fishway_flow(self) -> Tuple[float, int]:
        """
        计算鱼道流量和出口

        Returns:
            (流量, 出口编号)
        """
        if not self.fishway_enabled:
            return 0.0, 0

        # 根据水位选择出口
        outlet = self.curves.get_fishway_outlet(self.level)
        if outlet == 0:
            return 0.0, 0

        # 鱼道流量 (设计流量)
        Q = self.curves.FISHWAY_DESIGN_FLOW

        return Q, outlet

    def update_gates(self, dt: float):
        """更新闸门位置"""
        for gate in self.spillway_gates:
            if gate.is_fault:
                continue
            error = gate.target_opening - gate.opening
            delta = np.clip(error, -self.gate_max_rate * dt, self.gate_max_rate * dt)
            gate.opening = np.clip(gate.opening + delta, 0.0, 1.0)

        # 进水口闸门
        error = self.intake_target - self.intake_opening
        delta = np.clip(error, -self.gate_max_rate * dt, self.gate_max_rate * dt)
        self.intake_opening = np.clip(self.intake_opening + delta, 0.0, 1.0)

    def step(self, dt: float, inflow: float = None,
             power_target: float = None,
             timestamp: datetime = None) -> ReservoirStateV2:
        """
        推进一个时间步

        Args:
            dt: 时间步长 (s)
            inflow: 入库流量 (m³/s)
            power_target: 发电目标功率 (MW)
            timestamp: 时间戳

        Returns:
            当前水库状态
        """
        if inflow is not None:
            self.inflow = inflow

        # 更新闸门位置
        self.update_gates(dt)

        # 计算各出流
        Q_spill = self.compute_total_spillway_flow()
        Q_intake = self.compute_intake_flow()
        Q_fishway, fishway_outlet = self.compute_fishway_flow()

        # 计算尾水位和水头
        total_discharge = Q_spill + Q_intake + Q_fishway
        tailwater = self.get_tailwater_level(total_discharge)
        head = self.level - tailwater

        # 更新电站
        station_state = self.power_station.step(
            dt, head, tailwater, power_target)
        Q_power = station_state['total_flow']
        P_power = station_state['total_power']

        # 水量平衡
        total_outflow = Q_spill + Q_intake + Q_fishway + Q_power
        net_flow = self.inflow - total_outflow
        self.storage += net_flow * dt

        # 约束库容
        V_dead = self.get_volume(self.cfg.DEAD_LEVEL)
        V_max = self.get_volume(self.cfg.CHECK_FLOOD_LEVEL)
        self.storage = np.clip(self.storage, V_dead, V_max)

        # 更新水位和面积
        self.level = self.get_level(self.storage)
        self.surface_area = self.get_area(self.level)
        self.fishway_outlet = fishway_outlet

        # 构造状态
        state = ReservoirStateV2(
            timestamp=timestamp,
            level=self.level,
            storage=self.storage,
            surface_area=self.surface_area,
            inflow=self.inflow,
            spillway_flow=Q_spill,
            intake_flow=Q_intake,
            power_flow=Q_power,
            fishway_flow=Q_fishway,
            total_outflow=total_outflow,
            power_output=P_power,
            head=head,
            tailwater_level=tailwater,
            fishway_outlet=fishway_outlet
        )

        # 记录历史
        self.history.append(state)
        if len(self.history) > self.max_history_length:
            self.history = self.history[-self.max_history_length:]

        return state

    def set_spillway_opening(self, gate_id: int, opening: float):
        """设置溢洪道闸门目标开度"""
        if 0 <= gate_id < len(self.spillway_gates):
            self.spillway_gates[gate_id].target_opening = np.clip(opening, 0.0, 1.0)

    def set_all_spillway_openings(self, openings: List[float]):
        """设置所有溢洪道闸门开度"""
        for i, opening in enumerate(openings):
            if i < len(self.spillway_gates):
                self.spillway_gates[i].target_opening = np.clip(opening, 0.0, 1.0)

    def set_intake_opening(self, opening: float):
        """设置进水口开度"""
        self.intake_target = np.clip(opening, 0.0, 1.0)

    def set_fishway_enabled(self, enabled: bool):
        """设置鱼道启用状态"""
        self.fishway_enabled = enabled

    def set_gate_fault(self, gate_id: int, fault: bool):
        """设置闸门故障状态"""
        if 0 <= gate_id < len(self.spillway_gates):
            self.spillway_gates[gate_id].is_fault = fault

    def get_state(self) -> ReservoirStateV2:
        """获取当前状态"""
        Q_spill = sum(g.flow for g in self.spillway_gates)
        Q_fishway, outlet = self.compute_fishway_flow()
        tailwater = self.get_tailwater_level(Q_spill)

        return ReservoirStateV2(
            level=self.level,
            storage=self.storage,
            surface_area=self.surface_area,
            inflow=self.inflow,
            spillway_flow=Q_spill,
            intake_flow=self.compute_intake_flow(),
            power_flow=self.power_station.total_flow,
            fishway_flow=Q_fishway,
            total_outflow=Q_spill + self.compute_intake_flow() +
                          Q_fishway + self.power_station.total_flow,
            power_output=self.power_station.total_power,
            head=self.level - tailwater,
            tailwater_level=tailwater,
            fishway_outlet=outlet
        )

    def get_operating_summary(self) -> Dict:
        """获取运行摘要"""
        state = self.get_state()
        return {
            'level_m': round(state.level, 2),
            'storage_billion_m3': round(state.storage / 1e8, 3),
            'inflow_m3s': round(state.inflow, 2),
            'spillway_flow_m3s': round(state.spillway_flow, 2),
            'intake_flow_m3s': round(state.intake_flow, 2),
            'power_mw': round(state.power_output, 2),
            'running_units': self.power_station.get_running_units(),
            'fishway_active': self.fishway_enabled,
            'fishway_outlet': state.fishway_outlet,
            'gate_openings': [round(g.opening, 2) for g in self.spillway_gates]
        }

    def reset(self, level: float = None):
        """重置水库状态"""
        self.level = level if level else self.cfg.NORMAL_LEVEL
        self.storage = self.get_volume(self.level)
        self.surface_area = self.get_area(self.level)

        for gate in self.spillway_gates:
            gate.opening = 0.0
            gate.target_opening = 0.0
            gate.flow = 0.0
            gate.is_fault = False

        self.intake_opening = 0.0
        self.intake_target = 0.0
        self.inflow = 0.0
        self.fishway_enabled = False

        self.power_station.reset()
        self.history.clear()


# 导出
__all__ = [
    'ReservoirStateV2',
    'SpillwayGateState',
    'WendegenReservoir'
]
