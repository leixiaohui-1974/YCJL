"""
水库模型 - 文得根水利枢纽
=========================

物理特性:
- 库容-水位关系曲线
- 溢洪道弧形闸门流量特性
- 进水口淹没/自由出流切换
- 水轮机发电流量
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from ..config.settings import Config


@dataclass
class ReservoirState:
    """水库状态"""
    level: float           # 水位 (m)
    storage: float         # 蓄水量 (m³)
    inflow: float         # 入库流量 (m³/s)
    spillway_flow: float  # 溢洪流量 (m³/s)
    intake_flow: float    # 引水流量 (m³/s)
    power_flow: float     # 发电流量 (m³/s)


class Reservoir:
    """
    水库仿真模型

    基于文得根水利枢纽参数，实现:
    - 水量平衡计算
    - 溢洪道弧形闸门流量计算
    - 进水口闸门流量计算（考虑淹没/自由出流）
    - 发电机组流量
    """

    def __init__(self, config: Optional[Config] = None):
        self.cfg = config.reservoir if config else Config.reservoir
        self.physics = Config.physics

        # 状态变量
        self.level = self.cfg.normal_level
        self.storage = self._level_to_storage(self.level)

        # 闸门状态
        self.spillway_openings = np.zeros(self.cfg.spillway_gates)
        self.intake_opening = 0.0
        self.power_units_running = 0

        # 入库流量 (外部输入)
        self.inflow = 0.0

        # 历史记录
        self.history = []

    def _level_to_storage(self, level: float) -> float:
        """水位-库容曲线插值"""
        curve = self.cfg.elevation_volume_curve
        levels = sorted(curve.keys())

        if level <= levels[0]:
            return curve[levels[0]]
        if level >= levels[-1]:
            return curve[levels[-1]]

        for i in range(len(levels) - 1):
            if levels[i] <= level < levels[i + 1]:
                ratio = (level - levels[i]) / (levels[i + 1] - levels[i])
                v1, v2 = curve[levels[i]], curve[levels[i + 1]]
                return v1 + ratio * (v2 - v1)

        return curve[levels[-1]]

    def _storage_to_level(self, storage: float) -> float:
        """库容-水位曲线反算"""
        curve = self.cfg.elevation_volume_curve
        levels = sorted(curve.keys())
        volumes = [curve[l] for l in levels]

        if storage <= volumes[0]:
            return levels[0]
        if storage >= volumes[-1]:
            return levels[-1]

        for i in range(len(volumes) - 1):
            if volumes[i] <= storage < volumes[i + 1]:
                ratio = (storage - volumes[i]) / (volumes[i + 1] - volumes[i])
                return levels[i] + ratio * (levels[i + 1] - levels[i])

        return levels[-1]

    def compute_spillway_flow(self) -> float:
        """
        计算溢洪道总流量

        弧形闸门流量公式 (考虑收缩系数):
        Q = Cd * B * e * sqrt(2 * g * H)

        其中 Cd 随相对开度变化 (Vuskovic经验公式)
        """
        total_flow = 0.0
        H_up = self.level - self.cfg.gate_sill_elevation

        if H_up <= 0:
            return 0.0

        for i in range(self.cfg.spillway_gates):
            e = self.spillway_openings[i] * self.cfg.gate_max_opening
            if e < 0.001:
                continue

            # 相对开度
            sigma = min(e / max(H_up, 0.1), 1.0)

            # 流量系数 (经验公式)
            Cd = 0.611 * np.sqrt(1 + 0.045 * sigma)

            # 判断流态
            if e >= H_up * 0.9:
                # 堰流
                Q = 0.42 * self.cfg.gate_width * H_up ** 1.5 * np.sqrt(2 * self.physics.G)
            else:
                # 孔流
                Q = Cd * self.cfg.gate_width * e * np.sqrt(2 * self.physics.G * H_up)

            total_flow += Q

        return total_flow

    def compute_intake_flow(self, downstream_level: float = 0.0) -> float:
        """
        计算进水口引水流量

        考虑淹没与自由出流的切换:
        - 自由出流: 下游水位 < 收缩断面
        - 淹没出流: 下游水位 >= 收缩断面
        """
        e = self.intake_opening * self.cfg.gate_max_opening
        if e < 0.001:
            return 0.0

        H_up = self.level - self.cfg.intake_sill_elevation
        if H_up <= 0:
            return 0.0

        B = self.cfg.intake_gate_width

        # 淹没判定
        contraction_level = self.cfg.intake_sill_elevation + e * 0.62
        is_submerged = downstream_level > contraction_level

        if is_submerged:
            # 淹没出流
            H_down = downstream_level - self.cfg.intake_sill_elevation
            delta_h = max(H_up - H_down, 0)
            Cs = 0.8  # 淹没系数
            Cd = 0.6
            Q = Cs * Cd * B * e * np.sqrt(2 * self.physics.G * delta_h)
        else:
            # 自由出流
            Cd = 0.62
            Q = Cd * B * e * np.sqrt(2 * self.physics.G * H_up)

        return Q

    def compute_power_flow(self) -> float:
        """
        计算发电流量

        简化为与机组数量成正比
        """
        if self.power_units_running == 0:
            return 0.0

        # 单机额定流量
        unit_flow = 15.0  # m³/s

        # 可用水头
        H_available = self.level - self.cfg.dead_level
        efficiency = np.clip(H_available / 30.0, 0.0, 1.0)

        return self.power_units_running * unit_flow * efficiency

    def step(self, dt: float, inflow: float = None) -> ReservoirState:
        """
        推进一个时间步

        Parameters:
            dt: 时间步长 (s)
            inflow: 入库流量 (m³/s)，如为None则使用上一步值

        Returns:
            ReservoirState: 当前状态
        """
        if inflow is not None:
            self.inflow = inflow

        # 计算各出流
        Q_spill = self.compute_spillway_flow()
        Q_intake = self.compute_intake_flow()
        Q_power = self.compute_power_flow()

        # 水量平衡
        net_flow = self.inflow - Q_spill - Q_intake - Q_power
        self.storage += net_flow * dt

        # 约束库容范围
        min_storage = self._level_to_storage(self.cfg.dead_level)
        max_storage = self._level_to_storage(self.cfg.design_level)
        self.storage = np.clip(self.storage, min_storage, max_storage)

        # 更新水位
        self.level = self._storage_to_level(self.storage)

        state = ReservoirState(
            level=self.level,
            storage=self.storage,
            inflow=self.inflow,
            spillway_flow=Q_spill,
            intake_flow=Q_intake,
            power_flow=Q_power
        )

        self.history.append(state)
        return state

    def set_spillway_opening(self, gate_index: int, opening: float):
        """设置溢洪闸开度 (0~1)"""
        if 0 <= gate_index < self.cfg.spillway_gates:
            self.spillway_openings[gate_index] = np.clip(opening, 0.0, 1.0)

    def set_intake_opening(self, opening: float):
        """设置进水口闸门开度 (0~1)"""
        self.intake_opening = np.clip(opening, 0.0, 1.0)

    def set_power_units(self, num_running: int):
        """设置发电机组运行数量"""
        self.power_units_running = np.clip(num_running, 0, 4)

    def get_state(self) -> ReservoirState:
        """获取当前状态"""
        return ReservoirState(
            level=self.level,
            storage=self.storage,
            inflow=self.inflow,
            spillway_flow=self.compute_spillway_flow(),
            intake_flow=self.compute_intake_flow(),
            power_flow=self.compute_power_flow()
        )

    def reset(self, level: float = None):
        """重置水库状态"""
        self.level = level if level else self.cfg.normal_level
        self.storage = self._level_to_storage(self.level)
        self.spillway_openings[:] = 0.0
        self.intake_opening = 0.0
        self.power_units_running = 0
        self.inflow = 0.0
        self.history.clear()
