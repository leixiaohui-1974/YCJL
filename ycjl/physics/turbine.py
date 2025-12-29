"""
水轮机模型
=========

基于文得根水电站水轮机特性参数，实现:
- Hill Chart效率插值
- 功率-流量计算
- 空化校核
- 机组启停逻辑
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum, auto

from ..config.config_database import (
    ProjectParams, CurveDatabase, SourceConfig, GlobalConfig
)


class TurbineState(Enum):
    """水轮机运行状态"""
    STOPPED = auto()        # 停机
    STARTING = auto()       # 启动中
    RUNNING = auto()        # 运行
    STOPPING = auto()       # 停机中
    FAULT = auto()          # 故障
    MAINTENANCE = auto()    # 检修


class TurbineType(Enum):
    """水轮机类型"""
    LARGE = auto()          # 大机组 HLTF60-LJ-225
    SMALL = auto()          # 小机组


@dataclass
class TurbineSpec:
    """水轮机规格参数"""
    turbine_type: TurbineType
    rated_power: float      # 额定功率 (MW)
    rated_flow: float       # 额定流量 (m³/s)
    rated_head: float       # 额定水头 (m)
    max_head: float         # 最大水头 (m)
    min_head: float         # 最小水头 (m)
    runner_diameter: float  # 转轮直径 (m)
    rated_speed: float      # 额定转速 (rpm)
    runaway_speed: float    # 飞逸转速 (rpm)

    @classmethod
    def large_unit(cls) -> 'TurbineSpec':
        """大机组规格"""
        return cls(
            turbine_type=TurbineType.LARGE,
            rated_power=SourceConfig.TURBINE_L_POWER,
            rated_flow=SourceConfig.TURBINE_L_FLOW,
            rated_head=SourceConfig.HEAD_RATED,
            max_head=SourceConfig.HEAD_MAX,
            min_head=SourceConfig.HEAD_MIN,
            runner_diameter=2.25,
            rated_speed=214.3,
            runaway_speed=385.7
        )

    @classmethod
    def small_unit(cls) -> 'TurbineSpec':
        """小机组规格"""
        return cls(
            turbine_type=TurbineType.SMALL,
            rated_power=SourceConfig.TURBINE_S_POWER,
            rated_flow=SourceConfig.TURBINE_S_FLOW,
            rated_head=SourceConfig.HEAD_RATED,
            max_head=SourceConfig.HEAD_MAX,
            min_head=SourceConfig.HEAD_MIN,
            runner_diameter=1.6,
            rated_speed=300.0,
            runaway_speed=540.0
        )


@dataclass
class TurbineOperatingPoint:
    """水轮机运行工况点"""
    head: float             # 水头 (m)
    flow: float             # 流量 (m³/s)
    power: float            # 功率 (MW)
    efficiency: float       # 效率
    guide_vane_opening: float  # 导叶开度 (%)
    cavitation_sigma: float    # 空化系数
    is_valid: bool          # 是否在有效运行区


class HillChartInterpolator:
    """
    Hill Chart双线性插值器

    基于数字化的Hill Chart骨架点进行插值
    """

    def __init__(self, hill_points: List[Tuple[float, float, float]]):
        """
        初始化插值器

        Args:
            hill_points: [(水头, 功率, 效率), ...]
        """
        self.points = np.array(hill_points)
        self.heads = self.points[:, 0]
        self.powers = self.points[:, 1]
        self.efficiencies = self.points[:, 2]

        # 构建边界
        self.head_min = self.heads.min()
        self.head_max = self.heads.max()
        self.power_min = self.powers.min()
        self.power_max = self.powers.max()

    def get_efficiency(self, head: float, power: float) -> float:
        """
        双线性插值获取效率

        使用反距离加权插值
        """
        # 边界检查
        head = np.clip(head, self.head_min, self.head_max)
        power = np.clip(power, self.power_min, self.power_max)

        # 计算到各点的距离
        dh = (self.heads - head) / (self.head_max - self.head_min)
        dp = (self.powers - power) / (self.power_max - self.power_min)
        distances = np.sqrt(dh**2 + dp**2)

        # 检查是否有精确匹配
        if np.min(distances) < 1e-6:
            idx = np.argmin(distances)
            return float(self.efficiencies[idx])

        # 反距离加权
        weights = 1.0 / (distances + 1e-10)
        weights = weights / np.sum(weights)

        return float(np.sum(weights * self.efficiencies))

    def get_max_efficiency_power(self, head: float) -> float:
        """获取给定水头下最佳效率对应的功率"""
        # 筛选该水头附近的点
        head_range = 2.0  # ±2m
        mask = np.abs(self.heads - head) < head_range

        if not np.any(mask):
            # 无匹配点，返回中间值
            return (self.power_min + self.power_max) / 2

        subset_powers = self.powers[mask]
        subset_effs = self.efficiencies[mask]

        # 返回最高效率对应的功率
        best_idx = np.argmax(subset_effs)
        return float(subset_powers[best_idx])


class Turbine:
    """
    水轮机模型

    包含:
    - 效率特性计算
    - 功率-流量关系
    - 空化校核
    - 启停控制逻辑
    """

    def __init__(self, unit_id: int, spec: TurbineSpec):
        """
        初始化水轮机

        Args:
            unit_id: 机组编号
            spec: 机组规格
        """
        self.unit_id = unit_id
        self.spec = spec

        # 状态变量
        self.state = TurbineState.STOPPED
        self.guide_vane_opening = 0.0       # 导叶开度 (0-1)
        self.target_opening = 0.0           # 目标开度
        self.current_power = 0.0            # 当前功率 (MW)
        self.current_flow = 0.0             # 当前流量 (m³/s)
        self.current_efficiency = 0.0       # 当前效率
        self.running_hours = 0.0            # 累计运行小时

        # 启停计时
        self.state_timer = 0.0
        self.startup_time = 120.0           # 启动时间 (s)
        self.shutdown_time = 90.0           # 停机时间 (s)

        # 初始化Hill Chart插值器
        hill_data = (CurveDatabase.TURBINE_LARGE_HILL_POINTS
                     if spec.turbine_type == TurbineType.LARGE
                     else CurveDatabase.TURBINE_SMALL_HILL_POINTS)
        self.hill_chart = HillChartInterpolator(hill_data)

        # 历史记录
        self.history: List[TurbineOperatingPoint] = []

    def compute_efficiency(self, head: float, power: float) -> float:
        """计算效率"""
        return self.hill_chart.get_efficiency(head, power)

    def compute_flow(self, head: float, power: float) -> float:
        """
        根据水头和功率计算流量

        P = rho * g * Q * H * eta
        Q = P / (rho * g * H * eta)
        """
        eta = self.compute_efficiency(head, power)
        if head <= 0 or eta <= 0:
            return 0.0

        rho = GlobalConfig.RHO_WATER
        g = GlobalConfig.G

        # 功率单位转换: MW -> W
        power_w = power * 1e6

        flow = power_w / (rho * g * head * eta)
        return flow

    def compute_power(self, head: float, flow: float) -> float:
        """
        根据水头和流量计算功率

        首先估算功率，然后迭代求解
        """
        if head <= 0 or flow <= 0:
            return 0.0

        rho = GlobalConfig.RHO_WATER
        g = GlobalConfig.G

        # 初始估算 (假设效率0.9)
        power_est = rho * g * flow * head * 0.9 / 1e6

        # 迭代求解
        for _ in range(10):
            eta = self.compute_efficiency(head, power_est)
            power_new = rho * g * flow * head * eta / 1e6
            if abs(power_new - power_est) < 0.01:
                break
            power_est = power_new

        return power_est

    def compute_cavitation_sigma(self, head: float, tailwater_el: float,
                                 runner_el: float = 335.0) -> float:
        """
        计算空化系数

        sigma = (H_atm - H_vapor - H_s) / H
        H_s = 转轮中心高程 - 尾水位

        Args:
            head: 水头 (m)
            tailwater_el: 尾水位高程 (m)
            runner_el: 转轮中心高程 (m)
        """
        H_atm = GlobalConfig.PATM_HEAD
        H_vapor = abs(GlobalConfig.VAPOR_PRESSURE_HEAD)
        H_s = runner_el - tailwater_el

        if head <= 0:
            return 1.0

        sigma = (H_atm - H_vapor - H_s) / head
        return sigma

    def check_operating_range(self, head: float, power: float) -> bool:
        """检查是否在有效运行区"""
        # 水头范围
        if head < self.spec.min_head or head > self.spec.max_head:
            return False

        # 功率范围 (10% ~ 100%)
        if power < self.spec.rated_power * 0.1 or power > self.spec.rated_power:
            return False

        return True

    def compute_guide_vane_opening(self, flow: float, head: float) -> float:
        """
        根据流量和水头计算导叶开度

        简化模型：开度与流量成正比，水头修正
        """
        head_factor = math.sqrt(self.spec.rated_head / max(head, 1.0))
        opening = (flow / self.spec.rated_flow) * head_factor
        return np.clip(opening, 0.0, 1.0)

    def start(self):
        """启动机组"""
        if self.state == TurbineState.STOPPED:
            self.state = TurbineState.STARTING
            self.state_timer = 0.0

    def stop(self):
        """停止机组"""
        if self.state == TurbineState.RUNNING:
            self.state = TurbineState.STOPPING
            self.state_timer = 0.0

    def set_target_opening(self, opening: float):
        """设置目标导叶开度"""
        self.target_opening = np.clip(opening, 0.0, 1.0)

    def step(self, dt: float, head: float, tailwater_el: float) -> TurbineOperatingPoint:
        """
        推进一个时间步

        Args:
            dt: 时间步长 (s)
            head: 当前水头 (m)
            tailwater_el: 尾水位 (m)

        Returns:
            当前运行工况点
        """
        self.state_timer += dt

        # 状态机
        if self.state == TurbineState.STARTING:
            # 启动过程
            progress = min(self.state_timer / self.startup_time, 1.0)
            self.guide_vane_opening = progress * self.target_opening
            if progress >= 1.0:
                self.state = TurbineState.RUNNING

        elif self.state == TurbineState.STOPPING:
            # 停机过程
            progress = min(self.state_timer / self.shutdown_time, 1.0)
            self.guide_vane_opening = (1.0 - progress) * self.guide_vane_opening
            if progress >= 1.0:
                self.state = TurbineState.STOPPED
                self.guide_vane_opening = 0.0

        elif self.state == TurbineState.RUNNING:
            # 正常运行：导叶跟踪目标
            max_rate = 0.02  # 2%/s
            error = self.target_opening - self.guide_vane_opening
            delta = np.clip(error, -max_rate * dt, max_rate * dt)
            self.guide_vane_opening += delta
            self.running_hours += dt / 3600.0

        # 计算流量和功率
        if self.state in [TurbineState.RUNNING, TurbineState.STARTING,
                          TurbineState.STOPPING]:
            # 流量与开度成正比，水头修正
            head_factor = math.sqrt(head / self.spec.rated_head)
            self.current_flow = (self.guide_vane_opening *
                                self.spec.rated_flow * head_factor)
            self.current_power = self.compute_power(head, self.current_flow)
            self.current_efficiency = self.compute_efficiency(
                head, self.current_power)
        else:
            self.current_flow = 0.0
            self.current_power = 0.0
            self.current_efficiency = 0.0

        # 空化校核
        sigma = self.compute_cavitation_sigma(head, tailwater_el)

        # 运行区检查
        is_valid = self.check_operating_range(head, self.current_power)

        point = TurbineOperatingPoint(
            head=head,
            flow=self.current_flow,
            power=self.current_power,
            efficiency=self.current_efficiency,
            guide_vane_opening=self.guide_vane_opening,
            cavitation_sigma=sigma,
            is_valid=is_valid
        )

        self.history.append(point)
        return point

    def get_optimal_opening(self, head: float, target_power: float) -> float:
        """
        获取最佳导叶开度

        根据目标功率和当前水头计算
        """
        target_flow = self.compute_flow(head, target_power)
        return self.compute_guide_vane_opening(target_flow, head)

    def reset(self):
        """重置机组状态"""
        self.state = TurbineState.STOPPED
        self.guide_vane_opening = 0.0
        self.target_opening = 0.0
        self.current_power = 0.0
        self.current_flow = 0.0
        self.current_efficiency = 0.0
        self.state_timer = 0.0
        self.history.clear()


class PowerStation:
    """
    水电站模型

    管理多台水轮机组的协调运行
    """

    def __init__(self):
        """初始化电站"""
        # 创建机组
        self.units: Dict[int, Turbine] = {}

        # 大机组 (3台)
        for i in range(SourceConfig.TURBINE_L_COUNT):
            spec = TurbineSpec.large_unit()
            self.units[i + 1] = Turbine(i + 1, spec)

        # 小机组 (1台)
        spec = TurbineSpec.small_unit()
        self.units[4] = Turbine(4, spec)

        # 电站状态
        self.total_power = 0.0
        self.total_flow = 0.0
        self.available_units = list(self.units.keys())

    def get_total_capacity(self) -> float:
        """获取总装机容量 (MW)"""
        return (SourceConfig.TURBINE_L_COUNT * SourceConfig.TURBINE_L_POWER +
                SourceConfig.TURBINE_S_COUNT * SourceConfig.TURBINE_S_POWER)

    def get_running_units(self) -> List[int]:
        """获取运行中的机组编号"""
        return [uid for uid, unit in self.units.items()
                if unit.state == TurbineState.RUNNING]

    def dispatch_power(self, target_power: float, head: float) -> Dict[int, float]:
        """
        功率分配

        根据目标功率在机组间分配负荷

        Args:
            target_power: 目标功率 (MW)
            head: 当前水头 (m)

        Returns:
            各机组目标功率
        """
        allocation = {}
        remaining_power = target_power

        # 优先使用大机组
        large_units = [uid for uid, u in self.units.items()
                       if u.spec.turbine_type == TurbineType.LARGE]
        small_units = [uid for uid, u in self.units.items()
                       if u.spec.turbine_type == TurbineType.SMALL]

        # 确定需要多少台大机组
        large_power = SourceConfig.TURBINE_L_POWER
        small_power = SourceConfig.TURBINE_S_POWER

        n_large_needed = min(
            int(remaining_power / large_power) + 1,
            len(large_units)
        )

        # 分配大机组
        for i, uid in enumerate(large_units[:n_large_needed]):
            if remaining_power >= large_power:
                allocation[uid] = large_power
                remaining_power -= large_power
            elif remaining_power > large_power * 0.3:  # 最低30%负荷
                allocation[uid] = remaining_power
                remaining_power = 0
            else:
                allocation[uid] = 0

        # 如果剩余功率较小，使用小机组
        if 0 < remaining_power <= small_power:
            for uid in small_units:
                if remaining_power > small_power * 0.2:
                    allocation[uid] = min(remaining_power, small_power)
                    remaining_power = 0
                    break

        return allocation

    def step(self, dt: float, head: float, tailwater_el: float,
             target_power: float = None) -> Dict:
        """
        推进一个时间步

        Args:
            dt: 时间步长 (s)
            head: 水头 (m)
            tailwater_el: 尾水位 (m)
            target_power: 目标功率 (MW)，如为None则维持当前状态

        Returns:
            电站运行状态
        """
        # 功率分配
        if target_power is not None:
            allocation = self.dispatch_power(target_power, head)

            # 启停控制
            for uid, unit in self.units.items():
                if uid in allocation and allocation[uid] > 0:
                    if unit.state == TurbineState.STOPPED:
                        unit.start()
                    opening = unit.get_optimal_opening(head, allocation[uid])
                    unit.set_target_opening(opening)
                else:
                    if unit.state == TurbineState.RUNNING:
                        unit.stop()

        # 更新各机组
        self.total_power = 0.0
        self.total_flow = 0.0
        unit_states = {}

        for uid, unit in self.units.items():
            point = unit.step(dt, head, tailwater_el)
            self.total_power += unit.current_power
            self.total_flow += unit.current_flow
            unit_states[uid] = {
                'state': unit.state.name,
                'power': unit.current_power,
                'flow': unit.current_flow,
                'efficiency': unit.current_efficiency,
                'opening': unit.guide_vane_opening
            }

        return {
            'total_power': self.total_power,
            'total_flow': self.total_flow,
            'running_units': self.get_running_units(),
            'units': unit_states
        }

    def reset(self):
        """重置电站"""
        for unit in self.units.values():
            unit.reset()
        self.total_power = 0.0
        self.total_flow = 0.0


# 导出
__all__ = [
    'TurbineState',
    'TurbineType',
    'TurbineSpec',
    'TurbineOperatingPoint',
    'HillChartInterpolator',
    'Turbine',
    'PowerStation'
]
