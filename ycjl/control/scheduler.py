"""
兴利调度模块
===========

基于《引绰济辽工程调度原则和调度运用技术要点》实现:
- 调度图分区判断
- 供水调度决策
- 水库蓄泄策略
- 发电调度优化
"""

import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum, auto

from ..config.config_database import (
    ProjectParams, CurveDatabase, SourceConfig, HydroConfig
)


class OperationZone(Enum):
    """调度分区"""
    UPPER = auto()      # 弃水区 - 超蓄
    NORMAL = auto()     # 正常供水区
    LOWER = auto()      # 限制供水区
    DEAD = auto()       # 死库容区
    FLOOD = auto()      # 防洪调度区


class SupplyMode(Enum):
    """供水模式"""
    FULL = auto()       # 全额供水
    REDUCED = auto()    # 减量供水
    MINIMUM = auto()    # 最小供水
    STOPPED = auto()    # 停止供水
    EMERGENCY = auto()  # 应急供水


class FloodControlLevel(Enum):
    """防洪等级"""
    NORMAL = auto()     # 正常
    ALERT = auto()      # 警戒
    WARNING = auto()    # 预警
    EMERGENCY = auto()  # 紧急


@dataclass
class ScheduleDecision:
    """调度决策"""
    timestamp: datetime
    zone: OperationZone
    supply_mode: SupplyMode
    target_supply_flow: float       # 目标供水流量 (m³/s)
    target_power: float             # 目标发电功率 (MW)
    spillway_flow: float            # 溢洪流量 (m³/s)
    supply_reduction_factor: float  # 供水削减系数 (0-1)
    remarks: List[str] = field(default_factory=list)


@dataclass
class FloodForecast:
    """洪水预报"""
    peak_time: datetime
    peak_flow: float                # 洪峰流量 (m³/s)
    total_volume: float             # 洪水总量 (m³)
    duration_hours: float           # 洪水历时 (h)
    return_period: float            # 重现期 (年)


class OperationRuleChart:
    """
    兴利调度图

    基于Table 3.3-2实现月度调度边界
    """

    def __init__(self):
        """初始化调度图"""
        self.rule_limits = CurveDatabase.OPERATION_RULE_LIMITS
        self.dead_level = SourceConfig.DEAD_LEVEL
        self.normal_level = SourceConfig.NORMAL_LEVEL
        self.flood_limit = SourceConfig.FLOOD_LIMIT_LEVEL

    def get_zone_limits(self, month: int) -> Tuple[float, float]:
        """
        获取指定月份的调度分区边界

        Args:
            month: 月份 (1-12)

        Returns:
            (上限水位, 下限水位)
        """
        if month in self.rule_limits:
            return self.rule_limits[month]
        return (self.normal_level, self.dead_level)

    def get_operation_zone(self, month: int, level: float) -> OperationZone:
        """
        判断调度分区

        Args:
            month: 月份
            level: 当前水位 (m)

        Returns:
            调度分区
        """
        upper, lower = self.get_zone_limits(month)

        # 汛期特殊处理
        if month in [7, 8, 9] and level > self.flood_limit:
            return OperationZone.FLOOD

        if level > upper:
            return OperationZone.UPPER
        elif level >= lower:
            return OperationZone.NORMAL
        elif level > self.dead_level:
            return OperationZone.LOWER
        else:
            return OperationZone.DEAD

    def get_target_level(self, month: int, zone: OperationZone) -> float:
        """获取目标水位"""
        upper, lower = self.get_zone_limits(month)

        if zone == OperationZone.NORMAL:
            # 目标在正常区中部偏上
            return lower + (upper - lower) * 0.6
        elif zone == OperationZone.LOWER:
            # 目标在限制区上部
            return lower + (lower - self.dead_level) * 0.8
        elif zone == OperationZone.UPPER:
            return upper
        else:
            return lower

    def interpolate_limits(self, date: datetime) -> Tuple[float, float]:
        """
        日尺度调度边界插值

        在月边界之间线性插值
        """
        month = date.month
        day = date.day
        days_in_month = 30  # 简化

        # 当月和下月边界
        upper_cur, lower_cur = self.get_zone_limits(month)
        next_month = month % 12 + 1
        upper_next, lower_next = self.get_zone_limits(next_month)

        # 线性插值
        ratio = day / days_in_month
        upper = upper_cur + (upper_next - upper_cur) * ratio
        lower = lower_cur + (lower_next - lower_cur) * ratio

        return upper, lower


class SupplyScheduler:
    """
    供水调度器

    根据水库蓄水状态和用户需求制定供水策略
    """

    def __init__(self):
        """初始化调度器"""
        self.rule_chart = OperationRuleChart()
        self.design_supply = SourceConfig.INTAKE_DESIGN_FLOW

        # 用户需求
        from ..config.config_database import UserConfig
        self.user_demands = UserConfig.FLOW_DEMANDS

    def compute_supply_factor(self, zone: OperationZone,
                              level: float, month: int) -> float:
        """
        计算供水削减系数

        Args:
            zone: 调度分区
            level: 当前水位
            month: 月份

        Returns:
            供水系数 (0-1)
        """
        if zone == OperationZone.DEAD:
            return 0.0  # 停止供水

        if zone == OperationZone.NORMAL:
            return 1.0  # 全额供水

        if zone == OperationZone.UPPER:
            return 1.0  # 可超额供水

        # LOWER区：线性削减
        upper, lower = self.rule_chart.get_zone_limits(month)
        dead = self.rule_chart.dead_level

        if level <= dead:
            return 0.0

        # 从下限到死水位线性降至30%
        ratio = (level - dead) / (lower - dead)
        return 0.3 + 0.7 * ratio

    def get_supply_mode(self, factor: float) -> SupplyMode:
        """根据削减系数确定供水模式"""
        if factor >= 0.95:
            return SupplyMode.FULL
        elif factor >= 0.5:
            return SupplyMode.REDUCED
        elif factor > 0:
            return SupplyMode.MINIMUM
        else:
            return SupplyMode.STOPPED

    def allocate_supply(self, available_flow: float,
                        priority_users: List[str] = None) -> Dict[str, float]:
        """
        分配供水量

        优先保障重要用户

        Args:
            available_flow: 可供水量 (m³/s)
            priority_users: 优先用户列表

        Returns:
            各用户分配流量
        """
        allocation = {}
        remaining = available_flow

        # 默认优先级
        if priority_users is None:
            priority_users = ['Keerqin', 'Zalute', 'Keyouzhong',
                             'DevZone', 'Park', 'Tuquan',
                             'Kezuozhong', 'Kailu']

        # 按优先级分配
        for user in priority_users:
            if user in self.user_demands:
                demand = self.user_demands[user]
                allocated = min(demand, remaining)
                allocation[user] = allocated
                remaining -= allocated

        # 剩余用户
        for user, demand in self.user_demands.items():
            if user not in allocation:
                allocated = min(demand, remaining)
                allocation[user] = allocated
                remaining -= allocated

        return allocation


class FloodDispatcher:
    """
    防洪调度器

    处理汛期洪水调度决策
    """

    def __init__(self):
        """初始化防洪调度器"""
        self.flood_limit = SourceConfig.FLOOD_LIMIT_LEVEL
        self.check_level = SourceConfig.CHECK_FLOOD_LEVEL
        self.design_level = SourceConfig.DESIGN_FLOOD_LEVEL
        self.spillway_capacity = SourceConfig.SPILLWAY_MAX_DISCHARGE

    def get_flood_control_level(self, reservoir_level: float,
                                 forecast: Optional[FloodForecast] = None) -> FloodControlLevel:
        """
        确定防洪等级

        Args:
            reservoir_level: 当前库水位
            forecast: 洪水预报

        Returns:
            防洪等级
        """
        if reservoir_level >= self.check_level:
            return FloodControlLevel.EMERGENCY
        elif reservoir_level >= self.design_level:
            return FloodControlLevel.WARNING
        elif reservoir_level >= self.flood_limit:
            return FloodControlLevel.ALERT
        else:
            return FloodControlLevel.NORMAL

    def compute_required_discharge(self, current_level: float,
                                    target_level: float,
                                    inflow: float,
                                    time_hours: float) -> float:
        """
        计算所需泄量

        Args:
            current_level: 当前水位 (m)
            target_level: 目标水位 (m)
            inflow: 入库流量 (m³/s)
            time_hours: 可用时间 (h)

        Returns:
            所需泄量 (m³/s)
        """
        # 计算需要削减的库容
        v_current = CurveDatabase.get_wendegen_volume(current_level)
        v_target = CurveDatabase.get_wendegen_volume(target_level)
        delta_v = v_current - v_target

        if delta_v <= 0:
            # 不需要泄洪
            return 0.0

        # 所需平均泄量
        time_seconds = time_hours * 3600
        avg_outflow = delta_v / time_seconds + inflow

        return min(avg_outflow, self.spillway_capacity)

    def compute_spillway_opening(self, required_discharge: float,
                                  upstream_level: float) -> List[float]:
        """
        计算溢洪道开度

        Args:
            required_discharge: 所需泄量 (m³/s)
            upstream_level: 上游水位 (m)

        Returns:
            各闸门开度列表 (0-1)
        """
        n_gates = SourceConfig.SPILLWAY_GATE_COUNT
        gate_height = SourceConfig.SPILLWAY_GATE_HEIGHT

        # 堰顶水深
        H = upstream_level - SourceConfig.SPILLWAY_WEIR_EL
        if H <= 0:
            return [0.0] * n_gates

        # 单孔最大泄量
        single_max = CurveDatabase.get_spillway_discharge(upstream_level) / n_gates

        # 分配到各孔
        openings = []
        remaining = required_discharge

        for i in range(n_gates):
            if remaining <= 0:
                openings.append(0.0)
            elif remaining >= single_max:
                openings.append(1.0)
                remaining -= single_max
            else:
                # 部分开度
                ratio = remaining / single_max
                openings.append(ratio)
                remaining = 0

        return openings


class ReservoirScheduler:
    """
    水库综合调度器

    整合供水、发电、防洪调度
    """

    def __init__(self):
        """初始化调度器"""
        self.rule_chart = OperationRuleChart()
        self.supply_scheduler = SupplyScheduler()
        self.flood_dispatcher = FloodDispatcher()

        # 发电参数
        self.power_price_peak = 0.8      # 峰时电价 (元/kWh)
        self.power_price_valley = 0.3   # 谷时电价
        self.power_price_normal = 0.5   # 平时电价

    def get_power_price(self, hour: int) -> float:
        """获取分时电价"""
        if 8 <= hour < 11 or 17 <= hour < 21:
            return self.power_price_peak
        elif 23 <= hour or hour < 6:
            return self.power_price_valley
        else:
            return self.power_price_normal

    def compute_optimal_power(self, level: float, month: int,
                               hour: int, inflow: float) -> float:
        """
        计算最优发电功率

        考虑:
        - 调度分区
        - 电价时段
        - 来水情况
        """
        zone = self.rule_chart.get_operation_zone(month, level)
        price = self.get_power_price(hour)

        # 最大装机
        max_power = (SourceConfig.TURBINE_L_COUNT * SourceConfig.TURBINE_L_POWER +
                     SourceConfig.TURBINE_S_COUNT * SourceConfig.TURBINE_S_POWER)

        # 根据分区调整
        if zone == OperationZone.DEAD:
            return 0.0  # 禁止发电
        elif zone == OperationZone.LOWER:
            # 限制发电
            return max_power * 0.3
        elif zone == OperationZone.UPPER:
            # 加大发电消落
            return max_power
        elif zone == OperationZone.FLOOD:
            # 防洪期尽量发电消落
            return max_power
        else:
            # 正常区：根据电价和来水
            if price == self.power_price_peak:
                return max_power * 0.9
            elif price == self.power_price_valley:
                return max_power * 0.5 if inflow > 50 else max_power * 0.3
            else:
                return max_power * 0.7

    def make_decision(self, current_time: datetime,
                      level: float,
                      inflow: float,
                      demand_factor: float = 1.0,
                      flood_forecast: Optional[FloodForecast] = None) -> ScheduleDecision:
        """
        生成调度决策

        Args:
            current_time: 当前时间
            level: 当前水位 (m)
            inflow: 入库流量 (m³/s)
            demand_factor: 需求系数
            flood_forecast: 洪水预报

        Returns:
            调度决策
        """
        month = current_time.month
        hour = current_time.hour
        remarks = []

        # 1. 判断调度分区
        zone = self.rule_chart.get_operation_zone(month, level)
        remarks.append(f"调度分区: {zone.name}")

        # 2. 计算供水
        supply_factor = self.supply_scheduler.compute_supply_factor(zone, level, month)
        supply_factor *= demand_factor  # 考虑需求变化
        supply_mode = self.supply_scheduler.get_supply_mode(supply_factor)
        target_supply = self.supply_scheduler.design_supply * supply_factor

        if supply_factor < 1.0:
            remarks.append(f"供水削减至{supply_factor*100:.1f}%")

        # 3. 计算发电
        target_power = self.compute_optimal_power(level, month, hour, inflow)

        # 4. 防洪调度
        spillway_flow = 0.0
        if zone == OperationZone.FLOOD or zone == OperationZone.UPPER:
            # 需要泄洪
            target_level = self.flood_dispatcher.flood_limit
            spillway_flow = self.flood_dispatcher.compute_required_discharge(
                level, target_level, inflow, 6.0)  # 6小时内降到汛限
            remarks.append(f"启动泄洪, 目标水位{target_level:.1f}m")

        # 5. 考虑洪水预报
        if flood_forecast and flood_forecast.peak_flow > 3000:
            # 预腾库容
            remarks.append(f"洪水预报: 峰值{flood_forecast.peak_flow:.0f}m³/s")
            if level > self.flood_dispatcher.flood_limit - 2:
                spillway_flow = max(spillway_flow, inflow * 0.5)
                remarks.append("预泄腾库")

        return ScheduleDecision(
            timestamp=current_time,
            zone=zone,
            supply_mode=supply_mode,
            target_supply_flow=target_supply,
            target_power=target_power,
            spillway_flow=spillway_flow,
            supply_reduction_factor=supply_factor,
            remarks=remarks
        )


class FloodHydrograph:
    """
    洪水过程线生成器

    基于标准洪水形状因子生成不同频率的洪水过程
    """

    def __init__(self):
        """初始化生成器"""
        self.pattern_double_peak = CurveDatabase.FLOOD_PATTERN_1998_DOUBLE_PEAK
        self.pattern_single_peak = CurveDatabase.FLOOD_PATTERN_SINGLE_PEAK
        self.flood_peaks = HydroConfig.FLOOD_FREQUENCY

    def generate(self, return_period: str,
                 duration_days: float = 30,
                 pattern: str = 'double_peak',
                 dt_hours: float = 1.0) -> List[Tuple[float, float]]:
        """
        生成洪水过程线

        Args:
            return_period: 重现期 (如 "P=1%")
            duration_days: 洪水历时 (天)
            pattern: 形态 ('double_peak' 或 'single_peak')
            dt_hours: 时间步长 (小时)

        Returns:
            [(时间h, 流量m³/s), ...]
        """
        # 获取洪峰流量
        if return_period in self.flood_peaks:
            peak_flow = self.flood_peaks[return_period]
        else:
            peak_flow = 3000.0  # 默认值

        # 选择形态
        if pattern == 'double_peak':
            shape = self.pattern_double_peak
        else:
            shape = self.pattern_single_peak

        # 生成过程线
        total_hours = duration_days * 24
        times = np.arange(0, total_hours, dt_hours)

        hydrograph = []
        for t in times:
            # 归一化时间
            t_norm = t / total_hours

            # 形状因子插值
            x = [p[0] for p in shape]
            y = [p[1] for p in shape]
            q_factor = np.interp(t_norm, x, y)

            # 实际流量
            q = peak_flow * q_factor
            hydrograph.append((float(t), float(q)))

        return hydrograph

    def generate_annual_inflow(self, year_type: str = 'average',
                               dt_days: float = 1.0) -> List[Tuple[int, float]]:
        """
        生成年来水过程

        Args:
            year_type: 年型 ('wet', 'average', 'dry')
            dt_days: 时间步长 (天)

        Returns:
            [(日序号, 日均流量m³/s), ...]
        """
        # 年来水量
        if year_type == 'wet':
            annual_volume = HydroConfig.ANNUAL_INFLOW_MEAN * 1.3
        elif year_type == 'dry':
            annual_volume = HydroConfig.ANNUAL_INFLOW_P95
        else:
            annual_volume = HydroConfig.ANNUAL_INFLOW_MEAN

        # 月分配
        monthly_pattern = HydroConfig.MONTHLY_INFLOW_PATTERN

        inflow = []
        day = 0
        for month in range(1, 13):
            # 该月来水量
            month_volume = annual_volume * monthly_pattern[month]

            # 该月天数
            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month-1]

            # 日均流量
            daily_flow = month_volume / (days_in_month * 86400)

            for d in range(days_in_month):
                # 添加随机波动
                noise = np.random.normal(1.0, 0.1)
                inflow.append((day, daily_flow * noise))
                day += 1

        return inflow


# 导出
__all__ = [
    'OperationZone',
    'SupplyMode',
    'FloodControlLevel',
    'ScheduleDecision',
    'FloodForecast',
    'OperationRuleChart',
    'SupplyScheduler',
    'FloodDispatcher',
    'ReservoirScheduler',
    'FloodHydrograph'
]
