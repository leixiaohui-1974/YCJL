"""
冰期控制策略
============

引绰济辽工程冰期运行控制策略:
1. 冰期运行模式识别
2. 流量控制策略
3. 水位约束管理
4. 糙率与过流能力调整
5. 安全保护机制
6. 开河期特殊处理

版本: 3.3.0

设计原则:
- 稳定流量，避免水锤
- 维持足够水深，防止封闭
- 保证最小流速，防止冰塞
- 预防机械开河引发的洪水
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum, auto
from datetime import datetime

from ..config.ice_parameters import (
    IcePhase, BreakupType,
    IceParams, YCJLIceParams, BreakupParams
)
from ..physics.ice_model import IceState, IcePeriodSimulator


class IceOperationMode(Enum):
    """冰期运行模式"""
    NORMAL = auto()           # 正常模式 (无冰期)
    PRE_ICE = auto()          # 预冰期 (11月)
    FREEZE_UP = auto()        # 封冻期
    STABLE_ICE = auto()       # 稳定冰期
    PRE_BREAKUP = auto()      # 预开河期
    BREAKUP = auto()          # 开河期
    POST_ICE = auto()         # 冰后期


class IceAlarmLevel(Enum):
    """冰期报警等级"""
    NORMAL = 0                # 正常
    ATTENTION = 1             # 关注
    WARNING = 2               # 预警
    ALARM = 3                 # 报警
    EMERGENCY = 4             # 紧急


@dataclass
class IceOperationConstraints:
    """冰期运行约束"""
    min_flow_rate: float          # 最小流量 (m³/s)
    max_flow_rate: float          # 最大流量 (m³/s)
    max_flow_change_rate: float   # 最大流量变化率 (%/h)
    min_water_level: float        # 最小水位 (m)
    max_water_level: float        # 最大水位 (m)
    min_velocity: float           # 最小流速 (m/s)
    max_velocity: float           # 最大流速 (m/s)
    roughness_factor: float       # 糙率增加因子
    capacity_factor: float        # 过流能力折减系数


@dataclass
class IceControlDecision:
    """冰期控制决策"""
    mode: IceOperationMode        # 运行模式
    alarm_level: IceAlarmLevel    # 报警等级
    constraints: IceOperationConstraints  # 运行约束
    flow_setpoint: float          # 流量设定值 (m³/s)
    flow_ramp_rate: float         # 流量变化速率 (%/h)
    valve_constraints: Dict[str, Tuple[float, float]]  # 阀门开度约束
    warnings: List[str]           # 警告信息
    recommendations: List[str]    # 操作建议
    timestamp: datetime = field(default_factory=datetime.now)


class IcePeriodController:
    """
    冰期控制器

    根据冰期状态和气象条件生成控制策略
    """

    def __init__(self, params: 'YCJLIceParams' = None):
        self.params = params or IceParams.YCJL
        self.ice_simulator = IcePeriodSimulator()
        self.current_mode = IceOperationMode.NORMAL
        self.last_decision: Optional[IceControlDecision] = None

    def detect_operation_mode(self, ice_state: IceState,
                               air_temp: float,
                               current_month: int) -> IceOperationMode:
        """
        检测当前应采用的运行模式

        Args:
            ice_state: 冰期状态
            air_temp: 气温 (°C)
            current_month: 当前月份

        Returns:
            运行模式
        """
        # 基于月份的初步判断
        ice_months = self.params.STABLE_ICE_MONTHS  # [12, 1, 2]
        start_month = self.params.ICE_PERIOD_START_MONTH  # 11
        end_month = self.params.ICE_PERIOD_END_MONTH  # 3

        # 正常期 (4-10月)
        if 4 <= current_month <= 10:
            return IceOperationMode.NORMAL

        # 预冰期 (11月初)
        if current_month == start_month and ice_state.phase == IcePhase.OPEN_WATER:
            if air_temp < 5:
                return IceOperationMode.PRE_ICE
            return IceOperationMode.NORMAL

        # 封冻期
        if ice_state.phase == IcePhase.FREEZE_UP:
            return IceOperationMode.FREEZE_UP

        # 稳定冰期
        if ice_state.phase == IcePhase.FROZEN:
            return IceOperationMode.STABLE_ICE

        # 预开河期
        if current_month == end_month and ice_state.mdd > 10:
            if ice_state.phase == IcePhase.FROZEN:
                return IceOperationMode.PRE_BREAKUP

        # 开河期
        if ice_state.phase == IcePhase.BREAKUP:
            return IceOperationMode.BREAKUP

        # 冰后期
        if current_month == end_month + 1 or (current_month == 4):
            if ice_state.ice_thickness < 0.1:
                return IceOperationMode.POST_ICE

        # 默认
        if current_month in ice_months:
            return IceOperationMode.STABLE_ICE

        return IceOperationMode.NORMAL

    def get_constraints(self, mode: IceOperationMode,
                        ice_state: IceState) -> IceOperationConstraints:
        """
        获取运行约束

        Args:
            mode: 运行模式
            ice_state: 冰期状态

        Returns:
            运行约束
        """
        # 基础约束 (正常期)
        base_constraints = IceOperationConstraints(
            min_flow_rate=5.0,
            max_flow_rate=18.58,
            max_flow_change_rate=10.0,
            min_water_level=351.0,
            max_water_level=377.0,
            min_velocity=0.3,
            max_velocity=3.0,
            roughness_factor=1.0,
            capacity_factor=1.0
        )

        if mode == IceOperationMode.NORMAL:
            return base_constraints

        elif mode == IceOperationMode.PRE_ICE:
            # 预冰期: 开始降低流量变化率
            return IceOperationConstraints(
                min_flow_rate=5.0,
                max_flow_rate=18.58 * 0.95,
                max_flow_change_rate=5.0,  # 降低变化率
                min_water_level=self.params.MIN_RESERVOIR_LEVEL_ICE,
                max_water_level=377.0,
                min_velocity=self.params.PIPELINE_MIN_VELOCITY,
                max_velocity=2.5,
                roughness_factor=1.05,
                capacity_factor=0.98
            )

        elif mode == IceOperationMode.FREEZE_UP:
            # 封冻期: 严格限制流量变化
            return IceOperationConstraints(
                min_flow_rate=6.0,  # 提高最小流量
                max_flow_rate=18.58 * 0.90,
                max_flow_change_rate=self.params.MAX_DISCHARGE_CHANGE_RATE * 100,  # 5%
                min_water_level=self.params.MIN_RESERVOIR_LEVEL_ICE,
                max_water_level=375.0,
                min_velocity=self.params.PIPELINE_MIN_VELOCITY,
                max_velocity=2.0,
                roughness_factor=1.15,
                capacity_factor=0.90
            )

        elif mode == IceOperationMode.STABLE_ICE:
            # 稳定冰期: 保持稳定运行
            roughness_factor = ice_state.composite_roughness / 0.014 if ice_state.composite_roughness > 0 else 1.3
            capacity_factor = ice_state.conveyance_factor if ice_state.conveyance_factor > 0 else 0.75

            return IceOperationConstraints(
                min_flow_rate=6.0,
                max_flow_rate=18.58 * self.params.FLOW_SAFETY_FACTOR,
                max_flow_change_rate=self.params.MAX_DISCHARGE_CHANGE_RATE * 100,
                min_water_level=self.params.MIN_RESERVOIR_LEVEL_ICE,
                max_water_level=373.0,
                min_velocity=self.params.PIPELINE_MIN_VELOCITY,
                max_velocity=1.5,  # 进一步限制
                roughness_factor=roughness_factor,
                capacity_factor=capacity_factor
            )

        elif mode == IceOperationMode.PRE_BREAKUP:
            # 预开河期: 准备降低水位
            return IceOperationConstraints(
                min_flow_rate=5.0,
                max_flow_rate=18.58 * 0.80,
                max_flow_change_rate=3.0,  # 更严格
                min_water_level=self.params.MIN_RESERVOIR_LEVEL_ICE - 2,
                max_water_level=370.0,  # 降低水位
                min_velocity=0.3,
                max_velocity=1.5,
                roughness_factor=1.2,
                capacity_factor=0.85
            )

        elif mode == IceOperationMode.BREAKUP:
            # 开河期: 最严格限制
            return IceOperationConstraints(
                min_flow_rate=3.0,  # 可以降低
                max_flow_rate=18.58 * 0.70,
                max_flow_change_rate=2.0,  # 极严格
                min_water_level=360.0,  # 可以更低
                max_water_level=368.0,  # 留洪水余量
                min_velocity=0.2,
                max_velocity=1.0,
                roughness_factor=1.1,
                capacity_factor=0.90
            )

        elif mode == IceOperationMode.POST_ICE:
            # 冰后期: 逐步恢复
            return IceOperationConstraints(
                min_flow_rate=5.0,
                max_flow_rate=18.58 * 0.95,
                max_flow_change_rate=8.0,  # 逐步放宽
                min_water_level=355.0,
                max_water_level=377.0,
                min_velocity=0.3,
                max_velocity=2.5,
                roughness_factor=1.02,
                capacity_factor=0.98
            )

        return base_constraints

    def evaluate_alarm_level(self, ice_state: IceState,
                              mode: IceOperationMode,
                              current_flow: float,
                              constraints: IceOperationConstraints) -> IceAlarmLevel:
        """
        评估报警等级

        Args:
            ice_state: 冰期状态
            mode: 运行模式
            current_flow: 当前流量 (m³/s)
            constraints: 运行约束

        Returns:
            报警等级
        """
        level = IceAlarmLevel.NORMAL

        # 流量超限
        if current_flow > constraints.max_flow_rate * 1.1:
            level = max(level, IceAlarmLevel.ALARM)
        elif current_flow > constraints.max_flow_rate:
            level = max(level, IceAlarmLevel.WARNING)

        # 冰厚异常
        if ice_state.ice_thickness > self.params.MAX_ICE_THICKNESS:
            level = max(level, IceAlarmLevel.ALARM)
        elif ice_state.ice_thickness > self.params.DESIGN_ICE_THICKNESS:
            level = max(level, IceAlarmLevel.WARNING)

        # 开河期特殊判断
        if mode == IceOperationMode.BREAKUP:
            # 机械开河风险
            if ice_state.mdd < BreakupParams.MDD_THRESHOLD * 0.5:
                level = max(level, IceAlarmLevel.WARNING)

        # 封冻期frazil浓度
        if mode == IceOperationMode.FREEZE_UP:
            if ice_state.frazil_concentration > 0.1:
                level = max(level, IceAlarmLevel.ATTENTION)
            if ice_state.frazil_concentration > 0.2:
                level = max(level, IceAlarmLevel.WARNING)

        return level

    def generate_decision(self, ice_state: IceState,
                           air_temp: float,
                           current_month: int,
                           current_flow: float,
                           flow_setpoint_request: float = None) -> IceControlDecision:
        """
        生成控制决策

        Args:
            ice_state: 冰期状态
            air_temp: 气温 (°C)
            current_month: 当前月份
            current_flow: 当前流量 (m³/s)
            flow_setpoint_request: 请求的流量设定值

        Returns:
            控制决策
        """
        # 检测运行模式
        mode = self.detect_operation_mode(ice_state, air_temp, current_month)
        self.current_mode = mode

        # 获取约束
        constraints = self.get_constraints(mode, ice_state)

        # 评估报警等级
        alarm_level = self.evaluate_alarm_level(ice_state, mode, current_flow, constraints)

        # 计算流量设定值
        if flow_setpoint_request is not None:
            # 限制在约束范围内
            flow_setpoint = max(constraints.min_flow_rate,
                               min(flow_setpoint_request, constraints.max_flow_rate))
        else:
            # 使用当前流量，但限制在范围内
            flow_setpoint = max(constraints.min_flow_rate,
                               min(current_flow, constraints.max_flow_rate))

        # 计算流量变化速率
        flow_ramp_rate = constraints.max_flow_change_rate

        # 阀门约束
        valve_constraints = self._compute_valve_constraints(mode, constraints)

        # 生成警告和建议
        warnings, recommendations = self._generate_warnings_recommendations(
            mode, ice_state, alarm_level, constraints, current_flow
        )

        decision = IceControlDecision(
            mode=mode,
            alarm_level=alarm_level,
            constraints=constraints,
            flow_setpoint=flow_setpoint,
            flow_ramp_rate=flow_ramp_rate,
            valve_constraints=valve_constraints,
            warnings=warnings,
            recommendations=recommendations
        )

        self.last_decision = decision
        return decision

    def _compute_valve_constraints(self, mode: IceOperationMode,
                                    constraints: IceOperationConstraints) -> Dict[str, Tuple[float, float]]:
        """计算阀门开度约束"""
        valve_constraints = {}

        if mode in [IceOperationMode.STABLE_ICE, IceOperationMode.FREEZE_UP]:
            # 冰期限制阀门变化
            valve_constraints['inline_1'] = (20.0, 85.0)  # 不全开避免水锤
            valve_constraints['inline_2'] = (20.0, 85.0)
            valve_constraints['inline_3'] = (20.0, 85.0)
            valve_constraints['end_valves'] = (15.0, 90.0)
        elif mode == IceOperationMode.BREAKUP:
            # 开河期进一步限制
            valve_constraints['inline_1'] = (30.0, 70.0)
            valve_constraints['inline_2'] = (30.0, 70.0)
            valve_constraints['inline_3'] = (30.0, 70.0)
            valve_constraints['end_valves'] = (20.0, 80.0)
        else:
            # 正常期
            valve_constraints['inline_1'] = (0.0, 100.0)
            valve_constraints['inline_2'] = (0.0, 100.0)
            valve_constraints['inline_3'] = (0.0, 100.0)
            valve_constraints['end_valves'] = (0.0, 100.0)

        return valve_constraints

    def _generate_warnings_recommendations(self, mode: IceOperationMode,
                                            ice_state: IceState,
                                            alarm_level: IceAlarmLevel,
                                            constraints: IceOperationConstraints,
                                            current_flow: float) -> Tuple[List[str], List[str]]:
        """生成警告和建议"""
        warnings = []
        recommendations = []

        # 模式相关警告
        if mode == IceOperationMode.PRE_ICE:
            recommendations.append("进入预冰期，建议开始降低流量变化率")
            recommendations.append("检查冰期运行准备工作")

        elif mode == IceOperationMode.FREEZE_UP:
            warnings.append("封冻期运行，严禁快速调节流量")
            recommendations.append("保持流量稳定至少{}小时".format(
                self.params.STABLE_FLOW_DURATION))

        elif mode == IceOperationMode.STABLE_ICE:
            recommendations.append("冰期稳定运行，糙率增加{:.1%}".format(
                constraints.roughness_factor - 1))
            recommendations.append("过流能力降至{:.1%}".format(
                constraints.capacity_factor))

        elif mode == IceOperationMode.PRE_BREAKUP:
            warnings.append("预开河期，注意观测冰情变化")
            recommendations.append("建议逐步降低水库水位")
            recommendations.append("准备开河期应急措施")

        elif mode == IceOperationMode.BREAKUP:
            warnings.append("开河期，机械开河风险较高")
            warnings.append("严格限制流量变化，防止冰坝溃决")
            recommendations.append("加强下游冰情监测")
            recommendations.append("必要时启动应急预案")

        # 状态相关警告
        if ice_state.ice_thickness > self.params.DESIGN_ICE_THICKNESS:
            warnings.append("冰厚超过设计值: {:.2f}m > {:.2f}m".format(
                ice_state.ice_thickness, self.params.DESIGN_ICE_THICKNESS))

        if ice_state.frazil_concentration > 0.1:
            warnings.append("Frazil冰浓度较高: {:.1%}".format(
                ice_state.frazil_concentration))
            recommendations.append("可能需要增加取水口加热")

        # 流量相关警告
        if current_flow > constraints.max_flow_rate * 0.95:
            warnings.append("流量接近冰期上限")
            recommendations.append("考虑降低流量以保证安全裕度")

        # 报警等级相关
        if alarm_level.value >= IceAlarmLevel.WARNING.value:
            recommendations.append("建议增加巡检频率")

        if alarm_level.value >= IceAlarmLevel.ALARM.value:
            warnings.append("系统处于报警状态，请立即检查")
            recommendations.append("考虑启动应急响应程序")

        return warnings, recommendations

    def apply_ice_correction(self, base_flow: float, mode: IceOperationMode,
                              constraints: IceOperationConstraints) -> float:
        """
        应用冰期修正

        Args:
            base_flow: 基础流量需求 (m³/s)
            mode: 运行模式
            constraints: 运行约束

        Returns:
            修正后的流量 (m³/s)
        """
        if mode == IceOperationMode.NORMAL:
            return base_flow

        # 考虑过流能力折减
        capacity_corrected = base_flow / constraints.capacity_factor

        # 限制在约束范围
        corrected = max(constraints.min_flow_rate,
                       min(capacity_corrected, constraints.max_flow_rate))

        return corrected


class IceFlowRateLimiter:
    """
    冰期流量变化率限制器

    防止快速调节导致的水锤和冰盖破坏
    """

    def __init__(self, max_rate_percent_per_hour: float = 5.0):
        self.max_rate = max_rate_percent_per_hour
        self.last_flow: Optional[float] = None
        self.last_time: Optional[datetime] = None

    def limit(self, requested_flow: float, current_flow: float,
              dt_seconds: float) -> float:
        """
        限制流量变化

        Args:
            requested_flow: 请求流量 (m³/s)
            current_flow: 当前流量 (m³/s)
            dt_seconds: 时间步长 (s)

        Returns:
            限制后的流量 (m³/s)
        """
        if current_flow <= 0:
            return requested_flow

        # 计算最大允许变化
        dt_hours = dt_seconds / 3600.0
        max_change = current_flow * (self.max_rate / 100.0) * dt_hours

        # 计算实际变化
        requested_change = requested_flow - current_flow

        # 限制变化
        if abs(requested_change) <= max_change:
            return requested_flow
        else:
            if requested_change > 0:
                return current_flow + max_change
            else:
                return current_flow - max_change

    def set_max_rate(self, rate: float):
        """设置最大变化率"""
        self.max_rate = max(0.1, min(rate, 20.0))  # 限制在合理范围


class IceMonitor:
    """
    冰期监测器

    监测冰期关键参数并生成报告
    """

    def __init__(self):
        self.history: List[Dict] = []

    def record(self, ice_state: IceState, decision: IceControlDecision,
               actual_flow: float, air_temp: float):
        """记录状态"""
        record = {
            'timestamp': datetime.now(),
            'ice_phase': ice_state.phase.name,
            'ice_thickness': ice_state.ice_thickness,
            'ice_cover_fraction': ice_state.ice_cover_fraction,
            'composite_roughness': ice_state.composite_roughness,
            'conveyance_factor': ice_state.conveyance_factor,
            'afdd': ice_state.afdd,
            'mdd': ice_state.mdd,
            'mode': decision.mode.name,
            'alarm_level': decision.alarm_level.name,
            'flow_setpoint': decision.flow_setpoint,
            'actual_flow': actual_flow,
            'air_temp': air_temp
        }
        self.history.append(record)

    def get_daily_report(self) -> Dict:
        """获取日报"""
        if not self.history:
            return {}

        recent = self.history[-24*60:]  # 最近24小时 (假设1分钟采样)

        return {
            'period': 'daily',
            'max_ice_thickness': max(r['ice_thickness'] for r in recent),
            'min_ice_thickness': min(r['ice_thickness'] for r in recent),
            'avg_conveyance_factor': sum(r['conveyance_factor'] for r in recent) / len(recent),
            'max_alarm_level': max(r['alarm_level'] for r in recent),
            'warning_count': sum(1 for r in recent if r['alarm_level'] in ['WARNING', 'ALARM', 'EMERGENCY'])
        }


# ==========================================
# 导出
# ==========================================
__all__ = [
    # 枚举
    'IceOperationMode',
    'IceAlarmLevel',
    # 数据类
    'IceOperationConstraints',
    'IceControlDecision',
    # 控制器
    'IcePeriodController',
    'IceFlowRateLimiter',
    'IceMonitor'
]
