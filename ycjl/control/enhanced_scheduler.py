"""
引绰济辽工程L5级增强调度器 (Enhanced Scheduler) v1.0
====================================================

在原有ReservoirScheduler基础上增强:
1. [NEW] 数据完备性诊断 - 集成GapAnalyzer
2. [NEW] 场景感知调度 - 集成83种场景库
3. [NEW] 月度约束检查 - 调度图分区限制
4. [NEW] 综合诊断报告 - 系统健康评估
"""

from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from .scheduler import (
    ReservoirScheduler,
    ScheduleDecision,
    OperationZone,
    SupplyMode,
    FloodControlLevel,
    FloodForecast
)
from ..config.gap_analyzer import YinChuoGapAnalyzer
from ..config.config_database import CurveDatabase, SourceConfig
from ..core.gap_analyzer import DataGapReport, DataReadinessLevel


# 尝试导入场景引擎
try:
    from ..scenarios.scenario_database import (
        ScenarioType, ScenarioCategory, ScenarioSeverity,
        SCENARIO_DB
    )
    from ..scenarios.scenario_engine import ScenarioDetector, ScenarioState
    SCENARIO_ENGINE_AVAILABLE = True
except ImportError:
    SCENARIO_ENGINE_AVAILABLE = False


@dataclass
class EnhancedScheduleDecision(ScheduleDecision):
    """增强调度决策"""
    detected_scenarios: List[str] = field(default_factory=list)
    data_readiness: str = "未检测"
    constraint_violations: List[str] = field(default_factory=list)
    health_score: float = 100.0


@dataclass
class SystemHealthReport:
    """系统健康报告"""
    timestamp: datetime
    overall_score: float          # 总体得分 0-100
    data_readiness: DataReadinessLevel
    active_scenarios: List[str]
    warnings: List[str]
    recommendations: List[str]
    zone: OperationZone
    supply_mode: SupplyMode


class YinChuoEnhancedScheduler:
    """
    引绰济辽工程L5级增强调度器

    集成数据诊断、场景感知和综合调度功能
    """

    def __init__(self):
        """初始化增强调度器"""
        self.base_scheduler = ReservoirScheduler()
        self.gap_analyzer = YinChuoGapAnalyzer()

        # 场景检测器
        self._scenario_detector = None
        if SCENARIO_ENGINE_AVAILABLE:
            self._scenario_detector = ScenarioDetector()

        # 缓存
        self._last_gap_report: Optional[DataGapReport] = None
        self._active_scenarios: List[ScenarioState] = []

    # ---------------------------------------------------------
    # 1. 数据完备性诊断
    # ---------------------------------------------------------
    def check_data_readiness(self, refresh: bool = False) -> DataGapReport:
        """
        检查数据完备性

        Args:
            refresh: 是否刷新缓存

        Returns:
            DataGapReport: 数据缺口报告
        """
        if refresh or self._last_gap_report is None:
            self._last_gap_report = self.gap_analyzer.analyze()
        return self._last_gap_report

    def print_data_readiness(self) -> None:
        """打印数据完备性报告"""
        self.gap_analyzer.print_report()

    # ---------------------------------------------------------
    # 2. 月度调度约束检查
    # ---------------------------------------------------------
    def check_monthly_constraints(
        self,
        level: float,
        month: Optional[int] = None
    ) -> Tuple[bool, List[str]]:
        """
        检查月度调度约束

        Args:
            level: 当前水位 (m)
            month: 月份 (1-12)，默认当前月

        Returns:
            (是否合规, 约束信息列表)
        """
        if month is None:
            month = datetime.now().month

        upper, lower = self.base_scheduler.rule_chart.get_zone_limits(month)
        zone = self.base_scheduler.rule_chart.get_operation_zone(month, level)

        violations = []
        is_valid = True

        # 检查水位约束
        if level > SourceConfig.CHECK_FLOOD_LEVEL:
            is_valid = False
            violations.append(
                f"水位超校核洪水位: {level:.2f} > {SourceConfig.CHECK_FLOOD_LEVEL:.2f} m"
            )
        elif level > SourceConfig.DESIGN_FLOOD_LEVEL:
            violations.append(
                f"水位接近设计洪水位: {level:.2f} m (设计={SourceConfig.DESIGN_FLOOD_LEVEL:.2f} m)"
            )

        # 检查死水位
        if level < SourceConfig.DEAD_LEVEL:
            is_valid = False
            violations.append(
                f"水位低于死水位: {level:.2f} < {SourceConfig.DEAD_LEVEL:.2f} m"
            )

        # 检查调度分区
        if zone == OperationZone.DEAD:
            is_valid = False
            violations.append("处于死库容区，禁止供水和发电")
        elif zone == OperationZone.LOWER:
            violations.append(f"处于限制供水区: {lower:.2f} > 水位 > {SourceConfig.DEAD_LEVEL:.2f} m")
        elif zone == OperationZone.FLOOD:
            violations.append("处于防洪调度区，需启动泄洪")
        elif zone == OperationZone.UPPER:
            violations.append("处于弃水区，建议加大供水/发电")
        else:
            violations.append(f"水位合规: {lower:.2f} ≤ {level:.2f} ≤ {upper:.2f} m (月份={month})")

        return is_valid, violations

    def get_recommended_level_range(
        self,
        month: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        获取当月推荐水位范围

        Returns:
            (上限水位, 下限水位)
        """
        if month is None:
            month = datetime.now().month
        return self.base_scheduler.rule_chart.get_zone_limits(month)

    # ---------------------------------------------------------
    # 3. 场景感知调度
    # ---------------------------------------------------------
    def detect_scenarios(
        self,
        state: Dict
    ) -> List[str]:
        """
        检测当前运行场景

        Args:
            state: 系统状态字典，应包含:
                - level: 水位 (m)
                - inflow: 入库流量 (m³/s)
                - outflow: 出库流量 (m³/s)
                - temperature: 温度 (°C)
                - 其他传感器数据

        Returns:
            检测到的场景类型列表
        """
        if not SCENARIO_ENGINE_AVAILABLE:
            return ["场景引擎不可用"]

        detected = []

        # 基于水位判断
        level = state.get("level", SourceConfig.NORMAL_LEVEL)
        month = datetime.now().month
        zone = self.base_scheduler.rule_chart.get_operation_zone(month, level)

        # 正常运行场景
        flow = state.get("outflow", 0)
        if zone == OperationZone.NORMAL:
            if flow < 5:
                detected.append("NORMAL_LOW_FLOW")
            elif flow < 12:
                detected.append("NORMAL_MEDIUM_FLOW")
            else:
                detected.append("NORMAL_HIGH_FLOW")

        # 极端工况
        if zone == OperationZone.FLOOD:
            detected.append("EXTREME_FLOOD")
        elif zone == OperationZone.DEAD:
            detected.append("EXTREME_DROUGHT")

        # 冰期检测
        temp = state.get("temperature", 15)
        if temp < 0:
            detected.append("EXTREME_ICE_SEVERE")
        elif temp < 5:
            detected.append("NORMAL_SEASONAL_WINTER")

        # 设备故障检测
        if state.get("valve_stuck", False):
            detected.append("FAULT_VALVE_STUCK")
        if state.get("pump_trip", False):
            detected.append("FAULT_PUMP_TRIP")
        if state.get("sensor_drift", False):
            detected.append("FAULT_SENSOR_DRIFT")

        # 事故场景
        if state.get("pipe_burst", False):
            detected.append("ACCIDENT_PIPE_BURST")
        if state.get("power_loss", False):
            detected.append("FAULT_POWER_LOSS")

        return detected if detected else ["NORMAL_STEADY"]

    def get_scenario_response(
        self,
        scenario_id: str
    ) -> List[str]:
        """
        获取场景响应措施

        Args:
            scenario_id: 场景ID

        Returns:
            响应措施列表
        """
        responses = {
            "NORMAL_STEADY": ["保持当前运行状态", "定期巡检"],
            "NORMAL_LOW_FLOW": ["减少泵运行台数", "关闭部分分水口"],
            "NORMAL_MEDIUM_FLOW": ["优化泵组效率", "平衡供水分配"],
            "NORMAL_HIGH_FLOW": ["增加泵运行台数", "开启备用管线"],
            "EXTREME_FLOOD": ["启动溢洪道泄洪", "加大发电消落", "通知下游预警"],
            "EXTREME_DROUGHT": ["启动限水措施", "优先保障重要用户", "申请上游补水"],
            "EXTREME_ICE_SEVERE": ["启动冰期运行模式", "加强隧洞巡检", "控制流速"],
            "FAULT_VALVE_STUCK": ["切换备用阀门", "派人现场检修", "调整流量分配"],
            "FAULT_PUMP_TRIP": ["启动备用泵", "检查电气系统", "降低流量需求"],
            "FAULT_SENSOR_DRIFT": ["启用冗余传感器", "人工校核数据", "申请传感器更换"],
            "ACCIDENT_PIPE_BURST": ["紧急关闭上游阀门", "启动应急预案", "疏散影响区域"],
            "FAULT_POWER_LOSS": ["启动UPS供电", "切换柴油发电机", "执行安全停机"],
        }
        return responses.get(scenario_id, ["未知场景，请人工判断"])

    # ---------------------------------------------------------
    # 4. 综合调度决策
    # ---------------------------------------------------------
    def make_enhanced_decision(
        self,
        current_time: datetime,
        level: float,
        inflow: float,
        demand_factor: float = 1.0,
        flood_forecast: Optional[FloodForecast] = None,
        system_state: Optional[Dict] = None
    ) -> EnhancedScheduleDecision:
        """
        生成增强调度决策

        Args:
            current_time: 当前时间
            level: 水位 (m)
            inflow: 入库流量 (m³/s)
            demand_factor: 需求系数
            flood_forecast: 洪水预报
            system_state: 系统状态

        Returns:
            EnhancedScheduleDecision
        """
        # 1. 基础调度决策
        base_decision = self.base_scheduler.make_decision(
            current_time, level, inflow, demand_factor, flood_forecast
        )

        # 2. 数据完备性检查
        gap_report = self.check_data_readiness()

        # 3. 月度约束检查
        is_valid, constraint_msgs = self.check_monthly_constraints(
            level, current_time.month
        )

        # 4. 场景检测
        state = system_state or {"level": level, "inflow": inflow}
        scenarios = self.detect_scenarios(state)

        # 5. 计算健康得分
        health_score = self._calculate_health_score(
            gap_report, is_valid, scenarios, base_decision
        )

        # 构建增强决策
        enhanced = EnhancedScheduleDecision(
            timestamp=base_decision.timestamp,
            zone=base_decision.zone,
            supply_mode=base_decision.supply_mode,
            target_supply_flow=base_decision.target_supply_flow,
            target_power=base_decision.target_power,
            spillway_flow=base_decision.spillway_flow,
            supply_reduction_factor=base_decision.supply_reduction_factor,
            remarks=base_decision.remarks,
            detected_scenarios=scenarios,
            data_readiness=gap_report.readiness_level.value,
            constraint_violations=constraint_msgs if not is_valid else [],
            health_score=health_score
        )

        return enhanced

    def _calculate_health_score(
        self,
        gap_report: DataGapReport,
        constraints_valid: bool,
        scenarios: List[str],
        decision: ScheduleDecision
    ) -> float:
        """计算系统健康得分"""
        score = 100.0

        # 数据完备性扣分
        level_scores = {
            DataReadinessLevel.L5_AUTONOMOUS: 0,
            DataReadinessLevel.L4_OPTIMIZED: 5,
            DataReadinessLevel.L3_OPERATIONAL: 10,
            DataReadinessLevel.L2_PARTIAL: 20,
            DataReadinessLevel.L1_MINIMAL: 30,
            DataReadinessLevel.L0_UNAVAILABLE: 50,
        }
        score -= level_scores.get(gap_report.readiness_level, 30)

        # 约束违规扣分
        if not constraints_valid:
            score -= 20

        # 场景严重性扣分
        critical_scenarios = [
            "EXTREME_FLOOD", "EXTREME_DROUGHT", "ACCIDENT_PIPE_BURST",
            "FAULT_POWER_LOSS", "EXTREME_ICE_SEVERE"
        ]
        for s in scenarios:
            if s in critical_scenarios:
                score -= 15
            elif s.startswith("FAULT_"):
                score -= 10
            elif s.startswith("EXTREME_"):
                score -= 5

        # 调度分区扣分
        zone_penalties = {
            OperationZone.DEAD: 30,
            OperationZone.FLOOD: 15,
            OperationZone.LOWER: 10,
            OperationZone.UPPER: 5,
            OperationZone.NORMAL: 0
        }
        score -= zone_penalties.get(decision.zone, 0)

        return max(0, min(100, score))

    # ---------------------------------------------------------
    # 5. 系统健康报告
    # ---------------------------------------------------------
    def generate_health_report(
        self,
        level: float,
        inflow: float,
        system_state: Optional[Dict] = None
    ) -> SystemHealthReport:
        """生成系统健康报告"""
        current_time = datetime.now()

        # 调度决策
        decision = self.base_scheduler.make_decision(
            current_time, level, inflow
        )

        # 数据完备性
        gap_report = self.check_data_readiness()

        # 场景检测
        state = system_state or {"level": level, "inflow": inflow}
        scenarios = self.detect_scenarios(state)

        # 约束检查
        is_valid, violations = self.check_monthly_constraints(level)

        # 生成警告和建议
        warnings = []
        recommendations = []

        # 基于场景的警告
        for s in scenarios:
            if s.startswith("FAULT_"):
                warnings.append(f"检测到故障场景: {s}")
            elif s.startswith("EXTREME_"):
                warnings.append(f"检测到极端工况: {s}")
            elif s.startswith("ACCIDENT_"):
                warnings.append(f"检测到事故场景: {s}")

        # 基于约束的警告
        if not is_valid:
            warnings.extend([v for v in violations if "超" in v or "低于" in v])

        # 生成建议
        if gap_report.critical_missing > 0:
            recommendations.append(
                f"建议补全 {gap_report.critical_missing} 项关键数据"
            )

        for s in scenarios:
            responses = self.get_scenario_response(s)
            recommendations.extend(responses[:2])  # 取前2条建议

        # 计算得分
        score = self._calculate_health_score(
            gap_report, is_valid, scenarios, decision
        )

        return SystemHealthReport(
            timestamp=current_time,
            overall_score=score,
            data_readiness=gap_report.readiness_level,
            active_scenarios=scenarios,
            warnings=warnings,
            recommendations=list(set(recommendations))[:5],  # 去重，最多5条
            zone=decision.zone,
            supply_mode=decision.supply_mode
        )

    def print_health_report(
        self,
        level: float,
        inflow: float,
        system_state: Optional[Dict] = None
    ) -> None:
        """打印系统健康报告"""
        report = self.generate_health_report(level, inflow, system_state)

        print(f"\n{'='*70}")
        print("引绰济辽工程 L5级系统健康报告")
        print(f"{'='*70}")
        print(f"报告时间: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"健康得分: {report.overall_score:.1f}/100")
        print(f"数据就绪: {report.data_readiness.value}")
        print(f"调度分区: {report.zone.name}")
        print(f"供水模式: {report.supply_mode.name}")

        print(f"\n活跃场景 ({len(report.active_scenarios)} 个):")
        for s in report.active_scenarios:
            print(f"  - {s}")

        if report.warnings:
            print(f"\n告警 ({len(report.warnings)} 项):")
            for w in report.warnings:
                print(f"  ⚠ {w}")

        if report.recommendations:
            print(f"\n建议:")
            for r in report.recommendations:
                print(f"  → {r}")

        print(f"{'='*70}")


# ==========================================
# 模块级实例
# ==========================================
EnhancedScheduler = YinChuoEnhancedScheduler()


# ==========================================
# 便捷函数
# ==========================================
def check_readiness() -> DataGapReport:
    """检查数据完备性"""
    return EnhancedScheduler.check_data_readiness()


def make_decision(level: float, inflow: float) -> EnhancedScheduleDecision:
    """生成增强调度决策"""
    return EnhancedScheduler.make_enhanced_decision(
        datetime.now(), level, inflow
    )


def health_report(level: float, inflow: float) -> None:
    """打印健康报告"""
    EnhancedScheduler.print_health_report(level, inflow)


# ==========================================
# 导出
# ==========================================
__all__ = [
    'EnhancedScheduleDecision',
    'SystemHealthReport',
    'YinChuoEnhancedScheduler',
    'EnhancedScheduler',
    'check_readiness',
    'make_decision',
    'health_report'
]
