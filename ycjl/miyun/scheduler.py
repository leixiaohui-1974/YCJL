"""
密云水库调蓄工程L5级自主调度器 (Autonomous Scheduler) v1.0
=========================================================

功能：
1. 数字孪生体检 - 全参数系统诊断
2. 运行优化 - 泵站运行方案优化
3. 调度决策 - 基于当前工况的调度建议
4. 场景仿真 - 不同流量场景下的系统响应
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum, auto
import datetime

from .config_database import (
    RouteType, STATION_DB, MiyunParams,
    MiyunCurveDatabase, MiyunReservoirCfg
)
from .physics_engine import (
    MiyunSimulationEngine, SimEngine,
    SystemStatus, HeadCalculationResult, SystemDiagnosisResult
)
from .gap_analyzer import GapAnalyzer, DataGapReport


# ==========================================
# 1. 调度模式枚举
# ==========================================
class ScheduleMode(Enum):
    """调度模式"""
    NORMAL = "正常供水"
    PEAK = "高峰供水"
    LOW_FLOW = "低流量运行"
    MAINTENANCE = "检修模式"
    EMERGENCY = "应急模式"
    STANDBY = "待机模式"


class OperationZone(Enum):
    """运行分区"""
    OPTIMAL = "最优区"           # 效率最高
    ACCEPTABLE = "可接受区"      # 正常运行
    SUBOPTIMAL = "次优区"        # 效率偏低
    WARNING = "警戒区"           # 接近限制
    FORBIDDEN = "禁止区"         # 超出限制


# ==========================================
# 2. 调度决策数据类
# ==========================================
@dataclass
class PumpScheduleItem:
    """单站泵组调度"""
    station_key: str
    station_name: str
    recommended_count: int      # 建议运行台数
    current_count: int          # 当前运行台数
    action: str                 # 调度动作 (启动/停止/保持)
    target_flow: float          # 目标流量
    estimated_power: float      # 预估功耗


@dataclass
class ScheduleDecision:
    """调度决策"""
    timestamp: str
    mode: ScheduleMode
    target_flow: float          # 目标总流量 (m³/s)
    estimated_power: float      # 预估总功耗 (kW)
    pump_schedules: List[PumpScheduleItem]
    system_status: SystemStatus
    warnings: List[str]
    recommendations: List[str]


# ==========================================
# 3. 数字孪生调度器
# ==========================================
class MiyunDigitalTwinScheduler:
    """
    密云水库调蓄工程L5级自主调度器

    集成物理仿真、数据诊断和调度优化功能
    """

    def __init__(self):
        """初始化调度器"""
        self.engine = SimEngine
        self.gap_analyzer = GapAnalyzer
        self.curves = MiyunCurveDatabase
        self.config = MiyunParams

    # ---------------------------------------------------------
    # 3.1 系统诊断
    # ---------------------------------------------------------
    def run_diagnosis(
        self,
        flow_scenario: float,
        print_output: bool = True
    ) -> SystemDiagnosisResult:
        """
        运行全参数数字孪生体检

        Args:
            flow_scenario: 流量场景 (m³/s)
            print_output: 是否打印输出

        Returns:
            SystemDiagnosisResult: 诊断结果
        """
        result = self.engine.run_system_diagnosis(flow_scenario)

        if print_output:
            self._print_diagnosis(result)

        return result

    def _print_diagnosis(self, result: SystemDiagnosisResult) -> None:
        """打印诊断结果"""
        print(f"\n{'='*80}")
        print(f"全参数数字孪生体检 | 流量场景: {result.flow_rate} m³/s")
        print(f"{'='*80}")
        print(f"{'枢纽名称':<12} | {'实测扬程(m)':<12} | {'功率(kW)':<10} | {'效率':<8} | {'健康诊断'}")
        print("-" * 80)

        for sr in result.station_results:
            diag_msg = "运行平稳"
            if sr.warnings:
                diag_msg = sr.warnings[0][:30] if len(sr.warnings[0]) > 30 else sr.warnings[0]

            status_icon = ""
            if sr.status == SystemStatus.CRITICAL:
                status_icon = " "
            elif sr.status == SystemStatus.WARNING:
                status_icon = " "
            else:
                status_icon = " "

            print(f"{sr.station_name:<12} | {sr.total_head:<12.2f} | "
                  f"{sr.power_required:<10.0f} | {sr.efficiency:<8.2%} | "
                  f"{status_icon}{diag_msg}")

        print("-" * 80)
        print(f"系统总瞬时功耗: {result.total_power_mw:.2f} MW")
        print(f"系统总扬程: {result.total_head:.2f} m")
        print(f"整体状态: {result.overall_status.value}")

        if result.system_warnings:
            print(f"\n系统告警 ({len(result.system_warnings)} 项):")
            for warn in result.system_warnings[:5]:  # 最多显示5条
                print(f"  - {warn}")

    # ---------------------------------------------------------
    # 3.2 流量场景分析
    # ---------------------------------------------------------
    def analyze_flow_scenarios(
        self,
        flow_range: Optional[List[float]] = None
    ) -> Dict[float, SystemDiagnosisResult]:
        """
        分析多个流量场景

        Args:
            flow_range: 流量列表，默认 [4, 6, 10, 15, 20]

        Returns:
            {流量: 诊断结果}
        """
        if flow_range is None:
            flow_range = [4.0, 6.0, 10.0, 15.0, 20.0]

        results = {}
        for flow in flow_range:
            results[flow] = self.engine.run_system_diagnosis(flow)

        return results

    def print_flow_analysis(
        self,
        flow_range: Optional[List[float]] = None
    ) -> None:
        """打印流量场景分析"""
        results = self.analyze_flow_scenarios(flow_range)

        print(f"\n{'='*80}")
        print("多流量场景系统响应分析")
        print(f"{'='*80}")
        print(f"{'流量(m³/s)':<12} | {'总功耗(MW)':<12} | {'总扬程(m)':<12} | "
              f"{'告警数':<8} | {'系统状态'}")
        print("-" * 70)

        for flow, result in sorted(results.items()):
            warn_count = len(result.system_warnings)
            status = result.overall_status.value

            print(f"{flow:<12.1f} | {result.total_power_mw:<12.2f} | "
                  f"{result.total_head:<12.2f} | {warn_count:<8} | {status}")

        print("-" * 70)

    # ---------------------------------------------------------
    # 3.3 调度决策生成
    # ---------------------------------------------------------
    def generate_schedule(
        self,
        target_flow: float,
        current_pump_status: Optional[Dict[str, int]] = None
    ) -> ScheduleDecision:
        """
        生成调度决策

        Args:
            target_flow: 目标流量 (m³/s)
            current_pump_status: 当前各站运行台数 {station_key: count}

        Returns:
            ScheduleDecision: 调度决策
        """
        if current_pump_status is None:
            current_pump_status = {k: 0 for k in STATION_DB.keys()}

        # 运行诊断
        diagnosis = self.engine.run_system_diagnosis(target_flow)

        # 获取优化建议
        optimization = self.engine.optimize_pump_operation(target_flow)

        # 生成调度项
        pump_schedules = []
        total_estimated_power = 0.0
        warnings = list(diagnosis.system_warnings)
        recommendations = []

        for station_key, opt in optimization.items():
            current = current_pump_status.get(station_key, 0)
            recommended = opt["optimal_pump_count"]

            if recommended > current:
                action = f"启动 {recommended - current} 台"
            elif recommended < current:
                action = f"停止 {current - recommended} 台"
            else:
                action = "保持"

            # 从诊断结果获取功耗
            station_result = next(
                (r for r in diagnosis.station_results if r.station_key == station_key),
                None
            )
            power = station_result.power_required if station_result else 0.0
            total_estimated_power += power

            pump_schedules.append(PumpScheduleItem(
                station_key=station_key,
                station_name=opt["station_name"],
                recommended_count=recommended,
                current_count=current,
                action=action,
                target_flow=opt["flow_per_pump"] * recommended,
                estimated_power=power
            ))

            # 生成建议
            if opt["status"] == "suboptimal":
                recommendations.append(
                    f"{opt['station_name']}: 负荷率{opt['load_ratio']:.1%}，"
                    f"建议调整运行台数"
                )

        # 确定调度模式
        mode = self._determine_mode(target_flow, diagnosis.overall_status)

        return ScheduleDecision(
            timestamp=datetime.datetime.now().isoformat(),
            mode=mode,
            target_flow=target_flow,
            estimated_power=total_estimated_power,
            pump_schedules=pump_schedules,
            system_status=diagnosis.overall_status,
            warnings=warnings,
            recommendations=recommendations
        )

    def _determine_mode(
        self,
        target_flow: float,
        status: SystemStatus
    ) -> ScheduleMode:
        """确定调度模式"""
        if status == SystemStatus.CRITICAL:
            return ScheduleMode.EMERGENCY
        elif status == SystemStatus.SHUTDOWN:
            return ScheduleMode.STANDBY
        elif target_flow >= 18.0:
            return ScheduleMode.PEAK
        elif target_flow <= 6.0:
            return ScheduleMode.LOW_FLOW
        else:
            return ScheduleMode.NORMAL

    def print_schedule(self, decision: ScheduleDecision) -> None:
        """打印调度决策"""
        print(f"\n{'='*80}")
        print(f"调度决策 | 目标流量: {decision.target_flow} m³/s | 模式: {decision.mode.value}")
        print(f"{'='*80}")
        print(f"{'泵站':<12} | {'当前':<6} | {'建议':<6} | {'动作':<12} | "
              f"{'目标流量':<10} | {'功耗(kW)':<10}")
        print("-" * 75)

        for ps in decision.pump_schedules:
            print(f"{ps.station_name:<12} | {ps.current_count:<6} | "
                  f"{ps.recommended_count:<6} | {ps.action:<12} | "
                  f"{ps.target_flow:<10.2f} | {ps.estimated_power:<10.0f}")

        print("-" * 75)
        print(f"预估总功耗: {decision.estimated_power/1000:.2f} MW")
        print(f"系统状态: {decision.system_status.value}")

        if decision.warnings:
            print(f"\n告警 ({len(decision.warnings)} 项):")
            for warn in decision.warnings[:5]:
                print(f"  - {warn}")

        if decision.recommendations:
            print(f"\n建议:")
            for rec in decision.recommendations:
                print(f"  - {rec}")

    # ---------------------------------------------------------
    # 3.4 数据完备性检查
    # ---------------------------------------------------------
    def check_data_readiness(self, print_output: bool = True) -> DataGapReport:
        """
        检查数据完备性

        Args:
            print_output: 是否打印输出

        Returns:
            DataGapReport: 数据缺口报告
        """
        report = self.gap_analyzer.analyze_readiness()

        if print_output:
            self.gap_analyzer.print_report()

        return report

    # ---------------------------------------------------------
    # 3.5 综合仿真
    # ---------------------------------------------------------
    def run_comprehensive_simulation(
        self,
        target_flow: float = 10.0
    ) -> Dict:
        """
        运行综合仿真

        Args:
            target_flow: 目标流量

        Returns:
            综合仿真结果
        """
        print(f"\n{'='*80}")
        print("密云水库调蓄工程 L5级数字孪生系统")
        print(f"{'='*80}")

        # 1. 数据完备性检查
        print("\n[1/3] 数据完备性检查...")
        gap_report = self.check_data_readiness(print_output=False)
        print(f"  - 当前支持级别: {gap_report.current_level.value}")
        print(f"  - 缺失数据项: {gap_report.total_missing} (关键: {gap_report.critical_missing})")

        # 2. 系统诊断
        print(f"\n[2/3] 系统诊断 (流量: {target_flow} m³/s)...")
        diagnosis = self.run_diagnosis(target_flow, print_output=False)
        print(f"  - 系统状态: {diagnosis.overall_status.value}")
        print(f"  - 总功耗: {diagnosis.total_power_mw:.2f} MW")
        print(f"  - 告警数: {len(diagnosis.system_warnings)}")

        # 3. 调度决策
        print(f"\n[3/3] 生成调度决策...")
        schedule = self.generate_schedule(target_flow)
        print(f"  - 调度模式: {schedule.mode.value}")
        print(f"  - 预估功耗: {schedule.estimated_power/1000:.2f} MW")

        # 详细报告
        print(f"\n{'='*80}")
        print("详细诊断报告")
        self._print_diagnosis(diagnosis)

        print(f"\n{'='*80}")
        print("调度建议")
        self.print_schedule(schedule)

        return {
            "gap_report": gap_report,
            "diagnosis": diagnosis,
            "schedule": schedule
        }


# ==========================================
# 模块级实例
# ==========================================
Scheduler = MiyunDigitalTwinScheduler()


# ==========================================
# 便捷函数
# ==========================================
def run_diagnosis(flow: float = 10.0) -> SystemDiagnosisResult:
    """运行系统诊断"""
    return Scheduler.run_diagnosis(flow)


def generate_schedule(target_flow: float = 10.0) -> ScheduleDecision:
    """生成调度决策"""
    return Scheduler.generate_schedule(target_flow)


def run_simulation(flow: float = 10.0) -> Dict:
    """运行综合仿真"""
    return Scheduler.run_comprehensive_simulation(flow)


# ==========================================
# 导出
# ==========================================
__all__ = [
    'ScheduleMode',
    'OperationZone',
    'PumpScheduleItem',
    'ScheduleDecision',
    'MiyunDigitalTwinScheduler',
    'Scheduler',
    'run_diagnosis',
    'generate_schedule',
    'run_simulation'
]
