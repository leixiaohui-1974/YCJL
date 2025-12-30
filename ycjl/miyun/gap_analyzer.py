"""
密云水库调蓄工程数据完备性诊断器 (Gap Analyzer) v1.0
====================================================

功能：
1. 分析系统参数数据库的完备性
2. 识别L5级自主运行所需的缺失数据
3. 评估当前系统支持的仿真能力级别
4. 生成数据补全优先级建议
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum, auto
import datetime

from .config_database import RouteType, STATION_DB, MiyunParams


# ==========================================
# 1. 数据完备性等级
# ==========================================
class DataReadinessLevel(Enum):
    """数据完备性等级"""
    L1_BASIC = "L1-基础仿真 (稳态计算)"
    L2_STANDARD = "L2-标准仿真 (动态响应)"
    L3_ADVANCED = "L3-高级仿真 (过渡过程)"
    L4_TRANSIENT = "L4-瞬态仿真 (水锤分析)"
    L5_FULL = "L5-完全自主 (全工况运行)"


class DataPriority(Enum):
    """数据补全优先级"""
    CRITICAL = "CRITICAL"       # 关键 - 影响安全
    HIGH = "HIGH"               # 高 - 影响精度
    MEDIUM = "MEDIUM"           # 中 - 影响优化
    LOW = "LOW"                 # 低 - 锦上添花


# ==========================================
# 2. 缺失数据项
# ==========================================
@dataclass
class MissingDataItem:
    """缺失数据项"""
    station_key: str
    station_name: str
    parameter_name: str
    parameter_description: str
    priority: DataPriority
    impact: str
    required_for_level: DataReadinessLevel


@dataclass
class DataGapReport:
    """数据缺口分析报告"""
    timestamp: str
    total_stations: int
    missing_items: List[MissingDataItem]
    current_level: DataReadinessLevel
    level_breakdown: Dict[str, int]  # 各级别缺失数量
    recommendations: List[str]

    @property
    def total_missing(self) -> int:
        return len(self.missing_items)

    @property
    def critical_missing(self) -> int:
        return len([m for m in self.missing_items
                   if m.priority == DataPriority.CRITICAL])


# ==========================================
# 3. 数据完备性诊断器
# ==========================================
class MiyunDataGapAnalyzer:
    """
    密云水库调蓄工程数据完备性诊断器

    分析系统参数数据库，识别缺失数据并评估仿真能力
    """

    def __init__(self):
        """初始化诊断器"""
        self.station_db = STATION_DB
        self.config = MiyunParams

    def analyze_readiness(self) -> DataGapReport:
        """
        分析数据完备性

        Returns:
            DataGapReport: 完整的分析报告
        """
        missing_items = []

        for station_key, data in self.station_db.items():
            station_name = data["name"]
            station_type = data["type"]

            # 检查泵参数
            pump_missing = self._check_pump_parameters(
                station_key, station_name, data.get("pump", {})
            )
            missing_items.extend(pump_missing)

            # 检查闸门参数
            gate_missing = self._check_gate_parameters(
                station_key, station_name, data.get("gate", {})
            )
            missing_items.extend(gate_missing)

            # 检查管道参数 (仅有压管段)
            if station_type == RouteType.PIPELINE:
                pipe_missing = self._check_pipe_parameters(
                    station_key, station_name, data.get("pipe_geo", {})
                )
                missing_items.extend(pipe_missing)

            # 检查渠道参数 (仅明渠段)
            if station_type == RouteType.CHANNEL:
                channel_missing = self._check_channel_parameters(
                    station_key, station_name, data.get("channel_geo", {})
                )
                missing_items.extend(channel_missing)

        # 评估当前支持级别
        current_level, level_breakdown = self._evaluate_readiness_level(missing_items)

        # 生成建议
        recommendations = self._generate_recommendations(missing_items, current_level)

        return DataGapReport(
            timestamp=datetime.datetime.now().isoformat(),
            total_stations=len(self.station_db),
            missing_items=missing_items,
            current_level=current_level,
            level_breakdown=level_breakdown,
            recommendations=recommendations
        )

    def _check_pump_parameters(
        self,
        station_key: str,
        station_name: str,
        pump_cfg: dict
    ) -> List[MissingDataItem]:
        """检查泵参数完备性"""
        missing = []

        # 转动惯量 (GD²) - 用于停泵水锤计算
        if pump_cfg.get("Inertia_GD2") is None:
            missing.append(MissingDataItem(
                station_key=station_key,
                station_name=station_name,
                parameter_name="Inertia_GD2",
                parameter_description="机组转动惯量 (kg·m²)",
                priority=DataPriority.CRITICAL,
                impact="无法计算停泵水锤/倒流时间",
                required_for_level=DataReadinessLevel.L4_TRANSIENT
            ))

        # Hill Chart - 全工况效率图谱
        if pump_cfg.get("Hill_Chart") is None:
            missing.append(MissingDataItem(
                station_key=station_key,
                station_name=station_name,
                parameter_name="Hill_Chart",
                parameter_description="泵全工况效率图谱",
                priority=DataPriority.MEDIUM,
                impact="无法进行极致能效寻优",
                required_for_level=DataReadinessLevel.L5_FULL
            ))

        # Suter曲线 - 全特性曲线 (反转特性)
        if pump_cfg.get("Suter_Curve") is None and station_key == "Xiwengzhuang":
            missing.append(MissingDataItem(
                station_key=station_key,
                station_name=station_name,
                parameter_name="Suter_Curve",
                parameter_description="水泵全特性曲线 (Suter变换)",
                priority=DataPriority.HIGH,
                impact="无法分析泵反转/飞逸工况",
                required_for_level=DataReadinessLevel.L4_TRANSIENT
            ))

        return missing

    def _check_gate_parameters(
        self,
        station_key: str,
        station_name: str,
        gate_cfg: dict
    ) -> List[MissingDataItem]:
        """检查闸门参数完备性"""
        missing = []

        if gate_cfg and gate_cfg.get("Cd_Curve") is None:
            missing.append(MissingDataItem(
                station_key=station_key,
                station_name=station_name,
                parameter_name="Cd_Curve",
                parameter_description="节制闸流量系数曲线 Q=f(e,ΔH)",
                priority=DataPriority.MEDIUM,
                impact="放水控制精度低",
                required_for_level=DataReadinessLevel.L3_ADVANCED
            ))

        return missing

    def _check_pipe_parameters(
        self,
        station_key: str,
        station_name: str,
        pipe_cfg: dict
    ) -> List[MissingDataItem]:
        """检查管道参数完备性"""
        missing = []

        # 波速 - 用于水锤计算
        if pipe_cfg.get("Wave_Speed_a") is None:
            # 雁栖和溪翁庄为高扬程段，更关键
            priority = DataPriority.CRITICAL if station_key in ["Yanqi", "Xiwengzhuang"] else DataPriority.HIGH

            missing.append(MissingDataItem(
                station_key=station_key,
                station_name=station_name,
                parameter_name="Wave_Speed_a",
                parameter_description="管道压力波速 (m/s)",
                priority=priority,
                impact="水锤压力计算失真",
                required_for_level=DataReadinessLevel.L4_TRANSIENT
            ))

        # 阀门关闭规律
        if pipe_cfg.get("Valve_Closure_Curve") is None:
            missing.append(MissingDataItem(
                station_key=station_key,
                station_name=station_name,
                parameter_name="Valve_Closure_Curve",
                parameter_description="出口阀关闭时间规律",
                priority=DataPriority.CRITICAL,
                impact="无法设计防爆管策略",
                required_for_level=DataReadinessLevel.L4_TRANSIENT
            ))

        return missing

    def _check_channel_parameters(
        self,
        station_key: str,
        station_name: str,
        channel_cfg: dict
    ) -> List[MissingDataItem]:
        """检查渠道参数完备性"""
        missing = []

        # 糙率季节性变化
        if channel_cfg.get("Roughness_Seasonality") is None:
            missing.append(MissingDataItem(
                station_key=station_key,
                station_name=station_name,
                parameter_name="Roughness_Seasonality",
                parameter_description="渠道糙率季节变化系数",
                priority=DataPriority.LOW,
                impact="季节性水力计算精度下降",
                required_for_level=DataReadinessLevel.L5_FULL
            ))

        return missing

    def _evaluate_readiness_level(
        self,
        missing_items: List[MissingDataItem]
    ) -> Tuple[DataReadinessLevel, Dict[str, int]]:
        """评估当前系统支持的仿真级别"""
        level_counts = {
            DataReadinessLevel.L4_TRANSIENT.value: 0,
            DataReadinessLevel.L5_FULL.value: 0,
            DataReadinessLevel.L3_ADVANCED.value: 0
        }

        for item in missing_items:
            if item.required_for_level.value in level_counts:
                level_counts[item.required_for_level.value] += 1

        # 判断当前支持级别
        critical_count = len([m for m in missing_items
                             if m.priority == DataPriority.CRITICAL])

        if critical_count == 0:
            current_level = DataReadinessLevel.L5_FULL
        elif level_counts[DataReadinessLevel.L4_TRANSIENT.value] > 0:
            current_level = DataReadinessLevel.L3_ADVANCED
        elif level_counts[DataReadinessLevel.L5_FULL.value] > 0:
            current_level = DataReadinessLevel.L4_TRANSIENT
        else:
            current_level = DataReadinessLevel.L2_STANDARD

        return current_level, level_counts

    def _generate_recommendations(
        self,
        missing_items: List[MissingDataItem],
        current_level: DataReadinessLevel
    ) -> List[str]:
        """生成数据补全建议"""
        recommendations = []

        # 按优先级统计
        critical = [m for m in missing_items if m.priority == DataPriority.CRITICAL]
        high = [m for m in missing_items if m.priority == DataPriority.HIGH]

        if critical:
            recommendations.append(
                f"[紧急] 需优先补全 {len(critical)} 项关键参数，以支持事故瞬态分析"
            )
            # 列出关键参数
            for item in critical[:3]:  # 最多列出3项
                recommendations.append(
                    f"  - {item.station_name}: {item.parameter_description}"
                )

        if high:
            recommendations.append(
                f"[重要] 建议补全 {len(high)} 项高优先级参数，提升仿真精度"
            )

        # 根据当前级别给出建议
        if current_level == DataReadinessLevel.L3_ADVANCED:
            recommendations.append(
                "当前系统可支持[稳态运行仿真]和[过渡过程分析]，"
                "但暂不支持[事故瞬态分析]"
            )
        elif current_level == DataReadinessLevel.L4_TRANSIENT:
            recommendations.append(
                "当前系统可支持[事故瞬态分析]，但暂不支持[完全自主运行]"
            )

        return recommendations

    def print_report(self) -> None:
        """打印诊断报告到控制台"""
        report = self.analyze_readiness()

        print(f"\n{'='*80}")
        print("L5级数字孪生·数据完备性诊断报告")
        print("密云水库调蓄工程")
        print(f"{'='*80}")
        print(f"诊断时间: {report.timestamp}")
        print(f"泵站总数: {report.total_stations}")
        print(f"缺失数据项: {report.total_missing} (其中关键项: {report.critical_missing})")
        print(f"当前支持级别: {report.current_level.value}")
        print()

        # 打印详细缺失列表
        print(f"{'枢纽节点':<12} | {'类型':<6} | {'缺失参数':<25} | {'优先级':<10} | {'影响后果'}")
        print("-" * 100)

        # 按优先级排序
        sorted_items = sorted(report.missing_items,
                             key=lambda x: list(DataPriority).index(x.priority))

        for item in sorted_items:
            station_type = STATION_DB[item.station_key]["type"].name[:4]
            print(f"{item.station_name:<12} | {station_type:<6} | "
                  f"{item.parameter_description[:25]:<25} | {item.priority.value:<10} | "
                  f"{item.impact[:30]}")

        print("-" * 100)

        # 打印建议
        print("\n建议:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")

        print(f"\n{'='*80}")

    def get_station_readiness(self, station_key: str) -> Dict:
        """获取单个泵站的数据完备性"""
        report = self.analyze_readiness()
        station_items = [m for m in report.missing_items
                        if m.station_key == station_key]

        return {
            "station_key": station_key,
            "station_name": STATION_DB[station_key]["name"] if station_key in STATION_DB else "Unknown",
            "missing_count": len(station_items),
            "critical_count": len([m for m in station_items
                                  if m.priority == DataPriority.CRITICAL]),
            "missing_items": station_items,
            "is_ready_for_l4": all(m.required_for_level != DataReadinessLevel.L4_TRANSIENT
                                   for m in station_items if m.priority == DataPriority.CRITICAL)
        }


# ==========================================
# 模块级实例
# ==========================================
GapAnalyzer = MiyunDataGapAnalyzer()


# ==========================================
# 便捷函数
# ==========================================
def analyze_data_readiness() -> DataGapReport:
    """分析数据完备性"""
    return GapAnalyzer.analyze_readiness()


def print_data_report() -> None:
    """打印数据完备性报告"""
    GapAnalyzer.print_report()


# ==========================================
# 导出
# ==========================================
__all__ = [
    'DataReadinessLevel',
    'DataPriority',
    'MissingDataItem',
    'DataGapReport',
    'MiyunDataGapAnalyzer',
    'GapAnalyzer',
    'analyze_data_readiness',
    'print_data_report'
]
