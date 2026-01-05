"""
设计运行域分析器 (Operational Design Domain Analyzer)
=====================================================

提供水利智能系统的设计运行域（ODD）分析框架，支持：
- 多维度ODD边界定义与监测
- 可观性/可控性评估
- 运行域可靠性计算
- 自主等级判定与优雅降级

ODD定义了智能调度算法或具身智能体在什么情况下是保证管用的。
它是从"经验描述"向"数字契约"转化的新标准。

四个关键维度：
1. 环境与边界 (Environment & Geography)
2. 动力学限制 (Dynamic Constraints)
3. 设备能力 (Infrastructure Capability)
4. 数字化支撑 (Digital Backbone)
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Callable
from enum import Enum, auto
from datetime import datetime


# ==========================================
# 枚举定义
# ==========================================

class ODDDimension(Enum):
    """ODD维度分类"""
    ENVIRONMENT = "环境与边界"           # 工程适用性、地理围栏、气象窗口
    DYNAMICS = "动力学限制"              # 流态边界、水情边界、波动变率
    INFRASTRUCTURE = "设备能力"          # 执行器特性、传感器精度、冗余度
    DIGITAL = "数字化支撑"               # 通信指标、算力保障


class ODDStatus(Enum):
    """ODD状态"""
    NOMINAL = "正常"                    # 在ODD范围内
    MARGINAL = "边界"                   # 接近ODD边界（80%-100%）
    EXCEEDED = "超出"                   # 超出ODD边界
    UNKNOWN = "未知"                    # 无法确定（数据缺失）
    DEGRADED = "降级"                   # 部分功能受限


class AutonomyLevel(Enum):
    """自主运行等级"""
    L0_MANUAL = (0, "人工操作")         # 完全人工
    L1_ASSISTED = (1, "辅助决策")       # 提供建议
    L2_PARTIAL = (2, "部分自动")        # 特定功能自动
    L3_CONDITIONAL = (3, "条件自动")    # 有条件自主
    L4_HIGH = (4, "高度自主")           # 大部分场景自主
    L5_FULL = (5, "完全自主")           # 全自主运行

    def __init__(self, level: int, description: str):
        self.level = level
        self.description = description


class ConstraintType(Enum):
    """约束类型"""
    HARD = "硬约束"                     # 不可违反
    SOFT = "软约束"                     # 可临时违反
    ADVISORY = "建议约束"               # 仅作参考


class ViolationSeverity(Enum):
    """违规严重程度"""
    CRITICAL = (1, "严重", 0.0)         # 立即触发MRM
    MAJOR = (2, "重大", 0.3)            # 需降级运行
    MINOR = (3, "轻微", 0.7)            # 可继续但需关注
    WARNING = (4, "警告", 0.9)          # 仅记录

    def __init__(self, level: int, name: str, min_score: float):
        self._level = level
        self._name = name
        self.min_score = min_score


# ==========================================
# 数据类定义
# ==========================================

@dataclass
class ODDBoundary:
    """
    ODD边界参数定义

    定义单个ODD参数的边界条件
    """
    name: str                                   # 参数名称
    dimension: ODDDimension                     # 所属维度
    description: str                            # 参数描述
    unit: str                                   # 单位

    # 边界值
    min_value: Optional[float] = None           # 最小允许值
    max_value: Optional[float] = None           # 最大允许值
    nominal_value: Optional[float] = None       # 标称值

    # 约束特性
    constraint_type: ConstraintType = ConstraintType.HARD
    violation_severity: ViolationSeverity = ViolationSeverity.MAJOR

    # 边界裕度
    warning_margin: float = 0.1                 # 警告裕度（10%）
    critical_margin: float = 0.0                # 临界裕度

    # 权重
    weight: float = 1.0                         # 评分权重

    def check_value(self, value: float) -> Tuple[ODDStatus, float]:
        """
        检查值是否在ODD范围内

        Args:
            value: 当前值

        Returns:
            (状态, 得分0-1)
        """
        if value is None:
            return ODDStatus.UNKNOWN, 0.5

        # 计算归一化偏离度
        if self.min_value is not None and self.max_value is not None:
            range_val = self.max_value - self.min_value
            if range_val <= 0:
                return ODDStatus.UNKNOWN, 0.5

            # 计算偏离
            if value < self.min_value:
                deviation = (self.min_value - value) / range_val
                score = max(0, 1 - deviation)
                status = ODDStatus.EXCEEDED if deviation > self.critical_margin else ODDStatus.MARGINAL
            elif value > self.max_value:
                deviation = (value - self.max_value) / range_val
                score = max(0, 1 - deviation)
                status = ODDStatus.EXCEEDED if deviation > self.critical_margin else ODDStatus.MARGINAL
            else:
                # 在范围内，计算距离边界的裕度
                dist_to_min = value - self.min_value
                dist_to_max = self.max_value - value
                margin_dist = min(dist_to_min, dist_to_max) / range_val

                if margin_dist < self.warning_margin:
                    status = ODDStatus.MARGINAL
                    score = 0.8 + 0.2 * (margin_dist / self.warning_margin)
                else:
                    status = ODDStatus.NOMINAL
                    score = 1.0

        elif self.min_value is not None:
            if value < self.min_value:
                deviation = (self.min_value - value) / abs(self.min_value) if self.min_value != 0 else 1
                score = max(0, 1 - deviation)
                status = ODDStatus.EXCEEDED
            else:
                score = 1.0
                status = ODDStatus.NOMINAL

        elif self.max_value is not None:
            if value > self.max_value:
                deviation = (value - self.max_value) / abs(self.max_value) if self.max_value != 0 else 1
                score = max(0, 1 - deviation)
                status = ODDStatus.EXCEEDED
            else:
                score = 1.0
                status = ODDStatus.NOMINAL
        else:
            return ODDStatus.UNKNOWN, 0.5

        return status, score

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "name": self.name,
            "dimension": self.dimension.name,
            "description": self.description,
            "unit": self.unit,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "constraint_type": self.constraint_type.name,
            "weight": self.weight
        }


@dataclass
class ODDViolation:
    """ODD违规记录"""
    timestamp: datetime
    boundary: ODDBoundary
    actual_value: float
    status: ODDStatus
    score: float
    message: str

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "parameter": self.boundary.name,
            "dimension": self.boundary.dimension.name,
            "actual_value": self.actual_value,
            "status": self.status.name,
            "score": self.score,
            "message": self.message
        }


@dataclass
class DimensionScore:
    """维度评分"""
    dimension: ODDDimension
    score: float                                # 0-1
    status: ODDStatus
    parameter_count: int
    violated_count: int
    violations: List[ODDViolation] = field(default_factory=list)
    details: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "dimension": self.dimension.name,
            "dimension_name": self.dimension.value,
            "score": self.score,
            "status": self.status.name,
            "parameter_count": self.parameter_count,
            "violated_count": self.violated_count,
            "details": self.details
        }


@dataclass
class ObservabilityMetrics:
    """
    可观性指标

    评估系统状态是否可通过测量输出完全观测
    """
    sensor_coverage: float = 1.0                # 传感器覆盖率 (0-1)
    sensor_health: float = 1.0                  # 传感器健康度 (0-1)
    data_quality: float = 1.0                   # 数据质量 (0-1)
    communication_reliability: float = 1.0      # 通信可靠性 (0-1)
    redundancy_level: int = 1                   # 冗余度 (>=1)

    # 扩展指标
    state_estimation_confidence: float = 1.0    # 状态估计置信度
    observability_gramian_rank: Optional[int] = None  # 可观性矩阵秩

    @property
    def overall_score(self) -> float:
        """综合可观性得分"""
        base_score = (
            self.sensor_coverage * 0.25 +
            self.sensor_health * 0.25 +
            self.data_quality * 0.2 +
            self.communication_reliability * 0.2 +
            self.state_estimation_confidence * 0.1
        )
        # 冗余度加成
        redundancy_bonus = min(0.1, (self.redundancy_level - 1) * 0.05)
        return min(1.0, base_score + redundancy_bonus)

    def to_dict(self) -> Dict:
        return {
            "sensor_coverage": self.sensor_coverage,
            "sensor_health": self.sensor_health,
            "data_quality": self.data_quality,
            "communication_reliability": self.communication_reliability,
            "redundancy_level": self.redundancy_level,
            "overall_score": self.overall_score
        }


@dataclass
class ControllabilityMetrics:
    """
    可控性指标

    评估系统是否可通过控制输入调整到目标状态
    """
    actuator_availability: float = 1.0          # 执行器可用率 (0-1)
    actuator_health: float = 1.0                # 执行器健康度 (0-1)
    control_authority: float = 1.0              # 控制权限 (0-1)
    response_capability: float = 1.0            # 响应能力 (0-1)

    # 执行器特性
    dead_zone_ratio: float = 0.0                # 死区比例 (0-1, 越小越好)
    response_delay: float = 0.0                 # 响应延迟 (s)
    max_response_delay: float = 5.0             # 最大允许延迟 (s)

    # 扩展指标
    controllability_gramian_rank: Optional[int] = None  # 可控性矩阵秩

    @property
    def overall_score(self) -> float:
        """综合可控性得分"""
        # 死区惩罚
        dead_zone_penalty = self.dead_zone_ratio * 0.3

        # 延迟惩罚
        delay_ratio = min(1.0, self.response_delay / self.max_response_delay)
        delay_penalty = delay_ratio * 0.2

        base_score = (
            self.actuator_availability * 0.3 +
            self.actuator_health * 0.25 +
            self.control_authority * 0.25 +
            self.response_capability * 0.2
        )

        return max(0, base_score - dead_zone_penalty - delay_penalty)

    def to_dict(self) -> Dict:
        return {
            "actuator_availability": self.actuator_availability,
            "actuator_health": self.actuator_health,
            "control_authority": self.control_authority,
            "response_capability": self.response_capability,
            "dead_zone_ratio": self.dead_zone_ratio,
            "response_delay": self.response_delay,
            "overall_score": self.overall_score
        }


@dataclass
class ODDReport:
    """
    ODD完整评估报告

    包含所有维度的评估结果和自主等级判定
    """
    timestamp: datetime
    system_name: str

    # 综合评估
    overall_score: float = 0.0                  # 综合ODD得分 (0-1)
    overall_status: ODDStatus = ODDStatus.UNKNOWN
    autonomy_level: AutonomyLevel = AutonomyLevel.L0_MANUAL

    # 维度得分
    dimension_scores: Dict[ODDDimension, DimensionScore] = field(default_factory=dict)

    # 可观性/可控性
    observability: Optional[ObservabilityMetrics] = None
    controllability: Optional[ControllabilityMetrics] = None

    # 违规记录
    violations: List[ODDViolation] = field(default_factory=list)
    critical_violations: List[str] = field(default_factory=list)

    # 建议和行动
    recommended_actions: List[str] = field(default_factory=list)
    mrm_triggered: bool = False                 # 是否触发最小风险动作
    degradation_mode: Optional[str] = None      # 降级模式

    def add_violation(self, violation: ODDViolation):
        """添加违规记录"""
        self.violations.append(violation)
        if violation.boundary.violation_severity == ViolationSeverity.CRITICAL:
            self.critical_violations.append(violation.message)

    def summary(self) -> str:
        """生成报告摘要"""
        lines = [
            "=" * 70,
            f"ODD评估报告 - {self.system_name}",
            "=" * 70,
            f"评估时间: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"📊 综合评估:",
            f"   ODD得分: {self.overall_score:.1%}",
            f"   运行状态: {self.overall_status.value}",
            f"   自主等级: {self.autonomy_level.name} ({self.autonomy_level.description})",
            "",
            f"📋 维度得分:",
        ]

        for dim, score in self.dimension_scores.items():
            status_icon = "✅" if score.status == ODDStatus.NOMINAL else \
                         "⚠️" if score.status == ODDStatus.MARGINAL else "❌"
            lines.append(f"   {status_icon} {dim.value}: {score.score:.1%} "
                        f"({score.violated_count}/{score.parameter_count}违规)")

        if self.observability:
            lines.extend([
                "",
                f"🔍 可观性: {self.observability.overall_score:.1%}",
                f"   传感器覆盖: {self.observability.sensor_coverage:.1%}",
                f"   数据质量: {self.observability.data_quality:.1%}"
            ])

        if self.controllability:
            lines.extend([
                "",
                f"🎮 可控性: {self.controllability.overall_score:.1%}",
                f"   执行器可用: {self.controllability.actuator_availability:.1%}",
                f"   死区比例: {self.controllability.dead_zone_ratio:.1%}"
            ])

        if self.critical_violations:
            lines.extend([
                "",
                "🚨 严重违规:"
            ])
            for v in self.critical_violations[:5]:
                lines.append(f"   • {v}")

        if self.mrm_triggered:
            lines.extend([
                "",
                "⛔ 触发最小风险动作 (MRM)!",
                f"   降级模式: {self.degradation_mode or '安全停车'}"
            ])

        if self.recommended_actions:
            lines.extend([
                "",
                "💡 建议行动:"
            ])
            for action in self.recommended_actions[:5]:
                lines.append(f"   • {action}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """导出为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "system_name": self.system_name,
            "overall_score": self.overall_score,
            "overall_status": self.overall_status.name,
            "autonomy_level": {
                "level": self.autonomy_level.level,
                "name": self.autonomy_level.name,
                "description": self.autonomy_level.description
            },
            "dimension_scores": {
                dim.name: score.to_dict()
                for dim, score in self.dimension_scores.items()
            },
            "observability": self.observability.to_dict() if self.observability else None,
            "controllability": self.controllability.to_dict() if self.controllability else None,
            "violations": [v.to_dict() for v in self.violations],
            "mrm_triggered": self.mrm_triggered,
            "degradation_mode": self.degradation_mode,
            "recommended_actions": self.recommended_actions
        }


# ==========================================
# ODD分析器基类
# ==========================================

class BaseODDAnalyzer(ABC):
    """
    ODD分析器抽象基类

    提供设计运行域分析的通用框架。子类需实现
    具体的边界定义和状态获取方法。
    """

    def __init__(self, system_name: str):
        self.system_name = system_name
        self._boundaries: Dict[str, ODDBoundary] = {}
        self._report: Optional[ODDReport] = None

        # 维度权重配置
        self.dimension_weights: Dict[ODDDimension, float] = {
            ODDDimension.ENVIRONMENT: 0.2,
            ODDDimension.DYNAMICS: 0.3,
            ODDDimension.INFRASTRUCTURE: 0.25,
            ODDDimension.DIGITAL: 0.25
        }

        # 自主等级阈值
        self.autonomy_thresholds: Dict[AutonomyLevel, float] = {
            AutonomyLevel.L5_FULL: 0.95,
            AutonomyLevel.L4_HIGH: 0.85,
            AutonomyLevel.L3_CONDITIONAL: 0.70,
            AutonomyLevel.L2_PARTIAL: 0.50,
            AutonomyLevel.L1_ASSISTED: 0.30,
            AutonomyLevel.L0_MANUAL: 0.0
        }

        # 初始化边界
        self._initialize_boundaries()

    @abstractmethod
    def _initialize_boundaries(self):
        """
        初始化ODD边界定义

        子类必须实现此方法，定义具体的ODD边界参数
        """
        pass

    @abstractmethod
    def get_current_state(self) -> Dict[str, float]:
        """
        获取当前系统状态

        Returns:
            {参数名: 当前值}
        """
        pass

    def add_boundary(self, boundary: ODDBoundary):
        """添加ODD边界"""
        self._boundaries[boundary.name] = boundary

    def remove_boundary(self, name: str):
        """移除ODD边界"""
        if name in self._boundaries:
            del self._boundaries[name]

    def get_boundary(self, name: str) -> Optional[ODDBoundary]:
        """获取ODD边界"""
        return self._boundaries.get(name)

    def get_boundaries_by_dimension(self, dimension: ODDDimension) -> List[ODDBoundary]:
        """获取指定维度的所有边界"""
        return [b for b in self._boundaries.values() if b.dimension == dimension]

    def analyze(self,
                current_state: Optional[Dict[str, float]] = None,
                observability: Optional[ObservabilityMetrics] = None,
                controllability: Optional[ControllabilityMetrics] = None) -> ODDReport:
        """
        执行ODD分析

        Args:
            current_state: 当前状态，若不提供则调用get_current_state()
            observability: 可观性指标
            controllability: 可控性指标

        Returns:
            ODDReport
        """
        if current_state is None:
            current_state = self.get_current_state()

        timestamp = datetime.now()
        report = ODDReport(
            timestamp=timestamp,
            system_name=self.system_name
        )

        # 评估各维度
        for dimension in ODDDimension:
            dim_score = self._evaluate_dimension(dimension, current_state, timestamp)
            report.dimension_scores[dimension] = dim_score
            report.violations.extend(dim_score.violations)

        # 计算综合得分
        report.overall_score = self._calculate_overall_score(report.dimension_scores)

        # 加入可观性/可控性
        report.observability = observability
        report.controllability = controllability

        if observability and controllability:
            # 调整综合得分
            oc_score = (observability.overall_score + controllability.overall_score) / 2
            report.overall_score = report.overall_score * 0.7 + oc_score * 0.3

        # 确定状态
        report.overall_status = self._determine_status(report.overall_score, report.violations)

        # 确定自主等级
        report.autonomy_level = self._determine_autonomy_level(
            report.overall_score,
            report.critical_violations
        )

        # 检查是否需要MRM
        report.mrm_triggered = self._check_mrm_condition(report)
        if report.mrm_triggered:
            report.degradation_mode = self._determine_degradation_mode(report)

        # 生成建议
        report.recommended_actions = self._generate_recommendations(report)

        self._report = report
        return report

    def _evaluate_dimension(self,
                           dimension: ODDDimension,
                           current_state: Dict[str, float],
                           timestamp: datetime) -> DimensionScore:
        """评估单个维度"""
        boundaries = self.get_boundaries_by_dimension(dimension)

        if not boundaries:
            return DimensionScore(
                dimension=dimension,
                score=1.0,
                status=ODDStatus.NOMINAL,
                parameter_count=0,
                violated_count=0
            )

        total_weight = sum(b.weight for b in boundaries)
        weighted_score = 0.0
        violations = []
        details = {}

        for boundary in boundaries:
            value = current_state.get(boundary.name)
            status, score = boundary.check_value(value)

            weighted_score += score * boundary.weight
            details[boundary.name] = score

            if status in (ODDStatus.EXCEEDED, ODDStatus.MARGINAL):
                violation = ODDViolation(
                    timestamp=timestamp,
                    boundary=boundary,
                    actual_value=value if value is not None else float('nan'),
                    status=status,
                    score=score,
                    message=f"{boundary.name}: 当前值 {value} "
                           f"超出范围 [{boundary.min_value}, {boundary.max_value}]"
                )
                violations.append(violation)

        final_score = weighted_score / total_weight if total_weight > 0 else 0

        # 确定维度状态
        if any(v.status == ODDStatus.EXCEEDED for v in violations):
            dim_status = ODDStatus.EXCEEDED
        elif any(v.status == ODDStatus.MARGINAL for v in violations):
            dim_status = ODDStatus.MARGINAL
        else:
            dim_status = ODDStatus.NOMINAL

        return DimensionScore(
            dimension=dimension,
            score=final_score,
            status=dim_status,
            parameter_count=len(boundaries),
            violated_count=len(violations),
            violations=violations,
            details=details
        )

    def _calculate_overall_score(self,
                                dimension_scores: Dict[ODDDimension, DimensionScore]) -> float:
        """计算综合得分"""
        total_weight = sum(self.dimension_weights.values())
        weighted_sum = sum(
            dimension_scores[dim].score * self.dimension_weights[dim]
            for dim in ODDDimension
            if dim in dimension_scores
        )
        return weighted_sum / total_weight if total_weight > 0 else 0

    def _determine_status(self, score: float, violations: List[ODDViolation]) -> ODDStatus:
        """确定整体状态"""
        # 检查是否有严重违规
        has_critical = any(
            v.boundary.violation_severity == ViolationSeverity.CRITICAL
            for v in violations
        )

        if has_critical or score < 0.3:
            return ODDStatus.EXCEEDED
        elif score < 0.7:
            return ODDStatus.DEGRADED
        elif score < 0.85:
            return ODDStatus.MARGINAL
        else:
            return ODDStatus.NOMINAL

    def _determine_autonomy_level(self,
                                  score: float,
                                  critical_violations: List[str]) -> AutonomyLevel:
        """确定自主等级"""
        # 有严重违规直接降到L0
        if critical_violations:
            return AutonomyLevel.L0_MANUAL

        for level, threshold in sorted(
            self.autonomy_thresholds.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if score >= threshold:
                return level

        return AutonomyLevel.L0_MANUAL

    def _check_mrm_condition(self, report: ODDReport) -> bool:
        """检查是否触发最小风险动作"""
        # 有严重违规
        if report.critical_violations:
            return True
        # 综合得分过低
        if report.overall_score < 0.3:
            return True
        # 多个维度同时超标
        exceeded_dims = sum(
            1 for s in report.dimension_scores.values()
            if s.status == ODDStatus.EXCEEDED
        )
        if exceeded_dims >= 2:
            return True
        return False

    def _determine_degradation_mode(self, report: ODDReport) -> str:
        """确定降级模式"""
        # 根据违规类型确定降级模式
        exceeded_dims = [
            dim for dim, score in report.dimension_scores.items()
            if score.status == ODDStatus.EXCEEDED
        ]

        if ODDDimension.DIGITAL in exceeded_dims:
            return "通信降级：切换到本地控制"
        elif ODDDimension.INFRASTRUCTURE in exceeded_dims:
            return "设备降级：启用备用执行器"
        elif ODDDimension.DYNAMICS in exceeded_dims:
            return "动态降级：降低控制增益"
        elif ODDDimension.ENVIRONMENT in exceeded_dims:
            return "环境超限：安全停车"
        else:
            return "标准降级：切换人工模式"

    def _generate_recommendations(self, report: ODDReport) -> List[str]:
        """生成建议行动"""
        recommendations = []

        for dim, score in report.dimension_scores.items():
            if score.status == ODDStatus.EXCEEDED:
                if dim == ODDDimension.ENVIRONMENT:
                    recommendations.append("环境超限：建议暂停自动控制，等待条件恢复")
                elif dim == ODDDimension.DYNAMICS:
                    recommendations.append("动力学超限：建议降低控制强度，避免激进操作")
                elif dim == ODDDimension.INFRASTRUCTURE:
                    recommendations.append("设备异常：建议检查执行器和传感器状态")
                elif dim == ODDDimension.DIGITAL:
                    recommendations.append("通信异常：建议检查网络连接和数据链路")
            elif score.status == ODDStatus.MARGINAL:
                recommendations.append(f"{dim.value}接近边界，建议加强监控")

        if report.observability and report.observability.overall_score < 0.7:
            recommendations.append("可观性不足：建议增加传感器冗余或检查数据质量")

        if report.controllability and report.controllability.overall_score < 0.7:
            recommendations.append("可控性不足：建议检查执行器响应和控制权限")

        return recommendations

    def get_report(self) -> Optional[ODDReport]:
        """获取最近的分析报告"""
        return self._report

    def print_report(self):
        """打印分析报告"""
        if self._report:
            print(self._report.summary())
        else:
            print("尚未执行分析，请先调用 analyze() 方法")


# ==========================================
# 水网系统ODD分析器实现
# ==========================================

class WaterNetworkODDAnalyzer(BaseODDAnalyzer):
    """
    水网系统ODD分析器

    针对水利工程的ODD分析实现，包含：
    - 水位/流量边界
    - 闸门/泵站约束
    - 传感器/通信要求
    """

    def __init__(self, system_name: str = "水网智能体"):
        self._current_state: Dict[str, float] = {}
        super().__init__(system_name)

    def _initialize_boundaries(self):
        """初始化水网ODD边界"""
        # ========== 环境与边界维度 ==========
        self.add_boundary(ODDBoundary(
            name="water_level_error",
            dimension=ODDDimension.ENVIRONMENT,
            description="水位偏差",
            unit="m",
            min_value=-0.15,
            max_value=0.15,
            nominal_value=0.0,
            constraint_type=ConstraintType.HARD,
            violation_severity=ViolationSeverity.MAJOR,
            weight=1.5
        ))

        self.add_boundary(ODDBoundary(
            name="rainfall_intensity",
            dimension=ODDDimension.ENVIRONMENT,
            description="降雨强度",
            unit="mm/h",
            min_value=0,
            max_value=50,
            constraint_type=ConstraintType.SOFT,
            violation_severity=ViolationSeverity.WARNING,
            weight=0.8
        ))

        self.add_boundary(ODDBoundary(
            name="wind_speed",
            dimension=ODDDimension.ENVIRONMENT,
            description="风速(风壅)",
            unit="m/s",
            min_value=0,
            max_value=15,
            constraint_type=ConstraintType.SOFT,
            violation_severity=ViolationSeverity.MINOR,
            weight=0.5
        ))

        # ========== 动力学限制维度 ==========
        self.add_boundary(ODDBoundary(
            name="water_level_rate",
            dimension=ODDDimension.DYNAMICS,
            description="水位变化率",
            unit="m/h",
            min_value=-0.15,
            max_value=0.15,
            nominal_value=0.0,
            constraint_type=ConstraintType.HARD,
            violation_severity=ViolationSeverity.MAJOR,
            weight=1.2
        ))

        self.add_boundary(ODDBoundary(
            name="flow_disturbance_rate",
            dimension=ODDDimension.DYNAMICS,
            description="流量扰动率",
            unit="%",
            min_value=0,
            max_value=15,
            constraint_type=ConstraintType.HARD,
            violation_severity=ViolationSeverity.MAJOR,
            weight=1.3
        ))

        self.add_boundary(ODDBoundary(
            name="froude_number",
            dimension=ODDDimension.DYNAMICS,
            description="弗劳德数(流态)",
            unit="-",
            min_value=0,
            max_value=0.9,  # 保持缓流
            constraint_type=ConstraintType.HARD,
            violation_severity=ViolationSeverity.CRITICAL,
            weight=1.5
        ))

        # ========== 设备能力维度 ==========
        self.add_boundary(ODDBoundary(
            name="gate_deadzone",
            dimension=ODDDimension.INFRASTRUCTURE,
            description="闸门死区",
            unit="m",
            min_value=0,
            max_value=0.05,
            constraint_type=ConstraintType.SOFT,
            violation_severity=ViolationSeverity.MINOR,
            weight=1.0
        ))

        self.add_boundary(ODDBoundary(
            name="sensor_accuracy",
            dimension=ODDDimension.INFRASTRUCTURE,
            description="传感器精度",
            unit="-",
            min_value=0.95,
            max_value=1.0,
            constraint_type=ConstraintType.HARD,
            violation_severity=ViolationSeverity.MAJOR,
            weight=1.2
        ))

        self.add_boundary(ODDBoundary(
            name="actuator_response_time",
            dimension=ODDDimension.INFRASTRUCTURE,
            description="执行器响应时间",
            unit="s",
            min_value=0,
            max_value=30,
            constraint_type=ConstraintType.SOFT,
            violation_severity=ViolationSeverity.MINOR,
            weight=0.8
        ))

        # ========== 数字化支撑维度 ==========
        self.add_boundary(ODDBoundary(
            name="communication_latency",
            dimension=ODDDimension.DIGITAL,
            description="通信时延",
            unit="s",
            min_value=0,
            max_value=1.5,
            constraint_type=ConstraintType.HARD,
            violation_severity=ViolationSeverity.MAJOR,
            weight=1.3
        ))

        self.add_boundary(ODDBoundary(
            name="packet_loss_rate",
            dimension=ODDDimension.DIGITAL,
            description="丢包率",
            unit="%",
            min_value=0,
            max_value=0.1,
            constraint_type=ConstraintType.HARD,
            violation_severity=ViolationSeverity.MAJOR,
            weight=1.2
        ))

        self.add_boundary(ODDBoundary(
            name="compute_load",
            dimension=ODDDimension.DIGITAL,
            description="计算负载",
            unit="%",
            min_value=0,
            max_value=80,
            constraint_type=ConstraintType.SOFT,
            violation_severity=ViolationSeverity.WARNING,
            weight=0.7
        ))

    def set_state(self, state: Dict[str, float]):
        """设置当前状态"""
        self._current_state = state.copy()

    def update_state(self, **kwargs):
        """更新部分状态"""
        self._current_state.update(kwargs)

    def get_current_state(self) -> Dict[str, float]:
        """获取当前状态"""
        return self._current_state.copy()

    def quick_check(self, state: Dict[str, float]) -> Tuple[bool, AutonomyLevel, List[str]]:
        """
        快速ODD检查

        Args:
            state: 当前状态

        Returns:
            (是否在ODD内, 自主等级, 警告列表)
        """
        self.set_state(state)
        report = self.analyze()

        warnings = []
        for v in report.violations:
            if v.status == ODDStatus.EXCEEDED:
                warnings.append(f"[超出] {v.message}")
            elif v.status == ODDStatus.MARGINAL:
                warnings.append(f"[边界] {v.message}")

        in_odd = report.overall_status in (ODDStatus.NOMINAL, ODDStatus.MARGINAL)
        return in_odd, report.autonomy_level, warnings


# ==========================================
# 便捷函数
# ==========================================

def calculate_odd_reliability(
    physical_score: float,
    digital_score: float,
    actuator_score: float,
    aggregation: str = "multiplicative"
) -> float:
    """
    计算ODD可靠性

    Args:
        physical_score: 物理环境得分 (0-1)
        digital_score: 数字化得分 (0-1)
        actuator_score: 执行器得分 (0-1)
        aggregation: 聚合方式
            - "multiplicative": 乘法模型（木桶效应）
            - "weighted": 加权平均

    Returns:
        可靠性得分 (0-1)
    """
    scores = [physical_score, digital_score, actuator_score]

    if aggregation == "multiplicative":
        # 乘法模型：任一维度崩溃则整体失效
        reliability = 1.0
        for s in scores:
            reliability *= s
        return reliability
    else:
        # 加权平均
        weights = [0.4, 0.3, 0.3]
        return sum(s * w for s, w in zip(scores, weights))


def determine_autonomy_from_score(score: float) -> AutonomyLevel:
    """
    根据ODD得分确定自主等级

    Args:
        score: ODD综合得分 (0-1)

    Returns:
        AutonomyLevel
    """
    if score >= 0.95:
        return AutonomyLevel.L5_FULL
    elif score >= 0.85:
        return AutonomyLevel.L4_HIGH
    elif score >= 0.70:
        return AutonomyLevel.L3_CONDITIONAL
    elif score >= 0.50:
        return AutonomyLevel.L2_PARTIAL
    elif score >= 0.30:
        return AutonomyLevel.L1_ASSISTED
    else:
        return AutonomyLevel.L0_MANUAL


def create_water_odd_analyzer(
    system_name: str,
    custom_boundaries: Optional[List[ODDBoundary]] = None
) -> WaterNetworkODDAnalyzer:
    """
    创建水网ODD分析器（工厂函数）

    Args:
        system_name: 系统名称
        custom_boundaries: 自定义边界列表

    Returns:
        WaterNetworkODDAnalyzer
    """
    analyzer = WaterNetworkODDAnalyzer(system_name)

    if custom_boundaries:
        for boundary in custom_boundaries:
            analyzer.add_boundary(boundary)

    return analyzer


# ==========================================
# ODD自动构建器 - 从设备配置推导ODD边界
# ==========================================

@dataclass
class SensorProfile:
    """
    传感器配置描述

    用于从传感器参数自动推导可观性边界
    """
    name: str                               # 传感器名称
    sensor_type: str                        # 类型: level, flow, pressure, temperature
    position: str                           # 安装位置

    # 测量特性
    range_min: float = 0.0                  # 量程下限
    range_max: float = 100.0                # 量程上限
    accuracy: float = 0.01                  # 精度 (绝对值)
    accuracy_percent: float = 0.1           # 精度 (百分比)
    resolution: float = 0.001               # 分辨率

    # 动态特性
    time_constant: float = 0.5              # 时间常数 (s)
    sampling_rate: float = 1.0              # 采样率 (Hz)

    # 环境适应性
    temp_range: Tuple[float, float] = (-20, 60)  # 工作温度范围
    ip_rating: str = "IP68"                 # 防护等级

    # 可靠性
    mtbf: float = 50000                     # 平均故障间隔 (h)
    is_redundant: bool = False              # 是否有冗余
    redundancy_count: int = 1               # 冗余数量

    def get_observability_contribution(self) -> float:
        """计算对可观性的贡献度 (0-1)"""
        # 基础分
        base = 0.7

        # 精度加成
        if self.accuracy_percent < 0.5:
            base += 0.1
        elif self.accuracy_percent > 2.0:
            base -= 0.2

        # 冗余加成
        if self.redundancy_count >= 2:
            base += 0.1
        elif self.redundancy_count >= 3:
            base += 0.15

        # 时间常数惩罚
        if self.time_constant > 2.0:
            base -= 0.1

        return min(1.0, max(0.0, base))


@dataclass
class ActuatorProfile:
    """
    执行器配置描述

    用于从执行器参数自动推导可控性边界
    """
    name: str                               # 执行器名称
    actuator_type: str                      # 类型: gate, valve, pump
    position: str                           # 安装位置

    # 行程与量程
    stroke_min: float = 0.0                 # 最小开度
    stroke_max: float = 1.0                 # 最大开度
    max_opening: float = 6.0                # 最大开度 (物理量, m)

    # 速率特性
    max_rate: float = 0.01                  # 最大速率 (%/s)
    min_step: float = 0.005                 # 最小步进

    # 死区与精度
    dead_band: float = 0.005                # 死区
    positioning_accuracy: float = 0.01      # 定位精度

    # 响应特性
    response_time: float = 5.0              # 响应时间 (s)
    settling_time: float = 10.0             # 稳定时间 (s)

    # 力矩与负载
    max_torque: float = 1000                # 最大力矩 (N·m)
    stall_protection: bool = True           # 堵转保护

    # 可靠性
    mtbf: float = 30000                     # 平均故障间隔 (h)
    is_redundant: bool = False              # 是否有备用
    backup_mode: str = "manual"             # 备用模式

    def get_controllability_contribution(self) -> float:
        """计算对可控性的贡献度 (0-1)"""
        base = 0.7

        # 死区惩罚
        if self.dead_band > 0.02:
            base -= 0.15
        elif self.dead_band > 0.01:
            base -= 0.05

        # 响应时间惩罚
        if self.response_time > 30:
            base -= 0.2
        elif self.response_time > 10:
            base -= 0.1

        # 速率加成
        if self.max_rate > 0.02:
            base += 0.1

        # 冗余加成
        if self.is_redundant:
            base += 0.1

        return min(1.0, max(0.0, base))


@dataclass
class ChannelProfile:
    """
    渠道/管道配置描述

    用于推导动力学边界
    """
    name: str
    channel_type: str                       # channel, pipeline, tunnel
    length: float                           # 长度 (m)
    diameter: float = 0.0                   # 直径/水力半径 (m)
    bottom_width: float = 0.0               # 底宽 (明渠)
    side_slope: float = 0.0                 # 边坡系数

    # 水力参数
    design_flow: float = 0.0                # 设计流量
    max_flow: float = 0.0                   # 最大流量
    min_flow: float = 0.0                   # 最小流量
    design_velocity: float = 0.0            # 设计流速

    # 坡降与糙率
    bed_slope: float = 0.0001               # 底坡
    manning_n: float = 0.014                # 糙率

    # 水位边界
    normal_level: float = 0.0               # 正常水位
    max_level: float = 0.0                  # 最高水位
    min_level: float = 0.0                  # 最低水位


@dataclass
class CommunicationProfile:
    """
    通信配置描述

    用于推导数字化支撑边界
    """
    network_type: str = "5G"                # 网络类型
    bandwidth: float = 100.0                # 带宽 (Mbps)
    latency_typical: float = 0.1            # 典型时延 (s)
    latency_max: float = 1.0                # 最大时延 (s)
    packet_loss_rate: float = 0.001         # 丢包率
    availability: float = 0.9999            # 可用率

    # 冗余
    has_backup: bool = True                 # 是否有备份链路
    backup_type: str = "4G"                 # 备份类型

    # 边缘计算
    edge_compute_available: bool = True     # 边缘计算可用
    edge_response_time: float = 0.01        # 边缘响应时间 (s)


class ODDProfileBuilder:
    """
    ODD边界自动构建器

    从传感器、执行器、渠道、通信配置自动推导ODD边界。
    这是ODD从"手动定义"到"自动推导"的核心组件。
    """

    def __init__(self, system_name: str):
        self.system_name = system_name
        self.sensors: List[SensorProfile] = []
        self.actuators: List[ActuatorProfile] = []
        self.channels: List[ChannelProfile] = []
        self.communication: Optional[CommunicationProfile] = None

        # 推导的边界
        self._boundaries: List[ODDBoundary] = []
        self._observability: Optional[ObservabilityMetrics] = None
        self._controllability: Optional[ControllabilityMetrics] = None

    def add_sensor(self, sensor: SensorProfile):
        """添加传感器配置"""
        self.sensors.append(sensor)

    def add_actuator(self, actuator: ActuatorProfile):
        """添加执行器配置"""
        self.actuators.append(actuator)

    def add_channel(self, channel: ChannelProfile):
        """添加渠道配置"""
        self.channels.append(channel)

    def set_communication(self, comm: CommunicationProfile):
        """设置通信配置"""
        self.communication = comm

    def build(self) -> Tuple[List[ODDBoundary], ObservabilityMetrics, ControllabilityMetrics]:
        """
        构建ODD边界

        Returns:
            (边界列表, 可观性指标, 可控性指标)
        """
        self._boundaries = []

        # 1. 从传感器推导可观性和设备边界
        self._build_sensor_boundaries()

        # 2. 从执行器推导可控性和设备边界
        self._build_actuator_boundaries()

        # 3. 从渠道推导动力学边界
        self._build_channel_boundaries()

        # 4. 从通信推导数字化边界
        self._build_communication_boundaries()

        # 5. 计算综合可观性/可控性
        self._compute_observability()
        self._compute_controllability()

        return self._boundaries, self._observability, self._controllability

    def _build_sensor_boundaries(self):
        """从传感器配置推导边界"""
        if not self.sensors:
            return

        # 统计传感器类型
        level_sensors = [s for s in self.sensors if s.sensor_type == 'level']
        flow_sensors = [s for s in self.sensors if s.sensor_type == 'flow']

        # 水位测量精度边界
        if level_sensors:
            best_accuracy = min(s.accuracy for s in level_sensors)
            worst_accuracy = max(s.accuracy for s in level_sensors)

            self._boundaries.append(ODDBoundary(
                name="level_sensor_accuracy",
                dimension=ODDDimension.INFRASTRUCTURE,
                description="水位传感器精度要求",
                unit="m",
                min_value=0,
                max_value=worst_accuracy * 3,  # 3倍精度作为边界
                nominal_value=best_accuracy,
                constraint_type=ConstraintType.HARD,
                violation_severity=ViolationSeverity.MAJOR
            ))

            # 水位变化率边界（基于采样率和精度）
            max_rate = min(s.accuracy / s.time_constant for s in level_sensors)
            self._boundaries.append(ODDBoundary(
                name="level_rate_limit",
                dimension=ODDDimension.DYNAMICS,
                description="可观测的水位变化率上限",
                unit="m/s",
                min_value=-max_rate * 10,
                max_value=max_rate * 10,
                constraint_type=ConstraintType.HARD,
                violation_severity=ViolationSeverity.MAJOR
            ))

        # 流量测量边界
        if flow_sensors:
            min_flow = max(s.range_min for s in flow_sensors)
            max_flow = min(s.range_max for s in flow_sensors)

            self._boundaries.append(ODDBoundary(
                name="flow_measurement_range",
                dimension=ODDDimension.ENVIRONMENT,
                description="可测量流量范围",
                unit="m³/s",
                min_value=min_flow,
                max_value=max_flow,
                constraint_type=ConstraintType.HARD,
                violation_severity=ViolationSeverity.MAJOR
            ))

    def _build_actuator_boundaries(self):
        """从执行器配置推导边界"""
        if not self.actuators:
            return

        gates = [a for a in self.actuators if a.actuator_type == 'gate']
        valves = [a for a in self.actuators if a.actuator_type == 'valve']
        pumps = [a for a in self.actuators if a.actuator_type == 'pump']

        # 闸门死区边界
        if gates:
            max_deadband = max(g.dead_band for g in gates)
            max_response = max(g.response_time for g in gates)

            self._boundaries.append(ODDBoundary(
                name="gate_deadzone_limit",
                dimension=ODDDimension.INFRASTRUCTURE,
                description="闸门死区上限",
                unit="-",
                min_value=0,
                max_value=max_deadband * 2,
                nominal_value=max_deadband,
                constraint_type=ConstraintType.SOFT,
                violation_severity=ViolationSeverity.MINOR
            ))

            self._boundaries.append(ODDBoundary(
                name="gate_response_limit",
                dimension=ODDDimension.INFRASTRUCTURE,
                description="闸门响应时间上限",
                unit="s",
                min_value=0,
                max_value=max_response * 2,
                nominal_value=max_response,
                constraint_type=ConstraintType.SOFT,
                violation_severity=ViolationSeverity.MINOR
            ))

            # 控制步长边界（基于最大速率）
            min_rate = min(g.max_rate for g in gates)
            self._boundaries.append(ODDBoundary(
                name="min_control_interval",
                dimension=ODDDimension.DYNAMICS,
                description="最小控制间隔",
                unit="s",
                min_value=1.0 / min_rate if min_rate > 0 else 100,
                max_value=3600,
                constraint_type=ConstraintType.ADVISORY,
                violation_severity=ViolationSeverity.WARNING
            ))

    def _build_channel_boundaries(self):
        """从渠道配置推导动力学边界"""
        if not self.channels:
            return

        for channel in self.channels:
            # 流量边界
            if channel.max_flow > 0:
                self._boundaries.append(ODDBoundary(
                    name=f"{channel.name}_flow_range",
                    dimension=ODDDimension.ENVIRONMENT,
                    description=f"{channel.name}流量范围",
                    unit="m³/s",
                    min_value=channel.min_flow,
                    max_value=channel.max_flow,
                    nominal_value=channel.design_flow,
                    constraint_type=ConstraintType.HARD,
                    violation_severity=ViolationSeverity.MAJOR
                ))

            # 水位边界
            if channel.max_level > 0:
                self._boundaries.append(ODDBoundary(
                    name=f"{channel.name}_level_range",
                    dimension=ODDDimension.ENVIRONMENT,
                    description=f"{channel.name}水位范围",
                    unit="m",
                    min_value=channel.min_level,
                    max_value=channel.max_level,
                    nominal_value=channel.normal_level,
                    constraint_type=ConstraintType.HARD,
                    violation_severity=ViolationSeverity.CRITICAL
                ))

            # 弗劳德数边界（保持缓流）
            if channel.design_velocity > 0 and channel.diameter > 0:
                # Fr = v / sqrt(g * h)
                g = 9.81
                fr_design = channel.design_velocity / np.sqrt(g * channel.diameter)
                self._boundaries.append(ODDBoundary(
                    name=f"{channel.name}_froude",
                    dimension=ODDDimension.DYNAMICS,
                    description=f"{channel.name}弗劳德数",
                    unit="-",
                    min_value=0,
                    max_value=0.9,  # 缓流上限
                    nominal_value=fr_design,
                    constraint_type=ConstraintType.HARD,
                    violation_severity=ViolationSeverity.CRITICAL
                ))

    def _build_communication_boundaries(self):
        """从通信配置推导数字化边界"""
        if not self.communication:
            # 使用默认值
            self._boundaries.append(ODDBoundary(
                name="comm_latency",
                dimension=ODDDimension.DIGITAL,
                description="通信时延",
                unit="s",
                min_value=0,
                max_value=1.5,
                constraint_type=ConstraintType.HARD,
                violation_severity=ViolationSeverity.MAJOR
            ))
            return

        comm = self.communication

        self._boundaries.append(ODDBoundary(
            name="comm_latency",
            dimension=ODDDimension.DIGITAL,
            description="通信时延",
            unit="s",
            min_value=0,
            max_value=comm.latency_max,
            nominal_value=comm.latency_typical,
            constraint_type=ConstraintType.HARD,
            violation_severity=ViolationSeverity.MAJOR
        ))

        self._boundaries.append(ODDBoundary(
            name="packet_loss",
            dimension=ODDDimension.DIGITAL,
            description="丢包率",
            unit="%",
            min_value=0,
            max_value=comm.packet_loss_rate * 100 * 10,  # 10倍作为边界
            nominal_value=comm.packet_loss_rate * 100,
            constraint_type=ConstraintType.HARD,
            violation_severity=ViolationSeverity.MAJOR
        ))

    def _compute_observability(self):
        """计算综合可观性"""
        if not self.sensors:
            self._observability = ObservabilityMetrics()
            return

        # 传感器覆盖率（假设需要水位和流量两种）
        sensor_types = set(s.sensor_type for s in self.sensors)
        required_types = {'level', 'flow'}
        coverage = len(sensor_types & required_types) / len(required_types)

        # 传感器健康度（基于贡献度平均）
        health = np.mean([s.get_observability_contribution() for s in self.sensors])

        # 冗余度
        redundancy = max(s.redundancy_count for s in self.sensors)

        # 数据质量（基于精度）
        avg_accuracy_pct = np.mean([s.accuracy_percent for s in self.sensors])
        data_quality = max(0, 1 - avg_accuracy_pct / 5)  # 5%精度时质量为0

        # 通信可靠性
        comm_reliability = 1.0
        if self.communication:
            comm_reliability = self.communication.availability

        self._observability = ObservabilityMetrics(
            sensor_coverage=coverage,
            sensor_health=health,
            data_quality=data_quality,
            communication_reliability=comm_reliability,
            redundancy_level=redundancy
        )

    def _compute_controllability(self):
        """计算综合可控性"""
        if not self.actuators:
            self._controllability = ControllabilityMetrics()
            return

        # 执行器可用率
        availability = 1.0  # 假设全部可用

        # 执行器健康度
        health = np.mean([a.get_controllability_contribution() for a in self.actuators])

        # 死区比例
        avg_deadband = np.mean([a.dead_band for a in self.actuators])

        # 响应延迟
        max_response = max(a.response_time for a in self.actuators)

        self._controllability = ControllabilityMetrics(
            actuator_availability=availability,
            actuator_health=health,
            control_authority=1.0,
            response_capability=health,
            dead_zone_ratio=avg_deadband,
            response_delay=max_response,
            max_response_delay=max_response * 2
        )

    def create_analyzer(self) -> 'ConfiguredODDAnalyzer':
        """
        创建配置好的ODD分析器

        Returns:
            ConfiguredODDAnalyzer实例
        """
        boundaries, obs, ctrl = self.build()
        return ConfiguredODDAnalyzer(
            self.system_name,
            boundaries,
            obs,
            ctrl
        )


class ConfiguredODDAnalyzer(BaseODDAnalyzer):
    """
    通过配置构建的ODD分析器

    与WaterNetworkODDAnalyzer不同，此分析器的边界
    完全由ODDProfileBuilder从设备配置推导
    """

    def __init__(self,
                 system_name: str,
                 boundaries: List[ODDBoundary],
                 observability: ObservabilityMetrics,
                 controllability: ControllabilityMetrics):
        self._preset_boundaries = boundaries
        self._preset_observability = observability
        self._preset_controllability = controllability
        self._current_state: Dict[str, float] = {}
        super().__init__(system_name)

    def _initialize_boundaries(self):
        """初始化边界（使用预设）"""
        for boundary in self._preset_boundaries:
            self.add_boundary(boundary)

    def get_current_state(self) -> Dict[str, float]:
        return self._current_state.copy()

    def set_state(self, state: Dict[str, float]):
        self._current_state = state.copy()

    def analyze_with_preset(self,
                           current_state: Dict[str, float]) -> ODDReport:
        """使用预设的可观性/可控性进行分析"""
        self.set_state(current_state)
        return self.analyze(
            current_state,
            self._preset_observability,
            self._preset_controllability
        )


# ==========================================
# 世界模型约束接口
# ==========================================

@dataclass
class WorldModelConstraint:
    """
    世界模型生成约束

    定义数字孪生/仿真模型必须遵守的边界条件。
    确保生成的场景在ODD范围内。
    """
    name: str
    parameter: str                          # 参数名称
    min_value: float                        # 最小值
    max_value: float                        # 最大值
    nominal_value: Optional[float] = None   # 标称值
    distribution: str = "uniform"           # 分布类型: uniform, normal, truncated_normal
    std_dev: Optional[float] = None         # 标准差（正态分布时）

    def sample(self, rng: Optional[np.random.Generator] = None) -> float:
        """
        在约束范围内采样

        Args:
            rng: 随机数生成器

        Returns:
            采样值
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.distribution == "uniform":
            return rng.uniform(self.min_value, self.max_value)

        elif self.distribution == "normal":
            if self.nominal_value is None or self.std_dev is None:
                return rng.uniform(self.min_value, self.max_value)
            value = rng.normal(self.nominal_value, self.std_dev)
            return np.clip(value, self.min_value, self.max_value)

        elif self.distribution == "truncated_normal":
            if self.nominal_value is None or self.std_dev is None:
                return rng.uniform(self.min_value, self.max_value)
            # 截断正态分布
            while True:
                value = rng.normal(self.nominal_value, self.std_dev)
                if self.min_value <= value <= self.max_value:
                    return value

        return rng.uniform(self.min_value, self.max_value)

    def is_valid(self, value: float) -> bool:
        """检查值是否在约束范围内"""
        return self.min_value <= value <= self.max_value


class WorldModelODDEnvelope:
    """
    世界模型ODD包络

    为数字孪生和仿真引擎提供ODD约束接口。
    确保模型生成的所有场景都在设计运行域内。
    """

    def __init__(self, analyzer: BaseODDAnalyzer):
        self.analyzer = analyzer
        self._constraints: Dict[str, WorldModelConstraint] = {}
        self._build_constraints()

    def _build_constraints(self):
        """从ODD边界构建世界模型约束"""
        for name, boundary in self.analyzer._boundaries.items():
            if boundary.min_value is not None and boundary.max_value is not None:
                # 确定分布类型
                if boundary.nominal_value is not None:
                    distribution = "truncated_normal"
                    range_val = boundary.max_value - boundary.min_value
                    std_dev = range_val / 6  # 99.7%落在范围内
                else:
                    distribution = "uniform"
                    std_dev = None

                self._constraints[name] = WorldModelConstraint(
                    name=boundary.description,
                    parameter=name,
                    min_value=boundary.min_value,
                    max_value=boundary.max_value,
                    nominal_value=boundary.nominal_value,
                    distribution=distribution,
                    std_dev=std_dev
                )

    def get_constraint(self, parameter: str) -> Optional[WorldModelConstraint]:
        """获取指定参数的约束"""
        return self._constraints.get(parameter)

    def get_all_constraints(self) -> Dict[str, WorldModelConstraint]:
        """获取所有约束"""
        return self._constraints.copy()

    def sample_scenario(self,
                       parameters: Optional[List[str]] = None,
                       rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        """
        采样一个符合ODD的场景

        Args:
            parameters: 要采样的参数列表，None表示全部
            rng: 随机数生成器

        Returns:
            {参数名: 采样值}
        """
        if rng is None:
            rng = np.random.default_rng()

        if parameters is None:
            parameters = list(self._constraints.keys())

        scenario = {}
        for param in parameters:
            if param in self._constraints:
                scenario[param] = self._constraints[param].sample(rng)

        return scenario

    def validate_scenario(self, scenario: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        验证场景是否在ODD内

        Args:
            scenario: {参数名: 值}

        Returns:
            (是否有效, 违规参数列表)
        """
        violations = []

        for param, value in scenario.items():
            constraint = self._constraints.get(param)
            if constraint and not constraint.is_valid(value):
                violations.append(
                    f"{param}: {value} 超出范围 [{constraint.min_value}, {constraint.max_value}]"
                )

        return len(violations) == 0, violations

    def clip_scenario(self, scenario: Dict[str, float]) -> Dict[str, float]:
        """
        将场景裁剪到ODD范围内

        Args:
            scenario: 原始场景

        Returns:
            裁剪后的场景
        """
        clipped = {}
        for param, value in scenario.items():
            constraint = self._constraints.get(param)
            if constraint:
                clipped[param] = np.clip(value, constraint.min_value, constraint.max_value)
            else:
                clipped[param] = value
        return clipped

    def get_safe_perturbation_range(self,
                                   current_state: Dict[str, float],
                                   parameter: str,
                                   margin: float = 0.1) -> Tuple[float, float]:
        """
        获取安全扰动范围

        在当前状态基础上，确定某参数可以安全扰动的范围

        Args:
            current_state: 当前状态
            parameter: 参数名
            margin: 安全裕度 (0-1)

        Returns:
            (最小扰动, 最大扰动)
        """
        constraint = self._constraints.get(parameter)
        if not constraint:
            return (0, 0)

        current = current_state.get(parameter, constraint.nominal_value or 0)
        range_val = constraint.max_value - constraint.min_value
        safe_margin = range_val * margin

        min_delta = constraint.min_value + safe_margin - current
        max_delta = constraint.max_value - safe_margin - current

        return (min_delta, max_delta)

    def summary(self) -> str:
        """生成约束摘要"""
        lines = [
            "=" * 60,
            f"世界模型ODD包络 - {self.analyzer.system_name}",
            "=" * 60,
            f"约束参数数量: {len(self._constraints)}",
            "",
            "参数约束列表:"
        ]

        for name, constraint in self._constraints.items():
            nom = f", 标称={constraint.nominal_value:.3f}" if constraint.nominal_value else ""
            lines.append(
                f"  {name}: [{constraint.min_value:.3f}, {constraint.max_value:.3f}]{nom}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)


# ==========================================
# 便捷函数（扩展）
# ==========================================

def build_odd_from_config(
    system_name: str,
    sensors: List[SensorProfile],
    actuators: List[ActuatorProfile],
    channels: Optional[List[ChannelProfile]] = None,
    communication: Optional[CommunicationProfile] = None
) -> ConfiguredODDAnalyzer:
    """
    从配置构建ODD分析器（便捷函数）

    Args:
        system_name: 系统名称
        sensors: 传感器配置列表
        actuators: 执行器配置列表
        channels: 渠道配置列表
        communication: 通信配置

    Returns:
        ConfiguredODDAnalyzer
    """
    builder = ODDProfileBuilder(system_name)

    for sensor in sensors:
        builder.add_sensor(sensor)

    for actuator in actuators:
        builder.add_actuator(actuator)

    if channels:
        for channel in channels:
            builder.add_channel(channel)

    if communication:
        builder.set_communication(communication)

    return builder.create_analyzer()


def create_world_model_envelope(
    analyzer: BaseODDAnalyzer
) -> WorldModelODDEnvelope:
    """
    创建世界模型ODD包络（便捷函数）

    Args:
        analyzer: ODD分析器

    Returns:
        WorldModelODDEnvelope
    """
    return WorldModelODDEnvelope(analyzer)


# ==========================================
# 导出
# ==========================================

__all__ = [
    # 枚举
    'ODDDimension',
    'ODDStatus',
    'AutonomyLevel',
    'ConstraintType',
    'ViolationSeverity',
    # 数据类
    'ODDBoundary',
    'ODDViolation',
    'DimensionScore',
    'ObservabilityMetrics',
    'ControllabilityMetrics',
    'ODDReport',
    # 分析器
    'BaseODDAnalyzer',
    'WaterNetworkODDAnalyzer',
    # 便捷函数
    'calculate_odd_reliability',
    'determine_autonomy_from_score',
    'create_water_odd_analyzer',
    # ODD自动构建器
    'SensorProfile',
    'ActuatorProfile',
    'ChannelProfile',
    'CommunicationProfile',
    'ODDProfileBuilder',
    'ConfiguredODDAnalyzer',
    'build_odd_from_config',
    # 世界模型约束
    'WorldModelConstraint',
    'WorldModelODDEnvelope',
    'create_world_model_envelope'
]
