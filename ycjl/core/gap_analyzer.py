"""
数据完备性诊断器 (Data Gap Analyzer)
====================================

提供工程数据完备性分析的通用框架，支持：
- L5级自主运行数据需求分析
- 缺失数据识别和优先级排序
- 数据质量评估
- 补全建议生成
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from enum import Enum, auto
from datetime import datetime


class DataReadinessLevel(Enum):
    """数据就绪等级"""
    L0_UNAVAILABLE = 0          # 不可用
    L1_MINIMAL = 1              # 最小可用（手动控制）
    L2_PARTIAL = 2              # 部分可用（辅助决策）
    L3_OPERATIONAL = 3          # 可运行（自动控制）
    L4_OPTIMIZED = 4            # 已优化（自适应控制）
    L5_AUTONOMOUS = 5           # 全自主（L5级无人驾驶）


class DataPriority(Enum):
    """数据优先级"""
    CRITICAL = 1                # 关键：L5必需
    HIGH = 2                    # 高：影响安全或效率
    MEDIUM = 3                  # 中：影响精度
    LOW = 4                     # 低：增强功能
    OPTIONAL = 5                # 可选：锦上添花


class DataCategory(Enum):
    """数据类别"""
    GEOMETRY = auto()           # 几何参数
    HYDRAULIC = auto()          # 水力参数
    TRANSIENT = auto()          # 瞬态参数
    CONTROL = auto()            # 控制参数
    CURVE = auto()              # 特性曲线
    OPERATIONAL = auto()        # 运行参数
    SAFETY = auto()             # 安全参数
    ENVIRONMENTAL = auto()      # 环境参数


@dataclass
class MissingDataItem:
    """缺失数据项"""
    name: str                                   # 数据名称
    category: DataCategory                      # 数据类别
    priority: DataPriority                      # 优先级
    description: str                            # 描述
    impact: str                                 # 缺失影响
    suggestion: str                             # 获取建议
    source: str = ""                            # 数据来源
    component: str = ""                         # 相关组件

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "category": self.category.name,
            "priority": self.priority.name,
            "description": self.description,
            "impact": self.impact,
            "suggestion": self.suggestion
        }


@dataclass
class DataGapReport:
    """数据完备性报告"""
    project_name: str                           # 项目名称
    analysis_time: datetime                     # 分析时间
    readiness_level: DataReadinessLevel         # 就绪等级

    # 统计信息
    total_parameters: int = 0                   # 总参数数
    available_count: int = 0                    # 已有数
    missing_count: int = 0                      # 缺失数
    completeness_ratio: float = 0.0             # 完备率

    # 缺失数据列表
    missing_items: List[MissingDataItem] = field(default_factory=list)

    # 按优先级分类
    critical_missing: List[str] = field(default_factory=list)
    high_missing: List[str] = field(default_factory=list)
    medium_missing: List[str] = field(default_factory=list)
    low_missing: List[str] = field(default_factory=list)

    # 按类别分类
    by_category: Dict[str, int] = field(default_factory=dict)

    # 行动建议
    recommendations: List[str] = field(default_factory=list)

    def add_missing_item(self, item: MissingDataItem):
        """添加缺失项"""
        self.missing_items.append(item)

        # 分类统计
        if item.priority == DataPriority.CRITICAL:
            self.critical_missing.append(item.name)
        elif item.priority == DataPriority.HIGH:
            self.high_missing.append(item.name)
        elif item.priority == DataPriority.MEDIUM:
            self.medium_missing.append(item.name)
        else:
            self.low_missing.append(item.name)

        cat_name = item.category.name
        self.by_category[cat_name] = self.by_category.get(cat_name, 0) + 1
        self.missing_count += 1

    def finalize(self):
        """完成报告"""
        if self.total_parameters > 0:
            self.completeness_ratio = self.available_count / self.total_parameters

        # 生成建议
        if self.critical_missing:
            self.recommendations.append(
                f"🚨 需紧急补充 {len(self.critical_missing)} 个关键参数以支持L5运行"
            )
        if self.high_missing:
            self.recommendations.append(
                f"⚠️ 建议优先获取 {len(self.high_missing)} 个高优先级参数"
            )

    def summary(self) -> str:
        """生成摘要"""
        lines = [
            "=" * 60,
            f"数据完备性分析报告 - {self.project_name}",
            "=" * 60,
            f"分析时间: {self.analysis_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"就绪等级: {self.readiness_level.name} ({self.readiness_level.value}/5)",
            "",
            f"📊 完备性统计:",
            f"   总参数数: {self.total_parameters}",
            f"   已有数据: {self.available_count}",
            f"   缺失数据: {self.missing_count}",
            f"   完备率: {self.completeness_ratio:.1%}",
            "",
            f"📋 缺失数据分布:",
            f"   🔴 关键: {len(self.critical_missing)}",
            f"   🟠 高: {len(self.high_missing)}",
            f"   🟡 中: {len(self.medium_missing)}",
            f"   🟢 低/可选: {len(self.low_missing)}",
        ]

        if self.by_category:
            lines.append("\n📁 按类别分布:")
            for cat, count in sorted(self.by_category.items()):
                lines.append(f"   - {cat}: {count}")

        if self.recommendations:
            lines.append("\n💡 建议:")
            for rec in self.recommendations:
                lines.append(f"   {rec}")

        lines.append("=" * 60)
        return "\n".join(lines)


class BaseGapAnalyzer(ABC):
    """
    数据完备性诊断器抽象基类

    提供数据完备性分析的通用框架
    """

    def __init__(self, project_name: str):
        self.project_name = project_name
        self._report: Optional[DataGapReport] = None

        # L5级数据需求定义（子类可扩展）
        self.l5_requirements: Dict[str, Dict] = {
            "geometry": {
                "description": "几何参数",
                "required_for": DataReadinessLevel.L3_OPERATIONAL
            },
            "hydraulic": {
                "description": "水力参数",
                "required_for": DataReadinessLevel.L3_OPERATIONAL
            },
            "transient": {
                "description": "瞬态参数",
                "required_for": DataReadinessLevel.L5_AUTONOMOUS
            },
            "control": {
                "description": "控制参数",
                "required_for": DataReadinessLevel.L4_OPTIMIZED
            },
            "curves": {
                "description": "特性曲线",
                "required_for": DataReadinessLevel.L4_OPTIMIZED
            }
        }

    @abstractmethod
    def analyze(self) -> DataGapReport:
        """
        执行数据完备性分析

        Returns:
            分析报告
        """
        pass

    @abstractmethod
    def check_component(self, component_name: str,
                        component_data: Dict) -> List[MissingDataItem]:
        """
        检查单个组件的数据完备性

        Args:
            component_name: 组件名称
            component_data: 组件数据

        Returns:
            缺失数据项列表
        """
        pass

    def determine_readiness_level(self, report: DataGapReport) -> DataReadinessLevel:
        """
        根据缺失数据确定就绪等级

        Args:
            report: 数据完备性报告

        Returns:
            就绪等级
        """
        if report.critical_missing:
            return DataReadinessLevel.L1_MINIMAL

        if report.completeness_ratio < 0.5:
            return DataReadinessLevel.L2_PARTIAL

        if report.high_missing:
            return DataReadinessLevel.L3_OPERATIONAL

        if report.completeness_ratio < 0.9:
            return DataReadinessLevel.L4_OPTIMIZED

        return DataReadinessLevel.L5_AUTONOMOUS

    def get_report(self) -> Optional[DataGapReport]:
        """获取最近的分析报告"""
        return self._report

    def print_report(self):
        """打印分析报告"""
        if self._report:
            print(self._report.summary())
        else:
            print("尚未执行分析，请先调用 analyze() 方法")

    def get_priority_items(self, priority: DataPriority) -> List[MissingDataItem]:
        """获取指定优先级的缺失项"""
        if not self._report:
            return []
        return [item for item in self._report.missing_items
                if item.priority == priority]

    def get_category_items(self, category: DataCategory) -> List[MissingDataItem]:
        """获取指定类别的缺失项"""
        if not self._report:
            return []
        return [item for item in self._report.missing_items
                if item.category == category]

    def export_to_dict(self) -> Dict:
        """导出报告为字典"""
        if not self._report:
            return {}

        return {
            "project_name": self._report.project_name,
            "analysis_time": self._report.analysis_time.isoformat(),
            "readiness_level": self._report.readiness_level.name,
            "statistics": {
                "total": self._report.total_parameters,
                "available": self._report.available_count,
                "missing": self._report.missing_count,
                "completeness": self._report.completeness_ratio
            },
            "missing_items": [item.to_dict() for item in self._report.missing_items],
            "recommendations": self._report.recommendations
        }


# ==========================================
# 便捷函数
# ==========================================
def create_missing_item(name: str,
                        category: DataCategory,
                        priority: DataPriority,
                        description: str,
                        impact: str = "",
                        suggestion: str = "",
                        component: str = "") -> MissingDataItem:
    """创建缺失数据项（便捷函数）"""
    return MissingDataItem(
        name=name,
        category=category,
        priority=priority,
        description=description,
        impact=impact or f"缺失 {name} 将影响相关计算",
        suggestion=suggestion or f"请提供 {name} 的数据",
        component=component
    )


__all__ = [
    'DataReadinessLevel',
    'DataPriority',
    'DataCategory',
    'MissingDataItem',
    'DataGapReport',
    'BaseGapAnalyzer',
    'create_missing_item'
]
