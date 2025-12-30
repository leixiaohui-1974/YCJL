"""
引绰济辽工程数据完备性诊断器 (Gap Analyzer) v1.0
===============================================

基于核心框架的BaseGapAnalyzer，针对引绰济辽工程进行定制

功能：
1. L5级数据需求分析
2. 缺失数据识别
3. 数据质量评估
4. 补全优先级建议
"""

from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from ..core.gap_analyzer import (
    BaseGapAnalyzer,
    DataReadinessLevel,
    DataPriority,
    DataCategory,
    MissingDataItem,
    DataGapReport,
    create_missing_item
)

from .config_database import (
    ProjectParams,
    YinChuoProjectConfig,
    CurveDatabase,
    SourceConfig,
    TunnelConfig,
    PoolConfig,
    PipeConfig,
    SurgeConfig,
    ValveConfig,
    UserConfig,
    SafetyConfig
)


class YinChuoGapAnalyzer(BaseGapAnalyzer):
    """
    引绰济辽工程数据完备性诊断器

    分析工程参数的完备性，识别L5级自主运行所需的缺失数据
    """

    def __init__(self):
        super().__init__("引绰济辽工程")
        self.config = ProjectParams

    def analyze(self) -> DataGapReport:
        """执行数据完备性分析"""
        report = DataGapReport(
            project_name=self.project_name,
            analysis_time=datetime.now(),
            readiness_level=DataReadinessLevel.L1_MINIMAL
        )

        # 统计总参数数
        total_params = self._count_total_parameters()
        report.total_parameters = total_params

        # 检查各个子系统
        missing_items = []
        missing_items.extend(self._check_source_hub())
        missing_items.extend(self._check_tunnel_system())
        missing_items.extend(self._check_pipeline_system())
        missing_items.extend(self._check_valve_system())
        missing_items.extend(self._check_safety_system())
        missing_items.extend(self._check_control_parameters())
        missing_items.extend(self._check_transient_parameters())

        # 添加缺失项
        for item in missing_items:
            report.add_missing_item(item)

        # 计算已有数据
        report.available_count = total_params - report.missing_count

        # 确定就绪等级
        report.readiness_level = self.determine_readiness_level(report)

        # 完成报告
        report.finalize()

        self._report = report
        return report

    def check_component(self, component_name: str,
                        component_data: Dict) -> List[MissingDataItem]:
        """检查单个组件的数据完备性"""
        missing = []

        # 检查必要字段
        required_fields = {
            "source": ["NORMAL_LEVEL", "DEAD_LEVEL", "TOTAL_STORAGE"],
            "tunnel": ["TOTAL_LENGTH", "SECTION_WIDTH", "MANNING_N_NORMAL"],
            "pipeline": ["TOTAL_LENGTH", "INNER_DIAMETER", "WAVE_SPEED"],
            "valve": ["INLINE_VALVE_DN", "VALVE_CLOSE_TIMES"],
            "safety": ["PRESSURE_ALARM_HIGH", "PRESSURE_TRIP_HIGH"]
        }

        fields = required_fields.get(component_name, [])
        for field_name in fields:
            if field_name not in component_data or component_data[field_name] is None:
                missing.append(create_missing_item(
                    name=f"{component_name}.{field_name}",
                    category=DataCategory.GEOMETRY,
                    priority=DataPriority.HIGH,
                    description=f"{component_name}组件的{field_name}参数",
                    component=component_name
                ))

        return missing

    def _count_total_parameters(self) -> int:
        """统计总参数数"""
        # 估算各子系统参数数
        counts = {
            "source": 30,       # 水源枢纽
            "tunnel": 20,       # 隧洞系统
            "pool": 15,         # 稳流池
            "pipeline": 25,     # 管道系统
            "surge_tank": 12,   # 调压塔
            "valve": 20,        # 阀门系统
            "end_user": 15,     # 用户配置
            "safety": 15,       # 安全设施
            "control": 20,      # 控制参数
            "curves": 50,       # 特性曲线
            "transient": 30     # 瞬态参数
        }
        return sum(counts.values())

    def _check_source_hub(self) -> List[MissingDataItem]:
        """检查水源枢纽参数"""
        missing = []

        # 检查溢洪道调度曲线
        if not hasattr(CurveDatabase, 'SPILLWAY_DISPATCH_CURVE'):
            missing.append(create_missing_item(
                name="溢洪道调度曲线",
                category=DataCategory.CURVE,
                priority=DataPriority.MEDIUM,
                description="溢洪道闸门开度与泄流量的调度关系曲线",
                impact="无法精确计算溢洪道调度",
                suggestion="从设计文件获取溢洪道调度图",
                component="SourceHub"
            ))

        # 检查水轮机详细特性
        if not hasattr(CurveDatabase, 'TURBINE_DETAILED_HILL_CHART'):
            missing.append(create_missing_item(
                name="水轮机详细Hill图",
                category=DataCategory.CURVE,
                priority=DataPriority.LOW,
                description="水轮机全工况效率等值线图",
                impact="影响发电优化精度",
                suggestion="从机组厂家获取完整Hill图数据",
                component="PowerStation"
            ))

        return missing

    def _check_tunnel_system(self) -> List[MissingDataItem]:
        """检查隧洞系统参数"""
        missing = []

        # 检查冰期糙率实测数据
        if not hasattr(TunnelConfig, 'ICE_ROUGHNESS_MEASURED'):
            missing.append(create_missing_item(
                name="冰期糙率实测数据",
                category=DataCategory.HYDRAULIC,
                priority=DataPriority.HIGH,
                description="冰期实测糙率随时间/温度变化数据",
                impact="影响冰期流量预测精度",
                suggestion="在冰期运行期间进行实测",
                component="Tunnel"
            ))

        # 检查隧洞沿程高程
        if not hasattr(TunnelConfig, 'PROFILE_ELEVATIONS'):
            missing.append(create_missing_item(
                name="隧洞沿程高程剖面",
                category=DataCategory.GEOMETRY,
                priority=DataPriority.MEDIUM,
                description="隧洞沿程高程变化曲线",
                impact="影响空气入侵和负压分析",
                suggestion="从设计纵断面图获取",
                component="Tunnel"
            ))

        return missing

    def _check_pipeline_system(self) -> List[MissingDataItem]:
        """检查管道系统参数"""
        missing = []

        # 检查PCCP管道实测波速
        if PipeConfig.WAVE_SPEED == 1050.0:  # 使用默认值
            missing.append(create_missing_item(
                name="PCCP管道实测波速",
                category=DataCategory.TRANSIENT,
                priority=DataPriority.CRITICAL,
                description="PCCP管道压力波速实测值",
                impact="水锤计算结果可能不准确",
                suggestion="进行管道压力波速测试",
                component="Pipeline"
            ))

        # 检查管道分段高程
        if not hasattr(PipeConfig, 'SEGMENT_ELEVATIONS'):
            missing.append(create_missing_item(
                name="管道分段高程数据",
                category=DataCategory.GEOMETRY,
                priority=DataPriority.HIGH,
                description="PCCP管道沿程高程和关键高点位置",
                impact="无法准确识别负压风险点",
                suggestion="从竣工图获取管道纵断面",
                component="Pipeline"
            ))

        # 检查空气阀位置
        if not hasattr(SafetyConfig, 'AIR_VALVE_LOCATIONS'):
            missing.append(create_missing_item(
                name="空气阀具体位置",
                category=DataCategory.SAFETY,
                priority=DataPriority.HIGH,
                description="232个空气阀的具体桩号和高程",
                impact="影响负压保护和水锤分析",
                suggestion="从设计文件获取空气阀布置图",
                component="Safety"
            ))

        return missing

    def _check_valve_system(self) -> List[MissingDataItem]:
        """检查阀门系统参数"""
        missing = []

        # 检查阀门实测Cv曲线
        if not hasattr(CurveDatabase, 'VALVE_MEASURED_CV_CURVES'):
            missing.append(create_missing_item(
                name="阀门实测Cv曲线",
                category=DataCategory.CURVE,
                priority=DataPriority.MEDIUM,
                description="调流调压阀门的实测流量系数曲线",
                impact="阀门调节计算可能有偏差",
                suggestion="进行阀门特性测试或参考厂家数据",
                component="Valve"
            ))

        # 检查阀门执行器特性
        if not hasattr(ValveConfig, 'ACTUATOR_RESPONSE_TIME'):
            missing.append(create_missing_item(
                name="阀门执行器响应时间",
                category=DataCategory.CONTROL,
                priority=DataPriority.MEDIUM,
                description="阀门执行器的响应时间和速率限制",
                impact="影响闭环控制精度",
                suggestion="从阀门厂家获取执行器参数",
                component="Valve"
            ))

        return missing

    def _check_safety_system(self) -> List[MissingDataItem]:
        """检查安全系统参数"""
        missing = []

        # 检查泄压阀详细特性
        if not hasattr(SafetyConfig, 'RELIEF_VALVE_DETAILED_CURVE'):
            missing.append(create_missing_item(
                name="泄压阀详细特性曲线",
                category=DataCategory.SAFETY,
                priority=DataPriority.HIGH,
                description="泄压阀开启/回座特性和流量曲线",
                impact="影响水锤保护分析",
                suggestion="从阀门厂家获取详细测试数据",
                component="Safety"
            ))

        # 检查联通阀操作规程
        if not hasattr(SafetyConfig, 'INTERCONNECT_VALVE_RULES'):
            missing.append(create_missing_item(
                name="联通阀操作规程",
                category=DataCategory.OPERATIONAL,
                priority=DataPriority.MEDIUM,
                description="联通阀的自动/手动操作条件和时序",
                impact="影响应急工况切换策略",
                suggestion="从运行规程中提取联通阀逻辑",
                component="Safety"
            ))

        return missing

    def _check_control_parameters(self) -> List[MissingDataItem]:
        """检查控制参数"""
        missing = []

        # 检查PID参数整定数据
        if not hasattr(self.config.Control, 'PID_TUNING_RECORDS'):
            missing.append(create_missing_item(
                name="PID参数整定记录",
                category=DataCategory.CONTROL,
                priority=DataPriority.MEDIUM,
                description="现场整定的PID参数及响应测试记录",
                impact="控制器参数可能不是最优",
                suggestion="在调试期间进行参数整定测试",
                component="Control"
            ))

        # 检查MPC模型参数
        if not hasattr(self.config.Control, 'MPC_SYSTEM_MODEL'):
            missing.append(create_missing_item(
                name="MPC系统辨识模型",
                category=DataCategory.CONTROL,
                priority=DataPriority.LOW,
                description="用于MPC的系统降阶模型参数",
                impact="MPC控制器可能不是最优",
                suggestion="进行系统辨识实验获取模型",
                component="Control"
            ))

        return missing

    def _check_transient_parameters(self) -> List[MissingDataItem]:
        """检查瞬态参数"""
        missing = []

        # 检查泵组惯性
        if not hasattr(SourceConfig, 'PUMP_INERTIA_GD2'):
            missing.append(create_missing_item(
                name="泵组转动惯量GD²",
                category=DataCategory.TRANSIENT,
                priority=DataPriority.CRITICAL,
                description="水泵机组的转动惯量数据",
                impact="无法准确计算停泵水锤",
                suggestion="从机组厂家获取GD²数据",
                component="PowerStation"
            ))

        # 检查调压塔阻抗特性
        if not hasattr(SurgeConfig, 'IMPEDANCE_MEASURED_COEFFICIENTS'):
            missing.append(create_missing_item(
                name="调压塔阻抗实测系数",
                category=DataCategory.TRANSIENT,
                priority=DataPriority.HIGH,
                description="调压塔入流/出流阻抗系数的实测值",
                impact="调压塔涌浪计算可能不准确",
                suggestion="进行调压塔阻抗测试",
                component="SurgeTank"
            ))

        # 检查管道壁厚分布
        if not hasattr(PipeConfig, 'WALL_THICKNESS_DISTRIBUTION'):
            missing.append(create_missing_item(
                name="管道壁厚沿程分布",
                category=DataCategory.TRANSIENT,
                priority=DataPriority.MEDIUM,
                description="不同压力等级管段的壁厚变化",
                impact="局部波速计算可能不准确",
                suggestion="从设计文件获取管道规格表",
                component="Pipeline"
            ))

        return missing


# ==========================================
# 模块级实例
# ==========================================
YCJLGapAnalyzer = YinChuoGapAnalyzer()


# ==========================================
# 便捷函数
# ==========================================
def analyze_data_readiness() -> DataGapReport:
    """分析数据完备性"""
    return YCJLGapAnalyzer.analyze()


def print_gap_report():
    """打印数据缺口报告"""
    YCJLGapAnalyzer.analyze()
    YCJLGapAnalyzer.print_report()


def get_critical_missing() -> List[MissingDataItem]:
    """获取关键缺失项"""
    return YCJLGapAnalyzer.get_priority_items(DataPriority.CRITICAL)


# ==========================================
# 导出
# ==========================================
__all__ = [
    'YinChuoGapAnalyzer',
    'YCJLGapAnalyzer',
    'analyze_data_readiness',
    'print_gap_report',
    'get_critical_missing'
]
