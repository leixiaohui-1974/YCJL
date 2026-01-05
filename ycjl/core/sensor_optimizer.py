"""
传感器点位与参数优化模块 (Sensor Placement & Parameter Optimization)
==================================================================

提供水利工程传感器布置的通用优化框架，支持：
- 传感器位置优化（覆盖率、冗余度、可观测性）
- 传感器参数优化（采样率、精度、量程）
- 可观测性分析（状态可观测性评估）
- 成本效益评估（投资回报分析）
- 鲁棒性评估（故障容错能力）

设计原则：
1. 通用性：适用于各类水利工程（调水、水电、灌溉等）
2. 可扩展：支持自定义优化目标和约束
3. 参数驱动：基于工程基本参数自动配置
4. 多目标优化：平衡覆盖率、成本、冗余度等多个目标

版本: 1.0.0
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set, Callable
from enum import Enum, auto
from datetime import datetime
import math


# ==========================================
# 枚举定义
# ==========================================

class SensorType(Enum):
    """传感器类型"""
    # 基础水力参数传感器
    LEVEL = ("水位", "m")                   # 水位传感器
    PRESSURE = ("压力", "m水头")            # 压力传感器
    FLOW = ("流量", "m³/s")                 # 流量传感器
    VELOCITY = ("流速", "m/s")              # 流速传感器
    TEMPERATURE = ("温度", "°C")            # 温度传感器

    # 水质传感器
    TURBIDITY = ("浊度", "NTU")             # 浊度传感器
    PH = ("pH值", "pH")                     # pH传感器
    DISSOLVED_OXYGEN = ("溶解氧", "mg/L")   # 溶解氧传感器
    CONDUCTIVITY = ("电导率", "μS/cm")      # 电导率传感器
    CHLOROPHYLL = ("叶绿素", "μg/L")        # 叶绿素传感器
    AMMONIA = ("氨氮", "mg/L")              # 氨氮传感器
    COD = ("COD", "mg/L")                   # 化学需氧量
    QUALITY = ("水质综合", "-")             # 水质综合传感器

    # 结构安全传感器
    VIBRATION = ("振动", "mm/s")            # 振动传感器
    DISPLACEMENT = ("位移", "mm")           # 位移传感器
    STRAIN = ("应变", "με")                 # 应变传感器
    CRACK = ("裂缝", "mm")                  # 裂缝计
    SETTLEMENT = ("沉降", "mm")             # 沉降仪
    INCLINATION = ("倾斜", "°")             # 倾斜仪
    SEEPAGE = ("渗流", "L/min")             # 渗流量计
    PORE_PRESSURE = ("孔隙水压", "kPa")     # 孔隙水压力计
    STRESS = ("应力", "MPa")                # 应力计

    # 气象与环境传感器
    AIR_TEMPERATURE = ("气温", "°C")        # 气温传感器
    HUMIDITY = ("湿度", "%RH")              # 湿度传感器
    RAINFALL = ("降雨量", "mm")             # 雨量计
    WIND_SPEED = ("风速", "m/s")            # 风速仪
    WIND_DIRECTION = ("风向", "°")          # 风向仪
    EVAPORATION = ("蒸发量", "mm")          # 蒸发计
    SOLAR_RADIATION = ("太阳辐射", "W/m²")  # 太阳辐射计

    # 冰期与寒区传感器
    ICE_THICKNESS = ("冰厚", "m")           # 冰厚传感器
    ICE_COVER = ("冰盖", "%")               # 冰盖覆盖率
    FREEZE_DEPTH = ("冻深", "m")            # 冻结深度

    # 设备状态传感器
    POWER = ("功率", "kW")                  # 功率传感器
    CURRENT = ("电流", "A")                 # 电流传感器
    VOLTAGE = ("电压", "V")                 # 电压传感器
    ROTATION_SPEED = ("转速", "rpm")        # 转速传感器
    BEARING_TEMP = ("轴承温度", "°C")       # 轴承温度
    OIL_PRESSURE = ("油压", "MPa")          # 油压传感器
    VALVE_POSITION = ("阀位", "%")          # 阀门开度
    GATE_POSITION = ("闸位", "m")           # 闸门开度

    def __init__(self, name: str, unit: str):
        self._name = name
        self._unit = unit

    @property
    def display_name(self) -> str:
        return self._name

    @property
    def unit(self) -> str:
        return self._unit

    @classmethod
    def get_category(cls, sensor_type: 'SensorType') -> str:
        """获取传感器类别"""
        hydraulic = {cls.LEVEL, cls.PRESSURE, cls.FLOW, cls.VELOCITY, cls.TEMPERATURE}
        quality = {cls.TURBIDITY, cls.PH, cls.DISSOLVED_OXYGEN, cls.CONDUCTIVITY,
                   cls.CHLOROPHYLL, cls.AMMONIA, cls.COD, cls.QUALITY}
        structural = {cls.VIBRATION, cls.DISPLACEMENT, cls.STRAIN, cls.CRACK,
                      cls.SETTLEMENT, cls.INCLINATION, cls.SEEPAGE, cls.PORE_PRESSURE, cls.STRESS}
        meteorological = {cls.AIR_TEMPERATURE, cls.HUMIDITY, cls.RAINFALL,
                          cls.WIND_SPEED, cls.WIND_DIRECTION, cls.EVAPORATION, cls.SOLAR_RADIATION}
        ice = {cls.ICE_THICKNESS, cls.ICE_COVER, cls.FREEZE_DEPTH}
        equipment = {cls.POWER, cls.CURRENT, cls.VOLTAGE, cls.ROTATION_SPEED,
                     cls.BEARING_TEMP, cls.OIL_PRESSURE, cls.VALVE_POSITION, cls.GATE_POSITION}

        if sensor_type in hydraulic:
            return "水力参数"
        elif sensor_type in quality:
            return "水质监测"
        elif sensor_type in structural:
            return "结构安全"
        elif sensor_type in meteorological:
            return "气象环境"
        elif sensor_type in ice:
            return "冰期寒区"
        elif sensor_type in equipment:
            return "设备状态"
        return "其他"


class MeasurementPriority(Enum):
    """测量优先级"""
    CRITICAL = (1, "关键")                  # 安全关键，必须测量
    HIGH = (2, "高")                        # 运行重要，强烈建议
    MEDIUM = (3, "中")                      # 优化相关，推荐安装
    LOW = (4, "低")                         # 辅助信息，可选
    OPTIONAL = (5, "可选")                  # 额外信息，按需

    def __init__(self, level: int, description: str):
        self._level = level
        self._description = description

    @property
    def level(self) -> int:
        return self._level


class ComponentType(Enum):
    """工程组件类型"""
    RESERVOIR = "水库"                      # 水库
    TUNNEL = "隧洞"                         # 隧洞
    PIPELINE = "管道"                       # 压力管道
    PUMP_STATION = "泵站"                   # 泵站
    POWER_STATION = "电站"                  # 水电站
    VALVE = "阀门"                          # 阀门
    GATE = "闸门"                           # 闸门
    SURGE_TANK = "调压井"                   # 调压井
    POOL = "稳流池"                         # 稳流池/前池
    AQUEDUCT = "渡槽"                       # 渡槽
    SIPHON = "虹吸"                         # 虹吸管
    JUNCTION = "节点"                       # 管道节点
    BIFURCATION = "分叉"                    # 分叉结构


class OptimizationObjective(Enum):
    """优化目标"""
    COVERAGE = "覆盖率最大化"               # 最大化测量覆盖
    OBSERVABILITY = "可观测性最大化"        # 最大化状态可观测性
    REDUNDANCY = "冗余度优化"               # 优化故障冗余
    COST = "成本最小化"                     # 最小化总投资
    ACCURACY = "精度最大化"                 # 最大化测量精度
    ROBUSTNESS = "鲁棒性最大化"             # 最大化系统鲁棒性
    BALANCED = "多目标平衡"                 # 多目标均衡优化


class PlacementStrategy(Enum):
    """布置策略"""
    UNIFORM = "均匀分布"                    # 均匀间隔布置
    CRITICAL_POINTS = "关键点位"            # 关键位置优先
    GRADIENT_BASED = "梯度驱动"             # 基于变化梯度
    OBSERVABILITY_BASED = "可观测性驱动"    # 基于可观测性分析
    HYBRID = "混合策略"                     # 组合多种策略


# ==========================================
# 数据类定义
# ==========================================

@dataclass
class SensorSpec:
    """
    传感器规格定义

    定义传感器的技术参数和成本信息
    """
    sensor_type: SensorType                 # 传感器类型
    name: str = ""                          # 型号名称
    manufacturer: str = ""                  # 制造商

    # 测量参数
    range_min: float = 0.0                  # 量程下限
    range_max: float = 100.0                # 量程上限
    accuracy: float = 0.01                  # 精度 (相对或绝对)
    accuracy_type: str = "absolute"         # "absolute" 或 "relative"
    resolution: float = 0.001               # 分辨率

    # 动态特性
    response_time: float = 1.0              # 响应时间 (s)
    sampling_rate_max: float = 100.0        # 最大采样率 (Hz)
    bandwidth: float = 10.0                 # 带宽 (Hz)

    # 可靠性
    mtbf: float = 50000.0                   # 平均故障间隔 (h)
    operating_temp_min: float = -20.0       # 工作温度下限 (°C)
    operating_temp_max: float = 60.0        # 工作温度上限 (°C)
    ip_rating: str = "IP68"                 # 防护等级

    # 成本
    purchase_cost: float = 5000.0           # 采购成本 (元)
    installation_cost: float = 2000.0       # 安装成本 (元)
    maintenance_cost_annual: float = 500.0  # 年维护成本 (元)
    lifespan_years: float = 10.0            # 预期寿命 (年)

    # 通信
    output_type: str = "4-20mA"             # 输出类型
    communication: str = "HART"             # 通信协议
    power_requirement: float = 24.0         # 供电电压 (V)
    power_consumption: float = 5.0          # 功耗 (W)

    @property
    def total_lifecycle_cost(self) -> float:
        """全生命周期成本"""
        return (self.purchase_cost + self.installation_cost +
                self.maintenance_cost_annual * self.lifespan_years)

    @property
    def annual_cost(self) -> float:
        """年均成本"""
        return self.total_lifecycle_cost / self.lifespan_years

    def is_suitable_for_range(self, min_val: float, max_val: float) -> bool:
        """检查是否适合给定量程"""
        # 允许小裕度，确保边界情况也能通过
        margin = 0.05 * (max_val - min_val) if max_val > min_val else 0
        return (self.range_min <= min_val + margin and
                self.range_max >= max_val - margin)

    def to_dict(self) -> Dict:
        return {
            "type": self.sensor_type.name,
            "name": self.name,
            "range": f"{self.range_min}~{self.range_max} {self.sensor_type.unit}",
            "accuracy": f"±{self.accuracy}{self.sensor_type.unit if self.accuracy_type == 'absolute' else '%'}",
            "response_time": f"{self.response_time}s",
            "cost": f"¥{self.total_lifecycle_cost:.0f}"
        }


@dataclass
class MeasurementPoint:
    """
    测量点位定义

    定义工程中的一个测量位置
    """
    point_id: str                           # 点位ID
    name: str                               # 点位名称
    component_type: ComponentType           # 所属组件类型
    component_id: str                       # 组件ID

    # 位置信息
    chainage: float = 0.0                   # 桩号/里程 (m)
    elevation: float = 0.0                  # 高程 (m)
    position_description: str = ""          # 位置描述

    # 测量需求
    required_measurements: List[SensorType] = field(default_factory=list)
    priority: MeasurementPriority = MeasurementPriority.MEDIUM

    # 物理约束
    expected_value_range: Dict[SensorType, Tuple[float, float]] = field(default_factory=dict)
    expected_variation_rate: Dict[SensorType, float] = field(default_factory=dict)  # 变化率

    # 环境条件
    ambient_temp_range: Tuple[float, float] = (-10.0, 40.0)
    humidity_range: Tuple[float, float] = (20.0, 100.0)
    is_submerged: bool = False              # 是否水下
    is_pressurized: bool = False            # 是否承压
    has_ice_risk: bool = False              # 是否有冰期影响

    # 访问性
    accessibility: float = 1.0              # 可达性 0-1 (1=易访问)
    maintenance_difficulty: float = 0.5     # 维护难度 0-1

    # 安全相关
    is_safety_critical: bool = False        # 是否安全关键点

    def get_priority_weight(self) -> float:
        """获取优先级权重"""
        weights = {
            MeasurementPriority.CRITICAL: 1.0,
            MeasurementPriority.HIGH: 0.8,
            MeasurementPriority.MEDIUM: 0.6,
            MeasurementPriority.LOW: 0.4,
            MeasurementPriority.OPTIONAL: 0.2
        }
        return weights.get(self.priority, 0.5)

    def to_dict(self) -> Dict:
        return {
            "id": self.point_id,
            "name": self.name,
            "component": f"{self.component_type.value}({self.component_id})",
            "chainage": f"{self.chainage:.1f}m",
            "priority": self.priority.name,
            "measurements": [m.display_name for m in self.required_measurements]
        }


@dataclass
class SensorPlacement:
    """
    传感器布置方案

    定义某个测量点位的传感器配置
    """
    placement_id: str                       # 布置ID
    point: MeasurementPoint                 # 测量点位
    sensor_spec: SensorSpec                 # 传感器规格

    # 配置参数
    sampling_rate: float = 1.0              # 采样率 (Hz)
    filtering_enabled: bool = True          # 是否启用滤波
    filter_cutoff: float = 0.5              # 滤波截止频率 (Hz)

    # 冗余配置
    redundancy_count: int = 1               # 冗余数量 (1=无冗余)
    redundancy_type: str = "none"           # "none", "hot", "cold"

    # 数据传输
    transmission_interval: float = 1.0      # 数据传输间隔 (s)
    local_storage_hours: float = 24.0       # 本地存储时长 (h)

    # 告警配置
    alarm_high: Optional[float] = None      # 高限告警
    alarm_low: Optional[float] = None       # 低限告警
    alarm_rate_of_change: Optional[float] = None  # 变化率告警

    # 评分
    coverage_score: float = 1.0             # 覆盖得分 0-1
    observability_score: float = 1.0        # 可观测性得分 0-1
    reliability_score: float = 1.0          # 可靠性得分 0-1

    @property
    def total_cost(self) -> float:
        """布置总成本"""
        return self.sensor_spec.total_lifecycle_cost * self.redundancy_count

    @property
    def annual_cost(self) -> float:
        """年度成本"""
        return self.sensor_spec.annual_cost * self.redundancy_count

    @property
    def effective_mtbf(self) -> float:
        """有效MTBF（考虑冗余）"""
        base_mtbf = self.sensor_spec.mtbf
        if self.redundancy_count > 1 and self.redundancy_type == "hot":
            # 热备冗余提高可靠性
            return base_mtbf * (self.redundancy_count + 1) / 2
        return base_mtbf

    @property
    def overall_score(self) -> float:
        """综合得分"""
        return (self.coverage_score * 0.3 +
                self.observability_score * 0.4 +
                self.reliability_score * 0.3)

    def to_dict(self) -> Dict:
        return {
            "id": self.placement_id,
            "point": self.point.name,
            "sensor": self.sensor_spec.name,
            "type": self.sensor_spec.sensor_type.display_name,
            "sampling_rate": f"{self.sampling_rate}Hz",
            "redundancy": self.redundancy_count,
            "cost": f"¥{self.total_cost:.0f}",
            "score": f"{self.overall_score:.1%}"
        }


@dataclass
class OptimizationConstraint:
    """
    优化约束条件

    定义优化过程中的约束
    """
    name: str                               # 约束名称
    constraint_type: str = "inequality"     # "equality" 或 "inequality"

    # 预算约束
    max_total_cost: Optional[float] = None  # 最大总投资 (元)
    max_annual_cost: Optional[float] = None # 最大年度成本 (元)

    # 数量约束
    max_sensors: Optional[int] = None       # 最大传感器数量
    min_sensors: Optional[int] = None       # 最小传感器数量
    max_per_type: Optional[Dict[SensorType, int]] = None  # 每类型最大数量

    # 性能约束
    min_coverage: float = 0.8               # 最小覆盖率
    min_observability: float = 0.7          # 最小可观测性
    min_redundancy: float = 0.0             # 最小冗余度

    # 可靠性约束
    min_system_availability: float = 0.99   # 最小系统可用性
    max_single_point_failure: int = 0       # 最大单点故障数

    # 响应约束
    max_response_time: float = 5.0          # 最大响应时间 (s)
    min_sampling_rate: float = 0.1          # 最小采样率 (Hz)

    def is_satisfied(self, solution: 'OptimizationSolution') -> Tuple[bool, List[str]]:
        """检查约束是否满足"""
        violations = []

        if self.max_total_cost and solution.total_cost > self.max_total_cost:
            violations.append(f"总成本{solution.total_cost:.0f}超出预算{self.max_total_cost:.0f}")

        if self.max_sensors and solution.sensor_count > self.max_sensors:
            violations.append(f"传感器数量{solution.sensor_count}超出限制{self.max_sensors}")

        if self.min_sensors and solution.sensor_count < self.min_sensors:
            violations.append(f"传感器数量{solution.sensor_count}低于要求{self.min_sensors}")

        if solution.coverage_rate < self.min_coverage:
            violations.append(f"覆盖率{solution.coverage_rate:.1%}低于要求{self.min_coverage:.1%}")

        if solution.observability_score < self.min_observability:
            violations.append(f"可观测性{solution.observability_score:.1%}低于要求{self.min_observability:.1%}")

        return len(violations) == 0, violations


@dataclass
class OptimizationSolution:
    """
    优化结果方案

    包含完整的传感器布置优化结果
    """
    solution_id: str                        # 方案ID
    name: str                               # 方案名称
    timestamp: datetime = field(default_factory=datetime.now)

    # 布置方案
    placements: List[SensorPlacement] = field(default_factory=list)

    # 评估指标
    coverage_rate: float = 0.0              # 覆盖率
    observability_score: float = 0.0        # 可观测性得分
    redundancy_score: float = 0.0           # 冗余度得分
    robustness_score: float = 0.0           # 鲁棒性得分

    # 成本统计
    total_cost: float = 0.0                 # 总投资
    annual_cost: float = 0.0                # 年度成本

    # 统计信息
    sensor_count: int = 0                   # 传感器总数
    sensor_by_type: Dict[SensorType, int] = field(default_factory=dict)
    covered_points: int = 0                 # 覆盖点位数
    total_points: int = 0                   # 总点位数

    # 优化过程
    optimization_iterations: int = 0        # 迭代次数
    optimization_time: float = 0.0          # 优化耗时 (s)
    objective_value: float = 0.0            # 目标函数值

    # 约束满足
    constraints_satisfied: bool = True
    constraint_violations: List[str] = field(default_factory=list)

    def update_statistics(self):
        """更新统计信息"""
        self.sensor_count = sum(p.redundancy_count for p in self.placements)
        self.total_cost = sum(p.total_cost for p in self.placements)
        self.annual_cost = sum(p.annual_cost for p in self.placements)

        # 按类型统计
        self.sensor_by_type = {}
        for p in self.placements:
            st = p.sensor_spec.sensor_type
            self.sensor_by_type[st] = self.sensor_by_type.get(st, 0) + p.redundancy_count

        # 覆盖点位
        self.covered_points = len(set(p.point.point_id for p in self.placements))

    def summary(self) -> str:
        """生成方案摘要"""
        lines = [
            "=" * 70,
            f"传感器布置优化方案 - {self.name}",
            "=" * 70,
            f"方案ID: {self.solution_id}",
            f"生成时间: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "📊 总体评估:",
            f"   覆盖率: {self.coverage_rate:.1%}",
            f"   可观测性: {self.observability_score:.1%}",
            f"   冗余度: {self.redundancy_score:.1%}",
            f"   鲁棒性: {self.robustness_score:.1%}",
            "",
            "💰 成本统计:",
            f"   总投资: ¥{self.total_cost:,.0f}",
            f"   年度成本: ¥{self.annual_cost:,.0f}",
            "",
            "📦 设备统计:",
            f"   传感器总数: {self.sensor_count}",
            f"   覆盖点位: {self.covered_points}/{self.total_points}",
        ]

        if self.sensor_by_type:
            lines.append("   按类型分布:")
            for st, count in sorted(self.sensor_by_type.items(), key=lambda x: -x[1]):
                lines.append(f"      {st.display_name}: {count}")

        if not self.constraints_satisfied:
            lines.extend([
                "",
                "⚠️ 约束违反:"
            ])
            for v in self.constraint_violations:
                lines.append(f"   • {v}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "id": self.solution_id,
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "coverage_rate": self.coverage_rate,
            "observability_score": self.observability_score,
            "redundancy_score": self.redundancy_score,
            "total_cost": self.total_cost,
            "sensor_count": self.sensor_count,
            "placements": [p.to_dict() for p in self.placements]
        }


# ==========================================
# 传感器库
# ==========================================

class SensorCatalog:
    """
    传感器产品目录

    管理可用的传感器规格库
    """

    def __init__(self):
        self._sensors: Dict[str, SensorSpec] = {}
        self._by_type: Dict[SensorType, List[SensorSpec]] = {}
        self._initialize_default_catalog()

    def _initialize_default_catalog(self):
        """初始化默认传感器目录"""
        # 水位传感器
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.LEVEL,
            name="LV-100-P",
            manufacturer="通用仪表",
            range_min=0, range_max=20,
            accuracy=0.01, accuracy_type="absolute",
            response_time=0.5, sampling_rate_max=10,
            mtbf=60000,
            purchase_cost=3000, installation_cost=1500,
            maintenance_cost_annual=300
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.LEVEL,
            name="LV-200-U",
            manufacturer="通用仪表",
            range_min=0, range_max=50,
            accuracy=0.02, accuracy_type="absolute",
            response_time=1.0, sampling_rate_max=5,
            mtbf=50000,
            purchase_cost=5000, installation_cost=2000,
            maintenance_cost_annual=500
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.LEVEL,
            name="LV-300-R",
            manufacturer="精密仪器",
            range_min=0, range_max=100,
            accuracy=0.005, accuracy_type="absolute",
            response_time=0.2, sampling_rate_max=20,
            mtbf=80000,
            purchase_cost=15000, installation_cost=3000,
            maintenance_cost_annual=800
        ))

        # 压力传感器
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.PRESSURE,
            name="PT-100",
            manufacturer="通用仪表",
            range_min=0, range_max=100,
            accuracy=0.5, accuracy_type="absolute",
            response_time=0.01, sampling_rate_max=100,
            mtbf=70000,
            purchase_cost=4000, installation_cost=1000,
            maintenance_cost_annual=400
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.PRESSURE,
            name="PT-200",
            manufacturer="精密仪器",
            range_min=0, range_max=200,
            accuracy=0.25, accuracy_type="absolute",
            response_time=0.005, sampling_rate_max=200,
            mtbf=80000,
            purchase_cost=8000, installation_cost=1500,
            maintenance_cost_annual=600
        ))

        # 流量传感器
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.FLOW,
            name="FM-100-U",
            manufacturer="流量科技",
            range_min=0, range_max=30,
            accuracy=0.5, accuracy_type="relative",
            response_time=1.0, sampling_rate_max=10,
            mtbf=50000,
            purchase_cost=20000, installation_cost=5000,
            maintenance_cost_annual=2000
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.FLOW,
            name="FM-200-E",
            manufacturer="流量科技",
            range_min=0, range_max=50,
            accuracy=0.3, accuracy_type="relative",
            response_time=0.5, sampling_rate_max=20,
            mtbf=60000,
            purchase_cost=35000, installation_cost=8000,
            maintenance_cost_annual=3000
        ))

        # 温度传感器
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.TEMPERATURE,
            name="TT-100",
            manufacturer="温控仪表",
            range_min=-30, range_max=50,
            accuracy=0.1, accuracy_type="absolute",
            response_time=5.0, sampling_rate_max=1,
            mtbf=100000,
            purchase_cost=500, installation_cost=200,
            maintenance_cost_annual=50
        ))

        # 流速传感器
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.VELOCITY,
            name="VT-100",
            manufacturer="流量科技",
            range_min=0, range_max=10,
            accuracy=0.02, accuracy_type="absolute",
            response_time=0.5, sampling_rate_max=20,
            mtbf=40000,
            purchase_cost=8000, installation_cost=2000,
            maintenance_cost_annual=800
        ))

        # 冰厚传感器
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.ICE_THICKNESS,
            name="ICE-100",
            manufacturer="寒区仪器",
            range_min=0, range_max=2,
            accuracy=0.01, accuracy_type="absolute",
            response_time=10.0, sampling_rate_max=0.1,
            mtbf=30000,
            operating_temp_min=-40, operating_temp_max=30,
            purchase_cost=25000, installation_cost=10000,
            maintenance_cost_annual=3000
        ))

        # ==========================================
        # 扩展传感器类型 (v1.1)
        # ==========================================

        # 水质传感器
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.TURBIDITY,
            name="TUR-100",
            manufacturer="水质仪器",
            range_min=0, range_max=1000,
            accuracy=2, accuracy_type="relative",
            response_time=2.0, sampling_rate_max=1,
            mtbf=40000,
            purchase_cost=8000, installation_cost=2000,
            maintenance_cost_annual=1500
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.PH,
            name="PH-100",
            manufacturer="水质仪器",
            range_min=0, range_max=14,
            accuracy=0.02, accuracy_type="absolute",
            response_time=5.0, sampling_rate_max=0.5,
            mtbf=30000,
            purchase_cost=3000, installation_cost=1000,
            maintenance_cost_annual=800
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.DISSOLVED_OXYGEN,
            name="DO-100",
            manufacturer="水质仪器",
            range_min=0, range_max=20,
            accuracy=0.1, accuracy_type="absolute",
            response_time=30.0, sampling_rate_max=0.1,
            mtbf=25000,
            purchase_cost=5000, installation_cost=1500,
            maintenance_cost_annual=1200
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.CONDUCTIVITY,
            name="EC-100",
            manufacturer="水质仪器",
            range_min=0, range_max=5000,
            accuracy=1, accuracy_type="relative",
            response_time=2.0, sampling_rate_max=1,
            mtbf=50000,
            purchase_cost=2500, installation_cost=800,
            maintenance_cost_annual=500
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.AMMONIA,
            name="NH3-100",
            manufacturer="水质仪器",
            range_min=0, range_max=50,
            accuracy=5, accuracy_type="relative",
            response_time=60.0, sampling_rate_max=0.05,
            mtbf=20000,
            purchase_cost=15000, installation_cost=3000,
            maintenance_cost_annual=3000
        ))

        # 结构安全传感器
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.VIBRATION,
            name="VIB-100",
            manufacturer="振动监测",
            range_min=0, range_max=100,
            accuracy=0.1, accuracy_type="absolute",
            response_time=0.001, sampling_rate_max=5000,
            mtbf=80000,
            purchase_cost=6000, installation_cost=1500,
            maintenance_cost_annual=600
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.DISPLACEMENT,
            name="DIS-100",
            manufacturer="位移监测",
            range_min=-50, range_max=50,
            accuracy=0.01, accuracy_type="absolute",
            response_time=0.1, sampling_rate_max=100,
            mtbf=100000,
            purchase_cost=4000, installation_cost=1000,
            maintenance_cost_annual=400
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.STRAIN,
            name="STR-100",
            manufacturer="应变监测",
            range_min=-3000, range_max=3000,
            accuracy=1, accuracy_type="absolute",
            response_time=0.01, sampling_rate_max=1000,
            mtbf=150000,
            purchase_cost=800, installation_cost=500,
            maintenance_cost_annual=100
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.CRACK,
            name="CRK-100",
            manufacturer="裂缝监测",
            range_min=0, range_max=30,
            accuracy=0.01, accuracy_type="absolute",
            response_time=1.0, sampling_rate_max=10,
            mtbf=100000,
            purchase_cost=2000, installation_cost=800,
            maintenance_cost_annual=200
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.SETTLEMENT,
            name="SET-100",
            manufacturer="沉降监测",
            range_min=-500, range_max=500,
            accuracy=0.1, accuracy_type="absolute",
            response_time=10.0, sampling_rate_max=0.1,
            mtbf=80000,
            purchase_cost=5000, installation_cost=2000,
            maintenance_cost_annual=500
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.INCLINATION,
            name="INC-100",
            manufacturer="倾斜监测",
            range_min=-30, range_max=30,
            accuracy=0.001, accuracy_type="absolute",
            response_time=1.0, sampling_rate_max=10,
            mtbf=100000,
            purchase_cost=8000, installation_cost=2500,
            maintenance_cost_annual=800
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.SEEPAGE,
            name="SEE-100",
            manufacturer="渗流监测",
            range_min=0, range_max=100,
            accuracy=1, accuracy_type="relative",
            response_time=5.0, sampling_rate_max=1,
            mtbf=50000,
            purchase_cost=3000, installation_cost=1500,
            maintenance_cost_annual=500
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.PORE_PRESSURE,
            name="POR-100",
            manufacturer="渗压监测",
            range_min=0, range_max=1000,
            accuracy=0.5, accuracy_type="relative",
            response_time=1.0, sampling_rate_max=10,
            mtbf=80000,
            purchase_cost=2500, installation_cost=1000,
            maintenance_cost_annual=300
        ))

        # 气象传感器
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.AIR_TEMPERATURE,
            name="AT-100",
            manufacturer="气象仪器",
            range_min=-50, range_max=60,
            accuracy=0.2, accuracy_type="absolute",
            response_time=10.0, sampling_rate_max=0.5,
            mtbf=80000,
            purchase_cost=500, installation_cost=300,
            maintenance_cost_annual=100
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.HUMIDITY,
            name="HUM-100",
            manufacturer="气象仪器",
            range_min=0, range_max=100,
            accuracy=2, accuracy_type="absolute",
            response_time=10.0, sampling_rate_max=0.5,
            mtbf=60000,
            purchase_cost=400, installation_cost=200,
            maintenance_cost_annual=80
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.RAINFALL,
            name="RG-100",
            manufacturer="气象仪器",
            range_min=0, range_max=500,
            accuracy=0.2, accuracy_type="absolute",
            response_time=60.0, sampling_rate_max=0.1,
            mtbf=50000,
            purchase_cost=3000, installation_cost=1500,
            maintenance_cost_annual=500
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.WIND_SPEED,
            name="WS-100",
            manufacturer="气象仪器",
            range_min=0, range_max=60,
            accuracy=0.3, accuracy_type="absolute",
            response_time=1.0, sampling_rate_max=10,
            mtbf=40000,
            purchase_cost=2000, installation_cost=1000,
            maintenance_cost_annual=400
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.EVAPORATION,
            name="EVP-100",
            manufacturer="气象仪器",
            range_min=0, range_max=100,
            accuracy=0.1, accuracy_type="absolute",
            response_time=3600.0, sampling_rate_max=0.001,
            mtbf=60000,
            purchase_cost=5000, installation_cost=2000,
            maintenance_cost_annual=800
        ))

        # 冰期扩展传感器
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.ICE_COVER,
            name="ICOV-100",
            manufacturer="寒区仪器",
            range_min=0, range_max=100,
            accuracy=5, accuracy_type="absolute",
            response_time=60.0, sampling_rate_max=0.05,
            mtbf=25000,
            operating_temp_min=-45, operating_temp_max=25,
            purchase_cost=20000, installation_cost=8000,
            maintenance_cost_annual=4000
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.FREEZE_DEPTH,
            name="FRZ-100",
            manufacturer="寒区仪器",
            range_min=0, range_max=5,
            accuracy=0.05, accuracy_type="absolute",
            response_time=3600.0, sampling_rate_max=0.001,
            mtbf=50000,
            operating_temp_min=-50, operating_temp_max=30,
            purchase_cost=8000, installation_cost=5000,
            maintenance_cost_annual=1000
        ))

        # 设备状态传感器
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.POWER,
            name="PWR-100",
            manufacturer="电力仪表",
            range_min=0, range_max=10000,
            accuracy=0.5, accuracy_type="relative",
            response_time=0.1, sampling_rate_max=100,
            mtbf=100000,
            purchase_cost=2000, installation_cost=500,
            maintenance_cost_annual=200
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.CURRENT,
            name="CUR-100",
            manufacturer="电力仪表",
            range_min=0, range_max=1000,
            accuracy=0.2, accuracy_type="relative",
            response_time=0.05, sampling_rate_max=200,
            mtbf=120000,
            purchase_cost=800, installation_cost=300,
            maintenance_cost_annual=80
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.ROTATION_SPEED,
            name="RPM-100",
            manufacturer="转速监测",
            range_min=0, range_max=3000,
            accuracy=0.1, accuracy_type="relative",
            response_time=0.1, sampling_rate_max=100,
            mtbf=80000,
            purchase_cost=1500, installation_cost=500,
            maintenance_cost_annual=150
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.BEARING_TEMP,
            name="BT-100",
            manufacturer="温度监测",
            range_min=0, range_max=150,
            accuracy=0.5, accuracy_type="absolute",
            response_time=2.0, sampling_rate_max=5,
            mtbf=100000,
            purchase_cost=600, installation_cost=200,
            maintenance_cost_annual=60
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.OIL_PRESSURE,
            name="OP-100",
            manufacturer="油压监测",
            range_min=0, range_max=2,
            accuracy=0.01, accuracy_type="absolute",
            response_time=0.1, sampling_rate_max=50,
            mtbf=90000,
            purchase_cost=1200, installation_cost=400,
            maintenance_cost_annual=120
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.VALVE_POSITION,
            name="VP-100",
            manufacturer="阀门监测",
            range_min=0, range_max=100,
            accuracy=0.5, accuracy_type="absolute",
            response_time=0.2, sampling_rate_max=20,
            mtbf=100000,
            purchase_cost=1000, installation_cost=300,
            maintenance_cost_annual=100
        ))
        self.add_sensor(SensorSpec(
            sensor_type=SensorType.GATE_POSITION,
            name="GP-100",
            manufacturer="闸门监测",
            range_min=0, range_max=20,
            accuracy=0.01, accuracy_type="absolute",
            response_time=0.5, sampling_rate_max=10,
            mtbf=80000,
            purchase_cost=3000, installation_cost=1000,
            maintenance_cost_annual=300
        ))

    def add_sensor(self, spec: SensorSpec):
        """添加传感器规格"""
        self._sensors[spec.name] = spec
        if spec.sensor_type not in self._by_type:
            self._by_type[spec.sensor_type] = []
        self._by_type[spec.sensor_type].append(spec)

    def get_sensor(self, name: str) -> Optional[SensorSpec]:
        """获取传感器规格"""
        return self._sensors.get(name)

    def get_by_type(self, sensor_type: SensorType) -> List[SensorSpec]:
        """获取某类型的所有传感器"""
        return self._by_type.get(sensor_type, [])

    def find_suitable(self, sensor_type: SensorType,
                      value_range: Tuple[float, float],
                      max_cost: Optional[float] = None) -> List[SensorSpec]:
        """查找适合的传感器"""
        candidates = self.get_by_type(sensor_type)
        suitable = []
        for spec in candidates:
            if spec.is_suitable_for_range(value_range[0], value_range[1]):
                if max_cost is None or spec.total_lifecycle_cost <= max_cost:
                    suitable.append(spec)
        return sorted(suitable, key=lambda x: x.accuracy)

    def list_all(self) -> List[SensorSpec]:
        """列出所有传感器"""
        return list(self._sensors.values())


# ==========================================
# 可观测性分析器
# ==========================================

class ObservabilityAnalyzer:
    """
    系统可观测性分析器

    评估传感器配置对系统状态的可观测能力
    """

    def __init__(self):
        # 状态变量权重
        self.state_weights: Dict[str, float] = {
            "level": 1.0,           # 水位
            "pressure": 0.9,        # 压力
            "flow": 0.95,           # 流量
            "velocity": 0.7,        # 流速
            "temperature": 0.5,     # 温度
        }

        # 关键状态识别
        self.critical_states: Set[str] = {
            "reservoir_level",
            "pipeline_pressure",
            "main_flow",
            "surge_tank_level"
        }

    def analyze_coverage(self,
                         measurement_points: List[MeasurementPoint],
                         placements: List[SensorPlacement]) -> Dict:
        """
        分析测量覆盖率

        Args:
            measurement_points: 所有测量点位
            placements: 当前布置方案

        Returns:
            覆盖率分析结果
        """
        # 统计各类型测量需求
        required_by_type: Dict[SensorType, int] = {}
        for point in measurement_points:
            for st in point.required_measurements:
                required_by_type[st] = required_by_type.get(st, 0) + 1

        # 统计已覆盖
        covered_by_type: Dict[SensorType, int] = {}
        covered_points: Set[str] = set()
        for p in placements:
            st = p.sensor_spec.sensor_type
            covered_by_type[st] = covered_by_type.get(st, 0) + 1
            covered_points.add(p.point.point_id)

        # 计算覆盖率
        coverage_by_type = {}
        for st, required in required_by_type.items():
            covered = covered_by_type.get(st, 0)
            coverage_by_type[st] = min(1.0, covered / required) if required > 0 else 1.0

        # 综合覆盖率（加权）
        total_weight = sum(self.state_weights.get(st.name.lower(), 0.5)
                          for st in required_by_type.keys())
        weighted_coverage = sum(
            coverage_by_type[st] * self.state_weights.get(st.name.lower(), 0.5)
            for st in coverage_by_type.keys()
        ) / total_weight if total_weight > 0 else 0

        # 关键点位覆盖
        critical_points = [p for p in measurement_points if p.is_safety_critical]
        critical_covered = sum(1 for p in critical_points if p.point_id in covered_points)
        critical_coverage = critical_covered / len(critical_points) if critical_points else 1.0

        return {
            "overall_coverage": weighted_coverage,
            "critical_coverage": critical_coverage,
            "coverage_by_type": {st.name: rate for st, rate in coverage_by_type.items()},
            "total_points": len(measurement_points),
            "covered_points": len(covered_points),
            "critical_points": len(critical_points),
            "critical_covered": critical_covered
        }

    def analyze_observability_matrix(self,
                                     system_states: List[str],
                                     measurements: List[str],
                                     sensitivity_matrix: Optional[np.ndarray] = None) -> Dict:
        """
        分析可观测性矩阵

        Args:
            system_states: 系统状态变量列表
            measurements: 测量变量列表
            sensitivity_matrix: 灵敏度矩阵 (可选)

        Returns:
            可观测性分析结果
        """
        n_states = len(system_states)
        n_measurements = len(measurements)

        if sensitivity_matrix is None:
            # 生成默认灵敏度矩阵（假设直接测量）
            sensitivity_matrix = np.eye(n_states, n_measurements)

        # 计算可观测性矩阵秩
        rank = np.linalg.matrix_rank(sensitivity_matrix)
        observability_index = rank / n_states if n_states > 0 else 0

        # 识别不可观测状态
        u, s, vh = np.linalg.svd(sensitivity_matrix)
        threshold = 1e-10
        unobservable_count = sum(1 for sv in s if sv < threshold)

        # 计算条件数（数值稳定性）
        if len(s) > 0 and s[-1] > threshold:
            condition_number = s[0] / s[-1]
        else:
            condition_number = float('inf')

        return {
            "rank": rank,
            "n_states": n_states,
            "n_measurements": n_measurements,
            "observability_index": observability_index,
            "unobservable_states": unobservable_count,
            "condition_number": condition_number,
            "is_fully_observable": rank >= n_states,
            "singular_values": s.tolist() if isinstance(s, np.ndarray) else list(s)
        }

    def calculate_information_gain(self,
                                   current_placements: List[SensorPlacement],
                                   new_placement: SensorPlacement) -> float:
        """
        计算新增传感器的信息增益

        Args:
            current_placements: 当前布置
            new_placement: 新增布置

        Returns:
            信息增益值 0-1
        """
        # 检查是否已有相同类型传感器在附近
        new_type = new_placement.sensor_spec.sensor_type
        new_chainage = new_placement.point.chainage

        nearby_same_type = []
        for p in current_placements:
            if p.sensor_spec.sensor_type == new_type:
                distance = abs(p.point.chainage - new_chainage)
                if distance < 100:  # 100m范围内
                    nearby_same_type.append((p, distance))

        if not nearby_same_type:
            # 无重复，信息增益最大
            return 1.0

        # 根据距离计算增益衰减
        min_distance = min(d for _, d in nearby_same_type)
        distance_factor = min(1.0, min_distance / 100)

        # 根据精度差异计算增益
        min_nearby_accuracy = min(p.sensor_spec.accuracy for p, _ in nearby_same_type)
        new_accuracy = new_placement.sensor_spec.accuracy
        accuracy_factor = 1.0 if new_accuracy < min_nearby_accuracy else 0.5

        return distance_factor * accuracy_factor * new_placement.point.get_priority_weight()


# ==========================================
# 成本效益分析器
# ==========================================

class CostBenefitAnalyzer:
    """
    成本效益分析器

    评估传感器配置的投资回报
    """

    def __init__(self):
        # 效益计算参数
        self.benefit_factors = {
            "safety_improvement": 10000,      # 安全性提升价值 (元/%)
            "efficiency_gain": 5000,          # 效率提升价值 (元/%)
            "maintenance_saving": 2000,       # 维护节省价值 (元/%)
            "downtime_reduction": 50000,      # 停机减少价值 (元/h)
        }

        # 折现率
        self.discount_rate = 0.08

    def calculate_roi(self, solution: OptimizationSolution,
                      project_lifetime: float = 20.0) -> Dict:
        """
        计算投资回报率

        Args:
            solution: 优化方案
            project_lifetime: 项目寿命 (年)

        Returns:
            ROI分析结果
        """
        # 初始投资
        initial_investment = sum(
            p.sensor_spec.purchase_cost + p.sensor_spec.installation_cost
            for p in solution.placements
        ) * 1.0  # 可加系数调整

        # 年运营成本
        annual_operating_cost = sum(
            p.sensor_spec.maintenance_cost_annual +
            p.sensor_spec.power_consumption * 8760 * 0.5 / 1000  # 电费
            for p in solution.placements
        )

        # 年效益估算
        annual_benefit = (
            solution.coverage_rate * self.benefit_factors["safety_improvement"] +
            solution.observability_score * self.benefit_factors["efficiency_gain"] +
            solution.redundancy_score * self.benefit_factors["maintenance_saving"]
        )

        # 净现值计算
        npv = -initial_investment
        for year in range(1, int(project_lifetime) + 1):
            net_cash_flow = annual_benefit - annual_operating_cost
            npv += net_cash_flow / ((1 + self.discount_rate) ** year)

        # 内部收益率估算
        irr = self._estimate_irr(initial_investment, annual_benefit - annual_operating_cost,
                                  project_lifetime)

        # 投资回收期
        if annual_benefit > annual_operating_cost:
            payback_period = initial_investment / (annual_benefit - annual_operating_cost)
        else:
            payback_period = float('inf')

        return {
            "initial_investment": initial_investment,
            "annual_operating_cost": annual_operating_cost,
            "annual_benefit": annual_benefit,
            "net_present_value": npv,
            "internal_rate_of_return": irr,
            "payback_period": payback_period,
            "benefit_cost_ratio": annual_benefit / (annual_operating_cost +
                                                    initial_investment / project_lifetime)
        }

    def _estimate_irr(self, investment: float, annual_cash_flow: float,
                      years: float) -> float:
        """估算内部收益率"""
        if annual_cash_flow <= 0:
            return 0.0

        # 简化IRR计算
        for rate in np.arange(0.01, 0.5, 0.01):
            npv = -investment
            for y in range(1, int(years) + 1):
                npv += annual_cash_flow / ((1 + rate) ** y)
            if npv < 0:
                return rate - 0.01
        return 0.5

    def compare_solutions(self, solutions: List[OptimizationSolution]) -> Dict:
        """
        比较多个方案的成本效益

        Args:
            solutions: 方案列表

        Returns:
            比较结果
        """
        comparisons = []
        for sol in solutions:
            roi = self.calculate_roi(sol)
            comparisons.append({
                "solution_id": sol.solution_id,
                "name": sol.name,
                "investment": roi["initial_investment"],
                "npv": roi["net_present_value"],
                "irr": roi["internal_rate_of_return"],
                "payback": roi["payback_period"],
                "coverage": sol.coverage_rate,
                "observability": sol.observability_score
            })

        # 排序（按NPV）
        comparisons.sort(key=lambda x: x["npv"], reverse=True)

        return {
            "comparisons": comparisons,
            "best_npv": comparisons[0] if comparisons else None,
            "lowest_cost": min(comparisons, key=lambda x: x["investment"]) if comparisons else None
        }


# ==========================================
# 鲁棒性分析器
# ==========================================

class RobustnessAnalyzer:
    """
    系统鲁棒性分析器

    评估传感器配置的故障容错能力
    """

    def __init__(self):
        self.failure_modes = ["stuck", "drift", "noise", "communication", "power"]

    def analyze_redundancy(self, placements: List[SensorPlacement]) -> Dict:
        """
        分析冗余配置

        Args:
            placements: 布置方案

        Returns:
            冗余分析结果
        """
        # 按类型和位置分组
        by_type_location: Dict[Tuple[SensorType, str], List[SensorPlacement]] = {}
        for p in placements:
            key = (p.sensor_spec.sensor_type, p.point.component_id)
            if key not in by_type_location:
                by_type_location[key] = []
            by_type_location[key].append(p)

        # 统计冗余度
        redundancy_counts = []
        single_points = []
        for (st, loc), ps in by_type_location.items():
            total_redundancy = sum(p.redundancy_count for p in ps)
            redundancy_counts.append(total_redundancy)
            if total_redundancy == 1:
                single_points.append(f"{st.display_name}@{loc}")

        avg_redundancy = np.mean(redundancy_counts) if redundancy_counts else 0
        min_redundancy = min(redundancy_counts) if redundancy_counts else 0

        return {
            "average_redundancy": avg_redundancy,
            "minimum_redundancy": min_redundancy,
            "single_point_failures": single_points,
            "single_point_count": len(single_points),
            "redundancy_score": min(1.0, (avg_redundancy - 1) / 2)  # 归一化
        }

    def analyze_availability(self, placements: List[SensorPlacement]) -> Dict:
        """
        分析系统可用性

        Args:
            placements: 布置方案

        Returns:
            可用性分析结果
        """
        if not placements:
            return {"system_availability": 0, "critical_availability": 0}

        # 单传感器可用性
        availabilities = []
        critical_availabilities = []

        for p in placements:
            # 基于MTBF和修复时间计算
            mttr = 8.0  # 平均修复时间 (h)
            mtbf = p.effective_mtbf
            single_availability = mtbf / (mtbf + mttr)

            # 考虑冗余
            if p.redundancy_count > 1 and p.redundancy_type == "hot":
                # 并联系统可用性
                unavailability = (1 - single_availability) ** p.redundancy_count
                availability = 1 - unavailability
            else:
                availability = single_availability

            availabilities.append(availability)
            if p.point.is_safety_critical:
                critical_availabilities.append(availability)

        # 系统可用性（串联模型）
        system_availability = np.prod(availabilities)
        critical_availability = np.prod(critical_availabilities) if critical_availabilities else 1.0

        return {
            "system_availability": system_availability,
            "critical_availability": critical_availability,
            "average_sensor_availability": np.mean(availabilities),
            "min_sensor_availability": np.min(availabilities),
            "availability_score": system_availability
        }

    def simulate_failures(self, placements: List[SensorPlacement],
                          n_simulations: int = 1000) -> Dict:
        """
        蒙特卡洛故障模拟

        Args:
            placements: 布置方案
            n_simulations: 模拟次数

        Returns:
            故障模拟结果
        """
        if not placements:
            return {"failure_rate": 1.0, "critical_failure_rate": 1.0}

        system_failures = 0
        critical_failures = 0

        for _ in range(n_simulations):
            # 模拟每个传感器的故障状态
            failed_types = set()
            critical_failed = False

            for p in placements:
                # 基于MTBF计算年故障概率
                annual_failure_prob = 1 - np.exp(-8760 / p.effective_mtbf)

                # 所有冗余都失效才算失效
                all_failed = all(
                    np.random.random() < annual_failure_prob
                    for _ in range(p.redundancy_count)
                )

                if all_failed:
                    failed_types.add(p.sensor_spec.sensor_type)
                    if p.point.is_safety_critical:
                        critical_failed = True

            # 判断系统失效（关键测量类型丢失）
            critical_types = {SensorType.LEVEL, SensorType.PRESSURE, SensorType.FLOW}
            if failed_types & critical_types:
                system_failures += 1
            if critical_failed:
                critical_failures += 1

        return {
            "system_failure_rate": system_failures / n_simulations,
            "critical_failure_rate": critical_failures / n_simulations,
            "robustness_score": 1 - system_failures / n_simulations
        }


# ==========================================
# 优化器基类
# ==========================================

class BaseSensorOptimizer(ABC):
    """
    传感器优化器抽象基类

    定义传感器布置优化的通用框架
    """

    def __init__(self, project_name: str):
        self.project_name = project_name

        # 组件
        self.sensor_catalog = SensorCatalog()
        self.observability_analyzer = ObservabilityAnalyzer()
        self.cost_analyzer = CostBenefitAnalyzer()
        self.robustness_analyzer = RobustnessAnalyzer()

        # 测量点位和约束
        self._measurement_points: List[MeasurementPoint] = []
        self._constraints: OptimizationConstraint = OptimizationConstraint(name="default")

        # 优化配置
        self._objective: OptimizationObjective = OptimizationObjective.BALANCED
        self._strategy: PlacementStrategy = PlacementStrategy.HYBRID

        # 权重配置
        self.objective_weights = {
            OptimizationObjective.COVERAGE: {"coverage": 1.0},
            OptimizationObjective.OBSERVABILITY: {"observability": 1.0},
            OptimizationObjective.COST: {"cost": 1.0},
            OptimizationObjective.REDUNDANCY: {"redundancy": 1.0},
            OptimizationObjective.ROBUSTNESS: {"robustness": 1.0},
            OptimizationObjective.BALANCED: {
                "coverage": 0.25,
                "observability": 0.25,
                "cost": 0.2,
                "redundancy": 0.15,
                "robustness": 0.15
            }
        }

    @abstractmethod
    def _initialize_measurement_points(self):
        """
        初始化测量点位

        子类必须实现此方法，定义具体的测量点位
        """
        pass

    def add_measurement_point(self, point: MeasurementPoint):
        """添加测量点位"""
        self._measurement_points.append(point)

    def set_constraints(self, constraints: OptimizationConstraint):
        """设置约束条件"""
        self._constraints = constraints

    def set_objective(self, objective: OptimizationObjective):
        """设置优化目标"""
        self._objective = objective

    def set_strategy(self, strategy: PlacementStrategy):
        """设置布置策略"""
        self._strategy = strategy

    def _select_sensor_for_point(self, point: MeasurementPoint,
                                  sensor_type: SensorType) -> Optional[SensorSpec]:
        """为测量点选择合适的传感器"""
        # 获取预期值范围
        value_range = point.expected_value_range.get(sensor_type, (0, 100))

        # 查找候选传感器
        candidates = self.sensor_catalog.find_suitable(
            sensor_type, value_range,
            max_cost=self._constraints.max_total_cost / 10 if self._constraints.max_total_cost else None
        )

        if not candidates:
            # 如果找不到合适的，选择该类型中量程最大的
            candidates = self.sensor_catalog.get_by_type(sensor_type)

        if not candidates:
            return None

        # 根据优化目标选择
        if self._objective == OptimizationObjective.COST:
            return min(candidates, key=lambda x: x.total_lifecycle_cost)
        elif self._objective == OptimizationObjective.ACCURACY:
            return min(candidates, key=lambda x: x.accuracy)
        else:
            # 平衡考虑精度和成本
            scored = [(c, c.accuracy * c.total_lifecycle_cost) for c in candidates]
            return min(scored, key=lambda x: x[1])[0]

    def _determine_redundancy(self, point: MeasurementPoint) -> Tuple[int, str]:
        """确定冗余配置"""
        if point.is_safety_critical:
            return (2, "hot")  # 安全关键点热备
        elif point.priority == MeasurementPriority.CRITICAL:
            return (2, "cold")
        elif point.priority == MeasurementPriority.HIGH:
            return (1, "none")  # 高优先级，考虑冷备
        else:
            return (1, "none")

    def _calculate_scores(self, placements: List[SensorPlacement]) -> Dict[str, float]:
        """计算各项评分"""
        # 覆盖率分析
        coverage_result = self.observability_analyzer.analyze_coverage(
            self._measurement_points, placements
        )

        # 冗余分析
        redundancy_result = self.robustness_analyzer.analyze_redundancy(placements)

        # 可用性分析
        availability_result = self.robustness_analyzer.analyze_availability(placements)

        # 成本计算
        total_cost = sum(p.total_cost for p in placements)
        cost_score = 1.0 - min(1.0, total_cost / (self._constraints.max_total_cost or total_cost * 2))

        return {
            "coverage": coverage_result["overall_coverage"],
            "critical_coverage": coverage_result["critical_coverage"],
            "observability": availability_result["system_availability"],
            "redundancy": redundancy_result["redundancy_score"],
            "robustness": availability_result["availability_score"],
            "cost": cost_score
        }

    def _evaluate_objective(self, scores: Dict[str, float]) -> float:
        """计算目标函数值"""
        weights = self.objective_weights.get(self._objective,
                                             self.objective_weights[OptimizationObjective.BALANCED])

        objective_value = 0.0
        for metric, weight in weights.items():
            if metric == "cost":
                # 成本是要最小化的，所以取补
                objective_value += weight * scores.get(metric, 0.5)
            else:
                objective_value += weight * scores.get(metric, 0)

        return objective_value

    def optimize(self) -> OptimizationSolution:
        """
        执行优化

        Returns:
            优化结果方案
        """
        import time
        start_time = time.time()

        # 初始化测量点
        if not self._measurement_points:
            self._initialize_measurement_points()

        # 生成初始方案
        placements = self._generate_initial_solution()

        # 迭代优化
        best_placements = placements
        best_score = self._evaluate_objective(self._calculate_scores(placements))
        iterations = 0
        max_iterations = 100

        for i in range(max_iterations):
            iterations += 1

            # 尝试改进
            improved_placements = self._try_improve(best_placements)
            improved_scores = self._calculate_scores(improved_placements)
            improved_value = self._evaluate_objective(improved_scores)

            if improved_value > best_score:
                best_placements = improved_placements
                best_score = improved_value
            else:
                # 收敛检查
                if i > 10 and improved_value <= best_score:
                    break

        # 构建解决方案
        final_scores = self._calculate_scores(best_placements)

        solution = OptimizationSolution(
            solution_id=f"OPT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            name=f"{self.project_name}传感器优化方案",
            placements=best_placements,
            coverage_rate=final_scores["coverage"],
            observability_score=final_scores["observability"],
            redundancy_score=final_scores["redundancy"],
            robustness_score=final_scores["robustness"],
            total_points=len(self._measurement_points),
            optimization_iterations=iterations,
            optimization_time=time.time() - start_time,
            objective_value=best_score
        )

        solution.update_statistics()

        # 检查约束
        satisfied, violations = self._constraints.is_satisfied(solution)
        solution.constraints_satisfied = satisfied
        solution.constraint_violations = violations

        return solution

    def _generate_initial_solution(self) -> List[SensorPlacement]:
        """生成初始方案"""
        placements = []
        placement_id = 0

        for point in self._measurement_points:
            for sensor_type in point.required_measurements:
                spec = self._select_sensor_for_point(point, sensor_type)
                if spec:
                    redundancy_count, redundancy_type = self._determine_redundancy(point)

                    placement = SensorPlacement(
                        placement_id=f"P{placement_id:04d}",
                        point=point,
                        sensor_spec=spec,
                        redundancy_count=redundancy_count,
                        redundancy_type=redundancy_type,
                        sampling_rate=min(1.0, spec.sampling_rate_max)
                    )
                    placements.append(placement)
                    placement_id += 1

        return placements

    def _try_improve(self, current: List[SensorPlacement]) -> List[SensorPlacement]:
        """尝试改进当前方案"""
        improved = list(current)

        # 随机选择一个布置进行调整
        if improved and np.random.random() < 0.5:
            idx = np.random.randint(len(improved))
            p = improved[idx]

            # 尝试不同的传感器型号
            alternatives = self.sensor_catalog.get_by_type(p.sensor_spec.sensor_type)
            if len(alternatives) > 1:
                new_spec = np.random.choice([a for a in alternatives if a != p.sensor_spec])
                improved[idx] = SensorPlacement(
                    placement_id=p.placement_id,
                    point=p.point,
                    sensor_spec=new_spec,
                    redundancy_count=p.redundancy_count,
                    redundancy_type=p.redundancy_type,
                    sampling_rate=min(p.sampling_rate, new_spec.sampling_rate_max)
                )

        # 尝试增加/减少冗余
        if improved and np.random.random() < 0.3:
            idx = np.random.randint(len(improved))
            p = improved[idx]

            if p.redundancy_count == 1 and np.random.random() < 0.5:
                # 增加冗余
                improved[idx] = SensorPlacement(
                    placement_id=p.placement_id,
                    point=p.point,
                    sensor_spec=p.sensor_spec,
                    redundancy_count=2,
                    redundancy_type="hot",
                    sampling_rate=p.sampling_rate
                )
            elif p.redundancy_count > 1 and not p.point.is_safety_critical:
                # 减少冗余
                improved[idx] = SensorPlacement(
                    placement_id=p.placement_id,
                    point=p.point,
                    sensor_spec=p.sensor_spec,
                    redundancy_count=1,
                    redundancy_type="none",
                    sampling_rate=p.sampling_rate
                )

        return improved

    def generate_report(self, solution: OptimizationSolution) -> str:
        """
        生成详细报告

        Args:
            solution: 优化方案

        Returns:
            报告文本
        """
        lines = [
            solution.summary(),
            "",
            "详细布置清单:",
            "-" * 70
        ]

        # 按组件分组
        by_component: Dict[str, List[SensorPlacement]] = {}
        for p in solution.placements:
            comp = f"{p.point.component_type.value}({p.point.component_id})"
            if comp not in by_component:
                by_component[comp] = []
            by_component[comp].append(p)

        for comp, ps in by_component.items():
            lines.append(f"\n【{comp}】")
            for p in ps:
                lines.append(
                    f"  {p.placement_id}: {p.sensor_spec.sensor_type.display_name} "
                    f"@ {p.point.name} (桩号{p.point.chainage:.0f}m)"
                )
                lines.append(
                    f"      型号: {p.sensor_spec.name}, "
                    f"冗余: {p.redundancy_count}({p.redundancy_type}), "
                    f"成本: ¥{p.total_cost:.0f}"
                )

        # ROI分析
        roi = self.cost_analyzer.calculate_roi(solution)
        lines.extend([
            "",
            "投资回报分析:",
            "-" * 70,
            f"初始投资: ¥{roi['initial_investment']:,.0f}",
            f"年运营成本: ¥{roi['annual_operating_cost']:,.0f}",
            f"年预期收益: ¥{roi['annual_benefit']:,.0f}",
            f"净现值(NPV): ¥{roi['net_present_value']:,.0f}",
            f"内部收益率(IRR): {roi['internal_rate_of_return']:.1%}",
            f"投资回收期: {roi['payback_period']:.1f}年"
        ])

        return "\n".join(lines)


# ==========================================
# 通用工程优化器
# ==========================================

class WaterProjectSensorOptimizer(BaseSensorOptimizer):
    """
    通用水利工程传感器优化器

    适用于调水、水电、灌溉等各类水利工程
    """

    def __init__(self, project_name: str, project_params: Dict[str, Any]):
        """
        初始化优化器

        Args:
            project_name: 工程名称
            project_params: 工程基本参数字典，包含:
                - components: 组件列表
                - total_length: 总长度 (m)
                - design_flow: 设计流量 (m³/s)
                - design_head: 设计水头 (m)
                - etc.
        """
        super().__init__(project_name)
        self.project_params = project_params

        # 从参数推导测量点
        self._initialize_measurement_points()

    def _initialize_measurement_points(self):
        """基于工程参数初始化测量点位"""
        components = self.project_params.get("components", [])

        point_id = 0
        for comp in components:
            comp_type = ComponentType[comp.get("type", "PIPELINE").upper()]
            comp_id = comp.get("id", f"C{point_id}")

            # 根据组件类型生成测量点
            points = self._generate_points_for_component(comp_type, comp_id, comp, point_id)
            for p in points:
                self.add_measurement_point(p)
                point_id += 1

    def _generate_points_for_component(self, comp_type: ComponentType,
                                        comp_id: str,
                                        comp_params: Dict,
                                        start_id: int) -> List[MeasurementPoint]:
        """为组件生成测量点位"""
        points = []

        if comp_type == ComponentType.RESERVOIR:
            # 水库：水位、温度
            points.append(MeasurementPoint(
                point_id=f"MP{start_id:04d}",
                name=f"{comp_params.get('name', comp_id)}水位",
                component_type=comp_type,
                component_id=comp_id,
                required_measurements=[SensorType.LEVEL, SensorType.TEMPERATURE],
                priority=MeasurementPriority.CRITICAL,
                is_safety_critical=True,
                expected_value_range={
                    SensorType.LEVEL: (
                        comp_params.get("dead_level", 0),
                        comp_params.get("normal_level", 50)
                    )
                }
            ))

        elif comp_type == ComponentType.PIPELINE:
            # 管道：进口压力、出口压力、流量
            length = comp_params.get("length", 1000)

            # 进口
            points.append(MeasurementPoint(
                point_id=f"MP{start_id:04d}",
                name=f"{comp_params.get('name', comp_id)}进口",
                component_type=comp_type,
                component_id=comp_id,
                chainage=comp_params.get("start_chainage", 0),
                required_measurements=[SensorType.PRESSURE, SensorType.FLOW],
                priority=MeasurementPriority.HIGH,
                is_pressurized=True,
                expected_value_range={
                    SensorType.PRESSURE: (0, comp_params.get("design_pressure", 100)),
                    SensorType.FLOW: (0, comp_params.get("design_flow", 30))
                }
            ))

            # 中间点（每5km一个）
            mid_points = int(length / 5000)
            for i in range(1, mid_points + 1):
                chainage = comp_params.get("start_chainage", 0) + i * 5000
                points.append(MeasurementPoint(
                    point_id=f"MP{start_id + i:04d}",
                    name=f"{comp_params.get('name', comp_id)}桩号{chainage/1000:.1f}km",
                    component_type=comp_type,
                    component_id=comp_id,
                    chainage=chainage,
                    required_measurements=[SensorType.PRESSURE],
                    priority=MeasurementPriority.MEDIUM,
                    is_pressurized=True
                ))

            # 出口
            points.append(MeasurementPoint(
                point_id=f"MP{start_id + mid_points + 1:04d}",
                name=f"{comp_params.get('name', comp_id)}出口",
                component_type=comp_type,
                component_id=comp_id,
                chainage=comp_params.get("start_chainage", 0) + length,
                required_measurements=[SensorType.PRESSURE, SensorType.FLOW],
                priority=MeasurementPriority.HIGH,
                is_pressurized=True
            ))

        elif comp_type == ComponentType.PUMP_STATION:
            # 泵站：进水池水位、出水压力、流量
            points.append(MeasurementPoint(
                point_id=f"MP{start_id:04d}",
                name=f"{comp_params.get('name', comp_id)}进水池",
                component_type=comp_type,
                component_id=comp_id,
                required_measurements=[SensorType.LEVEL],
                priority=MeasurementPriority.CRITICAL,
                is_safety_critical=True
            ))
            points.append(MeasurementPoint(
                point_id=f"MP{start_id + 1:04d}",
                name=f"{comp_params.get('name', comp_id)}出水",
                component_type=comp_type,
                component_id=comp_id,
                required_measurements=[SensorType.PRESSURE, SensorType.FLOW],
                priority=MeasurementPriority.CRITICAL,
                is_safety_critical=True,
                is_pressurized=True
            ))

        elif comp_type == ComponentType.SURGE_TANK:
            # 调压井：水位
            points.append(MeasurementPoint(
                point_id=f"MP{start_id:04d}",
                name=f"{comp_params.get('name', comp_id)}水位",
                component_type=comp_type,
                component_id=comp_id,
                required_measurements=[SensorType.LEVEL],
                priority=MeasurementPriority.CRITICAL,
                is_safety_critical=True
            ))

        elif comp_type == ComponentType.VALVE:
            # 阀门：上下游压力
            points.append(MeasurementPoint(
                point_id=f"MP{start_id:04d}",
                name=f"{comp_params.get('name', comp_id)}上游",
                component_type=comp_type,
                component_id=comp_id,
                required_measurements=[SensorType.PRESSURE],
                priority=MeasurementPriority.HIGH,
                is_pressurized=True
            ))
            points.append(MeasurementPoint(
                point_id=f"MP{start_id + 1:04d}",
                name=f"{comp_params.get('name', comp_id)}下游",
                component_type=comp_type,
                component_id=comp_id,
                required_measurements=[SensorType.PRESSURE],
                priority=MeasurementPriority.HIGH,
                is_pressurized=True
            ))

        return points


# ==========================================
# YCJL专用优化器
# ==========================================

class YCJLSensorOptimizer(BaseSensorOptimizer):
    """
    引绰济辽工程专用传感器优化器

    针对引绰济辽输水工程的特定约束和需求进行优化，包括：
    - 冰期运行特殊需求
    - 长距离输水管道
    - 水库群联合调度
    - 电站运行监测
    """

    def __init__(self, config: Optional[Any] = None):
        """
        初始化YCJL专用优化器

        Args:
            config: YinChuoProjectConfig 配置对象（可选）
        """
        super().__init__("引绰济辽输水工程")
        self.config = config

        # YCJL特有配置
        self.ice_period_enabled = True
        self.power_generation_enabled = True

        # 添加冰厚传感器到目录
        self._add_ice_sensors()

        # 初始化测量点
        self._initialize_measurement_points()

    def _add_ice_sensors(self):
        """添加冰期专用传感器"""
        self.sensor_catalog.add_sensor(SensorSpec(
            sensor_type=SensorType.ICE_THICKNESS,
            name="ICE-200-YCJL",
            manufacturer="寒区专用",
            range_min=0, range_max=1.5,
            accuracy=0.005, accuracy_type="absolute",
            response_time=30.0, sampling_rate_max=0.1,
            mtbf=25000,
            operating_temp_min=-45, operating_temp_max=25,
            purchase_cost=30000, installation_cost=15000,
            maintenance_cost_annual=5000,
            lifespan_years=8
        ))

    def _initialize_measurement_points(self):
        """初始化YCJL测量点位"""
        # 文得根水库
        self.add_measurement_point(MeasurementPoint(
            point_id="YCJL-WDG-001",
            name="文得根水库库水位",
            component_type=ComponentType.RESERVOIR,
            component_id="WDG",
            required_measurements=[SensorType.LEVEL, SensorType.TEMPERATURE],
            priority=MeasurementPriority.CRITICAL,
            is_safety_critical=True,
            has_ice_risk=True,
            expected_value_range={
                SensorType.LEVEL: (574.6, 617.0),
                SensorType.TEMPERATURE: (-2, 25)
            },
            ambient_temp_range=(-35, 35)
        ))

        # 进水口
        self.add_measurement_point(MeasurementPoint(
            point_id="YCJL-INL-001",
            name="进水口水位",
            component_type=ComponentType.GATE,
            component_id="INLET",
            chainage=0,
            required_measurements=[SensorType.LEVEL, SensorType.FLOW],
            priority=MeasurementPriority.CRITICAL,
            is_safety_critical=True,
            expected_value_range={
                SensorType.LEVEL: (574, 620),
                SensorType.FLOW: (0, 40)
            }
        ))

        # 引水隧洞
        tunnel_length = 230000  # 230km
        tunnel_segments = [
            (0, "进口"),
            (30000, "30km"),
            (60000, "60km"),
            (90000, "90km"),
            (115000, "电站前"),
            (120000, "电站后"),
            (150000, "150km"),
            (180000, "180km"),
            (210000, "210km"),
            (230000, "出口")
        ]

        for chainage, name in tunnel_segments:
            priority = MeasurementPriority.CRITICAL if name in ["进口", "电站前", "出口"] else MeasurementPriority.HIGH
            measurements = [SensorType.PRESSURE]
            if name in ["进口", "出口", "电站前", "电站后"]:
                measurements.append(SensorType.FLOW)

            self.add_measurement_point(MeasurementPoint(
                point_id=f"YCJL-TUN-{chainage//1000:03d}",
                name=f"隧洞{name}",
                component_type=ComponentType.TUNNEL,
                component_id="MAIN_TUNNEL",
                chainage=chainage,
                required_measurements=measurements,
                priority=priority,
                is_safety_critical=(priority == MeasurementPriority.CRITICAL),
                is_pressurized=True,
                has_ice_risk=(chainage < 50000),  # 前50km有冰期风险
                expected_value_range={
                    SensorType.PRESSURE: (0, 150),
                    SensorType.FLOW: (0, 40)
                }
            ))

        # 地下电站
        self.add_measurement_point(MeasurementPoint(
            point_id="YCJL-PWR-001",
            name="电站进水压力",
            component_type=ComponentType.POWER_STATION,
            component_id="POWER",
            chainage=115000,
            required_measurements=[SensorType.PRESSURE, SensorType.FLOW],
            priority=MeasurementPriority.CRITICAL,
            is_safety_critical=True,
            is_pressurized=True,
            expected_value_range={
                SensorType.PRESSURE: (50, 120),
                SensorType.FLOW: (0, 40)
            }
        ))

        self.add_measurement_point(MeasurementPoint(
            point_id="YCJL-PWR-002",
            name="电站尾水水位",
            component_type=ComponentType.POWER_STATION,
            component_id="POWER",
            chainage=120000,
            required_measurements=[SensorType.LEVEL],
            priority=MeasurementPriority.HIGH,
            expected_value_range={
                SensorType.LEVEL: (400, 450)
            }
        ))

        # 调压井
        self.add_measurement_point(MeasurementPoint(
            point_id="YCJL-SRG-001",
            name="上游调压井水位",
            component_type=ComponentType.SURGE_TANK,
            component_id="SURGE_UP",
            chainage=110000,
            required_measurements=[SensorType.LEVEL],
            priority=MeasurementPriority.CRITICAL,
            is_safety_critical=True,
            expected_value_range={
                SensorType.LEVEL: (550, 620)
            }
        ))

        self.add_measurement_point(MeasurementPoint(
            point_id="YCJL-SRG-002",
            name="下游调压井水位",
            component_type=ComponentType.SURGE_TANK,
            component_id="SURGE_DOWN",
            chainage=125000,
            required_measurements=[SensorType.LEVEL],
            priority=MeasurementPriority.CRITICAL,
            is_safety_critical=True,
            expected_value_range={
                SensorType.LEVEL: (400, 480)
            }
        ))

        # 出口闸
        self.add_measurement_point(MeasurementPoint(
            point_id="YCJL-OUT-001",
            name="出口闸上游",
            component_type=ComponentType.GATE,
            component_id="OUTLET",
            chainage=230000,
            required_measurements=[SensorType.LEVEL, SensorType.FLOW],
            priority=MeasurementPriority.CRITICAL,
            is_safety_critical=True,
            expected_value_range={
                SensorType.LEVEL: (260, 280),
                SensorType.FLOW: (0, 40)
            }
        ))

        # 冰期专用监测点
        if self.ice_period_enabled:
            ice_monitoring_points = [
                (5000, "进口段"),
                (15000, "明流段"),
                (30000, "过渡段")
            ]

            for chainage, name in ice_monitoring_points:
                self.add_measurement_point(MeasurementPoint(
                    point_id=f"YCJL-ICE-{chainage//1000:03d}",
                    name=f"冰期监测-{name}",
                    component_type=ComponentType.TUNNEL,
                    component_id="MAIN_TUNNEL",
                    chainage=chainage,
                    required_measurements=[SensorType.ICE_THICKNESS, SensorType.TEMPERATURE],
                    priority=MeasurementPriority.HIGH,
                    has_ice_risk=True,
                    ambient_temp_range=(-40, 30),
                    expected_value_range={
                        SensorType.ICE_THICKNESS: (0, 1.2),
                        SensorType.TEMPERATURE: (-30, 25)
                    }
                ))

    def optimize_for_ice_period(self) -> OptimizationSolution:
        """
        针对冰期运行进行优化

        Returns:
            冰期优化方案
        """
        # 保存原始设置
        original_objective = self._objective

        # 设置冰期优化参数
        self.set_objective(OptimizationObjective.ROBUSTNESS)

        # 增加冰期相关约束
        ice_constraints = OptimizationConstraint(
            name="ice_period",
            min_coverage=0.9,
            min_redundancy=0.5,
            min_system_availability=0.995
        )
        self.set_constraints(ice_constraints)

        # 执行优化
        solution = self.optimize()
        solution.name = "引绰济辽冰期运行优化方案"

        # 恢复设置
        self._objective = original_objective

        return solution

    def optimize_for_normal_operation(self) -> OptimizationSolution:
        """
        针对常规运行进行优化

        Returns:
            常规优化方案
        """
        self.set_objective(OptimizationObjective.BALANCED)

        normal_constraints = OptimizationConstraint(
            name="normal_operation",
            min_coverage=0.85,
            min_observability=0.8,
            max_total_cost=5000000  # 500万预算
        )
        self.set_constraints(normal_constraints)

        solution = self.optimize()
        solution.name = "引绰济辽常规运行优化方案"

        return solution


# ==========================================
# 高级优化算法 (v1.1)
# ==========================================

class OptimizationAlgorithm(Enum):
    """优化算法类型"""
    GREEDY = "贪心算法"                     # 基础贪心
    GENETIC = "遗传算法"                    # 遗传算法
    PSO = "粒子群优化"                      # 粒子群优化
    SIMULATED_ANNEALING = "模拟退火"        # 模拟退火


@dataclass
class GeneticAlgorithmConfig:
    """遗传算法配置"""
    population_size: int = 50               # 种群大小
    generations: int = 100                  # 迭代代数
    crossover_rate: float = 0.8             # 交叉率
    mutation_rate: float = 0.1              # 变异率
    elite_ratio: float = 0.1                # 精英比例
    tournament_size: int = 3                # 锦标赛选择大小


@dataclass
class PSOConfig:
    """粒子群优化配置"""
    swarm_size: int = 30                    # 粒子数量
    iterations: int = 100                   # 迭代次数
    w: float = 0.7                          # 惯性权重
    c1: float = 1.5                         # 认知参数
    c2: float = 1.5                         # 社会参数
    w_decay: float = 0.99                   # 惯性衰减


class GeneticSensorOptimizer:
    """
    遗传算法传感器优化器

    使用遗传算法进行传感器布置优化，特点：
    - 全局搜索能力强
    - 适合大规模组合优化问题
    - 支持多目标优化
    """

    def __init__(self, base_optimizer: BaseSensorOptimizer,
                 config: Optional[GeneticAlgorithmConfig] = None):
        self.base_optimizer = base_optimizer
        self.config = config or GeneticAlgorithmConfig()

        # 确保测量点已初始化
        if not base_optimizer._measurement_points:
            base_optimizer._initialize_measurement_points()

        self._best_solution: Optional[OptimizationSolution] = None
        self._generation_history: List[float] = []

    def _encode_solution(self, placements: List[SensorPlacement]) -> np.ndarray:
        """将方案编码为染色体"""
        # 编码: [传感器选择(0-N), 冗余度(1-3)]
        chromosome = []
        for p in placements:
            # 传感器型号索引
            all_sensors = self.base_optimizer.sensor_catalog.get_by_type(p.sensor_spec.sensor_type)
            sensor_idx = all_sensors.index(p.sensor_spec) if p.sensor_spec in all_sensors else 0
            chromosome.extend([sensor_idx, p.redundancy_count])
        return np.array(chromosome, dtype=float)

    def _decode_chromosome(self, chromosome: np.ndarray) -> List[SensorPlacement]:
        """将染色体解码为方案"""
        placements = []
        points = self.base_optimizer._measurement_points
        idx = 0
        placement_id = 0

        for point in points:
            for sensor_type in point.required_measurements:
                if idx + 1 >= len(chromosome):
                    break

                all_sensors = self.base_optimizer.sensor_catalog.get_by_type(sensor_type)
                if not all_sensors:
                    continue

                sensor_idx = int(chromosome[idx]) % len(all_sensors)
                redundancy = max(1, min(3, int(chromosome[idx + 1])))

                placements.append(SensorPlacement(
                    placement_id=f"GA-{placement_id:04d}",
                    point=point,
                    sensor_spec=all_sensors[sensor_idx],
                    redundancy_count=redundancy,
                    redundancy_type="hot" if redundancy > 1 else "none"
                ))
                placement_id += 1
                idx += 2

        return placements

    def _initialize_population(self) -> List[np.ndarray]:
        """初始化种群"""
        population = []
        # 第一个个体使用贪心解
        greedy_solution = self.base_optimizer._generate_initial_solution()
        population.append(self._encode_solution(greedy_solution))

        # 其余随机生成
        chromosome_length = len(population[0])
        for _ in range(self.config.population_size - 1):
            chromosome = np.zeros(chromosome_length)
            for i in range(0, chromosome_length, 2):
                chromosome[i] = np.random.randint(0, 10)  # 传感器选择
                chromosome[i + 1] = np.random.randint(1, 4)  # 冗余度
            population.append(chromosome)

        return population

    def _evaluate_fitness(self, chromosome: np.ndarray) -> float:
        """评估适应度"""
        placements = self._decode_chromosome(chromosome)
        if not placements:
            return 0.0

        scores = self.base_optimizer._calculate_scores(placements)
        fitness = self.base_optimizer._evaluate_objective(scores)

        # 约束惩罚
        total_cost = sum(p.total_cost for p in placements)
        if self.base_optimizer._constraints.max_total_cost:
            if total_cost > self.base_optimizer._constraints.max_total_cost:
                penalty = (total_cost - self.base_optimizer._constraints.max_total_cost) / \
                          self.base_optimizer._constraints.max_total_cost
                fitness *= (1 - min(0.5, penalty))

        return fitness

    def _selection(self, population: List[np.ndarray],
                   fitness_scores: List[float]) -> List[np.ndarray]:
        """锦标赛选择"""
        selected = []
        for _ in range(len(population)):
            tournament_idx = np.random.choice(len(population), self.config.tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_idx]
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        return selected

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """双点交叉"""
        if np.random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()

        length = len(parent1)
        point1, point2 = sorted(np.random.choice(length, 2, replace=False))

        child1 = parent1.copy()
        child2 = parent2.copy()
        child1[point1:point2] = parent2[point1:point2]
        child2[point1:point2] = parent1[point1:point2]

        return child1, child2

    def _mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """变异操作"""
        mutated = chromosome.copy()
        for i in range(len(mutated)):
            if np.random.random() < self.config.mutation_rate:
                if i % 2 == 0:  # 传感器选择
                    mutated[i] = np.random.randint(0, 10)
                else:  # 冗余度
                    mutated[i] = np.random.randint(1, 4)
        return mutated

    def optimize(self) -> OptimizationSolution:
        """执行遗传算法优化"""
        import time
        start_time = time.time()

        # 初始化种群
        population = self._initialize_population()
        best_fitness = -float('inf')
        best_chromosome = None

        for generation in range(self.config.generations):
            # 评估适应度
            fitness_scores = [self._evaluate_fitness(c) for c in population]

            # 更新最优解
            max_idx = np.argmax(fitness_scores)
            if fitness_scores[max_idx] > best_fitness:
                best_fitness = fitness_scores[max_idx]
                best_chromosome = population[max_idx].copy()

            self._generation_history.append(best_fitness)

            # 精英保留
            elite_count = int(self.config.population_size * self.config.elite_ratio)
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            elites = [population[i].copy() for i in elite_indices]

            # 选择
            selected = self._selection(population, fitness_scores)

            # 交叉和变异
            new_population = elites.copy()
            while len(new_population) < self.config.population_size:
                idx1, idx2 = np.random.choice(len(selected), 2, replace=False)
                child1, child2 = self._crossover(selected[idx1], selected[idx2])
                new_population.append(self._mutate(child1))
                if len(new_population) < self.config.population_size:
                    new_population.append(self._mutate(child2))

            population = new_population[:self.config.population_size]

        # 构建最优解
        best_placements = self._decode_chromosome(best_chromosome)
        final_scores = self.base_optimizer._calculate_scores(best_placements)

        solution = OptimizationSolution(
            solution_id=f"GA-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            name=f"{self.base_optimizer.project_name}遗传算法优化方案",
            placements=best_placements,
            coverage_rate=final_scores["coverage"],
            observability_score=final_scores["observability"],
            redundancy_score=final_scores["redundancy"],
            robustness_score=final_scores["robustness"],
            total_points=len(self.base_optimizer._measurement_points),
            optimization_iterations=self.config.generations,
            optimization_time=time.time() - start_time,
            objective_value=best_fitness
        )
        solution.update_statistics()

        # 检查约束
        satisfied, violations = self.base_optimizer._constraints.is_satisfied(solution)
        solution.constraints_satisfied = satisfied
        solution.constraint_violations = violations

        self._best_solution = solution
        return solution

    def get_convergence_history(self) -> List[float]:
        """获取收敛历史"""
        return self._generation_history


class PSOSensorOptimizer:
    """
    粒子群优化传感器布置器

    使用粒子群优化算法进行传感器布置优化，特点：
    - 收敛速度快
    - 实现简单
    - 适合连续优化问题
    """

    def __init__(self, base_optimizer: BaseSensorOptimizer,
                 config: Optional[PSOConfig] = None):
        self.base_optimizer = base_optimizer
        self.config = config or PSOConfig()

        if not base_optimizer._measurement_points:
            base_optimizer._initialize_measurement_points()

        self._best_solution: Optional[OptimizationSolution] = None
        self._iteration_history: List[float] = []

    def _calculate_dimension(self) -> int:
        """计算问题维度"""
        dim = 0
        for point in self.base_optimizer._measurement_points:
            dim += len(point.required_measurements) * 2  # 传感器选择 + 冗余
        return dim

    def _position_to_placements(self, position: np.ndarray) -> List[SensorPlacement]:
        """将粒子位置转换为布置方案"""
        placements = []
        idx = 0
        placement_id = 0

        for point in self.base_optimizer._measurement_points:
            for sensor_type in point.required_measurements:
                if idx + 1 >= len(position):
                    break

                all_sensors = self.base_optimizer.sensor_catalog.get_by_type(sensor_type)
                if not all_sensors:
                    continue

                sensor_idx = int(abs(position[idx])) % len(all_sensors)
                redundancy = max(1, min(3, int(abs(position[idx + 1])) % 4 + 1))

                placements.append(SensorPlacement(
                    placement_id=f"PSO-{placement_id:04d}",
                    point=point,
                    sensor_spec=all_sensors[sensor_idx],
                    redundancy_count=redundancy,
                    redundancy_type="hot" if redundancy > 1 else "none"
                ))
                placement_id += 1
                idx += 2

        return placements

    def _evaluate_particle(self, position: np.ndarray) -> float:
        """评估粒子适应度"""
        placements = self._position_to_placements(position)
        if not placements:
            return 0.0

        scores = self.base_optimizer._calculate_scores(placements)
        fitness = self.base_optimizer._evaluate_objective(scores)

        # 约束惩罚
        total_cost = sum(p.total_cost for p in placements)
        if self.base_optimizer._constraints.max_total_cost:
            if total_cost > self.base_optimizer._constraints.max_total_cost:
                penalty = (total_cost - self.base_optimizer._constraints.max_total_cost) / \
                          self.base_optimizer._constraints.max_total_cost
                fitness *= (1 - min(0.5, penalty))

        return fitness

    def optimize(self) -> OptimizationSolution:
        """执行粒子群优化"""
        import time
        start_time = time.time()

        dim = self._calculate_dimension()
        swarm_size = self.config.swarm_size

        # 初始化粒子群
        positions = np.random.uniform(0, 10, (swarm_size, dim))
        velocities = np.random.uniform(-1, 1, (swarm_size, dim))

        # 个体最优
        personal_best_positions = positions.copy()
        personal_best_scores = np.array([self._evaluate_particle(p) for p in positions])

        # 全局最优
        global_best_idx = np.argmax(personal_best_scores)
        global_best_position = positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]

        w = self.config.w

        for iteration in range(self.config.iterations):
            for i in range(swarm_size):
                # 更新速度
                r1, r2 = np.random.random(dim), np.random.random(dim)
                cognitive = self.config.c1 * r1 * (personal_best_positions[i] - positions[i])
                social = self.config.c2 * r2 * (global_best_position - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social

                # 限制速度
                velocities[i] = np.clip(velocities[i], -5, 5)

                # 更新位置
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], 0, 20)

                # 评估
                score = self._evaluate_particle(positions[i])

                # 更新个体最优
                if score > personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()

                    # 更新全局最优
                    if score > global_best_score:
                        global_best_score = score
                        global_best_position = positions[i].copy()

            # 惯性权重衰减
            w *= self.config.w_decay
            self._iteration_history.append(global_best_score)

        # 构建最优解
        best_placements = self._position_to_placements(global_best_position)
        final_scores = self.base_optimizer._calculate_scores(best_placements)

        solution = OptimizationSolution(
            solution_id=f"PSO-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            name=f"{self.base_optimizer.project_name}粒子群优化方案",
            placements=best_placements,
            coverage_rate=final_scores["coverage"],
            observability_score=final_scores["observability"],
            redundancy_score=final_scores["redundancy"],
            robustness_score=final_scores["robustness"],
            total_points=len(self.base_optimizer._measurement_points),
            optimization_iterations=self.config.iterations,
            optimization_time=time.time() - start_time,
            objective_value=global_best_score
        )
        solution.update_statistics()

        satisfied, violations = self.base_optimizer._constraints.is_satisfied(solution)
        solution.constraints_satisfied = satisfied
        solution.constraint_violations = violations

        self._best_solution = solution
        return solution

    def get_convergence_history(self) -> List[float]:
        """获取收敛历史"""
        return self._iteration_history


class MultiAlgorithmOptimizer:
    """
    多算法组合优化器

    自动选择或组合多种算法进行优化
    """

    def __init__(self, base_optimizer: BaseSensorOptimizer):
        self.base_optimizer = base_optimizer
        self._results: Dict[str, OptimizationSolution] = {}

    def optimize_all(self) -> Dict[str, OptimizationSolution]:
        """使用所有算法进行优化"""
        # 贪心算法
        print("  运行贪心算法...")
        greedy_solution = self.base_optimizer.optimize()
        greedy_solution.name += "(贪心)"
        self._results["greedy"] = greedy_solution

        # 遗传算法
        print("  运行遗传算法...")
        ga_optimizer = GeneticSensorOptimizer(self.base_optimizer)
        ga_solution = ga_optimizer.optimize()
        self._results["genetic"] = ga_solution

        # 粒子群优化
        print("  运行粒子群优化...")
        pso_optimizer = PSOSensorOptimizer(self.base_optimizer)
        pso_solution = pso_optimizer.optimize()
        self._results["pso"] = pso_solution

        return self._results

    def get_best_solution(self) -> OptimizationSolution:
        """获取最优方案"""
        if not self._results:
            self.optimize_all()

        best = max(self._results.values(), key=lambda s: s.objective_value)
        return best

    def compare_results(self) -> str:
        """比较各算法结果"""
        if not self._results:
            self.optimize_all()

        lines = [
            "=" * 70,
            "多算法优化结果比较",
            "=" * 70,
            f"{'算法':<15} {'目标值':<10} {'覆盖率':<10} {'成本(万)':<12} {'耗时(s)':<10}",
            "-" * 70
        ]

        for name, sol in sorted(self._results.items(), key=lambda x: -x[1].objective_value):
            lines.append(
                f"{name:<15} {sol.objective_value:.4f}     "
                f"{sol.coverage_rate:.1%}     "
                f"¥{sol.total_cost/10000:.1f}        "
                f"{sol.optimization_time:.2f}"
            )

        best = self.get_best_solution()
        lines.extend([
            "-" * 70,
            f"推荐方案: {best.name}",
            f"目标函数值: {best.objective_value:.4f}",
            "=" * 70
        ])

        return "\n".join(lines)


# ==========================================
# 工厂函数
# ==========================================

def create_sensor_optimizer(project_type: str,
                            project_name: str,
                            project_params: Optional[Dict] = None,
                            config: Optional[Any] = None) -> BaseSensorOptimizer:
    """
    创建传感器优化器的工厂函数

    Args:
        project_type: 项目类型 ("ycjl", "miyun", "generic")
        project_name: 项目名称
        project_params: 项目参数（通用类型需要）
        config: 配置对象（专用类型可选）

    Returns:
        对应的优化器实例
    """
    if project_type.lower() == "ycjl":
        return YCJLSensorOptimizer(config)
    elif project_type.lower() == "generic" or project_type.lower() == "water":
        if project_params is None:
            raise ValueError("通用优化器需要提供project_params")
        return WaterProjectSensorOptimizer(project_name, project_params)
    else:
        raise ValueError(f"不支持的项目类型: {project_type}")


# ==========================================
# 可视化报告生成器 (v1.1)
# ==========================================

class ReportFormat(Enum):
    """报告格式"""
    TEXT = "文本"
    HTML = "HTML"
    MARKDOWN = "Markdown"
    JSON = "JSON"


class SensorOptimizationReporter:
    """
    传感器优化报告生成器

    支持多种格式的报告输出：
    - 文本报告
    - HTML报告（含图表）
    - Markdown报告
    - JSON数据导出
    """

    def __init__(self, solution: OptimizationSolution,
                 optimizer: Optional[BaseSensorOptimizer] = None):
        self.solution = solution
        self.optimizer = optimizer
        self._convergence_history: Optional[List[float]] = None

    def set_convergence_history(self, history: List[float]):
        """设置收敛历史（用于图表）"""
        self._convergence_history = history

    def generate_text_report(self) -> str:
        """生成文本报告"""
        return self.solution.summary()

    def generate_markdown_report(self) -> str:
        """生成Markdown报告"""
        sol = self.solution
        lines = [
            f"# {sol.name}",
            "",
            f"**方案ID**: {sol.solution_id}",
            f"**生成时间**: {sol.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 总体评估",
            "",
            "| 指标 | 数值 |",
            "|------|------|",
            f"| 覆盖率 | {sol.coverage_rate:.1%} |",
            f"| 可观测性 | {sol.observability_score:.1%} |",
            f"| 冗余度 | {sol.redundancy_score:.1%} |",
            f"| 鲁棒性 | {sol.robustness_score:.1%} |",
            "",
            "## 成本统计",
            "",
            f"- **总投资**: ¥{sol.total_cost:,.0f}",
            f"- **年度成本**: ¥{sol.annual_cost:,.0f}",
            "",
            "## 设备统计",
            "",
            f"- **传感器总数**: {sol.sensor_count}",
            f"- **覆盖点位**: {sol.covered_points}/{sol.total_points}",
            "",
            "### 按类型分布",
            "",
            "| 类型 | 数量 |",
            "|------|------|",
        ]

        for st, count in sorted(sol.sensor_by_type.items(), key=lambda x: -x[1]):
            lines.append(f"| {st.display_name} | {count} |")

        # 详细布置表
        lines.extend([
            "",
            "## 详细布置清单",
            "",
            "| 点位 | 传感器 | 型号 | 冗余 | 成本 |",
            "|------|--------|------|------|------|",
        ])

        for p in sol.placements:
            lines.append(
                f"| {p.point.name} | {p.sensor_spec.sensor_type.display_name} | "
                f"{p.sensor_spec.name} | {p.redundancy_count} | ¥{p.total_cost:,.0f} |"
            )

        # 约束检查
        if not sol.constraints_satisfied:
            lines.extend([
                "",
                "## ⚠️ 约束违反",
                ""
            ])
            for v in sol.constraint_violations:
                lines.append(f"- {v}")

        return "\n".join(lines)

    def generate_html_report(self) -> str:
        """生成HTML报告（含图表）"""
        sol = self.solution

        # 生成图表数据
        chart_data = self._generate_chart_data()

        html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{sol.name}</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .info-box {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
        .metric-card.cost {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
        .metric-value {{ font-size: 32px; font-weight: bold; }}
        .metric-label {{ font-size: 14px; opacity: 0.9; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .chart-container {{ margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 10px; }}
        .bar {{ height: 30px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 5px; margin: 5px 0; transition: width 0.5s; }}
        .bar-label {{ display: flex; justify-content: space-between; margin-bottom: 5px; }}
        .pie-chart {{ width: 300px; height: 300px; margin: 0 auto; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
        .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 {sol.name}</h1>

        <div class="info-box">
            <strong>方案ID:</strong> {sol.solution_id} |
            <strong>生成时间:</strong> {sol.timestamp.strftime('%Y-%m-%d %H:%M:%S')} |
            <strong>优化耗时:</strong> {sol.optimization_time:.2f}秒
        </div>

        <h2>📈 总体评估</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{sol.coverage_rate:.1%}</div>
                <div class="metric-label">覆盖率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{sol.observability_score:.1%}</div>
                <div class="metric-label">可观测性</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{sol.redundancy_score:.1%}</div>
                <div class="metric-label">冗余度</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{sol.robustness_score:.1%}</div>
                <div class="metric-label">鲁棒性</div>
            </div>
        </div>

        <div class="metric-grid">
            <div class="metric-card cost">
                <div class="metric-value">¥{sol.total_cost/10000:.1f}万</div>
                <div class="metric-label">总投资</div>
            </div>
            <div class="metric-card cost">
                <div class="metric-value">¥{sol.annual_cost/10000:.1f}万</div>
                <div class="metric-label">年度成本</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{sol.sensor_count}</div>
                <div class="metric-label">传感器总数</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{sol.covered_points}/{sol.total_points}</div>
                <div class="metric-label">覆盖点位</div>
            </div>
        </div>

        <h2>📊 传感器类型分布</h2>
        <div class="chart-container">
            {chart_data['type_distribution']}
        </div>

        <h2>📋 详细布置清单</h2>
        <table>
            <thead>
                <tr>
                    <th>点位</th>
                    <th>组件</th>
                    <th>传感器类型</th>
                    <th>型号</th>
                    <th>冗余</th>
                    <th>成本</th>
                </tr>
            </thead>
            <tbody>
'''

        for p in sol.placements:
            html += f'''                <tr>
                    <td>{p.point.name}</td>
                    <td>{p.point.component_type.value}</td>
                    <td>{p.sensor_spec.sensor_type.display_name}</td>
                    <td>{p.sensor_spec.name}</td>
                    <td>{p.redundancy_count}×</td>
                    <td>¥{p.total_cost:,.0f}</td>
                </tr>
'''

        html += f'''            </tbody>
        </table>

        <div class="footer">
            生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 传感器优化模块 v1.1
        </div>
    </div>
</body>
</html>'''

        return html

    def _generate_chart_data(self) -> Dict[str, str]:
        """生成图表HTML"""
        sol = self.solution

        # 类型分布条形图
        type_bars = []
        if sol.sensor_by_type:
            max_count = max(sol.sensor_by_type.values())
            for st, count in sorted(sol.sensor_by_type.items(), key=lambda x: -x[1]):
                width = int(count / max_count * 100) if max_count > 0 else 0
                type_bars.append(f'''
            <div class="bar-label">
                <span>{st.display_name}</span>
                <span>{count}个</span>
            </div>
            <div class="bar" style="width: {width}%;"></div>
''')

        return {
            'type_distribution': '\n'.join(type_bars)
        }

    def generate_json_report(self) -> str:
        """生成JSON报告"""
        import json
        return json.dumps(self.solution.to_dict(), ensure_ascii=False, indent=2)

    def generate_report(self, format: ReportFormat = ReportFormat.TEXT) -> str:
        """
        生成报告

        Args:
            format: 报告格式

        Returns:
            报告内容
        """
        if format == ReportFormat.TEXT:
            return self.generate_text_report()
        elif format == ReportFormat.MARKDOWN:
            return self.generate_markdown_report()
        elif format == ReportFormat.HTML:
            return self.generate_html_report()
        elif format == ReportFormat.JSON:
            return self.generate_json_report()
        else:
            return self.generate_text_report()

    def save_report(self, filepath: str, format: Optional[ReportFormat] = None):
        """
        保存报告到文件

        Args:
            filepath: 文件路径
            format: 报告格式（自动从扩展名推断）
        """
        if format is None:
            ext = filepath.lower().split('.')[-1]
            format_map = {
                'txt': ReportFormat.TEXT,
                'md': ReportFormat.MARKDOWN,
                'html': ReportFormat.HTML,
                'json': ReportFormat.JSON
            }
            format = format_map.get(ext, ReportFormat.TEXT)

        content = self.generate_report(format)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    @staticmethod
    def generate_comparison_report(solutions: List[OptimizationSolution],
                                   format: ReportFormat = ReportFormat.TEXT) -> str:
        """
        生成多方案比较报告

        Args:
            solutions: 方案列表
            format: 报告格式

        Returns:
            比较报告
        """
        if format == ReportFormat.MARKDOWN:
            lines = [
                "# 传感器优化方案比较报告",
                "",
                "## 方案对比",
                "",
                "| 方案 | 覆盖率 | 可观测性 | 鲁棒性 | 成本(万) | 传感器数 |",
                "|------|--------|----------|--------|----------|----------|"
            ]

            for sol in sorted(solutions, key=lambda x: -x.objective_value):
                lines.append(
                    f"| {sol.name} | {sol.coverage_rate:.1%} | "
                    f"{sol.observability_score:.1%} | {sol.robustness_score:.1%} | "
                    f"¥{sol.total_cost/10000:.1f} | {sol.sensor_count} |"
                )

            # 找出最优
            best = max(solutions, key=lambda x: x.objective_value)
            lines.extend([
                "",
                f"## 推荐方案: {best.name}",
                f"- 目标函数值: {best.objective_value:.4f}",
                f"- 总投资: ¥{best.total_cost:,.0f}"
            ])

            return "\n".join(lines)

        else:
            # 文本格式
            lines = [
                "=" * 70,
                "传感器优化方案比较报告",
                "=" * 70,
                "",
                f"{'方案':<25} {'覆盖率':<10} {'成本(万)':<12} {'传感器数':<10}",
                "-" * 70
            ]

            for sol in sorted(solutions, key=lambda x: -x.objective_value):
                lines.append(
                    f"{sol.name:<25} {sol.coverage_rate:.1%}     "
                    f"¥{sol.total_cost/10000:<10.1f} {sol.sensor_count}"
                )

            best = max(solutions, key=lambda x: x.objective_value)
            lines.extend([
                "-" * 70,
                f"推荐: {best.name}",
                "=" * 70
            ])

            return "\n".join(lines)


# ==========================================
# 导出
# ==========================================

__all__ = [
    # 枚举
    'SensorType',
    'MeasurementPriority',
    'ComponentType',
    'OptimizationObjective',
    'PlacementStrategy',
    'OptimizationAlgorithm',

    # 数据类
    'SensorSpec',
    'MeasurementPoint',
    'SensorPlacement',
    'OptimizationConstraint',
    'OptimizationSolution',
    'GeneticAlgorithmConfig',
    'PSOConfig',

    # 分析器
    'SensorCatalog',
    'ObservabilityAnalyzer',
    'CostBenefitAnalyzer',
    'RobustnessAnalyzer',

    # 优化器
    'BaseSensorOptimizer',
    'WaterProjectSensorOptimizer',
    'YCJLSensorOptimizer',

    # 高级优化算法
    'GeneticSensorOptimizer',
    'PSOSensorOptimizer',
    'MultiAlgorithmOptimizer',

    # 工厂
    'create_sensor_optimizer',

    # 报告生成器
    'ReportFormat',
    'SensorOptimizationReporter'
]
