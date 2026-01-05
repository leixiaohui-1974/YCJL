"""
水利工程智能输水系统通用核心框架 (Core Framework)
=================================================

版本: 2.1.0
日期: 2026-01-05

本模块提供可复用的基础设施，支持多个水利工程项目：
- 引绰济辽工程
- 密云水库调蓄工程
- 其他水利工程项目（可扩展）

核心组件:
---------
1. base_config - 配置基类和验证框架
2. interpolators - 插值器工厂（PCHIP、线性、双线性）
3. base_physics - 物理模型基类（水库、管道、泵站、阀门）
4. base_simulation - 仿真引擎基类
5. base_scheduler - 调度器基类
6. gap_analyzer - 通用数据完备性诊断器
7. odd_analyzer - 设计运行域(ODD)分析器
8. constants - 全局物理常数

设计原则:
---------
- 模块化：各组件独立可测试
- 可扩展：通过继承支持特定工程
- 类型安全：使用dataclass和Enum
- 文档化：完整的docstring和类型注解
"""

__version__ = "2.1.0"
__author__ = "YCJL Development Team"

# ==========================================
# 物理常数
# ==========================================
from .constants import (
    PhysicsConstants,
    GRAVITY,
    WATER_DENSITY,
    ATMOSPHERIC_PRESSURE_HEAD,
    KINEMATIC_VISCOSITY,
    WATER_BULK_MODULUS,
    VAPOR_PRESSURE_HEAD
)

# ==========================================
# 插值器工厂
# ==========================================
from .interpolators import (
    InterpolatorType,
    InterpolatorFactory,
    create_interpolator,
    create_bilinear_interpolator,
    create_curve_lookup
)

# ==========================================
# 配置基类
# ==========================================
from .base_config import (
    BaseProjectConfig,
    BaseGlobalPhysicsConfig,
    BaseReservoirConfig,
    BasePipelineConfig,
    BasePumpStationConfig,
    BaseControlConfig,
    BaseSafetyConfig,
    ConfigValidator,
    ValidationResult
)

# ==========================================
# 物理模型基类
# ==========================================
from .base_physics import (
    ComponentType,
    ComponentStatus,
    BaseHydraulicComponent,
    BaseReservoir,
    BasePipeline,
    BasePumpStation,
    BaseValve,
    BaseChannel,
    HydraulicState
)

# ==========================================
# 仿真基类
# ==========================================
from .base_simulation import (
    SimulationMode,
    SimulationStatus,
    BaseSimulationEngine,
    SimulationResult,
    TimeSeriesData
)

# ==========================================
# 调度器基类
# ==========================================
from .base_scheduler import (
    ScheduleMode,
    OperationZone,
    BaseScheduler,
    ScheduleDecision,
    ScheduleConstraint,
    # 增强调度器接口 (v2.0)
    HealthLevel,
    EnhancedScheduleDecision,
    HealthReport,
    IEnhancedScheduler,
    calculate_health_score,
    get_health_level
)

# ==========================================
# 数据完备性诊断器
# ==========================================
from .gap_analyzer import (
    DataReadinessLevel,
    DataPriority,
    DataCategory,
    MissingDataItem,
    DataGapReport,
    BaseGapAnalyzer
)

# ==========================================
# 设计运行域(ODD)分析器
# ==========================================
from .odd_analyzer import (
    # 枚举
    ODDDimension,
    ODDStatus,
    AutonomyLevel,
    ConstraintType,
    ViolationSeverity,
    # 数据类
    ODDBoundary,
    ODDViolation,
    DimensionScore,
    ObservabilityMetrics,
    ControllabilityMetrics,
    ODDReport,
    # 分析器
    BaseODDAnalyzer,
    WaterNetworkODDAnalyzer,
    # 便捷函数
    calculate_odd_reliability,
    determine_autonomy_from_score,
    create_water_odd_analyzer,
    # ODD自动构建器 (从设备配置推导)
    SensorProfile,
    ActuatorProfile,
    ChannelProfile,
    CommunicationProfile,
    ODDProfileBuilder,
    ConfiguredODDAnalyzer,
    build_odd_from_config,
    # 世界模型约束
    WorldModelConstraint,
    WorldModelODDEnvelope,
    create_world_model_envelope
)

# ==========================================
# 导出列表
# ==========================================
__all__ = [
    # 版本
    '__version__',

    # 物理常数
    'PhysicsConstants',
    'GRAVITY',
    'WATER_DENSITY',
    'ATMOSPHERIC_PRESSURE_HEAD',
    'KINEMATIC_VISCOSITY',
    'WATER_BULK_MODULUS',
    'VAPOR_PRESSURE_HEAD',

    # 插值器
    'InterpolatorType',
    'InterpolatorFactory',
    'create_interpolator',
    'create_bilinear_interpolator',
    'create_curve_lookup',

    # 配置基类
    'BaseProjectConfig',
    'BaseGlobalPhysicsConfig',
    'BaseReservoirConfig',
    'BasePipelineConfig',
    'BasePumpStationConfig',
    'BaseControlConfig',
    'BaseSafetyConfig',
    'ConfigValidator',
    'ValidationResult',

    # 物理模型基类
    'ComponentType',
    'ComponentStatus',
    'BaseHydraulicComponent',
    'BaseReservoir',
    'BasePipeline',
    'BasePumpStation',
    'BaseValve',
    'BaseChannel',
    'HydraulicState',

    # 仿真基类
    'SimulationMode',
    'SimulationStatus',
    'BaseSimulationEngine',
    'SimulationResult',
    'TimeSeriesData',

    # 调度器基类
    'ScheduleMode',
    'OperationZone',
    'BaseScheduler',
    'ScheduleDecision',
    'ScheduleConstraint',
    # 增强调度器接口
    'HealthLevel',
    'EnhancedScheduleDecision',
    'HealthReport',
    'IEnhancedScheduler',
    'calculate_health_score',
    'get_health_level',

    # 数据诊断器
    'DataReadinessLevel',
    'DataPriority',
    'DataCategory',
    'MissingDataItem',
    'DataGapReport',
    'BaseGapAnalyzer',

    # ODD分析器
    'ODDDimension',
    'ODDStatus',
    'AutonomyLevel',
    'ConstraintType',
    'ViolationSeverity',
    'ODDBoundary',
    'ODDViolation',
    'DimensionScore',
    'ObservabilityMetrics',
    'ControllabilityMetrics',
    'ODDReport',
    'BaseODDAnalyzer',
    'WaterNetworkODDAnalyzer',
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
