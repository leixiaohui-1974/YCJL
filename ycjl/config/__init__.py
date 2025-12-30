"""
配置模块
========

包含引绰济辽工程的全部配置参数。

模块:
- settings: 原有配置 (向后兼容)
- config_database: 工程级参数数据库 v3.2
- ice_parameters: 冰期参数数据库 v3.3
- gap_analyzer: 数据完备性诊断器 v1.0 [NEW]
"""
from .settings import (
    Config,
    SeasonMode,
    ScenarioType,
    PhysicsConstants,
    ReservoirConfig,
    TunnelConfig,
    PoolConfig,
    SurgeTankConfig,
    PipelineConfig,
    ValveConfig,
    SafetyConfig,
    SimulationConfig,
    ControlConfig
)

from .config_database import (
    YinChuoProjectConfig,
    ProjectParams,
    GlobalPhysicsConfig,
    CharacteristicCurves,
    SourceHubConfig,
    TunnelSystemConfig,
    StabilizingPoolConfig,
    PipelineSystemConfig,
    SurgeTankConfig as SurgeTankConfigV2,
    RegulatingValveConfig,
    EndUserConfig,
    SafetySystemConfig,
    SimulationHardwareConfig,
    ControlParameterConfig,
    HydrologyConfig,
    GlobalConfig,
    CurveDatabase,
    SourceConfig,
    PoolConfig as PoolConfigV2,
    PipeConfig,
    SurgeConfig,
    ValveConfig as ValveConfigV2,
    UserConfig,
    SafetyConfig as SafetyConfigV2,
    HardwareConfig,
    ControlConfig as ControlConfigV2,
    HydroConfig,
    create_interpolator
)

# 冰期参数数据库 v3.3
from .ice_parameters import (
    IceType,
    IcePhase,
    BreakupType,
    IcePhysicalProperties,
    StefanEquationParams,
    CompositeRoughnessParams,
    FrazilIceParams,
    AnchorIceParams,
    IceCoverFormationParams,
    IceJamParams,
    BreakupCriteriaParams,
    ThermalExchangeParams,
    YCJLIcePeriodParams,
    IceHydraulicsConfig,
    IceParams,
    IcePhysical,
    StefanParams,
    RoughnessParams,
    FrazilParams,
    AnchorParams,
    CoverParams,
    JamParams,
    BreakupParams,
    ThermalParams,
    YCJLIceParams
)

__all__ = [
    # 原有配置
    'Config',
    'SeasonMode',
    'ScenarioType',
    'PhysicsConstants',
    'ReservoirConfig',
    'TunnelConfig',
    'PoolConfig',
    'SurgeTankConfig',
    'PipelineConfig',
    'ValveConfig',
    'SafetyConfig',
    'SimulationConfig',
    'ControlConfig',
    # 新配置数据库
    'YinChuoProjectConfig',
    'ProjectParams',
    'GlobalPhysicsConfig',
    'CharacteristicCurves',
    'SourceHubConfig',
    'TunnelSystemConfig',
    'StabilizingPoolConfig',
    'PipelineSystemConfig',
    'RegulatingValveConfig',
    'EndUserConfig',
    'SafetySystemConfig',
    'SimulationHardwareConfig',
    'ControlParameterConfig',
    'HydrologyConfig',
    'GlobalConfig',
    'CurveDatabase',
    'SourceConfig',
    'PipeConfig',
    'SurgeConfig',
    'UserConfig',
    'HardwareConfig',
    'HydroConfig',
    'create_interpolator',
    # 冰期参数数据库
    'IceType',
    'IcePhase',
    'BreakupType',
    'IcePhysicalProperties',
    'StefanEquationParams',
    'CompositeRoughnessParams',
    'FrazilIceParams',
    'AnchorIceParams',
    'IceCoverFormationParams',
    'IceJamParams',
    'BreakupCriteriaParams',
    'ThermalExchangeParams',
    'YCJLIcePeriodParams',
    'IceHydraulicsConfig',
    'IceParams',
    'IcePhysical',
    'StefanParams',
    'RoughnessParams',
    'FrazilParams',
    'AnchorParams',
    'CoverParams',
    'JamParams',
    'BreakupParams',
    'ThermalParams',
    'YCJLIceParams',
    # 数据完备性诊断器
    'YinChuoGapAnalyzer',
    'YCJLGapAnalyzer',
    'analyze_data_readiness',
    'print_gap_report',
    'get_critical_missing'
]

# 数据完备性诊断器 v1.0
from .gap_analyzer import (
    YinChuoGapAnalyzer,
    YCJLGapAnalyzer,
    analyze_data_readiness,
    print_gap_report,
    get_critical_missing
)
