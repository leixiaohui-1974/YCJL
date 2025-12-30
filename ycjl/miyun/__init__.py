"""
密云水库调蓄工程智能输水系统 (Miyun Reservoir Water Transfer System)
===================================================================

版本: 1.1.0
日期: 2024-12-30

模块概述:
---------
本模块实现密云水库调蓄工程的L5级数字孪生与自主调度系统，包括：

1. config_database - 全系统参数配置数据库
   - 9个泵站枢纽参数 (6个明渠段 + 3个有压管段)
   - 密云水库特征水位与库容
   - 京密引水渠参数
   - 有压管道参数

2. physics_engine - 物理仿真引擎
   - 扬程/功率计算
   - 水力损失计算
   - 安全诊断
   - 水锤估算

3. gap_analyzer - 数据完备性诊断器
   - L5级数据需求分析
   - 缺失数据识别
   - 补全优先级建议

4. scheduler - L5级自主调度器
   - 数字孪生体检
   - 调度决策生成
   - 多场景仿真

5. scenarios - 场景库 [NEW v1.1]
   - 20+种工况场景定义
   - 场景检测器
   - 响应措施库

工程概况:
---------
密云水库调蓄工程是北京市重要的水源工程，包括：
- 京密引水渠：57km明渠，6座梯级泵站
- 有压管道：约26.6km PCCP/钢管，3座泵站
- 设计流量：20 m³/s
- 总扬程：约100m

快速使用:
---------
    # 导入模块
    from ycjl.miyun import run_simulation, run_diagnosis

    # 运行综合仿真
    result = run_simulation(flow=10.0)

    # 单独运行诊断
    diagnosis = run_diagnosis(flow=15.0)

详细使用:
---------
    from ycjl.miyun import (
        MiyunParams,           # 项目配置
        STATION_DB,            # 泵站数据库
        SimEngine,             # 仿真引擎
        GapAnalyzer,           # 数据诊断器
        Scheduler              # 调度器
    )

    # 查看配置摘要
    print(MiyunParams.get_summary())

    # 运行数据完备性检查
    GapAnalyzer.print_report()

    # 生成调度决策
    decision = Scheduler.generate_schedule(target_flow=15.0)
    Scheduler.print_schedule(decision)
"""

__version__ = "1.1.0"
__author__ = "YCJL Development Team"
__project__ = "密云水库调蓄工程智能输水系统"

# ==========================================
# 配置数据库
# ==========================================
from .config_database import (
    # 枚举类型
    RouteType,

    # 泵站数据库
    STATION_DB,

    # 配置类
    MiyunProjectConfig,
    MiyunGlobalPhysicsConfig,
    MiyunCharacteristicCurves,
    MiyunReservoirConfig,
    JingMiChannelConfig,
    MiyunPipelineConfig,
    MiyunSafetyConfig,
    MiyunControlConfig,

    # 配置实例
    MiyunParams,
    MiyunGlobalConfig,
    MiyunCurveDatabase,
    MiyunReservoirCfg,
    MiyunChannelCfg,
    MiyunPipelineCfg,
    MiyunSafetyCfg,
    MiyunControlCfg,

    # 工具函数
    create_interpolator
)

# ==========================================
# 物理仿真引擎
# ==========================================
from .physics_engine import (
    # 枚举类型
    SystemStatus,
    PumpStatus,

    # 数据类
    HeadCalculationResult,
    SystemDiagnosisResult,

    # 引擎类
    MiyunSimulationEngine,

    # 引擎实例
    SimEngine
)

# ==========================================
# 数据完备性诊断器
# ==========================================
from .gap_analyzer import (
    # 枚举类型
    DataReadinessLevel,
    DataPriority,

    # 数据类
    MissingDataItem,
    DataGapReport,

    # 诊断器类
    MiyunDataGapAnalyzer,

    # 诊断器实例
    GapAnalyzer,

    # 便捷函数
    analyze_data_readiness,
    print_data_report
)

# ==========================================
# L5级自主调度器
# ==========================================
from .scheduler import (
    # 枚举类型
    ScheduleMode,
    OperationZone,

    # 数据类
    PumpScheduleItem,
    ScheduleDecision,

    # 调度器类
    MiyunDigitalTwinScheduler,

    # 调度器实例
    Scheduler,

    # 便捷函数
    run_diagnosis,
    generate_schedule,
    run_simulation
)

# ==========================================
# 场景库 [NEW v1.1]
# ==========================================
from .scenarios import (
    # 枚举类型
    ScenarioType,
    ScenarioSeverity,
    ResponsePriority,

    # 数据类
    ScenarioDefinition,
    ScenarioEvent,

    # 场景数据库
    MIYUN_SCENARIO_DATABASE,

    # 检测器类
    MiyunScenarioDetector,

    # 检测器实例
    ScenarioDetector,

    # 便捷函数
    get_all_scenarios,
    get_scenario_count,
    print_scenario_summary
)


# ==========================================
# 模块级便捷函数
# ==========================================
def get_version() -> str:
    """获取模块版本"""
    return __version__


def get_project_info() -> dict:
    """获取项目信息"""
    return {
        "version": __version__,
        "project": __project__,
        "config_version": MiyunParams.VERSION,
        "station_count": len(STATION_DB),
        "summary": MiyunParams.get_summary()
    }


def validate_config() -> list:
    """验证配置完整性"""
    return MiyunParams.validate()


# ==========================================
# 导出列表
# ==========================================
__all__ = [
    # 版本信息
    '__version__',
    '__author__',
    '__project__',

    # 配置数据库
    'RouteType',
    'STATION_DB',
    'MiyunProjectConfig',
    'MiyunGlobalPhysicsConfig',
    'MiyunCharacteristicCurves',
    'MiyunReservoirConfig',
    'JingMiChannelConfig',
    'MiyunPipelineConfig',
    'MiyunSafetyConfig',
    'MiyunControlConfig',
    'MiyunParams',
    'MiyunGlobalConfig',
    'MiyunCurveDatabase',
    'MiyunReservoirCfg',
    'MiyunChannelCfg',
    'MiyunPipelineCfg',
    'MiyunSafetyCfg',
    'MiyunControlCfg',
    'create_interpolator',

    # 物理仿真引擎
    'SystemStatus',
    'PumpStatus',
    'HeadCalculationResult',
    'SystemDiagnosisResult',
    'MiyunSimulationEngine',
    'SimEngine',

    # 数据完备性诊断器
    'DataReadinessLevel',
    'DataPriority',
    'MissingDataItem',
    'DataGapReport',
    'MiyunDataGapAnalyzer',
    'GapAnalyzer',
    'analyze_data_readiness',
    'print_data_report',

    # L5级自主调度器
    'ScheduleMode',
    'OperationZone',
    'PumpScheduleItem',
    'ScheduleDecision',
    'MiyunDigitalTwinScheduler',
    'Scheduler',
    'run_diagnosis',
    'generate_schedule',
    'run_simulation',

    # 模块级函数
    'get_version',
    'get_project_info',
    'validate_config',

    # 场景库 [NEW v1.1]
    'ScenarioType',
    'ScenarioSeverity',
    'ResponsePriority',
    'ScenarioDefinition',
    'ScenarioEvent',
    'MIYUN_SCENARIO_DATABASE',
    'MiyunScenarioDetector',
    'ScenarioDetector',
    'get_all_scenarios',
    'get_scenario_count',
    'print_scenario_summary'
]
