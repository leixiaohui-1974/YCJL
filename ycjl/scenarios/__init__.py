"""
全工况场景模块
==============

覆盖引绰济辽工程所有运行场景和工况:
- 正常运行场景
- 启停与过渡场景
- 极端工况场景
- 故障与事故场景
- 应急响应场景
- 维护检修场景

版本: 3.4.0 - 全工况L5自主运行版
"""

from .scenario_database import (
    # 场景分类
    ScenarioCategory,
    ScenarioSeverity,
    ScenarioPhase,
    OperationMode,
    # 场景定义
    ScenarioType,
    ScenarioDefinition,
    ScenarioTrigger,
    ScenarioResponse,
    # 场景数据库
    ScenarioDatabase,
    SCENARIO_DB,
    # 便捷访问
    get_scenario,
    get_scenarios_by_category,
    get_response_strategy
)

from .scenario_engine import (
    ScenarioState,
    ScenarioEvent,
    ScenarioDetector,
    ScenarioClassifier,
    ScenarioEngine
)

__all__ = [
    # 场景分类
    'ScenarioCategory',
    'ScenarioSeverity',
    'ScenarioPhase',
    'OperationMode',
    # 场景定义
    'ScenarioType',
    'ScenarioDefinition',
    'ScenarioTrigger',
    'ScenarioResponse',
    # 场景数据库
    'ScenarioDatabase',
    'SCENARIO_DB',
    'get_scenario',
    'get_scenarios_by_category',
    'get_response_strategy',
    # 场景引擎
    'ScenarioState',
    'ScenarioEvent',
    'ScenarioDetector',
    'ScenarioClassifier',
    'ScenarioEngine'
]
