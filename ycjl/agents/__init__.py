"""
L5级自主运行智能体模块
======================

实现引绰济辽工程L5级全自主运行能力:
- 态势感知智能体
- 决策规划智能体
- 执行控制智能体
- 协调管理智能体
- 学习优化智能体

泵站群多智能体系统 (v1.0):
- 泵站群安全智能体 (L1层)
- 泵站群经济优化智能体 (L3层)
- 泵站群协调智能体 (融合安全与经济)

L5级定义: 全工况全自主,无需人工干预

版本: 3.8.0
"""

from .l5_autonomous import (
    # 智能等级
    AutonomyLevel,
    AgentRole,
    AgentStatus,
    # 智能体
    SituationAwarenessAgent,
    DecisionPlanningAgent,
    ExecutionControlAgent,
    CoordinationAgent,
    LearningAgent,
    # L5系统
    L5AutonomousSystem,
    L5SystemState,
    L5Decision,
    # 工厂函数
    create_l5_system
)

from .emergency_agent import (
    EmergencyLevel,
    EmergencyType,
    EmergencyState,
    EmergencyResponse,
    EmergencyAgent,
    create_emergency_agent
)

from .fault_tolerance import (
    FaultType,
    FaultState,
    RedundancyMode,
    FaultToleranceManager,
    ComponentHealth,
    SystemResilience
)

# 泵站群多智能体系统 (v1.0)
from .pump_group_agents import (
    # 数据结构
    PumpStatus,
    ProtectionType,
    PumpState,
    StationState,
    PumpGroupState,
    PumpScheduleAction,
    # 智能体
    PumpGroupSafetyAgent,
    PumpGroupEconomicAgent,
    PumpGroupCoordinatorAgent,
    PumpGroupMultiAgentSystem,
    # 便捷函数
    create_pump_group_system,
)

__all__ = [
    # 智能等级
    'AutonomyLevel',
    'AgentRole',
    'AgentStatus',
    # 智能体
    'SituationAwarenessAgent',
    'DecisionPlanningAgent',
    'ExecutionControlAgent',
    'CoordinationAgent',
    'LearningAgent',
    # L5系统
    'L5AutonomousSystem',
    'L5SystemState',
    'L5Decision',
    'create_l5_system',
    # 应急
    'EmergencyLevel',
    'EmergencyType',
    'EmergencyState',
    'EmergencyResponse',
    'EmergencyAgent',
    'create_emergency_agent',
    # 容错
    'FaultType',
    'FaultState',
    'RedundancyMode',
    'FaultToleranceManager',
    'ComponentHealth',
    'SystemResilience',
    # 泵站群多智能体系统
    'PumpStatus',
    'ProtectionType',
    'PumpState',
    'StationState',
    'PumpGroupState',
    'PumpScheduleAction',
    'PumpGroupSafetyAgent',
    'PumpGroupEconomicAgent',
    'PumpGroupCoordinatorAgent',
    'PumpGroupMultiAgentSystem',
    'create_pump_group_system',
]
