"""
多层级智能体系统
================

三层分布式控制架构:
- L1 Reflex Agents: 毫秒级安全响应
- L2 Tactical Agents: 基于MPC的管段优化
- L3 Strategic Agent: 多目标经济调度

协同算法:
- ADMM分布式优化
- 事件触发通信
"""

from .base_agent import BaseAgent, AgentState, AgentMessage
from .reflex_agent import ReflexAgent, SafetyRule
from .tactical_agent import TacticalAgent, MPCController
from .strategic_agent import StrategicAgent
from .coordinator import ADMMCoordinator, CommunicationHub

__all__ = [
    'BaseAgent',
    'AgentState',
    'AgentMessage',
    'ReflexAgent',
    'SafetyRule',
    'TacticalAgent',
    'MPCController',
    'StrategicAgent',
    'ADMMCoordinator',
    'CommunicationHub'
]
