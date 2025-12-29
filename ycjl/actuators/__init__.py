"""
执行器仿真模块
==============

仿真各类执行器的物理特性:
- 动作速率限制
- 机械死区
- 故障模式
- 位置反馈
"""

from .gate import GateActuator
from .valve import ValveActuator
from .pump import PumpActuator

__all__ = [
    'GateActuator',
    'ValveActuator',
    'PumpActuator'
]
