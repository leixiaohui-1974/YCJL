"""
控制算法模块
============

经典控制与先进控制算法:
- PID控制器 (带抗积分饱和)
- 自适应控制
- 串级控制
- 前馈补偿
- 史密斯预估器
"""

from .pid import PIDController, CascadePID, PIDAutotuner
from .adaptive import AdaptiveController, MRACController, STRController
from .feedforward import FeedforwardCompensator, SmithPredictor
from .coordinator import ControlCoordinator, ControlMode

__all__ = [
    'PIDController',
    'CascadePID',
    'PIDAutotuner',
    'AdaptiveController',
    'MRACController',
    'STRController',
    'FeedforwardCompensator',
    'SmithPredictor',
    'ControlCoordinator',
    'ControlMode'
]
