"""
控制算法模块
============

经典控制与先进控制算法:
- PID控制器 (带抗积分饱和)
- 自适应控制
- 串级控制
- 前馈补偿
- 史密斯预估器
- 兴利调度 (调度图、供水调度、防洪调度)
"""

from .pid import PIDController, CascadePID, PIDAutotuner
from .adaptive import AdaptiveController, MRACController, STRController
from .feedforward import FeedforwardCompensator, SmithPredictor
from .coordinator import ControlCoordinator, ControlMode
from .scheduler import (
    OperationZone,
    SupplyMode,
    FloodControlLevel,
    ScheduleDecision,
    FloodForecast,
    OperationRuleChart,
    SupplyScheduler,
    FloodDispatcher,
    ReservoirScheduler,
    FloodHydrograph
)

__all__ = [
    # 经典控制
    'PIDController',
    'CascadePID',
    'PIDAutotuner',
    'AdaptiveController',
    'MRACController',
    'STRController',
    'FeedforwardCompensator',
    'SmithPredictor',
    'ControlCoordinator',
    'ControlMode',
    # 调度模块
    'OperationZone',
    'SupplyMode',
    'FloodControlLevel',
    'ScheduleDecision',
    'FloodForecast',
    'OperationRuleChart',
    'SupplyScheduler',
    'FloodDispatcher',
    'ReservoirScheduler',
    'FloodHydrograph'
]
