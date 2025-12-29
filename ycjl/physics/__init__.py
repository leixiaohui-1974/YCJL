"""
物理模型模块
============

包含所有水工建筑物的高保真物理仿真模型：
- Reservoir: 水库模型（库容特性、溢洪道、进水口）
- Tunnel: 无压隧洞模型（圣维南方程）
- Pool: 稳流连接池模型（积分环节）
- SurgeTank: 调压塔模型（阻抗式）
- Pipeline: PCCP管道模型（特征线法MOC）
- Siphon: 倒虹吸模型
- Valves: 各类阀门模型
"""

from .reservoir import Reservoir
from .tunnel import TunnelSolver
from .pool import StabilizingPool
from .surge_tank import SurgeTank
from .pipeline import PipelineMOC
from .siphon import Siphon
from .valves import (
    RadialGate,
    PlungerValve,
    ButterflyValve,
    ReliefValve,
    AirValve
)

__all__ = [
    'Reservoir',
    'TunnelSolver',
    'StabilizingPool',
    'SurgeTank',
    'PipelineMOC',
    'Siphon',
    'RadialGate',
    'PlungerValve',
    'ButterflyValve',
    'ReliefValve',
    'AirValve'
]
