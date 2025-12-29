"""
物理模型模块
============

包含所有水工建筑物的高保真物理仿真模型：
- Reservoir: 水库模型（库容特性、溢洪道、进水口）
- ReservoirV2: 增强水库模型（使用配置数据库v3.2）
- Tunnel: 无压隧洞模型（圣维南方程）
- Pool: 稳流连接池模型（积分环节）
- SurgeTank: 调压塔模型（阻抗式）
- Pipeline: PCCP管道模型（特征线法MOC）
- Siphon: 倒虹吸模型
- Valves: 各类阀门模型
- ValvesV2: 增强阀门模型（数字化流阻特性）
- Turbine: 水轮机模型（Hill Chart效率）
"""

from .reservoir import Reservoir
from .reservoir_v2 import WendegenReservoir, ReservoirStateV2, SpillwayGateState
from .tunnel import TunnelSolver, Tunnel, TunnelState
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
from .valves_v2 import (
    InlineRegulatingValve,
    EndRegulatingValve,
    ButterflyValveV2,
    ReliefValveV2,
    AirValveV2,
    ValveSystem,
    ValveState,
    ValveFaultType
)
from .turbine import (
    Turbine,
    TurbineState,
    TurbineType,
    TurbineSpec,
    PowerStation,
    HillChartInterpolator
)
from .ice_model import (
    IceState,
    ChannelGeometry,
    MeteoCondition,
    IceThicknessModel,
    IceCoverHydraulics,
    FrazilIceModel,
    IceCoverFormation,
    BreakupModel,
    IcePeriodSimulator
)

__all__ = [
    # 原有模型
    'Reservoir',
    'TunnelSolver',
    'Tunnel',  # 向后兼容别名
    'TunnelState',
    'StabilizingPool',
    'SurgeTank',
    'PipelineMOC',
    'Siphon',
    'RadialGate',
    'PlungerValve',
    'ButterflyValve',
    'ReliefValve',
    'AirValve',
    # 增强模型 V2
    'WendegenReservoir',
    'ReservoirStateV2',
    'SpillwayGateState',
    'InlineRegulatingValve',
    'EndRegulatingValve',
    'ButterflyValveV2',
    'ReliefValveV2',
    'AirValveV2',
    'ValveSystem',
    'ValveState',
    'ValveFaultType',
    # 水轮机模型
    'Turbine',
    'TurbineState',
    'TurbineType',
    'TurbineSpec',
    'PowerStation',
    'HillChartInterpolator',
    # 冰期模型
    'IceState',
    'ChannelGeometry',
    'MeteoCondition',
    'IceThicknessModel',
    'IceCoverHydraulics',
    'FrazilIceModel',
    'IceCoverFormation',
    'BreakupModel',
    'IcePeriodSimulator'
]
