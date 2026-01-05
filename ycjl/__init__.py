"""
引绰济辽智能输水系统 (YCJL Intelligent Water Transfer System)
============================================================

产品化重构版本：包含完整的水动力学仿真、多层级智能体控制、
场景识别和全自主运行能力。

版本: 3.8.0 - ODD设计运行域分析模块
- 完整工程参数数据库 (config_database)
- 数字化特性曲线 (Hill Chart, 流阻曲线等)
- 水轮机电站模型
- 兴利调度逻辑
- 工程部署接口
- 冰期水力学参数库 (IAHR/ASCE标准)
- 冰期物理模型 (Stefan方程, Belokon-Sabaneev复合糙率)
- 冰期控制策略 (运行模式识别, 约束管理, 安全保护)
- 全工况场景数据库 (83种场景覆盖)
- L5级自主运行多智能体系统
- 应急响应智能体
- 故障容错系统
- 密云水库调蓄工程模块 (miyun)
- 增强调度器 - 场景感知、数据诊断、健康评估
- 统一调度接口 IEnhancedScheduler
- [NEW] ODD设计运行域分析器 - 四维度边界监测、可观性/可控性评估、自主等级判定

模块结构:
- config: 全局配置参数与工程数据库
- physics: 物理模型（水库、隧洞、管道、水轮机、冰期等）
- sensors: 传感器仿真
- actuators: 执行器仿真
- estimation: 参数辨识与状态同化
- models: 降阶模型与数字孪生
- agents: L5自主运行智能体（态势感知/决策规划/执行控制/协调管理/学习优化）
- control: 控制算法（PID/MPC/自适应/调度/冰期控制）
- scenarios: 全工况场景识别与处理
- simulation: 仿真引擎
- deployment: 工程部署接口
- miyun: 密云水库调蓄工程（泵站数据库、物理仿真、数据诊断、调度器、场景库）
- core: 通用核心框架（插值器、物理基类、仿真基类、调度基类、数据诊断基类、ODD分析器）
"""

__version__ = "3.8.0"
__author__ = "YCJL Control Team"

# 原有配置
from .config.settings import Config, ScenarioType, SeasonMode

# 新配置数据库
from .config.config_database import (
    YinChuoProjectConfig,
    ProjectParams,
    CurveDatabase,
    SourceConfig,
    GlobalConfig
)

# 冰期参数
from .config.ice_parameters import (
    IceParams,
    IceType,
    IcePhase,
    BreakupType,
    IceHydraulicsConfig
)

# 部署接口
from .deployment import (
    DeploymentManager,
    DeploymentEnvironment,
    ConfigValidator,
    SCADAInterface,
    create_production_deployment,
    create_testing_deployment
)

# 增强物理模型 - 使用延迟导入避免循环依赖
def get_wendegen_reservoir():
    """获取增强水库模型"""
    from .physics.reservoir_v2 import WendegenReservoir
    return WendegenReservoir

def get_power_station():
    """获取电站模型"""
    from .physics.turbine import PowerStation
    return PowerStation

def get_valve_system():
    """获取阀门系统"""
    from .physics.valves_v2 import ValveSystem
    return ValveSystem

def get_reservoir_scheduler():
    """获取调度器"""
    from .control.scheduler import ReservoirScheduler
    return ReservoirScheduler

def get_flood_hydrograph():
    """获取洪水过程生成器"""
    from .control.scheduler import FloodHydrograph
    return FloodHydrograph


# ==========================================
# 密云水库调蓄工程模块
# ==========================================
def get_miyun_scheduler():
    """获取密云水库调度器"""
    from .miyun import Scheduler
    return Scheduler


def get_miyun_simulation_engine():
    """获取密云水库仿真引擎"""
    from .miyun import SimEngine
    return SimEngine


def get_miyun_config():
    """获取密云水库配置"""
    from .miyun import MiyunParams
    return MiyunParams


def run_miyun_simulation(flow: float = 10.0):
    """运行密云水库仿真"""
    from .miyun import run_simulation
    return run_simulation(flow)


__all__ = [
    # 版本信息
    '__version__',
    # 配置
    'Config',
    'ScenarioType',
    'SeasonMode',
    'YinChuoProjectConfig',
    'ProjectParams',
    'CurveDatabase',
    'SourceConfig',
    'GlobalConfig',
    # 冰期参数
    'IceParams',
    'IceType',
    'IcePhase',
    'BreakupType',
    'IceHydraulicsConfig',
    # 部署
    'DeploymentManager',
    'DeploymentEnvironment',
    'ConfigValidator',
    'SCADAInterface',
    'create_production_deployment',
    'create_testing_deployment',
    # 工厂函数
    'get_wendegen_reservoir',
    'get_power_station',
    'get_valve_system',
    'get_reservoir_scheduler',
    'get_flood_hydrograph',
    # 密云水库调蓄工程
    'get_miyun_scheduler',
    'get_miyun_simulation_engine',
    'get_miyun_config',
    'run_miyun_simulation'
]
