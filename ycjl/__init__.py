"""
引绰济辽智能输水系统 (YCJL Intelligent Water Transfer System)
============================================================

产品化重构版本：包含完整的水动力学仿真、多层级智能体控制、
场景识别和全自主运行能力。

模块结构:
- config: 全局配置参数
- physics: 物理模型（水库、隧洞、管道等）
- sensors: 传感器仿真
- actuators: 执行器仿真
- estimation: 参数辨识与状态同化
- models: 降阶模型与数字孪生
- agents: 多层级智能体（L1/L2/L3）
- control: 控制算法（PID/MPC/自适应）
- scenarios: 场景识别与处理
- simulation: 仿真引擎
- tests: 测试套件
"""

__version__ = "2.0.0"
__author__ = "YCJL Control Team"

from .config.settings import Config, ScenarioType, SeasonMode
from .simulation.plant import WaterTransferPlant, PlantState
from .simulation.runner import SimulationRunner, SimulationConfig, run_scenario_test
from .agents.coordinator import MultiAgentSystem
from .scenarios.detector import ScenarioDetector
from .models.digital_twin import DigitalTwin

__all__ = [
    'Config',
    'ScenarioType',
    'SeasonMode',
    'WaterTransferPlant',
    'PlantState',
    'SimulationRunner',
    'SimulationConfig',
    'run_scenario_test',
    'MultiAgentSystem',
    'ScenarioDetector',
    'DigitalTwin'
]
