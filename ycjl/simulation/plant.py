"""
水利系统工厂模型
================

集成所有物理组件的完整系统仿真
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto

from ..config.settings import Config, ScenarioType, SeasonMode
from ..physics.reservoir import Reservoir, ReservoirState
from ..physics.tunnel import Tunnel, TunnelState
from ..physics.pool import StabilizingPool, PoolState
from ..physics.pipeline import Pipeline, PipelineState
from ..physics.surge_tank import SurgeTank, SurgeTankState
from ..sensors.level_sensor import LevelSensor
from ..sensors.pressure_sensor import PressureSensor
from ..sensors.flow_sensor import FlowSensor
from ..actuators.gate import Gate
from ..actuators.valve import Valve


class PlantMode(Enum):
    """系统运行模式"""
    STARTUP = auto()
    NORMAL = auto()
    EMERGENCY = auto()
    SHUTDOWN = auto()
    MAINTENANCE = auto()


@dataclass
class PlantState:
    """系统状态"""
    timestamp: float
    mode: PlantMode

    # 物理状态
    reservoir: ReservoirState
    tunnel: TunnelState
    pool: PoolState
    pipeline: PipelineState
    surge_tank: SurgeTankState

    # 测量值
    measurements: Dict[str, float]

    # 控制输入
    control_inputs: Dict[str, float]

    # 综合指标
    total_flow: float
    total_head_loss: float
    efficiency: float


class WaterTransferPlant:
    """
    引水工程完整系统仿真

    包含:
    - 文得根水库
    - 输水隧洞
    - 稳流连接池
    - 调压井
    - PCCP压力管道
    - 传感器系统
    - 执行器系统
    """

    def __init__(self, config: Config = None):
        self.cfg = config or Config

        # 物理组件
        self.reservoir = Reservoir()
        self.tunnel = Tunnel()
        self.pool = StabilizingPool()
        self.surge_tank = SurgeTank()
        self.pipeline = Pipeline(num_sections=100)

        # 传感器
        self.sensors = self._init_sensors()

        # 执行器
        self.actuators = self._init_actuators()

        # 系统状态
        self.mode = PlantMode.NORMAL
        self.time = 0.0
        self.dt = 1.0  # 默认时间步长

        # 历史记录
        self.history: List[PlantState] = []
        self.max_history = 10000

        # 外部输入
        self.inflow = 10.0  # 入库流量
        self.demand = 10.0  # 末端需求

        # 场景模式
        self.scenario = ScenarioType.NORMAL
        self.season = SeasonMode.NORMAL

    def _init_sensors(self) -> Dict[str, object]:
        """初始化传感器"""
        return {
            'reservoir_level': LevelSensor(
                sensor_id='reservoir_level',
                noise_std=0.01,
                range_min=0, range_max=150
            ),
            'pool_level': LevelSensor(
                sensor_id='pool_level',
                noise_std=0.01,
                range_min=0, range_max=15
            ),
            'surge_tank_level': LevelSensor(
                sensor_id='surge_tank_level',
                noise_std=0.02,
                range_min=0, range_max=50
            ),
            'tunnel_flow': FlowSensor(
                sensor_id='tunnel_flow',
                noise_std=0.1,
                range_min=0, range_max=30
            ),
            'pipe_flow': FlowSensor(
                sensor_id='pipe_flow',
                noise_std=0.1,
                range_min=0, range_max=20
            ),
            'pipe_pressure_up': PressureSensor(
                sensor_id='pipe_pressure_up',
                noise_std=0.5,
                range_min=0, range_max=150
            ),
            'pipe_pressure_down': PressureSensor(
                sensor_id='pipe_pressure_down',
                noise_std=0.5,
                range_min=0, range_max=100
            )
        }

    def _init_actuators(self) -> Dict[str, object]:
        """初始化执行器"""
        return {
            'gate_intake': Gate(
                gate_id='gate_intake',
                max_opening=5.0,
                rate_limit=0.001
            ),
            'gate_pool_out': Gate(
                gate_id='gate_pool_out',
                max_opening=3.0,
                rate_limit=0.002
            ),
            'valve_mid': Valve(
                valve_id='valve_mid',
                Cv=1000,
                rate_limit=0.01
            ),
            'valve_end': Valve(
                valve_id='valve_end',
                Cv=800,
                rate_limit=0.01
            ),
            'valve_relief': Valve(
                valve_id='valve_relief',
                Cv=500,
                rate_limit=0.1
            ),
            'valve_air': Valve(
                valve_id='valve_air',
                Cv=100,
                rate_limit=0.1
            )
        }

    def set_mode(self, mode: PlantMode):
        """设置运行模式"""
        self.mode = mode

    def set_scenario(self, scenario: ScenarioType):
        """设置场景"""
        self.scenario = scenario

    def set_season(self, season: SeasonMode):
        """设置季节模式"""
        self.season = season

        # 根据季节调整参数
        if season == SeasonMode.ICE:
            self.tunnel.set_ice_mode(True, 0.5)
            self.pool.set_ice_mode(True, 0.3)

    def apply_control(self, control_inputs: Dict[str, float]):
        """应用控制输入"""
        for actuator_id, target in control_inputs.items():
            if actuator_id in self.actuators:
                actuator = self.actuators[actuator_id]
                actuator.set_target(target)

    def get_measurements(self) -> Dict[str, float]:
        """获取传感器测量值"""
        measurements = {}

        # 读取各传感器
        measurements['reservoir_level'] = self.sensors['reservoir_level'].read(
            self.reservoir.level
        )
        measurements['pool_level'] = self.sensors['pool_level'].read(
            self.pool.level
        )
        measurements['surge_tank_level'] = self.sensors['surge_tank_level'].read(
            self.surge_tank.level
        )
        measurements['tunnel_flow'] = self.sensors['tunnel_flow'].read(
            self.tunnel.flow_out
        )
        measurements['pipe_flow'] = self.sensors['pipe_flow'].read(
            self.pipeline.flow_out
        )
        measurements['pipe_pressure_up'] = self.sensors['pipe_pressure_up'].read(
            self.pipeline.H[0]
        )
        measurements['pipe_pressure_down'] = self.sensors['pipe_pressure_down'].read(
            self.pipeline.H[-1]
        )

        return measurements

    def _compute_coupling(self):
        """计算组件间耦合"""
        # 水库 -> 隧洞
        self.tunnel.upstream_head = self.reservoir.level

        # 隧洞 -> 稳流池
        self.pool.inflow = self.tunnel.flow_out

        # 稳流池 -> 调压井
        self.surge_tank.inflow = self.pool.outflow

        # 调压井 -> 管道
        self.pipeline.H[0] = self.surge_tank.level + 50  # 基准高程

        # 管道末端 -> 需求
        self.pipeline.Q_out = self.demand

    def step(self, dt: float = None, control_inputs: Dict[str, float] = None) -> PlantState:
        """
        推进一个时间步

        Parameters:
            dt: 时间步长
            control_inputs: 控制输入

        Returns:
            PlantState: 当前状态
        """
        if dt is None:
            dt = self.dt

        # 应用控制
        if control_inputs:
            self.apply_control(control_inputs)

        # 更新执行器
        actuator_states = {}
        for actuator_id, actuator in self.actuators.items():
            actuator.step(dt)
            actuator_states[actuator_id] = actuator.get_opening()

        # 计算耦合
        self._compute_coupling()

        # 更新物理组件
        reservoir_state = self.reservoir.step(dt, self.inflow)

        # 隧洞入口由进水闸控制
        intake_opening = actuator_states.get('gate_intake', 0.5)
        self.tunnel.upstream_control = intake_opening
        tunnel_state = self.tunnel.step(dt)

        # 稳流池出口由出水闸控制
        pool_out_opening = actuator_states.get('gate_pool_out', 0.5)
        self.pool.outflow = self.tunnel.flow_out * pool_out_opening
        pool_state = self.pool.step(dt, self.tunnel.flow_out)

        # 调压井
        surge_state = self.surge_tank.step(dt, self.pool.outflow, self.pipeline.Q[0])

        # 管道边界条件
        valve_mid = actuator_states.get('valve_mid', 0.7)
        valve_end = actuator_states.get('valve_end', 0.8)

        # 管道仿真
        self.pipeline.valve_opening_mid = valve_mid
        self.pipeline.valve_opening_end = valve_end
        pipeline_state = self.pipeline.step(dt)

        # 获取测量值
        measurements = self.get_measurements()

        # 计算综合指标
        total_flow = self.pipeline.Q[-1] if len(self.pipeline.Q) > 0 else 0
        total_head_loss = (self.reservoir.level - self.pipeline.H[-1]) if len(self.pipeline.H) > 0 else 0
        efficiency = total_flow / max(self.inflow, 0.1)

        # 构建状态
        state = PlantState(
            timestamp=self.time,
            mode=self.mode,
            reservoir=reservoir_state,
            tunnel=tunnel_state,
            pool=pool_state,
            pipeline=pipeline_state,
            surge_tank=surge_state,
            measurements=measurements,
            control_inputs=actuator_states,
            total_flow=total_flow,
            total_head_loss=total_head_loss,
            efficiency=efficiency
        )

        # 记录历史
        self.history.append(state)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        self.time += dt

        return state

    def get_state_vector(self) -> np.ndarray:
        """获取状态向量（用于控制器）"""
        return np.array([
            self.reservoir.level,
            self.pool.level,
            self.surge_tank.level,
            self.tunnel.flow_out,
            self.pipeline.Q[-1] if len(self.pipeline.Q) > 0 else 0,
            self.pipeline.H[0] if len(self.pipeline.H) > 0 else 50,
            self.pipeline.H[-1] if len(self.pipeline.H) > 0 else 30
        ])

    def get_state_dict(self) -> Dict[str, float]:
        """获取状态字典"""
        measurements = self.get_measurements()

        state_dict = {
            'reservoir_level': self.reservoir.level,
            'pool_level': self.pool.level,
            'surge_tank_level': self.surge_tank.level,
            'tunnel_flow': self.tunnel.flow_out,
            'pipe_flow': self.pipeline.Q[-1] if len(self.pipeline.Q) > 0 else 0,
            'pipe_pressure': self.pipeline.H[0] if len(self.pipeline.H) > 0 else 50,
            'total_flow': self.pipeline.Q[-1] if len(self.pipeline.Q) > 0 else 0,
            'demand': self.demand,
            'inflow': self.inflow,
            'scenario': self.scenario,
            'season': self.season,
            'mode': self.mode
        }

        # 添加测量值
        state_dict.update(measurements)

        # 添加执行器状态
        for actuator_id, actuator in self.actuators.items():
            state_dict[f'{actuator_id}_opening'] = actuator.get_opening()

        return state_dict

    def inject_fault(self, fault_type: str, location: float = 0.5, severity: float = 1.0):
        """注入故障"""
        if fault_type == 'leak':
            # 管道泄漏
            leak_index = int(location * len(self.pipeline.Q))
            self.pipeline.inject_leak(leak_index, severity * 0.1)

        elif fault_type == 'blockage':
            # 管道堵塞
            block_index = int(location * len(self.pipeline.sections))
            # 增加局部阻力
            self.pipeline.sections[block_index].friction *= (1 + severity)

        elif fault_type == 'sensor_fault':
            # 传感器故障
            for sensor in self.sensors.values():
                sensor.inject_fault('bias', severity * 0.1)

    def clear_faults(self):
        """清除故障"""
        self.pipeline.clear_leak()
        for sensor in self.sensors.values():
            sensor.clear_fault()

    def reset(self, initial_state: Dict[str, float] = None):
        """重置系统"""
        self.reservoir.reset()
        self.tunnel.reset()
        self.pool.reset()
        self.surge_tank.reset()
        self.pipeline.reset()

        for sensor in self.sensors.values():
            sensor.reset()

        for actuator in self.actuators.values():
            actuator.reset()

        self.time = 0.0
        self.mode = PlantMode.NORMAL
        self.scenario = ScenarioType.NORMAL
        self.history.clear()

        if initial_state:
            if 'reservoir_level' in initial_state:
                self.reservoir.level = initial_state['reservoir_level']
            if 'pool_level' in initial_state:
                self.pool.level = initial_state['pool_level']
