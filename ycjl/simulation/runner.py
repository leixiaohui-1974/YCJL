"""
仿真运行器
==========

完整系统仿真运行
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import time

from ..config.settings import Config, ScenarioType
from .plant import WaterTransferPlant, PlantState, PlantMode
from .scenario_injector import ScenarioInjector, InjectionEvent
from ..agents.coordinator import MultiAgentSystem
from ..scenarios.detector import ScenarioDetector
from ..models.digital_twin import DigitalTwin


@dataclass
class SimulationConfig:
    """仿真配置"""
    duration: float = 3600.0    # 仿真时长 (s)
    dt: float = 1.0             # 时间步长 (s)

    # 控制
    enable_control: bool = True
    control_interval: float = 1.0

    # 场景检测
    enable_detection: bool = True
    detection_interval: float = 5.0

    # 数字孪生
    enable_twin: bool = True
    twin_interval: float = 1.0

    # 日志
    log_interval: float = 10.0
    verbose: bool = True


@dataclass
class SimulationResult:
    """仿真结果"""
    success: bool
    duration: float
    steps: int

    # 时间序列
    time_series: np.ndarray
    states: List[PlantState]

    # 场景统计
    scenario_history: List[Tuple[float, ScenarioType]]
    scenario_durations: Dict[ScenarioType, float]

    # 性能指标
    metrics: Dict[str, float]

    # 控制统计
    control_actions: int
    safety_interventions: int

    # 错误
    errors: List[str]


class SimulationRunner:
    """
    仿真运行器

    集成:
    - 物理系统仿真
    - 多智能体控制
    - 场景检测
    - 数字孪生
    """

    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()

        # 系统组件
        self.plant = WaterTransferPlant()
        self.injector = ScenarioInjector()
        self.detector = ScenarioDetector()
        self.mas = MultiAgentSystem()
        self.twin = DigitalTwin()

        # 状态
        self.current_time = 0.0
        self.step_count = 0

        # 记录
        self.states: List[PlantState] = []
        self.scenario_history: List[Tuple[float, ScenarioType]] = []
        self.errors: List[str] = []

        # 统计
        self.control_actions = 0
        self.safety_interventions = 0

        # 回调
        self.callbacks: Dict[str, Callable] = {}

    def register_callback(self, event: str, callback: Callable):
        """注册回调"""
        self.callbacks[event] = callback

    def run(self, initial_state: Dict[str, float] = None,
            scenario_events: List[InjectionEvent] = None) -> SimulationResult:
        """
        运行仿真

        Parameters:
            initial_state: 初始状态
            scenario_events: 场景事件列表

        Returns:
            SimulationResult: 仿真结果
        """
        start_time = time.time()

        # 初始化
        self._initialize(initial_state, scenario_events)

        # 主循环
        try:
            while self.current_time < self.config.duration:
                self._step()

                # 检查终止条件
                if self._check_termination():
                    break

        except Exception as e:
            self.errors.append(f"Simulation error at t={self.current_time}: {str(e)}")

        # 生成结果
        result = self._generate_result(time.time() - start_time)

        return result

    def _initialize(self, initial_state: Dict, scenario_events: List):
        """初始化仿真"""
        # 重置组件
        self.plant.reset(initial_state)
        self.injector.reset()
        self.detector.reset()
        self.mas.reset()
        self.twin.reset()

        # 重置状态
        self.current_time = 0.0
        self.step_count = 0
        self.states.clear()
        self.scenario_history.clear()
        self.errors.clear()
        self.control_actions = 0
        self.safety_interventions = 0

        # 调度场景事件
        if scenario_events:
            for event in scenario_events:
                self.injector.schedule_event(event)

        # 记录初始场景
        self.scenario_history.append((0.0, ScenarioType.NORMAL))

        if self.config.verbose:
            print(f"Simulation initialized. Duration: {self.config.duration}s, dt: {self.config.dt}s")

    def _step(self):
        """执行一个仿真步"""
        dt = self.config.dt

        # 1. 获取当前状态
        state_dict = self.plant.get_state_dict()

        # 2. 注入场景
        state_dict = self.injector.step(self.current_time, state_dict)

        # 3. 场景检测
        if self.config.enable_detection:
            if self.step_count % int(self.config.detection_interval / dt) == 0:
                self._run_detection(state_dict)

        # 4. 多智能体控制
        control_inputs = {}
        if self.config.enable_control:
            if self.step_count % int(self.config.control_interval / dt) == 0:
                control_inputs = self._run_control(state_dict)

        # 5. 数字孪生更新
        if self.config.enable_twin:
            if self.step_count % int(self.config.twin_interval / dt) == 0:
                self._run_twin(state_dict, control_inputs)

        # 6. 物理仿真步进
        plant_state = self.plant.step(dt, control_inputs)
        self.states.append(plant_state)

        # 7. 日志
        if self.config.verbose:
            if self.step_count % int(self.config.log_interval / dt) == 0:
                self._log_state(plant_state)

        # 8. 回调
        if 'on_step' in self.callbacks:
            self.callbacks['on_step'](self.current_time, plant_state)

        self.current_time += dt
        self.step_count += 1

    def _run_detection(self, state_dict: Dict):
        """运行场景检测"""
        # 更新检测器
        self.detector.update(state_dict, self.current_time)

        # 执行检测
        detection_result = self.detector.detect()

        # 检查场景变化
        if len(self.scenario_history) == 0 or \
           detection_result.scenario != self.scenario_history[-1][1]:
            self.scenario_history.append(
                (self.current_time, detection_result.scenario)
            )

            if self.config.verbose:
                print(f"[{self.current_time:.1f}s] Scenario changed to: "
                      f"{detection_result.scenario.name} "
                      f"(confidence: {detection_result.confidence:.2f})")

    def _run_control(self, state_dict: Dict) -> Dict[str, float]:
        """运行多智能体控制"""
        # 执行MAS步进
        mas_result = self.mas.step(state_dict)

        # 统计
        actions = mas_result.get('all_actions', [])
        self.control_actions += len(actions)

        # 检查安全干预
        l1_actions = mas_result.get('L1', {}).get('actions', [])
        if l1_actions:
            self.safety_interventions += 1

        # 转换为控制输入
        control_inputs = {}
        for action in actions:
            actuator = action.get('actuator', '')
            value = action.get('value', 0)
            control_inputs[actuator] = value

        return control_inputs

    def _run_twin(self, state_dict: Dict, control_inputs: Dict):
        """运行数字孪生"""
        # 更新物理状态
        self.twin.update_physical_state(state_dict)

        # 步进
        twin_state = self.twin.step(self.config.dt, control_inputs)

        # 检测异常
        is_anomaly, anomaly_type, severity = self.twin.detect_anomaly()
        if is_anomaly and severity > 0.5:
            if self.config.verbose:
                print(f"[{self.current_time:.1f}s] Digital twin anomaly: "
                      f"{anomaly_type} (severity: {severity:.2f})")

    def _check_termination(self) -> bool:
        """检查终止条件"""
        # 检查紧急状态
        if self.plant.mode == PlantMode.EMERGENCY:
            if self.config.verbose:
                print(f"[{self.current_time:.1f}s] Emergency mode - simulation terminated")
            return True

        return False

    def _log_state(self, state: PlantState):
        """记录状态日志"""
        print(f"[{self.current_time:.1f}s] "
              f"Pool: {state.pool.level:.2f}m, "
              f"Flow: {state.total_flow:.2f}m³/s, "
              f"Scenario: {self.detector.current_scenario.name}")

    def _generate_result(self, elapsed_time: float) -> SimulationResult:
        """生成仿真结果"""
        # 时间序列
        time_series = np.arange(0, self.current_time, self.config.dt)

        # 场景持续时间
        scenario_durations = {s: 0.0 for s in ScenarioType}
        for i in range(len(self.scenario_history)):
            start_time = self.scenario_history[i][0]
            if i < len(self.scenario_history) - 1:
                end_time = self.scenario_history[i + 1][0]
            else:
                end_time = self.current_time

            scenario = self.scenario_history[i][1]
            scenario_durations[scenario] += end_time - start_time

        # 性能指标
        metrics = self._compute_metrics()

        return SimulationResult(
            success=len(self.errors) == 0,
            duration=elapsed_time,
            steps=self.step_count,
            time_series=time_series,
            states=self.states,
            scenario_history=self.scenario_history,
            scenario_durations=scenario_durations,
            metrics=metrics,
            control_actions=self.control_actions,
            safety_interventions=self.safety_interventions,
            errors=self.errors
        )

    def _compute_metrics(self) -> Dict[str, float]:
        """计算性能指标"""
        if not self.states:
            return {}

        metrics = {}

        # 水位指标
        pool_levels = [s.pool.level for s in self.states]
        metrics['pool_level_mean'] = np.mean(pool_levels)
        metrics['pool_level_std'] = np.std(pool_levels)
        metrics['pool_level_min'] = np.min(pool_levels)
        metrics['pool_level_max'] = np.max(pool_levels)

        # 流量指标
        flows = [s.total_flow for s in self.states]
        metrics['flow_mean'] = np.mean(flows)
        metrics['flow_std'] = np.std(flows)

        # 效率
        efficiencies = [s.efficiency for s in self.states]
        metrics['efficiency_mean'] = np.mean(efficiencies)

        # 控制指标
        metrics['control_actions_per_hour'] = self.control_actions / \
            (self.current_time / 3600) if self.current_time > 0 else 0
        metrics['safety_interventions_per_hour'] = self.safety_interventions / \
            (self.current_time / 3600) if self.current_time > 0 else 0

        return metrics

    def reset(self):
        """重置运行器"""
        self.plant.reset()
        self.injector.reset()
        self.detector.reset()
        self.mas.reset()
        self.twin.reset()
        self.current_time = 0.0
        self.step_count = 0
        self.states.clear()
        self.scenario_history.clear()
        self.errors.clear()


def run_scenario_test(scenario: ScenarioType, duration: float = 3600,
                      **kwargs) -> SimulationResult:
    """
    运行场景测试

    Parameters:
        scenario: 场景类型
        duration: 仿真时长
        **kwargs: 场景参数

    Returns:
        SimulationResult: 测试结果
    """
    config = SimulationConfig(
        duration=duration,
        dt=1.0,
        verbose=True
    )

    runner = SimulationRunner(config)

    # 创建场景事件
    events = []

    if scenario == ScenarioType.DEMAND_SURGE:
        events.append(InjectionEvent(
            time=300,  # 5分钟后
            scenario=ScenarioType.DEMAND_SURGE,
            duration=1800,  # 30分钟
            severity=0.5,
            parameters={'surge_factor': kwargs.get('surge_factor', 1.5)}
        ))

    elif scenario == ScenarioType.PIPE_BURST:
        events.append(InjectionEvent(
            time=300,
            scenario=ScenarioType.PIPE_BURST,
            duration=float('inf'),
            severity=kwargs.get('leak_rate', 0.1),
            parameters={
                'location': kwargs.get('location', 0.5),
                'leak_rate': kwargs.get('leak_rate', 0.1)
            }
        ))

    elif scenario == ScenarioType.ICE_PERIOD:
        events.append(InjectionEvent(
            time=0,
            scenario=ScenarioType.ICE_PERIOD,
            duration=duration,
            severity=0.5,
            parameters={'temperature': kwargs.get('temperature', -5)}
        ))

    elif scenario == ScenarioType.POWER_FAILURE:
        events.append(InjectionEvent(
            time=300,
            scenario=ScenarioType.POWER_FAILURE,
            duration=kwargs.get('failure_duration', 300),
            severity=1.0,
            parameters={}
        ))

    # 运行仿真
    result = runner.run(scenario_events=events)

    return result
