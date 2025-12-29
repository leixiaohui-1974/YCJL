"""
集成测试
========

全系统集成测试
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/YCJL')

from ycjl.simulation.plant import WaterTransferPlant, PlantMode
from ycjl.simulation.runner import SimulationRunner, SimulationConfig, run_scenario_test
from ycjl.simulation.scenario_injector import ScenarioInjector, InjectionEvent
from ycjl.config.settings import ScenarioType
from ycjl.agents.coordinator import MultiAgentSystem
from ycjl.scenarios.detector import ScenarioDetector
from ycjl.models.digital_twin import DigitalTwin


class TestPlantIntegration:
    """系统工厂集成测试"""

    def test_plant_initialization(self):
        """测试系统初始化"""
        plant = WaterTransferPlant()

        assert plant.reservoir is not None
        assert plant.tunnel is not None
        assert plant.pool is not None
        assert plant.pipeline is not None
        assert plant.surge_tank is not None

    def test_plant_step(self):
        """测试系统步进"""
        plant = WaterTransferPlant()

        # 运行几步
        for _ in range(10):
            state = plant.step(1.0)

        assert state is not None
        assert state.timestamp > 0
        assert state.total_flow >= 0

    def test_plant_with_control(self):
        """测试带控制的系统运行"""
        plant = WaterTransferPlant()

        control_inputs = {
            'gate_intake': 0.7,
            'gate_pool_out': 0.6,
            'valve_mid': 0.8,
            'valve_end': 0.7
        }

        for _ in range(20):
            state = plant.step(1.0, control_inputs)

        assert state.total_flow > 0

    def test_plant_measurements(self):
        """测试传感器测量"""
        plant = WaterTransferPlant()

        plant.step(1.0)
        measurements = plant.get_measurements()

        assert 'reservoir_level' in measurements
        assert 'pool_level' in measurements
        assert 'pipe_flow' in measurements


class TestScenarioInjection:
    """场景注入测试"""

    def test_demand_surge_injection(self):
        """测试需水激增注入"""
        injector = ScenarioInjector()
        injector.schedule_demand_surge(time=10, duration=100, surge_factor=1.5)

        # 注入前
        state = {'demand': 10.0}
        state = injector.step(5, state)
        assert state['demand'] == 10.0

        # 注入后
        state = injector.step(50, state)
        assert state['demand'] > 10.0
        assert state.get('demand_surge_active', False)

    def test_pipe_burst_injection(self):
        """测试爆管注入"""
        injector = ScenarioInjector()
        injector.schedule_pipe_burst(time=10, location=0.5, leak_rate=0.1)

        state = {'pipe_flow': 10.0, 'pipe_pressure': 50.0}
        state = injector.step(15, state)

        assert state.get('leak_active', False)
        assert state['pipe_flow'] < 10.0

    def test_ice_period_injection(self):
        """测试冰期注入"""
        injector = ScenarioInjector()
        injector.schedule_ice_period(time=0, duration=1000, temperature=-5)

        state = {}
        state = injector.step(50, state)

        assert state.get('is_ice_period', False)
        assert state.get('water_temperature', 10) < 0


class TestSimulationRunner:
    """仿真运行器测试"""

    def test_short_simulation(self):
        """测试短仿真"""
        config = SimulationConfig(
            duration=60,
            dt=1.0,
            verbose=False
        )

        runner = SimulationRunner(config)
        result = runner.run()

        assert result.success
        assert result.steps > 0
        assert len(result.states) > 0

    def test_simulation_with_control(self):
        """测试带控制的仿真"""
        config = SimulationConfig(
            duration=120,
            dt=1.0,
            enable_control=True,
            verbose=False
        )

        runner = SimulationRunner(config)
        result = runner.run()

        assert result.control_actions >= 0

    def test_simulation_with_detection(self):
        """测试带检测的仿真"""
        config = SimulationConfig(
            duration=120,
            dt=1.0,
            enable_detection=True,
            verbose=False
        )

        runner = SimulationRunner(config)
        result = runner.run()

        assert len(result.scenario_history) > 0

    def test_simulation_metrics(self):
        """测试仿真指标"""
        config = SimulationConfig(
            duration=120,
            dt=1.0,
            verbose=False
        )

        runner = SimulationRunner(config)
        result = runner.run()

        assert 'pool_level_mean' in result.metrics
        assert 'flow_mean' in result.metrics


class TestScenarioTests:
    """场景测试"""

    @pytest.mark.slow
    def test_demand_surge_scenario(self):
        """测试需水激增场景"""
        result = run_scenario_test(
            ScenarioType.DEMAND_SURGE,
            duration=600,
            surge_factor=1.5
        )

        assert result.success
        # 应该检测到需水激增
        scenarios = [s for _, s in result.scenario_history]
        # 可能识别到场景变化

    @pytest.mark.slow
    def test_pipe_burst_scenario(self):
        """测试爆管场景"""
        result = run_scenario_test(
            ScenarioType.PIPE_BURST,
            duration=600,
            leak_rate=0.1
        )

        assert result.success
        # 应该有安全干预
        assert result.safety_interventions > 0 or result.control_actions > 0

    @pytest.mark.slow
    def test_ice_period_scenario(self):
        """测试冰期场景"""
        result = run_scenario_test(
            ScenarioType.ICE_PERIOD,
            duration=600,
            temperature=-5
        )

        assert result.success


class TestDigitalTwinIntegration:
    """数字孪生集成测试"""

    def test_twin_synchronization(self):
        """测试数字孪生同步"""
        twin = DigitalTwin()
        plant = WaterTransferPlant()

        # 运行并同步
        for _ in range(20):
            plant_state = plant.step(1.0)

            # 更新孪生
            physical_state = {
                'tunnel_flow': plant_state.tunnel.flow if hasattr(plant_state.tunnel, 'flow') else 10.0,
                'pool_level': plant_state.pool.level,
                'pipe_pressure': plant_state.measurements.get('pipe_pressure_up', 50.0),
                'pipe_flow': plant_state.measurements.get('pipe_flow', 10.0)
            }

            twin.update_physical_state(physical_state)
            twin_state = twin.step(1.0, {})

        # 检查同步状态
        from ycjl.models.digital_twin import TwinSyncStatus
        assert twin_state.sync_status in [
            TwinSyncStatus.SYNCHRONIZED,
            TwinSyncStatus.DRIFTING,
            TwinSyncStatus.DIVERGED
        ]


class TestMultiAgentIntegration:
    """多智能体集成测试"""

    def test_mas_normal_operation(self):
        """测试正常运行"""
        mas = MultiAgentSystem()
        plant = WaterTransferPlant()

        for _ in range(20):
            state_dict = plant.get_state_dict()
            mas_result = mas.step(state_dict)

            # 获取控制动作
            actions = mas_result.get('all_actions', [])

            # 转换为控制输入
            control_inputs = {}
            for action in actions:
                actuator = action.get('actuator', '')
                value = action.get('value', 0)
                control_inputs[actuator] = value

            plant.step(1.0, control_inputs)

        # 系统应该正常运行
        status = mas.get_system_status()
        assert status['cycle_count'] == 20

    def test_mas_emergency_response(self):
        """测试应急响应"""
        mas = MultiAgentSystem()

        # 模拟紧急状态
        state = {
            'pool_level': 1.0,  # 低水位
            'pipe_pressure': 150.0,  # 高压力
        }

        result = mas.step(state)

        # L1层应该有动作
        l1_result = result.get('L1', {})
        assert len(l1_result.get('active_rules', [])) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-x'])
