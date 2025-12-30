"""
泵站群多智能体系统测试
======================

测试内容:
1. 数据结构
2. 泵站群安全智能体 (L1层)
3. 泵站群经济智能体 (L3层)
4. 多智能体系统集成
5. 级联泵站协调调度
"""

import pytest
import sys
import time
import math
sys.path.insert(0, '/home/user/YCJL')

from ycjl.agents.pump_group_agents import (
    PumpStatus, ProtectionType,
    PumpState, StationState, PumpGroupState,
    PumpScheduleAction,
    PumpGroupSafetyAgent,
    PumpGroupEconomicAgent,
    PumpGroupCoordinatorAgent,
    PumpGroupMultiAgentSystem,
    create_pump_group_system,
)
from ycjl.agents.base_agent import AgentPriority


# ============================================================================
#  基础数据结构测试
# ============================================================================

class TestDataStructures:
    """数据结构测试"""

    def test_pump_status_enum(self):
        """测试泵状态枚举"""
        statuses = list(PumpStatus)
        assert PumpStatus.STOPPED in statuses
        assert PumpStatus.RUNNING in statuses
        assert PumpStatus.STARTING in statuses
        assert PumpStatus.STOPPING in statuses
        assert PumpStatus.FAULT in statuses

    def test_protection_type_enum(self):
        """测试保护类型枚举"""
        types = list(ProtectionType)
        assert ProtectionType.OVERLOAD in types
        assert ProtectionType.CAVITATION in types
        assert ProtectionType.VIBRATION in types
        assert ProtectionType.OVERHEAT in types

    def test_pump_state(self):
        """测试泵状态"""
        pump = PumpState(
            pump_id='pump_1',
            station_id='test_station',
            status=PumpStatus.RUNNING,
            flow_rate=2.5,
            head=35.0,
            power=300.0,
            efficiency=0.8
        )

        assert pump.pump_id == 'pump_1'
        assert pump.status == PumpStatus.RUNNING
        assert pump.flow_rate == 2.5

    def test_station_state(self):
        """测试泵站状态"""
        pump1 = PumpState(pump_id='pump_1', station_id='test_station', status=PumpStatus.RUNNING)
        pump2 = PumpState(pump_id='pump_2', station_id='test_station', status=PumpStatus.STOPPED)

        station = StationState(
            station_id='test_station',
            station_name='测试泵站',
            pumps={'pump_1': pump1, 'pump_2': pump2},
            total_pump_count=2,
            running_pump_count=1,
            forebay_level=2.8
        )

        assert station.station_id == 'test_station'
        assert len(station.pumps) == 2
        assert station.forebay_level == 2.8

    def test_pump_group_state(self):
        """测试泵站群状态"""
        station = StationState(
            station_id='test_station',
            station_name='测试泵站'
        )

        group = PumpGroupState(
            stations={'test_station': station},
            total_flow=10.0,
            total_power=500.0,
            current_hour=12,
            electricity_price=0.55
        )

        assert 'test_station' in group.stations
        assert group.total_flow == 10.0

    def test_pump_schedule_action(self):
        """测试泵调度动作"""
        action = PumpScheduleAction(
            station_id='tundian',
            pump_id='pump_1',
            action='start',
            target_value=2.5,
            priority=AgentPriority.SAFETY,
            reason='满足需水量'
        )

        assert action.station_id == 'tundian'
        assert action.action == 'start'
        assert action.priority == AgentPriority.SAFETY


# ============================================================================
#  安全智能体测试
# ============================================================================

class TestPumpGroupSafetyAgent:
    """L1安全层智能体测试"""

    @pytest.fixture
    def safety_agent(self):
        """创建安全智能体"""
        return PumpGroupSafetyAgent()

    def test_initialization(self, safety_agent):
        """测试初始化"""
        assert safety_agent.agent_id == 'PumpGroup_L1_Safety'
        # 检查rules属性存在
        assert hasattr(safety_agent, 'rules')

    def test_perceive(self, safety_agent):
        """测试感知"""
        state = {
            'stations': {
                'test': StationState(
                    station_id='test',
                    station_name='测试站',
                    forebay_level=2.8
                )
            }
        }

        observations = safety_agent.perceive(state)
        assert observations is not None

    def test_decide(self, safety_agent):
        """测试决策"""
        state = {
            'stations': {
                'test': StationState(
                    station_id='test',
                    station_name='测试站',
                    forebay_level=2.8
                )
            }
        }
        safety_agent.perceive(state)

        actions = safety_agent.decide()
        assert isinstance(actions, list)

    def test_get_safety_status(self, safety_agent):
        """测试获取安全状态"""
        status = safety_agent.get_safety_status()

        assert 'rule_count' in status
        assert 'active_rules' in status


# ============================================================================
#  经济智能体测试
# ============================================================================

class TestPumpGroupEconomicAgent:
    """L3经济层智能体测试"""

    @pytest.fixture
    def economic_agent(self):
        """创建经济智能体"""
        return PumpGroupEconomicAgent()

    def test_initialization(self, economic_agent):
        """测试初始化"""
        assert economic_agent.agent_id == 'PumpGroup_L3_Economic'

    def test_perceive(self, economic_agent):
        """测试感知"""
        state = {
            'stations': {
                'test': StationState(
                    station_id='test',
                    station_name='测试站',
                    forebay_level=2.8
                )
            },
            'current_hour': 12,
            'target_flow': 10.0
        }

        observations = economic_agent.perceive(state)
        assert observations is not None

    def test_decide(self, economic_agent):
        """测试决策"""
        state = {
            'stations': {
                'test': StationState(
                    station_id='test',
                    station_name='测试站',
                    forebay_level=2.8
                )
            },
            'current_hour': 12,
            'target_flow': 10.0
        }
        economic_agent.perceive(state)

        actions = economic_agent.decide()
        assert isinstance(actions, list)

    def test_get_optimization_summary(self, economic_agent):
        """测试获取优化摘要"""
        summary = economic_agent.get_optimization_summary()

        assert 'params' in summary or 'schedule_length' in summary


# ============================================================================
#  协调智能体测试
# ============================================================================

class TestPumpGroupCoordinatorAgent:
    """协调智能体测试"""

    @pytest.fixture
    def coordinator_agent(self):
        """创建协调智能体"""
        return PumpGroupCoordinatorAgent()

    def test_initialization(self, coordinator_agent):
        """测试初始化"""
        assert coordinator_agent.agent_id == 'PumpGroup_Coordinator'

    def test_set_sub_agents(self, coordinator_agent):
        """测试设置子智能体"""
        safety = PumpGroupSafetyAgent()
        economic = PumpGroupEconomicAgent()
        coordinator_agent.set_sub_agents(safety, economic)

        assert coordinator_agent.safety_agent is not None
        assert coordinator_agent.economic_agent is not None

    def test_perceive(self, coordinator_agent):
        """测试感知"""
        state = {
            'stations': {
                'test': StationState(
                    station_id='test',
                    station_name='测试站',
                    forebay_level=2.8
                )
            }
        }

        observations = coordinator_agent.perceive(state)
        assert observations is not None

    def test_decide(self, coordinator_agent):
        """测试决策"""
        state = {
            'stations': {
                'test': StationState(
                    station_id='test',
                    station_name='测试站',
                    forebay_level=2.8
                )
            }
        }
        coordinator_agent.perceive(state)

        actions = coordinator_agent.decide()
        assert isinstance(actions, list)

    def test_get_coordination_status(self, coordinator_agent):
        """测试获取协调状态"""
        status = coordinator_agent.get_coordination_status()

        assert 'mode' in status


# ============================================================================
#  多智能体系统测试
# ============================================================================

class TestPumpGroupMultiAgentSystem:
    """多智能体系统测试"""

    @pytest.fixture
    def system(self):
        """创建多智能体系统"""
        return create_pump_group_system()

    def test_initialization(self, system):
        """测试初始化"""
        assert system is not None
        assert system.safety_agent is not None
        assert system.economic_agent is not None
        assert system.coordinator is not None

    def test_has_pump_group_state(self, system):
        """测试泵站群状态"""
        assert hasattr(system, 'pump_group_state')
        assert system.pump_group_state is not None

    def test_get_system_status(self, system):
        """测试获取系统状态"""
        status = system.get_system_status()
        assert isinstance(status, dict)

    def test_get_optimization_report(self, system):
        """测试获取优化报告"""
        report = system.get_optimization_report()
        assert isinstance(report, dict)


# ============================================================================
#  级联协调测试
# ============================================================================

from ycjl.agents.cascade_coordination import (
    CoordinationStrategy, WaveType,
    CanalSection, Forebay,
    WavePropagationPredictor,
    ForebayLevelController,
    CascadeCoordinator,
    create_cascade_coordinator,
)


class TestCascadeDataStructures:
    """级联协调数据结构测试"""

    def test_coordination_strategy(self):
        """测试协调策略枚举"""
        strategies = list(CoordinationStrategy)
        assert CoordinationStrategy.RELAY in strategies
        assert CoordinationStrategy.PARALLEL in strategies
        assert CoordinationStrategy.ADAPTIVE in strategies

    def test_wave_type(self):
        """测试波类型枚举"""
        types = list(WaveType)
        assert WaveType.POSITIVE in types
        assert WaveType.NEGATIVE in types

    def test_canal_section(self):
        """测试渠道段"""
        section = CanalSection(
            section_id='sec1',
            length=4000.0,
            width=8.0,
            slope=0.0001,
            roughness=0.015,  # 曼宁糙率
            water_depth=3.0
        )

        assert section.length == 4000.0
        assert section.water_depth == 3.0

    def test_forebay(self):
        """测试前池"""
        forebay = Forebay(
            forebay_id='fb1',
            area=2000.0,
            min_level=2.0,
            max_level=3.5,
            alarm_low=2.3,
            alarm_high=3.3,
            level=2.8
        )

        assert forebay.area == 2000.0
        assert forebay.level == 2.8


class TestWavePropagation:
    """波传播预测测试"""

    @pytest.fixture
    def predictor(self):
        """创建波传播预测器"""
        predictor = WavePropagationPredictor()
        section = CanalSection(
            section_id='sec1',
            length=4000.0,
            width=8.0,
            slope=0.0001,
            roughness=0.015,
            water_depth=3.0
        )
        predictor.add_canal_section(section)
        return predictor

    def test_initialization(self, predictor):
        """测试初始化"""
        assert predictor is not None
        assert len(predictor.canal_sections) > 0

    def test_wave_celerity_from_canal(self, predictor):
        """测试渠道段波速计算"""
        # c = sqrt(g * h)
        section = predictor.canal_sections['sec1']
        expected_celerity = math.sqrt(9.81 * section.water_depth)

        # 波速是CanalSection的属性
        celerity = section.wave_celerity

        assert abs(celerity - expected_celerity) < 0.01

    def test_active_waves_empty(self, predictor):
        """测试初始活跃波为空"""
        assert len(predictor.active_waves) == 0


class TestForebayController:
    """前池控制器测试"""

    @pytest.fixture
    def controller(self):
        """创建前池控制器"""
        forebay = Forebay(
            forebay_id='test_forebay',
            area=2000.0,
            min_level=2.0,
            max_level=3.5,
            alarm_low=2.3,
            alarm_high=3.3,
            level=2.8
        )
        return ForebayLevelController(forebay)

    def test_initialization(self, controller):
        """测试初始化"""
        assert controller is not None
        assert controller.forebay is not None

    def test_level_prediction(self, controller):
        """测试水位预测"""
        current_level = 2.8
        inflow = 5.0
        outflow = 4.5
        dt = 3600  # 1小时

        predicted = controller.predict_level(current_level, inflow, outflow, dt)

        # 预测水位应该在合理范围内
        assert predicted >= controller.forebay.min_level
        assert predicted <= controller.forebay.max_level

    def test_buffer_time(self, controller):
        """测试缓冲时间计算"""
        current_level = 2.8
        inflow = 5.0
        outflow = 4.5

        time_to_high, time_to_low = controller.estimate_buffer_time(
            current_level, inflow, outflow
        )

        # 入流大于出流，应该有到高水位的时间
        assert time_to_high > 0


class TestCascadeCoordinator:
    """级联协调器测试"""

    @pytest.fixture
    def coordinator(self):
        """创建级联协调器"""
        return create_cascade_coordinator()

    def test_initialization(self, coordinator):
        """测试初始化"""
        assert coordinator is not None
        assert len(coordinator.station_order) > 0

    def test_station_order_has_tundian(self, coordinator):
        """测试站点顺序包含屯佃"""
        assert 'tundian' in coordinator.station_order

    def test_upstream_action_response(self, coordinator):
        """测试上游动作响应"""
        commands = coordinator.on_upstream_action(
            station_id='tundian',
            action='start_pump',
            flow_change=2.5
        )

        # 应该返回指令列表
        assert isinstance(commands, list)

    def test_wave_propagation_info(self, coordinator):
        """测试波传播信息"""
        info = coordinator.get_wave_propagation_info()

        assert 'canal_sections' in info
        assert 'estimated_travel_times' in info

    def test_optimal_schedule(self, coordinator):
        """测试最优调度计划"""
        schedule = coordinator.get_optimal_schedule(
            target_flow=15.0,
            horizon_hours=4
        )

        assert isinstance(schedule, list)
        assert len(schedule) == 4  # 4小时

    def test_coordination_status(self, coordinator):
        """测试协调状态"""
        status = coordinator.get_coordination_status()

        assert 'strategy' in status
        assert 'station_count' in status
        assert 'station_order' in status


# ============================================================================
#  L5集成测试
# ============================================================================

from ycjl.agents.l5_pump_group_integration import (
    L5PumpGroupSystem,
    PumpGroupAwarenessExtension,
    PumpGroupPlanningExtension,
    PumpGroupExecutionExtension,
    create_l5_pump_group_system,
)


class TestL5PumpGroupIntegration:
    """L5泵站群集成测试"""

    @pytest.fixture
    def l5_system(self):
        """创建L5集成系统"""
        return create_l5_pump_group_system()

    def test_initialization(self, l5_system):
        """测试初始化"""
        assert l5_system is not None
        assert l5_system.pump_system is not None

    def test_has_awareness(self, l5_system):
        """测试态势感知扩展"""
        assert l5_system.pump_awareness is not None

    def test_has_planning(self, l5_system):
        """测试规划扩展"""
        assert l5_system.pump_planning is not None

    def test_has_execution(self, l5_system):
        """测试执行扩展"""
        assert l5_system.pump_execution is not None


# ============================================================================
#  性能测试
# ============================================================================

class TestPerformance:
    """性能测试"""

    def test_system_creation_time(self):
        """测试系统创建时间"""
        start = time.time()
        system = create_pump_group_system()
        elapsed = time.time() - start

        # 创建应该在1秒内完成
        assert elapsed < 1.0

    def test_coordinator_creation_time(self):
        """测试协调器创建时间"""
        start = time.time()
        coordinator = create_cascade_coordinator()
        elapsed = time.time() - start

        # 创建应该在1秒内完成
        assert elapsed < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
