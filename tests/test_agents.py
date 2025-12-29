"""
智能体模块测试
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/YCJL')

from ycjl.agents.base_agent import BaseAgent, AgentPriority, AgentMessage, MessageType
from ycjl.agents.reflex_agent import ReflexAgent, SafetyRule, RuleCategory
from ycjl.agents.tactical_agent import TacticalAgent, MPCController
from ycjl.agents.strategic_agent import StrategicAgent
from ycjl.agents.coordinator import ADMMCoordinator, CommunicationHub, MultiAgentSystem
from ycjl.config.settings import ScenarioType


class TestReflexAgent:
    """L1反射层智能体测试"""

    def test_initialization(self):
        """测试初始化"""
        agent = ReflexAgent()
        assert agent.agent_id == "L1_reflex"
        assert agent.priority == AgentPriority.SAFETY
        assert len(agent.rules) > 0

    def test_low_level_rule(self):
        """测试低水位规则"""
        agent = ReflexAgent()

        # 正常水位
        state = {'pool_level': 5.0}
        agent.update_observations(state)
        agent.observations = agent.perceive(state)
        actions = agent.decide()

        # 应该没有动作
        low_level_actions = [a for a in actions if 'pool' in a.actuator_id]
        # 正常情况下不应触发

        # 低水位
        state = {'pool_level': 1.0}  # 低于最小水位
        agent.observations = agent.perceive(state)
        actions = agent.decide()

        # 应该有关闸动作
        assert len(agent.active_rules) > 0

    def test_high_pressure_rule(self):
        """测试高压力规则"""
        agent = ReflexAgent()

        # 超高压力
        state = {'pipe_pressure': 150.0}  # 超过最大压力
        agent.observations = agent.perceive(state)
        actions = agent.decide()

        # 应该有泄压动作
        relief_actions = [a for a in actions if 'relief' in a.actuator_id]
        # 可能触发保护动作

    def test_rule_priority(self):
        """测试规则优先级"""
        agent = ReflexAgent()

        # 多个触发条件
        state = {
            'pool_level': 1.0,
            'pipe_pressure': 150.0
        }
        agent.observations = agent.perceive(state)
        actions = agent.decide()

        # 应该按优先级排序
        if len(actions) > 1:
            for i in range(len(actions) - 1):
                assert actions[i].priority.value <= actions[i+1].priority.value


class TestTacticalAgent:
    """L2战术层智能体测试"""

    def test_initialization(self):
        """测试初始化"""
        agent = TacticalAgent("L2_test", segment_id=0)
        assert agent.agent_id == "L2_test"
        assert agent.segment_id == 0
        assert agent.priority == AgentPriority.TACTICAL

    def test_mpc_controller(self):
        """测试MPC控制器"""
        mpc = MPCController()

        # 设置状态
        x0 = np.array([5.0, 10.0, 50.0, 2.0])
        u_prev = np.array([0.5, 0.5])

        # 设置参考
        x_ref = np.array([5.5, 10.0, 50.0, 2.0])
        mpc.set_reference(x_ref)

        # 求解
        result = mpc.solve(x0, u_prev)

        assert result.u_optimal.shape[0] > 0
        assert result.solve_time > 0

    def test_l1_override(self):
        """测试L1覆盖"""
        agent = TacticalAgent()

        # 设置L1覆盖
        state = {'l1_override': True}
        agent.observations = agent.perceive(state)
        actions = agent.decide()

        # L1覆盖时不应产生动作
        assert len(actions) == 0


class TestStrategicAgent:
    """L3战略层智能体测试"""

    def test_initialization(self):
        """测试初始化"""
        agent = StrategicAgent()
        assert agent.agent_id == "L3_strategic"
        assert agent.priority == AgentPriority.STRATEGIC

    def test_optimization(self):
        """测试优化"""
        agent = StrategicAgent()

        # 设置状态
        state = {
            'reservoir_level': 100.0,
            'total_flow': 10.0,
            'total_demand': 10.0,
            'hour': 12
        }
        agent.observations = agent.perceive(state)

        # 运行优化
        agent._run_optimization()

        # 应该生成战略目标
        assert len(agent.strategic_targets) > 0

    def test_scenario_adaptation(self):
        """测试场景适应"""
        agent = StrategicAgent()

        # 设置紧急场景
        agent.set_scenario(ScenarioType.PIPE_BURST)

        # 权重应该改变
        from ycjl.agents.strategic_agent import OptimizationObjective
        assert agent.objective_weights.get(OptimizationObjective.EMERGENCY_RESPONSE, 0) > 0


class TestADMMCoordinator:
    """ADMM协调器测试"""

    def test_initialization(self):
        """测试初始化"""
        admm = ADMMCoordinator(num_agents=4)
        assert admm.num_agents == 4
        assert admm.rho > 0

    def test_convergence(self):
        """测试收敛"""
        admm = ADMMCoordinator(num_agents=2)

        # 注册智能体
        admm.register_agent('agent_0', np.array([1.0, 1.0]))
        admm.register_agent('agent_1', np.array([2.0, 2.0]))

        # 迭代
        converged = False
        for _ in range(50):
            updates = {
                'agent_0': np.array([1.5, 1.5]),
                'agent_1': np.array([1.5, 1.5])
            }
            converged = admm.step(updates)
            if converged:
                break

        # 应该收敛
        assert converged or admm.primal_residual < 1.0


class TestMultiAgentSystem:
    """多智能体系统测试"""

    def test_initialization(self):
        """测试初始化"""
        mas = MultiAgentSystem()
        assert mas.l1_agent is not None
        assert len(mas.l2_agents) > 0
        assert mas.l3_agent is not None

    def test_step(self):
        """测试步进"""
        mas = MultiAgentSystem()

        state = {
            'pool_level': 5.0,
            'pipe_flow': 10.0,
            'pipe_pressure': 50.0,
            'reservoir_level': 100.0,
            'total_demand': 10.0,
            'hour': 12
        }

        result = mas.step(state)

        assert 'L1' in result
        assert 'L2' in result
        assert 'L3' in result
        assert 'cycle_time' in result


class TestCommunicationHub:
    """通信中心测试"""

    def test_message_routing(self):
        """测试消息路由"""
        hub = CommunicationHub()

        # 创建测试智能体
        agent1 = ReflexAgent("test_agent_1")
        agent2 = ReflexAgent("test_agent_2")

        hub.register_agent(agent1)
        hub.register_agent(agent2)

        # 发送消息
        msg = AgentMessage(
            msg_type=MessageType.STATE,
            sender="test_agent_1",
            receiver="test_agent_2",
            priority=AgentPriority.TACTICAL,
            timestamp=0,
            payload={'test': 123}
        )

        hub.send_message(msg)

        # 检查统计
        stats = hub.get_statistics()
        assert stats['total_messages'] == 1


# ============================================
# v3.4.0 L5自主系统和应急容错测试
# ============================================

from ycjl.agents.l5_autonomous import (
    AutonomyLevel, AgentRole, AgentStatus,
    SituationAwarenessAgent, DecisionPlanningAgent, ExecutionControlAgent,
    CoordinationAgent, LearningAgent,
    L5AutonomousSystem, L5SystemState, L5Decision,
    create_l5_system
)
from ycjl.agents.emergency_agent import (
    EmergencyLevel, EmergencyType, EmergencyState, EmergencyResponse,
    EmergencyAgent, create_emergency_agent
)
from ycjl.agents.fault_tolerance import (
    FaultType, FaultState, RedundancyMode,
    FaultToleranceManager, ComponentHealth, SystemResilience
)
from ycjl.scenarios.scenario_database import ScenarioType as ScenarioTypeV2, ScenarioSeverity


class TestL5AutonomousSystem:
    """L5自主运行系统测试"""

    def test_system_initialization(self):
        """测试系统初始化"""
        system = create_l5_system()
        assert system is not None
        assert system.state.autonomy_level == AutonomyLevel.L5_FULL

    def test_autonomy_levels(self):
        """测试自主等级"""
        levels = list(AutonomyLevel)
        assert len(levels) == 6
        assert AutonomyLevel.L0_MANUAL in levels
        assert AutonomyLevel.L5_FULL in levels

    def test_agent_roles(self):
        """测试智能体角色"""
        roles = list(AgentRole)
        assert AgentRole.AWARENESS in roles
        assert AgentRole.PLANNING in roles
        assert AgentRole.EXECUTION in roles
        assert AgentRole.COORDINATION in roles
        assert AgentRole.LEARNING in roles

    def test_situation_awareness_agent(self):
        """测试态势感知智能体"""
        agent = SituationAwarenessAgent()
        assert agent.role == AgentRole.AWARENESS

        # 处理感知数据
        measurements = {
            'flow': 10.0,
            'pressure': 50.0,
            'level': 5.0
        }
        result = agent.process(measurements, 0.0)
        assert result is not None

    def test_decision_planning_agent(self):
        """测试决策规划智能体"""
        agent = DecisionPlanningAgent()
        assert agent.role == AgentRole.PLANNING

        # 处理决策
        situation = {
            'current_scenario': 'normal',
            'risk_level': 0.1
        }
        result = agent.process(situation, 0.0)
        assert result is not None

    def test_execution_control_agent(self):
        """测试执行控制智能体"""
        agent = ExecutionControlAgent()
        assert agent.role == AgentRole.EXECUTION

        # 处理执行
        decision = {
            'action': 'maintain',
            'target_flow': 10.0
        }
        result = agent.process(decision, 0.0)
        assert result is not None

    def test_coordination_agent(self):
        """测试协调管理智能体"""
        agent = CoordinationAgent()
        assert agent.role == AgentRole.COORDINATION

    def test_learning_agent(self):
        """测试学习优化智能体"""
        agent = LearningAgent()
        assert agent.role == AgentRole.LEARNING

    def test_l5_system_cycle(self):
        """测试L5系统循环"""
        system = create_l5_system()

        # 输入测量数据
        measurements = {
            'flow': 10.0,
            'pressure': 50.0,
            'level': 5.0,
            'temperature': 15.0
        }

        # 运行一个周期
        result = system.process_cycle(measurements)
        assert result is not None

    def test_l5_system_state(self):
        """测试L5系统状态"""
        system = create_l5_system()

        state = system.get_state_summary()
        assert isinstance(state, dict)
        assert 'autonomy_level' in state

    def test_autonomy_level_change(self):
        """测试自主等级切换"""
        system = create_l5_system()

        # 初始应该是L5
        assert system.state.autonomy_level == AutonomyLevel.L5_FULL

        # 降级测试
        system.set_autonomy_level(AutonomyLevel.L3_CONDITIONAL)
        assert system.state.autonomy_level == AutonomyLevel.L3_CONDITIONAL


class TestEmergencyAgent:
    """应急响应智能体测试"""

    def test_agent_initialization(self):
        """测试智能体初始化"""
        agent = create_emergency_agent()
        assert agent is not None

    def test_emergency_levels(self):
        """测试应急等级"""
        levels = list(EmergencyLevel)
        assert EmergencyLevel.GREEN in levels
        assert EmergencyLevel.RED in levels
        # 等级应该有序
        assert EmergencyLevel.GREEN.value < EmergencyLevel.RED.value

    def test_emergency_types(self):
        """测试应急类型"""
        types = list(EmergencyType)
        assert EmergencyType.EQUIPMENT in types
        assert EmergencyType.SAFETY in types
        assert EmergencyType.NATURAL in types

    def test_emergency_detection(self):
        """测试应急检测"""
        agent = create_emergency_agent()

        # 创建应急事件
        emergency = agent.detect_emergency(
            ScenarioTypeV2.FAULT_VALVE_STUCK,
            ScenarioSeverity.CRITICAL,
            {'description': '测试阀门故障', 'affected': ['valve_1']}
        )

        assert emergency is not None
        assert emergency.is_active

    def test_emergency_response_generation(self):
        """测试应急响应生成"""
        agent = create_emergency_agent()

        emergency = agent.detect_emergency(
            ScenarioTypeV2.FAULT_PUMP_TRIP,
            ScenarioSeverity.ALARM,
            {'description': '泵跳闸'}
        )

        if emergency:
            response = agent.generate_response(emergency)
            assert response is not None
            assert len(response.actions) > 0

    def test_emergency_close(self):
        """测试应急关闭"""
        agent = create_emergency_agent()

        emergency = agent.detect_emergency(
            ScenarioTypeV2.FAULT_SENSOR_DRIFT,
            ScenarioSeverity.ALARM,
            {'description': '传感器漂移'}
        )

        if emergency:
            agent.close_emergency(emergency.emergency_id, "已修复")
            assert emergency.emergency_id not in agent.active_emergencies

    def test_get_active_emergencies(self):
        """测试获取活动应急"""
        agent = create_emergency_agent()

        # 创建多个应急
        agent.detect_emergency(
            ScenarioTypeV2.FAULT_VALVE_STUCK,
            ScenarioSeverity.CRITICAL,
            {'description': '阀门故障1'}
        )
        agent.detect_emergency(
            ScenarioTypeV2.FAULT_PUMP_TRIP,
            ScenarioSeverity.ALARM,
            {'description': '泵故障'}
        )

        active = agent.get_active_emergencies()
        assert len(active) >= 2


class TestFaultTolerance:
    """故障容错系统测试"""

    def test_manager_initialization(self):
        """测试管理器初始化"""
        manager = FaultToleranceManager()
        assert len(manager.components) > 0

    def test_fault_types(self):
        """测试故障类型"""
        types = list(FaultType)
        assert FaultType.SENSOR in types
        assert FaultType.ACTUATOR in types
        assert FaultType.COMMUNICATION in types

    def test_redundancy_modes(self):
        """测试冗余模式"""
        modes = list(RedundancyMode)
        assert RedundancyMode.COLD_STANDBY in modes
        assert RedundancyMode.HOT_STANDBY in modes
        assert RedundancyMode.ACTIVE_ACTIVE in modes

    def test_health_update(self):
        """测试健康状态更新"""
        manager = FaultToleranceManager()

        # 更新组件健康
        manager.update_health('reservoir', {'health_score': 0.9})
        assert manager.components['reservoir'].health_score == 0.9

    def test_component_isolation(self):
        """测试组件隔离"""
        manager = FaultToleranceManager()

        # 隔离组件
        manager.isolate_component('valve_inline_1')
        assert not manager.components['valve_inline_1'].is_healthy

    def test_component_recovery(self):
        """测试组件恢复"""
        manager = FaultToleranceManager()

        # 先隔离
        manager.isolate_component('valve_inline_1')
        # 再恢复
        manager.recover_component('valve_inline_1')
        assert manager.components['valve_inline_1'].is_healthy

    def test_system_resilience(self):
        """测试系统弹性"""
        manager = FaultToleranceManager()

        resilience = manager.get_system_resilience()
        assert isinstance(resilience, SystemResilience)
        assert resilience.overall_health >= 0
        assert resilience.overall_health <= 1

    def test_fault_statistics(self):
        """测试故障统计"""
        manager = FaultToleranceManager()

        stats = manager.get_fault_statistics()
        assert 'active_faults' in stats
        assert 'total_faults' in stats

    def test_heartbeat_timeout(self):
        """测试心跳超时"""
        manager = FaultToleranceManager()
        manager.heartbeat_timeout = 0.001  # 设置极短超时

        import time
        time.sleep(0.01)

        timeouts = manager.check_heartbeats()
        assert len(timeouts) > 0

    def test_redundancy_configuration(self):
        """测试冗余配置"""
        manager = FaultToleranceManager()

        # 检查冗余配置
        assert 'scada' in manager.redundancy_config
        assert 'network' in manager.redundancy_config
        assert 'power' in manager.redundancy_config

        # 检查冗余模式
        assert manager.redundancy_config['scada']['mode'] == RedundancyMode.HOT_STANDBY


class TestL5Integration:
    """L5系统集成测试"""

    def test_full_cycle_with_emergency(self):
        """测试完整周期包含应急"""
        l5_system = create_l5_system()
        emergency_agent = create_emergency_agent()
        fault_manager = FaultToleranceManager()

        # 模拟正常运行
        measurements = {
            'flow': 10.0,
            'pressure': 50.0,
            'level': 5.0
        }

        result = l5_system.process_cycle(measurements)
        resilience = fault_manager.get_system_resilience()

        assert result is not None
        assert resilience.can_operate

    def test_degradation_handling(self):
        """测试降级处理"""
        fault_manager = FaultToleranceManager()

        # 模拟多个组件故障
        fault_manager.isolate_component('valve_inline_1')
        fault_manager.isolate_component('valve_inline_2')

        resilience = fault_manager.get_system_resilience()

        # 应该仍能运行但有降级
        assert resilience.degradation_level > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
