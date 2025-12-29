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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
