"""
智能体协调器
============

分布式协调算法:
- ADMM分布式优化
- 事件触发通信
- 优先级仲裁
- 故障处理
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import time

from .base_agent import (
    BaseAgent, AgentPriority, AgentState, AgentMessage,
    ControlAction, MessageType, AgentRegistry
)
from ..config.settings import Config


class CoordinationStatus(Enum):
    """协调状态"""
    IDLE = auto()
    COLLECTING = auto()
    OPTIMIZING = auto()
    DISTRIBUTING = auto()
    CONVERGED = auto()
    TIMEOUT = auto()


@dataclass
class ADMMState:
    """ADMM状态"""
    agent_id: str
    x_local: np.ndarray       # 本地变量
    z_shared: np.ndarray      # 共享变量
    lambda_dual: np.ndarray   # 拉格朗日乘子
    primal_residual: float    # 原始残差
    dual_residual: float      # 对偶残差


@dataclass
class CommunicationEvent:
    """通信事件"""
    timestamp: float
    sender: str
    receiver: str
    msg_type: MessageType
    size_bytes: int
    latency: float = 0.0


class ADMMCoordinator:
    """
    ADMM分布式优化协调器

    算法:
    1. x更新: 各智能体本地优化
    2. z更新: 协调器汇总平均
    3. lambda更新: 对偶变量更新
    4. 收敛检查
    """

    def __init__(self, num_agents: int = 4):
        self.num_agents = num_agents

        # ADMM参数
        self.rho = 1.0              # 惩罚参数
        self.rho_min = 0.1
        self.rho_max = 100.0
        self.tau_incr = 2.0         # 自适应增益
        self.tau_decr = 2.0
        self.mu = 10.0              # 残差比阈值

        # 变量维度
        self.var_dim = 2            # 每个智能体的共享变量维度

        # 智能体状态
        self.agent_states: Dict[str, ADMMState] = {}

        # 共享变量
        self.z_global = np.zeros(self.var_dim)
        self.z_prev = np.zeros(self.var_dim)

        # 收敛判据
        self.eps_abs = 1e-4
        self.eps_rel = 1e-3
        self.max_iterations = 100

        # 状态
        self.status = CoordinationStatus.IDLE
        self.iteration = 0
        self.primal_residual = float('inf')
        self.dual_residual = float('inf')

        # 历史
        self.convergence_history: List[Dict] = []

    def register_agent(self, agent_id: str, initial_x: np.ndarray = None):
        """注册智能体"""
        if initial_x is None:
            initial_x = np.zeros(self.var_dim)

        self.agent_states[agent_id] = ADMMState(
            agent_id=agent_id,
            x_local=initial_x.copy(),
            z_shared=np.zeros(self.var_dim),
            lambda_dual=np.zeros(self.var_dim),
            primal_residual=0.0,
            dual_residual=0.0
        )

    def collect_local_updates(self, updates: Dict[str, np.ndarray]):
        """收集各智能体的本地更新"""
        self.status = CoordinationStatus.COLLECTING

        for agent_id, x_new in updates.items():
            if agent_id in self.agent_states:
                self.agent_states[agent_id].x_local = x_new.copy()

    def z_update(self):
        """z更新 (共享变量平均)"""
        self.status = CoordinationStatus.OPTIMIZING
        self.z_prev = self.z_global.copy()

        # 简单平均
        sum_x_lambda = np.zeros(self.var_dim)
        for state in self.agent_states.values():
            sum_x_lambda += state.x_local + state.lambda_dual / self.rho

        self.z_global = sum_x_lambda / len(self.agent_states)

        # 更新各智能体的z_shared
        for state in self.agent_states.values():
            state.z_shared = self.z_global.copy()

    def lambda_update(self):
        """拉格朗日乘子更新"""
        for state in self.agent_states.values():
            state.lambda_dual += self.rho * (state.x_local - self.z_global)

    def compute_residuals(self) -> Tuple[float, float]:
        """计算残差"""
        # 原始残差: ||x - z||
        primal = 0.0
        for state in self.agent_states.values():
            primal += np.linalg.norm(state.x_local - self.z_global) ** 2
        primal = np.sqrt(primal)

        # 对偶残差: ||rho * (z - z_prev)||
        dual = self.rho * np.sqrt(len(self.agent_states)) * \
               np.linalg.norm(self.z_global - self.z_prev)

        self.primal_residual = primal
        self.dual_residual = dual

        # 更新各智能体残差
        for state in self.agent_states.values():
            state.primal_residual = np.linalg.norm(state.x_local - self.z_global)
            state.dual_residual = self.rho * np.linalg.norm(self.z_global - self.z_prev)

        return primal, dual

    def check_convergence(self) -> bool:
        """检查收敛"""
        n = len(self.agent_states)
        p = self.var_dim

        # 原始残差阈值
        eps_pri = np.sqrt(n * p) * self.eps_abs + self.eps_rel * max(
            np.sqrt(sum(np.linalg.norm(s.x_local)**2 for s in self.agent_states.values())),
            np.sqrt(n) * np.linalg.norm(self.z_global)
        )

        # 对偶残差阈值
        eps_dual = np.sqrt(n * p) * self.eps_abs + self.eps_rel * \
                   np.sqrt(sum(np.linalg.norm(s.lambda_dual)**2 for s in self.agent_states.values()))

        return self.primal_residual < eps_pri and self.dual_residual < eps_dual

    def adapt_rho(self):
        """自适应调整rho"""
        if self.primal_residual > self.mu * self.dual_residual:
            self.rho = min(self.rho * self.tau_incr, self.rho_max)
            # 调整lambda
            for state in self.agent_states.values():
                state.lambda_dual /= self.tau_incr
        elif self.dual_residual > self.mu * self.primal_residual:
            self.rho = max(self.rho / self.tau_decr, self.rho_min)
            # 调整lambda
            for state in self.agent_states.values():
                state.lambda_dual *= self.tau_decr

    def step(self, local_updates: Dict[str, np.ndarray]) -> bool:
        """
        执行一次ADMM迭代

        Parameters:
            local_updates: 各智能体的本地更新

        Returns:
            是否收敛
        """
        # 1. 收集本地更新
        self.collect_local_updates(local_updates)

        # 2. z更新
        self.z_update()

        # 3. lambda更新
        self.lambda_update()

        # 4. 计算残差
        self.compute_residuals()

        # 5. 检查收敛
        converged = self.check_convergence()

        # 6. 自适应rho
        self.adapt_rho()

        # 记录历史
        self.convergence_history.append({
            'iteration': self.iteration,
            'primal_residual': self.primal_residual,
            'dual_residual': self.dual_residual,
            'rho': self.rho,
            'converged': converged
        })

        self.iteration += 1

        if converged:
            self.status = CoordinationStatus.CONVERGED
        elif self.iteration >= self.max_iterations:
            self.status = CoordinationStatus.TIMEOUT

        return converged

    def get_shared_variable(self) -> np.ndarray:
        """获取共享变量"""
        return self.z_global.copy()

    def get_agent_dual(self, agent_id: str) -> np.ndarray:
        """获取智能体的对偶变量"""
        if agent_id in self.agent_states:
            return self.agent_states[agent_id].lambda_dual.copy()
        return np.zeros(self.var_dim)

    def distribute_updates(self) -> Dict[str, Dict]:
        """分发更新给各智能体"""
        self.status = CoordinationStatus.DISTRIBUTING

        updates = {}
        for agent_id, state in self.agent_states.items():
            updates[agent_id] = {
                'z': state.z_shared.copy(),
                'lambda': state.lambda_dual.copy(),
                'rho': self.rho
            }

        return updates

    def reset(self):
        """重置"""
        self.z_global = np.zeros(self.var_dim)
        self.z_prev = np.zeros(self.var_dim)
        self.iteration = 0
        self.primal_residual = float('inf')
        self.dual_residual = float('inf')
        self.status = CoordinationStatus.IDLE
        self.convergence_history.clear()

        for state in self.agent_states.values():
            state.x_local = np.zeros(self.var_dim)
            state.z_shared = np.zeros(self.var_dim)
            state.lambda_dual = np.zeros(self.var_dim)


class CommunicationHub:
    """
    通信中心

    功能:
    - 消息路由
    - 事件触发通信
    - 带宽管理
    - 通信日志
    """

    def __init__(self):
        # 智能体注册表
        self.registry = AgentRegistry()

        # 消息队列
        self.message_queue: deque = deque(maxlen=1000)

        # 事件触发器
        self.event_triggers: Dict[str, Callable] = {}
        self.trigger_thresholds: Dict[str, float] = {}
        self.last_triggered: Dict[str, float] = {}

        # 通信日志
        self.comm_log: List[CommunicationEvent] = []
        self.max_log_size = 10000

        # 统计
        self.total_messages = 0
        self.total_bytes = 0

        # 通信参数
        self.base_latency = 0.001  # 1ms
        self.bandwidth_limit = 1e6  # 1 MB/s

    def register_agent(self, agent: BaseAgent):
        """注册智能体"""
        self.registry.register(agent)

    def unregister_agent(self, agent_id: str):
        """注销智能体"""
        self.registry.unregister(agent_id)

    def register_event_trigger(self, event_id: str, condition: Callable,
                               threshold: float = 0.0):
        """
        注册事件触发器

        Parameters:
            event_id: 事件标识
            condition: 触发条件函数
            threshold: 最小触发间隔
        """
        self.event_triggers[event_id] = condition
        self.trigger_thresholds[event_id] = threshold
        self.last_triggered[event_id] = 0.0

    def check_events(self, state: Dict) -> List[str]:
        """检查并触发事件"""
        current_time = time.time()
        triggered = []

        for event_id, condition in self.event_triggers.items():
            # 检查时间间隔
            if current_time - self.last_triggered.get(event_id, 0) < \
               self.trigger_thresholds.get(event_id, 0):
                continue

            # 检查条件
            try:
                if condition(state):
                    triggered.append(event_id)
                    self.last_triggered[event_id] = current_time
            except Exception:
                pass

        return triggered

    def send_message(self, msg: AgentMessage):
        """发送消息"""
        # 计算消息大小 (估算)
        size = self._estimate_message_size(msg)

        # 计算延迟
        latency = self.base_latency + size / self.bandwidth_limit

        # 记录日志
        event = CommunicationEvent(
            timestamp=time.time(),
            sender=msg.sender,
            receiver=msg.receiver,
            msg_type=msg.msg_type,
            size_bytes=size,
            latency=latency
        )
        self._log_event(event)

        # 路由消息
        if msg.receiver == 'broadcast':
            self.registry.broadcast(msg)
        else:
            receiver = self.registry.get(msg.receiver)
            if receiver:
                receiver.receive_message(msg)

        self.total_messages += 1
        self.total_bytes += size

    def _estimate_message_size(self, msg: AgentMessage) -> int:
        """估算消息大小"""
        # 基础大小
        base_size = 100  # bytes

        # payload大小
        import json
        try:
            payload_str = json.dumps(msg.payload)
            payload_size = len(payload_str)
        except:
            payload_size = 0

        return base_size + payload_size

    def _log_event(self, event: CommunicationEvent):
        """记录通信事件"""
        self.comm_log.append(event)
        if len(self.comm_log) > self.max_log_size:
            self.comm_log = self.comm_log[-self.max_log_size:]

    def route_all_messages(self):
        """路由所有待发消息"""
        self.registry.route_messages()

    def step(self, system_state: Dict):
        """
        通信步进

        Parameters:
            system_state: 系统状态
        """
        # 检查事件触发
        triggered_events = self.check_events(system_state)

        # 处理触发事件
        for event_id in triggered_events:
            # 广播事件通知
            msg = AgentMessage(
                msg_type=MessageType.ALERT,
                sender='communication_hub',
                receiver='broadcast',
                priority=AgentPriority.SAFETY,
                timestamp=time.time(),
                payload={'event': event_id}
            )
            self.send_message(msg)

        # 路由消息
        self.route_all_messages()

    def get_statistics(self) -> Dict:
        """获取通信统计"""
        if not self.comm_log:
            return {
                'total_messages': 0,
                'total_bytes': 0,
                'average_latency': 0,
                'messages_by_type': {}
            }

        # 按类型统计
        by_type: Dict[str, int] = {}
        total_latency = 0.0

        for event in self.comm_log:
            type_name = event.msg_type.name
            by_type[type_name] = by_type.get(type_name, 0) + 1
            total_latency += event.latency

        return {
            'total_messages': self.total_messages,
            'total_bytes': self.total_bytes,
            'average_latency': total_latency / len(self.comm_log),
            'messages_by_type': by_type,
            'recent_events': len(self.comm_log)
        }

    def get_agent_comm_stats(self, agent_id: str) -> Dict:
        """获取特定智能体的通信统计"""
        sent = 0
        received = 0

        for event in self.comm_log:
            if event.sender == agent_id:
                sent += 1
            if event.receiver == agent_id or event.receiver == 'broadcast':
                received += 1

        return {
            'agent_id': agent_id,
            'messages_sent': sent,
            'messages_received': received
        }

    def reset(self):
        """重置"""
        self.message_queue.clear()
        self.comm_log.clear()
        self.last_triggered.clear()
        self.total_messages = 0
        self.total_bytes = 0


class MultiAgentSystem:
    """
    多智能体系统

    整合L1/L2/L3层智能体和协调器
    """

    def __init__(self):
        from .reflex_agent import ReflexAgent
        from .tactical_agent import TacticalAgent
        from .strategic_agent import StrategicAgent

        # 通信中心
        self.comm_hub = CommunicationHub()

        # L1层
        self.l1_agent = ReflexAgent("L1_reflex")
        self.comm_hub.register_agent(self.l1_agent)

        # L2层 (多个管段)
        self.l2_agents: List[TacticalAgent] = []
        for i in range(4):
            agent = TacticalAgent(f"L2_tactical_{i}", segment_id=i)
            self.l2_agents.append(agent)
            self.comm_hub.register_agent(agent)

        # L3层
        self.l3_agent = StrategicAgent("L3_strategic")
        self.comm_hub.register_agent(self.l3_agent)

        # ADMM协调器
        self.admm = ADMMCoordinator(num_agents=4)
        for agent in self.l2_agents:
            self.admm.register_agent(agent.agent_id)

        # 状态
        self.cycle_count = 0
        self.last_cycle_time = 0.0

    def step(self, system_state: Dict) -> Dict:
        """
        执行一个系统周期

        Parameters:
            system_state: 系统状态

        Returns:
            所有控制动作
        """
        start_time = time.time()
        results = {}

        # 1. L1层 (最高优先级)
        l1_result = self.l1_agent.step(system_state)
        results['L1'] = l1_result

        # 检查L1是否有覆盖动作
        l1_override = len(l1_result.get('actions', [])) > 0
        system_state['l1_override'] = l1_override
        system_state['l1_actions'] = l1_result.get('actions', [])

        # 2. L3层 (战略决策)
        l3_result = self.l3_agent.step(system_state)
        results['L3'] = l3_result

        # 3. L2层 (战术控制)
        l2_results = []
        local_updates = {}

        for agent in self.l2_agents:
            result = agent.step(system_state)
            l2_results.append(result)
            local_updates[agent.agent_id] = agent.u

        results['L2'] = l2_results

        # 4. ADMM协调
        if not l1_override:
            converged = self.admm.step(local_updates)
            admm_updates = self.admm.distribute_updates()

            # 分发ADMM更新
            for agent in self.l2_agents:
                if agent.agent_id in admm_updates:
                    update = admm_updates[agent.agent_id]
                    msg = AgentMessage(
                        msg_type=MessageType.COORDINATION,
                        sender='ADMM_coordinator',
                        receiver=agent.agent_id,
                        priority=AgentPriority.TACTICAL,
                        timestamp=time.time(),
                        payload=update
                    )
                    self.comm_hub.send_message(msg)

            results['ADMM'] = {
                'converged': converged,
                'iteration': self.admm.iteration,
                'primal_residual': self.admm.primal_residual,
                'dual_residual': self.admm.dual_residual
            }

        # 5. 通信处理
        self.comm_hub.step(system_state)

        # 合并所有控制动作
        all_actions = []
        all_actions.extend(l1_result.get('actions', []))
        for l2_result in l2_results:
            all_actions.extend(l2_result.get('actions', []))

        results['all_actions'] = all_actions

        self.cycle_count += 1
        self.last_cycle_time = time.time() - start_time
        results['cycle_time'] = self.last_cycle_time

        return results

    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return {
            'cycle_count': self.cycle_count,
            'last_cycle_time': self.last_cycle_time,
            'l1_active_rules': self.l1_agent.get_active_rules(),
            'l3_scenario': self.l3_agent.current_scenario.name,
            'admm_status': self.admm.status.name,
            'comm_stats': self.comm_hub.get_statistics()
        }

    def reset(self):
        """重置系统"""
        self.l1_agent.reset()
        for agent in self.l2_agents:
            agent.reset()
        self.l3_agent.reset()
        self.admm.reset()
        self.comm_hub.reset()
        self.cycle_count = 0
