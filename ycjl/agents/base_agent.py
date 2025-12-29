"""
智能体基类
=========

定义所有层级智能体的通用接口和通信协议。
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import time


class AgentPriority(Enum):
    """智能体优先级"""
    SAFETY = 0      # 最高优先级 (L1)
    TACTICAL = 1    # 战术层 (L2)
    STRATEGIC = 2   # 战略层 (L3)
    ADVISORY = 3    # 建议级


class MessageType(Enum):
    """消息类型"""
    COMMAND = auto()       # 控制命令
    STATE = auto()         # 状态更新
    ALERT = auto()         # 报警
    REQUEST = auto()       # 请求
    RESPONSE = auto()      # 响应
    HEARTBEAT = auto()     # 心跳
    COORDINATION = auto()  # 协调消息


@dataclass
class AgentState:
    """智能体状态"""
    agent_id: str
    priority: AgentPriority
    is_active: bool = True
    last_update: float = 0.0

    # 性能指标
    response_time: float = 0.0
    action_count: int = 0
    error_count: int = 0

    # 局部状态
    local_state: Dict = field(default_factory=dict)

    # 目标跟踪
    current_objective: str = ""
    objective_progress: float = 0.0


@dataclass
class AgentMessage:
    """智能体间通信消息"""
    msg_type: MessageType
    sender: str
    receiver: str  # 可以是 'broadcast'
    priority: AgentPriority
    timestamp: float

    # 消息内容
    payload: Dict = field(default_factory=dict)

    # 元数据
    sequence: int = 0
    requires_ack: bool = False
    ttl: float = 5.0  # 生存时间 (s)

    def is_expired(self, current_time: float) -> bool:
        """检查消息是否过期"""
        return current_time - self.timestamp > self.ttl


@dataclass
class ControlAction:
    """控制动作"""
    actuator_id: str
    action_type: str  # 'set', 'increment', 'rate'
    value: float
    priority: AgentPriority
    timestamp: float
    source_agent: str

    # 约束
    min_value: float = 0.0
    max_value: float = 1.0
    rate_limit: float = float('inf')

    # 执行状态
    executed: bool = False
    execution_time: float = 0.0


class BaseAgent(ABC):
    """
    智能体基类

    所有层级的智能体都继承此类，实现:
    - 感知-决策-执行循环
    - 消息通信
    - 状态管理
    """

    def __init__(self, agent_id: str, priority: AgentPriority):
        self.agent_id = agent_id
        self.priority = priority

        # 状态
        self.state = AgentState(
            agent_id=agent_id,
            priority=priority
        )

        # 消息队列
        self.inbox: deque = deque(maxlen=100)
        self.outbox: deque = deque(maxlen=100)

        # 动作缓冲
        self.pending_actions: List[ControlAction] = []

        # 感知数据
        self.observations: Dict = {}

        # 回调
        self.message_handlers: Dict[MessageType, Callable] = {}

        # 时间
        self.last_cycle_time = 0.0
        self.cycle_count = 0

        # 注册默认消息处理器
        self._register_default_handlers()

    def _register_default_handlers(self):
        """注册默认消息处理器"""
        self.message_handlers[MessageType.HEARTBEAT] = self._handle_heartbeat
        self.message_handlers[MessageType.STATE] = self._handle_state_update

    def _handle_heartbeat(self, msg: AgentMessage):
        """处理心跳消息"""
        # 发送心跳响应
        response = AgentMessage(
            msg_type=MessageType.RESPONSE,
            sender=self.agent_id,
            receiver=msg.sender,
            priority=self.priority,
            timestamp=time.time(),
            payload={'type': 'heartbeat_ack'}
        )
        self.outbox.append(response)

    def _handle_state_update(self, msg: AgentMessage):
        """处理状态更新消息"""
        # 更新本地观测
        if 'state' in msg.payload:
            self.observations.update(msg.payload['state'])

    def receive_message(self, msg: AgentMessage):
        """接收消息"""
        if msg.receiver == self.agent_id or msg.receiver == 'broadcast':
            self.inbox.append(msg)

    def send_message(self, msg: AgentMessage):
        """发送消息到发件箱"""
        msg.sender = self.agent_id
        msg.timestamp = time.time()
        self.outbox.append(msg)

    def broadcast(self, msg_type: MessageType, payload: Dict):
        """广播消息"""
        msg = AgentMessage(
            msg_type=msg_type,
            sender=self.agent_id,
            receiver='broadcast',
            priority=self.priority,
            timestamp=time.time(),
            payload=payload
        )
        self.outbox.append(msg)

    def process_messages(self):
        """处理收件箱中的消息"""
        current_time = time.time()

        while self.inbox:
            msg = self.inbox.popleft()

            # 检查是否过期
            if msg.is_expired(current_time):
                continue

            # 调用处理器
            handler = self.message_handlers.get(msg.msg_type)
            if handler:
                try:
                    handler(msg)
                except Exception as e:
                    self.state.error_count += 1

    def update_observations(self, observations: Dict):
        """更新感知数据"""
        self.observations.update(observations)
        self.state.last_update = time.time()

    @abstractmethod
    def perceive(self, system_state: Dict) -> Dict:
        """
        感知环节

        从系统状态中提取本智能体关注的信息
        """
        pass

    @abstractmethod
    def decide(self) -> List[ControlAction]:
        """
        决策环节

        基于当前观测生成控制动作
        """
        pass

    @abstractmethod
    def act(self, actions: List[ControlAction]) -> Dict:
        """
        执行环节

        执行控制动作并返回执行结果
        """
        pass

    def step(self, system_state: Dict) -> Dict:
        """
        执行一个感知-决策-执行循环

        Parameters:
            system_state: 系统当前状态

        Returns:
            执行结果
        """
        cycle_start = time.time()

        # 1. 处理消息
        self.process_messages()

        # 2. 感知
        self.observations = self.perceive(system_state)

        # 3. 决策
        actions = self.decide()
        self.pending_actions = actions

        # 4. 执行
        result = self.act(actions)

        # 更新统计
        self.last_cycle_time = time.time() - cycle_start
        self.state.response_time = self.last_cycle_time
        self.state.action_count += len(actions)
        self.cycle_count += 1

        return result

    def get_state(self) -> AgentState:
        """获取智能体状态"""
        return self.state

    def set_objective(self, objective: str):
        """设置当前目标"""
        self.state.current_objective = objective
        self.state.objective_progress = 0.0

    def update_progress(self, progress: float):
        """更新目标进度"""
        self.state.objective_progress = np.clip(progress, 0.0, 1.0)

    def reset(self):
        """重置智能体"""
        self.inbox.clear()
        self.outbox.clear()
        self.pending_actions.clear()
        self.observations.clear()
        self.state.action_count = 0
        self.state.error_count = 0
        self.cycle_count = 0


class AgentRegistry:
    """智能体注册表"""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent):
        """注册智能体"""
        self.agents[agent.agent_id] = agent

    def unregister(self, agent_id: str):
        """注销智能体"""
        if agent_id in self.agents:
            del self.agents[agent_id]

    def get(self, agent_id: str) -> Optional[BaseAgent]:
        """获取智能体"""
        return self.agents.get(agent_id)

    def get_by_priority(self, priority: AgentPriority) -> List[BaseAgent]:
        """按优先级获取智能体"""
        return [a for a in self.agents.values() if a.priority == priority]

    def broadcast(self, msg: AgentMessage):
        """向所有智能体广播消息"""
        for agent in self.agents.values():
            if agent.agent_id != msg.sender:
                agent.receive_message(msg)

    def route_messages(self):
        """路由消息"""
        # 收集所有发件箱消息
        all_messages = []
        for agent in self.agents.values():
            while agent.outbox:
                all_messages.append(agent.outbox.popleft())

        # 分发消息
        for msg in all_messages:
            if msg.receiver == 'broadcast':
                self.broadcast(msg)
            else:
                receiver = self.agents.get(msg.receiver)
                if receiver:
                    receiver.receive_message(msg)
