"""
L1反射层智能体
==============

毫秒级安全响应:
- 硬约束规则引擎
- 条件-动作对
- 优先级仲裁
- 无需优化计算
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import time

from .base_agent import (
    BaseAgent, AgentPriority, AgentState, AgentMessage,
    ControlAction, MessageType
)
from ..config.settings import Config


class RuleCategory(Enum):
    """规则类别"""
    EMERGENCY = auto()      # 紧急停机
    PROTECTION = auto()     # 保护动作
    INTERLOCK = auto()      # 联锁逻辑
    LIMIT = auto()          # 限幅控制


class RuleStatus(Enum):
    """规则状态"""
    INACTIVE = auto()
    TRIGGERED = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    BLOCKED = auto()


@dataclass
class SafetyRule:
    """安全规则"""
    rule_id: str
    category: RuleCategory
    description: str

    # 触发条件
    condition: Callable[[Dict], bool]

    # 动作
    action: Callable[[Dict], List[ControlAction]]

    # 优先级 (0最高)
    priority: int = 0

    # 状态
    status: RuleStatus = RuleStatus.INACTIVE
    trigger_count: int = 0
    last_trigger: float = 0.0

    # 时序约束
    min_interval: float = 0.0  # 最小触发间隔
    hold_time: float = 0.0     # 动作保持时间

    # 联锁
    requires: List[str] = field(default_factory=list)   # 前置条件
    inhibits: List[str] = field(default_factory=list)   # 抑制规则


class ReflexAgent(BaseAgent):
    """
    L1反射层智能体

    实现确定性的条件-动作规则:
    - 毫秒级响应
    - 无优化计算
    - 基于预定义规则
    """

    def __init__(self, agent_id: str = "L1_reflex"):
        super().__init__(agent_id, AgentPriority.SAFETY)

        self.cfg = Config

        # 规则库
        self.rules: Dict[str, SafetyRule] = {}

        # 活跃规则
        self.active_rules: List[str] = []

        # 规则日志
        self.rule_log: List[Dict] = []
        self.max_log_size = 1000

        # 初始化默认规则
        self._init_default_rules()

    def _init_default_rules(self):
        """初始化默认安全规则"""

        # R1: 稳流池低水位紧急关闸
        self.add_rule(SafetyRule(
            rule_id="R1_pool_low_level",
            category=RuleCategory.EMERGENCY,
            description="稳流池水位过低，紧急关闭出水闸门",
            condition=lambda s: s.get('pool_level', 5.0) < Config.pool.min_level,
            action=lambda s: [
                ControlAction(
                    actuator_id='valve_pool_out',
                    action_type='set',
                    value=0.0,
                    priority=AgentPriority.SAFETY,
                    timestamp=time.time(),
                    source_agent='L1_reflex'
                )
            ],
            priority=0,
            hold_time=30.0
        ))

        # R2: 稳流池高水位溢流报警
        self.add_rule(SafetyRule(
            rule_id="R2_pool_high_level",
            category=RuleCategory.PROTECTION,
            description="稳流池水位过高，开启溢流",
            condition=lambda s: s.get('pool_level', 5.0) > Config.pool.warning_high,
            action=lambda s: [
                ControlAction(
                    actuator_id='valve_overflow',
                    action_type='set',
                    value=1.0,
                    priority=AgentPriority.SAFETY,
                    timestamp=time.time(),
                    source_agent='L1_reflex'
                )
            ],
            priority=1
        ))

        # R3: 管道压力超限保护
        self.add_rule(SafetyRule(
            rule_id="R3_pipe_overpressure",
            category=RuleCategory.PROTECTION,
            description="管道压力超限，打开泄压阀",
            condition=lambda s: s.get('pipe_pressure', 50.0) > Config.pipeline.max_pressure,
            action=lambda s: [
                ControlAction(
                    actuator_id='valve_relief',
                    action_type='set',
                    value=1.0,
                    priority=AgentPriority.SAFETY,
                    timestamp=time.time(),
                    source_agent='L1_reflex'
                )
            ],
            priority=0,
            hold_time=10.0
        ))

        # R4: 负压保护（防止管道抽空）
        self.add_rule(SafetyRule(
            rule_id="R4_negative_pressure",
            category=RuleCategory.PROTECTION,
            description="检测到负压，打开进气阀",
            condition=lambda s: s.get('pipe_pressure', 50.0) < 0.5,
            action=lambda s: [
                ControlAction(
                    actuator_id='valve_air',
                    action_type='set',
                    value=1.0,
                    priority=AgentPriority.SAFETY,
                    timestamp=time.time(),
                    source_agent='L1_reflex'
                )
            ],
            priority=0
        ))

        # R5: 调压井水位极限保护
        self.add_rule(SafetyRule(
            rule_id="R5_surge_tank_overflow",
            category=RuleCategory.PROTECTION,
            description="调压井水位过高，减小闸门开度",
            condition=lambda s: s.get('surge_tank_level', 20.0) > Config.surge_tank.max_level - 1.0,
            action=lambda s: [
                ControlAction(
                    actuator_id='gate_main',
                    action_type='increment',
                    value=-0.1,
                    priority=AgentPriority.SAFETY,
                    timestamp=time.time(),
                    source_agent='L1_reflex',
                    min_value=0.0,
                    max_value=1.0
                )
            ],
            priority=1,
            min_interval=5.0
        ))

        # R6: 隧洞流速超限
        self.add_rule(SafetyRule(
            rule_id="R6_tunnel_velocity",
            category=RuleCategory.LIMIT,
            description="隧洞流速超限，限制进水闸开度",
            condition=lambda s: s.get('tunnel_velocity', 2.0) > Config.tunnel.max_velocity,
            action=lambda s: [
                ControlAction(
                    actuator_id='gate_intake',
                    action_type='increment',
                    value=-0.05,
                    priority=AgentPriority.SAFETY,
                    timestamp=time.time(),
                    source_agent='L1_reflex',
                    rate_limit=0.01
                )
            ],
            priority=2,
            min_interval=2.0
        ))

        # R7: 联锁 - 进水闸与出水闸的开度比例
        self.add_rule(SafetyRule(
            rule_id="R7_gate_interlock",
            category=RuleCategory.INTERLOCK,
            description="出水闸开度不得超过进水闸",
            condition=lambda s: (
                s.get('gate_out_opening', 0) > s.get('gate_in_opening', 1) + 0.1
            ),
            action=lambda s: [
                ControlAction(
                    actuator_id='gate_out',
                    action_type='set',
                    value=s.get('gate_in_opening', 1),
                    priority=AgentPriority.SAFETY,
                    timestamp=time.time(),
                    source_agent='L1_reflex'
                )
            ],
            priority=1
        ))

        # R8: 流量突变检测（可能爆管）
        self.add_rule(SafetyRule(
            rule_id="R8_flow_surge",
            category=RuleCategory.EMERGENCY,
            description="检测到流量突变，启动紧急程序",
            condition=lambda s: abs(s.get('flow_rate_of_change', 0)) > 2.0,
            action=lambda s: self._emergency_flow_change_action(s),
            priority=0,
            hold_time=60.0
        ))

    def _emergency_flow_change_action(self, state: Dict) -> List[ControlAction]:
        """流量突变紧急动作"""
        actions = []

        # 分段关闭阀门
        for i in range(3):
            actions.append(ControlAction(
                actuator_id=f'valve_section_{i}',
                action_type='set',
                value=0.3,  # 先关到30%
                priority=AgentPriority.SAFETY,
                timestamp=time.time(),
                source_agent='L1_reflex',
                rate_limit=0.02  # 两段关闭
            ))

        # 广播报警
        self.broadcast(MessageType.ALERT, {
            'type': 'flow_surge',
            'severity': 'critical',
            'message': '检测到流量突变，可能发生爆管'
        })

        return actions

    def add_rule(self, rule: SafetyRule):
        """添加规则"""
        self.rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str):
        """移除规则"""
        if rule_id in self.rules:
            del self.rules[rule_id]

    def perceive(self, system_state: Dict) -> Dict:
        """感知系统状态"""
        # L1层关注的关键变量
        relevant_keys = [
            'pool_level', 'pipe_pressure', 'surge_tank_level',
            'tunnel_velocity', 'tunnel_flow', 'pipe_flow',
            'gate_in_opening', 'gate_out_opening',
            'flow_rate_of_change', 'is_ice_period'
        ]

        observations = {}
        for key in relevant_keys:
            if key in system_state:
                observations[key] = system_state[key]

        return observations

    def decide(self) -> List[ControlAction]:
        """基于规则决策"""
        current_time = time.time()
        triggered_rules: List[Tuple[int, SafetyRule]] = []

        # 检查所有规则
        for rule_id, rule in self.rules.items():
            # 检查时间间隔
            if rule.min_interval > 0:
                if current_time - rule.last_trigger < rule.min_interval:
                    continue

            # 检查前置条件
            if rule.requires:
                if not all(r in self.active_rules for r in rule.requires):
                    continue

            # 检查抑制条件
            if rule.inhibits:
                if any(r in self.active_rules for r in rule.inhibits):
                    continue

            # 评估触发条件
            try:
                if rule.condition(self.observations):
                    triggered_rules.append((rule.priority, rule))
                    rule.status = RuleStatus.TRIGGERED
                else:
                    # 检查是否需要保持
                    if rule.status == RuleStatus.EXECUTING:
                        if current_time - rule.last_trigger < rule.hold_time:
                            triggered_rules.append((rule.priority, rule))
                        else:
                            rule.status = RuleStatus.COMPLETED
                    else:
                        rule.status = RuleStatus.INACTIVE
            except Exception as e:
                rule.status = RuleStatus.BLOCKED
                self.state.error_count += 1

        # 按优先级排序
        triggered_rules.sort(key=lambda x: x[0])

        # 生成动作
        all_actions: List[ControlAction] = []
        self.active_rules = []

        for priority, rule in triggered_rules:
            try:
                actions = rule.action(self.observations)
                all_actions.extend(actions)

                rule.status = RuleStatus.EXECUTING
                rule.trigger_count += 1
                rule.last_trigger = current_time
                self.active_rules.append(rule.rule_id)

                # 记录日志
                self._log_rule(rule)

            except Exception as e:
                rule.status = RuleStatus.BLOCKED
                self.state.error_count += 1

        # 动作冲突仲裁
        resolved_actions = self._resolve_conflicts(all_actions)

        return resolved_actions

    def _resolve_conflicts(self, actions: List[ControlAction]) -> List[ControlAction]:
        """解决动作冲突"""
        # 按执行器分组
        by_actuator: Dict[str, List[ControlAction]] = {}

        for action in actions:
            if action.actuator_id not in by_actuator:
                by_actuator[action.actuator_id] = []
            by_actuator[action.actuator_id].append(action)

        # 对每个执行器，选择最高优先级动作
        resolved = []
        for actuator_id, acts in by_actuator.items():
            # 按优先级排序 (SAFETY < TACTICAL < STRATEGIC)
            acts.sort(key=lambda a: a.priority.value)
            resolved.append(acts[0])

        return resolved

    def act(self, actions: List[ControlAction]) -> Dict:
        """执行动作（返回动作列表供上层执行）"""
        result = {
            'agent': self.agent_id,
            'actions': [],
            'active_rules': self.active_rules.copy()
        }

        for action in actions:
            result['actions'].append({
                'actuator': action.actuator_id,
                'type': action.action_type,
                'value': action.value,
                'priority': action.priority.name
            })

        return result

    def _log_rule(self, rule: SafetyRule):
        """记录规则触发"""
        log_entry = {
            'timestamp': time.time(),
            'rule_id': rule.rule_id,
            'category': rule.category.name,
            'description': rule.description,
            'observations': self.observations.copy()
        }

        self.rule_log.append(log_entry)

        # 限制日志大小
        if len(self.rule_log) > self.max_log_size:
            self.rule_log = self.rule_log[-self.max_log_size:]

    def get_active_rules(self) -> List[str]:
        """获取当前活跃规则"""
        return self.active_rules.copy()

    def get_rule_statistics(self) -> Dict:
        """获取规则统计"""
        stats = {
            'total_rules': len(self.rules),
            'active_rules': len(self.active_rules),
            'by_category': {},
            'top_triggered': []
        }

        # 按类别统计
        for cat in RuleCategory:
            stats['by_category'][cat.name] = sum(
                1 for r in self.rules.values() if r.category == cat
            )

        # 触发次数最多的规则
        sorted_rules = sorted(
            self.rules.values(),
            key=lambda r: r.trigger_count,
            reverse=True
        )[:5]

        for rule in sorted_rules:
            stats['top_triggered'].append({
                'rule_id': rule.rule_id,
                'count': rule.trigger_count
            })

        return stats

    def reset(self):
        """重置"""
        super().reset()
        self.active_rules.clear()
        self.rule_log.clear()

        for rule in self.rules.values():
            rule.status = RuleStatus.INACTIVE
            rule.trigger_count = 0
            rule.last_trigger = 0.0
