"""
L5级泵站群多智能体集成系统 v1.0
=================================

将泵站群智能体深度融合到L5自主系统:

集成架构:
┌─────────────────────────────────────────────────────────────┐
│                   L5PumpGroupSystem                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  态势感知    │──│  决策规划    │──│     执行控制        │  │
│  │  Awareness  │  │  Planning   │  │    Execution        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│        │               │                    │                │
│        ▼               ▼                    ▼                │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              泵站群智能体层 (PumpGroup Layer)            ││
│  │  ┌───────────┐  ┌───────────┐  ┌───────────────────┐    ││
│  │  │ L1 Safety │  │L3 Economic│  │   Coordinator     │    ││
│  │  │ 安全规则  │  │ 经济优化  │  │   协调融合        │    ││
│  │  └───────────┘  └───────────┘  └───────────────────┘    ││
│  └─────────────────────────────────────────────────────────┘│
│        │               │                    │                │
│        ▼               ▼                    ▼                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  协调管理    │  │  学习优化    │  │   泵站群状态       │  │
│  │ Coordination│  │  Learning   │  │  PumpGroupState    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

关键集成点:
1. 态势感知 → 泵站群状态监测
2. 决策规划 → 经济优化目标融入
3. 执行控制 → 安全规则优先执行
4. 协调管理 → 安全与经济目标权衡
5. 学习优化 → 泵站运行模式学习

版本: 1.0
"""

import time
import math
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any
from datetime import datetime
from collections import deque

# L5自主系统
from .l5_autonomous import (
    L5AutonomousSystem, L5SystemState, L5Decision,
    AutonomyLevel, AgentRole, AgentStatus,
    SituationAwarenessAgent, DecisionPlanningAgent,
    ExecutionControlAgent, CoordinationAgent, LearningAgent,
    BaseAgent, AgentMessage
)

# 泵站群智能体
from .pump_group_agents import (
    PumpGroupSafetyAgent, PumpGroupEconomicAgent, PumpGroupCoordinatorAgent,
    PumpGroupMultiAgentSystem, PumpGroupState, StationState, PumpState,
    PumpStatus, ProtectionType, create_pump_group_system
)

# 场景系统
from ..scenarios.scenario_database import (
    ScenarioType, ScenarioCategory, ScenarioSeverity, OperationMode
)


# ============================================================
# 1. 集成数据结构
# ============================================================

class IntegrationMode(Enum):
    """集成模式"""
    STANDALONE = auto()      # 独立运行 (仅泵站群系统)
    EMBEDDED = auto()        # 嵌入式 (泵站群作为L5子系统)
    DISTRIBUTED = auto()     # 分布式 (多个L5系统协调)


@dataclass
class PumpGroupSituation:
    """泵站群态势"""
    timestamp: datetime = field(default_factory=datetime.now)

    # 泵站群状态
    total_stations: int = 0
    total_pumps: int = 0
    running_pumps: int = 0
    fault_pumps: int = 0

    # 水力状态
    total_flow: float = 0.0
    average_forebay_level: float = 0.0
    min_forebay_level: float = float('inf')
    max_forebay_level: float = 0.0

    # 电力状态
    total_power: float = 0.0
    current_electricity_price: float = 0.5

    # 风险评估
    safety_risk_level: float = 0.0
    economic_efficiency: float = 1.0
    overall_health: float = 1.0

    # 活跃场景
    active_safety_rules: List[str] = field(default_factory=list)
    active_scenarios: List[str] = field(default_factory=list)


@dataclass
class IntegratedDecision:
    """集成决策"""
    decision_id: str
    source: str  # 'l5_core', 'pump_safety', 'pump_economic', 'pump_coordinator'
    priority: int
    decision_type: str
    actions: List[Dict[str, Any]]
    rationale: str
    confidence: float
    safety_approved: bool = True
    economic_impact: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================
# 2. 泵站群态势感知扩展
# ============================================================

class PumpGroupAwarenessExtension:
    """
    泵站群态势感知扩展

    扩展L5态势感知智能体，增加泵站群专项感知能力
    """

    def __init__(self, pump_system: PumpGroupMultiAgentSystem):
        self.pump_system = pump_system
        self.situation_history: deque = deque(maxlen=1000)

        # 阈值参数
        self.thresholds = {
            'forebay_critical_low': 0.8,
            'forebay_critical_high': 5.2,
            'efficiency_warning': 0.6,
            'power_peak_threshold': 5000,  # kW
        }

    def perceive(self, raw_data: Dict[str, Any]) -> PumpGroupSituation:
        """
        感知泵站群态势

        Args:
            raw_data: 原始传感器数据

        Returns:
            PumpGroupSituation: 泵站群态势
        """
        situation = PumpGroupSituation()
        state = self.pump_system.pump_group_state

        # 基本统计
        situation.total_stations = len(state.stations)

        total_pumps = 0
        running_pumps = 0
        fault_pumps = 0
        forebay_levels = []

        for station in state.stations.values():
            total_pumps += station.total_pump_count
            running_pumps += station.running_pump_count
            forebay_levels.append(station.forebay_level)

            for pump in station.pumps.values():
                if pump.status == PumpStatus.FAULT:
                    fault_pumps += 1

        situation.total_pumps = total_pumps
        situation.running_pumps = running_pumps
        situation.fault_pumps = fault_pumps

        # 水力状态
        situation.total_flow = state.total_flow
        if forebay_levels:
            situation.average_forebay_level = sum(forebay_levels) / len(forebay_levels)
            situation.min_forebay_level = min(forebay_levels)
            situation.max_forebay_level = max(forebay_levels)

        # 电力状态
        situation.total_power = state.total_power
        situation.current_electricity_price = state.electricity_price

        # 风险评估
        situation.safety_risk_level = self._assess_safety_risk(state)
        situation.economic_efficiency = self._assess_economic_efficiency(state)
        situation.overall_health = self._assess_overall_health(state)

        # 活跃规则和场景
        situation.active_safety_rules = self.pump_system.safety_agent.active_rules.copy()

        # 记录历史
        self.situation_history.append(situation)

        return situation

    def _assess_safety_risk(self, state: PumpGroupState) -> float:
        """评估安全风险等级 (0-5)"""
        risk = 0.0

        for station in state.stations.values():
            # 前池水位风险
            if station.forebay_level < self.thresholds['forebay_critical_low']:
                risk += 3.0
            elif station.forebay_level < 1.5:
                risk += 1.5

            if station.forebay_level > self.thresholds['forebay_critical_high']:
                risk += 3.0
            elif station.forebay_level > 4.5:
                risk += 1.5

            # 泵故障风险
            for pump in station.pumps.values():
                if pump.status == PumpStatus.FAULT:
                    risk += 1.0
                if pump.vibration_level > 4.5:
                    risk += 0.5
                if pump.bearing_temp > 70:
                    risk += 0.5

        return min(risk, 5.0)

    def _assess_economic_efficiency(self, state: PumpGroupState) -> float:
        """评估经济效率 (0-1)"""
        if state.total_running_pumps == 0:
            return 1.0

        # 简化：基于当前电价和运行台数
        # 谷时效率高，峰时效率低
        price_factor = 1.0 - (state.electricity_price - 0.35) / 0.5
        return max(0.3, min(1.0, price_factor))

    def _assess_overall_health(self, state: PumpGroupState) -> float:
        """评估整体健康度 (0-1)"""
        health = 1.0

        for station in state.stations.values():
            # 可用率
            if station.total_pump_count > 0:
                availability = station.available_pump_count / station.total_pump_count
                health *= availability

        return max(0.0, health)

    def predict_trends(self, horizon_hours: int = 4) -> Dict[str, Any]:
        """预测趋势"""
        if len(self.situation_history) < 5:
            return {'status': 'insufficient_data'}

        # 分析最近的态势变化
        recent = list(self.situation_history)[-10:]

        flow_trend = self._calculate_trend([s.total_flow for s in recent])
        power_trend = self._calculate_trend([s.total_power for s in recent])
        level_trend = self._calculate_trend([s.average_forebay_level for s in recent])

        return {
            'flow_trend': flow_trend,
            'power_trend': power_trend,
            'level_trend': level_trend,
            'prediction_horizon': horizon_hours
        }

    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """计算趋势"""
        if len(values) < 2:
            return {'direction': 'stable', 'rate': 0.0}

        # 简单线性回归
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if abs(denominator) < 1e-10:
            slope = 0.0
        else:
            slope = numerator / denominator

        return {
            'direction': 'up' if slope > 0.01 else ('down' if slope < -0.01 else 'stable'),
            'rate': abs(slope)
        }


# ============================================================
# 3. 泵站群决策规划扩展
# ============================================================

class PumpGroupPlanningExtension:
    """
    泵站群决策规划扩展

    扩展L5决策规划智能体，融入泵站群经济优化
    """

    def __init__(self, pump_system: PumpGroupMultiAgentSystem):
        self.pump_system = pump_system
        self.economic_agent = pump_system.economic_agent
        self.decision_counter = 0

    def generate_economic_decisions(
        self,
        situation: PumpGroupSituation,
        constraints: Dict[str, Any]
    ) -> List[IntegratedDecision]:
        """
        生成经济优化决策

        Args:
            situation: 泵站群态势
            constraints: 约束条件

        Returns:
            List[IntegratedDecision]: 决策列表
        """
        decisions = []
        self.decision_counter += 1

        # 触发经济优化
        target_flow = constraints.get('target_flow', 15.0)
        self.economic_agent.last_optimization_time = 0  # 强制重新优化
        self.pump_system.step({
            'hour': situation.timestamp.hour,
            'target_flow': target_flow
        })

        # 从优化结果生成决策
        schedule = self.economic_agent.optimal_schedule
        if schedule:
            current_schedule = schedule[0]

            for station_id, station_data in current_schedule.get('stations', {}).items():
                optimal_count = station_data['optimal_pump_count']
                current_count = self.pump_system.pump_group_state.stations.get(
                    station_id, StationState(station_id=station_id, station_name=station_id)
                ).running_pump_count

                if optimal_count != current_count:
                    action_type = 'increase_pumps' if optimal_count > current_count else 'decrease_pumps'
                    diff = abs(optimal_count - current_count)

                    decisions.append(IntegratedDecision(
                        decision_id=f"ECON_{self.decision_counter}_{station_id}",
                        source='pump_economic',
                        priority=5,  # 经济决策优先级较低
                        decision_type='optimization',
                        actions=[{
                            'station_id': station_id,
                            'action': action_type,
                            'count': diff,
                            'target_count': optimal_count
                        }],
                        rationale=f"经济优化: {station_id} 调整到 {optimal_count} 台运行",
                        confidence=0.85,
                        safety_approved=True,  # 待安全审核
                        economic_impact=station_data.get('energy_cost', 0)
                    ))

        return decisions

    def optimize_for_price(self, current_hour: int) -> Dict[str, Any]:
        """
        根据电价优化

        Args:
            current_hour: 当前小时

        Returns:
            优化建议
        """
        price = self.economic_agent._get_current_price(current_hour)

        if price >= 0.8:  # 峰时
            strategy = 'reduce_load'
            recommendation = '峰时电价，建议减少运行泵数'
        elif price <= 0.4:  # 谷时
            strategy = 'increase_load'
            recommendation = '谷时电价，建议增加输水量'
        else:
            strategy = 'maintain'
            recommendation = '平时电价，维持正常运行'

        return {
            'hour': current_hour,
            'price': price,
            'strategy': strategy,
            'recommendation': recommendation
        }


# ============================================================
# 4. 泵站群执行控制扩展
# ============================================================

class PumpGroupExecutionExtension:
    """
    泵站群执行控制扩展

    扩展L5执行控制智能体，集成安全规则优先执行
    """

    def __init__(self, pump_system: PumpGroupMultiAgentSystem):
        self.pump_system = pump_system
        self.safety_agent = pump_system.safety_agent
        self.execution_log: deque = deque(maxlen=1000)

    def safety_check(self, decisions: List[IntegratedDecision]) -> List[IntegratedDecision]:
        """
        安全检查

        对所有决策进行安全审核

        Args:
            decisions: 待检查的决策列表

        Returns:
            List[IntegratedDecision]: 审核后的决策列表
        """
        checked_decisions = []

        for decision in decisions:
            # 检查是否违反安全规则
            is_safe, violations = self._check_safety_violations(decision)

            if is_safe:
                decision.safety_approved = True
                checked_decisions.append(decision)
            else:
                # 安全违规，阻止或修改决策
                decision.safety_approved = False
                decision.rationale += f" [BLOCKED: {', '.join(violations)}]"

                # 如果是紧急安全决策，仍然执行
                if decision.source == 'pump_safety':
                    decision.safety_approved = True
                    checked_decisions.append(decision)

        return checked_decisions

    def _check_safety_violations(self, decision: IntegratedDecision) -> Tuple[bool, List[str]]:
        """检查安全违规"""
        violations = []

        for action in decision.actions:
            station_id = action.get('station_id', '')
            action_type = action.get('action', '')

            if station_id in self.pump_system.pump_group_state.stations:
                station = self.pump_system.pump_group_state.stations[station_id]

                # 检查前池水位
                if action_type == 'increase_pumps':
                    if station.forebay_level < 1.5:
                        violations.append(f'{station_id}: 前池水位过低，不宜增加泵数')

                if action_type == 'decrease_pumps':
                    if station.forebay_level > 4.5:
                        violations.append(f'{station_id}: 前池水位过高，不宜减少泵数')

        return len(violations) == 0, violations

    def execute_with_safety(
        self,
        decisions: List[IntegratedDecision]
    ) -> List[Dict[str, Any]]:
        """
        带安全保护的执行

        Args:
            decisions: 决策列表

        Returns:
            执行结果列表
        """
        results = []

        # 按优先级排序（数值小优先级高）
        sorted_decisions = sorted(decisions, key=lambda d: d.priority)

        for decision in sorted_decisions:
            if not decision.safety_approved:
                results.append({
                    'decision_id': decision.decision_id,
                    'status': 'blocked',
                    'reason': 'safety_violation'
                })
                continue

            # 执行决策
            result = self._execute_decision(decision)
            results.append(result)

            # 记录日志
            self.execution_log.append({
                'timestamp': datetime.now(),
                'decision': decision,
                'result': result
            })

        return results

    def _execute_decision(self, decision: IntegratedDecision) -> Dict[str, Any]:
        """执行单个决策"""
        result = {
            'decision_id': decision.decision_id,
            'status': 'success',
            'actions_executed': [],
            'errors': []
        }

        for action in decision.actions:
            try:
                action_result = self._execute_action(action)
                result['actions_executed'].append(action_result)
            except Exception as e:
                result['errors'].append({
                    'action': action,
                    'error': str(e)
                })
                result['status'] = 'partial'

        return result

    def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个动作"""
        return {
            'action': action,
            'executed': True,
            'timestamp': datetime.now().isoformat()
        }

    def get_safety_override_actions(self) -> List[IntegratedDecision]:
        """
        获取安全覆盖动作

        当安全规则触发时，生成覆盖决策
        """
        override_decisions = []

        # 运行安全智能体
        self.pump_system.step({
            'hour': datetime.now().hour,
            'target_flow': 15.0
        })

        # 检查活跃的安全规则
        active_rules = self.safety_agent.active_rules

        if active_rules:
            override_decisions.append(IntegratedDecision(
                decision_id=f"SAFETY_OVERRIDE_{int(time.time())}",
                source='pump_safety',
                priority=0,  # 最高优先级
                decision_type='safety',
                actions=[{
                    'type': 'safety_override',
                    'active_rules': active_rules
                }],
                rationale=f"安全规则触发: {', '.join(active_rules)}",
                confidence=1.0,
                safety_approved=True
            ))

        return override_decisions


# ============================================================
# 5. L5泵站群集成系统
# ============================================================

class L5PumpGroupSystem(L5AutonomousSystem):
    """
    L5级泵站群集成系统

    继承L5AutonomousSystem，深度融合泵站群智能体
    """

    def __init__(self, station_configs: List[Dict] = None):
        # 初始化基础L5系统
        super().__init__()

        # 创建泵站群系统
        self.pump_system = create_pump_group_system(station_configs)

        # 创建扩展模块
        self.pump_awareness = PumpGroupAwarenessExtension(self.pump_system)
        self.pump_planning = PumpGroupPlanningExtension(self.pump_system)
        self.pump_execution = PumpGroupExecutionExtension(self.pump_system)

        # 集成模式
        self.integration_mode = IntegrationMode.EMBEDDED

        # 统计
        self.integrated_cycle_count = 0
        self.safety_override_count = 0
        self.economic_decisions_count = 0

    def process_cycle(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行集成处理周期

        Args:
            sensor_data: 传感器数据 (包含泵站群数据)

        Returns:
            周期结果
        """
        cycle_start = time.time()
        self.integrated_cycle_count += 1

        # ===== 1. 态势感知 (集成) =====

        # L5核心态势感知
        core_situation = self.awareness_agent.process(sensor_data)

        # 泵站群态势感知
        pump_situation = self.pump_awareness.perceive(sensor_data)

        # 融合态势
        integrated_situation = self._fuse_situations(core_situation, pump_situation)

        # ===== 2. 决策规划 (集成) =====

        # 获取安全覆盖决策 (最高优先级)
        safety_decisions = self.pump_execution.get_safety_override_actions()

        if safety_decisions:
            self.safety_override_count += 1

        # L5核心决策
        core_planning = self.planning_agent.process({
            'situation': core_situation,
            'constraints': self._get_constraints()
        })
        core_decisions = core_planning.get('decisions', [])

        # 泵站群经济优化决策
        economic_decisions = self.pump_planning.generate_economic_decisions(
            pump_situation,
            {'target_flow': sensor_data.get('target_flow', 15.0)}
        )
        self.economic_decisions_count += len(economic_decisions)

        # 合并所有决策
        all_decisions = safety_decisions + self._convert_l5_decisions(core_decisions) + economic_decisions

        # ===== 3. 安全检查 =====
        checked_decisions = self.pump_execution.safety_check(all_decisions)

        # ===== 4. 执行控制 =====
        if not self.state.human_override:
            execution_results = self.pump_execution.execute_with_safety(checked_decisions)
        else:
            execution_results = [{'status': 'skipped', 'reason': 'human_override'}]

        # ===== 5. 协调管理 =====
        coordination_result = self.coordination_agent.process({
            'agent_reports': self._collect_agent_reports(),
            'decisions': core_decisions
        })

        # 泵站群协调状态
        pump_coordination = self.pump_system.coordinator.get_coordination_status()

        # ===== 6. 学习优化 =====
        learning_result = self.learning_agent.process({
            'situation': integrated_situation,
            'decisions': all_decisions,
            'execution': execution_results,
            'pump_status': self.pump_system.get_system_status()
        })

        # ===== 7. 更新状态 =====
        self.state.last_update = datetime.now()
        self._update_system_health()

        cycle_time = time.time() - cycle_start

        return {
            'cycle_time': cycle_time,
            'cycle_count': self.integrated_cycle_count,

            # 态势
            'core_situation': core_situation,
            'pump_situation': pump_situation.__dict__,
            'integrated_situation': integrated_situation,

            # 决策
            'safety_decisions': len(safety_decisions),
            'economic_decisions': len(economic_decisions),
            'total_decisions': len(all_decisions),
            'approved_decisions': len([d for d in checked_decisions if d.safety_approved]),

            # 执行
            'execution_results': execution_results,

            # 协调
            'coordination': coordination_result,
            'pump_coordination': pump_coordination,

            # 学习
            'learning': learning_result,

            # 系统状态
            'system_state': self.get_integrated_state_summary()
        }

    def _fuse_situations(
        self,
        core_situation: Dict[str, Any],
        pump_situation: PumpGroupSituation
    ) -> Dict[str, Any]:
        """融合态势"""
        return {
            'timestamp': datetime.now().isoformat(),

            # 核心态势
            'core_risk_level': core_situation.get('risk_level', 0),
            'core_confidence': core_situation.get('confidence', 0.8),

            # 泵站群态势
            'pump_safety_risk': pump_situation.safety_risk_level,
            'pump_economic_efficiency': pump_situation.economic_efficiency,
            'pump_overall_health': pump_situation.overall_health,

            # 综合风险
            'integrated_risk': max(
                core_situation.get('risk_level', 0),
                pump_situation.safety_risk_level
            ),

            # 运行状态
            'running_pumps': pump_situation.running_pumps,
            'total_flow': pump_situation.total_flow,
            'total_power': pump_situation.total_power,

            # 活跃规则/场景
            'active_safety_rules': pump_situation.active_safety_rules,
            'active_scenarios': len(core_situation.get('active_scenarios', []))
        }

    def _convert_l5_decisions(self, l5_decisions: List[L5Decision]) -> List[IntegratedDecision]:
        """转换L5决策为集成决策"""
        converted = []
        for d in l5_decisions:
            converted.append(IntegratedDecision(
                decision_id=d.decision_id,
                source='l5_core',
                priority=d.priority,
                decision_type=d.decision_type,
                actions=d.actions,
                rationale=d.rationale,
                confidence=d.confidence,
                safety_approved=True
            ))
        return converted

    def get_integrated_state_summary(self) -> Dict[str, Any]:
        """获取集成状态摘要"""
        base_summary = self.get_state_summary()

        # 添加泵站群状态
        pump_status = self.pump_system.get_system_status()

        return {
            **base_summary,
            'integration_mode': self.integration_mode.name,
            'integrated_cycles': self.integrated_cycle_count,
            'safety_overrides': self.safety_override_count,
            'economic_decisions': self.economic_decisions_count,
            'pump_stations': len(self.pump_system.pump_group_state.stations),
            'pump_status': pump_status
        }

    def get_pump_optimization_report(self) -> Dict[str, Any]:
        """获取泵站优化报告"""
        return {
            'optimization_summary': self.pump_system.economic_agent.get_optimization_summary(),
            'schedule': self.pump_system.economic_agent.optimal_schedule[:6] if self.pump_system.economic_agent.optimal_schedule else [],
            'safety_rules': self.pump_system.safety_agent.get_safety_status(),
            'coordination': self.pump_system.coordinator.get_coordination_status()
        }

    def set_target_flow(self, target_flow: float):
        """设置目标流量"""
        self._target_flow = target_flow

    def _get_constraints(self) -> Dict[str, Any]:
        """获取约束条件"""
        constraints = super()._get_constraints()
        constraints['target_flow'] = getattr(self, '_target_flow', 15.0)
        return constraints


# ============================================================
# 6. 便捷函数
# ============================================================

def create_l5_pump_group_system(
    station_configs: List[Dict] = None
) -> L5PumpGroupSystem:
    """
    创建L5泵站群集成系统

    Args:
        station_configs: 泵站配置，默认使用密云工程配置

    Returns:
        L5PumpGroupSystem: 集成系统实例
    """
    system = L5PumpGroupSystem(station_configs)
    system.initialize()
    return system


# ============================================================
# 导出
# ============================================================

__all__ = [
    # 枚举
    'IntegrationMode',

    # 数据结构
    'PumpGroupSituation',
    'IntegratedDecision',

    # 扩展模块
    'PumpGroupAwarenessExtension',
    'PumpGroupPlanningExtension',
    'PumpGroupExecutionExtension',

    # 集成系统
    'L5PumpGroupSystem',

    # 便捷函数
    'create_l5_pump_group_system',
]
