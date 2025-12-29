"""
L5级全自主运行系统
==================

实现SAE J3016定义的L5级自动驾驶理念在水利工程中的应用:
- L5: 全工况全自主,无需人工干预
- L4: 特定条件下全自主
- L3: 有条件自动,需要人工监督
- L2: 部分自动,人工主导
- L1: 辅助控制

多智能体架构:
1. 态势感知智能体 (Situation Awareness Agent)
2. 决策规划智能体 (Decision Planning Agent)
3. 执行控制智能体 (Execution Control Agent)
4. 协调管理智能体 (Coordination Agent)
5. 学习优化智能体 (Learning Agent)

版本: 3.4.0
"""

import math
import time
import threading
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any, Set
from datetime import datetime, timedelta
from collections import deque
from abc import ABC, abstractmethod

from ..scenarios.scenario_database import (
    ScenarioType, ScenarioCategory, ScenarioSeverity,
    OperationMode, ScenarioDefinition, ScenarioResponse,
    SCENARIO_DB
)
from ..scenarios.scenario_engine import (
    ScenarioState, ScenarioEngine
)


# ==========================================
# 智能等级定义
# ==========================================
class AutonomyLevel(Enum):
    """自主等级 (参考SAE J3016)"""
    L0_MANUAL = 0         # 全手动
    L1_ASSISTED = 1       # 辅助控制
    L2_PARTIAL = 2        # 部分自动
    L3_CONDITIONAL = 3    # 有条件自动
    L4_HIGH = 4           # 高度自动
    L5_FULL = 5           # 完全自主


class AgentRole(Enum):
    """智能体角色"""
    AWARENESS = auto()     # 态势感知
    PLANNING = auto()      # 决策规划
    EXECUTION = auto()     # 执行控制
    COORDINATION = auto()  # 协调管理
    LEARNING = auto()      # 学习优化


class AgentStatus(Enum):
    """智能体状态"""
    IDLE = auto()         # 空闲
    ACTIVE = auto()       # 活动
    BUSY = auto()         # 忙碌
    ERROR = auto()        # 错误
    DEGRADED = auto()     # 降级


# ==========================================
# 数据结构
# ==========================================
@dataclass
class L5SystemState:
    """L5系统状态"""
    autonomy_level: AutonomyLevel = AutonomyLevel.L5_FULL
    operation_mode: OperationMode = OperationMode.AUTO_L5
    system_health: float = 1.0  # 0-1
    confidence: float = 1.0  # 系统决策置信度
    active_scenarios: List[ScenarioState] = field(default_factory=list)
    current_objectives: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.now)
    human_override: bool = False
    degradation_level: int = 0  # 0=正常, 1-5=降级程度


@dataclass
class L5Decision:
    """L5决策"""
    decision_id: str
    decision_type: str  # control, safety, optimization, emergency
    priority: int
    actions: List[Dict[str, Any]]
    rationale: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    requires_confirmation: bool = False
    timeout: float = 60.0
    executed: bool = False


@dataclass
class AgentMessage:
    """智能体消息"""
    sender: AgentRole
    receiver: AgentRole
    message_type: str
    content: Dict[str, Any]
    priority: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


# ==========================================
# 基础智能体类
# ==========================================
class BaseAgent(ABC):
    """智能体基类"""
    
    def __init__(self, role: AgentRole, name: str):
        self.role = role
        self.name = name
        self.status = AgentStatus.IDLE
        self.message_queue: deque = deque(maxlen=1000)
        self.output_queue: deque = deque(maxlen=1000)
        self.last_cycle_time: float = 0.0
        self.cycle_count: int = 0
        self.error_count: int = 0
        
    @abstractmethod
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """处理输入并产生输出"""
        pass
    
    def receive_message(self, message: AgentMessage):
        """接收消息"""
        self.message_queue.append(message)
    
    def send_message(self, receiver: AgentRole, message_type: str, 
                    content: Dict[str, Any], priority: int = 0) -> AgentMessage:
        """发送消息"""
        msg = AgentMessage(
            sender=self.role,
            receiver=receiver,
            message_type=message_type,
            content=content,
            priority=priority
        )
        self.output_queue.append(msg)
        return msg
    
    def get_pending_messages(self) -> List[AgentMessage]:
        """获取待处理消息"""
        messages = list(self.message_queue)
        self.message_queue.clear()
        return messages


# ==========================================
# 态势感知智能体
# ==========================================
class SituationAwarenessAgent(BaseAgent):
    """
    态势感知智能体
    
    职责:
    - 实时数据融合
    - 异常检测
    - 趋势预测
    - 风险评估
    """
    
    def __init__(self):
        super().__init__(AgentRole.AWARENESS, "态势感知智能体")
        self.scenario_engine = ScenarioEngine()
        self.data_history: Dict[str, deque] = {}
        self.anomaly_threshold = 3.0  # 标准差
        self.prediction_horizon = 300  # 秒
        
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """处理传感器数据"""
        self.status = AgentStatus.ACTIVE
        start_time = time.time()
        
        try:
            # 数据融合
            fused_data = self._fuse_data(inputs)
            
            # 异常检测
            anomalies = self._detect_anomalies(fused_data)
            
            # 趋势预测
            trends = self._predict_trends(fused_data)
            
            # 场景识别
            scenarios = self.scenario_engine.update(fused_data)
            
            # 风险评估
            risk_assessment = self._assess_risks(scenarios, anomalies, trends)
            
            # 构建态势图
            situation = {
                'timestamp': datetime.now(),
                'fused_data': fused_data,
                'anomalies': anomalies,
                'trends': trends,
                'active_scenarios': scenarios,
                'risk_level': risk_assessment['overall_risk'],
                'risk_details': risk_assessment,
                'confidence': self._calculate_confidence(fused_data)
            }
            
            # 向决策智能体发送态势
            self.send_message(
                AgentRole.PLANNING,
                'situation_update',
                situation,
                priority=risk_assessment['overall_risk']
            )
            
            self.status = AgentStatus.IDLE
            self.last_cycle_time = time.time() - start_time
            self.cycle_count += 1
            
            return situation
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.error_count += 1
            return {'error': str(e)}
    
    def _fuse_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """数据融合"""
        fused = {}
        
        for key, value in inputs.items():
            # 记录历史
            if key not in self.data_history:
                self.data_history[key] = deque(maxlen=1000)
            self.data_history[key].append((time.time(), value))
            
            # 融合(简单取最新值,可扩展为卡尔曼滤波等)
            fused[key] = value
        
        return fused
    
    def _detect_anomalies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """异常检测"""
        anomalies = []
        
        for key, value in data.items():
            if not isinstance(value, (int, float)):
                continue
            
            if key in self.data_history and len(self.data_history[key]) > 10:
                values = [v for _, v in self.data_history[key] 
                         if isinstance(v, (int, float))]
                if values:
                    mean = sum(values) / len(values)
                    std = math.sqrt(sum((v - mean)**2 for v in values) / len(values))
                    
                    if std > 0 and abs(value - mean) > self.anomaly_threshold * std:
                        anomalies.append({
                            'parameter': key,
                            'value': value,
                            'mean': mean,
                            'std': std,
                            'z_score': (value - mean) / std
                        })
        
        return anomalies
    
    def _predict_trends(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """趋势预测"""
        trends = {}
        
        for key in data.keys():
            if key in self.data_history and len(self.data_history[key]) > 5:
                values = [(t, v) for t, v in self.data_history[key] 
                         if isinstance(v, (int, float))]
                if len(values) >= 5:
                    # 简单线性回归
                    n = len(values)
                    sum_t = sum(t for t, _ in values)
                    sum_v = sum(v for _, v in values)
                    sum_tv = sum(t * v for t, v in values)
                    sum_t2 = sum(t * t for t, _ in values)
                    
                    denom = n * sum_t2 - sum_t * sum_t
                    if abs(denom) > 1e-10:
                        slope = (n * sum_tv - sum_t * sum_v) / denom
                        trends[key] = {
                            'slope': slope,
                            'direction': 'up' if slope > 0 else 'down',
                            'rate': abs(slope)
                        }
        
        return trends
    
    def _assess_risks(self, scenarios: List[ScenarioState], 
                     anomalies: List[Dict], trends: Dict) -> Dict[str, Any]:
        """风险评估"""
        risk_factors = []
        
        # 场景风险
        for scenario in scenarios:
            scenario_def = SCENARIO_DB.get_scenario(scenario.scenario_type)
            if scenario_def:
                risk_factors.append(scenario_def.severity.value * scenario.confidence)
        
        # 异常风险
        for anomaly in anomalies:
            risk_factors.append(min(abs(anomaly['z_score']), 5))
        
        # 计算总体风险
        if risk_factors:
            overall_risk = min(sum(risk_factors) / len(risk_factors), 6)
        else:
            overall_risk = 0
        
        return {
            'overall_risk': overall_risk,
            'scenario_count': len(scenarios),
            'anomaly_count': len(anomalies),
            'risk_factors': risk_factors
        }
    
    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """计算置信度"""
        # 基于数据完整性和质量
        expected_params = ['flow_rate', 'pressure', 'water_level', 'valve_position']
        available = sum(1 for p in expected_params if p in data)
        return available / len(expected_params)


# ==========================================
# 决策规划智能体
# ==========================================
class DecisionPlanningAgent(BaseAgent):
    """
    决策规划智能体
    
    职责:
    - 目标管理
    - 策略选择
    - 路径规划
    - 资源分配
    """
    
    def __init__(self):
        super().__init__(AgentRole.PLANNING, "决策规划智能体")
        self.objectives: List[str] = ['maintain_flow', 'ensure_safety', 'optimize_efficiency']
        self.decision_history: deque = deque(maxlen=1000)
        self.decision_counter = 0
        
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """处理态势信息生成决策"""
        self.status = AgentStatus.ACTIVE
        start_time = time.time()
        
        try:
            situation = inputs.get('situation', {})
            constraints = inputs.get('constraints', {})
            
            # 分析态势
            analysis = self._analyze_situation(situation)
            
            # 生成候选策略
            candidates = self._generate_strategies(analysis, constraints)
            
            # 评估和选择最优策略
            best_strategy = self._select_strategy(candidates, analysis)
            
            # 生成决策
            decisions = self._generate_decisions(best_strategy, situation)
            
            # 向执行智能体发送决策
            for decision in decisions:
                self.send_message(
                    AgentRole.EXECUTION,
                    'execute_decision',
                    {'decision': decision.__dict__},
                    priority=decision.priority
                )
            
            result = {
                'decisions': decisions,
                'strategy': best_strategy,
                'analysis': analysis
            }
            
            self.status = AgentStatus.IDLE
            self.last_cycle_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.error_count += 1
            return {'error': str(e)}
    
    def _analyze_situation(self, situation: Dict[str, Any]) -> Dict[str, Any]:
        """分析态势"""
        risk_level = situation.get('risk_level', 0)
        scenarios = situation.get('active_scenarios', [])
        
        # 确定运行优先级
        if risk_level >= 5:
            priority = 'emergency'
        elif risk_level >= 3:
            priority = 'safety'
        elif risk_level >= 1:
            priority = 'attention'
        else:
            priority = 'normal'
        
        return {
            'priority': priority,
            'risk_level': risk_level,
            'scenario_count': len(scenarios),
            'requires_action': risk_level > 1
        }
    
    def _generate_strategies(self, analysis: Dict[str, Any], 
                            constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成候选策略"""
        strategies = []
        priority = analysis.get('priority', 'normal')
        
        if priority == 'emergency':
            strategies.append({
                'name': 'emergency_shutdown',
                'type': 'safety',
                'actions': ['close_inlet', 'open_relief', 'notify_all'],
                'risk_reduction': 5.0
            })
            strategies.append({
                'name': 'controlled_reduction',
                'type': 'safety',
                'actions': ['reduce_flow', 'monitor_pressure'],
                'risk_reduction': 3.0
            })
        elif priority == 'safety':
            strategies.append({
                'name': 'safe_operation',
                'type': 'safety',
                'actions': ['adjust_flow', 'increase_monitoring'],
                'risk_reduction': 2.0
            })
        else:
            strategies.append({
                'name': 'normal_operation',
                'type': 'optimization',
                'actions': ['maintain_setpoint', 'optimize_efficiency'],
                'risk_reduction': 0.0
            })
            strategies.append({
                'name': 'efficiency_boost',
                'type': 'optimization',
                'actions': ['optimize_valves', 'reduce_losses'],
                'risk_reduction': 0.0
            })
        
        return strategies
    
    def _select_strategy(self, candidates: List[Dict[str, Any]], 
                        analysis: Dict[str, Any]) -> Dict[str, Any]:
        """选择最优策略"""
        if not candidates:
            return {'name': 'hold', 'type': 'safety', 'actions': ['maintain_state']}
        
        # 根据优先级选择
        priority = analysis.get('priority', 'normal')
        
        for strategy in candidates:
            if priority == 'emergency' and strategy['type'] == 'safety':
                return strategy
        
        # 默认选择第一个
        return candidates[0]
    
    def _generate_decisions(self, strategy: Dict[str, Any], 
                           situation: Dict[str, Any]) -> List[L5Decision]:
        """生成决策"""
        decisions = []
        self.decision_counter += 1
        
        for i, action in enumerate(strategy.get('actions', [])):
            decision = L5Decision(
                decision_id=f"DEC_{self.decision_counter}_{i}",
                decision_type=strategy.get('type', 'control'),
                priority=10 if strategy.get('type') == 'safety' else 5,
                actions=[{'action': action, 'parameters': {}}],
                rationale=f"Strategy: {strategy.get('name')}",
                confidence=situation.get('confidence', 0.8),
                requires_confirmation=strategy.get('type') == 'emergency'
            )
            decisions.append(decision)
            self.decision_history.append(decision)
        
        return decisions


# ==========================================
# 执行控制智能体
# ==========================================
class ExecutionControlAgent(BaseAgent):
    """
    执行控制智能体
    
    职责:
    - 决策执行
    - 控制指令生成
    - 执行监督
    - 反馈收集
    """
    
    def __init__(self):
        super().__init__(AgentRole.EXECUTION, "执行控制智能体")
        self.pending_decisions: deque = deque(maxlen=100)
        self.execution_log: deque = deque(maxlen=1000)
        
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行决策"""
        self.status = AgentStatus.ACTIVE
        
        try:
            decisions = inputs.get('decisions', [])
            
            results = []
            for decision in decisions:
                result = self._execute_decision(decision)
                results.append(result)
                
                # 向协调智能体反馈
                self.send_message(
                    AgentRole.COORDINATION,
                    'execution_feedback',
                    {'decision_id': decision.decision_id, 'result': result}
                )
            
            self.status = AgentStatus.IDLE
            return {'execution_results': results}
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            return {'error': str(e)}
    
    def _execute_decision(self, decision: L5Decision) -> Dict[str, Any]:
        """执行单个决策"""
        start_time = time.time()
        
        execution_result = {
            'decision_id': decision.decision_id,
            'status': 'success',
            'actions_executed': [],
            'errors': [],
            'execution_time': 0
        }
        
        for action_spec in decision.actions:
            action = action_spec.get('action', '')
            params = action_spec.get('parameters', {})
            
            try:
                # 模拟执行
                action_result = self._execute_action(action, params)
                execution_result['actions_executed'].append({
                    'action': action,
                    'result': action_result
                })
            except Exception as e:
                execution_result['errors'].append({
                    'action': action,
                    'error': str(e)
                })
                execution_result['status'] = 'partial'
        
        execution_result['execution_time'] = time.time() - start_time
        
        decision.executed = True
        self.execution_log.append(execution_result)
        
        return execution_result
    
    def _execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个动作"""
        # 实际执行逻辑会连接到物理设备
        return {
            'action': action,
            'params': params,
            'executed': True,
            'timestamp': datetime.now().isoformat()
        }


# ==========================================
# 协调管理智能体
# ==========================================
class CoordinationAgent(BaseAgent):
    """
    协调管理智能体
    
    职责:
    - 智能体协调
    - 冲突解决
    - 资源调度
    - 状态同步
    """
    
    def __init__(self):
        super().__init__(AgentRole.COORDINATION, "协调管理智能体")
        self.agent_states: Dict[AgentRole, AgentStatus] = {}
        self.conflict_log: deque = deque(maxlen=1000)
        
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """协调智能体"""
        self.status = AgentStatus.ACTIVE
        
        try:
            # 更新智能体状态
            self._update_agent_states(inputs.get('agent_reports', {}))
            
            # 检测冲突
            conflicts = self._detect_conflicts(inputs.get('decisions', []))
            
            # 解决冲突
            if conflicts:
                self._resolve_conflicts(conflicts)
            
            # 生成协调指令
            coordination = self._generate_coordination()
            
            self.status = AgentStatus.IDLE
            return coordination
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            return {'error': str(e)}
    
    def _update_agent_states(self, reports: Dict[AgentRole, Dict]):
        """更新智能体状态"""
        for role, report in reports.items():
            self.agent_states[role] = report.get('status', AgentStatus.IDLE)
    
    def _detect_conflicts(self, decisions: List[L5Decision]) -> List[Dict]:
        """检测冲突"""
        conflicts = []
        
        # 检查决策冲突
        for i, d1 in enumerate(decisions):
            for d2 in decisions[i+1:]:
                if self._is_conflicting(d1, d2):
                    conflicts.append({
                        'decision1': d1.decision_id,
                        'decision2': d2.decision_id,
                        'type': 'action_conflict'
                    })
        
        return conflicts
    
    def _is_conflicting(self, d1: L5Decision, d2: L5Decision) -> bool:
        """判断是否冲突"""
        # 简化: 同优先级的不同类型决策可能冲突
        return d1.priority == d2.priority and d1.decision_type != d2.decision_type
    
    def _resolve_conflicts(self, conflicts: List[Dict]):
        """解决冲突"""
        for conflict in conflicts:
            self.conflict_log.append({
                'conflict': conflict,
                'resolution': 'priority_based',
                'timestamp': datetime.now()
            })
    
    def _generate_coordination(self) -> Dict[str, Any]:
        """生成协调指令"""
        return {
            'agent_states': {r.name: s.name for r, s in self.agent_states.items()},
            'coordination_status': 'normal',
            'pending_conflicts': len(self.conflict_log)
        }


# ==========================================
# 学习优化智能体
# ==========================================
class LearningAgent(BaseAgent):
    """
    学习优化智能体
    
    职责:
    - 性能分析
    - 参数优化
    - 模式学习
    - 知识积累
    """
    
    def __init__(self):
        super().__init__(AgentRole.LEARNING, "学习优化智能体")
        self.performance_history: deque = deque(maxlen=10000)
        self.learned_patterns: Dict[str, Any] = {}
        self.optimization_suggestions: deque = deque(maxlen=100)
        
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """学习和优化"""
        self.status = AgentStatus.ACTIVE
        
        try:
            # 记录性能
            self._record_performance(inputs)
            
            # 分析模式
            patterns = self._analyze_patterns()
            
            # 生成优化建议
            suggestions = self._generate_suggestions(patterns)
            
            self.status = AgentStatus.IDLE
            return {
                'patterns': patterns,
                'suggestions': suggestions
            }
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            return {'error': str(e)}
    
    def _record_performance(self, data: Dict[str, Any]):
        """记录性能数据"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'data': data
        })
    
    def _analyze_patterns(self) -> Dict[str, Any]:
        """分析模式"""
        if len(self.performance_history) < 10:
            return {}
        
        return {
            'data_points': len(self.performance_history),
            'analysis_status': 'completed'
        }
    
    def _generate_suggestions(self, patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成优化建议"""
        suggestions = []
        
        if patterns.get('data_points', 0) > 100:
            suggestions.append({
                'type': 'parameter_tuning',
                'description': '基于历史数据优化PID参数',
                'priority': 'low'
            })
        
        return suggestions


# ==========================================
# L5自主系统
# ==========================================
class L5AutonomousSystem:
    """
    L5级全自主运行系统
    
    整合所有智能体实现完全自主运行
    """
    
    def __init__(self):
        # 初始化智能体
        self.awareness_agent = SituationAwarenessAgent()
        self.planning_agent = DecisionPlanningAgent()
        self.execution_agent = ExecutionControlAgent()
        self.coordination_agent = CoordinationAgent()
        self.learning_agent = LearningAgent()
        
        # 智能体列表
        self.agents: Dict[AgentRole, BaseAgent] = {
            AgentRole.AWARENESS: self.awareness_agent,
            AgentRole.PLANNING: self.planning_agent,
            AgentRole.EXECUTION: self.execution_agent,
            AgentRole.COORDINATION: self.coordination_agent,
            AgentRole.LEARNING: self.learning_agent
        }
        
        # 系统状态
        self.state = L5SystemState()
        self.is_running = False
        self._lock = threading.Lock()
        
        # 消息总线
        self.message_bus: deque = deque(maxlen=10000)
        
        # 回调
        self.on_decision_made: Optional[Callable] = None
        self.on_mode_change: Optional[Callable] = None
        self.on_emergency: Optional[Callable] = None
    
    def initialize(self):
        """初始化系统"""
        self.state = L5SystemState()
        self.is_running = True
    
    def shutdown(self):
        """关闭系统"""
        self.is_running = False
    
    def process_cycle(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行一个处理周期
        
        Args:
            sensor_data: 传感器数据
            
        Returns:
            周期结果
        """
        with self._lock:
            cycle_start = time.time()
            
            # 1. 态势感知
            situation = self.awareness_agent.process(sensor_data)
            
            # 更新系统状态
            self.state.active_scenarios = situation.get('active_scenarios', [])
            self.state.confidence = situation.get('confidence', 0.8)
            
            # 2. 决策规划
            planning_result = self.planning_agent.process({
                'situation': situation,
                'constraints': self._get_constraints()
            })
            
            decisions = planning_result.get('decisions', [])
            
            # 3. 执行控制
            if decisions and not self.state.human_override:
                execution_result = self.execution_agent.process({
                    'decisions': decisions
                })
            else:
                execution_result = {'status': 'skipped'}
            
            # 4. 协调管理
            coordination_result = self.coordination_agent.process({
                'agent_reports': self._collect_agent_reports(),
                'decisions': decisions
            })
            
            # 5. 学习优化 (异步)
            learning_result = self.learning_agent.process({
                'situation': situation,
                'decisions': decisions,
                'execution': execution_result
            })
            
            # 6. 处理消息总线
            self._process_message_bus()
            
            # 更新系统状态
            self.state.last_update = datetime.now()
            self._update_system_health()
            
            cycle_time = time.time() - cycle_start
            
            return {
                'cycle_time': cycle_time,
                'situation': situation,
                'decisions': [d.__dict__ for d in decisions],
                'execution': execution_result,
                'coordination': coordination_result,
                'learning': learning_result,
                'system_state': self.get_state_summary()
            }
    
    def _get_constraints(self) -> Dict[str, Any]:
        """获取约束条件"""
        return {
            'max_flow_rate': 18.58,
            'min_flow_rate': 0.0,
            'max_pressure': 1.0,
            'autonomy_level': self.state.autonomy_level.value
        }
    
    def _collect_agent_reports(self) -> Dict[AgentRole, Dict]:
        """收集智能体报告"""
        reports = {}
        for role, agent in self.agents.items():
            reports[role] = {
                'status': agent.status,
                'cycle_count': agent.cycle_count,
                'error_count': agent.error_count,
                'last_cycle_time': agent.last_cycle_time
            }
        return reports
    
    def _process_message_bus(self):
        """处理消息总线"""
        for agent in self.agents.values():
            while agent.output_queue:
                msg = agent.output_queue.popleft()
                self.message_bus.append(msg)
                
                # 路由到目标智能体
                if msg.receiver in self.agents:
                    self.agents[msg.receiver].receive_message(msg)
    
    def _update_system_health(self):
        """更新系统健康度"""
        total_errors = sum(a.error_count for a in self.agents.values())
        total_cycles = sum(a.cycle_count for a in self.agents.values())
        
        if total_cycles > 0:
            self.state.system_health = 1.0 - (total_errors / max(total_cycles, 1))
        else:
            self.state.system_health = 1.0
    
    def set_autonomy_level(self, level: AutonomyLevel):
        """设置自主等级"""
        old_level = self.state.autonomy_level
        self.state.autonomy_level = level
        
        # 更新运行模式
        mode_map = {
            AutonomyLevel.L5_FULL: OperationMode.AUTO_L5,
            AutonomyLevel.L4_HIGH: OperationMode.AUTO_L4,
            AutonomyLevel.L3_CONDITIONAL: OperationMode.AUTO_L3,
            AutonomyLevel.L2_PARTIAL: OperationMode.SUPERVISED,
            AutonomyLevel.L1_ASSISTED: OperationMode.SUPERVISED,
            AutonomyLevel.L0_MANUAL: OperationMode.MANUAL
        }
        self.state.operation_mode = mode_map.get(level, OperationMode.SUPERVISED)
        
        if self.on_mode_change:
            self.on_mode_change(old_level, level)
    
    def request_human_override(self, reason: str):
        """请求人工接管"""
        self.state.human_override = True
        self.state.autonomy_level = AutonomyLevel.L0_MANUAL
        self.state.operation_mode = OperationMode.MANUAL
    
    def release_human_override(self):
        """释放人工接管"""
        self.state.human_override = False
        self.set_autonomy_level(AutonomyLevel.L5_FULL)
    
    def get_state_summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        return {
            'autonomy_level': self.state.autonomy_level.name,
            'operation_mode': self.state.operation_mode.name,
            'system_health': self.state.system_health,
            'confidence': self.state.confidence,
            'active_scenarios': len(self.state.active_scenarios),
            'human_override': self.state.human_override,
            'degradation_level': self.state.degradation_level,
            'last_update': self.state.last_update.isoformat()
        }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """获取智能体状态"""
        status = {}
        for role, agent in self.agents.items():
            status[role.name] = {
                'name': agent.name,
                'status': agent.status.name,
                'cycle_count': agent.cycle_count,
                'error_count': agent.error_count,
                'last_cycle_time': agent.last_cycle_time
            }
        return status


def create_l5_system() -> L5AutonomousSystem:
    """创建L5自主系统"""
    system = L5AutonomousSystem()
    system.initialize()
    return system


# ==========================================
# 导出
# ==========================================
__all__ = [
    # 枚举
    'AutonomyLevel',
    'AgentRole',
    'AgentStatus',
    # 智能体
    'SituationAwarenessAgent',
    'DecisionPlanningAgent',
    'ExecutionControlAgent',
    'CoordinationAgent',
    'LearningAgent',
    # L5系统
    'L5AutonomousSystem',
    'L5SystemState',
    'L5Decision',
    # 工厂函数
    'create_l5_system'
]
