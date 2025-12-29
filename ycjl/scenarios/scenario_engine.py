"""
场景识别引擎
============

实现场景检测、分类和状态管理:
1. 实时场景检测
2. 多场景并发管理
3. 场景转换跟踪
4. 置信度评估

版本: 3.4.0
"""

import math
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any, Set
from datetime import datetime, timedelta
from collections import deque
import threading

from .scenario_database import (
    ScenarioType, ScenarioCategory, ScenarioSeverity, ScenarioPhase,
    OperationMode, ScenarioDefinition, ScenarioTrigger, ScenarioResponse,
    SCENARIO_DB
)


@dataclass
class ScenarioState:
    """场景状态"""
    scenario_type: ScenarioType
    phase: ScenarioPhase = ScenarioPhase.DETECTION
    confidence: float = 0.0
    start_time: Optional[datetime] = None
    detection_time: Optional[datetime] = None
    confirmation_time: Optional[datetime] = None
    response_start_time: Optional[datetime] = None
    recovery_time: Optional[datetime] = None
    trigger_values: Dict[str, Any] = field(default_factory=dict)
    response_progress: Dict[str, float] = field(default_factory=dict)
    is_active: bool = False
    priority: int = 0
    escalation_count: int = 0


@dataclass
class ScenarioEvent:
    """场景事件"""
    event_id: str
    scenario_type: ScenarioType
    event_type: str  # detected, confirmed, responded, recovered, escalated
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    source: str = "engine"


class ScenarioDetector:
    """场景检测器"""
    
    def __init__(self):
        self.trigger_history: Dict[str, deque] = {}
        self.detection_window = 60.0  # 检测窗口 (秒)
        
    def evaluate_trigger(self, trigger: ScenarioTrigger, 
                        current_values: Dict[str, Any]) -> Tuple[bool, float]:
        """
        评估单个触发条件
        
        Returns:
            (是否触发, 置信度)
        """
        param = trigger.parameter
        if param not in current_values:
            return False, 0.0
        
        value = current_values[param]
        threshold = trigger.threshold
        operator = trigger.operator
        
        # 评估条件
        triggered = False
        if operator == "<":
            triggered = value < threshold
        elif operator == ">":
            triggered = value > threshold
        elif operator == "==":
            triggered = value == threshold
        elif operator == "!=":
            triggered = value != threshold
        elif operator == "in":
            triggered = value in threshold
        elif operator == "not_in":
            triggered = value not in threshold
        elif operator == "<=":
            triggered = value <= threshold
        elif operator == ">=":
            triggered = value >= threshold
        
        # 计算置信度
        confidence = 1.0 if triggered else 0.0
        
        # 考虑持续时间要求
        if trigger.duration > 0 and triggered:
            key = f"{param}_{trigger.condition_name}"
            if key not in self.trigger_history:
                self.trigger_history[key] = deque(maxlen=1000)
            
            self.trigger_history[key].append((time.time(), triggered))
            
            # 检查持续时间
            duration_met = self._check_duration(key, trigger.duration)
            if not duration_met:
                confidence *= 0.5
        
        return triggered, confidence * trigger.confidence
    
    def _check_duration(self, key: str, required_duration: float) -> bool:
        """检查条件是否持续满足"""
        if key not in self.trigger_history:
            return False
        
        history = self.trigger_history[key]
        if not history:
            return False
        
        current_time = time.time()
        start_time = current_time - required_duration
        
        # 检查窗口内所有记录是否都满足
        for ts, triggered in history:
            if ts >= start_time and not triggered:
                return False
        
        return True
    
    def detect_scenario(self, scenario_def: ScenarioDefinition,
                       current_values: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        检测单个场景
        
        Returns:
            (是否检测到, 置信度, 触发值)
        """
        if not scenario_def.triggers:
            return False, 0.0, {}
        
        all_triggered = True
        total_confidence = 0.0
        trigger_values = {}
        
        for trigger in scenario_def.triggers:
            triggered, confidence = self.evaluate_trigger(trigger, current_values)
            trigger_values[trigger.condition_name] = {
                'triggered': triggered,
                'confidence': confidence,
                'parameter': trigger.parameter,
                'value': current_values.get(trigger.parameter)
            }
            
            if not triggered:
                all_triggered = False
            total_confidence += confidence
        
        avg_confidence = total_confidence / len(scenario_def.triggers)
        
        return all_triggered, avg_confidence, trigger_values
    
    def detect_all_scenarios(self, current_values: Dict[str, Any]) -> List[Tuple[ScenarioType, float, Dict]]:
        """检测所有场景"""
        detected = []
        
        for scenario_type, scenario_def in SCENARIO_DB.get_all_scenarios().items():
            is_detected, confidence, trigger_values = self.detect_scenario(
                scenario_def, current_values
            )
            
            if is_detected or confidence > 0.5:
                detected.append((scenario_type, confidence, trigger_values))
        
        # 按置信度排序
        detected.sort(key=lambda x: x[1], reverse=True)
        
        return detected


class ScenarioClassifier:
    """场景分类器"""
    
    def __init__(self):
        self.classification_history: deque = deque(maxlen=1000)
        self.scenario_weights: Dict[ScenarioType, float] = {}
        
    def classify(self, detected_scenarios: List[Tuple[ScenarioType, float, Dict]],
                 current_mode: OperationMode) -> List[ScenarioState]:
        """
        分类和优先级排序
        
        Args:
            detected_scenarios: 检测到的场景列表
            current_mode: 当前运行模式
            
        Returns:
            排序后的场景状态列表
        """
        states = []
        
        for scenario_type, confidence, trigger_values in detected_scenarios:
            scenario_def = SCENARIO_DB.get_scenario(scenario_type)
            if not scenario_def:
                continue
            
            # 检查是否允许在当前模式下
            mode_allowed = current_mode in scenario_def.allowed_modes
            
            # 计算优先级
            priority = self._calculate_priority(scenario_def, confidence, mode_allowed)
            
            state = ScenarioState(
                scenario_type=scenario_type,
                phase=ScenarioPhase.DETECTION,
                confidence=confidence,
                detection_time=datetime.now(),
                trigger_values=trigger_values,
                is_active=True,
                priority=priority
            )
            
            states.append(state)
        
        # 按优先级排序
        states.sort(key=lambda x: x.priority, reverse=True)
        
        return states
    
    def _calculate_priority(self, scenario_def: ScenarioDefinition,
                           confidence: float, mode_allowed: bool) -> int:
        """计算场景优先级"""
        # 基础优先级 = 严重程度
        base_priority = scenario_def.severity.value * 100
        
        # 置信度加权
        priority = int(base_priority * confidence)
        
        # 需要人工干预的优先
        if scenario_def.requires_human:
            priority += 50
        
        # 不支持自动恢复的优先
        if not scenario_def.auto_recovery:
            priority += 30
        
        # 不允许当前模式的降低优先级
        if not mode_allowed:
            priority -= 100
        
        return max(0, priority)


class ScenarioEngine:
    """
    场景引擎
    
    整合检测、分类和状态管理
    """
    
    def __init__(self):
        self.detector = ScenarioDetector()
        self.classifier = ScenarioClassifier()
        
        # 状态管理
        self.active_scenarios: Dict[ScenarioType, ScenarioState] = {}
        self.scenario_history: deque = deque(maxlen=10000)
        self.event_queue: deque = deque(maxlen=1000)
        
        # 运行状态
        self.current_mode = OperationMode.AUTO_L5
        self.is_running = False
        self._lock = threading.Lock()
        
        # 回调
        self.on_scenario_detected: Optional[Callable] = None
        self.on_scenario_confirmed: Optional[Callable] = None
        self.on_scenario_recovered: Optional[Callable] = None
        
        # 配置
        self.confirmation_threshold = 0.8
        self.recovery_threshold = 0.3
        self.max_concurrent_scenarios = 10
    
    def update(self, current_values: Dict[str, Any]) -> List[ScenarioState]:
        """
        更新场景状态
        
        Args:
            current_values: 当前系统状态值
            
        Returns:
            活动场景列表
        """
        with self._lock:
            # 检测场景
            detected = self.detector.detect_all_scenarios(current_values)
            
            # 分类排序
            new_states = self.classifier.classify(detected, self.current_mode)
            
            # 更新现有场景状态
            self._update_existing_scenarios(current_values)
            
            # 处理新检测到的场景
            for state in new_states:
                self._process_detected_scenario(state)
            
            # 检查场景恢复
            self._check_recovery(current_values)
            
            # 返回活动场景
            return list(self.active_scenarios.values())
    
    def _update_existing_scenarios(self, current_values: Dict[str, Any]):
        """更新现有场景状态"""
        for scenario_type, state in list(self.active_scenarios.items()):
            scenario_def = SCENARIO_DB.get_scenario(scenario_type)
            if not scenario_def:
                continue
            
            # 重新评估置信度
            _, confidence, trigger_values = self.detector.detect_scenario(
                scenario_def, current_values
            )
            
            state.confidence = confidence
            state.trigger_values = trigger_values
            
            # 更新阶段
            self._update_phase(state, scenario_def)
    
    def _process_detected_scenario(self, state: ScenarioState):
        """处理检测到的场景"""
        scenario_type = state.scenario_type
        
        if scenario_type in self.active_scenarios:
            # 更新现有场景
            existing = self.active_scenarios[scenario_type]
            existing.confidence = max(existing.confidence, state.confidence)
            existing.trigger_values.update(state.trigger_values)
        else:
            # 新场景
            if len(self.active_scenarios) < self.max_concurrent_scenarios:
                state.start_time = datetime.now()
                self.active_scenarios[scenario_type] = state
                
                # 触发回调
                if self.on_scenario_detected:
                    self.on_scenario_detected(state)
                
                # 记录事件
                self._log_event(scenario_type, "detected", state.trigger_values)
    
    def _update_phase(self, state: ScenarioState, scenario_def: ScenarioDefinition):
        """更新场景阶段"""
        if state.phase == ScenarioPhase.DETECTION:
            if state.confidence >= self.confirmation_threshold:
                state.phase = ScenarioPhase.CONFIRMATION
                state.confirmation_time = datetime.now()
                
                if self.on_scenario_confirmed:
                    self.on_scenario_confirmed(state)
                
                self._log_event(state.scenario_type, "confirmed", {
                    'confidence': state.confidence
                })
        
        elif state.phase == ScenarioPhase.CONFIRMATION:
            state.phase = ScenarioPhase.RESPONSE
            state.response_start_time = datetime.now()
    
    def _check_recovery(self, current_values: Dict[str, Any]):
        """检查场景恢复"""
        for scenario_type, state in list(self.active_scenarios.items()):
            if state.confidence < self.recovery_threshold:
                scenario_def = SCENARIO_DB.get_scenario(scenario_type)
                
                if scenario_def and scenario_def.auto_recovery:
                    state.phase = ScenarioPhase.RECOVERY
                    state.recovery_time = datetime.now()
                    state.is_active = False
                    
                    # 移动到历史
                    self.scenario_history.append(state)
                    del self.active_scenarios[scenario_type]
                    
                    if self.on_scenario_recovered:
                        self.on_scenario_recovered(state)
                    
                    self._log_event(scenario_type, "recovered", {
                        'duration': (datetime.now() - state.start_time).total_seconds()
                        if state.start_time else 0
                    })
    
    def _log_event(self, scenario_type: ScenarioType, event_type: str, 
                   details: Dict[str, Any]):
        """记录场景事件"""
        event = ScenarioEvent(
            event_id=f"{scenario_type.name}_{int(time.time()*1000)}",
            scenario_type=scenario_type,
            event_type=event_type,
            timestamp=datetime.now(),
            details=details
        )
        self.event_queue.append(event)
    
    def set_mode(self, mode: OperationMode):
        """设置运行模式"""
        self.current_mode = mode
    
    def get_active_scenarios(self) -> List[ScenarioState]:
        """获取活动场景"""
        return list(self.active_scenarios.values())
    
    def get_highest_severity_scenario(self) -> Optional[ScenarioState]:
        """获取最高严重度场景"""
        if not self.active_scenarios:
            return None
        
        return max(self.active_scenarios.values(), 
                   key=lambda s: s.priority)
    
    def get_scenarios_by_category(self, category: ScenarioCategory) -> List[ScenarioState]:
        """按类别获取活动场景"""
        result = []
        for state in self.active_scenarios.values():
            scenario_def = SCENARIO_DB.get_scenario(state.scenario_type)
            if scenario_def and scenario_def.category == category:
                result.append(state)
        return result
    
    def clear_scenario(self, scenario_type: ScenarioType):
        """手动清除场景"""
        if scenario_type in self.active_scenarios:
            state = self.active_scenarios[scenario_type]
            state.is_active = False
            self.scenario_history.append(state)
            del self.active_scenarios[scenario_type]
            self._log_event(scenario_type, "cleared", {"manual": True})
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'active_count': len(self.active_scenarios),
            'history_count': len(self.scenario_history),
            'event_count': len(self.event_queue),
            'current_mode': self.current_mode.name,
            'active_by_severity': self._count_by_severity(),
            'active_by_category': self._count_by_category()
        }
    
    def _count_by_severity(self) -> Dict[str, int]:
        """按严重度统计"""
        counts = {}
        for state in self.active_scenarios.values():
            scenario_def = SCENARIO_DB.get_scenario(state.scenario_type)
            if scenario_def:
                severity = scenario_def.severity.name
                counts[severity] = counts.get(severity, 0) + 1
        return counts
    
    def _count_by_category(self) -> Dict[str, int]:
        """按类别统计"""
        counts = {}
        for state in self.active_scenarios.values():
            scenario_def = SCENARIO_DB.get_scenario(state.scenario_type)
            if scenario_def:
                category = scenario_def.category.name
                counts[category] = counts.get(category, 0) + 1
        return counts


# ==========================================
# 导出
# ==========================================
__all__ = [
    'ScenarioState',
    'ScenarioEvent',
    'ScenarioDetector',
    'ScenarioClassifier',
    'ScenarioEngine'
]
