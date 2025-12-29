"""
应急响应智能体
==============

处理各类紧急情况:
1. 应急检测
2. 响应决策
3. 资源调度
4. 恢复管理

版本: 3.4.0
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from collections import deque

from ..scenarios.scenario_database import (
    ScenarioType, ScenarioSeverity, ScenarioResponse, SCENARIO_DB
)


class EmergencyLevel(Enum):
    """应急等级"""
    GREEN = 0       # 正常
    BLUE = 1        # 一般
    YELLOW = 2      # 较重
    ORANGE = 3      # 严重
    RED = 4         # 特别严重


class EmergencyType(Enum):
    """应急类型"""
    EQUIPMENT = auto()      # 设备故障
    SAFETY = auto()         # 安全事故
    NATURAL = auto()        # 自然灾害
    WATER_QUALITY = auto()  # 水质事故
    ENGINEERING = auto()    # 工程事故
    CYBER = auto()          # 网络安全
    UNKNOWN = auto()        # 未知


@dataclass
class EmergencyState:
    """应急状态"""
    emergency_id: str
    level: EmergencyLevel
    type: EmergencyType
    scenario_type: Optional[ScenarioType]
    description: str
    start_time: datetime
    end_time: Optional[datetime] = None
    is_active: bool = True
    response_actions: List[str] = field(default_factory=list)
    affected_components: List[str] = field(default_factory=list)
    casualties: int = 0
    economic_loss: float = 0.0


@dataclass
class EmergencyResponse:
    """应急响应"""
    response_id: str
    emergency_id: str
    actions: List[Dict[str, Any]]
    responders: List[str]
    resources: List[str]
    timeline: List[Dict[str, Any]]
    status: str = "pending"


class EmergencyAgent:
    """
    应急响应智能体
    
    职责:
    - 应急检测与分级
    - 响应方案生成
    - 资源协调
    - 恢复管理
    """
    
    def __init__(self):
        self.active_emergencies: Dict[str, EmergencyState] = {}
        self.emergency_history: deque = deque(maxlen=1000)
        self.response_templates: Dict[EmergencyType, List[Dict]] = self._init_templates()
        self.emergency_counter = 0
        
    def _init_templates(self) -> Dict[EmergencyType, List[Dict]]:
        """初始化响应模板"""
        return {
            EmergencyType.EQUIPMENT: [
                {'action': 'isolate_component', 'priority': 1},
                {'action': 'switch_to_backup', 'priority': 2},
                {'action': 'notify_maintenance', 'priority': 3}
            ],
            EmergencyType.SAFETY: [
                {'action': 'emergency_shutdown', 'priority': 1},
                {'action': 'evacuate_personnel', 'priority': 1},
                {'action': 'notify_authorities', 'priority': 2}
            ],
            EmergencyType.NATURAL: [
                {'action': 'activate_protection', 'priority': 1},
                {'action': 'reduce_operations', 'priority': 2},
                {'action': 'standby_mode', 'priority': 3}
            ],
            EmergencyType.WATER_QUALITY: [
                {'action': 'stop_water_supply', 'priority': 1},
                {'action': 'isolate_contamination', 'priority': 2},
                {'action': 'notify_health_dept', 'priority': 2}
            ],
            EmergencyType.ENGINEERING: [
                {'action': 'reduce_load', 'priority': 1},
                {'action': 'structural_monitoring', 'priority': 2},
                {'action': 'expert_assessment', 'priority': 3}
            ],
            EmergencyType.CYBER: [
                {'action': 'network_isolation', 'priority': 1},
                {'action': 'switch_to_manual', 'priority': 2},
                {'action': 'forensic_analysis', 'priority': 3}
            ]
        }
    
    def detect_emergency(self, scenario_type: ScenarioType, 
                        severity: ScenarioSeverity,
                        details: Dict[str, Any]) -> Optional[EmergencyState]:
        """检测并创建应急状态"""
        # 判断是否构成应急
        if severity.value < ScenarioSeverity.ALARM.value:
            return None
        
        # 确定应急等级
        level = self._determine_level(severity)
        
        # 确定应急类型
        etype = self._determine_type(scenario_type)
        
        # 创建应急状态
        self.emergency_counter += 1
        emergency = EmergencyState(
            emergency_id=f"EMG_{self.emergency_counter:06d}",
            level=level,
            type=etype,
            scenario_type=scenario_type,
            description=details.get('description', str(scenario_type)),
            start_time=datetime.now(),
            affected_components=details.get('affected', [])
        )
        
        self.active_emergencies[emergency.emergency_id] = emergency
        
        return emergency
    
    def _determine_level(self, severity: ScenarioSeverity) -> EmergencyLevel:
        """确定应急等级"""
        mapping = {
            ScenarioSeverity.ALARM: EmergencyLevel.BLUE,
            ScenarioSeverity.CRITICAL: EmergencyLevel.YELLOW,
            ScenarioSeverity.EMERGENCY: EmergencyLevel.RED
        }
        return mapping.get(severity, EmergencyLevel.BLUE)
    
    def _determine_type(self, scenario_type: ScenarioType) -> EmergencyType:
        """确定应急类型"""
        type_value = scenario_type.value
        
        if 400 <= type_value < 500:
            return EmergencyType.EQUIPMENT
        elif 500 <= type_value < 600:
            return EmergencyType.SAFETY
        elif 300 <= type_value < 400:
            return EmergencyType.NATURAL
        elif 800 <= type_value < 900:
            return EmergencyType.WATER_QUALITY
        elif 900 <= type_value < 1000:
            return EmergencyType.ENGINEERING
        elif type_value == 605:
            return EmergencyType.CYBER
        else:
            return EmergencyType.UNKNOWN
    
    def generate_response(self, emergency: EmergencyState) -> EmergencyResponse:
        """生成应急响应方案"""
        templates = self.response_templates.get(emergency.type, [])
        
        actions = []
        for template in templates:
            actions.append({
                'action': template['action'],
                'priority': template['priority'],
                'status': 'pending',
                'assigned_to': None
            })
        
        # 从场景数据库获取额外响应
        if emergency.scenario_type:
            scenario_def = SCENARIO_DB.get_scenario(emergency.scenario_type)
            if scenario_def:
                for resp in scenario_def.responses:
                    actions.append({
                        'action': resp.action_name,
                        'priority': 5,
                        'status': 'pending',
                        'parameters': resp.parameters
                    })
        
        response = EmergencyResponse(
            response_id=f"RSP_{emergency.emergency_id}",
            emergency_id=emergency.emergency_id,
            actions=actions,
            responders=self._get_responders(emergency.level),
            resources=self._get_resources(emergency.type),
            timeline=[]
        )
        
        return response
    
    def _get_responders(self, level: EmergencyLevel) -> List[str]:
        """获取响应人员"""
        responders = ['on_duty_operator']
        
        if level.value >= EmergencyLevel.YELLOW.value:
            responders.extend(['shift_supervisor', 'maintenance_team'])
        
        if level.value >= EmergencyLevel.ORANGE.value:
            responders.extend(['plant_manager', 'safety_officer'])
        
        if level.value >= EmergencyLevel.RED.value:
            responders.extend(['emergency_team', 'external_authorities'])
        
        return responders
    
    def _get_resources(self, etype: EmergencyType) -> List[str]:
        """获取应急资源"""
        base_resources = ['communication', 'transportation']
        
        type_resources = {
            EmergencyType.EQUIPMENT: ['spare_parts', 'tools', 'backup_equipment'],
            EmergencyType.SAFETY: ['first_aid', 'fire_fighting', 'rescue_equipment'],
            EmergencyType.NATURAL: ['sandbags', 'pumps', 'shelter'],
            EmergencyType.WATER_QUALITY: ['water_testing', 'treatment_chemicals'],
            EmergencyType.ENGINEERING: ['monitoring_equipment', 'repair_materials'],
            EmergencyType.CYBER: ['backup_systems', 'forensic_tools']
        }
        
        return base_resources + type_resources.get(etype, [])
    
    def execute_response(self, response: EmergencyResponse) -> Dict[str, Any]:
        """执行应急响应"""
        results = {
            'response_id': response.response_id,
            'executed_actions': [],
            'failed_actions': [],
            'status': 'in_progress'
        }
        
        for action in sorted(response.actions, key=lambda x: x['priority']):
            try:
                # 模拟执行
                action['status'] = 'completed'
                results['executed_actions'].append(action['action'])
                
                response.timeline.append({
                    'time': datetime.now().isoformat(),
                    'action': action['action'],
                    'status': 'completed'
                })
            except Exception as e:
                action['status'] = 'failed'
                results['failed_actions'].append({
                    'action': action['action'],
                    'error': str(e)
                })
        
        if not results['failed_actions']:
            results['status'] = 'completed'
            response.status = 'completed'
        
        return results
    
    def close_emergency(self, emergency_id: str, resolution: str):
        """关闭应急"""
        if emergency_id in self.active_emergencies:
            emergency = self.active_emergencies[emergency_id]
            emergency.is_active = False
            emergency.end_time = datetime.now()
            
            self.emergency_history.append(emergency)
            del self.active_emergencies[emergency_id]
    
    def get_active_emergencies(self) -> List[EmergencyState]:
        """获取活动应急"""
        return list(self.active_emergencies.values())
    
    def get_highest_level_emergency(self) -> Optional[EmergencyState]:
        """获取最高等级应急"""
        if not self.active_emergencies:
            return None
        return max(self.active_emergencies.values(), 
                   key=lambda e: e.level.value)


def create_emergency_agent() -> EmergencyAgent:
    """创建应急智能体"""
    return EmergencyAgent()


__all__ = [
    'EmergencyLevel',
    'EmergencyType',
    'EmergencyState',
    'EmergencyResponse',
    'EmergencyAgent',
    'create_emergency_agent'
]
