"""
故障容错系统
============

实现系统级故障容错:
1. 故障检测
2. 冗余管理
3. 优雅降级
4. 自动恢复

版本: 3.4.0
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set
from datetime import datetime, timedelta
from collections import deque


class FaultType(Enum):
    """故障类型"""
    SENSOR = auto()       # 传感器故障
    ACTUATOR = auto()     # 执行器故障
    COMMUNICATION = auto() # 通讯故障
    POWER = auto()        # 电源故障
    SOFTWARE = auto()     # 软件故障
    HARDWARE = auto()     # 硬件故障
    NETWORK = auto()      # 网络故障
    DATA = auto()         # 数据故障


class FaultState(Enum):
    """故障状态"""
    DETECTED = auto()     # 已检测
    CONFIRMED = auto()    # 已确认
    ISOLATED = auto()     # 已隔离
    MITIGATED = auto()    # 已缓解
    RECOVERED = auto()    # 已恢复


class RedundancyMode(Enum):
    """冗余模式"""
    NONE = auto()         # 无冗余
    COLD_STANDBY = auto() # 冷备
    WARM_STANDBY = auto() # 温备
    HOT_STANDBY = auto()  # 热备
    ACTIVE_ACTIVE = auto() # 双活
    N_PLUS_ONE = auto()   # N+1
    TWO_N = auto()        # 2N


@dataclass
class ComponentHealth:
    """组件健康状态"""
    component_id: str
    component_type: str
    health_score: float  # 0-1
    is_healthy: bool
    last_heartbeat: datetime
    fault_count: int = 0
    recovery_count: int = 0
    mtbf: float = 0.0  # 平均无故障时间
    mttr: float = 0.0  # 平均恢复时间
    availability: float = 1.0


@dataclass
class FaultRecord:
    """故障记录"""
    fault_id: str
    fault_type: FaultType
    component_id: str
    state: FaultState
    detection_time: datetime
    confirmation_time: Optional[datetime] = None
    isolation_time: Optional[datetime] = None
    recovery_time: Optional[datetime] = None
    description: str = ""
    impact: str = ""
    root_cause: str = ""


@dataclass
class SystemResilience:
    """系统弹性状态"""
    overall_health: float  # 0-1
    available_capacity: float  # 0-1
    redundancy_status: Dict[str, RedundancyMode]
    active_faults: int
    degradation_level: int  # 0-5
    can_operate: bool
    recommendations: List[str]


class FaultToleranceManager:
    """
    故障容错管理器
    
    职责:
    - 健康监测
    - 故障检测
    - 冗余切换
    - 优雅降级
    - 自动恢复
    """
    
    def __init__(self):
        self.components: Dict[str, ComponentHealth] = {}
        self.redundancy_config: Dict[str, Dict] = {}
        self.active_faults: Dict[str, FaultRecord] = {}
        self.fault_history: deque = deque(maxlen=10000)
        self.fault_counter = 0
        
        # 配置
        self.heartbeat_timeout = 30.0  # 秒
        self.health_threshold = 0.5
        self.auto_recovery_enabled = True
        
        # 初始化组件
        self._init_components()
    
    def _init_components(self):
        """初始化组件监测"""
        # 定义系统关键组件
        components = [
            ('reservoir', 'source'),
            ('tunnel_1', 'tunnel'),
            ('tunnel_2', 'tunnel'),
            ('pool_1', 'pool'),
            ('pool_2', 'pool'),
            ('pipeline_1', 'pipeline'),
            ('pipeline_2', 'pipeline'),
            ('pipeline_3', 'pipeline'),
            ('valve_inline_1', 'valve'),
            ('valve_inline_2', 'valve'),
            ('valve_inline_3', 'valve'),
            ('valve_end_1', 'valve'),
            ('valve_end_2', 'valve'),
            ('surge_tank_1', 'surge_tank'),
            ('surge_tank_2', 'surge_tank'),
            ('turbine', 'power'),
            ('scada_main', 'control'),
            ('scada_backup', 'control'),
            ('plc_1', 'control'),
            ('plc_2', 'control'),
            ('network_main', 'network'),
            ('network_backup', 'network'),
            ('power_main', 'power'),
            ('power_ups', 'power'),
            ('power_diesel', 'power')
        ]
        
        for comp_id, comp_type in components:
            self.components[comp_id] = ComponentHealth(
                component_id=comp_id,
                component_type=comp_type,
                health_score=1.0,
                is_healthy=True,
                last_heartbeat=datetime.now()
            )
        
        # 配置冗余
        self.redundancy_config = {
            'scada': {
                'primary': 'scada_main',
                'backup': 'scada_backup',
                'mode': RedundancyMode.HOT_STANDBY
            },
            'network': {
                'primary': 'network_main',
                'backup': 'network_backup',
                'mode': RedundancyMode.HOT_STANDBY
            },
            'power': {
                'primary': 'power_main',
                'backups': ['power_ups', 'power_diesel'],
                'mode': RedundancyMode.N_PLUS_ONE
            },
            'plc': {
                'primary': 'plc_1',
                'backup': 'plc_2',
                'mode': RedundancyMode.WARM_STANDBY
            }
        }
    
    def update_health(self, component_id: str, health_data: Dict[str, Any]):
        """更新组件健康状态"""
        if component_id not in self.components:
            return
        
        component = self.components[component_id]
        
        # 更新心跳
        component.last_heartbeat = datetime.now()
        
        # 更新健康分数
        if 'health_score' in health_data:
            component.health_score = health_data['health_score']
        
        # 检查健康状态
        old_healthy = component.is_healthy
        component.is_healthy = component.health_score >= self.health_threshold
        
        # 状态变化处理
        if old_healthy and not component.is_healthy:
            self._on_component_unhealthy(component)
        elif not old_healthy and component.is_healthy:
            self._on_component_recovered(component)
    
    def check_heartbeats(self) -> List[str]:
        """检查心跳超时"""
        now = datetime.now()
        timeout_components = []
        
        for comp_id, component in self.components.items():
            elapsed = (now - component.last_heartbeat).total_seconds()
            if elapsed > self.heartbeat_timeout:
                timeout_components.append(comp_id)
                if component.is_healthy:
                    component.is_healthy = False
                    self._on_component_unhealthy(component)
        
        return timeout_components
    
    def _on_component_unhealthy(self, component: ComponentHealth):
        """组件不健康处理"""
        component.fault_count += 1
        
        # 创建故障记录
        self.fault_counter += 1
        fault = FaultRecord(
            fault_id=f"FLT_{self.fault_counter:06d}",
            fault_type=self._determine_fault_type(component.component_type),
            component_id=component.component_id,
            state=FaultState.DETECTED,
            detection_time=datetime.now(),
            description=f"Component {component.component_id} unhealthy"
        )
        
        self.active_faults[fault.fault_id] = fault
        
        # 尝试冗余切换
        self._try_redundancy_switch(component.component_id)
    
    def _on_component_recovered(self, component: ComponentHealth):
        """组件恢复处理"""
        component.recovery_count += 1
        
        # 更新故障记录
        for fault in self.active_faults.values():
            if fault.component_id == component.component_id:
                fault.state = FaultState.RECOVERED
                fault.recovery_time = datetime.now()
                
                # 移到历史
                self.fault_history.append(fault)
        
        # 清理已恢复的故障
        self.active_faults = {
            fid: f for fid, f in self.active_faults.items()
            if f.state != FaultState.RECOVERED
        }
    
    def _determine_fault_type(self, component_type: str) -> FaultType:
        """确定故障类型"""
        type_mapping = {
            'sensor': FaultType.SENSOR,
            'valve': FaultType.ACTUATOR,
            'control': FaultType.SOFTWARE,
            'network': FaultType.NETWORK,
            'power': FaultType.POWER,
            'pipeline': FaultType.HARDWARE
        }
        return type_mapping.get(component_type, FaultType.HARDWARE)
    
    def _try_redundancy_switch(self, component_id: str):
        """尝试冗余切换"""
        for group_name, config in self.redundancy_config.items():
            primary = config.get('primary')
            backup = config.get('backup')
            backups = config.get('backups', [backup] if backup else [])
            
            if component_id == primary:
                # 主设备故障,切换到备用
                for backup_id in backups:
                    if backup_id and backup_id in self.components:
                        backup_comp = self.components[backup_id]
                        if backup_comp.is_healthy:
                            self._activate_backup(group_name, backup_id)
                            return True
        return False
    
    def _activate_backup(self, group_name: str, backup_id: str):
        """激活备用设备"""
        # 实际实现会发送控制命令
        pass
    
    def get_system_resilience(self) -> SystemResilience:
        """获取系统弹性状态"""
        healthy_count = sum(1 for c in self.components.values() if c.is_healthy)
        total_count = len(self.components)
        
        overall_health = healthy_count / total_count if total_count > 0 else 0
        
        # 计算降级等级
        unhealthy_ratio = 1 - overall_health
        if unhealthy_ratio == 0:
            degradation_level = 0
        elif unhealthy_ratio < 0.1:
            degradation_level = 1
        elif unhealthy_ratio < 0.2:
            degradation_level = 2
        elif unhealthy_ratio < 0.3:
            degradation_level = 3
        elif unhealthy_ratio < 0.5:
            degradation_level = 4
        else:
            degradation_level = 5
        
        # 检查关键组件
        critical_components = ['reservoir', 'scada_main', 'power_main']
        critical_healthy = all(
            self.components.get(c, ComponentHealth(c, '', 0, False, datetime.now())).is_healthy
            for c in critical_components
        )
        
        # 检查冗余状态
        redundancy_status = {}
        for name, config in self.redundancy_config.items():
            redundancy_status[name] = config.get('mode', RedundancyMode.NONE)
        
        # 生成建议
        recommendations = []
        if degradation_level > 0:
            recommendations.append("建议检查故障组件")
        if degradation_level >= 3:
            recommendations.append("建议启动应急预案")
        if not critical_healthy:
            recommendations.append("关键组件故障,需要立即处理")
        
        return SystemResilience(
            overall_health=overall_health,
            available_capacity=overall_health,
            redundancy_status=redundancy_status,
            active_faults=len(self.active_faults),
            degradation_level=degradation_level,
            can_operate=degradation_level < 5 and critical_healthy,
            recommendations=recommendations
        )
    
    def get_fault_statistics(self) -> Dict[str, Any]:
        """获取故障统计"""
        fault_by_type = {}
        for fault in list(self.active_faults.values()) + list(self.fault_history):
            ftype = fault.fault_type.name
            fault_by_type[ftype] = fault_by_type.get(ftype, 0) + 1
        
        return {
            'active_faults': len(self.active_faults),
            'total_faults': len(self.fault_history) + len(self.active_faults),
            'by_type': fault_by_type,
            'components_unhealthy': sum(
                1 for c in self.components.values() if not c.is_healthy
            )
        }
    
    def isolate_component(self, component_id: str):
        """隔离组件"""
        if component_id in self.components:
            self.components[component_id].is_healthy = False
            
            for fault in self.active_faults.values():
                if fault.component_id == component_id:
                    fault.state = FaultState.ISOLATED
                    fault.isolation_time = datetime.now()
    
    def recover_component(self, component_id: str):
        """恢复组件"""
        if component_id in self.components:
            self.components[component_id].is_healthy = True
            self.components[component_id].health_score = 1.0
            self.components[component_id].last_heartbeat = datetime.now()
            
            self._on_component_recovered(self.components[component_id])


__all__ = [
    'FaultType',
    'FaultState',
    'RedundancyMode',
    'FaultToleranceManager',
    'ComponentHealth',
    'SystemResilience'
]
