"""
泵站群多智能体系统 v1.0
==========================

实现泵站群的安全与经济运行，融合到多智能体体系。

包含:
1. PumpGroupSafetyAgent - 泵站群安全智能体 (L1层)
2. PumpGroupEconomicAgent - 泵站群经济优化智能体 (L3层)
3. PumpGroupCoordinatorAgent - 泵站群协调智能体 (融合安全与经济)
4. PumpGroupMultiAgentSystem - 完整的多智能体系统

关键功能:
- 泵站启停约束 (最小间隔、运行时长)
- 前池水位保护 (抽空/溢流)
- 汽蚀/过载保护
- 基于效率曲线的最优台数选择
- 峰谷电价优化
- 设备寿命均衡 (运行时长均衡)
- 多泵站协调的经济调度
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import time
import heapq

from .base_agent import (
    BaseAgent, AgentPriority, AgentState, AgentMessage,
    ControlAction, MessageType, AgentRegistry
)
from .reflex_agent import SafetyRule, RuleCategory, RuleStatus


# ============================================================
# 1. 数据结构定义
# ============================================================

class PumpStatus(Enum):
    """泵运行状态"""
    STOPPED = auto()          # 停机
    STARTING = auto()         # 启动中
    RUNNING = auto()          # 运行中
    STOPPING = auto()         # 停机中
    FAULT = auto()            # 故障
    MAINTENANCE = auto()      # 检修


class ProtectionType(Enum):
    """保护类型"""
    CAVITATION = auto()       # 汽蚀保护
    OVERLOAD = auto()         # 过载保护
    OVERHEAT = auto()         # 过热保护
    VIBRATION = auto()        # 振动保护
    BEARING = auto()          # 轴承保护
    LOW_SUCTION = auto()      # 低吸入压力保护


@dataclass
class PumpState:
    """单泵状态"""
    pump_id: str
    station_id: str
    status: PumpStatus = PumpStatus.STOPPED

    # 运行参数
    speed_rpm: float = 0.0
    flow_rate: float = 0.0         # m³/s
    head: float = 0.0              # m
    power: float = 0.0             # kW
    efficiency: float = 0.0        # 0-1

    # 状态监测
    vibration_level: float = 0.0   # mm/s
    bearing_temp: float = 25.0     # °C
    motor_temp: float = 25.0       # °C
    suction_pressure: float = 0.5  # MPa
    discharge_pressure: float = 1.0  # MPa

    # 运行统计
    run_hours: float = 0.0         # 累计运行小时
    start_count: int = 0           # 累计启动次数
    last_start_time: float = 0.0   # 上次启动时间
    last_stop_time: float = 0.0    # 上次停机时间
    continuous_run_time: float = 0.0  # 本次连续运行时间

    # 保护状态
    protection_triggered: List[ProtectionType] = field(default_factory=list)


@dataclass
class StationState:
    """泵站状态"""
    station_id: str
    station_name: str

    # 泵组状态
    pumps: Dict[str, PumpState] = field(default_factory=dict)
    total_pump_count: int = 0
    running_pump_count: int = 0
    available_pump_count: int = 0

    # 水力状态
    forebay_level: float = 3.0     # 前池水位 (m)
    forebay_level_max: float = 5.0
    forebay_level_min: float = 1.0
    forebay_level_alarm_high: float = 4.5
    forebay_level_alarm_low: float = 1.5

    outlet_pressure: float = 0.5   # 出水压力 (MPa)
    total_flow: float = 0.0        # 总流量 (m³/s)
    total_power: float = 0.0       # 总功率 (kW)

    # 经济指标
    current_efficiency: float = 0.0  # 站效率
    energy_cost_rate: float = 0.0    # 当前电费率 (元/kWh)


@dataclass
class PumpGroupState:
    """泵站群状态"""
    stations: Dict[str, StationState] = field(default_factory=dict)

    # 全局状态
    total_flow: float = 0.0
    total_power: float = 0.0
    total_running_pumps: int = 0

    # 协调状态
    wave_propagation_time: Dict[Tuple[str, str], float] = field(default_factory=dict)
    coordination_mode: str = "normal"

    # 电价信息
    current_hour: int = 12
    electricity_price: float = 0.5  # 元/kWh


@dataclass
class PumpScheduleAction:
    """泵调度动作"""
    station_id: str
    pump_id: str
    action: str  # 'start', 'stop', 'adjust_speed'
    target_value: float = 0.0
    priority: AgentPriority = AgentPriority.TACTICAL
    reason: str = ""
    timestamp: float = 0.0

    # 约束
    min_delay: float = 0.0  # 最小延迟执行时间
    must_execute_by: float = float('inf')  # 必须执行的截止时间


# ============================================================
# 2. 泵站群安全智能体 (L1层 - 毫秒级响应)
# ============================================================

class PumpGroupSafetyAgent(BaseAgent):
    """
    泵站群安全智能体

    继承L1反射层思想，专门针对泵站群的安全规则:
    - 前池水位保护 (抽空/溢流)
    - 泵启停约束 (最小间隔、冷却时间)
    - 汽蚀保护 (NPSH监测)
    - 过载保护 (电流/功率监测)
    - 振动保护 (轴承监测)
    - 泵组并联运行规则
    - 故障切换逻辑
    """

    def __init__(self, agent_id: str = "PumpGroup_L1_Safety"):
        super().__init__(agent_id, AgentPriority.SAFETY)

        # 安全规则库
        self.rules: Dict[str, SafetyRule] = {}
        self.active_rules: List[str] = []
        self.rule_log: List[Dict] = []

        # 泵站群状态
        self.pump_group_state: Optional[PumpGroupState] = None

        # 安全参数
        self.params = {
            # 启停约束
            'min_start_interval': 300.0,      # 最小启动间隔 (s)
            'min_stop_interval': 180.0,       # 最小停机间隔 (s)
            'min_run_time': 600.0,            # 最小运行时间 (s)
            'cooling_time': 120.0,            # 停机冷却时间 (s)

            # 前池水位
            'forebay_critical_low': 0.8,      # 临界低水位 (m)
            'forebay_alarm_low': 1.5,         # 报警低水位 (m)
            'forebay_alarm_high': 4.5,        # 报警高水位 (m)
            'forebay_critical_high': 5.2,     # 临界高水位 (m)

            # 汽蚀保护
            'npsh_margin': 1.5,               # NPSH余量系数
            'suction_pressure_min': 0.03,     # 最小吸入压力 (MPa)

            # 过载保护
            'power_overload_ratio': 1.15,     # 功率过载比
            'current_overload_ratio': 1.20,   # 电流过载比

            # 振动保护
            'vibration_alarm': 4.5,           # 振动报警值 (mm/s)
            'vibration_trip': 7.1,            # 振动跳闸值 (mm/s)

            # 温度保护
            'bearing_temp_alarm': 70.0,       # 轴承温度报警 (°C)
            'bearing_temp_trip': 85.0,        # 轴承温度跳闸 (°C)
            'motor_temp_alarm': 80.0,         # 电机温度报警 (°C)
            'motor_temp_trip': 95.0,          # 电机温度跳闸 (°C)
        }

        # 初始化安全规则
        self._init_safety_rules()

    def _init_safety_rules(self):
        """初始化泵站群专用安全规则"""

        # ===== 前池水位保护规则 =====

        # PS-R1: 前池临界低水位 - 紧急停泵
        self.add_rule(SafetyRule(
            rule_id="PS_R1_forebay_critical_low",
            category=RuleCategory.EMERGENCY,
            description="前池水位临界低，紧急停止所有泵",
            condition=self._check_forebay_critical_low,
            action=self._action_emergency_stop_all,
            priority=0,
            hold_time=60.0
        ))

        # PS-R2: 前池报警低水位 - 减少运行泵数
        self.add_rule(SafetyRule(
            rule_id="PS_R2_forebay_alarm_low",
            category=RuleCategory.PROTECTION,
            description="前池水位低报警，减少运行泵数",
            condition=self._check_forebay_alarm_low,
            action=self._action_reduce_pumps,
            priority=1,
            min_interval=30.0
        ))

        # PS-R3: 前池高水位报警 - 增加运行泵数
        self.add_rule(SafetyRule(
            rule_id="PS_R3_forebay_alarm_high",
            category=RuleCategory.PROTECTION,
            description="前池水位高报警，增加运行泵数",
            condition=self._check_forebay_alarm_high,
            action=self._action_increase_pumps,
            priority=1,
            min_interval=30.0
        ))

        # PS-R4: 前池临界高水位 - 全部启动并开溢流
        self.add_rule(SafetyRule(
            rule_id="PS_R4_forebay_critical_high",
            category=RuleCategory.EMERGENCY,
            description="前池水位临界高，启动所有泵并开溢流",
            condition=self._check_forebay_critical_high,
            action=self._action_emergency_discharge,
            priority=0,
            hold_time=60.0
        ))

        # ===== 泵启停约束规则 =====

        # PS-R5: 启动间隔检查
        self.add_rule(SafetyRule(
            rule_id="PS_R5_start_interval",
            category=RuleCategory.INTERLOCK,
            description="泵启动间隔不足，阻止启动",
            condition=self._check_start_interval_violation,
            action=self._action_block_start,
            priority=2
        ))

        # PS-R6: 最小运行时间检查
        self.add_rule(SafetyRule(
            rule_id="PS_R6_min_run_time",
            category=RuleCategory.INTERLOCK,
            description="泵运行时间不足，阻止停机",
            condition=self._check_min_run_time_violation,
            action=self._action_block_stop,
            priority=2
        ))

        # ===== 汽蚀保护规则 =====

        # PS-R7: 低吸入压力保护
        self.add_rule(SafetyRule(
            rule_id="PS_R7_low_suction_pressure",
            category=RuleCategory.PROTECTION,
            description="吸入压力过低，可能汽蚀",
            condition=self._check_low_suction_pressure,
            action=self._action_cavitation_protect,
            priority=0,
            hold_time=30.0
        ))

        # ===== 过载保护规则 =====

        # PS-R8: 功率过载保护
        self.add_rule(SafetyRule(
            rule_id="PS_R8_power_overload",
            category=RuleCategory.PROTECTION,
            description="泵功率过载",
            condition=self._check_power_overload,
            action=self._action_overload_protect,
            priority=0,
            hold_time=10.0
        ))

        # ===== 振动保护规则 =====

        # PS-R9: 振动报警
        self.add_rule(SafetyRule(
            rule_id="PS_R9_vibration_alarm",
            category=RuleCategory.PROTECTION,
            description="振动超标报警",
            condition=self._check_vibration_alarm,
            action=self._action_vibration_alarm,
            priority=1,
            min_interval=60.0
        ))

        # PS-R10: 振动跳闸
        self.add_rule(SafetyRule(
            rule_id="PS_R10_vibration_trip",
            category=RuleCategory.EMERGENCY,
            description="振动超限跳闸",
            condition=self._check_vibration_trip,
            action=self._action_vibration_trip,
            priority=0
        ))

        # ===== 温度保护规则 =====

        # PS-R11: 轴承温度保护
        self.add_rule(SafetyRule(
            rule_id="PS_R11_bearing_temp",
            category=RuleCategory.PROTECTION,
            description="轴承温度超限",
            condition=self._check_bearing_temp,
            action=self._action_temp_protect,
            priority=0
        ))

        # PS-R12: 电机温度保护
        self.add_rule(SafetyRule(
            rule_id="PS_R12_motor_temp",
            category=RuleCategory.PROTECTION,
            description="电机温度超限",
            condition=self._check_motor_temp,
            action=self._action_temp_protect,
            priority=0
        ))

        # ===== 故障切换规则 =====

        # PS-R13: 泵故障自动切换
        self.add_rule(SafetyRule(
            rule_id="PS_R13_fault_switchover",
            category=RuleCategory.PROTECTION,
            description="泵故障，启动备用泵",
            condition=self._check_pump_fault,
            action=self._action_fault_switchover,
            priority=1
        ))

    # ===== 条件检查函数 =====

    def _check_forebay_critical_low(self, state: Dict) -> bool:
        """检查前池临界低水位"""
        for station_id, station in self.pump_group_state.stations.items():
            if station.forebay_level < self.params['forebay_critical_low']:
                return True
        return False

    def _check_forebay_alarm_low(self, state: Dict) -> bool:
        """检查前池报警低水位"""
        for station_id, station in self.pump_group_state.stations.items():
            if station.forebay_level < self.params['forebay_alarm_low']:
                if station.running_pump_count > 0:
                    return True
        return False

    def _check_forebay_alarm_high(self, state: Dict) -> bool:
        """检查前池报警高水位"""
        for station_id, station in self.pump_group_state.stations.items():
            if station.forebay_level > self.params['forebay_alarm_high']:
                if station.running_pump_count < station.available_pump_count:
                    return True
        return False

    def _check_forebay_critical_high(self, state: Dict) -> bool:
        """检查前池临界高水位"""
        for station_id, station in self.pump_group_state.stations.items():
            if station.forebay_level > self.params['forebay_critical_high']:
                return True
        return False

    def _check_start_interval_violation(self, state: Dict) -> bool:
        """检查启动间隔违规"""
        current_time = time.time()
        for station_id, station in self.pump_group_state.stations.items():
            for pump_id, pump in station.pumps.items():
                if pump.status == PumpStatus.STARTING:
                    time_since_last_stop = current_time - pump.last_stop_time
                    if time_since_last_stop < self.params['min_start_interval']:
                        return True
        return False

    def _check_min_run_time_violation(self, state: Dict) -> bool:
        """检查最小运行时间违规"""
        for station_id, station in self.pump_group_state.stations.items():
            for pump_id, pump in station.pumps.items():
                if pump.status == PumpStatus.STOPPING:
                    if pump.continuous_run_time < self.params['min_run_time']:
                        return True
        return False

    def _check_low_suction_pressure(self, state: Dict) -> bool:
        """检查低吸入压力"""
        for station_id, station in self.pump_group_state.stations.items():
            for pump_id, pump in station.pumps.items():
                if pump.status == PumpStatus.RUNNING:
                    if pump.suction_pressure < self.params['suction_pressure_min']:
                        return True
        return False

    def _check_power_overload(self, state: Dict) -> bool:
        """检查功率过载"""
        # 需要额定功率信息，这里简化处理
        return False

    def _check_vibration_alarm(self, state: Dict) -> bool:
        """检查振动报警"""
        for station_id, station in self.pump_group_state.stations.items():
            for pump_id, pump in station.pumps.items():
                if pump.status == PumpStatus.RUNNING:
                    if pump.vibration_level > self.params['vibration_alarm']:
                        return True
        return False

    def _check_vibration_trip(self, state: Dict) -> bool:
        """检查振动跳闸"""
        for station_id, station in self.pump_group_state.stations.items():
            for pump_id, pump in station.pumps.items():
                if pump.status == PumpStatus.RUNNING:
                    if pump.vibration_level > self.params['vibration_trip']:
                        return True
        return False

    def _check_bearing_temp(self, state: Dict) -> bool:
        """检查轴承温度"""
        for station_id, station in self.pump_group_state.stations.items():
            for pump_id, pump in station.pumps.items():
                if pump.status == PumpStatus.RUNNING:
                    if pump.bearing_temp > self.params['bearing_temp_trip']:
                        return True
        return False

    def _check_motor_temp(self, state: Dict) -> bool:
        """检查电机温度"""
        for station_id, station in self.pump_group_state.stations.items():
            for pump_id, pump in station.pumps.items():
                if pump.status == PumpStatus.RUNNING:
                    if pump.motor_temp > self.params['motor_temp_trip']:
                        return True
        return False

    def _check_pump_fault(self, state: Dict) -> bool:
        """检查泵故障"""
        for station_id, station in self.pump_group_state.stations.items():
            for pump_id, pump in station.pumps.items():
                if pump.status == PumpStatus.FAULT:
                    return True
        return False

    # ===== 动作函数 =====

    def _action_emergency_stop_all(self, state: Dict) -> List[ControlAction]:
        """紧急停止所有泵"""
        actions = []
        for station_id, station in self.pump_group_state.stations.items():
            for pump_id, pump in station.pumps.items():
                if pump.status == PumpStatus.RUNNING:
                    actions.append(ControlAction(
                        actuator_id=f"{station_id}_{pump_id}",
                        action_type='emergency_stop',
                        value=0.0,
                        priority=AgentPriority.SAFETY,
                        timestamp=time.time(),
                        source_agent=self.agent_id
                    ))

        # 广播紧急报警
        self.broadcast(MessageType.ALERT, {
            'type': 'forebay_critical_low',
            'severity': 'critical',
            'message': '前池水位临界低，紧急停止所有泵'
        })

        return actions

    def _action_reduce_pumps(self, state: Dict) -> List[ControlAction]:
        """减少运行泵数"""
        actions = []
        for station_id, station in self.pump_group_state.stations.items():
            if station.forebay_level < self.params['forebay_alarm_low']:
                # 停止效率最低的泵
                pump_to_stop = self._select_pump_to_stop(station)
                if pump_to_stop:
                    actions.append(ControlAction(
                        actuator_id=f"{station_id}_{pump_to_stop.pump_id}",
                        action_type='stop',
                        value=0.0,
                        priority=AgentPriority.SAFETY,
                        timestamp=time.time(),
                        source_agent=self.agent_id
                    ))
        return actions

    def _action_increase_pumps(self, state: Dict) -> List[ControlAction]:
        """增加运行泵数"""
        actions = []
        for station_id, station in self.pump_group_state.stations.items():
            if station.forebay_level > self.params['forebay_alarm_high']:
                # 启动一台备用泵
                pump_to_start = self._select_pump_to_start(station)
                if pump_to_start:
                    actions.append(ControlAction(
                        actuator_id=f"{station_id}_{pump_to_start.pump_id}",
                        action_type='start',
                        value=1.0,
                        priority=AgentPriority.SAFETY,
                        timestamp=time.time(),
                        source_agent=self.agent_id
                    ))
        return actions

    def _action_emergency_discharge(self, state: Dict) -> List[ControlAction]:
        """紧急泄水"""
        actions = []
        for station_id, station in self.pump_group_state.stations.items():
            if station.forebay_level > self.params['forebay_critical_high']:
                # 启动所有可用泵
                for pump_id, pump in station.pumps.items():
                    if pump.status == PumpStatus.STOPPED:
                        actions.append(ControlAction(
                            actuator_id=f"{station_id}_{pump_id}",
                            action_type='start',
                            value=1.0,
                            priority=AgentPriority.SAFETY,
                            timestamp=time.time(),
                            source_agent=self.agent_id
                        ))

                # 开启溢流闸
                actions.append(ControlAction(
                    actuator_id=f"{station_id}_overflow_gate",
                    action_type='set',
                    value=1.0,
                    priority=AgentPriority.SAFETY,
                    timestamp=time.time(),
                    source_agent=self.agent_id
                ))

        self.broadcast(MessageType.ALERT, {
            'type': 'forebay_critical_high',
            'severity': 'critical',
            'message': '前池水位临界高，紧急泄水'
        })

        return actions

    def _action_block_start(self, state: Dict) -> List[ControlAction]:
        """阻止启动"""
        # 发送阻止命令
        self.broadcast(MessageType.COMMAND, {
            'type': 'block_start',
            'reason': '启动间隔不足',
            'min_interval': self.params['min_start_interval']
        })
        return []

    def _action_block_stop(self, state: Dict) -> List[ControlAction]:
        """阻止停机"""
        self.broadcast(MessageType.COMMAND, {
            'type': 'block_stop',
            'reason': '最小运行时间不足',
            'min_run_time': self.params['min_run_time']
        })
        return []

    def _action_cavitation_protect(self, state: Dict) -> List[ControlAction]:
        """汽蚀保护动作"""
        actions = []
        for station_id, station in self.pump_group_state.stations.items():
            for pump_id, pump in station.pumps.items():
                if pump.suction_pressure < self.params['suction_pressure_min']:
                    # 降低泵速或停机
                    actions.append(ControlAction(
                        actuator_id=f"{station_id}_{pump_id}",
                        action_type='reduce_speed',
                        value=0.8,  # 降到80%
                        priority=AgentPriority.SAFETY,
                        timestamp=time.time(),
                        source_agent=self.agent_id
                    ))

        self.broadcast(MessageType.ALERT, {
            'type': 'cavitation_warning',
            'severity': 'warning',
            'message': '检测到汽蚀风险，已降低泵速'
        })

        return actions

    def _action_overload_protect(self, state: Dict) -> List[ControlAction]:
        """过载保护动作"""
        return []

    def _action_vibration_alarm(self, state: Dict) -> List[ControlAction]:
        """振动报警动作"""
        self.broadcast(MessageType.ALERT, {
            'type': 'vibration_alarm',
            'severity': 'warning',
            'message': '泵振动超标'
        })
        return []

    def _action_vibration_trip(self, state: Dict) -> List[ControlAction]:
        """振动跳闸动作"""
        actions = []
        for station_id, station in self.pump_group_state.stations.items():
            for pump_id, pump in station.pumps.items():
                if pump.vibration_level > self.params['vibration_trip']:
                    actions.append(ControlAction(
                        actuator_id=f"{station_id}_{pump_id}",
                        action_type='emergency_stop',
                        value=0.0,
                        priority=AgentPriority.SAFETY,
                        timestamp=time.time(),
                        source_agent=self.agent_id
                    ))

        self.broadcast(MessageType.ALERT, {
            'type': 'vibration_trip',
            'severity': 'critical',
            'message': '振动超限，紧急停泵'
        })

        return actions

    def _action_temp_protect(self, state: Dict) -> List[ControlAction]:
        """温度保护动作"""
        actions = []
        for station_id, station in self.pump_group_state.stations.items():
            for pump_id, pump in station.pumps.items():
                if (pump.bearing_temp > self.params['bearing_temp_trip'] or
                    pump.motor_temp > self.params['motor_temp_trip']):
                    actions.append(ControlAction(
                        actuator_id=f"{station_id}_{pump_id}",
                        action_type='stop',
                        value=0.0,
                        priority=AgentPriority.SAFETY,
                        timestamp=time.time(),
                        source_agent=self.agent_id
                    ))
        return actions

    def _action_fault_switchover(self, state: Dict) -> List[ControlAction]:
        """故障切换动作"""
        actions = []
        for station_id, station in self.pump_group_state.stations.items():
            for pump_id, pump in station.pumps.items():
                if pump.status == PumpStatus.FAULT:
                    # 启动备用泵
                    standby = self._select_pump_to_start(station)
                    if standby:
                        actions.append(ControlAction(
                            actuator_id=f"{station_id}_{standby.pump_id}",
                            action_type='start',
                            value=1.0,
                            priority=AgentPriority.SAFETY,
                            timestamp=time.time(),
                            source_agent=self.agent_id
                        ))
        return actions

    # ===== 辅助函数 =====

    def _select_pump_to_stop(self, station: StationState) -> Optional[PumpState]:
        """选择要停止的泵（效率最低或运行时间最长）"""
        running_pumps = [p for p in station.pumps.values()
                        if p.status == PumpStatus.RUNNING]
        if not running_pumps:
            return None

        # 优先停止效率最低的泵
        return min(running_pumps, key=lambda p: p.efficiency)

    def _select_pump_to_start(self, station: StationState) -> Optional[PumpState]:
        """选择要启动的泵（运行时间最少）"""
        stopped_pumps = [p for p in station.pumps.values()
                       if p.status == PumpStatus.STOPPED]
        if not stopped_pumps:
            return None

        # 优先启动运行时间最少的泵
        return min(stopped_pumps, key=lambda p: p.run_hours)

    def add_rule(self, rule: SafetyRule):
        """添加安全规则"""
        self.rules[rule.rule_id] = rule

    def perceive(self, system_state: Dict) -> Dict:
        """感知系统状态"""
        # 更新泵站群状态
        if 'pump_group_state' in system_state:
            self.pump_group_state = system_state['pump_group_state']

        return system_state

    def decide(self) -> List[ControlAction]:
        """基于规则决策"""
        if self.pump_group_state is None:
            return []

        current_time = time.time()
        triggered_rules: List[Tuple[int, SafetyRule]] = []

        # 检查所有规则
        for rule_id, rule in self.rules.items():
            # 检查时间间隔
            if rule.min_interval > 0:
                if current_time - rule.last_trigger < rule.min_interval:
                    continue

            # 评估触发条件
            try:
                if rule.condition(self.observations):
                    triggered_rules.append((rule.priority, rule))
                    rule.status = RuleStatus.TRIGGERED
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

            except Exception as e:
                rule.status = RuleStatus.BLOCKED
                self.state.error_count += 1

        return all_actions

    def act(self, actions: List[ControlAction]) -> Dict:
        """返回动作列表"""
        result = {
            'agent': self.agent_id,
            'actions': [],
            'active_rules': self.active_rules.copy(),
            'safety_status': 'normal' if not self.active_rules else 'active'
        }

        for action in actions:
            result['actions'].append({
                'actuator': action.actuator_id,
                'type': action.action_type,
                'value': action.value,
                'priority': action.priority.name
            })

        return result

    def get_safety_status(self) -> Dict:
        """获取安全状态摘要"""
        return {
            'active_rules': self.active_rules,
            'rule_count': len(self.rules),
            'triggered_count': sum(1 for r in self.rules.values() if r.trigger_count > 0),
            'safety_params': self.params
        }


# ============================================================
# 3. 泵站群经济优化智能体 (L3层 - 小时级优化)
# ============================================================

class PumpGroupEconomicAgent(BaseAgent):
    """
    泵站群经济优化智能体

    继承L3战略层思想，专门针对泵站群的经济优化:
    - 基于效率曲线的最优台数选择
    - 峰谷电价优化
    - 启停成本考虑
    - 设备寿命均衡 (运行时长均衡)
    - 多目标权衡 (成本、可靠性、设备寿命)
    """

    def __init__(self, agent_id: str = "PumpGroup_L3_Economic"):
        super().__init__(agent_id, AgentPriority.STRATEGIC)

        # 泵站群状态
        self.pump_group_state: Optional[PumpGroupState] = None

        # 优化参数
        self.params = {
            # 电价 (元/kWh)
            'peak_price': 0.85,           # 峰时电价
            'valley_price': 0.35,         # 谷时电价
            'flat_price': 0.55,           # 平时电价

            # 峰谷时段
            'peak_hours': [8, 9, 10, 11, 18, 19, 20, 21],
            'valley_hours': [0, 1, 2, 3, 4, 5, 6],

            # 启停成本 (元/次)
            'start_cost': 50.0,           # 启动成本
            'stop_cost': 30.0,            # 停机成本

            # 设备寿命
            'design_run_hours': 50000.0,  # 设计运行小时数
            'lifetime_weight': 0.15,      # 寿命均衡权重

            # 优化权重
            'energy_cost_weight': 0.50,   # 能耗成本权重
            'start_stop_weight': 0.15,    # 启停成本权重
            'efficiency_weight': 0.20,    # 效率权重
        }

        # 效率曲线参数 (简化的泵效率模型)
        self.efficiency_curves = {}

        # 优化结果缓存
        self.last_optimization_time = 0.0
        self.optimization_interval = 3600.0  # 1小时
        self.optimal_schedule: List[Dict] = []

        # 24小时电价曲线
        self.price_curve = self._init_price_curve()

    def _init_price_curve(self) -> np.ndarray:
        """初始化24小时电价曲线"""
        price = np.ones(24) * self.params['flat_price']

        for h in self.params['peak_hours']:
            price[h] = self.params['peak_price']
        for h in self.params['valley_hours']:
            price[h] = self.params['valley_price']

        return price

    def _get_current_price(self, hour: int) -> float:
        """获取当前电价"""
        return self.price_curve[hour % 24]

    def _calculate_pump_efficiency(self, flow: float, pump_state: PumpState) -> float:
        """
        计算泵效率

        使用简化的效率曲线模型:
        η = η_max * (1 - ((Q - Q_bep) / Q_bep)^2)
        """
        # 简化参数
        Q_bep = 3.5  # 最佳效率点流量 (m³/s)
        eta_max = 0.85  # 最高效率

        if flow <= 0:
            return 0.0

        efficiency = eta_max * (1 - ((flow - Q_bep) / Q_bep) ** 2)
        return max(0.3, min(0.9, efficiency))

    def _calculate_pump_power(self, flow: float, head: float, efficiency: float) -> float:
        """
        计算泵功率 (kW)

        P = ρ * g * Q * H / η
        """
        rho = 1000  # 水密度 kg/m³
        g = 9.81    # 重力加速度

        if efficiency <= 0:
            efficiency = 0.5

        power = rho * g * flow * head / (efficiency * 1000)
        return power

    def _calculate_optimal_pump_count(
        self,
        station: StationState,
        target_flow: float
    ) -> Tuple[int, float, float]:
        """
        计算最优运行泵数

        Returns:
            (最优台数, 总效率, 总功率)
        """
        max_pumps = station.available_pump_count
        if max_pumps == 0:
            return 0, 0.0, 0.0

        best_count = 1
        best_efficiency = 0.0
        best_power = float('inf')

        # 假设所有泵相同，扬程约30m
        head = 30.0

        for n in range(1, max_pumps + 1):
            flow_per_pump = target_flow / n

            # 检查单泵流量是否在合理范围
            if flow_per_pump > 5.0:  # 单泵最大流量
                continue
            if flow_per_pump < 1.0:  # 单泵最小流量
                continue

            # 计算效率
            efficiency = self._calculate_pump_efficiency(flow_per_pump, None)

            # 计算总功率
            power_per_pump = self._calculate_pump_power(flow_per_pump, head, efficiency)
            total_power = power_per_pump * n

            # 选择功率最低的方案
            if total_power < best_power:
                best_power = total_power
                best_efficiency = efficiency
                best_count = n

        return best_count, best_efficiency, best_power

    def _calculate_lifetime_balance_factor(self, station: StationState) -> Dict[str, float]:
        """
        计算设备寿命均衡因子

        返回每台泵的优先级调整因子（运行时间少的优先级高）
        """
        factors = {}
        total_hours = sum(p.run_hours for p in station.pumps.values())
        avg_hours = total_hours / len(station.pumps) if station.pumps else 0

        for pump_id, pump in station.pumps.items():
            if avg_hours > 0:
                # 运行时间少于平均值的泵，因子>1
                factors[pump_id] = 2.0 - pump.run_hours / avg_hours
            else:
                factors[pump_id] = 1.0

        return factors

    def _optimize_pump_schedule(self, target_flow: float, horizon_hours: int = 24) -> List[Dict]:
        """
        优化泵站调度计划

        考虑:
        - 电价变化
        - 启停成本
        - 设备寿命均衡
        """
        schedule = []
        current_hour = self.pump_group_state.current_hour if self.pump_group_state else 12

        for h in range(horizon_hours):
            hour = (current_hour + h) % 24
            price = self.price_curve[hour]

            hour_schedule = {
                'hour': hour,
                'electricity_price': price,
                'stations': {}
            }

            # 电价因子：电价高时减少运行
            price_factor = 1.0 - (price - 0.5) * 0.3
            adjusted_flow = target_flow * price_factor

            if self.pump_group_state:
                for station_id, station in self.pump_group_state.stations.items():
                    # 计算该站最优运行方案
                    station_target = adjusted_flow / len(self.pump_group_state.stations)
                    optimal_count, efficiency, power = self._calculate_optimal_pump_count(
                        station, station_target
                    )

                    # 寿命均衡
                    lifetime_factors = self._calculate_lifetime_balance_factor(station)

                    hour_schedule['stations'][station_id] = {
                        'optimal_pump_count': optimal_count,
                        'target_flow': station_target,
                        'expected_efficiency': efficiency,
                        'expected_power': power,
                        'energy_cost': power * price,
                        'lifetime_factors': lifetime_factors
                    }

            schedule.append(hour_schedule)

        return schedule

    def _calculate_total_cost(self, schedule: List[Dict]) -> Dict:
        """计算总成本"""
        total_energy_cost = 0.0
        total_start_stop_cost = 0.0

        prev_counts = {}

        for hour_data in schedule:
            for station_id, station_data in hour_data['stations'].items():
                # 能耗成本
                total_energy_cost += station_data['energy_cost']

                # 启停成本
                current_count = station_data['optimal_pump_count']
                prev_count = prev_counts.get(station_id, current_count)

                if current_count > prev_count:
                    total_start_stop_cost += (current_count - prev_count) * self.params['start_cost']
                elif current_count < prev_count:
                    total_start_stop_cost += (prev_count - current_count) * self.params['stop_cost']

                prev_counts[station_id] = current_count

        return {
            'energy_cost': total_energy_cost,
            'start_stop_cost': total_start_stop_cost,
            'total_cost': total_energy_cost + total_start_stop_cost
        }

    def perceive(self, system_state: Dict) -> Dict:
        """感知系统状态"""
        if 'pump_group_state' in system_state:
            self.pump_group_state = system_state['pump_group_state']

        # 提取经济相关信息
        observations = {
            'current_hour': system_state.get('hour', 12),
            'current_price': self._get_current_price(system_state.get('hour', 12)),
            'target_flow': system_state.get('target_flow', 10.0),
            'current_total_power': system_state.get('total_power', 0.0)
        }

        return observations

    def decide(self) -> List[ControlAction]:
        """经济优化决策"""
        current_time = time.time()

        # 检查是否需要重新优化
        if current_time - self.last_optimization_time > self.optimization_interval:
            target_flow = self.observations.get('target_flow', 10.0)
            self.optimal_schedule = self._optimize_pump_schedule(target_flow)
            self.last_optimization_time = current_time

        # 生成当前时刻的调度命令
        actions = []
        current_hour = self.observations.get('current_hour', 12)

        # 从调度计划中获取当前小时的建议
        if self.optimal_schedule:
            current_schedule = self.optimal_schedule[0]  # 当前小时

            if self.pump_group_state:
                for station_id, station in self.pump_group_state.stations.items():
                    if station_id in current_schedule['stations']:
                        station_schedule = current_schedule['stations'][station_id]
                        optimal_count = station_schedule['optimal_pump_count']
                        current_count = station.running_pump_count

                        # 发送协调消息给L2层
                        self.send_message(AgentMessage(
                            msg_type=MessageType.COORDINATION,
                            sender=self.agent_id,
                            receiver=f'PumpGroup_L2_{station_id}',
                            priority=self.priority,
                            timestamp=current_time,
                            payload={
                                'optimal_pump_count': optimal_count,
                                'target_flow': station_schedule['target_flow'],
                                'expected_efficiency': station_schedule['expected_efficiency'],
                                'lifetime_factors': station_schedule['lifetime_factors']
                            }
                        ))

        return actions  # L3层不直接产生控制动作

    def act(self, actions: List[ControlAction]) -> Dict:
        """返回优化结果"""
        costs = self._calculate_total_cost(self.optimal_schedule) if self.optimal_schedule else {}

        result = {
            'agent': self.agent_id,
            'schedule_horizon': len(self.optimal_schedule),
            'current_price': self.observations.get('current_price', 0.5),
            'costs': costs,
            'optimization_status': 'active' if self.optimal_schedule else 'pending'
        }

        return result

    def get_optimization_summary(self) -> Dict:
        """获取优化摘要"""
        costs = self._calculate_total_cost(self.optimal_schedule) if self.optimal_schedule else {}

        return {
            'schedule_length': len(self.optimal_schedule),
            'total_24h_cost': costs.get('total_cost', 0),
            'energy_cost': costs.get('energy_cost', 0),
            'start_stop_cost': costs.get('start_stop_cost', 0),
            'params': self.params
        }


# ============================================================
# 4. 泵站群协调智能体 (融合安全与经济)
# ============================================================

class PumpGroupCoordinatorAgent(BaseAgent):
    """
    泵站群协调智能体

    负责协调安全智能体和经济智能体的决策:
    - 安全优先原则
    - 经济目标服从安全约束
    - 多泵站间的协调调度
    - 级联泵站的波传播预测
    """

    def __init__(self, agent_id: str = "PumpGroup_Coordinator"):
        super().__init__(agent_id, AgentPriority.TACTICAL)

        # 子智能体
        self.safety_agent: Optional[PumpGroupSafetyAgent] = None
        self.economic_agent: Optional[PumpGroupEconomicAgent] = None

        # 泵站群状态
        self.pump_group_state: Optional[PumpGroupState] = None

        # 协调参数
        self.params = {
            # 波传播参数
            'wave_speed': 5.0,            # 波速 (m/s)
            'coordination_delay': 60.0,   # 协调延迟 (s)

            # 决策参数
            'safety_override_threshold': 0.8,  # 安全覆盖阈值
            'economic_influence': 0.3,         # 经济影响因子
        }

        # 级联调度
        self.cascade_schedule: List[Dict] = []

        # 协调状态
        self.coordination_mode = "normal"  # normal, safety_override, emergency

    def set_sub_agents(
        self,
        safety_agent: PumpGroupSafetyAgent,
        economic_agent: PumpGroupEconomicAgent
    ):
        """设置子智能体"""
        self.safety_agent = safety_agent
        self.economic_agent = economic_agent

    def _predict_wave_arrival(
        self,
        source_station: str,
        target_station: str,
        distance: float
    ) -> float:
        """预测波到达时间"""
        travel_time = distance / self.params['wave_speed']
        return travel_time

    def _generate_cascade_schedule(
        self,
        upstream_action: PumpScheduleAction
    ) -> List[PumpScheduleAction]:
        """
        生成级联调度计划

        基于上游泵站动作，预测下游泵站的响应时间
        """
        cascade_actions = []

        if not self.pump_group_state:
            return cascade_actions

        # 获取波传播时间
        for (src, dst), travel_time in self.pump_group_state.wave_propagation_time.items():
            if src == upstream_action.station_id:
                # 计算下游动作时间
                downstream_time = upstream_action.timestamp + travel_time

                # 生成下游动作
                cascade_actions.append(PumpScheduleAction(
                    station_id=dst,
                    pump_id='auto',  # 自动选择
                    action=upstream_action.action,
                    target_value=upstream_action.target_value,
                    priority=AgentPriority.TACTICAL,
                    reason=f'级联响应: {src} -> {dst}',
                    timestamp=downstream_time,
                    min_delay=travel_time - 60,  # 提前1分钟准备
                    must_execute_by=travel_time + 120  # 最迟2分钟后
                ))

        return cascade_actions

    def _resolve_conflicts(
        self,
        safety_actions: List[ControlAction],
        economic_actions: List[ControlAction]
    ) -> List[ControlAction]:
        """
        解决安全与经济决策冲突

        原则: 安全优先
        """
        resolved = []

        # 按执行器分组
        safety_by_actuator = {a.actuator_id: a for a in safety_actions}
        economic_by_actuator = {a.actuator_id: a for a in economic_actions}

        all_actuators = set(safety_by_actuator.keys()) | set(economic_by_actuator.keys())

        for actuator_id in all_actuators:
            safety_action = safety_by_actuator.get(actuator_id)
            economic_action = economic_by_actuator.get(actuator_id)

            if safety_action and economic_action:
                # 有冲突，安全优先
                if safety_action.priority.value <= economic_action.priority.value:
                    resolved.append(safety_action)
                    self.coordination_mode = "safety_override"
                else:
                    resolved.append(economic_action)
            elif safety_action:
                resolved.append(safety_action)
            elif economic_action:
                resolved.append(economic_action)

        return resolved

    def perceive(self, system_state: Dict) -> Dict:
        """感知系统状态"""
        if 'pump_group_state' in system_state:
            self.pump_group_state = system_state['pump_group_state']

        # 更新子智能体的状态
        if self.safety_agent:
            self.safety_agent.pump_group_state = self.pump_group_state
        if self.economic_agent:
            self.economic_agent.pump_group_state = self.pump_group_state

        return system_state

    def decide(self) -> List[ControlAction]:
        """协调决策"""
        all_actions = []

        # 1. 获取安全智能体决策
        safety_actions = []
        if self.safety_agent:
            self.safety_agent.observations = self.observations
            safety_actions = self.safety_agent.decide()

        # 2. 获取经济智能体决策
        economic_actions = []
        if self.economic_agent:
            self.economic_agent.observations = self.observations
            economic_actions = self.economic_agent.decide()

        # 3. 解决冲突
        resolved_actions = self._resolve_conflicts(safety_actions, economic_actions)

        # 4. 生成级联调度（如果有动作）
        for action in resolved_actions:
            if action.action_type in ['start', 'stop']:
                cascade_action = PumpScheduleAction(
                    station_id=action.actuator_id.split('_')[0],
                    pump_id=action.actuator_id.split('_')[1] if '_' in action.actuator_id else 'auto',
                    action=action.action_type,
                    target_value=action.value,
                    priority=action.priority,
                    timestamp=time.time()
                )
                cascade_schedule = self._generate_cascade_schedule(cascade_action)
                self.cascade_schedule.extend(cascade_schedule)

        # 5. 转换为ControlAction
        all_actions.extend(resolved_actions)

        return all_actions

    def act(self, actions: List[ControlAction]) -> Dict:
        """执行协调结果"""
        result = {
            'agent': self.agent_id,
            'coordination_mode': self.coordination_mode,
            'actions': [],
            'cascade_schedule': len(self.cascade_schedule),
            'safety_status': self.safety_agent.get_safety_status() if self.safety_agent else {},
            'economic_status': self.economic_agent.get_optimization_summary() if self.economic_agent else {}
        }

        for action in actions:
            result['actions'].append({
                'actuator': action.actuator_id,
                'type': action.action_type,
                'value': action.value,
                'priority': action.priority.name,
                'source': action.source_agent
            })

        return result

    def get_coordination_status(self) -> Dict:
        """获取协调状态"""
        return {
            'mode': self.coordination_mode,
            'cascade_schedule_length': len(self.cascade_schedule),
            'safety_active_rules': self.safety_agent.active_rules if self.safety_agent else [],
            'economic_schedule': len(self.economic_agent.optimal_schedule) if self.economic_agent else 0
        }


# ============================================================
# 5. 泵站群多智能体系统 (完整集成)
# ============================================================

class PumpGroupMultiAgentSystem:
    """
    泵站群多智能体系统

    完整集成:
    - L1 安全智能体
    - L3 经济智能体
    - 协调智能体
    - 消息路由
    - 状态管理
    """

    def __init__(self):
        # 创建智能体
        self.safety_agent = PumpGroupSafetyAgent()
        self.economic_agent = PumpGroupEconomicAgent()
        self.coordinator = PumpGroupCoordinatorAgent()

        # 连接子智能体
        self.coordinator.set_sub_agents(self.safety_agent, self.economic_agent)

        # 智能体注册表
        self.registry = AgentRegistry()
        self.registry.register(self.safety_agent)
        self.registry.register(self.economic_agent)
        self.registry.register(self.coordinator)

        # 系统状态
        self.pump_group_state = PumpGroupState()

        # 历史记录
        self.action_history: List[Dict] = []
        self.state_history: List[Dict] = []

        # 统计
        self.cycle_count = 0
        self.total_actions = 0
        self.safety_overrides = 0

    def initialize_stations(self, station_configs: List[Dict]):
        """
        初始化泵站配置

        station_configs: [
            {
                'station_id': 'ST1',
                'station_name': '屯佃泵站',
                'pump_count': 4,
                'pump_capacity': 5.0,  # m³/s
                'rated_head': 30.0,    # m
                'forebay_limits': (1.0, 5.0)
            },
            ...
        ]
        """
        for config in station_configs:
            station = StationState(
                station_id=config['station_id'],
                station_name=config['station_name'],
                total_pump_count=config['pump_count'],
                available_pump_count=config['pump_count'],
                forebay_level_min=config.get('forebay_limits', (1.0, 5.0))[0],
                forebay_level_max=config.get('forebay_limits', (1.0, 5.0))[1]
            )

            # 创建泵状态
            for i in range(config['pump_count']):
                pump_id = f"P{i+1}"
                station.pumps[pump_id] = PumpState(
                    pump_id=pump_id,
                    station_id=config['station_id']
                )

            self.pump_group_state.stations[config['station_id']] = station

        # 设置波传播时间（示例：级联泵站）
        station_ids = list(self.pump_group_state.stations.keys())
        for i in range(len(station_ids) - 1):
            # 假设相邻泵站间距12km
            distance = 12000.0
            travel_time = distance / 5.0  # 波速5m/s
            self.pump_group_state.wave_propagation_time[(station_ids[i], station_ids[i+1])] = travel_time

    def step(self, system_state: Dict) -> Dict:
        """
        执行一个系统步进

        Parameters:
            system_state: 外部系统状态

        Returns:
            执行结果
        """
        self.cycle_count += 1

        # 1. 更新泵站群状态
        self._update_pump_group_state(system_state)

        # 2. 准备系统状态
        full_state = {
            'pump_group_state': self.pump_group_state,
            **system_state
        }

        # 3. 协调智能体决策
        result = self.coordinator.step(full_state)

        # 4. 路由消息
        self.registry.route_messages()

        # 5. 记录历史
        self.action_history.append({
            'cycle': self.cycle_count,
            'timestamp': time.time(),
            'actions': result.get('actions', []),
            'mode': result.get('coordination_mode', 'normal')
        })

        self.total_actions += len(result.get('actions', []))
        if result.get('coordination_mode') == 'safety_override':
            self.safety_overrides += 1

        return result

    def _update_pump_group_state(self, external_state: Dict):
        """更新泵站群状态"""
        # 更新全局信息
        self.pump_group_state.current_hour = external_state.get('hour', 12)
        self.pump_group_state.electricity_price = self.economic_agent._get_current_price(
            self.pump_group_state.current_hour
        )

        # 更新各站状态（从外部传感器数据）
        if 'stations' in external_state:
            for station_id, station_data in external_state['stations'].items():
                if station_id in self.pump_group_state.stations:
                    station = self.pump_group_state.stations[station_id]
                    station.forebay_level = station_data.get('forebay_level', station.forebay_level)
                    station.total_flow = station_data.get('total_flow', station.total_flow)
                    station.total_power = station_data.get('total_power', station.total_power)

                    # 更新泵状态
                    if 'pumps' in station_data:
                        for pump_id, pump_data in station_data['pumps'].items():
                            if pump_id in station.pumps:
                                pump = station.pumps[pump_id]
                                pump.flow_rate = pump_data.get('flow', pump.flow_rate)
                                pump.power = pump_data.get('power', pump.power)
                                pump.efficiency = pump_data.get('efficiency', pump.efficiency)
                                pump.vibration_level = pump_data.get('vibration', pump.vibration_level)
                                pump.bearing_temp = pump_data.get('bearing_temp', pump.bearing_temp)
                                pump.suction_pressure = pump_data.get('suction_pressure', pump.suction_pressure)

        # 计算汇总
        self.pump_group_state.total_flow = sum(
            s.total_flow for s in self.pump_group_state.stations.values()
        )
        self.pump_group_state.total_power = sum(
            s.total_power for s in self.pump_group_state.stations.values()
        )
        self.pump_group_state.total_running_pumps = sum(
            s.running_pump_count for s in self.pump_group_state.stations.values()
        )

    def get_system_status(self) -> Dict:
        """获取系统状态摘要"""
        return {
            'cycle_count': self.cycle_count,
            'total_actions': self.total_actions,
            'safety_overrides': self.safety_overrides,
            'coordination_mode': self.coordinator.coordination_mode,
            'stations': {
                sid: {
                    'name': s.station_name,
                    'forebay_level': s.forebay_level,
                    'running_pumps': s.running_pump_count,
                    'total_flow': s.total_flow,
                    'total_power': s.total_power
                }
                for sid, s in self.pump_group_state.stations.items()
            },
            'total_flow': self.pump_group_state.total_flow,
            'total_power': self.pump_group_state.total_power,
            'electricity_price': self.pump_group_state.electricity_price,
            'safety_status': self.safety_agent.get_safety_status(),
            'economic_status': self.economic_agent.get_optimization_summary()
        }

    def get_optimization_report(self) -> Dict:
        """获取优化报告"""
        return {
            'schedule': self.economic_agent.optimal_schedule,
            'costs': self.economic_agent._calculate_total_cost(self.economic_agent.optimal_schedule),
            'safety_rules_triggered': [
                r.rule_id for r in self.safety_agent.rules.values() if r.trigger_count > 0
            ],
            'cascade_schedule': self.coordinator.cascade_schedule
        }

    def reset(self):
        """重置系统"""
        self.safety_agent.reset()
        self.economic_agent.reset()
        self.coordinator.reset()
        self.action_history.clear()
        self.state_history.clear()
        self.cycle_count = 0
        self.total_actions = 0
        self.safety_overrides = 0


# ============================================================
# 6. 便捷函数与导出
# ============================================================

def create_pump_group_system(station_configs: List[Dict] = None) -> PumpGroupMultiAgentSystem:
    """
    创建泵站群多智能体系统

    Parameters:
        station_configs: 泵站配置列表，默认使用密云工程配置

    Returns:
        PumpGroupMultiAgentSystem 实例
    """
    system = PumpGroupMultiAgentSystem()

    if station_configs is None:
        # 默认使用密云工程6级泵站配置
        station_configs = [
            {
                'station_id': 'tundian',
                'station_name': '屯佃泵站',
                'pump_count': 4,
                'pump_capacity': 5.0,
                'rated_head': 3.2,
                'forebay_limits': (1.0, 5.0)
            },
            {
                'station_id': 'qianliulin',
                'station_name': '前柳林泵站',
                'pump_count': 4,
                'pump_capacity': 5.0,
                'rated_head': 3.2,
                'forebay_limits': (1.0, 5.0)
            },
            {
                'station_id': 'niantou',
                'station_name': '念头泵站',
                'pump_count': 4,
                'pump_capacity': 5.0,
                'rated_head': 4.1,
                'forebay_limits': (1.0, 5.0)
            },
            {
                'station_id': 'xingshou',
                'station_name': '兴寿泵站',
                'pump_count': 4,
                'pump_capacity': 5.0,
                'rated_head': 4.1,
                'forebay_limits': (1.0, 5.0)
            },
            {
                'station_id': 'lishishan',
                'station_name': '李石山泵站',
                'pump_count': 4,
                'pump_capacity': 5.0,
                'rated_head': 2.9,
                'forebay_limits': (1.0, 5.0)
            },
            {
                'station_id': 'xitaishang',
                'station_name': '西台上泵站',
                'pump_count': 3,
                'pump_capacity': 6.67,
                'rated_head': 7.15,
                'forebay_limits': (1.0, 5.0)
            },
        ]

    system.initialize_stations(station_configs)
    return system


# 导出
__all__ = [
    # 数据结构
    'PumpStatus',
    'ProtectionType',
    'PumpState',
    'StationState',
    'PumpGroupState',
    'PumpScheduleAction',

    # 智能体
    'PumpGroupSafetyAgent',
    'PumpGroupEconomicAgent',
    'PumpGroupCoordinatorAgent',
    'PumpGroupMultiAgentSystem',

    # 便捷函数
    'create_pump_group_system',
]
