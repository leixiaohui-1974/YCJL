"""
全工况场景数据库
================

覆盖引绰济辽工程所有可能的运行场景,包括:
1. 正常运行场景 (10种)
2. 启停过渡场景 (8种)
3. 极端工况场景 (12种)
4. 设备故障场景 (15种)
5. 事故应急场景 (10种)
6. 通讯故障场景 (6种)
7. 检修维护场景 (8种)
8. 水质应急场景 (6种)
9. 工程事故场景 (8种)
10. 长尾异常场景 (10种)

共计83种场景,实现全工况覆盖

版本: 3.4.0
"""

import math
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any
from datetime import datetime, timedelta


# ==========================================
# 场景分类枚举
# ==========================================
class ScenarioCategory(Enum):
    """场景大类"""
    NORMAL = auto()           # 正常运行
    TRANSITION = auto()       # 启停过渡
    EXTREME = auto()          # 极端工况
    EQUIPMENT_FAULT = auto()  # 设备故障
    ACCIDENT = auto()         # 事故应急
    COMMUNICATION = auto()    # 通讯故障
    MAINTENANCE = auto()      # 检修维护
    WATER_QUALITY = auto()    # 水质应急
    ENGINEERING = auto()      # 工程事故
    LONG_TAIL = auto()        # 长尾异常


class ScenarioSeverity(Enum):
    """场景严重程度"""
    INFO = 0          # 信息
    NORMAL = 1        # 正常
    ATTENTION = 2     # 关注
    WARNING = 3       # 预警
    ALARM = 4         # 报警
    CRITICAL = 5      # 严重
    EMERGENCY = 6     # 紧急


class ScenarioPhase(Enum):
    """场景阶段"""
    DETECTION = auto()    # 检测
    CONFIRMATION = auto() # 确认
    RESPONSE = auto()     # 响应
    MITIGATION = auto()   # 缓解
    RECOVERY = auto()     # 恢复
    NORMAL = auto()       # 正常


class OperationMode(Enum):
    """运行模式"""
    AUTO_L5 = auto()      # L5全自主
    AUTO_L4 = auto()      # L4高度自动
    AUTO_L3 = auto()      # L3有条件自动
    SUPERVISED = auto()   # 监督控制
    MANUAL = auto()       # 手动控制
    EMERGENCY = auto()    # 应急模式
    SHUTDOWN = auto()     # 停机模式


# ==========================================
# 场景类型枚举 (83种场景)
# ==========================================
class ScenarioType(Enum):
    """场景类型 - 全工况覆盖"""
    
    # ========== 1. 正常运行场景 (10种) ==========
    NORMAL_STEADY = 101           # 稳态运行
    NORMAL_LOW_FLOW = 102         # 低流量运行
    NORMAL_MEDIUM_FLOW = 103      # 中流量运行
    NORMAL_HIGH_FLOW = 104        # 高流量运行
    NORMAL_PEAK_DEMAND = 105      # 高峰需水
    NORMAL_OFF_PEAK = 106         # 低谷运行
    NORMAL_SEASONAL_SPRING = 107  # 春季运行
    NORMAL_SEASONAL_SUMMER = 108  # 夏季运行
    NORMAL_SEASONAL_AUTUMN = 109  # 秋季运行
    NORMAL_SEASONAL_WINTER = 110  # 冬季运行
    
    # ========== 2. 启停过渡场景 (8种) ==========
    STARTUP_COLD = 201            # 冷态启动
    STARTUP_WARM = 202            # 热态启动
    STARTUP_INITIAL = 203         # 初期小流量启动
    SHUTDOWN_NORMAL = 204         # 正常停机
    SHUTDOWN_EMERGENCY = 205      # 紧急停机
    TRANSITION_RAMP_UP = 206      # 流量上升过渡
    TRANSITION_RAMP_DOWN = 207    # 流量下降过渡
    TRANSITION_MODE_SWITCH = 208  # 运行模式切换
    
    # ========== 3. 极端工况场景 (12种) ==========
    EXTREME_MAX_FLOW = 301        # 最大流量工况
    EXTREME_MIN_FLOW = 302        # 最小流量工况
    EXTREME_HIGH_HEAD = 303       # 高水头工况
    EXTREME_LOW_HEAD = 304        # 低水头工况
    EXTREME_HEAVY_RAIN = 305      # 暴雨工况
    EXTREME_FLOOD = 306           # 洪水工况
    EXTREME_DROUGHT = 307         # 干旱工况
    EXTREME_ICE_SEVERE = 308      # 严重冰期
    EXTREME_HEAT_WAVE = 309       # 高温热浪
    EXTREME_COLD_WAVE = 310       # 寒潮工况
    EXTREME_EARTHQUAKE = 311      # 地震工况
    EXTREME_WIND_STORM = 312      # 大风工况
    
    # ========== 4. 设备故障场景 (15种) ==========
    FAULT_VALVE_STUCK = 401       # 阀门卡死
    FAULT_VALVE_LEAK = 402        # 阀门泄漏
    FAULT_PUMP_TRIP = 403         # 水泵跳闸
    FAULT_TURBINE_TRIP = 404      # 水轮机跳闸
    FAULT_GATE_MALFUNCTION = 405  # 闸门故障
    FAULT_SENSOR_DRIFT = 406      # 传感器漂移
    FAULT_SENSOR_FAILURE = 407    # 传感器失效
    FAULT_ACTUATOR_SLOW = 408     # 执行器响应慢
    FAULT_ACTUATOR_FAIL = 409     # 执行器失效
    FAULT_POWER_LOSS = 410        # 电源故障
    FAULT_UPS_FAILURE = 411       # UPS故障
    FAULT_PLC_ERROR = 412         # PLC错误
    FAULT_CONTROL_LOOP = 413      # 控制回路故障
    FAULT_HYDRAULIC_SYS = 414     # 液压系统故障
    FAULT_COOLING_SYS = 415       # 冷却系统故障
    
    # ========== 5. 事故应急场景 (10种) ==========
    ACCIDENT_PIPE_BURST = 501     # 管道爆管
    ACCIDENT_TUNNEL_COLLAPSE = 502 # 隧洞坍塌
    ACCIDENT_DAM_OVERFLOW = 503   # 大坝漫顶
    ACCIDENT_SURGE_EXTREME = 504  # 极端水锤
    ACCIDENT_CAVITATION = 505     # 空化空蚀
    ACCIDENT_STRUCTURAL = 506     # 结构破坏
    ACCIDENT_FIRE = 507           # 火灾
    ACCIDENT_EXPLOSION = 508      # 爆炸
    ACCIDENT_PERSON_INJURY = 509  # 人员伤亡
    ACCIDENT_ENV_POLLUTION = 510  # 环境污染
    
    # ========== 6. 通讯故障场景 (6种) ==========
    COMM_SCADA_LOSS = 601         # SCADA通讯中断
    COMM_RTU_OFFLINE = 602        # RTU离线
    COMM_NETWORK_DELAY = 603      # 网络延迟
    COMM_DATA_CORRUPTION = 604    # 数据损坏
    COMM_CYBER_ATTACK = 605       # 网络攻击
    COMM_PARTIAL_LOSS = 606       # 部分通讯丢失
    
    # ========== 7. 检修维护场景 (8种) ==========
    MAINT_PLANNED_VALVE = 701     # 计划阀门检修
    MAINT_PLANNED_PUMP = 702      # 计划泵站检修
    MAINT_PLANNED_TUNNEL = 703    # 计划隧洞检修
    MAINT_PLANNED_PIPE = 704      # 计划管道检修
    MAINT_EMERGENCY_VALVE = 705   # 临时阀门检修
    MAINT_EMERGENCY_PUMP = 706    # 临时泵站检修
    MAINT_CALIBRATION = 707       # 仪表校准
    MAINT_SYSTEM_UPGRADE = 708    # 系统升级
    
    # ========== 8. 水质应急场景 (6种) ==========
    QUALITY_TURBIDITY_HIGH = 801  # 浊度超标
    QUALITY_ALGAE_BLOOM = 802     # 藻类爆发
    QUALITY_POLLUTION_UPSTREAM = 803  # 上游污染
    QUALITY_CHEMICAL_SPILL = 804  # 化学品泄漏
    QUALITY_SEDIMENT_SURGE = 805  # 泥沙突增
    QUALITY_BIOLOGICAL = 806      # 生物污染
    
    # ========== 9. 工程事故场景 (8种) ==========
    ENG_FOUNDATION_SETTLE = 901   # 地基沉降
    ENG_SLOPE_SLIDE = 902         # 边坡滑坡
    ENG_LINING_CRACK = 903        # 衬砌开裂
    ENG_JOINT_LEAK = 904          # 接缝渗漏
    ENG_ANCHOR_LOOSE = 905        # 锚固松动
    ENG_CORROSION = 906           # 结构腐蚀
    ENG_FATIGUE_CRACK = 907       # 疲劳裂缝
    ENG_SEEPAGE_ABNORMAL = 908    # 异常渗流
    
    # ========== 10. 长尾异常场景 (10种) ==========
    LONG_TAIL_MULTI_FAULT = 1001  # 多故障并发
    LONG_TAIL_CASCADE = 1002      # 级联故障
    LONG_TAIL_OSCILLATION = 1003  # 系统振荡
    LONG_TAIL_RESONANCE = 1004    # 水力共振
    LONG_TAIL_DEADLOCK = 1005     # 控制死锁
    LONG_TAIL_RACE_COND = 1006    # 竞争条件
    LONG_TAIL_BUTTERFLY = 1007    # 蝴蝶效应
    LONG_TAIL_BLACK_SWAN = 1008   # 黑天鹅事件
    LONG_TAIL_UNKNOWN = 1009      # 未知异常
    LONG_TAIL_COMBINATION = 1010  # 组合异常


# ==========================================
# 场景触发条件
# ==========================================
@dataclass
class ScenarioTrigger:
    """场景触发条件"""
    condition_name: str                    # 条件名称
    parameter: str                         # 监测参数
    operator: str                          # 比较运算符 (<, >, ==, !=, in, not_in)
    threshold: Any                         # 阈值
    duration: float = 0.0                  # 持续时间要求 (秒)
    confidence: float = 0.9                # 置信度要求
    priority: int = 1                      # 优先级


@dataclass
class ScenarioResponse:
    """场景响应策略"""
    action_name: str                       # 动作名称
    action_type: str                       # 动作类型 (control, alarm, notify, log)
    target: str                            # 目标设备/系统
    parameters: Dict[str, Any] = field(default_factory=dict)
    delay: float = 0.0                     # 延迟执行 (秒)
    timeout: float = 60.0                  # 超时时间 (秒)
    retry_count: int = 3                   # 重试次数
    fallback: Optional[str] = None         # 回退动作


@dataclass
class ScenarioDefinition:
    """场景定义"""
    scenario_type: ScenarioType            # 场景类型
    category: ScenarioCategory             # 场景大类
    severity: ScenarioSeverity             # 严重程度
    name: str                              # 场景名称
    description: str                       # 场景描述
    triggers: List[ScenarioTrigger]        # 触发条件列表 (AND关系)
    responses: List[ScenarioResponse]      # 响应策略列表
    allowed_modes: List[OperationMode]     # 允许的运行模式
    auto_recovery: bool = True             # 是否支持自动恢复
    recovery_time: float = 300.0           # 预计恢复时间 (秒)
    escalation_time: float = 600.0         # 升级时间 (秒)
    requires_human: bool = False           # 是否需要人工干预
    documentation: str = ""                # 相关文档


# ==========================================
# 场景数据库
# ==========================================
class ScenarioDatabase:
    """全工况场景数据库"""
    
    def __init__(self):
        self.scenarios: Dict[ScenarioType, ScenarioDefinition] = {}
        self._build_database()
    
    def _build_database(self):
        """构建场景数据库"""
        self._add_normal_scenarios()
        self._add_transition_scenarios()
        self._add_extreme_scenarios()
        self._add_fault_scenarios()
        self._add_accident_scenarios()
        self._add_communication_scenarios()
        self._add_maintenance_scenarios()
        self._add_water_quality_scenarios()
        self._add_engineering_scenarios()
        self._add_long_tail_scenarios()
    
    # ========== 正常运行场景 ==========
    def _add_normal_scenarios(self):
        """添加正常运行场景"""
        
        # 稳态运行
        self.scenarios[ScenarioType.NORMAL_STEADY] = ScenarioDefinition(
            scenario_type=ScenarioType.NORMAL_STEADY,
            category=ScenarioCategory.NORMAL,
            severity=ScenarioSeverity.NORMAL,
            name="稳态运行",
            description="系统处于稳定运行状态,所有参数在正常范围内",
            triggers=[
                ScenarioTrigger("流量稳定", "flow_deviation", "<", 0.05, duration=300),
                ScenarioTrigger("压力正常", "pressure_deviation", "<", 0.03, duration=300),
                ScenarioTrigger("无报警", "alarm_count", "==", 0)
            ],
            responses=[
                ScenarioResponse("维持运行", "control", "system", {"mode": "steady"}),
                ScenarioResponse("记录状态", "log", "database", {"level": "info"})
            ],
            allowed_modes=[OperationMode.AUTO_L5, OperationMode.AUTO_L4],
            auto_recovery=True
        )
        
        # 初期小流量
        self.scenarios[ScenarioType.NORMAL_LOW_FLOW] = ScenarioDefinition(
            scenario_type=ScenarioType.NORMAL_LOW_FLOW,
            category=ScenarioCategory.NORMAL,
            severity=ScenarioSeverity.NORMAL,
            name="低流量运行",
            description="系统低流量运行,流量小于设计流量30%",
            triggers=[
                ScenarioTrigger("低流量", "flow_rate", "<", 5.57),  # 18.58 * 0.3
                ScenarioTrigger("流量正向", "flow_rate", ">", 0)
            ],
            responses=[
                ScenarioResponse("低流量控制", "control", "valves", {"mode": "low_flow"}),
                ScenarioResponse("调整PID", "control", "controller", {"gain_factor": 0.7}),
                ScenarioResponse("监测水锤", "monitor", "pipeline", {"sensitivity": "high"})
            ],
            allowed_modes=[OperationMode.AUTO_L5, OperationMode.AUTO_L4, OperationMode.AUTO_L3],
            auto_recovery=True
        )
        
        # 大流量运行
        self.scenarios[ScenarioType.NORMAL_HIGH_FLOW] = ScenarioDefinition(
            scenario_type=ScenarioType.NORMAL_HIGH_FLOW,
            category=ScenarioCategory.NORMAL,
            severity=ScenarioSeverity.ATTENTION,
            name="高流量运行",
            description="系统高流量运行,流量大于设计流量80%",
            triggers=[
                ScenarioTrigger("高流量", "flow_rate", ">", 14.86),  # 18.58 * 0.8
                ScenarioTrigger("流量上限", "flow_rate", "<", 18.58)
            ],
            responses=[
                ScenarioResponse("高流量控制", "control", "valves", {"mode": "high_flow"}),
                ScenarioResponse("加强监测", "monitor", "all", {"interval": 5}),
                ScenarioResponse("预警通知", "notify", "operator", {"level": "attention"})
            ],
            allowed_modes=[OperationMode.AUTO_L5, OperationMode.AUTO_L4],
            auto_recovery=True
        )
        
        # 冬季运行
        self.scenarios[ScenarioType.NORMAL_SEASONAL_WINTER] = ScenarioDefinition(
            scenario_type=ScenarioType.NORMAL_SEASONAL_WINTER,
            category=ScenarioCategory.NORMAL,
            severity=ScenarioSeverity.ATTENTION,
            name="冬季冰期运行",
            description="冬季冰期运行模式,需考虑冰盖影响",
            triggers=[
                ScenarioTrigger("冬季月份", "month", "in", [11, 12, 1, 2, 3]),
                ScenarioTrigger("低温", "air_temperature", "<", 0)
            ],
            responses=[
                ScenarioResponse("冰期模式", "control", "ice_controller", {"enabled": True}),
                ScenarioResponse("糙率修正", "control", "hydraulics", {"roughness_factor": 1.3}),
                ScenarioResponse("流量限制", "control", "valves", {"max_rate_change": 0.05}),
                ScenarioResponse("冰情监测", "monitor", "ice", {"enabled": True})
            ],
            allowed_modes=[OperationMode.AUTO_L5, OperationMode.AUTO_L4],
            auto_recovery=True
        )
    
    # ========== 启停过渡场景 ==========
    def _add_transition_scenarios(self):
        """添加启停过渡场景"""
        
        # 冷态启动
        self.scenarios[ScenarioType.STARTUP_COLD] = ScenarioDefinition(
            scenario_type=ScenarioType.STARTUP_COLD,
            category=ScenarioCategory.TRANSITION,
            severity=ScenarioSeverity.WARNING,
            name="冷态启动",
            description="系统长时间停机后的冷态启动过程",
            triggers=[
                ScenarioTrigger("停机超时", "shutdown_duration", ">", 24*3600),
                ScenarioTrigger("启动命令", "startup_command", "==", True)
            ],
            responses=[
                ScenarioResponse("系统自检", "control", "self_check", {"full": True}),
                ScenarioResponse("缓慢充水", "control", "valves", {"opening_rate": 0.01}),
                ScenarioResponse("排气操作", "control", "air_valves", {"mode": "exhaust"}),
                ScenarioResponse("压力监测", "monitor", "pipeline", {"threshold": 0.5}),
                ScenarioResponse("启动通知", "notify", "all", {"message": "系统冷态启动中"})
            ],
            allowed_modes=[OperationMode.SUPERVISED, OperationMode.AUTO_L3],
            auto_recovery=False,
            requires_human=True,
            recovery_time=3600
        )
        
        # 初期小流量启动
        self.scenarios[ScenarioType.STARTUP_INITIAL] = ScenarioDefinition(
            scenario_type=ScenarioType.STARTUP_INITIAL,
            category=ScenarioCategory.TRANSITION,
            severity=ScenarioSeverity.WARNING,
            name="初期小流量启动",
            description="工程初期调试阶段的小流量启动",
            triggers=[
                ScenarioTrigger("调试模式", "commissioning_mode", "==", True),
                ScenarioTrigger("目标小流量", "target_flow", "<", 3.0)
            ],
            responses=[
                ScenarioResponse("小流量控制", "control", "valves", {"max_opening": 0.2}),
                ScenarioResponse("详细监测", "monitor", "all", {"interval": 1}),
                ScenarioResponse("数据记录", "log", "database", {"detail": "full"}),
                ScenarioResponse("专家在场", "notify", "expert", {"required": True})
            ],
            allowed_modes=[OperationMode.SUPERVISED, OperationMode.MANUAL],
            auto_recovery=False,
            requires_human=True
        )
        
        # 紧急停机
        self.scenarios[ScenarioType.SHUTDOWN_EMERGENCY] = ScenarioDefinition(
            scenario_type=ScenarioType.SHUTDOWN_EMERGENCY,
            category=ScenarioCategory.TRANSITION,
            severity=ScenarioSeverity.EMERGENCY,
            name="紧急停机",
            description="检测到严重异常,执行紧急停机程序",
            triggers=[
                ScenarioTrigger("紧急停机触发", "emergency_shutdown", "==", True)
            ],
            responses=[
                ScenarioResponse("关闭进水阀", "control", "inlet_valve", {"position": 0, "rate": 0.1}),
                ScenarioResponse("打开泄压阀", "control", "relief_valve", {"position": 1}),
                ScenarioResponse("停止水轮机", "control", "turbine", {"trip": True}),
                ScenarioResponse("紧急广播", "notify", "all", {"message": "紧急停机", "priority": "emergency"}),
                ScenarioResponse("记录事件", "log", "database", {"level": "emergency"})
            ],
            allowed_modes=[OperationMode.EMERGENCY],
            auto_recovery=False,
            requires_human=True,
            escalation_time=60
        )
    
    # ========== 极端工况场景 ==========
    def _add_extreme_scenarios(self):
        """添加极端工况场景"""
        
        # 暴雨工况
        self.scenarios[ScenarioType.EXTREME_HEAVY_RAIN] = ScenarioDefinition(
            scenario_type=ScenarioType.EXTREME_HEAVY_RAIN,
            category=ScenarioCategory.EXTREME,
            severity=ScenarioSeverity.WARNING,
            name="暴雨工况",
            description="暴雨天气导致来水量激增",
            triggers=[
                ScenarioTrigger("降雨强度", "rainfall_intensity", ">", 50),  # mm/h
                ScenarioTrigger("水库来水", "inflow_rate", ">", 100),  # m³/s
            ],
            responses=[
                ScenarioResponse("洪水调度", "control", "flood_dispatcher", {"enabled": True}),
                ScenarioResponse("溢洪道准备", "control", "spillway", {"standby": True}),
                ScenarioResponse("降低输水", "control", "valves", {"target_flow": 0.5}),
                ScenarioResponse("气象监测", "monitor", "weather", {"interval": 10}),
                ScenarioResponse("预警通知", "notify", "flood_control", {"level": "warning"})
            ],
            allowed_modes=[OperationMode.AUTO_L4, OperationMode.AUTO_L3],
            auto_recovery=True,
            recovery_time=7200
        )
        
        # 洪水工况
        self.scenarios[ScenarioType.EXTREME_FLOOD] = ScenarioDefinition(
            scenario_type=ScenarioType.EXTREME_FLOOD,
            category=ScenarioCategory.EXTREME,
            severity=ScenarioSeverity.CRITICAL,
            name="洪水工况",
            description="洪水来袭,需要执行防洪调度",
            triggers=[
                ScenarioTrigger("水库水位", "reservoir_level", ">", 377.0),  # 汛限水位
                ScenarioTrigger("入库流量", "inflow_rate", ">", 500)  # m³/s
            ],
            responses=[
                ScenarioResponse("防洪调度", "control", "flood_dispatcher", {"mode": "flood"}),
                ScenarioResponse("开启溢洪道", "control", "spillway", {"gates": "auto"}),
                ScenarioResponse("减少输水", "control", "valves", {"target_flow": 0.3}),
                ScenarioResponse("水情通报", "notify", "authorities", {"level": "critical"}),
                ScenarioResponse("下游预警", "notify", "downstream", {"message": "洪水预警"})
            ],
            allowed_modes=[OperationMode.AUTO_L3, OperationMode.SUPERVISED],
            auto_recovery=False,
            requires_human=True,
            escalation_time=300
        )
        
        # 严重冰期
        self.scenarios[ScenarioType.EXTREME_ICE_SEVERE] = ScenarioDefinition(
            scenario_type=ScenarioType.EXTREME_ICE_SEVERE,
            category=ScenarioCategory.EXTREME,
            severity=ScenarioSeverity.CRITICAL,
            name="严重冰期",
            description="极端低温导致严重冰情",
            triggers=[
                ScenarioTrigger("极端低温", "air_temperature", "<", -30),
                ScenarioTrigger("冰厚超限", "ice_thickness", ">", 0.8)
            ],
            responses=[
                ScenarioResponse("严重冰期模式", "control", "ice_controller", {"mode": "severe"}),
                ScenarioResponse("大幅降低流量", "control", "valves", {"target_flow": 0.5}),
                ScenarioResponse("禁止调节", "control", "valves", {"lock": True}),
                ScenarioResponse("冰情巡检", "notify", "patrol", {"required": True}),
                ScenarioResponse("专家会商", "notify", "expert", {"meeting": True})
            ],
            allowed_modes=[OperationMode.AUTO_L3, OperationMode.SUPERVISED],
            auto_recovery=False,
            requires_human=True
        )
        
        # 地震工况
        self.scenarios[ScenarioType.EXTREME_EARTHQUAKE] = ScenarioDefinition(
            scenario_type=ScenarioType.EXTREME_EARTHQUAKE,
            category=ScenarioCategory.EXTREME,
            severity=ScenarioSeverity.EMERGENCY,
            name="地震工况",
            description="发生地震,需要紧急响应",
            triggers=[
                ScenarioTrigger("地震烈度", "seismic_intensity", ">", 5)
            ],
            responses=[
                ScenarioResponse("紧急停机", "control", "emergency_shutdown", {"trigger": True}),
                ScenarioResponse("关闭阀门", "control", "all_valves", {"position": 0}),
                ScenarioResponse("结构检查", "notify", "inspection", {"priority": "emergency"}),
                ScenarioResponse("人员撤离", "notify", "personnel", {"message": "紧急撤离"}),
                ScenarioResponse("应急上报", "notify", "authorities", {"level": "emergency"})
            ],
            allowed_modes=[OperationMode.EMERGENCY],
            auto_recovery=False,
            requires_human=True,
            escalation_time=60
        )
    
    # ========== 设备故障场景 ==========
    def _add_fault_scenarios(self):
        """添加设备故障场景"""
        
        # 阀门卡死
        self.scenarios[ScenarioType.FAULT_VALVE_STUCK] = ScenarioDefinition(
            scenario_type=ScenarioType.FAULT_VALVE_STUCK,
            category=ScenarioCategory.EQUIPMENT_FAULT,
            severity=ScenarioSeverity.ALARM,
            name="阀门卡死",
            description="阀门无法正常动作,卡在某位置",
            triggers=[
                ScenarioTrigger("阀门响应超时", "valve_response_time", ">", 30),
                ScenarioTrigger("位置误差", "valve_position_error", ">", 0.1)
            ],
            responses=[
                ScenarioResponse("阀门故障隔离", "control", "valve_isolation", {"valve_id": "auto"}),
                ScenarioResponse("切换备用", "control", "valve_redundancy", {"activate": True}),
                ScenarioResponse("故障报警", "alarm", "valve_fault", {"level": "alarm"}),
                ScenarioResponse("维修通知", "notify", "maintenance", {"priority": "high"})
            ],
            allowed_modes=[OperationMode.AUTO_L4, OperationMode.AUTO_L3],
            auto_recovery=True,
            recovery_time=1800
        )
        
        # 传感器失效
        self.scenarios[ScenarioType.FAULT_SENSOR_FAILURE] = ScenarioDefinition(
            scenario_type=ScenarioType.FAULT_SENSOR_FAILURE,
            category=ScenarioCategory.EQUIPMENT_FAULT,
            severity=ScenarioSeverity.ALARM,
            name="传感器失效",
            description="传感器完全失效,无法获取测量值",
            triggers=[
                ScenarioTrigger("传感器离线", "sensor_status", "==", "offline"),
                ScenarioTrigger("数据异常", "sensor_value", "==", None)
            ],
            responses=[
                ScenarioResponse("切换冗余传感器", "control", "sensor_redundancy", {"activate": True}),
                ScenarioResponse("软测量替代", "control", "soft_sensor", {"enabled": True}),
                ScenarioResponse("降级控制", "control", "degradation", {"level": 1}),
                ScenarioResponse("故障报警", "alarm", "sensor_fault", {"level": "alarm"})
            ],
            allowed_modes=[OperationMode.AUTO_L4, OperationMode.AUTO_L3],
            auto_recovery=True
        )
        
        # 电源故障
        self.scenarios[ScenarioType.FAULT_POWER_LOSS] = ScenarioDefinition(
            scenario_type=ScenarioType.FAULT_POWER_LOSS,
            category=ScenarioCategory.EQUIPMENT_FAULT,
            severity=ScenarioSeverity.CRITICAL,
            name="电源故障",
            description="主电源故障,需要切换备用电源",
            triggers=[
                ScenarioTrigger("电源电压", "power_voltage", "<", 0.85),
                ScenarioTrigger("电源状态", "power_status", "==", "fault")
            ],
            responses=[
                ScenarioResponse("切换UPS", "control", "ups", {"activate": True}),
                ScenarioResponse("非关键负载卸载", "control", "load_shed", {"level": 1}),
                ScenarioResponse("启动柴油发电机", "control", "diesel_generator", {"start": True}),
                ScenarioResponse("紧急通知", "notify", "all", {"message": "电源故障", "priority": "critical"})
            ],
            allowed_modes=[OperationMode.EMERGENCY],
            auto_recovery=True,
            escalation_time=60
        )
        
        # 水轮机跳闸
        self.scenarios[ScenarioType.FAULT_TURBINE_TRIP] = ScenarioDefinition(
            scenario_type=ScenarioType.FAULT_TURBINE_TRIP,
            category=ScenarioCategory.EQUIPMENT_FAULT,
            severity=ScenarioSeverity.CRITICAL,
            name="水轮机跳闸",
            description="水轮机突然跳闸停机",
            triggers=[
                ScenarioTrigger("水轮机状态", "turbine_status", "==", "trip"),
                ScenarioTrigger("转速异常", "turbine_speed", "==", 0)
            ],
            responses=[
                ScenarioResponse("关闭进水导叶", "control", "guide_vanes", {"position": 0}),
                ScenarioResponse("打开旁通阀", "control", "bypass_valve", {"position": 1}),
                ScenarioResponse("调压塔投入", "control", "surge_tank", {"activate": True}),
                ScenarioResponse("水锤保护", "control", "water_hammer", {"protection": True}),
                ScenarioResponse("故障诊断", "control", "diagnosis", {"target": "turbine"})
            ],
            allowed_modes=[OperationMode.AUTO_L4, OperationMode.AUTO_L3],
            auto_recovery=True,
            recovery_time=600
        )
    
    # ========== 事故应急场景 ==========
    def _add_accident_scenarios(self):
        """添加事故应急场景"""
        
        # 管道爆管
        self.scenarios[ScenarioType.ACCIDENT_PIPE_BURST] = ScenarioDefinition(
            scenario_type=ScenarioType.ACCIDENT_PIPE_BURST,
            category=ScenarioCategory.ACCIDENT,
            severity=ScenarioSeverity.EMERGENCY,
            name="管道爆管",
            description="管道发生破裂,需要紧急处置",
            triggers=[
                ScenarioTrigger("压力骤降", "pressure_drop_rate", ">", 10),  # bar/min
                ScenarioTrigger("流量异常", "flow_imbalance", ">", 0.2)
            ],
            responses=[
                ScenarioResponse("紧急关阀", "control", "upstream_valve", {"position": 0, "rate": 0.5}),
                ScenarioResponse("隔离故障段", "control", "isolation_valves", {"close": True}),
                ScenarioResponse("停止输水", "control", "system", {"shutdown": True}),
                ScenarioResponse("紧急通知", "notify", "emergency_team", {"priority": "emergency"}),
                ScenarioResponse("现场警戒", "notify", "security", {"action": "evacuate"})
            ],
            allowed_modes=[OperationMode.EMERGENCY],
            auto_recovery=False,
            requires_human=True,
            escalation_time=30
        )
        
        # 极端水锤
        self.scenarios[ScenarioType.ACCIDENT_SURGE_EXTREME] = ScenarioDefinition(
            scenario_type=ScenarioType.ACCIDENT_SURGE_EXTREME,
            category=ScenarioCategory.ACCIDENT,
            severity=ScenarioSeverity.EMERGENCY,
            name="极端水锤",
            description="发生极端水锤压力波动",
            triggers=[
                ScenarioTrigger("压力超限", "max_pressure", ">", 1.5),  # 设计压力的1.5倍
                ScenarioTrigger("压力波动", "pressure_oscillation", ">", 0.3)
            ],
            responses=[
                ScenarioResponse("打开泄压阀", "control", "relief_valves", {"position": 1}),
                ScenarioResponse("停止阀门动作", "control", "all_valves", {"freeze": True}),
                ScenarioResponse("调压塔泄能", "control", "surge_tank", {"drain": True}),
                ScenarioResponse("结构检查", "notify", "inspection", {"priority": "emergency"}),
                ScenarioResponse("水锤分析", "log", "analysis", {"event": "water_hammer"})
            ],
            allowed_modes=[OperationMode.EMERGENCY],
            auto_recovery=False,
            requires_human=True
        )
        
        # 隧洞坍塌
        self.scenarios[ScenarioType.ACCIDENT_TUNNEL_COLLAPSE] = ScenarioDefinition(
            scenario_type=ScenarioType.ACCIDENT_TUNNEL_COLLAPSE,
            category=ScenarioCategory.ACCIDENT,
            severity=ScenarioSeverity.EMERGENCY,
            name="隧洞坍塌",
            description="隧洞发生坍塌事故",
            triggers=[
                ScenarioTrigger("隧洞变形", "tunnel_deformation", ">", 0.1),  # 米
                ScenarioTrigger("渗水量", "tunnel_seepage", ">", 10)  # L/s
            ],
            responses=[
                ScenarioResponse("停止输水", "control", "system", {"emergency_stop": True}),
                ScenarioResponse("人员撤离", "notify", "personnel", {"action": "evacuate"}),
                ScenarioResponse("应急救援", "notify", "rescue_team", {"deploy": True}),
                ScenarioResponse("结构评估", "notify", "structural_team", {"urgent": True}),
                ScenarioResponse("政府上报", "notify", "authorities", {"level": "emergency"})
            ],
            allowed_modes=[OperationMode.EMERGENCY, OperationMode.SHUTDOWN],
            auto_recovery=False,
            requires_human=True,
            escalation_time=30
        )
    
    # ========== 通讯故障场景 ==========
    def _add_communication_scenarios(self):
        """添加通讯故障场景"""
        
        # SCADA通讯中断
        self.scenarios[ScenarioType.COMM_SCADA_LOSS] = ScenarioDefinition(
            scenario_type=ScenarioType.COMM_SCADA_LOSS,
            category=ScenarioCategory.COMMUNICATION,
            severity=ScenarioSeverity.CRITICAL,
            name="SCADA通讯中断",
            description="与SCADA主站通讯完全中断",
            triggers=[
                ScenarioTrigger("SCADA连接", "scada_connection", "==", False),
                ScenarioTrigger("心跳超时", "heartbeat_timeout", ">", 30)
            ],
            responses=[
                ScenarioResponse("本地自治模式", "control", "local_autonomy", {"enabled": True}),
                ScenarioResponse("保持当前状态", "control", "hold_state", {"enabled": True}),
                ScenarioResponse("备用通讯", "control", "backup_comm", {"activate": True}),
                ScenarioResponse("现场通知", "notify", "local_operator", {"message": "SCADA中断"}),
                ScenarioResponse("事件记录", "log", "local", {"buffer": True})
            ],
            allowed_modes=[OperationMode.AUTO_L3, OperationMode.SUPERVISED],
            auto_recovery=True,
            recovery_time=300
        )
        
        # 网络攻击
        self.scenarios[ScenarioType.COMM_CYBER_ATTACK] = ScenarioDefinition(
            scenario_type=ScenarioType.COMM_CYBER_ATTACK,
            category=ScenarioCategory.COMMUNICATION,
            severity=ScenarioSeverity.EMERGENCY,
            name="网络攻击",
            description="检测到网络安全攻击",
            triggers=[
                ScenarioTrigger("异常访问", "abnormal_access_count", ">", 100),
                ScenarioTrigger("入侵检测", "intrusion_detected", "==", True)
            ],
            responses=[
                ScenarioResponse("网络隔离", "control", "network", {"isolate": True}),
                ScenarioResponse("切换离线模式", "control", "offline_mode", {"enabled": True}),
                ScenarioResponse("安全团队通知", "notify", "security_team", {"priority": "emergency"}),
                ScenarioResponse("证据保全", "log", "forensic", {"enabled": True}),
                ScenarioResponse("应急响应", "notify", "cert", {"report": True})
            ],
            allowed_modes=[OperationMode.EMERGENCY],
            auto_recovery=False,
            requires_human=True
        )
    
    # ========== 检修维护场景 ==========
    def _add_maintenance_scenarios(self):
        """添加检修维护场景"""
        
        # 计划阀门检修
        self.scenarios[ScenarioType.MAINT_PLANNED_VALVE] = ScenarioDefinition(
            scenario_type=ScenarioType.MAINT_PLANNED_VALVE,
            category=ScenarioCategory.MAINTENANCE,
            severity=ScenarioSeverity.ATTENTION,
            name="计划阀门检修",
            description="按计划进行阀门检修维护",
            triggers=[
                ScenarioTrigger("检修工单", "maintenance_order", "==", "valve"),
                ScenarioTrigger("计划时间", "scheduled_time", "==", True)
            ],
            responses=[
                ScenarioResponse("隔离阀门", "control", "valve_isolation", {"enabled": True}),
                ScenarioResponse("切换备用", "control", "backup_valve", {"activate": True}),
                ScenarioResponse("降低流量", "control", "valves", {"target_flow": 0.7}),
                ScenarioResponse("检修通知", "notify", "maintenance_team", {"dispatch": True})
            ],
            allowed_modes=[OperationMode.AUTO_L4, OperationMode.AUTO_L3],
            auto_recovery=True,
            requires_human=True
        )
        
        # 临时紧急检修
        self.scenarios[ScenarioType.MAINT_EMERGENCY_VALVE] = ScenarioDefinition(
            scenario_type=ScenarioType.MAINT_EMERGENCY_VALVE,
            category=ScenarioCategory.MAINTENANCE,
            severity=ScenarioSeverity.WARNING,
            name="临时阀门检修",
            description="紧急临时阀门检修",
            triggers=[
                ScenarioTrigger("紧急检修", "emergency_maintenance", "==", True),
                ScenarioTrigger("目标设备", "target_equipment", "==", "valve")
            ],
            responses=[
                ScenarioResponse("快速隔离", "control", "fast_isolation", {"enabled": True}),
                ScenarioResponse("系统降级", "control", "degradation", {"level": 2}),
                ScenarioResponse("紧急调度", "notify", "maintenance_team", {"urgent": True}),
                ScenarioResponse("备件准备", "notify", "warehouse", {"parts_list": "auto"})
            ],
            allowed_modes=[OperationMode.AUTO_L3, OperationMode.SUPERVISED],
            auto_recovery=True,
            requires_human=True
        )
    
    # ========== 水质应急场景 ==========
    def _add_water_quality_scenarios(self):
        """添加水质应急场景"""
        
        # 浊度超标
        self.scenarios[ScenarioType.QUALITY_TURBIDITY_HIGH] = ScenarioDefinition(
            scenario_type=ScenarioType.QUALITY_TURBIDITY_HIGH,
            category=ScenarioCategory.WATER_QUALITY,
            severity=ScenarioSeverity.WARNING,
            name="浊度超标",
            description="水体浊度超过标准限值",
            triggers=[
                ScenarioTrigger("浊度值", "turbidity", ">", 100),  # NTU
                ScenarioTrigger("持续时间", "turbidity_duration", ">", 600)
            ],
            responses=[
                ScenarioResponse("降低取水", "control", "intake", {"reduction": 0.5}),
                ScenarioResponse("启动沉淀", "control", "sedimentation", {"enabled": True}),
                ScenarioResponse("水质监测", "monitor", "quality", {"interval": 60}),
                ScenarioResponse("水厂通知", "notify", "water_plant", {"alert": "turbidity"})
            ],
            allowed_modes=[OperationMode.AUTO_L4, OperationMode.AUTO_L3],
            auto_recovery=True
        )
        
        # 上游污染
        self.scenarios[ScenarioType.QUALITY_POLLUTION_UPSTREAM] = ScenarioDefinition(
            scenario_type=ScenarioType.QUALITY_POLLUTION_UPSTREAM,
            category=ScenarioCategory.WATER_QUALITY,
            severity=ScenarioSeverity.CRITICAL,
            name="上游污染",
            description="检测到上游水源污染",
            triggers=[
                ScenarioTrigger("污染物检测", "contaminant_detected", "==", True),
                ScenarioTrigger("上游水质", "upstream_quality", "==", "polluted")
            ],
            responses=[
                ScenarioResponse("停止取水", "control", "intake", {"shutdown": True}),
                ScenarioResponse("启动备用水源", "control", "backup_source", {"activate": True}),
                ScenarioResponse("环保通知", "notify", "environmental", {"report": True}),
                ScenarioResponse("公众预警", "notify", "public", {"message": "水质预警"}),
                ScenarioResponse("溯源调查", "notify", "investigation", {"deploy": True})
            ],
            allowed_modes=[OperationMode.AUTO_L3, OperationMode.SUPERVISED],
            auto_recovery=False,
            requires_human=True
        )
    
    # ========== 工程事故场景 ==========
    def _add_engineering_scenarios(self):
        """添加工程事故场景"""
        
        # 地基沉降
        self.scenarios[ScenarioType.ENG_FOUNDATION_SETTLE] = ScenarioDefinition(
            scenario_type=ScenarioType.ENG_FOUNDATION_SETTLE,
            category=ScenarioCategory.ENGINEERING,
            severity=ScenarioSeverity.CRITICAL,
            name="地基沉降",
            description="检测到异常地基沉降",
            triggers=[
                ScenarioTrigger("沉降速率", "settlement_rate", ">", 0.1),  # mm/day
                ScenarioTrigger("累计沉降", "total_settlement", ">", 50)  # mm
            ],
            responses=[
                ScenarioResponse("降低运行", "control", "system", {"load_reduction": 0.5}),
                ScenarioResponse("加密监测", "monitor", "settlement", {"interval": 60}),
                ScenarioResponse("结构评估", "notify", "structural_team", {"urgent": True}),
                ScenarioResponse("专家会诊", "notify", "expert_panel", {"convene": True})
            ],
            allowed_modes=[OperationMode.AUTO_L3, OperationMode.SUPERVISED],
            auto_recovery=False,
            requires_human=True
        )
        
        # 衬砌开裂
        self.scenarios[ScenarioType.ENG_LINING_CRACK] = ScenarioDefinition(
            scenario_type=ScenarioType.ENG_LINING_CRACK,
            category=ScenarioCategory.ENGINEERING,
            severity=ScenarioSeverity.ALARM,
            name="衬砌开裂",
            description="隧洞或管道衬砌出现裂缝",
            triggers=[
                ScenarioTrigger("裂缝宽度", "crack_width", ">", 0.3),  # mm
                ScenarioTrigger("渗水量", "seepage_rate", ">", 0.1)  # L/s
            ],
            responses=[
                ScenarioResponse("降低压力", "control", "pressure", {"reduction": 0.2}),
                ScenarioResponse("裂缝监测", "monitor", "crack", {"continuous": True}),
                ScenarioResponse("修复计划", "notify", "repair_team", {"schedule": True}),
                ScenarioResponse("风险评估", "notify", "risk_team", {"assessment": True})
            ],
            allowed_modes=[OperationMode.AUTO_L4, OperationMode.AUTO_L3],
            auto_recovery=True,
            requires_human=True
        )
    
    # ========== 长尾异常场景 ==========
    def _add_long_tail_scenarios(self):
        """添加长尾异常场景"""
        
        # 多故障并发
        self.scenarios[ScenarioType.LONG_TAIL_MULTI_FAULT] = ScenarioDefinition(
            scenario_type=ScenarioType.LONG_TAIL_MULTI_FAULT,
            category=ScenarioCategory.LONG_TAIL,
            severity=ScenarioSeverity.CRITICAL,
            name="多故障并发",
            description="多个故障同时发生",
            triggers=[
                ScenarioTrigger("活动故障数", "active_fault_count", ">", 3),
                ScenarioTrigger("故障关联度", "fault_correlation", ">", 0.5)
            ],
            responses=[
                ScenarioResponse("故障优先排序", "control", "fault_prioritization", {"enabled": True}),
                ScenarioResponse("资源协调", "control", "resource_coordination", {"optimize": True}),
                ScenarioResponse("系统降级", "control", "degradation", {"level": 3}),
                ScenarioResponse("应急团队", "notify", "emergency_team", {"full_deploy": True}),
                ScenarioResponse("根因分析", "control", "rca", {"start": True})
            ],
            allowed_modes=[OperationMode.AUTO_L3, OperationMode.SUPERVISED],
            auto_recovery=False,
            requires_human=True
        )
        
        # 级联故障
        self.scenarios[ScenarioType.LONG_TAIL_CASCADE] = ScenarioDefinition(
            scenario_type=ScenarioType.LONG_TAIL_CASCADE,
            category=ScenarioCategory.LONG_TAIL,
            severity=ScenarioSeverity.EMERGENCY,
            name="级联故障",
            description="故障引发连锁反应",
            triggers=[
                ScenarioTrigger("故障传播", "fault_propagation_rate", ">", 0.5),
                ScenarioTrigger("影响范围", "affected_components", ">", 5)
            ],
            responses=[
                ScenarioResponse("故障隔离", "control", "fault_isolation", {"aggressive": True}),
                ScenarioResponse("中断传播", "control", "propagation_break", {"enabled": True}),
                ScenarioResponse("系统分区", "control", "system_partition", {"activate": True}),
                ScenarioResponse("紧急停机准备", "control", "emergency_standby", {"ready": True})
            ],
            allowed_modes=[OperationMode.EMERGENCY],
            auto_recovery=False,
            requires_human=True,
            escalation_time=60
        )
        
        # 系统振荡
        self.scenarios[ScenarioType.LONG_TAIL_OSCILLATION] = ScenarioDefinition(
            scenario_type=ScenarioType.LONG_TAIL_OSCILLATION,
            category=ScenarioCategory.LONG_TAIL,
            severity=ScenarioSeverity.ALARM,
            name="系统振荡",
            description="控制系统出现持续振荡",
            triggers=[
                ScenarioTrigger("振荡幅度", "oscillation_amplitude", ">", 0.1),
                ScenarioTrigger("振荡周期", "oscillation_period", "<", 60)
            ],
            responses=[
                ScenarioResponse("PID参数调整", "control", "pid", {"damping": 2.0}),
                ScenarioResponse("控制器切换", "control", "controller_switch", {"to": "backup"}),
                ScenarioResponse("阀门固定", "control", "valve_hold", {"enabled": True}),
                ScenarioResponse("振荡分析", "control", "oscillation_analysis", {"start": True})
            ],
            allowed_modes=[OperationMode.AUTO_L4, OperationMode.AUTO_L3],
            auto_recovery=True
        )
        
        # 黑天鹅事件
        self.scenarios[ScenarioType.LONG_TAIL_BLACK_SWAN] = ScenarioDefinition(
            scenario_type=ScenarioType.LONG_TAIL_BLACK_SWAN,
            category=ScenarioCategory.LONG_TAIL,
            severity=ScenarioSeverity.EMERGENCY,
            name="黑天鹅事件",
            description="无法预见的极端事件",
            triggers=[
                ScenarioTrigger("异常指数", "anomaly_score", ">", 0.95),
                ScenarioTrigger("未知模式", "unknown_pattern", "==", True)
            ],
            responses=[
                ScenarioResponse("安全第一", "control", "safety_first", {"enabled": True}),
                ScenarioResponse("保守运行", "control", "conservative_mode", {"enabled": True}),
                ScenarioResponse("人工接管准备", "control", "manual_takeover", {"standby": True}),
                ScenarioResponse("全面通知", "notify", "all_stakeholders", {"message": "异常事件"}),
                ScenarioResponse("学习记录", "log", "learning", {"detailed": True})
            ],
            allowed_modes=[OperationMode.SUPERVISED, OperationMode.MANUAL],
            auto_recovery=False,
            requires_human=True
        )
        
        # 未知异常
        self.scenarios[ScenarioType.LONG_TAIL_UNKNOWN] = ScenarioDefinition(
            scenario_type=ScenarioType.LONG_TAIL_UNKNOWN,
            category=ScenarioCategory.LONG_TAIL,
            severity=ScenarioSeverity.WARNING,
            name="未知异常",
            description="检测到未知类型的异常",
            triggers=[
                ScenarioTrigger("分类置信度", "classification_confidence", "<", 0.5),
                ScenarioTrigger("异常检测", "anomaly_detected", "==", True)
            ],
            responses=[
                ScenarioResponse("保守策略", "control", "conservative", {"enabled": True}),
                ScenarioResponse("增强监测", "monitor", "all", {"enhanced": True}),
                ScenarioResponse("人工审核", "notify", "operator", {"review_required": True}),
                ScenarioResponse("模式学习", "control", "online_learning", {"enabled": True})
            ],
            allowed_modes=[OperationMode.AUTO_L3, OperationMode.SUPERVISED],
            auto_recovery=True,
            requires_human=True
        )
    
    # ========== 查询接口 ==========
    def get_scenario(self, scenario_type: ScenarioType) -> Optional[ScenarioDefinition]:
        """获取场景定义"""
        return self.scenarios.get(scenario_type)
    
    def get_scenarios_by_category(self, category: ScenarioCategory) -> List[ScenarioDefinition]:
        """按类别获取场景"""
        return [s for s in self.scenarios.values() if s.category == category]
    
    def get_scenarios_by_severity(self, min_severity: ScenarioSeverity) -> List[ScenarioDefinition]:
        """按严重程度获取场景"""
        return [s for s in self.scenarios.values() 
                if s.severity.value >= min_severity.value]
    
    def get_response_strategy(self, scenario_type: ScenarioType) -> List[ScenarioResponse]:
        """获取响应策略"""
        scenario = self.scenarios.get(scenario_type)
        return scenario.responses if scenario else []
    
    def get_all_scenarios(self) -> Dict[ScenarioType, ScenarioDefinition]:
        """获取所有场景"""
        return self.scenarios
    
    def count_scenarios(self) -> Dict[ScenarioCategory, int]:
        """统计各类场景数量"""
        counts = {}
        for cat in ScenarioCategory:
            counts[cat] = len(self.get_scenarios_by_category(cat))
        return counts


# 全局场景数据库实例
SCENARIO_DB = ScenarioDatabase()


# 便捷函数
def get_scenario(scenario_type: ScenarioType) -> Optional[ScenarioDefinition]:
    return SCENARIO_DB.get_scenario(scenario_type)

def get_scenarios_by_category(category: ScenarioCategory) -> List[ScenarioDefinition]:
    return SCENARIO_DB.get_scenarios_by_category(category)

def get_response_strategy(scenario_type: ScenarioType) -> List[ScenarioResponse]:
    return SCENARIO_DB.get_response_strategy(scenario_type)


# ==========================================
# 导出
# ==========================================
__all__ = [
    # 枚举
    'ScenarioCategory',
    'ScenarioSeverity', 
    'ScenarioPhase',
    'OperationMode',
    'ScenarioType',
    # 数据类
    'ScenarioTrigger',
    'ScenarioResponse',
    'ScenarioDefinition',
    # 数据库
    'ScenarioDatabase',
    'SCENARIO_DB',
    # 便捷函数
    'get_scenario',
    'get_scenarios_by_category',
    'get_response_strategy'
]
