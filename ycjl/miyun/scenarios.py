"""
密云水库调蓄工程场景库 (Scenario Database) v1.0
=============================================

基于引绰济辽工程场景库框架，适配密云水库调蓄工程特点

场景分类:
1. 正常运行场景
2. 需水变化场景
3. 泵站故障场景
4. 管道事故场景
5. 电力系统场景
6. 气象极端场景
7. 调度优化场景
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum, auto


class ScenarioType(Enum):
    """场景类型"""
    NORMAL = auto()             # 正常运行
    DEMAND_CHANGE = auto()      # 需水变化
    PUMP_FAULT = auto()         # 泵站故障
    PIPELINE_FAULT = auto()     # 管道故障
    POWER_SYSTEM = auto()       # 电力系统
    WEATHER = auto()            # 气象因素
    OPTIMIZATION = auto()       # 优化场景
    EMERGENCY = auto()          # 紧急场景
    MAINTENANCE = auto()        # 检修场景


class ScenarioSeverity(Enum):
    """场景严重程度"""
    INFO = 0                    # 信息
    MINOR = 1                   # 轻微
    MODERATE = 2                # 中等
    MAJOR = 3                   # 重大
    CRITICAL = 4                # 严重


class ResponsePriority(Enum):
    """响应优先级"""
    IMMEDIATE = 1               # 立即响应
    URGENT = 2                  # 紧急
    NORMAL = 3                  # 正常
    SCHEDULED = 4               # 计划


@dataclass
class ScenarioDefinition:
    """场景定义"""
    id: str                                     # 场景ID
    name: str                                   # 场景名称
    type: ScenarioType                          # 场景类型
    severity: ScenarioSeverity                  # 严重程度
    description: str                            # 描述
    triggers: List[str]                         # 触发条件
    responses: List[str]                        # 响应措施
    priority: ResponsePriority                  # 响应优先级
    affected_components: List[str] = field(default_factory=list)  # 受影响组件
    recovery_time_hours: float = 0.0            # 预估恢复时间


@dataclass
class ScenarioEvent:
    """场景事件"""
    timestamp: float                            # 时间戳
    scenario_id: str                            # 场景ID
    detected_values: Dict[str, float]           # 检测到的值
    severity: ScenarioSeverity                  # 当前严重程度
    is_active: bool = True                      # 是否活跃


# ==========================================
# 密云水库场景库
# ==========================================
MIYUN_SCENARIO_DATABASE: Dict[str, ScenarioDefinition] = {
    # ==========================
    # 1. 正常运行场景
    # ==========================
    "NOR-001": ScenarioDefinition(
        id="NOR-001",
        name="正常输水运行",
        type=ScenarioType.NORMAL,
        severity=ScenarioSeverity.INFO,
        description="系统在设计参数范围内正常运行",
        triggers=["所有参数在正常范围"],
        responses=["监控运行", "记录数据"],
        priority=ResponsePriority.NORMAL
    ),

    "NOR-002": ScenarioDefinition(
        id="NOR-002",
        name="季节性流量调整",
        type=ScenarioType.NORMAL,
        severity=ScenarioSeverity.INFO,
        description="根据月度用水需求调整输水流量",
        triggers=["月度需水变化", "调度计划更新"],
        responses=["调整泵站运行台数", "更新流量设定点"],
        priority=ResponsePriority.SCHEDULED,
        recovery_time_hours=0.5
    ),

    # ==========================
    # 2. 需水变化场景
    # ==========================
    "DEM-001": ScenarioDefinition(
        id="DEM-001",
        name="用水高峰",
        type=ScenarioType.DEMAND_CHANGE,
        severity=ScenarioSeverity.MINOR,
        description="夏季用水高峰期，需水量增加",
        triggers=["流量需求 > 18 m³/s", "季节=夏季"],
        responses=["增加运行泵数", "开启备用机组", "优化调度策略"],
        priority=ResponsePriority.NORMAL,
        recovery_time_hours=2.0
    ),

    "DEM-002": ScenarioDefinition(
        id="DEM-002",
        name="用水低谷",
        type=ScenarioType.DEMAND_CHANGE,
        severity=ScenarioSeverity.INFO,
        description="冬季用水低谷期，需水量减少",
        triggers=["流量需求 < 8 m³/s", "季节=冬季"],
        responses=["减少运行泵数", "轮换检修"],
        priority=ResponsePriority.NORMAL,
        recovery_time_hours=1.0
    ),

    "DEM-003": ScenarioDefinition(
        id="DEM-003",
        name="突发用水需求",
        type=ScenarioType.DEMAND_CHANGE,
        severity=ScenarioSeverity.MODERATE,
        description="突发事件导致用水需求激增",
        triggers=["流量需求突增 > 30%", "非计划调度请求"],
        responses=["紧急启动备用泵", "协调上下游", "评估系统容量"],
        priority=ResponsePriority.URGENT,
        affected_components=["全部泵站"],
        recovery_time_hours=4.0
    ),

    # ==========================
    # 3. 泵站故障场景
    # ==========================
    "PMP-001": ScenarioDefinition(
        id="PMP-001",
        name="单台泵故障停机",
        type=ScenarioType.PUMP_FAULT,
        severity=ScenarioSeverity.MODERATE,
        description="单台泵机组发生故障停机",
        triggers=["泵机组振动超标", "电机温度过高", "轴承异常"],
        responses=["自动切换备用泵", "隔离故障机组", "派遣检修人员"],
        priority=ResponsePriority.URGENT,
        affected_components=["故障泵站"],
        recovery_time_hours=8.0
    ),

    "PMP-002": ScenarioDefinition(
        id="PMP-002",
        name="泵站全停",
        type=ScenarioType.PUMP_FAULT,
        severity=ScenarioSeverity.CRITICAL,
        description="某泵站所有机组全部停机",
        triggers=["泵站失电", "多台泵同时故障", "控制系统故障"],
        responses=["启动应急预案", "调整上下游流量", "组织抢修"],
        priority=ResponsePriority.IMMEDIATE,
        affected_components=["故障泵站", "上下游泵站"],
        recovery_time_hours=24.0
    ),

    "PMP-003": ScenarioDefinition(
        id="PMP-003",
        name="高扬程泵站停泵水锤",
        type=ScenarioType.PUMP_FAULT,
        severity=ScenarioSeverity.MAJOR,
        description="雁栖或溪翁庄高扬程泵站突然停泵导致水锤",
        triggers=["雁栖/溪翁庄泵站停泵", "扬程>30m"],
        responses=["关闭进出口阀门", "启动水锤保护", "检查管道"],
        priority=ResponsePriority.IMMEDIATE,
        affected_components=["Yanqi", "Xiwengzhuang", "PCCP管道"],
        recovery_time_hours=12.0
    ),

    # ==========================
    # 4. 管道事故场景
    # ==========================
    "PIP-001": ScenarioDefinition(
        id="PIP-001",
        name="管道泄漏",
        type=ScenarioType.PIPELINE_FAULT,
        severity=ScenarioSeverity.MAJOR,
        description="PCCP管道发生泄漏",
        triggers=["流量不平衡", "压力异常下降", "泄漏探测报警"],
        responses=["定位泄漏点", "关闭相关阀门", "启动应急供水"],
        priority=ResponsePriority.IMMEDIATE,
        affected_components=["PCCP管道段"],
        recovery_time_hours=72.0
    ),

    "PIP-002": ScenarioDefinition(
        id="PIP-002",
        name="管道负压",
        type=ScenarioType.PIPELINE_FAULT,
        severity=ScenarioSeverity.CRITICAL,
        description="管道高点出现负压，可能导致管道塌陷",
        triggers=["高点压力<-3m", "空气阀动作异常"],
        responses=["降低流量", "开启进气阀", "停止相关泵站"],
        priority=ResponsePriority.IMMEDIATE,
        affected_components=["管道高点", "空气阀"],
        recovery_time_hours=4.0
    ),

    "PIP-003": ScenarioDefinition(
        id="PIP-003",
        name="阀门故障",
        type=ScenarioType.PIPELINE_FAULT,
        severity=ScenarioSeverity.MODERATE,
        description="调节阀门无法正常动作",
        triggers=["阀门位置反馈异常", "阀门动作时间超标"],
        responses=["手动操作", "启用备用阀门", "安排检修"],
        priority=ResponsePriority.URGENT,
        affected_components=["故障阀门"],
        recovery_time_hours=6.0
    ),

    # ==========================
    # 5. 电力系统场景
    # ==========================
    "PWR-001": ScenarioDefinition(
        id="PWR-001",
        name="单路电源故障",
        type=ScenarioType.POWER_SYSTEM,
        severity=ScenarioSeverity.MODERATE,
        description="泵站单路供电故障",
        triggers=["主电源失电", "备电切换"],
        responses=["自动切换备电", "检查电力系统", "联系供电部门"],
        priority=ResponsePriority.URGENT,
        recovery_time_hours=4.0
    ),

    "PWR-002": ScenarioDefinition(
        id="PWR-002",
        name="全站失电",
        type=ScenarioType.POWER_SYSTEM,
        severity=ScenarioSeverity.CRITICAL,
        description="泵站双路电源全部失电",
        triggers=["主备电均失电", "UPS告警"],
        responses=["启动柴油发电机", "保护性停机", "通知调度"],
        priority=ResponsePriority.IMMEDIATE,
        affected_components=["失电泵站"],
        recovery_time_hours=8.0
    ),

    "PWR-003": ScenarioDefinition(
        id="PWR-003",
        name="电力限制",
        type=ScenarioType.POWER_SYSTEM,
        severity=ScenarioSeverity.MINOR,
        description="电网要求限制用电负荷",
        triggers=["电网调度通知", "错峰用电要求"],
        responses=["降低运行台数", "错峰调度", "优化运行效率"],
        priority=ResponsePriority.NORMAL,
        recovery_time_hours=12.0
    ),

    # ==========================
    # 6. 气象场景
    # ==========================
    "WEA-001": ScenarioDefinition(
        id="WEA-001",
        name="暴雨预警",
        type=ScenarioType.WEATHER,
        severity=ScenarioSeverity.MINOR,
        description="降雨可能影响明渠段运行",
        triggers=["降雨量>50mm/d", "暴雨预警"],
        responses=["加强巡查", "准备排水设施", "调整调度"],
        priority=ResponsePriority.NORMAL,
        recovery_time_hours=24.0
    ),

    "WEA-002": ScenarioDefinition(
        id="WEA-002",
        name="冬季防冻",
        type=ScenarioType.WEATHER,
        severity=ScenarioSeverity.MODERATE,
        description="冬季低温可能导致明渠结冰",
        triggers=["气温<0°C", "冬季运行"],
        responses=["保持流速>0.6m/s", "启动防冻保护", "加强监测"],
        priority=ResponsePriority.NORMAL,
        affected_components=["明渠段"],
        recovery_time_hours=0.0
    ),

    "WEA-003": ScenarioDefinition(
        id="WEA-003",
        name="高温预警",
        type=ScenarioType.WEATHER,
        severity=ScenarioSeverity.MINOR,
        description="高温可能影响设备运行",
        triggers=["气温>35°C", "高温预警"],
        responses=["加强设备散热", "监控电机温度", "备用冷却"],
        priority=ResponsePriority.NORMAL,
        recovery_time_hours=12.0
    ),

    # ==========================
    # 7. 优化场景
    # ==========================
    "OPT-001": ScenarioDefinition(
        id="OPT-001",
        name="效率优化运行",
        type=ScenarioType.OPTIMIZATION,
        severity=ScenarioSeverity.INFO,
        description="系统处于可优化状态",
        triggers=["系统效率<85%", "能耗偏高"],
        responses=["优化泵组运行组合", "调整流量分配"],
        priority=ResponsePriority.SCHEDULED,
        recovery_time_hours=1.0
    ),

    "OPT-002": ScenarioDefinition(
        id="OPT-002",
        name="峰谷电价调度",
        type=ScenarioType.OPTIMIZATION,
        severity=ScenarioSeverity.INFO,
        description="利用峰谷电价差优化运行成本",
        triggers=["谷电时段", "有调节裕度"],
        responses=["谷电期增加输水", "峰电期减少输水"],
        priority=ResponsePriority.SCHEDULED,
        recovery_time_hours=0.0
    ),

    # ==========================
    # 8. 检修场景
    # ==========================
    "MNT-001": ScenarioDefinition(
        id="MNT-001",
        name="计划检修",
        type=ScenarioType.MAINTENANCE,
        severity=ScenarioSeverity.INFO,
        description="按计划进行设备检修",
        triggers=["检修计划触发", "定期维护"],
        responses=["隔离待检设备", "调整运行方式", "执行检修"],
        priority=ResponsePriority.SCHEDULED,
        recovery_time_hours=24.0
    ),

    "MNT-002": ScenarioDefinition(
        id="MNT-002",
        name="紧急检修",
        type=ScenarioType.MAINTENANCE,
        severity=ScenarioSeverity.MODERATE,
        description="发现隐患需紧急处理",
        triggers=["设备异常预警", "缺陷升级"],
        responses=["评估风险", "安排紧急检修", "临时调整运行"],
        priority=ResponsePriority.URGENT,
        recovery_time_hours=8.0
    ),
}


class MiyunScenarioDetector:
    """
    密云水库场景检测器

    实时监测系统状态，检测和识别场景
    """

    def __init__(self):
        self.database = MIYUN_SCENARIO_DATABASE
        self.active_scenarios: List[ScenarioEvent] = []

    def detect_scenarios(self, system_state: Dict) -> List[ScenarioEvent]:
        """
        检测当前系统状态对应的场景

        Args:
            system_state: 系统状态字典

        Returns:
            检测到的场景事件列表
        """
        events = []
        timestamp = system_state.get("timestamp", 0.0)

        # 检测需水变化场景
        flow = system_state.get("flow", 10.0)
        if flow > 18.0:
            events.append(ScenarioEvent(
                timestamp=timestamp,
                scenario_id="DEM-001",
                detected_values={"flow": flow},
                severity=ScenarioSeverity.MINOR
            ))
        elif flow < 6.0:
            events.append(ScenarioEvent(
                timestamp=timestamp,
                scenario_id="DEM-002",
                detected_values={"flow": flow},
                severity=ScenarioSeverity.INFO
            ))

        # 检测泵站故障
        pump_status = system_state.get("pump_status", {})
        for station, status in pump_status.items():
            if status.get("fault", False):
                events.append(ScenarioEvent(
                    timestamp=timestamp,
                    scenario_id="PMP-001",
                    detected_values={"station": station},
                    severity=ScenarioSeverity.MODERATE
                ))

        # 检测管道负压
        pressures = system_state.get("pressures", {})
        for point, pressure in pressures.items():
            if pressure < -3.0:
                events.append(ScenarioEvent(
                    timestamp=timestamp,
                    scenario_id="PIP-002",
                    detected_values={"point": point, "pressure": pressure},
                    severity=ScenarioSeverity.CRITICAL
                ))

        # 如果没有检测到任何问题，返回正常运行场景
        if not events:
            events.append(ScenarioEvent(
                timestamp=timestamp,
                scenario_id="NOR-001",
                detected_values={"flow": flow},
                severity=ScenarioSeverity.INFO
            ))

        self.active_scenarios = events
        return events

    def get_scenario_info(self, scenario_id: str) -> Optional[ScenarioDefinition]:
        """获取场景定义"""
        return self.database.get(scenario_id)

    def get_response_actions(self, scenario_id: str) -> List[str]:
        """获取场景响应措施"""
        scenario = self.database.get(scenario_id)
        if scenario:
            return scenario.responses
        return []

    def get_scenarios_by_type(self, scenario_type: ScenarioType) -> List[ScenarioDefinition]:
        """按类型获取场景"""
        return [s for s in self.database.values() if s.type == scenario_type]

    def get_scenarios_by_severity(self, min_severity: ScenarioSeverity) -> List[ScenarioDefinition]:
        """获取指定严重程度以上的场景"""
        return [s for s in self.database.values()
                if s.severity.value >= min_severity.value]


# ==========================================
# 模块级实例
# ==========================================
ScenarioDetector = MiyunScenarioDetector()


# ==========================================
# 便捷函数
# ==========================================
def get_all_scenarios() -> Dict[str, ScenarioDefinition]:
    """获取所有场景定义"""
    return MIYUN_SCENARIO_DATABASE


def get_scenario_count() -> int:
    """获取场景总数"""
    return len(MIYUN_SCENARIO_DATABASE)


def print_scenario_summary():
    """打印场景库摘要"""
    print(f"\n{'='*60}")
    print("密云水库调蓄工程场景库摘要")
    print(f"{'='*60}")
    print(f"总场景数: {len(MIYUN_SCENARIO_DATABASE)}")

    by_type = {}
    by_severity = {}
    for s in MIYUN_SCENARIO_DATABASE.values():
        by_type[s.type.name] = by_type.get(s.type.name, 0) + 1
        by_severity[s.severity.name] = by_severity.get(s.severity.name, 0) + 1

    print("\n按类型分布:")
    for t, c in sorted(by_type.items()):
        print(f"  - {t}: {c}")

    print("\n按严重程度分布:")
    for s, c in sorted(by_severity.items()):
        print(f"  - {s}: {c}")


# ==========================================
# 导出
# ==========================================
__all__ = [
    'ScenarioType',
    'ScenarioSeverity',
    'ResponsePriority',
    'ScenarioDefinition',
    'ScenarioEvent',
    'MIYUN_SCENARIO_DATABASE',
    'MiyunScenarioDetector',
    'ScenarioDetector',
    'get_all_scenarios',
    'get_scenario_count',
    'print_scenario_summary'
]
