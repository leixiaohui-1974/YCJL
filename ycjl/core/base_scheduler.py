"""
调度器基类 (Base Scheduler)
===========================

提供水利工程调度决策的抽象基类，支持：
- 多种调度模式
- 约束管理
- 优化目标
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum, auto
from datetime import datetime


class ScheduleMode(Enum):
    """调度模式"""
    NORMAL = auto()             # 正常运行
    EMERGENCY = auto()          # 应急响应
    MAINTENANCE = auto()        # 检修模式
    FLOOD_CONTROL = auto()      # 防洪调度
    POWER_GENERATION = auto()   # 发电调度
    ECOLOGICAL = auto()         # 生态调度
    ICE_PERIOD = auto()         # 冰期运行
    STARTUP = auto()            # 启动模式
    SHUTDOWN = auto()           # 停机模式


class OperationZone(Enum):
    """运行分区"""
    UPPER = "弃水区"            # 需泄洪
    NORMAL = "正常区"           # 正常供水
    LOWER = "限制区"            # 限制供水
    DEAD = "死库容区"           # 停止供水
    FLOOD_CONTROL = "防洪区"    # 防洪调度


@dataclass
class ScheduleConstraint:
    """调度约束"""
    name: str                                   # 约束名称
    constraint_type: str                        # 约束类型 (equality, inequality)
    value: float                                # 约束值
    unit: str = ""                              # 单位
    is_hard: bool = True                        # 是否硬约束
    priority: int = 1                           # 优先级 (1最高)
    margin: float = 0.0                         # 裕度


@dataclass
class ScheduleDecision:
    """调度决策"""
    timestamp: datetime                         # 决策时间戳
    mode: ScheduleMode                          # 调度模式
    zone: OperationZone                         # 运行分区

    # 流量决策
    target_flow: float = 0.0                    # 目标流量 (m³/s)
    flow_range: Tuple[float, float] = (0.0, 0.0)  # 流量范围

    # 水位决策
    target_level: float = 0.0                   # 目标水位 (m)
    level_range: Tuple[float, float] = (0.0, 0.0)  # 水位范围

    # 设备控制
    equipment_commands: Dict[str, Any] = field(default_factory=dict)

    # 时间安排
    start_time: Optional[datetime] = None       # 开始时间
    duration_hours: float = 0.0                 # 持续时间 (小时)

    # 备注和风险
    notes: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    confidence: float = 1.0                     # 置信度 (0-1)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "mode": self.mode.name,
            "zone": self.zone.value,
            "target_flow": self.target_flow,
            "target_level": self.target_level,
            "equipment_commands": self.equipment_commands,
            "duration_hours": self.duration_hours,
            "confidence": self.confidence
        }


class BaseScheduler(ABC):
    """
    调度器抽象基类

    提供调度决策的通用框架
    """

    def __init__(self, name: str = "Scheduler"):
        self.name = name
        self.current_mode = ScheduleMode.NORMAL
        self.current_zone = OperationZone.NORMAL
        self.constraints: List[ScheduleConstraint] = []
        self._history: List[ScheduleDecision] = []

    def add_constraint(self, constraint: ScheduleConstraint):
        """添加约束"""
        self.constraints.append(constraint)

    def remove_constraint(self, name: str):
        """移除约束"""
        self.constraints = [c for c in self.constraints if c.name != name]

    def clear_constraints(self):
        """清除所有约束"""
        self.constraints.clear()

    @abstractmethod
    def determine_zone(self, **kwargs) -> OperationZone:
        """
        确定运行分区

        Returns:
            运行分区
        """
        pass

    @abstractmethod
    def determine_mode(self, **kwargs) -> ScheduleMode:
        """
        确定调度模式

        Returns:
            调度模式
        """
        pass

    @abstractmethod
    def generate_schedule(self, **kwargs) -> ScheduleDecision:
        """
        生成调度决策

        Returns:
            调度决策
        """
        pass

    def validate_decision(self, decision: ScheduleDecision) -> Tuple[bool, List[str]]:
        """
        验证决策是否满足约束

        Args:
            decision: 调度决策

        Returns:
            (是否有效, 违反的约束列表)
        """
        violations = []

        for constraint in self.constraints:
            if constraint.name.startswith("flow"):
                if decision.target_flow < constraint.value - constraint.margin:
                    if constraint.is_hard:
                        violations.append(f"违反约束 {constraint.name}: "
                                          f"{decision.target_flow} < {constraint.value}")
            elif constraint.name.startswith("level"):
                if decision.target_level < constraint.value - constraint.margin:
                    if constraint.is_hard:
                        violations.append(f"违反约束 {constraint.name}: "
                                          f"{decision.target_level} < {constraint.value}")

        return len(violations) == 0, violations

    def record_decision(self, decision: ScheduleDecision):
        """记录决策到历史"""
        self._history.append(decision)

    def get_history(self, limit: int = 100) -> List[ScheduleDecision]:
        """获取历史决策"""
        return self._history[-limit:]

    def print_schedule(self, decision: ScheduleDecision):
        """打印调度决策"""
        lines = [
            "=" * 60,
            "调度决策",
            "=" * 60,
            f"时间戳: {decision.timestamp}",
            f"模式: {decision.mode.name}",
            f"分区: {decision.zone.value}",
            f"目标流量: {decision.target_flow:.2f} m³/s",
            f"目标水位: {decision.target_level:.2f} m",
            f"置信度: {decision.confidence:.1%}"
        ]

        if decision.equipment_commands:
            lines.append("\n设备控制:")
            for equip, cmd in decision.equipment_commands.items():
                lines.append(f"  - {equip}: {cmd}")

        if decision.notes:
            lines.append("\n备注:")
            for note in decision.notes:
                lines.append(f"  • {note}")

        if decision.risks:
            lines.append("\n风险:")
            for risk in decision.risks:
                lines.append(f"  ⚠️ {risk}")

        lines.append("=" * 60)
        print("\n".join(lines))


__all__ = [
    'ScheduleMode',
    'OperationZone',
    'ScheduleConstraint',
    'ScheduleDecision',
    'BaseScheduler'
]
