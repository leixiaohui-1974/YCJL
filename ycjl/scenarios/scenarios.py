"""
场景处理器
==========

各类场景的专用处理逻辑:
- 正常运行
- 需水激增
- 管道破裂
- 冰期运行
- 电力故障
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto

from ..config.settings import ScenarioType, Config


@dataclass
class ControlRecommendation:
    """控制建议"""
    actuator_id: str
    action: str          # 'set', 'increment', 'rate'
    value: float
    priority: int
    reason: str


@dataclass
class ScenarioResponse:
    """场景响应"""
    scenario: ScenarioType
    control_recommendations: List[ControlRecommendation]
    setpoint_adjustments: Dict[str, float]
    mode_changes: Dict[str, str]
    notifications: List[str]


class ScenarioHandler(ABC):
    """场景处理器基类"""

    def __init__(self):
        self.cfg = Config
        self.active = False
        self.start_time = 0.0
        self.duration = 0.0

    @abstractmethod
    def get_scenario_type(self) -> ScenarioType:
        """获取场景类型"""
        pass

    @abstractmethod
    def evaluate_entry_conditions(self, state: Dict) -> Tuple[bool, float]:
        """
        评估进入条件

        Returns:
            (是否满足, 置信度)
        """
        pass

    @abstractmethod
    def evaluate_exit_conditions(self, state: Dict) -> Tuple[bool, float]:
        """
        评估退出条件

        Returns:
            (是否满足, 置信度)
        """
        pass

    @abstractmethod
    def generate_response(self, state: Dict) -> ScenarioResponse:
        """
        生成场景响应

        Returns:
            ScenarioResponse: 控制响应
        """
        pass

    def activate(self, timestamp: float):
        """激活场景"""
        self.active = True
        self.start_time = timestamp

    def deactivate(self):
        """停用场景"""
        self.active = False
        self.duration = 0.0

    def update_duration(self, current_time: float):
        """更新持续时间"""
        if self.active:
            self.duration = current_time - self.start_time


class NormalScenarioHandler(ScenarioHandler):
    """正常运行场景处理器"""

    def get_scenario_type(self) -> ScenarioType:
        return ScenarioType.NORMAL

    def evaluate_entry_conditions(self, state: Dict) -> Tuple[bool, float]:
        """正常场景的进入条件"""
        conditions = []

        # 水位在正常范围
        pool_level = state.get('pool_level', 5.0)
        level_ok = self.cfg.pool.warning_low < pool_level < self.cfg.pool.warning_high
        conditions.append(level_ok)

        # 压力在正常范围
        pressure = state.get('pipe_pressure', 50.0)
        pressure_ok = 10 < pressure < self.cfg.pipeline.max_pressure * 0.9
        conditions.append(pressure_ok)

        # 流量稳定
        flow_std = state.get('flow_std', 0.5)
        flow_stable = flow_std < 1.0
        conditions.append(flow_stable)

        satisfied = all(conditions)
        confidence = sum(conditions) / len(conditions)

        return satisfied, confidence

    def evaluate_exit_conditions(self, state: Dict) -> Tuple[bool, float]:
        """正常场景的退出条件 (任一异常触发退出)"""
        # 与进入条件相反
        entry_ok, entry_conf = self.evaluate_entry_conditions(state)
        return not entry_ok, 1.0 - entry_conf

    def generate_response(self, state: Dict) -> ScenarioResponse:
        """正常场景响应"""
        return ScenarioResponse(
            scenario=ScenarioType.NORMAL,
            control_recommendations=[],
            setpoint_adjustments={},
            mode_changes={'operation_mode': 'normal'},
            notifications=["系统正常运行"]
        )


class DemandSurgeHandler(ScenarioHandler):
    """需水激增场景处理器"""

    def __init__(self):
        super().__init__()
        self.surge_threshold = 1.5  # 流量增幅阈值

    def get_scenario_type(self) -> ScenarioType:
        return ScenarioType.DEMAND_SURGE

    def evaluate_entry_conditions(self, state: Dict) -> Tuple[bool, float]:
        """需水激增进入条件"""
        conditions = []
        scores = []

        # 出流增加
        flow = state.get('pipe_flow', 10.0)
        flow_baseline = state.get('flow_baseline', 10.0)
        flow_increase = flow / max(flow_baseline, 1.0)
        if flow_increase > self.surge_threshold:
            conditions.append(True)
            scores.append(min(flow_increase - 1.0, 1.0))
        else:
            conditions.append(False)
            scores.append(0)

        # 水位下降趋势
        level_trend = state.get('pool_level_trend', 0)
        if level_trend < -0.05:
            conditions.append(True)
            scores.append(min(abs(level_trend) * 10, 1.0))
        else:
            conditions.append(False)
            scores.append(0)

        # 压力下降
        pressure_trend = state.get('pressure_trend', 0)
        if pressure_trend < -0.1:
            conditions.append(True)
            scores.append(0.5)
        else:
            conditions.append(False)
            scores.append(0)

        satisfied = sum(conditions) >= 2
        confidence = np.mean(scores) if scores else 0

        return satisfied, confidence

    def evaluate_exit_conditions(self, state: Dict) -> Tuple[bool, float]:
        """需水激增退出条件"""
        # 流量恢复正常
        flow = state.get('pipe_flow', 10.0)
        flow_baseline = state.get('flow_baseline', 10.0)
        flow_ratio = flow / max(flow_baseline, 1.0)

        # 水位稳定
        level_trend = state.get('pool_level_trend', 0)

        exit_ok = flow_ratio < 1.2 and abs(level_trend) < 0.02
        confidence = 1.0 - (flow_ratio - 1.0) if flow_ratio > 1.0 else 1.0

        return exit_ok, confidence

    def generate_response(self, state: Dict) -> ScenarioResponse:
        """需水激增响应"""
        recommendations = []

        # 增大进水闸开度
        current_intake = state.get('gate_intake_opening', 0.5)
        if current_intake < 0.9:
            recommendations.append(ControlRecommendation(
                actuator_id='gate_intake',
                action='increment',
                value=0.1,
                priority=1,
                reason="增大进水以补充需求"
            ))

        # 调整中间阀门
        recommendations.append(ControlRecommendation(
            actuator_id='valve_mid',
            action='increment',
            value=0.05,
            priority=2,
            reason="增大中间段流量"
        ))

        # 水位设定点调整
        setpoints = {
            'pool_level_setpoint': self.cfg.pool.design_level - 0.5  # 允许水位略低
        }

        return ScenarioResponse(
            scenario=ScenarioType.DEMAND_SURGE,
            control_recommendations=recommendations,
            setpoint_adjustments=setpoints,
            mode_changes={'operation_mode': 'demand_surge'},
            notifications=[
                "检测到需水激增",
                "正在增大进水流量",
                f"当前流量超基准{(state.get('pipe_flow', 10) / 10 - 1) * 100:.1f}%"
            ]
        )


class PipeBurstHandler(ScenarioHandler):
    """管道破裂场景处理器"""

    def __init__(self):
        super().__init__()
        self.isolation_stage = 0  # 隔离阶段

    def get_scenario_type(self) -> ScenarioType:
        return ScenarioType.PIPE_BURST

    def evaluate_entry_conditions(self, state: Dict) -> Tuple[bool, float]:
        """爆管进入条件"""
        conditions = []
        scores = []

        # 流量突变
        flow_change = abs(state.get('flow_rate_of_change', 0))
        if flow_change > 2.0:
            conditions.append(True)
            scores.append(min(flow_change / 5.0, 1.0))
        else:
            conditions.append(False)
            scores.append(0)

        # 压力骤降
        pressure_drop = -state.get('pressure_trend', 0)
        if pressure_drop > 0.5:
            conditions.append(True)
            scores.append(min(pressure_drop, 1.0))
        else:
            conditions.append(False)
            scores.append(0)

        # 流量不平衡
        imbalance = abs(state.get('flow_imbalance', 0))
        if imbalance > 2.0:
            conditions.append(True)
            scores.append(min(imbalance / 5.0, 1.0))
        else:
            conditions.append(False)
            scores.append(0)

        satisfied = sum(conditions) >= 2
        confidence = np.mean(scores) if scores else 0

        return satisfied, confidence

    def evaluate_exit_conditions(self, state: Dict) -> Tuple[bool, float]:
        """爆管退出条件 (需要人工确认)"""
        # 爆管场景通常需要人工确认修复后才能退出
        manual_clear = state.get('burst_cleared', False)
        return manual_clear, 1.0 if manual_clear else 0.0

    def generate_response(self, state: Dict) -> ScenarioResponse:
        """爆管响应"""
        recommendations = []

        # 根据隔离阶段生成不同响应
        if self.isolation_stage == 0:
            # 第一阶段: 减小流量
            recommendations.append(ControlRecommendation(
                actuator_id='gate_intake',
                action='set',
                value=0.3,
                priority=0,
                reason="紧急减小入流"
            ))
            recommendations.append(ControlRecommendation(
                actuator_id='valve_end',
                action='set',
                value=0.5,
                priority=0,
                reason="减小末端流量"
            ))

        elif self.isolation_stage == 1:
            # 第二阶段: 分段隔离
            recommendations.append(ControlRecommendation(
                actuator_id='valve_section_0',
                action='set',
                value=0.0,
                priority=0,
                reason="隔离第一段"
            ))

        # 打开泄压阀
        recommendations.append(ControlRecommendation(
            actuator_id='valve_relief',
            action='set',
            value=1.0,
            priority=0,
            reason="释放管道压力"
        ))

        return ScenarioResponse(
            scenario=ScenarioType.PIPE_BURST,
            control_recommendations=recommendations,
            setpoint_adjustments={},
            mode_changes={
                'operation_mode': 'emergency',
                'control_mode': 'manual_override'
            },
            notifications=[
                "【紧急】检测到疑似管道破裂！",
                "已启动应急隔离程序",
                "请立即派遣巡检人员",
                f"当前隔离阶段: {self.isolation_stage + 1}"
            ]
        )


class IcePeriodHandler(ScenarioHandler):
    """冰期运行场景处理器"""

    def __init__(self):
        super().__init__()
        self.ice_severity = 0.0  # 冰情严重程度

    def get_scenario_type(self) -> ScenarioType:
        return ScenarioType.ICE_PERIOD

    def evaluate_entry_conditions(self, state: Dict) -> Tuple[bool, float]:
        """冰期进入条件"""
        conditions = []
        scores = []

        # 低温
        temp = state.get('water_temperature', 10.0)
        if temp < 4.0:
            conditions.append(True)
            scores.append(1.0 - temp / 4.0 if temp > 0 else 1.0)
        else:
            conditions.append(False)
            scores.append(0)

        # 流阻增大
        friction_increase = state.get('friction_increase', 0)
        if friction_increase > 0.1:
            conditions.append(True)
            scores.append(min(friction_increase, 1.0))
        else:
            conditions.append(False)
            scores.append(0)

        # 季节 (可选)
        is_winter = state.get('is_winter', False)
        if is_winter:
            conditions.append(True)
            scores.append(0.5)

        satisfied = temp < 4.0 and sum(conditions) >= 2
        confidence = np.mean(scores) if scores else 0

        return satisfied, confidence

    def evaluate_exit_conditions(self, state: Dict) -> Tuple[bool, float]:
        """冰期退出条件"""
        temp = state.get('water_temperature', 10.0)
        exit_ok = temp > 6.0
        confidence = min((temp - 4.0) / 2.0, 1.0) if temp > 4.0 else 0.0

        return exit_ok, confidence

    def generate_response(self, state: Dict) -> ScenarioResponse:
        """冰期响应"""
        recommendations = []

        # 降低流速
        current_flow = state.get('pipe_flow', 10.0)
        target_flow = current_flow * 0.8

        recommendations.append(ControlRecommendation(
            actuator_id='gate_intake',
            action='increment',
            value=-0.05,
            priority=1,
            reason="降低流速防止冰塞"
        ))

        # 调整曼宁系数
        setpoints = {
            'manning_n': 0.016,  # 增大糙率系数
            'max_velocity': 1.5   # 降低最大流速限制
        }

        return ScenarioResponse(
            scenario=ScenarioType.ICE_PERIOD,
            control_recommendations=recommendations,
            setpoint_adjustments=setpoints,
            mode_changes={
                'operation_mode': 'ice_period',
                'ice_protection': 'enabled'
            },
            notifications=[
                "已进入冰期运行模式",
                f"当前水温: {state.get('water_temperature', 0):.1f}°C",
                "流速已自动降低",
                "请加强巡检监控冰情"
            ]
        )


class PowerFailureHandler(ScenarioHandler):
    """电力故障场景处理器"""

    def __init__(self):
        super().__init__()
        self.backup_power_active = False

    def get_scenario_type(self) -> ScenarioType:
        return ScenarioType.POWER_FAILURE

    def evaluate_entry_conditions(self, state: Dict) -> Tuple[bool, float]:
        """电力故障进入条件"""
        conditions = []
        scores = []

        # 电力状态
        power_status = state.get('power_status', 'normal')
        if power_status == 'failure':
            conditions.append(True)
            scores.append(1.0)
        else:
            conditions.append(False)
            scores.append(0)

        # 流量骤降
        flow_drop = -state.get('flow_rate_of_change', 0)
        if flow_drop > 3.0:
            conditions.append(True)
            scores.append(min(flow_drop / 5.0, 1.0))
        else:
            conditions.append(False)
            scores.append(0)

        # 泵站状态
        pump_running = state.get('pump_running', True)
        if not pump_running:
            conditions.append(True)
            scores.append(1.0)
        else:
            conditions.append(False)
            scores.append(0)

        satisfied = sum(conditions) >= 2
        confidence = np.mean(scores) if scores else 0

        return satisfied, confidence

    def evaluate_exit_conditions(self, state: Dict) -> Tuple[bool, float]:
        """电力故障退出条件"""
        power_status = state.get('power_status', 'normal')
        pump_running = state.get('pump_running', False)

        exit_ok = power_status == 'normal' and pump_running
        confidence = 1.0 if exit_ok else 0.0

        return exit_ok, confidence

    def generate_response(self, state: Dict) -> ScenarioResponse:
        """电力故障响应"""
        recommendations = []

        # 切换备用电源
        if not self.backup_power_active:
            recommendations.append(ControlRecommendation(
                actuator_id='backup_power',
                action='set',
                value=1.0,
                priority=0,
                reason="启动备用电源"
            ))
            self.backup_power_active = True

        # 关闭非必要阀门
        recommendations.append(ControlRecommendation(
            actuator_id='valve_auxiliary',
            action='set',
            value=0.0,
            priority=1,
            reason="关闭辅助阀门节省电力"
        ))

        # 打开进气阀防止负压
        recommendations.append(ControlRecommendation(
            actuator_id='valve_air',
            action='set',
            value=1.0,
            priority=0,
            reason="防止管道负压"
        ))

        return ScenarioResponse(
            scenario=ScenarioType.POWER_FAILURE,
            control_recommendations=recommendations,
            setpoint_adjustments={},
            mode_changes={
                'operation_mode': 'emergency',
                'power_mode': 'backup',
                'control_mode': 'reduced'
            },
            notifications=[
                "【紧急】检测到电力故障！",
                "正在切换至备用电源",
                "非必要设备已关闭",
                "请联系电力部门"
            ]
        )


class ScenarioManager:
    """
    场景管理器

    管理所有场景处理器
    """

    def __init__(self):
        self.handlers: Dict[ScenarioType, ScenarioHandler] = {
            ScenarioType.NORMAL: NormalScenarioHandler(),
            ScenarioType.DEMAND_SURGE: DemandSurgeHandler(),
            ScenarioType.PIPE_BURST: PipeBurstHandler(),
            ScenarioType.ICE_PERIOD: IcePeriodHandler(),
            ScenarioType.POWER_FAILURE: PowerFailureHandler()
        }

        self.current_handler: Optional[ScenarioHandler] = self.handlers[ScenarioType.NORMAL]
        self.current_handler.activate(0)

    def evaluate_scenario(self, state: Dict) -> Tuple[ScenarioType, float]:
        """评估当前场景"""
        best_scenario = ScenarioType.NORMAL
        best_confidence = 0.0

        for scenario, handler in self.handlers.items():
            if scenario == ScenarioType.NORMAL:
                continue

            satisfied, confidence = handler.evaluate_entry_conditions(state)
            if satisfied and confidence > best_confidence:
                best_scenario = scenario
                best_confidence = confidence

        # 如果没有异常场景，检查正常场景
        if best_confidence < 0.5:
            normal_satisfied, normal_conf = self.handlers[ScenarioType.NORMAL].evaluate_entry_conditions(state)
            if normal_satisfied:
                best_scenario = ScenarioType.NORMAL
                best_confidence = normal_conf

        return best_scenario, best_confidence

    def transition_to(self, scenario: ScenarioType, timestamp: float):
        """转移到新场景"""
        if self.current_handler:
            self.current_handler.deactivate()

        self.current_handler = self.handlers.get(scenario)
        if self.current_handler:
            self.current_handler.activate(timestamp)

    def get_response(self, state: Dict) -> ScenarioResponse:
        """获取当前场景响应"""
        if self.current_handler:
            return self.current_handler.generate_response(state)

        return ScenarioResponse(
            scenario=ScenarioType.NORMAL,
            control_recommendations=[],
            setpoint_adjustments={},
            mode_changes={},
            notifications=[]
        )

    def get_current_scenario(self) -> ScenarioType:
        """获取当前场景"""
        if self.current_handler:
            return self.current_handler.get_scenario_type()
        return ScenarioType.NORMAL
