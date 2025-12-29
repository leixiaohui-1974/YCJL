"""
场景注入器
==========

用于测试的场景注入:
- 需水激增
- 管道破裂
- 冰期条件
- 电力故障
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum, auto

from ..config.settings import ScenarioType


@dataclass
class InjectionEvent:
    """注入事件"""
    time: float              # 触发时间
    scenario: ScenarioType   # 场景类型
    duration: float          # 持续时间
    severity: float          # 严重程度 (0~1)
    parameters: Dict = field(default_factory=dict)


class ScenarioInjector:
    """
    场景注入器

    模拟各种运行场景
    """

    def __init__(self):
        self.events: List[InjectionEvent] = []
        self.active_events: List[InjectionEvent] = []
        self.current_time = 0.0

        # 场景修改器
        self.modifiers: Dict[ScenarioType, Callable] = {
            ScenarioType.NORMAL: self._apply_normal,
            ScenarioType.DEMAND_SURGE: self._apply_demand_surge,
            ScenarioType.PIPE_BURST: self._apply_pipe_burst,
            ScenarioType.ICE_PERIOD: self._apply_ice_period,
            ScenarioType.POWER_FAILURE: self._apply_power_failure
        }

    def schedule_event(self, event: InjectionEvent):
        """调度事件"""
        self.events.append(event)
        self.events.sort(key=lambda e: e.time)

    def schedule_demand_surge(self, time: float, duration: float = 3600,
                               surge_factor: float = 1.5):
        """调度需水激增"""
        self.schedule_event(InjectionEvent(
            time=time,
            scenario=ScenarioType.DEMAND_SURGE,
            duration=duration,
            severity=min(surge_factor / 2.0, 1.0),
            parameters={'surge_factor': surge_factor}
        ))

    def schedule_pipe_burst(self, time: float, location: float = 0.5,
                            leak_rate: float = 0.1):
        """调度管道破裂"""
        self.schedule_event(InjectionEvent(
            time=time,
            scenario=ScenarioType.PIPE_BURST,
            duration=float('inf'),  # 需要手动清除
            severity=leak_rate,
            parameters={'location': location, 'leak_rate': leak_rate}
        ))

    def schedule_ice_period(self, time: float, duration: float = 86400,
                            temperature: float = -5.0):
        """调度冰期"""
        self.schedule_event(InjectionEvent(
            time=time,
            scenario=ScenarioType.ICE_PERIOD,
            duration=duration,
            severity=abs(temperature) / 10.0,
            parameters={'temperature': temperature}
        ))

    def schedule_power_failure(self, time: float, duration: float = 300):
        """调度电力故障"""
        self.schedule_event(InjectionEvent(
            time=time,
            scenario=ScenarioType.POWER_FAILURE,
            duration=duration,
            severity=1.0,
            parameters={}
        ))

    def step(self, current_time: float, state: Dict) -> Dict:
        """
        推进一步

        Parameters:
            current_time: 当前时间
            state: 系统状态

        Returns:
            修改后的状态
        """
        self.current_time = current_time
        modified_state = state.copy()

        # 检查新事件
        while self.events and self.events[0].time <= current_time:
            event = self.events.pop(0)
            self.active_events.append(event)

        # 清理过期事件
        self.active_events = [
            e for e in self.active_events
            if current_time < e.time + e.duration
        ]

        # 应用活跃事件
        for event in self.active_events:
            modifier = self.modifiers.get(event.scenario)
            if modifier:
                modified_state = modifier(event, modified_state, current_time)

        return modified_state

    def _apply_normal(self, event: InjectionEvent, state: Dict,
                      current_time: float) -> Dict:
        """应用正常场景"""
        return state

    def _apply_demand_surge(self, event: InjectionEvent, state: Dict,
                            current_time: float) -> Dict:
        """应用需水激增"""
        surge_factor = event.parameters.get('surge_factor', 1.5)

        # 计算渐进因子
        elapsed = current_time - event.time
        ramp_time = 60.0  # 1分钟渐进

        if elapsed < ramp_time:
            factor = 1.0 + (surge_factor - 1.0) * (elapsed / ramp_time)
        elif elapsed > event.duration - ramp_time:
            remaining = event.time + event.duration - current_time
            factor = 1.0 + (surge_factor - 1.0) * (remaining / ramp_time)
        else:
            factor = surge_factor

        state['demand'] = state.get('demand', 10.0) * factor
        state['demand_surge_active'] = True
        state['demand_factor'] = factor

        return state

    def _apply_pipe_burst(self, event: InjectionEvent, state: Dict,
                          current_time: float) -> Dict:
        """应用管道破裂"""
        location = event.parameters.get('location', 0.5)
        leak_rate = event.parameters.get('leak_rate', 0.1)

        state['leak_active'] = True
        state['leak_location'] = location
        state['leak_rate'] = leak_rate

        # 修改流量
        current_flow = state.get('pipe_flow', 10.0)
        state['pipe_flow'] = current_flow * (1 - leak_rate)

        # 修改压力
        current_pressure = state.get('pipe_pressure', 50.0)
        state['pipe_pressure'] = current_pressure * (1 - leak_rate * 0.5)

        return state

    def _apply_ice_period(self, event: InjectionEvent, state: Dict,
                          current_time: float) -> Dict:
        """应用冰期"""
        temperature = event.parameters.get('temperature', -5.0)

        state['is_ice_period'] = True
        state['water_temperature'] = temperature

        # 增加流阻
        friction_increase = 0.1 + 0.1 * abs(temperature) / 10.0
        state['friction_increase'] = friction_increase

        # 降低最大流量
        state['max_flow_factor'] = 0.8

        return state

    def _apply_power_failure(self, event: InjectionEvent, state: Dict,
                             current_time: float) -> Dict:
        """应用电力故障"""
        state['power_status'] = 'failure'
        state['pump_running'] = False

        # 流量骤降
        elapsed = current_time - event.time
        if elapsed < 10:  # 10秒内快速下降
            decay = np.exp(-elapsed / 3.0)
            state['pipe_flow'] = state.get('pipe_flow', 10.0) * decay

        return state

    def clear_event(self, scenario: ScenarioType):
        """清除特定场景事件"""
        self.active_events = [
            e for e in self.active_events
            if e.scenario != scenario
        ]

    def get_active_scenarios(self) -> List[ScenarioType]:
        """获取活跃场景列表"""
        return [e.scenario for e in self.active_events]

    def is_scenario_active(self, scenario: ScenarioType) -> bool:
        """检查场景是否活跃"""
        return scenario in self.get_active_scenarios()

    def reset(self):
        """重置"""
        self.events.clear()
        self.active_events.clear()
        self.current_time = 0.0
