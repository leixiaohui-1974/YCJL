"""
L3战略层智能体
==============

多目标经济调度:
- 全局优化
- 多目标权衡
- 长时域规划
- 向L2分配目标
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import time
from scipy.optimize import minimize, differential_evolution

from .base_agent import (
    BaseAgent, AgentPriority, AgentState, AgentMessage,
    ControlAction, MessageType
)
from ..config.settings import Config, SeasonMode, ScenarioType


class OptimizationObjective(Enum):
    """优化目标"""
    ENERGY_COST = auto()         # 能耗成本
    WATER_DELIVERY = auto()      # 供水保障
    EQUIPMENT_LIFE = auto()      # 设备寿命
    STABILITY = auto()           # 运行稳定性
    EMERGENCY_RESPONSE = auto()  # 应急响应


@dataclass
class StrategicTarget:
    """战略目标"""
    segment_id: int
    target_flow: float
    target_level: float
    priority: float = 1.0
    deadline: float = float('inf')


@dataclass
class SchedulePoint:
    """调度时刻点"""
    time: float
    flows: Dict[int, float]      # 各段流量
    levels: Dict[int, float]     # 各点水位
    valves: Dict[str, float]     # 阀门开度
    cost: float                  # 预计成本


class StrategicAgent(BaseAgent):
    """
    L3战略层智能体

    负责全局优化和经济调度:
    - 24小时滚动优化
    - 峰谷电价利用
    - 多目标权衡
    - 向L2层分配目标
    """

    def __init__(self, agent_id: str = "L3_strategic"):
        super().__init__(agent_id, AgentPriority.STRATEGIC)

        self.cfg = Config

        # 管段数量
        self.num_segments = 4

        # 优化目标权重
        self.objective_weights = {
            OptimizationObjective.ENERGY_COST: 0.4,
            OptimizationObjective.WATER_DELIVERY: 0.3,
            OptimizationObjective.EQUIPMENT_LIFE: 0.15,
            OptimizationObjective.STABILITY: 0.15
        }

        # 当前场景
        self.current_scenario = ScenarioType.NORMAL
        self.season_mode = SeasonMode.NORMAL

        # L2层状态
        self.l2_states: Dict[int, Dict] = {}

        # 调度计划
        self.schedule: List[SchedulePoint] = []
        self.schedule_horizon = 24 * 3600  # 24小时

        # 电价曲线 (元/kWh)
        self.electricity_price = self._init_price_curve()

        # 需水预测
        self.demand_forecast = np.ones(24) * 10.0  # m³/s

        # 战略目标
        self.strategic_targets: Dict[int, StrategicTarget] = {}

        # 优化结果
        self.last_optimization_time = 0.0
        self.optimization_interval = 3600  # 1小时重优化

        # 注册消息处理器
        self.message_handlers[MessageType.COORDINATION] = self._handle_coordination
        self.message_handlers[MessageType.ALERT] = self._handle_alert

    def _init_price_curve(self) -> np.ndarray:
        """初始化电价曲线"""
        # 24小时电价 (示例: 峰谷电价)
        price = np.ones(24) * 0.5  # 基础电价

        # 峰时 (08-12, 18-22)
        price[8:12] = 0.8
        price[18:22] = 0.8

        # 谷时 (00-06)
        price[0:6] = 0.3

        return price

    def _handle_coordination(self, msg: AgentMessage):
        """处理来自L2的协调消息"""
        if msg.sender.startswith('L2'):
            segment_id = msg.payload.get('segment_id', 0)
            self.l2_states[segment_id] = {
                'u': msg.payload.get('u', [0, 0]),
                'x': msg.payload.get('x', [0, 0, 0, 0]),
                'cost': msg.payload.get('cost', 0)
            }

    def _handle_alert(self, msg: AgentMessage):
        """处理报警消息"""
        alert_type = msg.payload.get('type', '')
        severity = msg.payload.get('severity', 'info')

        if severity == 'critical':
            # 切换到应急模式
            self.current_scenario = ScenarioType.PIPE_BURST
            self.objective_weights[OptimizationObjective.EMERGENCY_RESPONSE] = 0.5
            self._normalize_weights()

    def _normalize_weights(self):
        """归一化权重"""
        total = sum(self.objective_weights.values())
        for key in self.objective_weights:
            self.objective_weights[key] /= total

    def perceive(self, system_state: Dict) -> Dict:
        """感知全局状态"""
        observations = {}

        # 提取全局信息
        observations['reservoir_level'] = system_state.get('reservoir_level', 100.0)
        observations['total_flow'] = system_state.get('total_flow', 10.0)
        observations['total_demand'] = system_state.get('demand', 10.0)

        # 各段状态汇总
        for i in range(self.num_segments):
            prefix = f"seg{i}_"
            observations[f'seg{i}_flow'] = system_state.get(f'{prefix}flow', 10.0)
            observations[f'seg{i}_level'] = system_state.get(f'{prefix}level', 5.0)

        # 当前时间
        observations['hour'] = system_state.get('hour', 12)
        observations['current_price'] = self.electricity_price[int(observations['hour']) % 24]

        # 场景信息
        if 'scenario' in system_state:
            self.current_scenario = system_state['scenario']
        if 'season' in system_state:
            self.season_mode = system_state['season']

        return observations

    def decide(self) -> List[ControlAction]:
        """战略决策"""
        current_time = time.time()

        # 判断是否需要重新优化
        if current_time - self.last_optimization_time > self.optimization_interval:
            self._run_optimization()
            self.last_optimization_time = current_time

        # 获取当前时刻的调度目标
        targets = self._get_current_targets()

        # 生成向L2的命令
        actions = []
        for segment_id, target in targets.items():
            # 发送目标给L2
            self.send_message(AgentMessage(
                msg_type=MessageType.COMMAND,
                sender=self.agent_id,
                receiver=f'L2_tactical_{segment_id}',
                priority=self.priority,
                timestamp=current_time,
                payload={
                    'target_flow': target.target_flow,
                    'target_level': target.target_level,
                    'priority': target.priority
                }
            ))

        return actions  # L3不直接产生控制动作

    def act(self, actions: List[ControlAction]) -> Dict:
        """返回战略层结果"""
        result = {
            'agent': self.agent_id,
            'scenario': self.current_scenario.name,
            'targets': {},
            'schedule_status': 'active' if self.schedule else 'empty'
        }

        for seg_id, target in self.strategic_targets.items():
            result['targets'][seg_id] = {
                'flow': target.target_flow,
                'level': target.target_level
            }

        return result

    def _run_optimization(self):
        """运行全局优化"""
        # 优化变量: 各段流量分配
        n_vars = self.num_segments

        # 目标函数
        def objective(x):
            cost = 0.0

            # 能耗成本
            hour = int(self.observations.get('hour', 12)) % 24
            price = self.electricity_price[hour]
            energy_cost = self._compute_energy_cost(x, price)
            cost += self.objective_weights[OptimizationObjective.ENERGY_COST] * energy_cost

            # 供水保障 (偏离需求惩罚)
            demand = self.observations.get('total_demand', 10.0)
            delivery_gap = abs(sum(x) - demand)
            cost += self.objective_weights[OptimizationObjective.WATER_DELIVERY] * delivery_gap * 100

            # 设备寿命 (开度变化惩罚)
            if hasattr(self, '_last_flows'):
                change_cost = np.sum(np.abs(x - self._last_flows))
                cost += self.objective_weights[OptimizationObjective.EQUIPMENT_LIFE] * change_cost * 10

            # 稳定性 (流量均匀性)
            stability_cost = np.std(x) * 10
            cost += self.objective_weights[OptimizationObjective.STABILITY] * stability_cost

            return cost

        # 约束
        bounds = [(0, 15)] * n_vars  # 每段最大流量

        # 约束: 总流量满足需求
        demand = self.observations.get('total_demand', 10.0)

        def demand_constraint(x):
            return sum(x) - demand * 0.95  # 至少满足95%需求

        constraints = [{'type': 'ineq', 'fun': demand_constraint}]

        # 初始猜测
        x0 = np.ones(n_vars) * demand / n_vars

        # 求解
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100}
            )

            if result.success:
                optimal_flows = result.x
                self._last_flows = optimal_flows.copy()

                # 生成战略目标
                for i in range(n_vars):
                    self.strategic_targets[i] = StrategicTarget(
                        segment_id=i,
                        target_flow=optimal_flows[i],
                        target_level=self._compute_target_level(i, optimal_flows[i]),
                        priority=1.0
                    )

                # 生成调度计划
                self._generate_schedule(optimal_flows)

        except Exception as e:
            self.state.error_count += 1

    def _compute_energy_cost(self, flows: np.ndarray, price: float) -> float:
        """计算能耗成本"""
        # 简化模型: 能耗与流量的三次方成正比 (泵功率)
        power = np.sum(flows ** 2) * 0.1  # kW
        energy = power * 1.0  # 1小时能耗 kWh
        return energy * price

    def _compute_target_level(self, segment_id: int, flow: float) -> float:
        """计算目标水位"""
        # 基于流量计算平衡水位
        base_level = 5.0
        level_adjustment = (flow - 10.0) * 0.1
        return base_level + level_adjustment

    def _generate_schedule(self, optimal_flows: np.ndarray):
        """生成调度计划"""
        self.schedule = []

        current_hour = int(self.observations.get('hour', 12)) % 24

        for h in range(24):
            hour = (current_hour + h) % 24
            price = self.electricity_price[hour]

            # 根据电价调整流量
            price_factor = 1.0 - (price - 0.5) * 0.3  # 电价高时减少流量

            flows = {}
            levels = {}
            for i in range(self.num_segments):
                flows[i] = optimal_flows[i] * price_factor
                levels[i] = self._compute_target_level(i, flows[i])

            cost = self._compute_energy_cost(np.array(list(flows.values())), price)

            self.schedule.append(SchedulePoint(
                time=h * 3600,
                flows=flows,
                levels=levels,
                valves={},
                cost=cost
            ))

    def _get_current_targets(self) -> Dict[int, StrategicTarget]:
        """获取当前时刻的目标"""
        return self.strategic_targets.copy()

    def set_demand_forecast(self, forecast: np.ndarray):
        """设置需水预测"""
        self.demand_forecast = forecast.copy()

    def set_scenario(self, scenario: ScenarioType):
        """设置当前场景"""
        self.current_scenario = scenario

        # 根据场景调整权重
        if scenario == ScenarioType.PIPE_BURST:
            self.objective_weights[OptimizationObjective.EMERGENCY_RESPONSE] = 0.5
            self.objective_weights[OptimizationObjective.ENERGY_COST] = 0.2
        elif scenario == ScenarioType.ICE_PERIOD:
            self.objective_weights[OptimizationObjective.STABILITY] = 0.3
            self.objective_weights[OptimizationObjective.EQUIPMENT_LIFE] = 0.2
        else:
            # 恢复默认权重
            self.objective_weights = {
                OptimizationObjective.ENERGY_COST: 0.4,
                OptimizationObjective.WATER_DELIVERY: 0.3,
                OptimizationObjective.EQUIPMENT_LIFE: 0.15,
                OptimizationObjective.STABILITY: 0.15
            }

        self._normalize_weights()

    def get_schedule(self) -> List[Dict]:
        """获取调度计划"""
        return [
            {
                'time': sp.time,
                'flows': sp.flows,
                'levels': sp.levels,
                'cost': sp.cost
            }
            for sp in self.schedule
        ]

    def get_optimization_summary(self) -> Dict:
        """获取优化摘要"""
        total_cost = sum(sp.cost for sp in self.schedule)
        avg_flow = np.mean([
            list(sp.flows.values()) for sp in self.schedule
        ]) if self.schedule else 0

        return {
            'total_24h_cost': total_cost,
            'average_flow': avg_flow,
            'scenario': self.current_scenario.name,
            'objective_weights': {k.name: v for k, v in self.objective_weights.items()},
            'num_segments': self.num_segments,
            'schedule_points': len(self.schedule)
        }

    def reset(self):
        """重置"""
        super().reset()
        self.l2_states.clear()
        self.schedule.clear()
        self.strategic_targets.clear()
        self.last_optimization_time = 0.0
        self.current_scenario = ScenarioType.NORMAL

        self.objective_weights = {
            OptimizationObjective.ENERGY_COST: 0.4,
            OptimizationObjective.WATER_DELIVERY: 0.3,
            OptimizationObjective.EQUIPMENT_LIFE: 0.15,
            OptimizationObjective.STABILITY: 0.15
        }
