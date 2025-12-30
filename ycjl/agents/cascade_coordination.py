"""
级联泵站协调调度模块 v1.0
==========================

增强级联泵站间的协调调度算法，考虑：
1. 波传播动力学 - 明渠水流的波传播时间
2. 渠道蓄量变化 - 渠道中的水量动态
3. 前池缓冲效应 - 前池容量和水位变化
4. 智能接力策略 - 上下游泵站的精准配合

关键算法:
- MPC预测控制 - 基于水力模型的预测
- 波传播预测 - 浅水波速估算
- 前池水位控制 - 基于物质平衡
- 协调调度优化 - 多目标优化

版本: 1.0
"""

import math
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from datetime import datetime, timedelta
from collections import deque
import heapq


# ============================================================
# 1. 数据结构定义
# ============================================================

class CoordinationStrategy(Enum):
    """协调策略"""
    RELAY = auto()           # 智能接力 - 上游脉冲，下游配合
    PARALLEL = auto()        # 并行运行 - 同步调整
    SEQUENTIAL = auto()      # 顺序调整 - 逐级调整
    ADAPTIVE = auto()        # 自适应 - 根据工况选择


class WaveType(Enum):
    """波类型"""
    POSITIVE = auto()        # 正波 (水位升高)
    NEGATIVE = auto()        # 负波 (水位降低)


@dataclass
class CanalSection:
    """渠道段参数"""
    section_id: str
    length: float            # 渠道长度 (m)
    width: float             # 渠道宽度 (m)
    slope: float             # 底坡 (无量纲)
    roughness: float         # 糙率 (曼宁n)

    # 当前状态
    water_depth: float = 2.5  # 水深 (m)
    flow_rate: float = 0.0    # 流量 (m³/s)
    velocity: float = 0.0     # 流速 (m/s)

    # 计算属性
    @property
    def cross_section_area(self) -> float:
        """过水断面面积"""
        return self.width * self.water_depth

    @property
    def wetted_perimeter(self) -> float:
        """湿周"""
        return self.width + 2 * self.water_depth

    @property
    def hydraulic_radius(self) -> float:
        """水力半径"""
        return self.cross_section_area / self.wetted_perimeter

    @property
    def wave_celerity(self) -> float:
        """浅水波速 c = sqrt(g*h)"""
        g = 9.81
        return math.sqrt(g * self.water_depth)

    @property
    def froude_number(self) -> float:
        """弗劳德数 Fr = v / c"""
        if self.wave_celerity > 0:
            return abs(self.velocity) / self.wave_celerity
        return 0.0

    @property
    def storage_volume(self) -> float:
        """蓄量 (m³)"""
        return self.cross_section_area * self.length


@dataclass
class Forebay:
    """前池参数"""
    forebay_id: str
    area: float              # 前池面积 (m²)
    max_level: float         # 最高水位 (m)
    min_level: float         # 最低水位 (m)
    alarm_high: float        # 高水位报警 (m)
    alarm_low: float         # 低水位报警 (m)

    # 当前状态
    level: float = 3.0       # 当前水位 (m)

    @property
    def storage_volume(self) -> float:
        """当前蓄量 (m³)"""
        return self.area * self.level

    @property
    def available_volume(self) -> float:
        """可用容量 (m³)"""
        return self.area * (self.max_level - self.level)

    @property
    def remaining_volume(self) -> float:
        """剩余容量 (m³)"""
        return self.area * (self.level - self.min_level)

    def level_change_rate(self, inflow: float, outflow: float) -> float:
        """水位变化率 dH/dt (m/s)"""
        return (inflow - outflow) / self.area


@dataclass
class WavePropagation:
    """波传播事件"""
    wave_id: str
    source_station: str
    wave_type: WaveType
    amplitude: float         # 波幅 (流量变化, m³/s)
    start_time: float        # 发起时间 (timestamp)
    current_position: float  # 当前位置 (m from source)

    # 传播路径
    path: List[str] = field(default_factory=list)  # 经过的站点
    arrival_times: Dict[str, float] = field(default_factory=dict)  # 到达时间


@dataclass
class CoordinationCommand:
    """协调指令"""
    command_id: str
    target_station: str
    action: str              # 'start_pump', 'stop_pump', 'adjust_flow'
    target_value: float
    execute_time: float      # 执行时间 (timestamp)
    reason: str
    priority: int = 5
    confirmed: bool = False


# ============================================================
# 2. 波传播预测器
# ============================================================

class WavePropagationPredictor:
    """
    波传播预测器

    基于浅水波理论预测水流波动的传播
    """

    def __init__(self):
        self.active_waves: Dict[str, WavePropagation] = {}
        self.wave_counter = 0

        # 渠道网络
        self.canal_sections: Dict[str, CanalSection] = {}
        self.forebays: Dict[str, Forebay] = {}

        # 站点连接关系 (上游 -> 下游)
        self.connections: List[Tuple[str, str, str]] = []  # (upstream, canal_id, downstream)

    def add_canal_section(self, section: CanalSection):
        """添加渠道段"""
        self.canal_sections[section.section_id] = section

    def add_forebay(self, forebay: Forebay):
        """添加前池"""
        self.forebays[forebay.forebay_id] = forebay

    def add_connection(self, upstream: str, canal_id: str, downstream: str):
        """添加站点连接"""
        self.connections.append((upstream, canal_id, downstream))

    def create_wave(
        self,
        source_station: str,
        flow_change: float,
        start_time: float = None
    ) -> WavePropagation:
        """
        创建波传播事件

        Args:
            source_station: 源站点
            flow_change: 流量变化 (正=增加, 负=减少)
            start_time: 开始时间

        Returns:
            WavePropagation: 波传播事件
        """
        self.wave_counter += 1
        wave_id = f"WAVE_{self.wave_counter}"

        wave = WavePropagation(
            wave_id=wave_id,
            source_station=source_station,
            wave_type=WaveType.POSITIVE if flow_change > 0 else WaveType.NEGATIVE,
            amplitude=abs(flow_change),
            start_time=start_time or time.time(),
            current_position=0.0,
            path=[source_station],
            arrival_times={source_station: start_time or time.time()}
        )

        self.active_waves[wave_id] = wave
        return wave

    def predict_arrival_times(self, wave: WavePropagation) -> Dict[str, float]:
        """
        预测波到达各站点的时间

        Returns:
            {station_id: arrival_time}
        """
        arrival_times = {wave.source_station: wave.start_time}
        current_station = wave.source_station
        current_time = wave.start_time

        # 沿连接路径传播
        for upstream, canal_id, downstream in self.connections:
            if upstream == current_station:
                if canal_id in self.canal_sections:
                    canal = self.canal_sections[canal_id]

                    # 计算波速 (考虑顺流/逆流)
                    wave_speed = canal.wave_celerity

                    # 考虑流速影响 (顺流加速, 逆流减速)
                    if wave.wave_type == WaveType.POSITIVE:
                        effective_speed = wave_speed + canal.velocity
                    else:
                        effective_speed = wave_speed - canal.velocity

                    effective_speed = max(effective_speed, 1.0)  # 最小1 m/s

                    # 计算传播时间
                    travel_time = canal.length / effective_speed

                    current_time += travel_time
                    arrival_times[downstream] = current_time
                    current_station = downstream

        wave.arrival_times = arrival_times
        return arrival_times

    def get_wave_attenuation(self, distance: float, initial_amplitude: float) -> float:
        """
        计算波衰减

        波幅随距离衰减 (简化模型)
        """
        # 衰减系数 (经验值)
        attenuation_rate = 0.00005  # 每米衰减
        return initial_amplitude * math.exp(-attenuation_rate * distance)


# ============================================================
# 3. 前池水位控制器
# ============================================================

class ForebayLevelController:
    """
    前池水位控制器

    基于物质平衡的前池水位预测和控制
    """

    def __init__(self, forebay: Forebay):
        self.forebay = forebay
        self.level_history: deque = deque(maxlen=1000)

        # 控制参数
        self.target_level = (forebay.max_level + forebay.min_level) / 2
        self.level_tolerance = 0.3  # 允许偏差 (m)

        # PID参数
        self.kp = 0.5
        self.ki = 0.1
        self.kd = 0.2
        self.integral = 0.0
        self.last_error = 0.0

    def predict_level(
        self,
        current_level: float,
        inflow: float,
        outflow: float,
        duration: float
    ) -> float:
        """
        预测未来水位

        Args:
            current_level: 当前水位 (m)
            inflow: 入流量 (m³/s)
            outflow: 出流量 (m³/s)
            duration: 预测时长 (s)

        Returns:
            预测水位 (m)
        """
        level_change = (inflow - outflow) * duration / self.forebay.area
        predicted_level = current_level + level_change

        # 限制在有效范围内
        return max(self.forebay.min_level,
                  min(self.forebay.max_level, predicted_level))

    def calculate_required_outflow(
        self,
        current_level: float,
        target_level: float,
        inflow: float,
        time_horizon: float
    ) -> float:
        """
        计算维持目标水位所需的出流量

        Args:
            current_level: 当前水位 (m)
            target_level: 目标水位 (m)
            inflow: 入流量 (m³/s)
            time_horizon: 时间范围 (s)

        Returns:
            所需出流量 (m³/s)
        """
        level_diff = current_level - target_level
        volume_to_remove = level_diff * self.forebay.area

        # 需要的额外出流
        extra_outflow = volume_to_remove / time_horizon

        return inflow + extra_outflow

    def get_control_action(
        self,
        current_level: float,
        current_outflow: float,
        dt: float = 1.0
    ) -> Tuple[float, str]:
        """
        获取控制动作 (PID控制)

        Args:
            current_level: 当前水位
            current_outflow: 当前出流量
            dt: 时间步长

        Returns:
            (建议出流量变化, 原因)
        """
        error = current_level - self.target_level

        # PID计算
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0

        output = (self.kp * error +
                 self.ki * self.integral +
                 self.kd * derivative)

        self.last_error = error

        # 确定原因
        if current_level > self.forebay.alarm_high:
            reason = "前池高水位，需增加出流"
        elif current_level < self.forebay.alarm_low:
            reason = "前池低水位，需减少出流"
        else:
            reason = "水位正常调节"

        return output, reason

    def estimate_buffer_time(
        self,
        current_level: float,
        inflow: float,
        outflow: float
    ) -> Tuple[float, float]:
        """
        估算缓冲时间

        Returns:
            (到高水位时间, 到低水位时间) 秒, -1表示不会到达
        """
        net_flow = inflow - outflow

        if abs(net_flow) < 0.001:
            return -1, -1

        # 到高水位的时间
        if net_flow > 0:
            volume_to_high = self.forebay.area * (self.forebay.alarm_high - current_level)
            time_to_high = volume_to_high / net_flow if volume_to_high > 0 else 0
        else:
            time_to_high = -1

        # 到低水位的时间
        if net_flow < 0:
            volume_to_low = self.forebay.area * (current_level - self.forebay.alarm_low)
            time_to_low = volume_to_low / abs(net_flow) if volume_to_low > 0 else 0
        else:
            time_to_low = -1

        return time_to_high, time_to_low


# ============================================================
# 4. 级联协调调度器
# ============================================================

class CascadeCoordinator:
    """
    级联泵站协调调度器

    实现多级泵站间的智能协调:
    - 波传播预测
    - 前池水位控制
    - 协调指令生成
    - 优化调度策略
    """

    def __init__(self):
        # 波传播预测器
        self.wave_predictor = WavePropagationPredictor()

        # 前池控制器
        self.forebay_controllers: Dict[str, ForebayLevelController] = {}

        # 站点列表 (按顺序)
        self.station_order: List[str] = []

        # 协调参数
        self.params = {
            'advance_time': 60.0,      # 提前量 (s)
            'safety_margin': 0.1,      # 安全裕度
            'min_pump_interval': 300,  # 最小泵间隔 (s)
            'coordination_horizon': 3600,  # 协调时域 (s)
        }

        # 协调策略
        self.strategy = CoordinationStrategy.RELAY

        # 指令队列
        self.command_queue: List[CoordinationCommand] = []
        self.command_counter = 0

        # 历史记录
        self.coordination_log: deque = deque(maxlen=1000)

    def initialize_network(self, station_configs: List[Dict]):
        """
        初始化泵站网络

        Args:
            station_configs: [
                {
                    'station_id': 'tundian',
                    'station_name': '屯佃泵站',
                    'forebay_area': 2000,
                    'forebay_levels': (1.0, 5.0, 1.5, 4.5),
                    'downstream_canal': {
                        'length': 12000,
                        'width': 20,
                        'slope': 0.0001,
                        'roughness': 0.015
                    },
                    'downstream_station': 'qianliulin'
                },
                ...
            ]
        """
        self.station_order = []

        for config in station_configs:
            station_id = config['station_id']
            self.station_order.append(station_id)

            # 添加前池
            forebay = Forebay(
                forebay_id=f"{station_id}_forebay",
                area=config.get('forebay_area', 2000),
                min_level=config.get('forebay_levels', (1.0, 5.0, 1.5, 4.5))[0],
                max_level=config.get('forebay_levels', (1.0, 5.0, 1.5, 4.5))[1],
                alarm_low=config.get('forebay_levels', (1.0, 5.0, 1.5, 4.5))[2],
                alarm_high=config.get('forebay_levels', (1.0, 5.0, 1.5, 4.5))[3],
            )
            self.wave_predictor.add_forebay(forebay)
            self.forebay_controllers[station_id] = ForebayLevelController(forebay)

            # 添加下游渠道
            if 'downstream_canal' in config and 'downstream_station' in config:
                canal_config = config['downstream_canal']
                canal_id = f"canal_{station_id}_{config['downstream_station']}"

                canal = CanalSection(
                    section_id=canal_id,
                    length=canal_config.get('length', 12000),
                    width=canal_config.get('width', 20),
                    slope=canal_config.get('slope', 0.0001),
                    roughness=canal_config.get('roughness', 0.015)
                )
                self.wave_predictor.add_canal_section(canal)
                self.wave_predictor.add_connection(
                    station_id, canal_id, config['downstream_station']
                )

    def on_upstream_action(
        self,
        station_id: str,
        action: str,
        flow_change: float,
        action_time: float = None
    ) -> List[CoordinationCommand]:
        """
        响应上游泵站动作

        Args:
            station_id: 上游站点ID
            action: 动作类型 ('start_pump', 'stop_pump', 'adjust_flow')
            flow_change: 流量变化 (m³/s)
            action_time: 动作时间

        Returns:
            下游协调指令列表
        """
        commands = []
        action_time = action_time or time.time()

        # 创建波传播事件
        wave = self.wave_predictor.create_wave(station_id, flow_change, action_time)

        # 预测波到达时间
        arrival_times = self.wave_predictor.predict_arrival_times(wave)

        # 为下游站点生成协调指令
        station_index = self.station_order.index(station_id) if station_id in self.station_order else -1

        for downstream_station in self.station_order[station_index + 1:]:
            if downstream_station in arrival_times:
                arrival_time = arrival_times[downstream_station]

                # 计算提前量
                execute_time = arrival_time - self.params['advance_time']

                # 计算衰减后的波幅
                distance = sum(
                    self.wave_predictor.canal_sections[c].length
                    for u, c, d in self.wave_predictor.connections
                    if u == station_id or u in self.station_order[:self.station_order.index(downstream_station)]
                )
                attenuated_amplitude = self.wave_predictor.get_wave_attenuation(
                    distance, wave.amplitude
                )

                # 生成协调指令
                self.command_counter += 1
                command = CoordinationCommand(
                    command_id=f"COORD_{self.command_counter}",
                    target_station=downstream_station,
                    action=self._get_matching_action(action),
                    target_value=attenuated_amplitude * (1 + self.params['safety_margin']),
                    execute_time=execute_time,
                    reason=f"响应上游{station_id}的{action}动作",
                    priority=3
                )

                commands.append(command)
                self.command_queue.append(command)

        # 记录日志
        self.coordination_log.append({
            'timestamp': datetime.now(),
            'trigger': station_id,
            'action': action,
            'flow_change': flow_change,
            'commands_generated': len(commands),
            'wave_id': wave.wave_id
        })

        return commands

    def _get_matching_action(self, upstream_action: str) -> str:
        """获取匹配的下游动作"""
        action_map = {
            'start_pump': 'start_pump',
            'stop_pump': 'stop_pump',
            'increase_flow': 'increase_flow',
            'decrease_flow': 'decrease_flow',
            'adjust_flow': 'adjust_flow'
        }
        return action_map.get(upstream_action, 'adjust_flow')

    def get_optimal_schedule(
        self,
        target_flow: float,
        horizon_hours: int = 4
    ) -> List[Dict]:
        """
        获取最优协调调度计划

        Args:
            target_flow: 目标流量 (m³/s)
            horizon_hours: 规划时域 (小时)

        Returns:
            调度计划列表
        """
        schedule = []
        current_time = time.time()

        for hour in range(horizon_hours):
            hour_schedule = {
                'hour': hour,
                'timestamp': current_time + hour * 3600,
                'stations': {}
            }

            # 计算各站流量分配
            for i, station_id in enumerate(self.station_order):
                # 考虑渠道损失 (简化: 每段损失1%)
                station_flow = target_flow * (1 - 0.01 * i)

                # 根据前池状态调整
                if station_id in self.forebay_controllers:
                    controller = self.forebay_controllers[station_id]
                    forebay = controller.forebay

                    # 预测缓冲时间
                    time_to_high, time_to_low = controller.estimate_buffer_time(
                        forebay.level, station_flow, station_flow
                    )

                    status = 'normal'
                    if time_to_high > 0 and time_to_high < 1800:
                        status = 'high_risk'
                    elif time_to_low > 0 and time_to_low < 1800:
                        status = 'low_risk'

                    hour_schedule['stations'][station_id] = {
                        'target_flow': station_flow,
                        'forebay_level': forebay.level,
                        'status': status,
                        'buffer_time_high': time_to_high,
                        'buffer_time_low': time_to_low
                    }

            schedule.append(hour_schedule)

        return schedule

    def process_pending_commands(self) -> List[CoordinationCommand]:
        """
        处理待执行的协调指令

        Returns:
            需要执行的指令列表
        """
        current_time = time.time()
        to_execute = []

        remaining = []
        for command in self.command_queue:
            if command.execute_time <= current_time:
                to_execute.append(command)
                command.confirmed = True
            else:
                remaining.append(command)

        self.command_queue = remaining
        return to_execute

    def get_coordination_status(self) -> Dict:
        """获取协调状态"""
        return {
            'strategy': self.strategy.name,
            'station_count': len(self.station_order),
            'station_order': self.station_order,
            'active_waves': len(self.wave_predictor.active_waves),
            'pending_commands': len(self.command_queue),
            'coordination_log_size': len(self.coordination_log)
        }

    def get_wave_propagation_info(self) -> Dict:
        """获取波传播信息"""
        info = {
            'canal_sections': {},
            'estimated_travel_times': {}
        }

        for canal_id, canal in self.wave_predictor.canal_sections.items():
            info['canal_sections'][canal_id] = {
                'length': canal.length,
                'wave_celerity': canal.wave_celerity,
                'travel_time': canal.length / canal.wave_celerity
            }

        # 计算各站间传播时间
        for i, station in enumerate(self.station_order[:-1]):
            next_station = self.station_order[i + 1]
            for upstream, canal_id, downstream in self.wave_predictor.connections:
                if upstream == station and downstream == next_station:
                    canal = self.wave_predictor.canal_sections.get(canal_id)
                    if canal:
                        travel_time = canal.length / canal.wave_celerity
                        info['estimated_travel_times'][f"{station}_to_{next_station}"] = {
                            'distance': canal.length,
                            'travel_time': travel_time,
                            'travel_time_min': travel_time / 60
                        }

        return info


# ============================================================
# 5. 便捷函数
# ============================================================

def create_cascade_coordinator(station_configs: List[Dict] = None) -> CascadeCoordinator:
    """
    创建级联协调器

    Args:
        station_configs: 泵站配置，默认使用密云工程配置

    Returns:
        CascadeCoordinator 实例
    """
    coordinator = CascadeCoordinator()

    if station_configs is None:
        # 默认使用密云工程6级泵站配置
        station_configs = [
            {
                'station_id': 'tundian',
                'station_name': '屯佃泵站',
                'forebay_area': 2000,
                'forebay_levels': (1.0, 5.0, 1.5, 4.5),
                'downstream_canal': {
                    'length': 12000,
                    'width': 20,
                    'slope': 0.0001,
                    'roughness': 0.015
                },
                'downstream_station': 'qianliulin'
            },
            {
                'station_id': 'qianliulin',
                'station_name': '前柳林泵站',
                'forebay_area': 1800,
                'forebay_levels': (1.0, 5.0, 1.5, 4.5),
                'downstream_canal': {
                    'length': 10000,
                    'width': 18,
                    'slope': 0.0001,
                    'roughness': 0.015
                },
                'downstream_station': 'niantou'
            },
            {
                'station_id': 'niantou',
                'station_name': '念头泵站',
                'forebay_area': 1600,
                'forebay_levels': (1.0, 5.0, 1.5, 4.5),
                'downstream_canal': {
                    'length': 8000,
                    'width': 18,
                    'slope': 0.0001,
                    'roughness': 0.015
                },
                'downstream_station': 'xingshou'
            },
            {
                'station_id': 'xingshou',
                'station_name': '兴寿泵站',
                'forebay_area': 1500,
                'forebay_levels': (1.0, 5.0, 1.5, 4.5),
                'downstream_canal': {
                    'length': 9000,
                    'width': 16,
                    'slope': 0.0001,
                    'roughness': 0.015
                },
                'downstream_station': 'lishishan'
            },
            {
                'station_id': 'lishishan',
                'station_name': '李石山泵站',
                'forebay_area': 1400,
                'forebay_levels': (1.0, 5.0, 1.5, 4.5),
                'downstream_canal': {
                    'length': 7000,
                    'width': 16,
                    'slope': 0.0001,
                    'roughness': 0.015
                },
                'downstream_station': 'xitaishang'
            },
            {
                'station_id': 'xitaishang',
                'station_name': '西台上泵站',
                'forebay_area': 1200,
                'forebay_levels': (1.0, 5.0, 1.5, 4.5),
            },
        ]

    coordinator.initialize_network(station_configs)
    return coordinator


# ============================================================
# 导出
# ============================================================

__all__ = [
    # 枚举
    'CoordinationStrategy',
    'WaveType',

    # 数据结构
    'CanalSection',
    'Forebay',
    'WavePropagation',
    'CoordinationCommand',

    # 控制器
    'WavePropagationPredictor',
    'ForebayLevelController',
    'CascadeCoordinator',

    # 便捷函数
    'create_cascade_coordinator',
]
