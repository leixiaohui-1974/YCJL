"""
流量传感器模型
==============

支持多种类型:
- 超声波多声道
- 电磁流量计
- 文丘里管

包含流态影响、紊流误差等特性
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum, auto

from ..config.settings import Config


class FlowSensorType(Enum):
    """流量计类型"""
    ULTRASONIC = auto()      # 超声波
    ELECTROMAGNETIC = auto() # 电磁
    VENTURI = auto()         # 文丘里


@dataclass
class FlowReading:
    """流量读数"""
    value: float           # 瞬时流量 (m³/s)
    accumulated: float     # 累计流量 (m³)
    velocity: float        # 流速 (m/s)
    quality: float         # 质量
    is_valid: bool         # 是否有效
    is_reverse: bool       # 是否逆流
    timestamp: float       # 时间戳


class FlowSensor:
    """
    流量传感器仿真

    特性:
    - 多声道超声波测量
    - 流态校正
    - 低流速误差
    - 累计流量计算
    """

    def __init__(self, name: str, position: str,
                 sensor_type: FlowSensorType = FlowSensorType.ULTRASONIC,
                 pipe_diameter: float = 2.4):
        self.name = name
        self.position = position
        self.sensor_type = sensor_type
        self.pipe_diameter = pipe_diameter
        self.pipe_area = np.pi * (pipe_diameter / 2) ** 2

        # 测量参数
        self.range_min = 0.0           # 量程下限 (m³/s)
        self.range_max = 30.0          # 量程上限 (m³/s)
        self.accuracy_percent = 0.5    # 精度 (%)

        # 噪声参数
        self.noise_std = Config.simulation.sensor_noise_std.get('flow', 0.05)

        # 动态特性
        self.time_constant = 1.0       # 时间常数 (s)
        self.last_output = 10.0

        # 低流速校正
        self.min_velocity = 0.3        # 最低可测流速 (m/s)
        self.low_flow_error = 0.02     # 低流速附加误差

        # 累计流量
        self.accumulated_flow = 0.0

        # 故障状态
        self.is_fault = False
        self.fault_type = None

        # 历史
        self.history = []
        self.current_time = 0.0

    def _compute_velocity(self, flow: float) -> float:
        """计算流速"""
        return flow / self.pipe_area

    def _apply_dynamics(self, true_value: float, dt: float) -> float:
        """一阶动态响应"""
        tau = self.time_constant
        alpha = dt / (tau + dt)
        output = (1 - alpha) * self.last_output + alpha * true_value
        self.last_output = output
        return output

    def _apply_flow_correction(self, value: float) -> float:
        """流态校正"""
        velocity = self._compute_velocity(abs(value))

        if velocity < self.min_velocity:
            # 低流速区，精度下降
            error = self.low_flow_error * (1 - velocity / self.min_velocity)
            value *= (1 + np.random.uniform(-error, error))
        elif velocity > 5.0:
            # 高流速区，紊流增强
            error = 0.01 * (velocity - 5.0)
            value *= (1 + np.random.uniform(-error, error))

        return value

    def _apply_noise(self, value: float) -> float:
        """添加测量噪声"""
        # 相对误差
        relative_noise = np.random.normal(0, self.accuracy_percent / 100)
        # 绝对噪声
        absolute_noise = np.random.normal(0, self.noise_std)
        return value * (1 + relative_noise) + absolute_noise

    def measure(self, true_value: float, dt: float) -> FlowReading:
        """
        执行测量

        Parameters:
            true_value: 真实流量 (m³/s)
            dt: 时间步长 (s)

        Returns:
            FlowReading: 测量结果
        """
        self.current_time += dt

        # 检测逆流
        is_reverse = true_value < 0

        # 量程检查
        is_in_range = self.range_min <= abs(true_value) <= self.range_max

        # 应用传感器特性
        value = self._apply_dynamics(true_value, dt)
        value = self._apply_flow_correction(value)
        value = self._apply_noise(value)

        # 故障处理
        if self.is_fault:
            if self.fault_type == 'stuck':
                value = self.history[-1].value if self.history else 0
            elif self.fault_type == 'zero':
                value = 0.0
            elif self.fault_type == 'offset':
                value += 2.0  # 固定偏移

        # 累计流量
        self.accumulated_flow += abs(value) * dt

        # 计算流速
        velocity = self._compute_velocity(abs(value))

        # 质量评估
        quality = 1.0
        if not is_in_range:
            quality *= 0.5
        if velocity < self.min_velocity:
            quality *= 0.7
        if is_reverse:
            quality *= 0.9
        if self.is_fault:
            quality *= 0.3

        reading = FlowReading(
            value=value,
            accumulated=self.accumulated_flow,
            velocity=velocity,
            quality=quality,
            is_valid=is_in_range and not self.is_fault,
            is_reverse=is_reverse,
            timestamp=self.current_time
        )

        self.history.append(reading)
        return reading

    def get_daily_volume(self) -> float:
        """获取日累计流量"""
        return self.accumulated_flow

    def reset_accumulated(self):
        """重置累计流量"""
        self.accumulated_flow = 0.0

    def inject_fault(self, fault_type: str):
        """注入故障"""
        self.is_fault = True
        self.fault_type = fault_type

    def clear_fault(self):
        """清除故障"""
        self.is_fault = False
        self.fault_type = None

    def reset(self):
        """重置传感器"""
        self.last_output = 10.0
        self.accumulated_flow = 0.0
        self.is_fault = False
        self.history.clear()
        self.current_time = 0.0
