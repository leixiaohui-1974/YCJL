"""
压力传感器模型
==============

用于管道压力监测:
- 高频采样能力
- 水锤波形捕获
- 压力脉动滤波
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass

from ..config.settings import Config


@dataclass
class PressureReading:
    """压力读数"""
    value: float           # 测量值 (m水头)
    peak_value: float      # 峰值
    min_value: float       # 谷值
    rate_of_change: float  # 变化率 (m/s)
    quality: float         # 质量
    is_valid: bool         # 是否有效
    timestamp: float       # 时间戳


class PressureSensor:
    """
    压力传感器仿真

    特性:
    - 高频响应 (100Hz)
    - 脉动抑制
    - 峰值检测
    - 故障诊断
    """

    def __init__(self, name: str, position: str):
        self.name = name
        self.position = position

        # 测量参数
        self.range_min = 0.0           # 量程下限 (m)
        self.range_max = 150.0         # 量程上限 (m)
        self.accuracy = 0.5            # 精度 (m)
        self.sampling_rate = 100.0     # 采样率 (Hz)

        # 噪声参数
        self.noise_std = Config.simulation.sensor_noise_std.get('pressure', 0.5)

        # 动态特性
        self.time_constant = 0.01      # 时间常数 (s) - 快响应
        self.last_output = 50.0

        # 滤波器 (移动平均)
        self.filter_size = 5
        self.filter_buffer: List[float] = []

        # 峰值检测
        self.peak_window = 100         # 峰值窗口 (采样点)
        self.peak_buffer: List[float] = []

        # 故障状态
        self.is_fault = False
        self.fault_type = None

        # 历史
        self.history = []
        self.current_time = 0.0
        self.prev_value = 50.0

    def _apply_dynamics(self, true_value: float, dt: float) -> float:
        """快速一阶响应"""
        tau = self.time_constant
        alpha = dt / (tau + dt)
        output = (1 - alpha) * self.last_output + alpha * true_value
        self.last_output = output
        return output

    def _apply_noise(self, value: float) -> float:
        """添加测量噪声"""
        noise = np.random.normal(0, self.noise_std)
        return value + noise

    def _apply_filter(self, value: float) -> float:
        """移动平均滤波"""
        self.filter_buffer.append(value)
        if len(self.filter_buffer) > self.filter_size:
            self.filter_buffer.pop(0)
        return np.mean(self.filter_buffer)

    def _update_peaks(self, value: float) -> tuple:
        """更新峰值检测"""
        self.peak_buffer.append(value)
        if len(self.peak_buffer) > self.peak_window:
            self.peak_buffer.pop(0)

        if len(self.peak_buffer) > 0:
            return max(self.peak_buffer), min(self.peak_buffer)
        return value, value

    def _compute_rate(self, value: float, dt: float) -> float:
        """计算变化率"""
        rate = (value - self.prev_value) / dt
        self.prev_value = value
        return rate

    def measure(self, true_value: float, dt: float) -> PressureReading:
        """
        执行测量

        Parameters:
            true_value: 真实压力 (m水头)
            dt: 时间步长 (s)

        Returns:
            PressureReading: 测量结果
        """
        self.current_time += dt

        # 量程检查
        is_in_range = self.range_min <= true_value <= self.range_max

        # 应用传感器特性
        value = self._apply_dynamics(true_value, dt)
        raw_value = self._apply_noise(value)
        filtered_value = self._apply_filter(raw_value)

        # 峰值检测
        peak, trough = self._update_peaks(raw_value)

        # 变化率
        rate = self._compute_rate(filtered_value, dt)

        # 故障处理
        if self.is_fault:
            if self.fault_type == 'stuck':
                filtered_value = self.prev_value
            elif self.fault_type == 'spike':
                if np.random.random() < 0.1:
                    filtered_value += np.random.choice([-1, 1]) * 20.0

        # 质量评估
        quality = 1.0
        if not is_in_range:
            quality *= 0.5
        if abs(rate) > 50.0:  # 异常快速变化
            quality *= 0.8
        if self.is_fault:
            quality *= 0.3

        reading = PressureReading(
            value=filtered_value,
            peak_value=peak,
            min_value=trough,
            rate_of_change=rate,
            quality=quality,
            is_valid=is_in_range and not self.is_fault,
            timestamp=self.current_time
        )

        self.history.append(reading)
        return reading

    def detect_water_hammer(self, threshold: float = 30.0) -> bool:
        """检测水锤事件"""
        if len(self.peak_buffer) < 10:
            return False

        amplitude = max(self.peak_buffer) - min(self.peak_buffer)
        return amplitude > threshold

    def get_waveform(self, duration: float) -> np.ndarray:
        """获取波形数据"""
        num_samples = int(duration * self.sampling_rate)
        if len(self.history) < num_samples:
            num_samples = len(self.history)

        return np.array([r.value for r in self.history[-num_samples:]])

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
        self.last_output = 50.0
        self.filter_buffer.clear()
        self.peak_buffer.clear()
        self.prev_value = 50.0
        self.is_fault = False
        self.history.clear()
        self.current_time = 0.0
