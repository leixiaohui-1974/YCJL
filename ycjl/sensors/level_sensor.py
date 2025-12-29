"""
水位传感器模型
==============

支持多种类型:
- 静压式
- 超声波式
- 雷达式

包含噪声、漂移、冰期干扰等特性
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto

from ..config.settings import Config


class LevelSensorType(Enum):
    """传感器类型"""
    PRESSURE = auto()     # 静压式
    ULTRASONIC = auto()   # 超声波式
    RADAR = auto()        # 雷达式


@dataclass
class LevelReading:
    """水位读数"""
    value: float           # 测量值 (m)
    raw_value: float       # 原始值
    quality: float         # 质量 (0~1)
    is_valid: bool         # 是否有效
    timestamp: float       # 时间戳


class LevelSensor:
    """
    水位传感器仿真

    特性:
    - 高斯白噪声
    - 零点漂移
    - 一阶动态响应
    - 故障注入
    """

    def __init__(self, name: str, position: str,
                 sensor_type: LevelSensorType = LevelSensorType.PRESSURE):
        self.name = name
        self.position = position
        self.sensor_type = sensor_type

        # 测量参数
        self.range_min = 0.0          # 量程下限 (m)
        self.range_max = 20.0         # 量程上限 (m)
        self.accuracy = 0.01          # 精度 (m)
        self.resolution = 0.001       # 分辨率 (m)

        # 噪声参数
        self.noise_std = Config.simulation.sensor_noise_std.get('level', 0.01)

        # 动态特性
        self.time_constant = 0.5      # 时间常数 (s)
        self.last_output = 0.0        # 上次输出

        # 漂移参数
        self.drift_rate = 1e-6        # 漂移速率 (m/s)
        self.drift_accumulated = 0.0  # 累积漂移

        # 故障状态
        self.is_fault = False
        self.fault_type = None        # 'stuck', 'drift', 'noise', 'offset'
        self.fault_value = 0.0

        # 冰期影响
        self.ice_interference = 0.0   # 冰盖干扰量

        # 历史
        self.history = []
        self.current_time = 0.0

    def _apply_dynamics(self, true_value: float, dt: float) -> float:
        """应用一阶动态响应"""
        tau = self.time_constant
        alpha = dt / (tau + dt)
        output = (1 - alpha) * self.last_output + alpha * true_value
        self.last_output = output
        return output

    def _apply_noise(self, value: float) -> float:
        """添加测量噪声"""
        noise = np.random.normal(0, self.noise_std)
        return value + noise

    def _apply_drift(self, value: float, dt: float) -> float:
        """应用零点漂移"""
        self.drift_accumulated += self.drift_rate * dt
        return value + self.drift_accumulated

    def _apply_ice_effect(self, value: float) -> float:
        """应用冰期干扰"""
        if self.sensor_type == LevelSensorType.ULTRASONIC:
            # 超声波受冰面反射干扰较大
            if self.ice_interference > 0:
                value += np.random.uniform(-0.5, 0.5) * self.ice_interference
        elif self.sensor_type == LevelSensorType.RADAR:
            # 雷达式干扰较小
            if self.ice_interference > 0:
                value += np.random.uniform(-0.1, 0.1) * self.ice_interference
        # 静压式不受冰面影响
        return value

    def _apply_fault(self, value: float) -> float:
        """应用故障效果"""
        if not self.is_fault:
            return value

        if self.fault_type == 'stuck':
            return self.fault_value
        elif self.fault_type == 'drift':
            return value + self.fault_value
        elif self.fault_type == 'noise':
            return value + np.random.normal(0, self.fault_value)
        elif self.fault_type == 'offset':
            return value + self.fault_value
        return value

    def measure(self, true_value: float, dt: float) -> LevelReading:
        """
        执行测量

        Parameters:
            true_value: 真实水位 (m)
            dt: 时间步长 (s)

        Returns:
            LevelReading: 测量结果
        """
        self.current_time += dt

        # 量程检查
        is_in_range = self.range_min <= true_value <= self.range_max

        # 应用传感器特性
        raw = true_value
        value = self._apply_dynamics(true_value, dt)
        value = self._apply_drift(value, dt)
        value = self._apply_ice_effect(value)
        value = self._apply_noise(value)
        value = self._apply_fault(value)

        # 量化
        value = round(value / self.resolution) * self.resolution

        # 质量评估
        quality = 1.0
        if not is_in_range:
            quality *= 0.5
        if self.is_fault:
            quality *= 0.3
        if self.ice_interference > 0:
            quality *= 0.8

        reading = LevelReading(
            value=value,
            raw_value=raw,
            quality=quality,
            is_valid=is_in_range and not self.is_fault,
            timestamp=self.current_time
        )

        self.history.append(reading)
        return reading

    def inject_fault(self, fault_type: str, fault_value: float):
        """注入故障"""
        self.is_fault = True
        self.fault_type = fault_type
        self.fault_value = fault_value

    def clear_fault(self):
        """清除故障"""
        self.is_fault = False
        self.fault_type = None
        self.fault_value = 0.0

    def set_ice_mode(self, enable: bool, thickness: float = 0.5):
        """设置冰期模式"""
        if enable:
            self.ice_interference = thickness
        else:
            self.ice_interference = 0.0

    def calibrate(self, offset: float = None):
        """校准（消除漂移）"""
        if offset is not None:
            self.drift_accumulated = -offset
        else:
            self.drift_accumulated = 0.0

    def get_statistics(self) -> dict:
        """获取统计信息"""
        if len(self.history) < 2:
            return {}

        values = [r.value for r in self.history[-100:]]
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'drift': self.drift_accumulated
        }

    def reset(self):
        """重置传感器"""
        self.last_output = 0.0
        self.drift_accumulated = 0.0
        self.is_fault = False
        self.fault_type = None
        self.ice_interference = 0.0
        self.history.clear()
        self.current_time = 0.0
