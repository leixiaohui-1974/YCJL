"""
温度传感器模型
==============

用于:
- 冰期监测
- 糙率补偿
- 密度修正
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from ..config.settings import Config


@dataclass
class TemperatureReading:
    """温度读数"""
    water_temp: float      # 水温 (°C)
    air_temp: float        # 气温 (°C)
    is_freezing_risk: bool # 结冰风险
    ice_probability: float # 结冰概率
    quality: float         # 质量
    timestamp: float       # 时间戳


class TemperatureSensor:
    """
    温度传感器仿真

    特性:
    - 热惯性
    - 日变化周期
    - 结冰预警
    """

    def __init__(self, name: str, position: str):
        self.name = name
        self.position = position

        # 测量参数
        self.range_min = -30.0         # 量程下限 (°C)
        self.range_max = 50.0          # 量程上限 (°C)
        self.accuracy = 0.1            # 精度 (°C)

        # 噪声参数
        self.noise_std = Config.simulation.sensor_noise_std.get('temperature', 0.1)

        # 热惯性
        self.thermal_time_constant = 60.0  # 热时间常数 (s)
        self.last_water_temp = 15.0
        self.last_air_temp = 20.0

        # 故障状态
        self.is_fault = False

        # 历史
        self.history = []
        self.current_time = 0.0

    def _apply_thermal_dynamics(self, true_temp: float, last_temp: float,
                                  dt: float) -> float:
        """热惯性响应"""
        tau = self.thermal_time_constant
        alpha = dt / (tau + dt)
        return (1 - alpha) * last_temp + alpha * true_temp

    def _apply_noise(self, value: float) -> float:
        """添加测量噪声"""
        return value + np.random.normal(0, self.noise_std)

    def _estimate_ice_probability(self, water_temp: float, air_temp: float) -> float:
        """估计结冰概率"""
        if water_temp > 2.0:
            return 0.0
        elif water_temp > 0.5:
            base_prob = 0.3 * (2.0 - water_temp) / 1.5
        else:
            base_prob = 0.5 + 0.5 * (0.5 - water_temp) / 0.5

        # 气温影响
        if air_temp < -10:
            base_prob *= 1.5
        elif air_temp < 0:
            base_prob *= 1.2

        return min(base_prob, 1.0)

    def measure(self, true_water_temp: float, true_air_temp: float,
                dt: float) -> TemperatureReading:
        """
        执行测量

        Parameters:
            true_water_temp: 真实水温 (°C)
            true_air_temp: 真实气温 (°C)
            dt: 时间步长 (s)

        Returns:
            TemperatureReading: 测量结果
        """
        self.current_time += dt

        # 应用热惯性
        water_temp = self._apply_thermal_dynamics(
            true_water_temp, self.last_water_temp, dt
        )
        air_temp = self._apply_thermal_dynamics(
            true_air_temp, self.last_air_temp, dt
        )

        self.last_water_temp = water_temp
        self.last_air_temp = air_temp

        # 添加噪声
        water_temp = self._apply_noise(water_temp)
        air_temp = self._apply_noise(air_temp)

        # 故障处理
        if self.is_fault:
            water_temp = self.history[-1].water_temp if self.history else 10.0

        # 结冰风险评估
        is_freezing_risk = water_temp < 2.0 and air_temp < 0
        ice_probability = self._estimate_ice_probability(water_temp, air_temp)

        # 质量
        quality = 1.0 if not self.is_fault else 0.3

        reading = TemperatureReading(
            water_temp=water_temp,
            air_temp=air_temp,
            is_freezing_risk=is_freezing_risk,
            ice_probability=ice_probability,
            quality=quality,
            timestamp=self.current_time
        )

        self.history.append(reading)
        return reading

    def get_manning_correction(self) -> float:
        """
        获取曼宁糙率修正系数

        冰期糙率增加
        """
        if not self.history:
            return 1.0

        water_temp = self.history[-1].water_temp
        if water_temp > 4.0:
            return 1.0
        elif water_temp > 0.5:
            # 线性插值
            return 1.0 + 0.3 * (4.0 - water_temp) / 3.5
        else:
            return 1.3  # 冰期最大糙率系数

    def inject_fault(self):
        """注入故障"""
        self.is_fault = True

    def clear_fault(self):
        """清除故障"""
        self.is_fault = False

    def reset(self):
        """重置传感器"""
        self.last_water_temp = 15.0
        self.last_air_temp = 20.0
        self.is_fault = False
        self.history.clear()
        self.current_time = 0.0
