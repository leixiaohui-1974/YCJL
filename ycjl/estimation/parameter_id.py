"""
参数辨识模块
============

实现在线参数估计:
- 曼宁糙率 n
- 达西摩阻系数 f
- 阀门流量系数 Cv
- 泄漏系数

采用递推最小二乘法（RLS）和扩展卡尔曼滤波
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..config.settings import Config


@dataclass
class ParameterEstimate:
    """参数估计结果"""
    value: float           # 估计值
    variance: float        # 方差
    confidence: float      # 置信度
    residual: float        # 残差
    is_converged: bool     # 是否收敛


class ParameterIdentifier(ABC):
    """参数辨识基类"""

    def __init__(self, name: str, initial_value: float):
        self.name = name
        self.value = initial_value
        self.variance = 1.0
        self.history: List[float] = []

    @abstractmethod
    def update(self, measurement: Dict, dt: float) -> ParameterEstimate:
        """更新参数估计"""
        pass

    @abstractmethod
    def reset(self):
        """重置"""
        pass


class ManningEstimator(ParameterIdentifier):
    """
    曼宁糙率在线估计

    基于水位-流量观测，利用曼宁公式反演糙率:
    Q = (1/n) * A * R^(2/3) * S^(1/2)
    => n = A * R^(2/3) * S^(1/2) / Q
    """

    def __init__(self, initial_n: float = 0.014):
        super().__init__("Manning_n", initial_n)

        self.cfg = Config.tunnel

        # RLS参数
        self.P = 1.0                   # 协方差
        self.lambda_factor = 0.98      # 遗忘因子
        self.Q_process = 1e-8          # 过程噪声
        self.R_measure = 1e-4          # 观测噪声

        # 收敛判据
        self.residual_threshold = 0.001
        self.convergence_count = 0

    def _compute_theoretical_n(self, Q: float, h: float, S: float) -> float:
        """根据曼宁公式计算理论糙率"""
        if Q < 0.1 or h < 0.1:
            return self.value

        B = self.cfg.width
        A = B * h
        P = B + 2 * h
        R = A / P

        n_computed = A * (R ** (2 / 3)) * np.sqrt(S) / Q
        return np.clip(n_computed, 0.008, 0.025)

    def update(self, measurement: Dict, dt: float) -> ParameterEstimate:
        """
        更新糙率估计

        Parameters:
            measurement: 包含 Q (流量), h (水深), S (坡度)
            dt: 时间步长

        Returns:
            ParameterEstimate: 估计结果
        """
        Q = measurement.get('flow', 10.0)
        h = measurement.get('depth', 3.5)
        S = measurement.get('slope', self.cfg.bottom_slope)

        # 计算观测糙率
        n_observed = self._compute_theoretical_n(Q, h, S)

        # 递推最小二乘更新
        # 简化为一维参数估计

        # 预测
        n_pred = self.value
        P_pred = self.P + self.Q_process

        # 卡尔曼增益
        K = P_pred / (P_pred + self.R_measure)

        # 更新
        residual = n_observed - n_pred
        self.value = n_pred + K * residual
        self.P = (1 - K) * P_pred

        # 约束
        self.value = np.clip(self.value, 0.008, 0.025)

        # 遗忘因子
        self.P = self.P / self.lambda_factor

        # 收敛判断
        is_converged = abs(residual) < self.residual_threshold
        if is_converged:
            self.convergence_count += 1
        else:
            self.convergence_count = 0

        self.variance = self.P
        self.history.append(self.value)

        # 计算置信度
        confidence = 1.0 / (1.0 + self.P * 100)

        return ParameterEstimate(
            value=self.value,
            variance=self.P,
            confidence=confidence,
            residual=residual,
            is_converged=self.convergence_count > 10
        )

    def get_ice_correction(self, temperature: float) -> float:
        """
        获取冰期糙率修正

        Parameters:
            temperature: 水温 (°C)

        Returns:
            修正后的糙率
        """
        if temperature > 4.0:
            return self.value

        # 冰期糙率增加
        correction = 1.0 + 0.3 * (4.0 - temperature) / 4.0
        return self.value * correction

    def reset(self):
        """重置"""
        self.value = 0.014
        self.P = 1.0
        self.convergence_count = 0
        self.history.clear()


class FrictionEstimator(ParameterIdentifier):
    """
    管道摩阻系数在线估计

    基于Darcy-Weisbach公式:
    h_f = f * (L/D) * (v²/2g)

    利用压差和流量观测反演 f
    """

    def __init__(self, initial_f: float = 0.012):
        super().__init__("Darcy_f", initial_f)

        self.cfg = Config.pipeline

        # 状态空间 [f]
        self.x = np.array([initial_f])
        self.P = np.array([[0.01]])

        # 噪声
        self.Q = np.array([[1e-10]])  # 过程噪声
        self.R = np.array([[0.1]])    # 观测噪声

        # 历史
        self.residual_history: List[float] = []

    def _compute_theoretical_f(self, Q: float, H_up: float, H_down: float,
                                L: float) -> float:
        """根据压差和流量计算摩阻系数"""
        if Q < 0.1:
            return self.value

        D = self.cfg.diameter
        A = self.cfg.area
        v = Q / A
        g = Config.physics.G

        delta_H = H_up - H_down
        if delta_H < 0.1:
            return self.value

        # h_f = f * L/D * v²/2g
        # f = h_f * 2g * D / (L * v²)
        f_computed = delta_H * 2 * g * D / (L * v ** 2)
        return np.clip(f_computed, 0.005, 0.03)

    def update(self, measurement: Dict, dt: float) -> ParameterEstimate:
        """
        更新摩阻系数估计

        Parameters:
            measurement: 包含 Q, H_up, H_down, L
            dt: 时间步长

        Returns:
            ParameterEstimate: 估计结果
        """
        Q = measurement.get('flow', 10.0)
        H_up = measurement.get('H_up', 60.0)
        H_down = measurement.get('H_down', 50.0)
        L = measurement.get('length', self.cfg.total_length / 2)

        # 观测值
        f_observed = self._compute_theoretical_f(Q, H_up, H_down, L)

        # EKF更新
        # 状态转移 (随机游走)
        x_pred = self.x
        P_pred = self.P + self.Q

        # 观测模型 (线性)
        H = np.array([[1.0]])
        z = np.array([f_observed])
        z_pred = H @ x_pred

        # 卡尔曼增益
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)

        # 更新
        residual = z - z_pred
        self.x = x_pred + K @ residual
        self.P = (np.eye(1) - K @ H) @ P_pred

        # 约束
        self.x[0] = np.clip(self.x[0], 0.005, 0.03)
        self.value = self.x[0]

        self.history.append(self.value)
        self.residual_history.append(residual[0])

        # 收敛判断
        is_converged = np.std(self.history[-20:]) < 0.0005 if len(self.history) > 20 else False

        confidence = 1.0 / (1.0 + self.P[0, 0] * 100)

        return ParameterEstimate(
            value=self.value,
            variance=self.P[0, 0],
            confidence=confidence,
            residual=residual[0],
            is_converged=is_converged
        )

    def detect_anomaly(self) -> Tuple[bool, str]:
        """
        检测异常

        Returns:
            (是否异常, 异常类型)
        """
        if len(self.history) < 50:
            return False, ""

        recent = self.history[-50:]
        mean = np.mean(recent)
        std = np.std(recent)

        # 突变检测
        if abs(self.value - mean) > 3 * std:
            return True, "sudden_change"

        # 趋势检测
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        if abs(slope) > 0.0001:
            return True, "trending"

        return False, ""

    def reset(self):
        """重置"""
        self.value = 0.012
        self.x = np.array([0.012])
        self.P = np.array([[0.01]])
        self.history.clear()
        self.residual_history.clear()


class ValveCvEstimator(ParameterIdentifier):
    """阀门Cv系数估计"""

    def __init__(self, initial_cv: float = 20.0):
        super().__init__("Valve_Cv", initial_cv)
        self.P = 1.0
        self.Q_process = 0.01
        self.R_measure = 0.5

    def update(self, measurement: Dict, dt: float) -> ParameterEstimate:
        """
        更新Cv估计

        基于 Q = Cv * sqrt(ΔH)
        """
        Q = measurement.get('flow', 10.0)
        delta_H = measurement.get('delta_H', 10.0)
        opening = measurement.get('opening', 0.5)

        if delta_H < 0.1 or opening < 0.01:
            return ParameterEstimate(self.value, self.P, 0.5, 0.0, False)

        # 观测Cv
        Cv_observed = Q / np.sqrt(delta_H) / opening

        # 卡尔曼更新
        P_pred = self.P + self.Q_process
        K = P_pred / (P_pred + self.R_measure)

        residual = Cv_observed - self.value
        self.value = self.value + K * residual
        self.P = (1 - K) * P_pred

        self.value = np.clip(self.value, 5.0, 50.0)
        self.history.append(self.value)

        return ParameterEstimate(
            value=self.value,
            variance=self.P,
            confidence=1.0 / (1.0 + self.P),
            residual=residual,
            is_converged=abs(residual) < 0.5
        )

    def reset(self):
        """重置"""
        self.value = 20.0
        self.P = 1.0
        self.history.clear()
