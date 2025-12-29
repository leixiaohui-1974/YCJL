"""
前馈控制与补偿
==============

前馈补偿器
史密斯预估器
干扰观测器
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class FeedforwardConfig:
    """前馈配置"""
    gain: float = 1.0
    lead_time: float = 0.0
    filter_tau: float = 1.0


class FeedforwardCompensator:
    """
    前馈补偿器

    用于补偿可测量扰动
    """

    def __init__(self, config: FeedforwardConfig = None):
        self.config = config or FeedforwardConfig()

        # 滤波器状态
        self.filtered_disturbance = 0.0

        # 超前补偿缓冲
        self.disturbance_buffer: deque = deque(maxlen=100)

    def compute(self, disturbance: float, dt: float = 1.0) -> float:
        """
        计算前馈补偿信号

        Parameters:
            disturbance: 可测量扰动
            dt: 采样周期

        Returns:
            前馈补偿量
        """
        # 记录扰动
        self.disturbance_buffer.append(disturbance)

        # 滤波
        tau = self.config.filter_tau
        alpha = dt / (tau + dt)
        self.filtered_disturbance = alpha * disturbance + \
            (1 - alpha) * self.filtered_disturbance

        # 超前补偿 (如果有)
        lead_steps = int(self.config.lead_time / dt)
        if lead_steps > 0 and len(self.disturbance_buffer) > lead_steps:
            # 基于历史预测
            recent = list(self.disturbance_buffer)[-10:]
            if len(recent) >= 2:
                trend = (recent[-1] - recent[0]) / len(recent)
                predicted = disturbance + trend * lead_steps
                compensation = -self.config.gain * predicted
            else:
                compensation = -self.config.gain * self.filtered_disturbance
        else:
            compensation = -self.config.gain * self.filtered_disturbance

        return compensation

    def reset(self):
        """重置"""
        self.filtered_disturbance = 0.0
        self.disturbance_buffer.clear()


class SmithPredictor:
    """
    史密斯预估器

    用于补偿纯滞后过程
    """

    def __init__(self,
                 process_gain: float = 1.0,
                 process_tau: float = 10.0,
                 dead_time: float = 5.0,
                 dt: float = 1.0):
        """
        Parameters:
            process_gain: 过程增益
            process_tau: 过程时间常数
            dead_time: 纯滞后时间
            dt: 采样周期
        """
        self.K = process_gain
        self.tau = process_tau
        self.L = dead_time
        self.dt = dt

        # 滞后步数
        self.delay_steps = int(dead_time / dt)

        # 模型状态
        self.model_state = 0.0

        # 延迟缓冲
        self.delay_buffer: deque = deque(maxlen=self.delay_steps + 1)
        for _ in range(self.delay_steps + 1):
            self.delay_buffer.append(0.0)

        # 补偿信号
        self.compensation = 0.0

    def predict(self, u: float) -> float:
        """
        预测器步进

        Parameters:
            u: 控制输入

        Returns:
            预测输出 (无延迟)
        """
        # 模型 (一阶惯性): dy/dt = (K*u - y) / tau
        alpha = self.dt / (self.tau + self.dt)
        self.model_state = alpha * self.K * u + (1 - alpha) * self.model_state

        # 延迟模型输出
        self.delay_buffer.append(self.model_state)
        delayed_model = self.delay_buffer[0]

        return self.model_state, delayed_model

    def compute_compensation(self, u: float, y_measured: float) -> float:
        """
        计算补偿信号

        Parameters:
            u: 控制输入
            y_measured: 实际测量值

        Returns:
            补偿后的反馈信号
        """
        # 模型预测
        y_model_nodelay, y_model_delayed = self.predict(u)

        # 补偿信号 = 测量值 - 延迟模型 + 无延迟模型
        # 这样控制器看到的是"无延迟"的过程
        self.compensation = y_measured - y_model_delayed + y_model_nodelay

        return self.compensation

    def update_model(self, K: float = None, tau: float = None, L: float = None):
        """更新模型参数"""
        if K is not None:
            self.K = K
        if tau is not None:
            self.tau = tau
        if L is not None:
            self.L = L
            new_delay_steps = int(L / self.dt)
            if new_delay_steps != self.delay_steps:
                self.delay_steps = new_delay_steps
                self.delay_buffer = deque(maxlen=self.delay_steps + 1)
                for _ in range(self.delay_steps + 1):
                    self.delay_buffer.append(self.model_state)

    def reset(self):
        """重置"""
        self.model_state = 0.0
        self.delay_buffer.clear()
        for _ in range(self.delay_steps + 1):
            self.delay_buffer.append(0.0)
        self.compensation = 0.0


class DisturbanceObserver:
    """
    干扰观测器 (DOB)

    估计和补偿未建模干扰
    """

    def __init__(self,
                 process_gain: float = 1.0,
                 process_tau: float = 10.0,
                 filter_tau: float = 2.0,
                 dt: float = 1.0):
        """
        Parameters:
            process_gain: 过程增益
            process_tau: 过程时间常数
            filter_tau: Q滤波器时间常数
            dt: 采样周期
        """
        self.K = process_gain
        self.tau = process_tau
        self.filter_tau = filter_tau
        self.dt = dt

        # 模型逆
        self.G_inv_state = 0.0

        # Q滤波器状态
        self.Q_state = 0.0

        # 估计的干扰
        self.d_hat = 0.0

    def estimate(self, u: float, y: float) -> float:
        """
        估计干扰

        d_hat = Q * (G_inv * y - u)

        Parameters:
            u: 控制输入
            y: 测量输出

        Returns:
            估计的干扰
        """
        # 模型逆: G_inv(s) = (tau*s + 1) / K
        # 离散化: G_inv[k] = (tau/dt + 1) * y[k] / K - tau/dt * y[k-1] / K
        # 简化为一阶滤波逆
        alpha = self.dt / (self.tau + self.dt)
        self.G_inv_state = (y / self.K - (1 - alpha) * self.G_inv_state) / alpha

        # 干扰估计 (未滤波)
        d_raw = self.G_inv_state - u

        # Q滤波 (低通)
        alpha_q = self.dt / (self.filter_tau + self.dt)
        self.Q_state = alpha_q * d_raw + (1 - alpha_q) * self.Q_state

        self.d_hat = self.Q_state

        return self.d_hat

    def compensate(self, u: float, y: float) -> float:
        """
        计算补偿后的控制信号

        u_compensated = u - d_hat

        Parameters:
            u: 原始控制信号
            y: 测量输出

        Returns:
            补偿后的控制信号
        """
        d_hat = self.estimate(u, y)
        return u - d_hat

    def reset(self):
        """重置"""
        self.G_inv_state = 0.0
        self.Q_state = 0.0
        self.d_hat = 0.0


class FeedforwardFeedbackController:
    """
    前馈-反馈复合控制器
    """

    def __init__(self,
                 Kp: float = 1.0,
                 Ki: float = 0.1,
                 Kd: float = 0.01,
                 ff_gain: float = 0.5):
        """
        Parameters:
            Kp, Ki, Kd: PID参数
            ff_gain: 前馈增益
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.ff_gain = ff_gain

        # 状态
        self.integral = 0.0
        self.last_error = 0.0
        self.last_sp = 0.0

    def compute(self, sp: float, pv: float, disturbance: float = 0.0,
                dt: float = 1.0) -> float:
        """
        计算控制输出

        Parameters:
            sp: 设定点
            pv: 过程变量
            disturbance: 可测量扰动
            dt: 采样周期

        Returns:
            控制输出
        """
        error = sp - pv

        # 反馈控制
        P = self.Kp * error
        self.integral += self.Ki * error * dt
        self.integral = np.clip(self.integral, -10, 10)
        D = self.Kd * (error - self.last_error) / dt

        feedback = P + self.integral + D

        # 前馈控制
        # 设定点变化前馈
        sp_change = (sp - self.last_sp) / dt
        ff_sp = 0.5 * sp_change

        # 扰动前馈
        ff_disturbance = -self.ff_gain * disturbance

        feedforward = ff_sp + ff_disturbance

        # 总输出
        output = feedback + feedforward
        output = np.clip(output, 0, 1)

        # 更新状态
        self.last_error = error
        self.last_sp = sp

        return output

    def reset(self):
        """重置"""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_sp = 0.0
