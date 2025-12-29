"""
降阶模型
========

用于实时MPC的简化模型:
- 传递函数模型
- 状态空间降阶
- 模态截断
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from ..config.settings import Config


@dataclass
class ModelResponse:
    """模型响应"""
    output: np.ndarray
    state: np.ndarray
    time: float


class TransferFunctionModel:
    """
    传递函数模型

    用于管段输入输出关系的简化描述:
    G(s) = K * e^(-τs) / (Ts + 1)
    """

    def __init__(self, gain: float = 1.0, time_constant: float = 100.0,
                 delay: float = 0.0):
        """
        初始化传递函数模型

        Parameters:
            gain: 稳态增益 K
            time_constant: 时间常数 T (s)
            delay: 纯滞后时间 τ (s)
        """
        self.K = gain
        self.T = time_constant
        self.tau = delay

        # 滞后环节采用离散延迟线
        self.delay_buffer: List[float] = []
        self.delay_steps = 0

        # 一阶惯性环节状态
        self.y = 0.0

    def set_delay_buffer(self, dt: float):
        """设置延迟缓冲区"""
        self.delay_steps = int(self.tau / dt) + 1
        self.delay_buffer = [0.0] * self.delay_steps

    def step(self, u: float, dt: float) -> float:
        """
        单步推进

        Parameters:
            u: 输入
            dt: 时间步长

        Returns:
            输出
        """
        # 更新延迟缓冲
        if self.delay_steps > 0:
            if len(self.delay_buffer) < self.delay_steps:
                self.delay_buffer = [0.0] * self.delay_steps
            self.delay_buffer.append(u)
            u_delayed = self.delay_buffer.pop(0)
        else:
            u_delayed = u

        # 一阶惯性环节
        # dy/dt = (K*u - y) / T
        dy = (self.K * u_delayed - self.y) / self.T
        self.y += dy * dt

        return self.y

    def get_steady_state(self, u: float) -> float:
        """获取稳态输出"""
        return self.K * u

    def reset(self, initial_output: float = 0.0):
        """重置"""
        self.y = initial_output
        self.delay_buffer = [initial_output] * self.delay_steps


class ReducedOrderModel:
    """
    降阶状态空间模型

    用于实时MPC的管网简化模型:
    dx/dt = A*x + B*u
    y = C*x + D*u
    """

    def __init__(self, order: int = 4):
        """
        初始化降阶模型

        Parameters:
            order: 模型阶数
        """
        self.n = order
        self.cfg = Config

        # 状态空间矩阵
        self.A = np.zeros((order, order))
        self.B = np.zeros((order, 2))  # 2个输入
        self.C = np.zeros((2, order))  # 2个输出
        self.D = np.zeros((2, 2))

        # 状态
        self.x = np.zeros(order)

        # 初始化为简单的一阶模型
        self._initialize_default()

        # 辨识数据
        self.input_history: List[np.ndarray] = []
        self.output_history: List[np.ndarray] = []

    def _initialize_default(self):
        """初始化默认模型"""
        # 简单的对角系统
        # 时间常数: 60s, 300s, 1000s, 3600s
        time_constants = [60.0, 300.0, 1000.0, 3600.0][:self.n]

        for i, T in enumerate(time_constants):
            self.A[i, i] = -1.0 / T

        # 输入增益 (流量->压力, 阀门->流量)
        self.B[:, 0] = 1.0  # 第一个输入
        self.B[:, 1] = 0.5  # 第二个输入

        # 输出矩阵
        self.C[0, :] = 1.0
        self.C[1, :] = 0.5

    def discretize(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        离散化

        使用零阶保持器 (ZOH)

        Parameters:
            dt: 采样时间

        Returns:
            (Ad, Bd): 离散状态空间矩阵
        """
        n = self.n

        # 简化：使用前向欧拉
        Ad = np.eye(n) + self.A * dt
        Bd = self.B * dt

        return Ad, Bd

    def step(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        单步推进

        Parameters:
            u: 输入向量 [u1, u2]
            dt: 时间步长

        Returns:
            输出向量 [y1, y2]
        """
        # 状态更新
        dx = self.A @ self.x + self.B @ u
        self.x += dx * dt

        # 输出
        y = self.C @ self.x + self.D @ u

        return y

    def predict_horizon(self, x0: np.ndarray, u_seq: np.ndarray,
                        dt: float) -> np.ndarray:
        """
        预测时域内的输出

        Parameters:
            x0: 初始状态
            u_seq: 控制序列 (N, 2)
            dt: 时间步长

        Returns:
            输出序列 (N, 2)
        """
        N = len(u_seq)
        y_seq = np.zeros((N, 2))

        x = x0.copy()
        for k in range(N):
            dx = self.A @ x + self.B @ u_seq[k]
            x = x + dx * dt
            y_seq[k] = self.C @ x + self.D @ u_seq[k]

        return y_seq

    def identify(self, input_data: np.ndarray, output_data: np.ndarray,
                 dt: float):
        """
        系统辨识

        使用最小二乘法估计模型参数

        Parameters:
            input_data: 输入数据 (N, 2)
            output_data: 输出数据 (N, 2)
            dt: 采样时间
        """
        N = len(input_data)
        if N < self.n * 10:
            return  # 数据不足

        # 简化辨识：只估计时间常数
        # 使用输出的自相关特性

        # 计算输出的衰减率
        y = output_data[:, 0]
        y_mean = np.mean(y)
        y_centered = y - y_mean

        # 自相关
        autocorr = np.correlate(y_centered, y_centered, mode='full')
        autocorr = autocorr[N - 1:]

        # 估计主时间常数
        if autocorr[0] > 0:
            half_idx = np.argmax(autocorr < autocorr[0] * 0.5)
            T_est = half_idx * dt / np.log(2)

            if T_est > 10 and T_est < 10000:
                self.A[0, 0] = -1.0 / T_est

        self.input_history.extend(input_data.tolist())
        self.output_history.extend(output_data.tolist())

    def get_state_space(self) -> Tuple[np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray]:
        """获取状态空间矩阵"""
        return self.A.copy(), self.B.copy(), self.C.copy(), self.D.copy()

    def set_state_space(self, A: np.ndarray, B: np.ndarray,
                        C: np.ndarray, D: np.ndarray):
        """设置状态空间矩阵"""
        self.A = A.copy()
        self.B = B.copy()
        self.C = C.copy()
        self.D = D.copy()
        self.n = A.shape[0]
        self.x = np.zeros(self.n)

    def reset(self, x0: np.ndarray = None):
        """重置状态"""
        if x0 is not None:
            self.x = x0.copy()
        else:
            self.x = np.zeros(self.n)


class TunnelReducedModel(ReducedOrderModel):
    """
    隧洞降阶模型

    将圣维南方程简化为有限维状态空间
    """

    def __init__(self, num_modes: int = 4):
        super().__init__(order=num_modes)

        self.cfg = Config.tunnel

        # 隧洞特有参数
        self.L = self.cfg.total_length
        self.wave_speed = np.sqrt(Config.physics.G * 3.5)  # 近似波速

        # 模态频率
        self._compute_modal_parameters()

    def _compute_modal_parameters(self):
        """计算模态参数"""
        # 基于扩散波近似的特征值
        n = Config.tunnel.manning_n_normal
        S = Config.tunnel.bottom_slope
        B = Config.tunnel.width
        h = 3.5  # 平均水深

        # 扩散系数
        D = (5 / 3) * h * np.sqrt(h * S) / n

        # 模态衰减率
        for i in range(self.n):
            k = (i + 1) * np.pi / self.L  # 波数
            lambda_i = D * k ** 2
            self.A[i, i] = -lambda_i

        # 模态增益
        for i in range(self.n):
            self.B[i, 0] = 1.0 / (i + 1)
            self.C[0, i] = 1.0


class PipelineReducedModel(ReducedOrderModel):
    """
    管道降阶模型

    将MOC方程简化为传递函数级联
    """

    def __init__(self, num_segments: int = 4):
        super().__init__(order=num_segments)

        self.cfg = Config.pipeline

        # 管道分段
        self.segments = num_segments
        self.segment_length = self.cfg.total_length / num_segments

        # 每段的时间常数
        self._compute_segment_parameters()

    def _compute_segment_parameters(self):
        """计算分段参数"""
        L_seg = self.segment_length
        a = self.cfg.wave_speed
        D = self.cfg.diameter
        A = self.cfg.area
        g = Config.physics.G

        # 惯性时间常数
        T_inertia = L_seg * A / (g * A)  # 简化

        # 摩阻时间常数
        f = self.cfg.darcy_f
        v = 2.0  # 典型流速
        T_friction = 2 * D / (f * v)

        # 综合时间常数
        T_eff = min(T_inertia, T_friction)

        for i in range(self.n):
            self.A[i, i] = -1.0 / (T_eff * (i + 1))

        # 级联结构
        for i in range(self.n - 1):
            self.A[i + 1, i] = 1.0 / T_eff
