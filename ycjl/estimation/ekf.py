"""
扩展卡尔曼滤波 (EKF)
====================

用于非线性系统的状态估计和参数辨识:
- 管道水力状态估计
- 在线参数跟踪
- 数字孪生同步
"""

import numpy as np
from typing import Optional, Callable, Tuple, Dict
from dataclasses import dataclass

from ..config.settings import Config


@dataclass
class EKFResult:
    """EKF结果"""
    state: np.ndarray          # 后验状态估计
    covariance: np.ndarray     # 后验协方差
    innovation: np.ndarray     # 新息
    kalman_gain: np.ndarray    # 卡尔曼增益
    nees: float                # 归一化估计误差平方


class ExtendedKalmanFilter:
    """
    扩展卡尔曼滤波器

    用于联合估计系统状态和模型参数
    """

    def __init__(self, state_dim: int, meas_dim: int,
                 param_dim: int = 0):
        """
        初始化EKF

        Parameters:
            state_dim: 状态维度
            meas_dim: 观测维度
            param_dim: 参数维度 (联合估计时)
        """
        self.nx = state_dim
        self.nz = meas_dim
        self.np = param_dim
        self.n = state_dim + param_dim  # 增广状态维度

        # 增广状态 [x; p]
        self.x = np.zeros(self.n)
        self.P = np.eye(self.n)

        # 噪声协方差
        self.Q = np.eye(self.n) * 0.01    # 过程噪声
        self.R = np.eye(meas_dim) * 0.1   # 观测噪声

        # 非线性函数 (将由用户设置)
        self.f = None  # 状态转移函数
        self.h = None  # 观测函数
        self.F_jacobian = None  # 状态转移雅可比
        self.H_jacobian = None  # 观测雅可比

        # 历史
        self.history: Dict[str, list] = {
            'state': [],
            'covariance': [],
            'innovation': [],
            'nees': []
        }

    def set_dynamics(self, f: Callable, F_jacobian: Callable):
        """
        设置状态转移函数

        Parameters:
            f: x(k+1) = f(x(k), u(k), dt)
            F_jacobian: F = df/dx
        """
        self.f = f
        self.F_jacobian = F_jacobian

    def set_measurement(self, h: Callable, H_jacobian: Callable):
        """
        设置观测函数

        Parameters:
            h: z = h(x)
            H_jacobian: H = dh/dx
        """
        self.h = h
        self.H_jacobian = H_jacobian

    def set_noise(self, Q: np.ndarray, R: np.ndarray):
        """设置噪声协方差"""
        self.Q = Q
        self.R = R

    def initialize(self, x0: np.ndarray, P0: np.ndarray):
        """初始化状态"""
        self.x = x0.copy()
        self.P = P0.copy()

    def predict(self, u: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测步

        Parameters:
            u: 控制输入
            dt: 时间步长

        Returns:
            (预测状态, 预测协方差)
        """
        if self.f is None:
            raise ValueError("State transition function not set")

        # 状态预测
        x_pred = self.f(self.x, u, dt)

        # 雅可比计算
        if self.F_jacobian is not None:
            F = self.F_jacobian(self.x, u, dt)
        else:
            # 数值雅可比
            F = self._numerical_jacobian(self.f, self.x, u, dt)

        # 协方差预测
        P_pred = F @ self.P @ F.T + self.Q

        return x_pred, P_pred

    def update(self, z: np.ndarray, x_pred: np.ndarray,
               P_pred: np.ndarray) -> EKFResult:
        """
        更新步

        Parameters:
            z: 观测向量
            x_pred: 预测状态
            P_pred: 预测协方差

        Returns:
            EKFResult: 滤波结果
        """
        if self.h is None:
            raise ValueError("Measurement function not set")

        # 观测预测
        z_pred = self.h(x_pred)

        # 雅可比计算
        if self.H_jacobian is not None:
            H = self.H_jacobian(x_pred)
        else:
            H = self._numerical_jacobian_h(self.h, x_pred)

        # 新息
        innovation = z - z_pred

        # 新息协方差
        S = H @ P_pred @ H.T + self.R

        # 卡尔曼增益
        K = P_pred @ H.T @ np.linalg.inv(S)

        # 状态更新
        self.x = x_pred + K @ innovation

        # 协方差更新 (Joseph形式，数值稳定)
        IKH = np.eye(self.n) - K @ H
        self.P = IKH @ P_pred @ IKH.T + K @ self.R @ K.T

        # NEES (归一化估计误差平方)
        nees = innovation.T @ np.linalg.inv(S) @ innovation

        result = EKFResult(
            state=self.x.copy(),
            covariance=self.P.copy(),
            innovation=innovation,
            kalman_gain=K,
            nees=nees
        )

        # 记录历史
        self.history['state'].append(self.x.copy())
        self.history['covariance'].append(np.diag(self.P).copy())
        self.history['innovation'].append(innovation.copy())
        self.history['nees'].append(nees)

        return result

    def step(self, z: np.ndarray, u: np.ndarray, dt: float) -> EKFResult:
        """
        完整的预测-更新循环

        Parameters:
            z: 观测向量
            u: 控制输入
            dt: 时间步长

        Returns:
            EKFResult: 滤波结果
        """
        x_pred, P_pred = self.predict(u, dt)
        return self.update(z, x_pred, P_pred)

    def _numerical_jacobian(self, f: Callable, x: np.ndarray,
                            u: np.ndarray, dt: float,
                            eps: float = 1e-6) -> np.ndarray:
        """数值计算状态转移雅可比"""
        n = len(x)
        F = np.zeros((n, n))

        f0 = f(x, u, dt)

        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = f(x_plus, u, dt)
            F[:, i] = (f_plus - f0) / eps

        return F

    def _numerical_jacobian_h(self, h: Callable, x: np.ndarray,
                               eps: float = 1e-6) -> np.ndarray:
        """数值计算观测雅可比"""
        n = len(x)
        h0 = h(x)
        m = len(h0)
        H = np.zeros((m, n))

        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            h_plus = h(x_plus)
            H[:, i] = (h_plus - h0) / eps

        return H

    def get_state_estimate(self, index: int = None) -> np.ndarray:
        """获取状态估计"""
        if index is None:
            return self.x[:self.nx]
        return self.x[index]

    def get_parameter_estimate(self) -> np.ndarray:
        """获取参数估计"""
        if self.np > 0:
            return self.x[self.nx:]
        return np.array([])

    def get_uncertainty(self, index: int = None) -> float:
        """获取估计不确定性"""
        if index is None:
            return np.sqrt(np.diag(self.P[:self.nx, :self.nx]))
        return np.sqrt(self.P[index, index])

    def is_consistent(self, confidence: float = 0.95) -> bool:
        """
        检查滤波器一致性

        基于NEES检验
        """
        if len(self.history['nees']) < 20:
            return True

        recent_nees = self.history['nees'][-20:]
        avg_nees = np.mean(recent_nees)

        # 对于nz维观测，NEES应该接近nz (卡方分布)
        lower = self.nz * 0.5
        upper = self.nz * 2.0

        return lower < avg_nees < upper

    def reset(self, x0: np.ndarray = None, P0: np.ndarray = None):
        """重置滤波器"""
        if x0 is not None:
            self.x = x0.copy()
        else:
            self.x = np.zeros(self.n)

        if P0 is not None:
            self.P = P0.copy()
        else:
            self.P = np.eye(self.n)

        for key in self.history:
            self.history[key].clear()


class PipelineEKF(ExtendedKalmanFilter):
    """
    管道水力系统专用EKF

    状态: [Q_mid, H_mid, f]
    观测: [Q_measured, H_measured]
    """

    def __init__(self):
        super().__init__(state_dim=2, meas_dim=2, param_dim=1)

        self.cfg = Config.pipeline

        # 初始化
        self.x = np.array([10.0, 50.0, 0.012])  # [Q, H, f]
        self.P = np.diag([1.0, 10.0, 0.001])

        # 噪声
        self.Q = np.diag([0.1, 0.5, 1e-8])
        self.R = np.diag([0.1, 0.5])

        # 设置函数
        self.set_dynamics(self._state_transition, self._state_jacobian)
        self.set_measurement(self._measurement, self._measurement_jacobian)

    def _state_transition(self, x: np.ndarray, u: np.ndarray,
                          dt: float) -> np.ndarray:
        """状态转移"""
        Q, H, f = x
        u_valve = u[0] if len(u) > 0 else 0.5

        # 简化动力学 (管道惯性)
        # dQ/dt ≈ (H_up - H - h_f) * gA / L
        A = self.cfg.area
        L = self.cfg.total_length / 2
        g = Config.physics.G
        D = self.cfg.diameter

        # 摩阻
        v = Q / A
        h_f = f * L / D * (v ** 2) / (2 * g)

        # 假设H_up已知
        H_up = 60.0

        dQ = (H_up - H - h_f) * g * A / L * 0.01  # 缩放
        Q_new = Q + dQ * dt

        # f 随机游走
        f_new = f

        return np.array([Q_new, H, f_new])

    def _state_jacobian(self, x: np.ndarray, u: np.ndarray,
                        dt: float) -> np.ndarray:
        """状态转移雅可比"""
        Q, H, f = x
        A = self.cfg.area
        L = self.cfg.total_length / 2
        g = Config.physics.G
        D = self.cfg.diameter

        v = Q / A

        F = np.eye(3)
        F[0, 0] = 1 - 2 * f * L / D * v / A / (2 * g) * g * A / L * 0.01 * dt
        F[0, 1] = -g * A / L * 0.01 * dt
        F[0, 2] = -L / D * (v ** 2) / (2 * g) * g * A / L * 0.01 * dt

        return F

    def _measurement(self, x: np.ndarray) -> np.ndarray:
        """观测函数"""
        Q, H, f = x
        return np.array([Q, H])

    def _measurement_jacobian(self, x: np.ndarray) -> np.ndarray:
        """观测雅可比"""
        return np.array([[1, 0, 0], [0, 1, 0]])

    def get_friction_estimate(self) -> Tuple[float, float]:
        """获取摩阻系数估计和不确定性"""
        f_est = self.x[2]
        f_std = np.sqrt(self.P[2, 2])
        return f_est, f_std
