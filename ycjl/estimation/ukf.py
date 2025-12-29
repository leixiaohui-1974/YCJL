"""
无迹卡尔曼滤波 (UKF)
====================

相比EKF更适合强非线性系统:
- 避免雅可比计算
- 更好的概率近似
- 处理非高斯分布
"""

import numpy as np
from typing import Optional, Callable, Tuple, Dict
from dataclasses import dataclass

from ..config.settings import Config


@dataclass
class UKFResult:
    """UKF结果"""
    state: np.ndarray
    covariance: np.ndarray
    innovation: np.ndarray
    sigma_points: np.ndarray


class UnscentedKalmanFilter:
    """
    无迹卡尔曼滤波器

    使用sigma点进行非线性变换的统计逼近
    """

    def __init__(self, state_dim: int, meas_dim: int):
        self.n = state_dim
        self.m = meas_dim

        # UKF参数
        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0.0
        self.lambda_ = self.alpha ** 2 * (self.n + self.kappa) - self.n

        # 状态
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim)

        # 噪声
        self.Q = np.eye(state_dim) * 0.01
        self.R = np.eye(meas_dim) * 0.1

        # 非线性函数
        self.f = None
        self.h = None

        # 权重
        self._compute_weights()

        # 历史
        self.history = []

    def _compute_weights(self):
        """计算sigma点权重"""
        n = self.n
        lambda_ = self.lambda_

        # 均值权重
        self.Wm = np.zeros(2 * n + 1)
        self.Wm[0] = lambda_ / (n + lambda_)
        self.Wm[1:] = 1.0 / (2 * (n + lambda_))

        # 协方差权重
        self.Wc = np.zeros(2 * n + 1)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha ** 2 + self.beta)
        self.Wc[1:] = 1.0 / (2 * (n + lambda_))

    def _compute_sigma_points(self, x: np.ndarray,
                               P: np.ndarray) -> np.ndarray:
        """
        计算sigma点

        2n+1个点围绕均值分布
        """
        n = len(x)
        sigma_points = np.zeros((2 * n + 1, n))

        # Cholesky分解
        try:
            sqrt_P = np.linalg.cholesky((n + self.lambda_) * P)
        except np.linalg.LinAlgError:
            # 如果矩阵不正定，添加小扰动
            sqrt_P = np.linalg.cholesky((n + self.lambda_) * (P + np.eye(n) * 1e-6))

        sigma_points[0] = x

        for i in range(n):
            sigma_points[i + 1] = x + sqrt_P[i]
            sigma_points[n + i + 1] = x - sqrt_P[i]

        return sigma_points

    def set_dynamics(self, f: Callable):
        """设置状态转移函数"""
        self.f = f

    def set_measurement(self, h: Callable):
        """设置观测函数"""
        self.h = h

    def set_noise(self, Q: np.ndarray, R: np.ndarray):
        """设置噪声协方差"""
        self.Q = Q
        self.R = R

    def initialize(self, x0: np.ndarray, P0: np.ndarray):
        """初始化"""
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
        # 生成sigma点
        sigma_points = self._compute_sigma_points(self.x, self.P)

        # 传播sigma点
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(2 * self.n + 1):
            sigma_points_pred[i] = self.f(sigma_points[i], u, dt)

        # 计算预测均值
        x_pred = np.zeros(self.n)
        for i in range(2 * self.n + 1):
            x_pred += self.Wm[i] * sigma_points_pred[i]

        # 计算预测协方差
        P_pred = np.zeros((self.n, self.n))
        for i in range(2 * self.n + 1):
            diff = sigma_points_pred[i] - x_pred
            P_pred += self.Wc[i] * np.outer(diff, diff)
        P_pred += self.Q

        return x_pred, P_pred

    def update(self, z: np.ndarray, x_pred: np.ndarray,
               P_pred: np.ndarray) -> UKFResult:
        """
        更新步

        Parameters:
            z: 观测向量
            x_pred: 预测状态
            P_pred: 预测协方差

        Returns:
            UKFResult
        """
        # 生成sigma点
        sigma_points = self._compute_sigma_points(x_pred, P_pred)

        # 观测sigma点
        z_sigma = np.zeros((2 * self.n + 1, self.m))
        for i in range(2 * self.n + 1):
            z_sigma[i] = self.h(sigma_points[i])

        # 观测均值
        z_pred = np.zeros(self.m)
        for i in range(2 * self.n + 1):
            z_pred += self.Wm[i] * z_sigma[i]

        # 观测协方差
        Pzz = np.zeros((self.m, self.m))
        for i in range(2 * self.n + 1):
            diff = z_sigma[i] - z_pred
            Pzz += self.Wc[i] * np.outer(diff, diff)
        Pzz += self.R

        # 交叉协方差
        Pxz = np.zeros((self.n, self.m))
        for i in range(2 * self.n + 1):
            diff_x = sigma_points[i] - x_pred
            diff_z = z_sigma[i] - z_pred
            Pxz += self.Wc[i] * np.outer(diff_x, diff_z)

        # 卡尔曼增益
        K = Pxz @ np.linalg.inv(Pzz)

        # 更新
        innovation = z - z_pred
        self.x = x_pred + K @ innovation
        self.P = P_pred - K @ Pzz @ K.T

        result = UKFResult(
            state=self.x.copy(),
            covariance=self.P.copy(),
            innovation=innovation,
            sigma_points=sigma_points
        )

        self.history.append(result)
        return result

    def step(self, z: np.ndarray, u: np.ndarray, dt: float) -> UKFResult:
        """完整预测-更新循环"""
        x_pred, P_pred = self.predict(u, dt)
        return self.update(z, x_pred, P_pred)

    def reset(self, x0: np.ndarray = None, P0: np.ndarray = None):
        """重置"""
        if x0 is not None:
            self.x = x0.copy()
        else:
            self.x = np.zeros(self.n)
        if P0 is not None:
            self.P = P0.copy()
        else:
            self.P = np.eye(self.n)
        self.history.clear()
