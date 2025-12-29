"""
自适应控制器
============

模型参考自适应控制 (MRAC)
自校正调节器 (STR)
增益调度控制
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque


class AdaptiveMethod(Enum):
    """自适应方法"""
    MRAC = auto()      # 模型参考自适应
    STR = auto()       # 自校正调节器
    GAIN_SCHEDULING = auto()  # 增益调度


@dataclass
class AdaptiveState:
    """自适应状态"""
    theta: np.ndarray      # 参数估计
    P: np.ndarray          # 协方差矩阵
    error: float = 0.0
    adaptation_rate: float = 0.0


class AdaptiveController:
    """
    通用自适应控制器基类
    """

    def __init__(self, n_params: int = 4, learning_rate: float = 0.01):
        self.n_params = n_params
        self.learning_rate = learning_rate

        # 参数估计
        self.theta = np.zeros(n_params)

        # 遗忘因子
        self.forgetting_factor = 0.99

        # 历史
        self.history: deque = deque(maxlen=500)

    def update_params(self, error: float, regressor: np.ndarray):
        """更新参数估计"""
        raise NotImplementedError

    def compute_control(self, reference: float, measurement: float) -> float:
        """计算控制信号"""
        raise NotImplementedError

    def reset(self):
        """重置"""
        self.theta = np.zeros(self.n_params)
        self.history.clear()


class MRACController(AdaptiveController):
    """
    模型参考自适应控制器 (MRAC)

    基于MIT规则或Lyapunov方法
    """

    def __init__(self,
                 reference_model: Tuple[float, float] = (1.0, 1.0),
                 adaptation_gain: float = 0.1,
                 method: str = "mit"):
        """
        Parameters:
            reference_model: (wn, zeta) 参考模型参数
            adaptation_gain: 自适应增益
            method: "mit" 或 "lyapunov"
        """
        super().__init__(n_params=2)

        # 参考模型: G_m(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
        self.wn, self.zeta = reference_model
        self.method = method
        self.gamma = adaptation_gain

        # 控制器参数 [k1, k2]
        self.theta = np.array([1.0, 0.1])

        # 状态
        self.x_m = np.zeros(2)  # 参考模型状态
        self.x = np.zeros(2)    # 实际状态

        self.e = 0.0  # 跟踪误差
        self.u = 0.0  # 控制输入

    def reference_model_step(self, r: float, dt: float) -> float:
        """参考模型步进"""
        wn, zeta = self.wn, self.zeta

        # 状态空间形式
        A_m = np.array([
            [0, 1],
            [-wn**2, -2*zeta*wn]
        ])
        B_m = np.array([0, wn**2])

        # 欧拉法
        self.x_m = self.x_m + dt * (A_m @ self.x_m + B_m * r)

        return self.x_m[0]

    def compute_control(self, reference: float, measurement: float,
                        dt: float = 0.1) -> float:
        """
        计算控制信号

        Parameters:
            reference: 参考输入
            measurement: 测量值
            dt: 采样周期

        Returns:
            控制信号
        """
        # 参考模型输出
        y_m = self.reference_model_step(reference, dt)

        # 跟踪误差
        self.e = measurement - y_m

        # 控制律: u = k1*r + k2*y
        k1, k2 = self.theta
        self.u = k1 * reference + k2 * measurement

        # 自适应律
        self._adapt(reference, measurement, dt)

        return self.u

    def _adapt(self, r: float, y: float, dt: float):
        """自适应律"""
        if self.method == "mit":
            # MIT规则
            # dk1/dt = -gamma * e * r
            # dk2/dt = -gamma * e * y
            self.theta[0] -= self.gamma * self.e * r * dt
            self.theta[1] -= self.gamma * self.e * y * dt
        else:
            # Lyapunov方法 (更稳定)
            phi = np.array([r, y])
            self.theta -= self.gamma * self.e * phi * dt

        # 参数投影 (防止参数漂移)
        self.theta = np.clip(self.theta, -10, 10)

    def get_status(self) -> Dict:
        """获取状态"""
        return {
            'theta': self.theta.tolist(),
            'error': self.e,
            'control': self.u,
            'reference_model_output': self.x_m[0]
        }


class STRController(AdaptiveController):
    """
    自校正调节器 (STR)

    基于递推最小二乘辨识 + 极点配置设计
    """

    def __init__(self,
                 model_order: int = 2,
                 forgetting_factor: float = 0.98,
                 desired_poles: List[float] = None):
        """
        Parameters:
            model_order: 模型阶数
            forgetting_factor: 遗忘因子
            desired_poles: 期望极点
        """
        self.model_order = model_order
        self.na = model_order  # A多项式阶数
        self.nb = model_order  # B多项式阶数
        n_params = self.na + self.nb

        super().__init__(n_params=n_params)

        self.forgetting_factor = forgetting_factor

        # 期望极点
        if desired_poles is None:
            self.desired_poles = [0.8, 0.7]  # 默认稳定极点
        else:
            self.desired_poles = desired_poles

        # RLS参数
        self.theta = np.zeros(n_params)
        self.P = np.eye(n_params) * 1000  # 初始协方差

        # 数据缓冲
        self.y_buffer: deque = deque(maxlen=model_order + 1)
        self.u_buffer: deque = deque(maxlen=model_order + 1)

        # 控制器参数
        self.R = np.zeros(self.nb)  # R多项式
        self.S = np.zeros(self.na)  # S多项式

    def update_params(self, y: float, u: float):
        """
        RLS参数更新

        模型: y(t) = -a1*y(t-1) - ... - ana*y(t-na)
                    + b1*u(t-1) + ... + bnb*u(t-nb)
        """
        self.y_buffer.append(y)
        self.u_buffer.append(u)

        if len(self.y_buffer) < self.model_order + 1:
            return

        # 构建回归向量
        phi = np.zeros(self.na + self.nb)
        for i in range(self.na):
            phi[i] = -self.y_buffer[-(i+2)]
        for i in range(self.nb):
            phi[self.na + i] = self.u_buffer[-(i+2)]

        # 预测误差
        y_pred = phi @ self.theta
        e = y - y_pred

        # RLS更新
        lam = self.forgetting_factor
        K = self.P @ phi / (lam + phi @ self.P @ phi)
        self.theta = self.theta + K * e
        self.P = (self.P - np.outer(K, phi @ self.P)) / lam

        # 协方差重置 (防止过小)
        if np.linalg.det(self.P) < 1e-10:
            self.P = np.eye(self.na + self.nb) * 100

    def design_controller(self):
        """极点配置设计控制器"""
        # 提取A和B多项式系数
        a = self.theta[:self.na]
        b = self.theta[self.na:]

        # 构建丢番图方程求解R和S
        # A*R + B*S = A_d (期望闭环特征多项式)

        # 简化: 使用直接配置法
        # 对于一阶系统
        if self.model_order == 1 and len(b) > 0 and abs(b[0]) > 1e-6:
            p = self.desired_poles[0]
            self.R[0] = 1.0
            self.S[0] = (p + a[0]) / b[0]
        elif self.model_order == 2:
            # 二阶系统极点配置
            p1, p2 = self.desired_poles[:2]
            a1, a2 = a[0], a[1] if len(a) > 1 else 0
            b1, b2 = b[0], b[1] if len(b) > 1 else 0

            if abs(b1) > 1e-6:
                self.R[0] = 1.0
                self.S[0] = (p1 + p2 + a1) / b1
                if len(self.S) > 1:
                    self.S[1] = (p1 * p2 + a2 - b2 * self.S[0]) / b1

    def compute_control(self, reference: float, measurement: float,
                        dt: float = 0.1) -> float:
        """
        计算控制信号

        u(t) = [r(t) - S*y(t)] / R
        """
        # 更新参数估计
        if len(self.u_buffer) > 0:
            self.update_params(measurement, self.u_buffer[-1])

        # 设计控制器
        self.design_controller()

        # 计算控制
        S_y = 0.0
        for i, s in enumerate(self.S):
            if i < len(self.y_buffer):
                S_y += s * self.y_buffer[-(i+1)]

        R_sum = sum(self.R) if sum(abs(self.R)) > 1e-6 else 1.0
        u = (reference - S_y) / R_sum

        # 限幅
        u = np.clip(u, 0, 1)

        return u

    def get_model_params(self) -> Dict:
        """获取辨识的模型参数"""
        return {
            'A': self.theta[:self.na].tolist(),
            'B': self.theta[self.na:].tolist(),
            'R': self.R.tolist(),
            'S': self.S.tolist()
        }


class GainScheduler:
    """
    增益调度器

    根据工况自动调整控制器参数
    """

    def __init__(self):
        # 调度表: {工况标识: PID参数}
        self.schedule_table: Dict[str, Dict] = {}

        # 当前工况
        self.current_regime = "normal"

        # 插值使能
        self.interpolation_enabled = True

    def add_regime(self, regime_id: str, params: Dict,
                   condition: Callable[[Dict], bool] = None):
        """
        添加工况

        Parameters:
            regime_id: 工况标识
            params: 控制器参数
            condition: 触发条件函数
        """
        self.schedule_table[regime_id] = {
            'params': params,
            'condition': condition
        }

    def get_params(self, operating_point: Dict) -> Dict:
        """
        获取当前工况的控制器参数

        Parameters:
            operating_point: 当前运行点

        Returns:
            控制器参数
        """
        # 检查各工况条件
        for regime_id, regime in self.schedule_table.items():
            condition = regime.get('condition')
            if condition and condition(operating_point):
                self.current_regime = regime_id
                return regime['params']

        # 返回默认
        if 'normal' in self.schedule_table:
            return self.schedule_table['normal']['params']

        return {'Kp': 1.0, 'Ki': 0.1, 'Kd': 0.01}


class AdaptiveGainController:
    """
    自适应增益控制器

    结合PID和自适应机制
    """

    def __init__(self, base_Kp: float = 1.0, base_Ki: float = 0.1,
                 base_Kd: float = 0.01):
        # 基础增益
        self.base_Kp = base_Kp
        self.base_Ki = base_Ki
        self.base_Kd = base_Kd

        # 自适应因子
        self.Kp_factor = 1.0
        self.Ki_factor = 1.0
        self.Kd_factor = 1.0

        # 误差历史
        self.error_history: deque = deque(maxlen=50)

        # 自适应速率
        self.adaptation_rate = 0.01

        # PID状态
        self.integral = 0.0
        self.last_error = 0.0
        self.filtered_derivative = 0.0

    def compute(self, sp: float, pv: float, dt: float = 1.0) -> float:
        """计算控制输出"""
        error = sp - pv
        self.error_history.append(error)

        # 自适应增益调整
        self._adapt_gains()

        # 当前增益
        Kp = self.base_Kp * self.Kp_factor
        Ki = self.base_Ki * self.Ki_factor
        Kd = self.base_Kd * self.Kd_factor

        # PID计算
        P = Kp * error
        self.integral += Ki * error * dt
        self.integral = np.clip(self.integral, -10, 10)

        derivative = (error - self.last_error) / dt
        self.filtered_derivative = 0.1 * derivative + 0.9 * self.filtered_derivative
        D = Kd * self.filtered_derivative

        output = P + self.integral + D
        output = np.clip(output, 0, 1)

        self.last_error = error

        return output

    def _adapt_gains(self):
        """自适应增益调整"""
        if len(self.error_history) < 10:
            return

        errors = np.array(self.error_history)

        # 计算误差特征
        error_mean = np.mean(np.abs(errors))
        error_trend = np.polyfit(np.arange(len(errors)), errors, 1)[0]
        oscillation = np.std(np.diff(errors))

        # 调整策略
        # 大误差 -> 增大Kp
        if error_mean > 0.5:
            self.Kp_factor = min(self.Kp_factor + self.adaptation_rate, 2.0)
        elif error_mean < 0.1:
            self.Kp_factor = max(self.Kp_factor - self.adaptation_rate, 0.5)

        # 持续偏差 -> 增大Ki
        if abs(errors[-1]) > 0.1 and len(errors) > 20:
            recent_avg = np.mean(errors[-20:])
            if abs(recent_avg) > 0.1:
                self.Ki_factor = min(self.Ki_factor + self.adaptation_rate, 2.0)

        # 振荡 -> 减小Kp, 增大Kd
        if oscillation > 0.2:
            self.Kp_factor = max(self.Kp_factor - self.adaptation_rate, 0.5)
            self.Kd_factor = min(self.Kd_factor + self.adaptation_rate, 2.0)

    def get_gains(self) -> Dict:
        """获取当前增益"""
        return {
            'Kp': self.base_Kp * self.Kp_factor,
            'Ki': self.base_Ki * self.Ki_factor,
            'Kd': self.base_Kd * self.Kd_factor,
            'factors': {
                'Kp_factor': self.Kp_factor,
                'Ki_factor': self.Ki_factor,
                'Kd_factor': self.Kd_factor
            }
        }

    def reset(self):
        """重置"""
        self.Kp_factor = 1.0
        self.Ki_factor = 1.0
        self.Kd_factor = 1.0
        self.integral = 0.0
        self.last_error = 0.0
        self.error_history.clear()
