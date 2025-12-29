"""
状态观测器
==========

用于估计不可直接测量的系统状态:
- 隧洞内部水位分布
- 管道压力波位置
- 泄漏位置检测
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from ..config.settings import Config


@dataclass
class ObserverState:
    """观测器状态"""
    estimated_state: np.ndarray   # 估计状态向量
    estimation_error: np.ndarray  # 估计误差
    confidence: float             # 置信度
    innovation: np.ndarray        # 新息


class StateObserver:
    """
    Luenberger状态观测器

    用于隧洞和管道的分布式状态估计
    """

    def __init__(self, num_states: int, num_outputs: int):
        self.n = num_states
        self.m = num_outputs

        # 系统矩阵 (将由具体模型设置)
        self.A = np.eye(num_states)
        self.B = np.zeros((num_states, 1))
        self.C = np.zeros((num_outputs, num_states))

        # 观测器增益
        self.L = np.zeros((num_states, num_outputs))

        # 状态
        self.x_hat = np.zeros(num_states)

        # 设计参数
        self.poles_ratio = 2.0  # 观测器极点相对系统极点的倍数

        # 历史
        self.history: List[np.ndarray] = []

    def design_observer_gain(self, system_poles: np.ndarray):
        """
        设计观测器增益

        使观测器极点位于系统极点的左边 (更快收敛)
        """
        observer_poles = system_poles * self.poles_ratio

        # 极点配置 (简化版 - 使用Ackermann公式思路)
        # 实际工程中可使用scipy.signal.place_poles

        # 简化处理: 假设系统可观，使用固定增益
        self.L = np.ones((self.n, self.m)) * 0.1

    def set_system_matrices(self, A: np.ndarray, B: np.ndarray, C: np.ndarray):
        """设置系统矩阵"""
        self.A = A
        self.B = B
        self.C = C

    def update(self, y_measured: np.ndarray, u: np.ndarray,
               dt: float) -> ObserverState:
        """
        更新状态估计

        dx_hat/dt = A*x_hat + B*u + L*(y - C*x_hat)

        Parameters:
            y_measured: 实测输出
            u: 控制输入
            dt: 时间步长

        Returns:
            ObserverState: 估计结果
        """
        # 输出估计
        y_hat = self.C @ self.x_hat

        # 新息 (测量残差)
        innovation = y_measured - y_hat

        # 状态更新
        dx = self.A @ self.x_hat + self.B @ u + self.L @ innovation
        self.x_hat = self.x_hat + dx * dt

        # 估计误差 (基于新息)
        error = np.abs(innovation)

        # 置信度
        confidence = 1.0 / (1.0 + np.mean(error))

        self.history.append(self.x_hat.copy())

        return ObserverState(
            estimated_state=self.x_hat.copy(),
            estimation_error=error,
            confidence=confidence,
            innovation=innovation
        )

    def get_state_at(self, index: int) -> float:
        """获取指定位置的状态"""
        if 0 <= index < self.n:
            return self.x_hat[index]
        return 0.0

    def reset(self, initial_state: np.ndarray = None):
        """重置"""
        if initial_state is not None:
            self.x_hat = initial_state.copy()
        else:
            self.x_hat = np.zeros(self.n)
        self.history.clear()


class TunnelStateObserver(StateObserver):
    """
    隧洞状态观测器

    估计沿程水位和流量分布
    """

    def __init__(self, num_nodes: int = 50):
        # 状态: [h1, h2, ..., hn, Q1, Q2, ..., Qn]
        super().__init__(num_states=2 * num_nodes, num_outputs=6)

        self.num_nodes = num_nodes
        self.cfg = Config.tunnel

        # 监测点位置 (6个)
        self.monitor_indices = np.linspace(0, num_nodes - 1, 6).astype(int)

        # 初始化输出矩阵
        self._setup_output_matrix()

        # 初始化状态
        self.x_hat[:num_nodes] = 3.5  # 初始水深
        self.x_hat[num_nodes:] = 10.0  # 初始流量

    def _setup_output_matrix(self):
        """设置输出矩阵"""
        # 只观测监测点的水位
        self.C = np.zeros((6, 2 * self.num_nodes))
        for i, idx in enumerate(self.monitor_indices):
            self.C[i, idx] = 1.0

        # 观测器增益 (简化)
        self.L = np.zeros((2 * self.num_nodes, 6))
        for i, idx in enumerate(self.monitor_indices):
            # 水位观测影响附近节点
            for j in range(max(0, idx - 5), min(self.num_nodes, idx + 6)):
                weight = 1.0 - abs(j - idx) / 6.0
                self.L[j, i] = weight * 0.5

    def get_depth_profile(self) -> np.ndarray:
        """获取水深分布"""
        return self.x_hat[:self.num_nodes]

    def get_flow_profile(self) -> np.ndarray:
        """获取流量分布"""
        return self.x_hat[self.num_nodes:]

    def detect_wave_position(self) -> Optional[float]:
        """检测波前位置"""
        depths = self.get_depth_profile()
        gradients = np.abs(np.diff(depths))

        if np.max(gradients) > 0.05:
            wave_idx = np.argmax(gradients)
            return wave_idx * self.cfg.dx
        return None


class LeakDetector:
    """
    泄漏检测器

    基于流量差和压力波形分析
    """

    def __init__(self, num_sections: int = 10):
        self.num_sections = num_sections
        self.cfg = Config.pipeline

        # 历史数据
        self.flow_balance_history: List[np.ndarray] = []
        self.pressure_history: List[np.ndarray] = []

        # 检测阈值
        self.flow_imbalance_threshold = 0.5  # m³/s
        self.pressure_gradient_threshold = 5.0  # m/km

        # 状态
        self.leak_detected = False
        self.leak_position = None
        self.leak_magnitude = 0.0

    def update(self, flows: np.ndarray, pressures: np.ndarray) -> Dict:
        """
        更新检测

        Parameters:
            flows: 各断面流量
            pressures: 各断面压力

        Returns:
            检测结果
        """
        self.flow_balance_history.append(flows.copy())
        self.pressure_history.append(pressures.copy())

        # 保持历史长度
        max_history = 100
        if len(self.flow_balance_history) > max_history:
            self.flow_balance_history.pop(0)
            self.pressure_history.pop(0)

        # 流量不平衡检测
        if len(flows) > 1:
            flow_diff = np.diff(flows)
            max_imbalance = np.max(np.abs(flow_diff))

            if max_imbalance > self.flow_imbalance_threshold:
                self.leak_detected = True
                leak_section = np.argmax(np.abs(flow_diff))
                self.leak_position = (leak_section + 0.5) * self.cfg.total_length / self.num_sections
                self.leak_magnitude = flow_diff[leak_section]
            else:
                self.leak_detected = False
                self.leak_position = None
                self.leak_magnitude = 0.0

        # 压力梯度异常检测
        pressure_gradient_anomaly = False
        if len(pressures) > 2:
            gradients = np.diff(pressures) / (self.cfg.total_length / self.num_sections / 1000)
            if np.any(np.abs(gradients) > self.pressure_gradient_threshold):
                pressure_gradient_anomaly = True

        return {
            'leak_detected': self.leak_detected,
            'leak_position': self.leak_position,
            'leak_magnitude': self.leak_magnitude,
            'pressure_anomaly': pressure_gradient_anomaly
        }

    def get_leak_confidence(self) -> float:
        """获取泄漏检测置信度"""
        if not self.leak_detected:
            return 0.0

        if len(self.flow_balance_history) < 10:
            return 0.3

        # 持续性检测
        recent_imbalances = []
        for flows in self.flow_balance_history[-10:]:
            if len(flows) > 1:
                recent_imbalances.append(np.max(np.abs(np.diff(flows))))

        # 如果持续存在不平衡，置信度增加
        persistent_count = sum(1 for imb in recent_imbalances
                               if imb > self.flow_imbalance_threshold)
        return min(persistent_count / 10.0, 1.0)

    def reset(self):
        """重置"""
        self.flow_balance_history.clear()
        self.pressure_history.clear()
        self.leak_detected = False
        self.leak_position = None
        self.leak_magnitude = 0.0
