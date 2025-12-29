"""
无压隧洞模型 - 圣维南方程求解器
================================

实现扩散波近似的圣维南方程数值求解:
- 连续方程: ∂A/∂t + ∂Q/∂x = 0
- 动量方程: ∂Q/∂t + ∂(Q²/A)/∂x + gA(∂Z/∂x + Sf) = 0

采用MacCormack格式进行离散求解，支持:
- 冰期糙率变化
- 下游顶托（稳流池水位反馈）
- 重力波传播与衰减
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass, field

from ..config.settings import Config


@dataclass
class TunnelState:
    """隧洞状态"""
    water_depth: np.ndarray       # 各节点水深 (m)
    discharge: np.ndarray         # 各节点流量 (m³/s)
    velocity: np.ndarray          # 各节点流速 (m/s)
    inlet_flow: float            # 入口流量
    outlet_flow: float           # 出口流量
    max_depth: float             # 最大水深
    wave_position: float         # 波前位置估计


class TunnelSolver:
    """
    无压隧洞圣维南方程求解器

    采用扩散波近似 + MacCormack格式:
    - 预测步: 前向差分
    - 校正步: 后向差分
    - 人工粘性项确保稳定性
    """

    def __init__(self, config: Optional[Config] = None):
        self.cfg = config.tunnel if config else Config.tunnel
        self.physics = Config.physics

        # 网格参数
        self.N = self.cfg.num_nodes
        self.dx = self.cfg.dx
        self.L = self.cfg.total_length

        # 断面参数
        self.width = self.cfg.width
        self.height = self.cfg.height

        # 水力参数
        self.n = self.cfg.manning_n_normal  # 曼宁糙率
        self.S0 = self.cfg.bottom_slope      # 底坡

        # 状态变量
        self.h = np.ones(self.N) * 3.5      # 初始水深 (m)
        self.Q = np.ones(self.N) * 10.0     # 初始流量 (m³/s)
        self.A = self.h * self.width        # 过水断面积

        # 边界条件
        self.Q_inlet = 10.0                  # 入口流量
        self.H_outlet = 5.0                  # 出口水位 (稳流池水位)

        # 监测断面
        self.monitor_positions = [0.25, 0.5, 0.75]  # 相对位置
        self.monitor_indices = [int(p * self.N) for p in self.monitor_positions]

        # 历史记录
        self.history: List[TunnelState] = []

        # 糙率变化记录（用于冰期）
        self.n_profile = np.ones(self.N) * self.n

    def _compute_area(self, h: np.ndarray) -> np.ndarray:
        """计算过水断面积（城门洞型简化为矩形）"""
        return h * self.width

    def _compute_wetted_perimeter(self, h: np.ndarray) -> np.ndarray:
        """计算湿周"""
        return self.width + 2 * h

    def _compute_hydraulic_radius(self, h: np.ndarray) -> np.ndarray:
        """计算水力半径"""
        A = self._compute_area(h)
        P = self._compute_wetted_perimeter(h)
        return np.where(P > 0, A / P, 0)

    def _compute_friction_slope(self, Q: np.ndarray, A: np.ndarray,
                                 R: np.ndarray) -> np.ndarray:
        """
        计算摩阻坡度 Sf

        曼宁公式: Q = (1/n) * A * R^(2/3) * Sf^(1/2)
        => Sf = (Q * n)^2 / (A^2 * R^(4/3))
        """
        # 防止除零
        A_safe = np.maximum(A, 1e-6)
        R_safe = np.maximum(R, 1e-6)

        Sf = (Q * self.n_profile) ** 2 / (A_safe ** 2 * R_safe ** (4 / 3))
        return Sf

    def _compute_wave_celerity(self, h: np.ndarray) -> np.ndarray:
        """计算重力波波速 c = sqrt(g*h)"""
        return np.sqrt(self.physics.G * np.maximum(h, 0.1))

    def step(self, dt: float, Q_inlet: float = None,
             H_outlet: float = None) -> TunnelState:
        """
        推进一个时间步 (MacCormack格式)

        Parameters:
            dt: 时间步长 (s)
            Q_inlet: 入口流量 (m³/s)
            H_outlet: 出口水位 (m) - 稳流池水位

        Returns:
            TunnelState: 当前状态
        """
        if Q_inlet is not None:
            self.Q_inlet = Q_inlet
        if H_outlet is not None:
            self.H_outlet = H_outlet

        # 保存当前值
        h_old = self.h.copy()
        Q_old = self.Q.copy()

        # 计算水力参数
        A = self._compute_area(self.h)
        R = self._compute_hydraulic_radius(self.h)
        Sf = self._compute_friction_slope(self.Q, A, R)

        # ========== 预测步 (前向差分) ==========
        h_pred = np.zeros(self.N)
        Q_pred = np.zeros(self.N)

        # 内部节点
        for i in range(1, self.N - 1):
            # 连续方程
            dQ_dx = (self.Q[i + 1] - self.Q[i]) / self.dx
            h_pred[i] = self.h[i] - dt * dQ_dx / self.width

            # 动量方程
            dFlux_dx = ((self.Q[i + 1] ** 2 / A[i + 1]) -
                        (self.Q[i] ** 2 / A[i])) / self.dx
            dh_dx = (self.h[i + 1] - self.h[i]) / self.dx

            # 源项：重力 - 摩阻
            source = self.physics.G * A[i] * (self.S0 - Sf[i] - dh_dx)

            Q_pred[i] = self.Q[i] - dt * dFlux_dx + dt * source

        # ========== 边界条件 (预测) ==========
        # 上游边界: 给定流量
        Q_pred[0] = self.Q_inlet
        # 正常水深估计
        A_inlet = self.Q_inlet / np.sqrt(self.S0) * self.n
        h_pred[0] = A_inlet / self.width

        # 下游边界: 考虑稳流池顶托
        h_critical = (self.Q[-1] ** 2 / (self.physics.G * self.width ** 2)) ** (1 / 3)
        h_boundary = max(self.H_outlet, h_critical)
        h_pred[-1] = h_boundary
        Q_pred[-1] = Q_pred[-2]  # 零梯度

        # 确保h_pred为正
        h_pred = np.maximum(h_pred, 0.1)

        # ========== 校正步 (后向差分) ==========
        A_pred = self._compute_area(h_pred)
        R_pred = self._compute_hydraulic_radius(h_pred)
        Sf_pred = self._compute_friction_slope(Q_pred, A_pred, R_pred)

        for i in range(1, self.N - 1):
            # 连续方程
            dQ_dx = (Q_pred[i] - Q_pred[i - 1]) / self.dx
            h_corr = h_pred[i] - dt * dQ_dx / self.width

            # 动量方程
            dFlux_dx = ((Q_pred[i] ** 2 / A_pred[i]) -
                        (Q_pred[i - 1] ** 2 / A_pred[i - 1])) / self.dx
            dh_dx = (h_pred[i] - h_pred[i - 1]) / self.dx

            source = self.physics.G * A_pred[i] * (self.S0 - Sf_pred[i] - dh_dx)

            Q_corr = Q_pred[i] - dt * dFlux_dx + dt * source

            # MacCormack平均
            self.h[i] = 0.5 * (h_old[i] + h_corr)
            self.Q[i] = 0.5 * (Q_old[i] + Q_corr)

        # ========== 边界条件 (最终) ==========
        self.Q[0] = self.Q_inlet
        h_normal = (self.Q_inlet * self.n / (self.width * np.sqrt(self.S0))) ** (3 / 5)
        self.h[0] = max(h_normal, 1.0)

        self.h[-1] = 0.8 * self.h[-1] + 0.2 * h_boundary
        self.Q[-1] = self.Q[-2]

        # 确保物理有效性
        self.h = np.maximum(self.h, 0.1)
        self.Q = np.maximum(self.Q, 0.0)

        # 人工粘性（提高稳定性）
        nu = 0.1
        self.h[1:-1] += nu * (self.h[:-2] - 2 * self.h[1:-1] + self.h[2:])
        self.Q[1:-1] += nu * (self.Q[:-2] - 2 * self.Q[1:-1] + self.Q[2:])

        # 更新断面积
        self.A = self._compute_area(self.h)

        # 计算流速
        velocity = np.where(self.A > 0, self.Q / self.A, 0)

        state = TunnelState(
            water_depth=self.h.copy(),
            discharge=self.Q.copy(),
            velocity=velocity,
            inlet_flow=self.Q[0],
            outlet_flow=self.Q[-1],
            max_depth=np.max(self.h),
            wave_position=self._estimate_wave_position()
        )

        self.history.append(state)
        return state

    def _estimate_wave_position(self) -> float:
        """估计波前位置（基于水深梯度）"""
        dh = np.abs(np.diff(self.h))
        if np.max(dh) < 0.01:
            return self.L  # 无明显波

        wave_idx = np.argmax(dh)
        return wave_idx * self.dx

    def set_manning_n(self, n: float, section: Tuple[float, float] = None):
        """
        设置曼宁糙率

        Parameters:
            n: 糙率值
            section: 区间 (start, end) 以相对位置表示，None表示全线
        """
        if section is None:
            self.n = n
            self.n_profile[:] = n
        else:
            start_idx = int(section[0] * self.N)
            end_idx = int(section[1] * self.N)
            self.n_profile[start_idx:end_idx] = n

    def set_ice_mode(self, enable: bool):
        """设置冰期模式"""
        if enable:
            self.n = Config.tunnel.manning_n_ice
            self.n_profile[:] = self.n
        else:
            self.n = Config.tunnel.manning_n_normal
            self.n_profile[:] = self.n

    def get_depth_at(self, position: float) -> float:
        """获取指定位置的水深"""
        idx = int(position / self.dx)
        idx = np.clip(idx, 0, self.N - 1)
        return self.h[idx]

    def get_flow_at(self, position: float) -> float:
        """获取指定位置的流量"""
        idx = int(position / self.dx)
        idx = np.clip(idx, 0, self.N - 1)
        return self.Q[idx]

    def get_monitor_data(self) -> dict:
        """获取监测断面数据"""
        data = {}
        for i, pos in enumerate(self.monitor_positions):
            idx = self.monitor_indices[i]
            data[f'section_{i + 1}'] = {
                'position': pos * self.L,
                'depth': self.h[idx],
                'flow': self.Q[idx],
                'velocity': self.Q[idx] / (self.h[idx] * self.width) if self.h[idx] > 0 else 0
            }
        return data

    def check_overflow_risk(self) -> List[int]:
        """检查溢流风险节点"""
        critical_depth = self.height * 0.9  # 90%洞高
        risk_nodes = np.where(self.h > critical_depth)[0]
        return risk_nodes.tolist()

    def reset(self, initial_depth: float = 3.5, initial_flow: float = 10.0):
        """重置隧洞状态"""
        self.h = np.ones(self.N) * initial_depth
        self.Q = np.ones(self.N) * initial_flow
        self.A = self._compute_area(self.h)
        self.n = Config.tunnel.manning_n_normal
        self.n_profile[:] = self.n
        self.history.clear()

    def get_state_vector(self) -> np.ndarray:
        """获取状态向量（用于状态同化）"""
        return np.concatenate([self.h, self.Q])

    def set_state_vector(self, state: np.ndarray):
        """设置状态向量（用于状态同化）"""
        self.h = state[:self.N]
        self.Q = state[self.N:]
        self.A = self._compute_area(self.h)
