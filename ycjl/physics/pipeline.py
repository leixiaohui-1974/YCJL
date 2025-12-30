"""
PCCP管道模型 - 特征线法 (MOC) 求解器
=====================================

实现有压管道瞬变流计算:
- 连续方程与动量方程
- 特征线离散化
- 阀门边界条件
- 爆管泄漏模拟
- 水锤压力波传播
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field

from ..config.settings import Config


@dataclass
class PipelineState:
    """管道状态"""
    head: np.ndarray          # 各节点测压管水头 (m)
    discharge: np.ndarray     # 各节点流量 (m³/s)
    velocity: np.ndarray      # 各节点流速 (m/s)
    max_pressure: float       # 最大压力 (m)
    min_pressure: float       # 最小压力 (m)
    inlet_head: float         # 入口水头
    outlet_head: float        # 出口水头
    inlet_flow: float         # 入口流量
    outlet_flow: float        # 出口流量


@dataclass
class ValveNode:
    """阀门节点"""
    position: int      # 节点索引
    opening: float     # 开度 (0~1)
    k_coefficient: float  # 阻力系数
    cv_value: float    # 流量系数


@dataclass
class LeakNode:
    """泄漏节点"""
    position: int      # 节点索引
    coefficient: float  # 泄漏系数
    area: float        # 泄漏面积


class PipelineMOC:
    """
    PCCP管道特征线法求解器

    控制方程:
    - 连续方程: ∂H/∂t + (a²/gA)∂Q/∂x = 0
    - 动量方程: ∂Q/∂t + gA∂H/∂x + fQ|Q|/(2DA) = 0

    特征线方程:
    - C+: H_P - H_A + B(Q_P - Q_A) + R*Q_A*|Q_A| = 0, dx/dt = +a
    - C-: H_P - H_B - B(Q_P - Q_B) - R*Q_B*|Q_B| = 0, dx/dt = -a

    其中:
    - B = a/(gA): 管道特征阻抗
    - R = f*Δx/(2gDA²): 摩阻项系数
    """

    def __init__(self, config: Optional[Config] = None):
        self.cfg = config.pipeline if config else Config.pipeline
        self.physics = Config.physics

        # 管道参数
        self.L = self.cfg.total_length
        self.D = self.cfg.diameter
        self.A = self.cfg.area
        self.a = self.cfg.wave_speed  # 波速
        self.f = self.cfg.darcy_f     # 达西摩阻系数

        # 网格参数 (满足Courant条件: dx = a * dt)
        self.dt = Config.simulation.dt
        self.dx = self.a * self.dt
        self.N = int(self.L / self.dx) + 1

        # 重新计算实际dx
        self.dx = self.L / (self.N - 1)

        # 特征线系数
        self.B = self.a / (self.physics.G * self.A)
        self.R = self.f * self.dx / (2 * self.physics.G * self.D * self.A ** 2)

        # 状态变量
        self.H = np.ones(self.N) * 50.0   # 水头 (m)
        self.Q = np.ones(self.N) * 10.0   # 流量 (m³/s)

        # 阀门节点
        self.valves: Dict[str, ValveNode] = {}
        self._setup_default_valves()

        # 泄漏节点
        self.leaks: Dict[str, LeakNode] = {}

        # 分水口节点
        self.branches: Dict[str, int] = {}
        self._setup_branches()

        # 空气阀节点
        self.air_valves: List[int] = []
        self._setup_air_valves()

        # 边界条件
        self.H_upstream = 55.0  # 上游边界水头 (调压塔)
        self.demand_flow = 10.0  # 下游需求流量

        # 历史记录
        self.history: List[PipelineState] = []

    def _setup_default_valves(self):
        """设置默认阀门"""
        # T212调流阀 (管道中部)
        mid_idx = self.N // 2
        self.valves['T212'] = ValveNode(
            position=mid_idx,
            opening=0.5,
            k_coefficient=Config.valve.t212_k_full_open,
            cv_value=Config.valve.t212_cv_max
        )

        # 末端调流阀
        end_idx = self.N - 1
        self.valves['END'] = ValveNode(
            position=end_idx,
            opening=1.0,
            k_coefficient=0.1,
            cv_value=Config.valve.end_cv_max
        )

    def _setup_branches(self):
        """设置分水口节点"""
        for i, pos in enumerate(self.cfg.branch_positions):
            idx = int(pos / self.dx)
            self.branches[f'branch_{i + 1}'] = min(idx, self.N - 1)

    def _setup_air_valves(self):
        """设置空气阀节点"""
        spacing = Config.safety.air_valve_spacing
        num = int(self.L / spacing)
        self.air_valves = [int(i * spacing / self.dx) for i in range(1, num)]

    def compute_valve_k(self, opening: float, base_k: float = 0.15) -> float:
        """
        计算阀门阻力系数 K(opening)

        K = K_full / opening² (活塞阀特性)
        """
        opening = np.clip(opening, 0.01, 1.0)
        return base_k + 0.5 * ((1.0 / opening) ** 2 - 1.0)

    def solve_valve_boundary(self, Cp: float, Cm: float,
                              K_valve: float) -> Tuple[float, float]:
        """
        牛顿迭代法求解阀门边界

        方程: (Cp - B*Q) - (Cm + B*Q) = K * Q * |Q|
        简化: Cp - Cm - 2*B*Q - K*Q*|Q| = 0
        """
        # 初值估计
        Q_guess = (Cp - Cm) / (2 * self.B + K_valve * 10)

        for _ in range(10):
            F = Cp - Cm - 2 * self.B * Q_guess - K_valve * Q_guess * abs(Q_guess)
            dF = -2 * self.B - 2 * K_valve * abs(Q_guess)

            if abs(dF) < 1e-10:
                break

            Q_new = Q_guess - F / dF
            if abs(Q_new - Q_guess) < 1e-6:
                break
            Q_guess = Q_new

        Q = Q_guess
        H_up = Cp - self.B * Q
        H_down = Cm + self.B * Q

        return Q, (H_up + H_down) / 2

    def solve_leak_boundary(self, Cp: float, Cm: float,
                             leak_coeff: float) -> Tuple[float, float, float]:
        """
        求解泄漏点边界

        Q_in = Q_out + Q_leak
        Q_leak = Cd * A * sqrt(2*g*H)
        """
        # 简化: 假设泄漏较小
        H_node = (Cp + Cm) / 2
        Q_leak = leak_coeff * np.sqrt(max(H_node, 0))

        Q_in = (Cp - H_node) / self.B
        Q_out = Q_in - Q_leak

        return Q_in, Q_out, Q_leak

    def compute_Cp_Cm(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算特征线系数 C+ 和 C-

        C+ = H_A + B*Q_A - R*Q_A*|Q_A|  (从上游点A沿dx/dt=+a传来)
        C- = H_B - B*Q_B + R*Q_B*|Q_B|  (从下游点B沿dx/dt=-a传来)
        """
        Cp = np.zeros(self.N)
        Cm = np.zeros(self.N)

        # C+ 从上游传来 (i-1 -> i)
        Cp[1:] = self.H[:-1] + self.B * self.Q[:-1] - \
                 self.R * self.Q[:-1] * np.abs(self.Q[:-1])

        # C- 从下游传来 (i+1 -> i)
        Cm[:-1] = self.H[1:] - self.B * self.Q[1:] + \
                  self.R * self.Q[1:] * np.abs(self.Q[1:])

        return Cp, Cm

    def step(self, dt: float = None, H_upstream: float = None,
             demand_flow: float = None) -> PipelineState:
        """
        推进一个时间步

        Parameters:
            dt: 时间步长 (如果与构造时不同需要重新计算网格)
            H_upstream: 上游边界水头
            demand_flow: 下游需求流量

        Returns:
            PipelineState: 当前状态
        """
        if H_upstream is not None:
            self.H_upstream = H_upstream
        if demand_flow is not None:
            self.demand_flow = demand_flow

        # 计算特征线系数
        Cp, Cm = self.compute_Cp_Cm()

        # 新时刻状态
        H_new = np.zeros(self.N)
        Q_new = np.zeros(self.N)

        # ========== 内部节点 ==========
        for i in range(1, self.N - 1):
            # 检查是否是阀门节点
            is_valve = False
            for name, valve in self.valves.items():
                if valve.position == i:
                    is_valve = True
                    K_valve = self.compute_valve_k(valve.opening, valve.k_coefficient)
                    Q_new[i], H_new[i] = self.solve_valve_boundary(
                        Cp[i], Cm[i], K_valve
                    )
                    break

            # 检查是否是泄漏节点
            is_leak = False
            for name, leak in self.leaks.items():
                if leak.position == i:
                    is_leak = True
                    Q_in, Q_out, _ = self.solve_leak_boundary(
                        Cp[i], Cm[i], leak.coefficient
                    )
                    Q_new[i] = Q_out
                    H_new[i] = (Cp[i] + Cm[i]) / 2
                    break

            # 普通内部节点
            if not is_valve and not is_leak:
                H_new[i] = (Cp[i] + Cm[i]) / 2
                Q_new[i] = (Cp[i] - Cm[i]) / (2 * self.B)

        # ========== 上游边界 (调压塔/稳流池) ==========
        # 恒定水头边界
        H_new[0] = self.H_upstream
        Q_new[0] = (H_new[0] - Cm[0]) / self.B

        # ========== 下游边界 ==========
        # 处理末端阀门
        end_valve = self.valves.get('END')
        if end_valve:
            Cv_eff = end_valve.opening * end_valve.cv_value

            if Cv_eff > 1e-4 and Cp[-1] > 0:
                # 求解: Q = Cv * sqrt(H), H = Cp - B*Q
                # B*Q + (1/Cv²)*Q² = Cp
                coeff_a = 1.0 / (Cv_eff ** 2)
                coeff_b = self.B
                coeff_c = -Cp[-1]

                discriminant = coeff_b ** 2 - 4 * coeff_a * coeff_c
                if discriminant > 0:
                    Q_avail = (-coeff_b + np.sqrt(discriminant)) / (2 * coeff_a)
                else:
                    Q_avail = 0.0

                Q_new[-1] = min(Q_avail, self.demand_flow)
                H_new[-1] = Cp[-1] - self.B * Q_new[-1]
            else:
                Q_new[-1] = 0.0
                H_new[-1] = Cp[-1]
        else:
            # 无阀门，恒定流量边界
            Q_new[-1] = self.demand_flow
            H_new[-1] = Cp[-1] - self.B * Q_new[-1]

        # ========== 空气阀节点处理 ==========
        for idx in self.air_valves:
            if 0 < idx < self.N - 1:
                # 如果压力低于大气压，空气阀打开
                if H_new[idx] < Config.physics.PATM:
                    # 简化处理：限制负压
                    H_new[idx] = max(H_new[idx], -Config.physics.PATM * 0.5)

        # 更新状态
        self.H = H_new
        self.Q = Q_new

        # 计算流速
        velocity = self.Q / self.A

        state = PipelineState(
            head=self.H.copy(),
            discharge=self.Q.copy(),
            velocity=velocity,
            max_pressure=np.max(self.H),
            min_pressure=np.min(self.H),
            inlet_head=self.H[0],
            outlet_head=self.H[-1],
            inlet_flow=self.Q[0],
            outlet_flow=self.Q[-1]
        )

        self.history.append(state)
        return state

    def set_valve_opening(self, name: str, opening: float):
        """设置阀门开度"""
        if name in self.valves:
            self.valves[name].opening = np.clip(opening, 0.0, 1.0)

    def add_leak(self, name: str, position: float, coefficient: float):
        """添加泄漏点"""
        idx = int(position / self.dx)
        idx = np.clip(idx, 1, self.N - 2)
        self.leaks[name] = LeakNode(
            position=idx,
            coefficient=coefficient,
            area=0.01  # 默认泄漏面积
        )

    def remove_leak(self, name: str):
        """移除泄漏点"""
        if name in self.leaks:
            del self.leaks[name]

    def get_head_at(self, position: float) -> float:
        """获取指定位置的水头"""
        idx = int(position / self.dx)
        idx = np.clip(idx, 0, self.N - 1)
        return self.H[idx]

    def get_flow_at(self, position: float) -> float:
        """获取指定位置的流量"""
        idx = int(position / self.dx)
        idx = np.clip(idx, 0, self.N - 1)
        return self.Q[idx]

    def get_valve_state(self, name: str) -> Optional[dict]:
        """获取阀门状态"""
        if name not in self.valves:
            return None

        valve = self.valves[name]
        H = self.H[valve.position]
        Q = self.Q[valve.position]

        return {
            'position': valve.position * self.dx,
            'opening': valve.opening,
            'head': H,
            'flow': Q,
            'head_loss': self.compute_valve_k(valve.opening) * Q * abs(Q) / (2 * self.physics.G * self.A ** 2)
        }

    def detect_water_hammer(self, threshold: float = 30.0) -> List[int]:
        """检测水锤位置"""
        if len(self.history) < 2:
            return []

        prev_H = self.history[-2].head
        curr_H = self.history[-1].head

        delta_H = np.abs(curr_H - prev_H)
        hammer_nodes = np.where(delta_H > threshold)[0]

        return hammer_nodes.tolist()

    def get_hydraulic_grade_line(self) -> np.ndarray:
        """获取水力坡度线"""
        x = np.linspace(0, self.L, self.N)
        return np.column_stack([x, self.H])

    def update_friction(self, new_f: float, section: Tuple[int, int] = None):
        """更新摩阻系数"""
        self.f = new_f
        self.R = self.f * self.dx / (2 * self.physics.G * self.D * self.A ** 2)

    def reset(self, H_init: float = 50.0, Q_init: float = 10.0):
        """重置状态"""
        self.H = np.ones(self.N) * H_init
        self.Q = np.ones(self.N) * Q_init

        # 重置阀门
        for valve in self.valves.values():
            valve.opening = 0.5

        # 清除泄漏
        self.leaks.clear()

        self.history.clear()

    def get_state_vector(self) -> np.ndarray:
        """获取状态向量"""
        return np.concatenate([self.H, self.Q])

    def set_state_vector(self, state: np.ndarray):
        """设置状态向量"""
        self.H = state[:self.N]
        self.Q = state[self.N:]

# 向后兼容别名
Pipeline = PipelineMOC
