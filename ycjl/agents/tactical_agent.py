"""
L2战术层智能体
==============

基于MPC的管段优化控制:
- 局部MPC控制器
- 滚动时域优化
- 约束处理
- 与L1/L3层协调
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import time
from scipy.optimize import minimize, LinearConstraint

from .base_agent import (
    BaseAgent, AgentPriority, AgentState, AgentMessage,
    ControlAction, MessageType
)
from ..config.settings import Config


class MPCStatus(Enum):
    """MPC状态"""
    IDLE = auto()
    SOLVING = auto()
    OPTIMAL = auto()
    SUBOPTIMAL = auto()
    INFEASIBLE = auto()
    TIMEOUT = auto()


@dataclass
class MPCConfig:
    """MPC配置"""
    horizon: int = 20           # 预测时域
    control_horizon: int = 5    # 控制时域
    dt: float = 1.0             # 采样周期

    # 权重
    Q: np.ndarray = None        # 状态权重
    R: np.ndarray = None        # 控制权重
    P: np.ndarray = None        # 终端权重

    # 约束
    u_min: np.ndarray = None    # 控制下界
    u_max: np.ndarray = None    # 控制上界
    du_max: np.ndarray = None   # 控制增量限制

    x_min: np.ndarray = None    # 状态下界
    x_max: np.ndarray = None    # 状态上界

    # 求解器设置
    max_iter: int = 100
    tolerance: float = 1e-4
    timeout: float = 0.5        # 秒


@dataclass
class MPCResult:
    """MPC求解结果"""
    status: MPCStatus
    u_optimal: np.ndarray       # 最优控制序列
    x_predicted: np.ndarray     # 预测状态序列
    cost: float                 # 代价值
    solve_time: float          # 求解时间
    iterations: int             # 迭代次数


class MPCController:
    """
    模型预测控制器

    基于线性化模型的MPC:
    - 二次规划求解
    - 软约束处理
    - 参考轨迹跟踪
    """

    def __init__(self, config: MPCConfig = None):
        self.cfg = config or MPCConfig()

        # 模型维度
        self.nx = 4  # 状态维度 [level, flow, pressure, velocity]
        self.nu = 2  # 控制维度 [valve1, valve2]
        self.ny = 2  # 输出维度 [level, flow]

        # 初始化默认权重
        if self.cfg.Q is None:
            self.cfg.Q = np.diag([10.0, 5.0, 1.0, 1.0])
        if self.cfg.R is None:
            self.cfg.R = np.diag([0.1, 0.1])
        if self.cfg.P is None:
            self.cfg.P = self.cfg.Q * 10

        # 初始化约束
        if self.cfg.u_min is None:
            self.cfg.u_min = np.zeros(self.nu)
        if self.cfg.u_max is None:
            self.cfg.u_max = np.ones(self.nu)
        if self.cfg.du_max is None:
            self.cfg.du_max = np.ones(self.nu) * 0.1

        # 线性模型 x(k+1) = A*x(k) + B*u(k)
        self.A = np.eye(self.nx)
        self.B = np.zeros((self.nx, self.nu))
        self.C = np.zeros((self.ny, self.nx))

        # 工作点
        self.x_op = np.zeros(self.nx)
        self.u_op = np.zeros(self.nu)

        # 参考轨迹
        self.x_ref = np.zeros((self.cfg.horizon, self.nx))

        # 状态
        self.status = MPCStatus.IDLE
        self.last_result: Optional[MPCResult] = None

    def set_linear_model(self, A: np.ndarray, B: np.ndarray, C: np.ndarray = None):
        """设置线性模型"""
        self.A = A.copy()
        self.B = B.copy()
        if C is not None:
            self.C = C.copy()

    def linearize_at(self, x_op: np.ndarray, u_op: np.ndarray,
                     nonlinear_model: callable):
        """在工作点处线性化"""
        self.x_op = x_op.copy()
        self.u_op = u_op.copy()

        eps = 1e-6

        # 数值雅可比
        f0 = nonlinear_model(x_op, u_op)

        # A = df/dx
        for i in range(self.nx):
            x_plus = x_op.copy()
            x_plus[i] += eps
            f_plus = nonlinear_model(x_plus, u_op)
            self.A[:, i] = (f_plus - f0) / eps

        # B = df/du
        for i in range(self.nu):
            u_plus = u_op.copy()
            u_plus[i] += eps
            f_plus = nonlinear_model(x_op, u_plus)
            self.B[:, i] = (f_plus - f0) / eps

    def set_reference(self, x_ref: np.ndarray):
        """设置参考轨迹"""
        if x_ref.ndim == 1:
            # 常值参考
            self.x_ref = np.tile(x_ref, (self.cfg.horizon, 1))
        else:
            self.x_ref = x_ref.copy()

    def _build_qp_matrices(self, x0: np.ndarray, u_prev: np.ndarray) -> Tuple:
        """构建QP问题矩阵"""
        N = self.cfg.horizon
        M = self.cfg.control_horizon
        nx, nu = self.nx, self.nu

        # 预测矩阵
        # X = Phi * x0 + Gamma * U
        Phi = np.zeros((N * nx, nx))
        Gamma = np.zeros((N * nx, M * nu))

        A_power = np.eye(nx)
        for k in range(N):
            A_power = A_power @ self.A
            Phi[k*nx:(k+1)*nx, :] = A_power

            for j in range(min(k+1, M)):
                A_power_j = np.linalg.matrix_power(self.A, k-j)
                Gamma[k*nx:(k+1)*nx, j*nu:(j+1)*nu] = A_power_j @ self.B

        # 代价矩阵
        Q_bar = np.kron(np.eye(N), self.cfg.Q)
        Q_bar[-nx:, -nx:] = self.cfg.P  # 终端代价

        R_bar = np.kron(np.eye(M), self.cfg.R)

        # QP: min 0.5 * U' * H * U + f' * U
        H = Gamma.T @ Q_bar @ Gamma + R_bar
        H = 0.5 * (H + H.T)  # 保证对称

        # 参考轨迹偏差
        x_ref_flat = self.x_ref[:N, :].flatten()
        x0_effect = Phi @ x0

        f = Gamma.T @ Q_bar @ (x0_effect - x_ref_flat)

        return H, f, Phi, Gamma

    def _add_rate_constraint(self, u_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """添加控制增量约束"""
        M = self.cfg.control_horizon
        nu = self.nu

        # Delta矩阵: du = D * U - d0
        D = np.zeros((M * nu, M * nu))
        for k in range(M):
            D[k*nu:(k+1)*nu, k*nu:(k+1)*nu] = np.eye(nu)
            if k > 0:
                D[k*nu:(k+1)*nu, (k-1)*nu:k*nu] = -np.eye(nu)

        d0 = np.zeros(M * nu)
        d0[:nu] = u_prev

        # -du_max <= du <= du_max
        lb = -np.tile(self.cfg.du_max, M)
        ub = np.tile(self.cfg.du_max, M)

        return D, d0, lb, ub

    def solve(self, x0: np.ndarray, u_prev: np.ndarray = None) -> MPCResult:
        """
        求解MPC问题

        Parameters:
            x0: 当前状态
            u_prev: 上一步控制输入

        Returns:
            MPCResult: 求解结果
        """
        self.status = MPCStatus.SOLVING
        start_time = time.time()

        if u_prev is None:
            u_prev = self.u_op.copy()

        N = self.cfg.horizon
        M = self.cfg.control_horizon
        nu = self.nu

        # 构建QP
        H, f, Phi, Gamma = self._build_qp_matrices(x0, u_prev)

        # 约束
        # 控制约束
        u_lb = np.tile(self.cfg.u_min, M)
        u_ub = np.tile(self.cfg.u_max, M)

        # 控制增量约束
        D, d0, du_lb, du_ub = self._add_rate_constraint(u_prev)

        # 初始猜测
        u0 = np.tile(u_prev, M)

        # 代价函数
        def cost(U):
            return 0.5 * U @ H @ U + f @ U

        def grad(U):
            return H @ U + f

        # 约束
        bounds = [(u_lb[i], u_ub[i]) for i in range(M * nu)]

        # 控制增量约束
        linear_constraint = LinearConstraint(
            D, du_lb + d0, du_ub + d0
        )

        # 求解
        try:
            result = minimize(
                cost,
                u0,
                method='SLSQP',
                jac=grad,
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': lambda U: du_ub + d0 - D @ U},
                options={'maxiter': self.cfg.max_iter, 'ftol': self.cfg.tolerance}
            )

            solve_time = time.time() - start_time
            u_optimal = result.x.reshape(M, nu)

            if result.success:
                self.status = MPCStatus.OPTIMAL
            else:
                self.status = MPCStatus.SUBOPTIMAL

            # 预测状态
            x_pred = np.zeros((N, self.nx))
            x_pred[0] = x0
            for k in range(1, N):
                u_k = u_optimal[min(k-1, M-1)]
                x_pred[k] = self.A @ x_pred[k-1] + self.B @ u_k

            mpc_result = MPCResult(
                status=self.status,
                u_optimal=u_optimal,
                x_predicted=x_pred,
                cost=result.fun,
                solve_time=solve_time,
                iterations=result.nit
            )

        except Exception as e:
            solve_time = time.time() - start_time
            self.status = MPCStatus.INFEASIBLE

            mpc_result = MPCResult(
                status=self.status,
                u_optimal=np.tile(u_prev, (M, 1)),
                x_predicted=np.zeros((N, self.nx)),
                cost=float('inf'),
                solve_time=solve_time,
                iterations=0
            )

        self.last_result = mpc_result
        return mpc_result


class TacticalAgent(BaseAgent):
    """
    L2战术层智能体

    负责管段级的MPC优化控制:
    - 局部优化
    - 约束处理
    - 与L1协调 (接受覆盖)
    - 与L3协调 (接受目标)
    """

    def __init__(self, agent_id: str = "L2_tactical", segment_id: int = 0):
        super().__init__(agent_id, AgentPriority.TACTICAL)

        self.segment_id = segment_id
        self.cfg = Config

        # MPC控制器
        mpc_config = MPCConfig(
            horizon=20,
            control_horizon=5,
            dt=1.0,
            Q=np.diag([10.0, 5.0, 1.0, 1.0]),
            R=np.diag([0.1, 0.1])
        )
        self.mpc = MPCController(mpc_config)

        # 控制目标 (从L3接收)
        self.target_flow = 10.0
        self.target_level = 5.0

        # 本地状态
        self.x = np.zeros(4)  # [level, flow, pressure, velocity]
        self.u = np.zeros(2)  # [valve1, valve2]

        # L1覆盖标志
        self.l1_override = False
        self.l1_override_actions: List[ControlAction] = []

        # 协调参数 (ADMM)
        self.lambda_coord = np.zeros(2)  # 拉格朗日乘子
        self.z_shared = np.zeros(2)      # 共享变量
        self.rho = 1.0                   # ADMM惩罚参数

        # 注册消息处理器
        self.message_handlers[MessageType.COMMAND] = self._handle_command
        self.message_handlers[MessageType.COORDINATION] = self._handle_coordination

    def _handle_command(self, msg: AgentMessage):
        """处理来自L3的命令"""
        if msg.sender.startswith('L3'):
            if 'target_flow' in msg.payload:
                self.target_flow = msg.payload['target_flow']
            if 'target_level' in msg.payload:
                self.target_level = msg.payload['target_level']

    def _handle_coordination(self, msg: AgentMessage):
        """处理ADMM协调消息"""
        if 'lambda' in msg.payload:
            self.lambda_coord = np.array(msg.payload['lambda'])
        if 'z' in msg.payload:
            self.z_shared = np.array(msg.payload['z'])
        if 'rho' in msg.payload:
            self.rho = msg.payload['rho']

    def perceive(self, system_state: Dict) -> Dict:
        """感知局部状态"""
        prefix = f"seg{self.segment_id}_"

        observations = {}

        # 提取本管段相关状态
        level_key = f"{prefix}level"
        flow_key = f"{prefix}flow"
        pressure_key = f"{prefix}pressure"
        velocity_key = f"{prefix}velocity"

        observations['level'] = system_state.get(level_key, system_state.get('pool_level', 5.0))
        observations['flow'] = system_state.get(flow_key, system_state.get('pipe_flow', 10.0))
        observations['pressure'] = system_state.get(pressure_key, system_state.get('pipe_pressure', 50.0))
        observations['velocity'] = system_state.get(velocity_key, 2.0)

        # 更新状态向量
        self.x = np.array([
            observations['level'],
            observations['flow'],
            observations['pressure'],
            observations['velocity']
        ])

        # 检查是否有L1覆盖
        if 'l1_override' in system_state:
            self.l1_override = system_state['l1_override']
            self.l1_override_actions = system_state.get('l1_actions', [])

        return observations

    def decide(self) -> List[ControlAction]:
        """MPC决策"""
        # 如果L1层有覆盖，直接返回空
        if self.l1_override:
            return []

        # 设置参考轨迹
        x_ref = np.array([self.target_level, self.target_flow, 50.0, 2.0])
        self.mpc.set_reference(x_ref)

        # 修改代价函数以包含ADMM项
        # J = J_local + lambda' * (u - z) + (rho/2) * ||u - z||^2
        # 这里简化处理，直接将z作为参考调整

        # 求解MPC
        result = self.mpc.solve(self.x, self.u)

        if result.status in [MPCStatus.OPTIMAL, MPCStatus.SUBOPTIMAL]:
            # 取第一步控制
            u_next = result.u_optimal[0]
            self.u = u_next

            actions = [
                ControlAction(
                    actuator_id=f'valve_{self.segment_id}_0',
                    action_type='set',
                    value=float(u_next[0]),
                    priority=AgentPriority.TACTICAL,
                    timestamp=time.time(),
                    source_agent=self.agent_id,
                    min_value=0.0,
                    max_value=1.0
                ),
                ControlAction(
                    actuator_id=f'valve_{self.segment_id}_1',
                    action_type='set',
                    value=float(u_next[1]),
                    priority=AgentPriority.TACTICAL,
                    timestamp=time.time(),
                    source_agent=self.agent_id,
                    min_value=0.0,
                    max_value=1.0
                )
            ]

            return actions
        else:
            # MPC求解失败，保持当前控制
            self.state.error_count += 1
            return []

    def act(self, actions: List[ControlAction]) -> Dict:
        """返回控制动作"""
        result = {
            'agent': self.agent_id,
            'segment': self.segment_id,
            'mpc_status': self.mpc.status.name,
            'actions': []
        }

        for action in actions:
            result['actions'].append({
                'actuator': action.actuator_id,
                'value': action.value
            })

        # 发送协调信息给L3
        self.send_message(AgentMessage(
            msg_type=MessageType.COORDINATION,
            sender=self.agent_id,
            receiver='L3_strategic',
            priority=self.priority,
            timestamp=time.time(),
            payload={
                'segment_id': self.segment_id,
                'u': self.u.tolist(),
                'x': self.x.tolist(),
                'cost': self.mpc.last_result.cost if self.mpc.last_result else 0
            }
        ))

        return result

    def get_predicted_trajectory(self) -> np.ndarray:
        """获取预测轨迹"""
        if self.mpc.last_result:
            return self.mpc.last_result.x_predicted
        return np.zeros((self.mpc.cfg.horizon, 4))

    def update_model(self, A: np.ndarray, B: np.ndarray):
        """更新MPC模型"""
        self.mpc.set_linear_model(A, B)

    def set_targets(self, flow: float, level: float):
        """设置控制目标"""
        self.target_flow = flow
        self.target_level = level

    def reset(self):
        """重置"""
        super().reset()
        self.x = np.zeros(4)
        self.u = np.zeros(2)
        self.l1_override = False
        self.lambda_coord = np.zeros(2)
        self.z_shared = np.zeros(2)
