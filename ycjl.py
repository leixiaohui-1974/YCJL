import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import logging
import math
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

# ==========================================
# 模块一：严苛的物理常数与全局配置
# ==========================================
class Config:
    # 仿真精度控制
    DT = 0.5             # 物理仿真时间步长 (s) - 满足 Courant 条件
    TOTAL_TIME = 86400   # 24小时仿真
    
    # 物理常数
    G = 9.80665          # 重力加速度 (m/s^2)
    RHO = 998.2          # 20度水密度 (kg/m^3)
    PATM = 10.33         # 标准大气压水头 (m)
    KINEMATIC_VISCOSITY = 1.004e-6 # 运动粘度

    # 1. 隧洞段 (圣维南方程)
    TUNNEL_LENGTH = 140000.0 
    TUNNEL_WIDTH = 6.0       # 矩形断面宽
    TUNNEL_SLOPE = 0.0005    # 底坡 i
    TUNNEL_MANNING = 0.014   # 曼宁系数 n
    TUNNEL_DX = 2000.0       # 空间步长
    TUNNEL_NODES = int(TUNNEL_LENGTH / TUNNEL_DX) + 1

    # 2. 稳流连接池 (积分环节)
    POOL_AREA_BASE = 1500.0  
    POOL_AREA_COEFF = 100.0  # A = A0 + k*H
    POOL_BOTTOM_EL = 0.0     # 底板高程
    POOL_TARGET_LEVEL = 5.0  # 设计运行水位
    
    # 3. 调压塔 (阻抗式)
    SURGE_DIA = 20.0
    SURGE_AREA = math.pi * (SURGE_DIA/2)**2
    SURGE_R_IN = 45.5        # 入流阻抗系数 (精确标定值)
    SURGE_R_OUT = 18.2       # 出流阻抗系数 (精确标定值)
    SURGE_BASE_EL = 10.0     # 塔底相对高程

    # 4. PCCP管线 (特征线法 MOC)
    PIPE_LENGTH = 180000.0   
    PIPE_DIAMETER = 2.4      
    PIPE_AREA = math.pi * (PIPE_DIAMETER/2)**2
    PIPE_WAVE_SPEED = 1050.0 # 波速 a (m/s)
    # MOC网格: dx = a * dt
    PIPE_DX = PIPE_WAVE_SPEED * DT 
    PIPE_NODES = int(PIPE_LENGTH / PIPE_DX) + 1
    PIPE_DARCY_F = 0.012     # 初始达西系数

    # 5. 阀门特性
    VALVE_T212_K_FULL = 0.15 # T212全开阻力系数
    VALVE_END_CV_MAX = 18.5  # 末端阀全开Cv值 (m^2.5/s)

    # 6. 安全设施
    RELIEF_SET_P = 120.0     # 泄压阀开启压力 (m)

# ==========================================
# 模块二：高保真组件物理模型
# ==========================================

class ValvePhysics:
    """真实阀门流体力学模型"""
    
    @staticmethod
    def radial_gate_flow(opening, width, H_up, H_down=0.0):
        """弧形闸门：考虑淹没与收缩系数"""
        if H_up <= 0.001: return 0.0
        
        e = np.clip(opening * 6.0, 0.001, 6.0) # 最大开度6m
        sigma = e / H_up # 相对开度
        
        # 流量系数 Cd 经验公式 (Vuskovic)
        Cd = 0.611 * np.sqrt(1 + 0.045 * sigma)
        
        # 淹没判定
        is_submerged = H_down > e * 0.75 # 收缩断面淹没
        
        if is_submerged:
            # 淹没出流公式 (Orifice equation with submergence)
            delta_h = H_up - H_down
            if delta_h < 0: return 0.0
            # 淹没系数 Cs
            Cs = 0.8  # 简化，实际应查表
            return Cs * Cd * width * e * np.sqrt(2 * Config.G * delta_h)
        else:
            # 自由出流
            return Cd * width * e * np.sqrt(2 * Config.G * H_up)

    @staticmethod
    def plunger_valve_k(opening):
        """活塞阀阻力系数 K(theta)"""
        # 特性：线性行程，但阻力系数呈倒数平方关系
        s = np.clip(opening, 0.001, 1.0)
        # 经验公式：K = K_min + 0.5 * (1/s - 1)^2
        return Config.VALVE_T212_K_FULL + 0.5 * ((1.0/s)**2 - 1.0)

class PIDController:
    """具备抗饱和与微分先行功能的工业级PID"""
    def __init__(self, kp, ki, kd, out_min=0.0, out_max=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        
        self.integral = 0.0
        self.last_input = 0.0
        
    def compute(self, setpoint, measurement, dt):
        error = setpoint - measurement
        
        # 积分项 (带抗饱和 Clamping)
        self.integral += error * dt
        
        # 微分项 (使用测量值微分 Derivative on Measurement，防止设定点跳变冲击)
        d_input = (measurement - self.last_input) / dt
        self.last_input = measurement
        
        output = (self.kp * error) + (self.ki * self.integral) - (self.kd * d_input)
        
        # 输出限幅与积分抗饱和回算
        if output > self.out_max:
            output = self.out_max
            self.integral -= error * dt # 回退积分
        elif output < self.out_min:
            output = self.out_min
            self.integral -= error * dt # 回退积分
            
        return output

# ==========================================
# 模块三：核心物理求解器 (Solver Kernels)
# ==========================================

class SaintVenantSolver:
    """明流求解器：扩散波近似 (Diffusive Wave) + 下游顶托处理"""
    def __init__(self):
        self.N = Config.TUNNEL_NODES
        self.dx = Config.TUNNEL_DX
        self.h = np.ones(self.N) * 3.5 # 初始水深
        self.Q = np.ones(self.N) * 10.0
        self.n = Config.TUNNEL_MANNING

    def step(self, Q_in, H_downstream_bc, dt):
        """
        :param Q_in: 上游入流
        :param H_downstream_bc: 下游边界水位 (稳流池水位)
        """
        # 上游边界
        self.Q[0] = Q_in
        
        h_new = self.h.copy()
        Q_new = self.Q.copy()
        width = Config.TUNNEL_WIDTH
        
        # 显式差分求解
        for i in range(1, self.N - 1):
            # 连续方程: dA/dt + dQ/dx = 0
            dq_dx = (self.Q[i] - self.Q[i-1]) / self.dx
            h_new[i] = self.h[i] - (dt / width) * dq_dx
            
            # 动量方程 (简化为扩散波): S_f = S_0 - dy/dx - (1/g)*dv/dt...
            # 这里使用运动波+扩散项近似，保证数值稳定性
            
            # 计算水力半径与摩阻坡度
            depth = self.h[i]
            area = depth * width
            peri = width + 2 * depth
            R = area / peri
            v = self.Q[i] / area
            
            # 曼宁公式反算 Sf
            # Q = 1/n * A * R^(2/3) * Sf^(1/2)
            # Sf = (Q*n)^2 / (A^2 * R^(4/3))
            Sf = (self.Q[i] * self.n)**2 / (area**2 * R**(4/3))
            
            # 压力梯度项
            dy_dx = (self.h[i+1] - self.h[i-1]) / (2 * self.dx)
            
            # 动量更新 (重力 - 摩阻 - 压力梯度)
            accel = Config.G * area * (Config.TUNNEL_SLOPE - Sf - dy_dx)
            Q_new[i] = self.Q[i] + accel * dt * 0.2 # 增加数值阻尼
            
        # 下游边界处理 (Backwater Effect)
        # 如果稳流池水位高于临界水深，则顶托；否则自由出流
        # 简化：假设下游断面与稳流池连通
        h_critical = (self.Q[-1]**2 / (Config.G * width**2))**(1/3)
        h_boundary = max(H_downstream_bc - Config.POOL_BOTTOM_EL, h_critical)
        
        # 边界松弛更新
        h_new[-1] = 0.9 * h_new[-1] + 0.1 * h_boundary
        Q_new[-1] = Q_new[-2] # 零梯度流出
        
        self.h = h_new
        self.Q = Q_new
        return self.Q[-1]

class MOCSolver:
    """特征线法求解器：精确处理非线性边界"""
    def __init__(self):
        self.N = Config.PIPE_NODES
        self.a = Config.PIPE_WAVE_SPEED
        self.B = self.a / (Config.G * Config.PIPE_AREA)
        self.R = Config.PIPE_DARCY_F * Config.PIPE_DX / (2 * Config.G * Config.PIPE_DIAMETER * Config.PIPE_AREA**2)
        
        # 状态矩阵
        self.H = np.ones(self.N) * 50.0
        self.Q = np.ones(self.N) * 10.0
        
        # 阀门位置
        self.mid_idx = int(self.N / 2) # T212
        
        # 故障注入
        self.burst_coeff = 0.0

    def solve_valve_boundary(self, Cp, Cm, K_valve, H_leak_ref=0.0):
        """
        牛顿迭代法求解阀门边界非线性方程:
        F(Q) = (Cp - B*Q) - (Cm + B*Q) - K_valve * Q * |Q| = 0
        """
        # 考虑爆管泄漏 Q_leak (简化为在阀前节点泄漏)
        # H_u = Cp - B(Q_val + Q_leak)
        # H_d = Cm + B*Q_val
        # H_u - H_d = K Q^2
        
        Q_guess = 10.0 # 初值
        tol = 1e-4
        max_iter = 10
        
        # 简化泄漏影响：假设泄漏不影响特征线到达时的 B*Q 项结构
        # 修正 Cp: Cp_eff = Cp - B * Q_leak_approx
        # 这里为了稳定性，仅在方程中体现泄漏造成的能量损失
        
        for _ in range(max_iter):
            Head_loss = K_valve * Q_guess * abs(Q_guess)
            # 动量平衡
            # Hu = Cp - B*Q
            # Hd = Cm + B*Q
            # F = Hu - Hd - Head_loss
            F = (Cp - Cm) - 2 * self.B * Q_guess - Head_loss
            
            # 导数 dF/dQ
            dF = -2 * self.B - 2 * K_valve * abs(Q_guess)
            
            Q_new = Q_guess - F / dF
            if abs(Q_new - Q_guess) < tol:
                return Q_new
            Q_guess = Q_new
            
        return Q_guess

    def step(self, H_up_bc, valve_mid_open, valve_end_open, demand_flow, dt):
        # 1. 计算特征线系数 Cp, Cm
        Hp = np.zeros(self.N)
        Qp = np.zeros(self.N)
        
        Cp = self.H[:-1] + self.B * self.Q[:-1] - self.R * self.Q[:-1] * np.abs(self.Q[:-1])
        Cm = self.H[1:] - self.B * self.Q[1:] + self.R * self.Q[1:] * np.abs(self.Q[1:])
        
        # 2. 内部节点
        Hp[1:-1] = (Cp[:-1] + Cm[1:]) / 2
        Qp[1:-1] = (Cp[:-1] - Cm[1:]) / (2 * self.B)
        
        # 3. 上游边界 (调压塔水位强约束)
        Hp[0] = H_up_bc
        Qp[0] = (Hp[0] - Cm[0]) / self.B
        
        # 4. 中间阀门边界 (T212)
        idx = self.mid_idx
        K_val_coef = ValvePhysics.plunger_valve_k(valve_mid_open)
        # 转换为水头损失系数 coeff * Q^2
        K_hydraulic = K_val_coef / (2 * Config.G * Config.PIPE_AREA**2)
        
        # 求解
        Q_val = self.solve_valve_boundary(Cp[idx-1], Cm[idx], K_hydraulic)
        
        # 爆管处理
        Q_leak = 0.0
        if self.burst_coeff > 0:
            Q_leak = self.burst_coeff * np.sqrt(max(Hp[idx], 0))
            
        Qp[idx] = Q_val
        # 阀后压力
        Hp[idx] = Cm[idx] + self.B * Q_val
        # 记录阀前压力 (用于爆管监测)
        H_valve_up = Cp[idx-1] - self.B * (Q_val + Q_leak)
        
        # 5. 末端边界
        # 目标是满足 demand_flow，但受限于阀门开度和物理压差
        # 联立 H = Cp_end - B*Q 和 Q = Cv * sqrt(H)
        Cp_end = Cp[-1]
        Cv_eff = valve_end_open * Config.VALVE_END_CV_MAX
        
        # 求解二次方程 B*Q + (1/Cv^2)*Q^2 - Cp = 0
        if Cv_eff > 1e-4 and Cp_end > 0:
            coeff_a = 1.0 / (Cv_eff**2)
            coeff_b = self.B
            coeff_c = -Cp_end
            Q_avail = (-coeff_b + np.sqrt(coeff_b**2 - 4*coeff_a*coeff_c)) / (2*coeff_a)
        else:
            Q_avail = 0.0
            
        Qp[-1] = min(Q_avail, demand_flow)
        Hp[-1] = Cp_end - self.B * Qp[-1]
        
        self.H = Hp
        self.Q = Qp
        
        return H_valve_up, Qp[idx], Hp[-1], Qp[-1]

class PhysicalPlant:
    """物理工厂总成"""
    def __init__(self):
        self.sv_solver = SaintVenantSolver()
        self.moc_solver = MOCSolver()
        
        # 状态量
        self.pool_level = 5.0
        self.surge_level = 45.0
        self.source_open = 0.0
        self.pool_out_open = 0.0
        
        # 辅助
        self.relief_open = 0.0

    def step(self, cmd_source_open, cmd_pool_open, cmd_mid_open, cmd_end_open, demand, burst_c, dt):
        # 1. 源头流出
        q_source = ValvePhysics.radial_gate_flow(cmd_source_open, Config.TUNNEL_WIDTH, 50.0)
        
        # 2. 隧洞演进 (传入稳流池水位作为顶托边界)
        q_tunnel_out = self.sv_solver.step(q_source, self.pool_level, dt)
        
        # 3. 稳流池流出 (传入调压塔水位计算压差)
        # 假设调压塔底与稳流池底高差 10m
        h_diff = max(0, self.pool_level - (self.surge_level - 10.0))
        q_pool_out = ValvePhysics.radial_gate_flow(cmd_pool_open, 3.0, h_diff)
        
        # 稳流池水位更新
        pool_area = Config.POOL_AREA_BASE + Config.POOL_AREA_COEFF * self.pool_level
        self.pool_level += (q_tunnel_out - q_pool_out) / pool_area * dt
        
        # 4. 调压塔动力学
        q_pipe_in = self.moc_solver.Q[0]
        dq = q_pool_out - q_pipe_in
        
        # 阻抗损失
        R_surge = Config.SURGE_R_IN if dq > 0 else Config.SURGE_R_OUT
        h_loss = R_surge * dq * abs(dq) * 1e-4 # 缩放系数
        
        self.surge_level += dq / Config.SURGE_AREA * dt
        h_boundary_pipe = self.surge_level + Config.SURGE_BASE_EL + h_loss
        
        # 5. MOC 求解
        self.moc_solver.burst_coeff = burst_c
        p_mid_up, q_mid, p_end, q_end = self.moc_solver.step(
            h_boundary_pipe, cmd_mid_open, cmd_end_open, demand, dt
        )
        
        # 6. 泄压阀 (机械被动)
        self.relief_open = ValvePhysics.relief_valve_dynamics(
            self.relief_open, p_mid_up, Config.RELIEF_SET_P, dt
        )
        if self.relief_open > 0:
            # 泄流反馈: 简单减少节点流量
            q_relief = self.relief_open * 20.0 * np.sqrt(p_mid_up/100)
            # 下一帧生效，直接修改对象状态 (副作用)
            self.moc_solver.Q[self.moc_solver.mid_idx] -= q_relief

        return {
            'Q_source': q_source,
            'H_pool': self.pool_level,
            'H_surge': self.surge_level,
            'P_mid': p_mid_up,
            'Q_mid': q_mid,
            'Q_end': q_end,
            'Relief': self.relief_open
        }

# ==========================================
# 模块四：数字孪生 (Digital Twin) - EKF实现
# ==========================================

class DigitalTwinEKF:
    """
    扩展卡尔曼滤波 (EKF) 用于在线估计管道摩阻系数 f
    状态向量 x = [Q_mid, f]
    观测向量 z = [P_mid_measure, Q_mid_measure]
    """
    def __init__(self):
        self.x = np.array([10.0, 0.012]) # 初始状态 [Q, f]
        self.P = np.eye(2) * 0.1         # 协方差矩阵
        
        self.Q_cov = np.diag([0.1, 1e-6]) # 过程噪声 (流量易变，f较稳定)
        self.R_cov = np.diag([0.5, 0.1])  # 观测噪声
        
    def predict(self, u_valve_open, h_up, dt):
        """预测步: 基于简化物理模型推演"""
        q, f = self.x
        
        # 简化流体动量方程: dQ/dt = (H_up - H_mid - h_f) / L * gA
        # H_mid 估算: H_up - h_f (稳态假设用于趋势预测)
        # h_f = f * L/D * v^2/2g
        
        area = Config.PIPE_AREA
        v = q / area
        h_f = f * (Config.PIPE_LENGTH/2) / Config.PIPE_DIAMETER * (v**2) / (2*Config.G)
        
        # 阻力系数
        K_val = ValvePhysics.plunger_valve_k(u_valve_open)
        h_val = K_val * (v**2) / (2*Config.G)
        
        # 简单的惯性更新
        # 这里的 F 是雅可比矩阵的一部分，为简化，使用数值积分
        # q_new = q + (H_up - h_f - h_val - H_down_est) ... 
        # 为简化 EKF，我们假设 f 是随机游走，Q 遵循惯性
        
        x_pred = np.array([q, f]) # 简单预测
        
        # 雅可比 F = dx_new / dx
        F = np.eye(2)
        
        self.x = x_pred
        self.P = F @ self.P @ F.T + self.Q_cov
        return self.x

    def update(self, z_meas, h_up_meas, u_valve_open):
        """更新步: 利用观测校正 f"""
        # z_meas = [P_mid, Q_mid]
        # 观测函数 h(x):
        # P_mid = H_up - h_f(Q, f) - h_valve(Q, u)
        # Q_mid = Q
        
        q_pred, f_pred = self.x
        area = Config.PIPE_AREA
        v = q_pred / area
        
        # 计算预测观测值
        h_friction = f_pred * (Config.PIPE_LENGTH/2) / Config.PIPE_DIAMETER * (v**2) / (2*Config.G)
        # 阀门压降 (假设测量点在阀前)
        h_pred = h_up_meas - h_friction
        
        z_pred = np.array([h_pred, q_pred])
        y_residual = z_meas - z_pred
        
        # 计算观测雅可比 H_jac = dz / dx
        # dp/dq = - (f*L/D + K_val)/2gA^2 * 2Q ...
        # dp/df = - L/D * v^2/2g
        term_friction = (Config.PIPE_LENGTH/2) / Config.PIPE_DIAMETER / (2*Config.G * area**2)
        
        dp_dq = -2 * q_pred * (f_pred * term_friction)
        dp_df = -q_pred**2 * term_friction
        
        H_jac = np.array([
            [dp_dq, dp_df],
            [1.0,   0.0]
        ])
        
        # 卡尔曼增益
        S = H_jac @ self.P @ H_jac.T + self.R_cov
        K = self.P @ H_jac.T @ np.linalg.inv(S)
        
        # 更新
        self.x = self.x + K @ y_residual
        self.P = (np.eye(2) - K @ H_jac) @ self.P
        
        # 约束 f > 0
        self.x[1] = max(0.005, self.x[1])
        
        return self.x[1] # 返回估计的 f

# ==========================================
# 模块五：控制系统 SCADA (L1-L3 Agents)
# ==========================================

class ScadaSystem:
    def __init__(self):
        # 控制器实例
        self.pid_pool = PIDController(kp=0.1, ki=0.005, kd=0.1, out_max=1.0) # 控稳流池出水
        self.pid_mid = PIDController(kp=0.05, ki=0.01, kd=0.0, out_max=1.0)  # 控中间阀流量
        
        # 智能体状态
        self.l3_demand_forecast = 10.0
        self.l1_mode = "NORMAL"
        
        # 数字孪生
        self.twin = DigitalTwinEKF()
        
    def step(self, t, sensors, dt):
        """
        :param sensors: {H_pool, P_mid, Q_mid, Q_source_meas}
        """
        # --- L3 战略层: 需水预测 ---
        # 场景: T=12h 需水从 10 -> 15
        if 40000 < t < 60000:
            self.l3_demand_forecast = 15.0
        else:
            self.l3_demand_forecast = 10.0
            
        # 前馈计算源头开度 (Q ~ e^1.5 approx or look up table)
        # 简单反算: e = Q / (Cd * B * sqrt(2gH))
        target_source_q = self.l3_demand_forecast
        source_open_ff = target_source_q / (0.65 * Config.TUNNEL_WIDTH * np.sqrt(2*9.8*50))
        source_open_ff = np.clip(source_open_ff, 0, 1)

        # --- L2 战术层: 闭环控制 ---
        
        # 1. 稳流池控制 (维持水位)
        # 目标: 水位 5.0m. 如果水位高，开大出水阀；水位低，关小
        # 叠加 L3 前馈流量需求
        pool_valve_ff = target_source_q / 25.0 # 粗略前馈
        pool_valve_pid = self.pid_pool.compute(Config.POOL_TARGET_LEVEL, sensors['H_pool'], dt)
        # 组合: 这里的PID输出作为微调系数或加算
        # 逻辑: 阀门开度 = 前馈 + (水位PID修正: 水位高->正修正)
        # 注意PID方向: Setpoint - PV. Setpoint > PV (低水位) -> 正输出. 
        # 但我们希望低水位时关小阀门(负修正). 所以系数为负
        cmd_pool_valve = pool_valve_ff - 0.5 * pool_valve_pid
        cmd_pool_valve = np.clip(cmd_pool_valve, 0, 1)
        
        # 2. 中间阀控制 (流量追踪)
        cmd_mid_valve_delta = self.pid_mid.compute(self.l3_demand_forecast, sensors['Q_mid'], dt)
        # PID输出作为开度增量或绝对值? 
        # 这里PID设计为输出调整量有点麻烦，改为直接计算目标开度
        # 重新设计: PID输出绝对开度
        # 动态重置 integral 如果模式切换
        cmd_mid_valve = 0.5 + self.pid_mid.integral + self.pid_mid.kp * (self.l3_demand_forecast - sensors['Q_mid'])
        cmd_mid_valve = np.clip(cmd_mid_valve, 0.1, 1.0)
        
        # --- 数字孪生同步 ---
        est_f = self.twin.predict(cmd_mid_valve, sensors['H_surge'], dt) # H_surge 近似 H_up
        est_f = self.twin.update(np.array([sensors['P_mid'], sensors['Q_mid']]), sensors['H_surge'], cmd_mid_valve)

        # --- L1 安全层: 超驰控制 ---
        l1_override = {}
        
        # 爆管检测
        if sensors['P_mid'] < 20.0 and sensors['Q_mid'] > 5.0 and cmd_mid_valve > 0.1:
            # 压力过低但还在供水 -> 爆管
            self.l1_mode = "BURST"
            l1_override['valve_mid'] = 0.0 # 强制关闭
        
        # 超压检测
        if sensors['P_mid'] > 115.0:
            # 辅助泄压，强制关小上游? 不，超压通常由急关引起，此时应保持或慢动
            pass 

        # 最终指令合成
        actions = {
            'source_gate': source_open_ff,
            'pool_valve': cmd_pool_valve,
            'valve_mid': l1_override.get('valve_mid', cmd_mid_valve),
            'valve_end': 1.0, # 末端全开，由T212控流
            'est_f': est_f
        }
        return actions

# ==========================================
# 模块六：主程序
# ==========================================

def run_simulation():
    plant = PhysicalPlant()
    scada = ScadaSystem()
    
    # 数据记录
    history = {k: [] for k in ['time', 'Q_source', 'H_pool', 'P_mid', 'Q_mid', 'Relief', 'Est_f']}
    
    print(f"=== 开始全系统高保真仿真 (DT={Config.DT}s) ===")
    
    for i in range(int(Config.TOTAL_TIME / Config.DT)):
        t = i * Config.DT
        
        # 物理场景注入
        burst_c = 0.0
        if 72000 < t < 74000: burst_c = 5.0 # T=20h 爆管
        
        if t > 57600: plant.pipe.R *= 1.5 # T=16h 阻力增加 (冰期/结垢)

        # 1. 传感器采集 (加噪声)
        sensors = {
            'H_pool': plant.pool_level + np.random.normal(0, 0.005),
            'P_mid': plant.moc_solver.H[plant.moc_solver.mid_idx] + np.random.normal(0, 0.1),
            'Q_mid': plant.moc_solver.Q[plant.moc_solver.mid_idx] + np.random.normal(0, 0.05),
            'H_surge': plant.surge_level + 40.0,
            'Q_source_meas': plant.tunnel.Q[0]
        }
        
        # 2. SCADA 决策
        actions = scada.step(t, sensors, Config.DT)
        
        # 3. 物理执行
        state = plant.step(
            actions['source_gate'],
            actions['pool_valve'],
            actions['valve_mid'],
            actions['valve_end'],
            15.0, # User Demand max capability
            burst_c,
            Config.DT
        )
        
        # 记录
        if i % 120 == 0:
            history['time'].append(t/3600)
            history['Q_source'].append(state['Q_source'])
            history['H_pool'].append(state['H_pool'])
            history['P_mid'].append(state['P_mid'])
            history['Q_mid'].append(state['Q_mid'])
            history['Relief'].append(state['Relief'])
            history['Est_f'].append(actions['est_f'])
            
        if i % 36000 == 0:
            print(f"Progress: {t/3600:.1f}h | Pool: {state['H_pool']:.2f}m")

    return history

if __name__ == "__main__":
    data = run_simulation()
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    
    # 1. 流量追踪
    axes[0].plot(data['time'], data['Q_source'], label='Source')
    axes[0].plot(data['time'], data['Q_mid'], label='Mid Valve')
    axes[0].set_ylabel('Flow (m3/s)')
    axes[0].set_title('Flow Tracking (L3 Forecast + L2 PID)')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. 稳流池控制
    axes[1].plot(data['time'], data['H_pool'], 'g')
    axes[1].axhline(Config.POOL_TARGET_LEVEL, color='k', linestyle='--')
    axes[1].set_ylabel('Pool Level (m)')
    axes[1].set_title('Pool Level Control (PID)')
    axes[1].grid(True)
    
    # 3. 压力与安全
    axes[2].plot(data['time'], data['P_mid'], 'r')
    axes[2].set_ylabel('Pressure (m)')
    axes[2].set_title('Pipe Pressure (MOC + Burst Event)')
    axes[2].grid(True)
    
    # 4. 数字孪生参数辨识
    axes[3].plot(data['time'], data['Est_f'], 'purple')
    axes[3].set_ylabel('Estimated f')
    axes[3].set_title('Digital Twin: Online Friction Estimation (EKF)')
    axes[3].set_xlabel('Time (h)')
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.show()