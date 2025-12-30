
import numpy as np
import matplotlib.pyplot as plt
from ycjl_model import SaintVenantSolver, Config

# ==============================================================================
# L5 级数字孪生：多级泵站脉冲协同仿真 (Multi-Agent Pulse Coordination)
# ==============================================================================
# 模拟场景：屯佃泵站(Station 1) -> 渠道1 (12km) -> 前柳林泵站(Station 2)
# 验证目标：下游泵站能否精准配合上游脉冲，实现“智能接力”

class L5Coordinator:
    """中心化协调智能体 (或分布式协商逻辑)"""
    def __init__(self, distance_km, wave_speed_approx=5.0):
        self.distance = distance_km * 1000
        self.wave_speed = wave_speed_approx # m/s (浅水波速 sqrt(gh))
        
        # 预测波传播时间
        self.travel_time_sec = self.distance / self.wave_speed
        print(f"🌊 [L5 MPC] Estimated Wave Travel Time: {self.travel_time_sec/60:.1f} min")
        
    def get_downstream_schedule(self, t_now, upstream_schedule):
        """
        基于MPC思想，生成下游泵站的动作计划
        upstream_schedule: [(start_time, end_time, Q), ...]
        """
        downstream_schedule = []
        for (t_start, t_end, Q) in upstream_schedule:
            # 策略：提前一点开启下游泵站，甚至预降水位(Pre-discharge)
            # 这里简化为：紧跟波头，考虑滞后
            t_arrival = t_start + self.travel_time_sec
            t_departure = t_end + self.travel_time_sec
            
            # 为了安全，下游泵站稍微延后开启，利用前池容积缓冲头部波
            # 或者提前开启以腾容积？
            # L5策略：精准匹配。由于明渠有坦化作用，波形会变，这里做简单滞后。
            downstream_schedule.append((t_arrival, t_departure, Q))
            
        return downstream_schedule

class CascadedCanalSimulation:
    def __init__(self):
        # 渠道段1：屯佃 -> 前柳林
        self.len_1 = 12000.0 # 12km
        self.width = 20.0
        self.dx = 500.0
        self.dt = 30.0
        
        # 物理模型: Channel 1
        self.channel1 = SaintVenantSolver(length=self.len_1, width=self.width, dx=self.dx)
        
        # 初始状态
        self.channel1.h[:] = 2.5
        self.channel1.Q[:] = 0.0
        
        # 前池模型 (Forebay at Station 2 input)
        self.forebay_level = 2.5
        self.forebay_area = 2000.0 # m2
        
        # L5协调器
        # 估算波速: h=2.5, v~0 -> c = sqrt(9.8*2.5) = 4.95 m/s
        self.coordinator = L5Coordinator(distance_km=12.0, wave_speed_approx=5.0)
        
        # 泵站计划 (上游)
        # T=1h 开始脉冲，持续 4h
        self.res_schedule_up = [(3600, 3600 + 4*3600, 20.0)]
        self.res_schedule_down = self.coordinator.get_downstream_schedule(0, self.res_schedule_up)

    def get_pump_q(self, t, schedule):
        for (start, end, q_val) in schedule:
            if start <= t < end:
                return q_val
        return 0.0

    def run(self):
        history = {'time':[], 'H_up':[], 'H_down_forebay':[], 'Q_up':[], 'Q_down':[]}
        
        total_steps = int(12 * 3600 / self.dt) # 12 hours
        
        print("🚀 Starting Multi-Agent Simulation...")
        
        for i in range(total_steps):
            t = i * self.dt
            
            # 1. 上游泵站动作 (Station 1 Output)
            q_pump_1 = self.get_pump_q(t, self.res_schedule_up)
            
            # 2. 下游泵站动作 (Station 2 Input/Suction)
            # 智能体控制：严格执行计划
            q_pump_2 = self.get_pump_q(t, self.res_schedule_down)
            
            # 3. 渠道1演进
            # 上游边界: q_pump_1
            # 下游边界: 假设自由出流进前池 (Q_out depends on h_end vs h_forebay)
            # 简化: 下游通过 forebay level 顶托
            q_last = self.channel1.step(q_pump_1, self.forebay_level, self.dt)
            
            # 4. 下游前池水位更新
            # dH/dt = (Q_in_from_channel - Q_pump_2) / Area
            # Q_in_from_channel = q_last (Solver的末端流量)
            self.forebay_level += (q_last - q_pump_2) / self.forebay_area * self.dt
            
            if i % 10 == 0:
                history['time'].append(t/3600)
                history['H_up'].append(self.channel1.h[0])
                history['H_down_forebay'].append(self.forebay_level)
                history['Q_up'].append(q_pump_1)
                history['Q_down'].append(q_pump_2)
                
        return history

def plot_results(res):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. 流量协调
    axes[0].plot(res['time'], res['Q_up'], 'b-', label='Station 1 (Upstream)')
    axes[0].plot(res['time'], res['Q_down'], 'r--', label='Station 2 (Downstream)')
    axes[0].set_ylabel('Flow (m3/s)')
    axes[0].set_title('L5 Multi-Agent Coordination: Relay Pumping')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. 前池水位 (关键指标: 是否抽空或溢出)
    axes[1].plot(res['time'], res['H_down_forebay'], 'g', linewidth=2)
    axes[1].axhline(y=4.0, color='r', linestyle=':', label='Overflow Limit')
    axes[1].axhline(y=1.0, color='r', linestyle=':', label='Suction Limit')
    axes[1].set_ylabel('Forebay Level (m)')
    axes[1].set_title('Station 2 Forebay Stability')
    axes[1].legend()
    axes[1].grid(True)
    
    # 3. 上游水位
    axes[2].plot(res['time'], res['H_up'], 'k')
    axes[2].set_xlabel('Time (h)')
    axes[2].set_ylabel('Level (m)')
    axes[2].set_title('Station 1 Outlet Channel Level')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('miyun_l5_multi_agent.png')
    print("Saved to miyun_l5_multi_agent.png")

if __name__ == "__main__":
    sim = CascadedCanalSimulation()
    data = sim.run()
    plot_results(data)
