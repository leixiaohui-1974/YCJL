
import numpy as np
import matplotlib.pyplot as plt
from ycjl_model import SaintVenantSolver, Config

# ==============================================================================
# L5 级数字孪生：脉冲输水水力响应仿真 (Hydrodynamic Response of Pulse Pumping)
# ==============================================================================

class MiyunCanalSimulation:
    def __init__(self):
        # 提取自《项目建议书》
        self.LENGTH = 73000.0  # 73km
        self.WIDTH = 20.0      # 底宽 20m
        self.SLOPE = 0.00025   # 纵坡 1/4000
        self.MANNING = 0.017   # 糙率
        self.DX = 1000.0       # 网格步长
        self.DT = 60.0         # 仿真步长 (s) - 大步长加速
        
        # 初始条件
        self.init_depth = 2.5  # 初始水深
        
    def run_scenario(self, mode="TRADITIONAL"):
        """
        mode:
          - TRADITIONAL: 恒定 6 m3/s
          - PULSE: 20 m3/s (30% duty) -> 0 m3/s
        """
        # 初始化求解器
        solver = SaintVenantSolver(
            length=self.LENGTH,
            width=self.WIDTH, 
            slope=self.SLOPE,
            manning=self.MANNING,
            dx=self.DX
        )
        # 重置初始状态
        solver.h = np.ones(solver.N) * self.init_depth
        solver.Q = np.ones(solver.N) * (6.0 if mode == "TRADITIONAL" else 0.0)
        
        history = {
            'time': [],
            'Q_in': [],
            'H_start': [],
            'H_mid': [],
            'H_end': []
        }
        
        total_steps = int(24 * 3600 / self.DT)
        
        print(f"Running Scenario: {mode}...")
        
        for i in range(total_steps):
            t = i * self.DT
            t_hour = t / 3600.0
            
            # --- 边界条件控制 ---
            if mode == "TRADITIONAL":
                q_in = 6.0
            else: # PULSE
                # 30% Duty Cycle of 24h = 7.2h
                if t_hour < 7.2:
                    q_in = 20.0 # 全速高效区
                else:
                    q_in = 0.0
            
            # 下游边界：假设为固定水位（由于多级泵站控制，末端水位相对受控）
            # 或者简单的自由出流 Q_out = Q_in_avg (为了维持水量平衡)
            # 这里简化为：下游泵站按计划抽水
            # L5模式下，下游泵站也会联动（Smart Skipping），这里模拟这种联动
            # 假设下游抽水量紧跟上游（考虑滞后）
            # 为简化，假设下游水位维持在设计深度
            h_down_bc = 2.5 
            
            solver.step(q_in, h_down_bc + Config.POOL_BOTTOM_EL, self.DT)
            
            if i % 10 == 0:
                history['time'].append(t_hour)
                history['Q_in'].append(q_in)
                history['H_start'].append(solver.h[0])
                history['H_mid'].append(solver.h[int(solver.N/2)])
                history['H_end'].append(solver.h[-1])
                
        return history

def main():
    sim = MiyunCanalSimulation()
    
    # 运行两种方案
    res_trad = sim.run_scenario("TRADITIONAL")
    res_pulse = sim.run_scenario("PULSE")
    
    # 绘图对比
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. 流量输入对比
    axes[0].plot(res_trad['time'], res_trad['Q_in'], 'b--', label='Traditional (6 m3/s)')
    axes[0].plot(res_pulse['time'], res_pulse['Q_in'], 'r-', label='L5 Pulse (20 m3/s PWM)')
    axes[0].set_ylabel('Inflow Q (m3/s)')
    axes[0].set_title('Flow Control Strategy Comparison')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. 渠首水位响应 (Buffer)
    axes[1].plot(res_trad['time'], res_trad['H_start'], 'b--', label='Traditional Level')
    axes[1].plot(res_pulse['time'], res_pulse['H_start'], 'r-', label='L5 Pulse Level')
    # 警戒线
    axes[1].axhline(y=3.5, color='k', linestyle=':', label='Max Design Level')
    axes[1].axhline(y=1.5, color='k', linestyle='-.', label='Min Operating Level')
    axes[1].set_ylabel('Start Level (m)')
    axes[1].set_title('Canal Buffering Capability (Start Node)')
    axes[1].legend()
    axes[1].grid(True)
    
    # 3. 渠中水位响应
    axes[2].plot(res_trad['time'], res_trad['H_mid'], 'b--', label='Traditional Level')
    axes[2].plot(res_pulse['time'], res_pulse['H_mid'], 'r-', label='L5 Pulse Level')
    axes[2].set_ylabel('Mid Level (m)')
    axes[2].set_title('Wave Propagation (Mid Node ~36km)')
    axes[2].set_xlabel('Time (h)')
    axes[2].grid(True)
    
    print("\nSimulation Complete. Generating Report Plot...")
    plt.tight_layout()
    plt.savefig('miyun_l5_simulation_result.png')
    print("Saved to miyun_l5_simulation_result.png")

if __name__ == "__main__":
    main()
