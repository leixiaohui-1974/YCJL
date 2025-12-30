
import numpy as np
import matplotlib.pyplot as plt
from miyun_real_model import CascadedChannelSystem, L5Scheduler

def run_real_simulation():
    # 1. 初始化系统
    system = CascadedChannelSystem()
    scheduler = L5Scheduler(system)
    
    # 2. 生成 L5 调度计划 (目标: 50万m3/天 => 约 6 m3/s 平均流量)
    # Q_avg = 6 m3/s -> Vol = 6 * 24 * 3600 = 51.84 万 m3
    target_vol = 518400.0
    schedule = scheduler.plan_daily_schedule(target_vol)
    
    # 3. 仿真循环
    dt = 5.0
    total_steps = int(30 * 3600 / dt) # 30 hours full run
    
    history = {
        'time': [],
        'levels': [[] for _ in range(5)], # 5个中间前池
        'powers': [],
        'flows': [[] for _ in range(6)]
    }
    
    print("\n🚀 Starting High-Fidelity Simulation (6 Stages)...")
    
    for i in range(total_steps):
        t = i * dt
        
        # 获取当前时刻各泵站流量指令
        pump_cmds = []
        for s_idx in range(6):
            plan = schedule[s_idx]
            # 简单矩形波 -> 改为 Soft Start Trapezoid
            # Ramp time: 20 min (1200s)
            t_ramp = 1200.0
            
            if plan['t_start'] <= t < plan['t_end']:
                # Ramp Up
                if t < plan['t_start'] + t_ramp:
                    progress = (t - plan['t_start']) / t_ramp
                    q_cmd = plan['q_target'] * progress
                # Ramp Down
                elif t > plan['t_end'] - t_ramp:
                    progress = (plan['t_end'] - t) / t_ramp
                    q_cmd = plan['q_target'] * progress
                else:
                    q_cmd = plan['q_target']
            else:
                q_cmd = 0.0
            pump_cmds.append(q_cmd)
            
        # 物理步进
        state = system.step(pump_cmds, dt)
        
        # 记录
        if i % 10 == 0: # 10min record
            history['time'].append(t/3600.0)
            total_power = sum(state['powers'])
            history['powers'].append(total_power)
            
            for f_idx in range(5):
                history['levels'][f_idx].append(state['forebays'][f_idx]['level'])
            
            for p_idx in range(6):
                history['flows'][p_idx].append(pump_cmds[p_idx])
                
        if i % 600 == 0:
            print(f"T={t/3600:.1f}h | Power={sum(state['powers'])/1000:.1f}MW | FB1={state['forebays'][0]['level']:.2f}m")

    return history, schedule

def plot_real_results(history):
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    
    # 1. 流量接力 (Flow Relay)
    # 只画 Station 1, 3, 6 代表
    axes[0].plot(history['time'], history['flows'][0], 'b-', label='Stn 1 (Start)')
    axes[0].plot(history['time'], history['flows'][2], 'g--', label='Stn 3 (Mid)')
    axes[0].plot(history['time'], history['flows'][5], 'r-.', label='Stn 6 (End)')
    axes[0].set_ylabel('Flow (m3/s)')
    axes[0].set_title('L5 Precise Flow Relay (Pulse Coordination)')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. 前池水位稳定性
    for i in range(5):
        axes[1].plot(history['time'], history['levels'][i], label=f'Forebay {i+1}')
    
    axes[1].axhline(5.0, color='k', linestyle='--', label='Overflow')
    axes[1].axhline(1.0, color='k', linestyle=':', label='Suction Limit')
    axes[1].set_ylabel('Water Level (m)')
    axes[1].set_title('Forebay Stability')
    axes[1].legend(ncol=3)
    axes[1].grid(True)
    
    # 3. 功率脉冲
    axes[2].plot(history['time'], np.array(history['powers'])/1000, 'purple')
    axes[2].set_ylabel('Total Power (MW)')
    axes[2].set_xlabel('Time (h)')
    axes[2].set_title('System Power Consumption (Pulse Mode)')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('miyun_real_simulation.png')
    print("Saved miyun_real_simulation.png")
    
    # KPI Calculation
    # Power is recorded every 10 mins (600s). Unit: W (from Power calc which was roughly W or kW? Check model.)
    # In model: power = 9.81 * Q * H / eta. (kW). 
    # history['powers'] is list of total kW.
    
    avg_power_kw = np.mean(history['powers'])
    total_time_h = history['time'][-1]
    total_energy_kwh = avg_power_kw * total_time_h
    
    # Volume: Station 6 (End) flow integral
    # Flow recorded every 10 min.
    q_end_avg = np.mean(history['flows'][5])
    total_vol_m3 = q_end_avg * (total_time_h * 3600)
    
    if total_vol_m3 > 0:
        unit_consumption = total_energy_kwh / total_vol_m3
    else:
        unit_consumption = 0
        
    print(f"\n===== L5 Optimization Results =====")
    print(f"Total Energy: {total_energy_kwh:.2f} kWh")
    print(f"Total Water:  {total_vol_m3:.2f} m3")
    print(f"Unit Energy:  {unit_consumption:.4f} kWh/m3")
    print(f"Comparison:   Traditional ~0.048 kWh/m3 (Estimated for 6m3/s throttling)")
    print(f"efficiency:   {unit_consumption / 0.048 * 100:.1f}% of Traditional (Saving {(1 - unit_consumption/0.048)*100:.1f}%)")
    print(f"===================================")

if __name__ == "__main__":
    hist, sched = run_real_simulation()
    plot_real_results(hist)
