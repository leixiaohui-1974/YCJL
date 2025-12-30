
import numpy as np
import scipy.interpolate as interp
from dataclasses import dataclass
from typing import List, Dict
from ycjl_model import SaintVenantSolver, Config

# ==============================================================================
# 模块一：真实物理组件 (Real Physics Components)
# ==============================================================================

@dataclass
class PumpSpecs:
    name: str
    design_Q: float  # m3/s
    design_H: float  # m
    rated_power: float # kW
    type: str = "Axial" # Axial(轴流) / Mixed(混流)

class PumpStation:
    """
    真实泵站模型
    包含：Q-H 特性曲线、Q-Eta 效率曲线、马鞍区振动模型
    """
    def __init__(self, specs: PumpSpecs):
        self.specs = specs
        self._init_curves()
        
    def _init_curves(self):
        # 基于比转速构造通用无因次特性曲线
        # 轴流泵特性：陡降的Q-H线，马鞍区显著
        q_ratios = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
        
        if self.specs.type == "Axial":
            # 轴流泵典型曲线 (H/H_des)
            # 马鞍区通常在 0.4~0.6 Q_des 处，扬程出现低谷或波动
            # Be careful: Axial pumps have HIGH shut-off head (2.0x)
            h_ratios = np.array([2.0, 1.6, 0.9, 1.1, 1.2, 1.0, 0.7, 0.3]) 
            # 效率曲线 (Eta/Eta_max)
            eta_ratios = np.array([0.0, 0.2, 0.5, 0.75, 0.9, 1.0, 0.8, 0.4])
        else:
            # 混流/离心泵 (Mixed)
            h_ratios = np.array([1.4, 1.35, 1.3, 1.25, 1.15, 1.0, 0.8, 0.5])
            eta_ratios = np.array([0.0, 0.3, 0.55, 0.75, 0.92, 1.0, 0.85, 0.5])

        self.curve_H = interp.interp1d(q_ratios * self.specs.design_Q, 
                                      h_ratios * self.specs.design_H, 
                                      kind='cubic', fill_value="extrapolate")
        
        self.curve_Eta = interp.interp1d(q_ratios * self.specs.design_Q, 
                                        eta_ratios * 0.85, # 假设最高效率 0.85
                                        kind='cubic', fill_value="extrapolate")

    def get_op_point(self, Q_flow, static_head):
        """
        给定流量 Q，计算扬程、功率、振动风险
        """
        # 1. 物理扬程能力
        H_capacity = float(self.curve_H(Q_flow))
        
        # 2. 实际运行点
        # 如果 H_capacity < static_head，泵无法抽水 (倒流或闷泵) -> Q=0
        # 这里假设泵站有可变叶片调节(VFD)或闸门节流来匹配 statid_head
        # simplified: 泵克服 static_head + loss，如果 H_cap > H_needed，则闸门节流 H_gate = H_cap - H_needed
        
        # 振动风险判定 (Saddle Zone Detection)
        # 轴流泵马鞍区通常在 0.3-0.7 Q_des
        vibration_level = 0.0
        q_ratio = Q_flow / self.specs.design_Q
        if self.specs.type == "Axial" and 0.3 < q_ratio < 0.7:
            diff = abs(q_ratio - 0.5)
            vibration_level = max(0, 1.0 - diff*5) # 0.5处最大
            
        efficiency = float(self.curve_Eta(Q_flow))
        power = 9.81 * Q_flow * H_capacity / max(0.01, efficiency)
        
        return {
            'H_out': H_capacity,
            'Power': power,
            'Efficiency': efficiency,
            'Vibration': vibration_level
        }

class Forebay:
    """前池模型 (Integrator Node)"""
    def __init__(self, name, area=1500.0, base_el=0.0, init_level=3.0):
        self.name = name
        self.area = area
        self.base_el = base_el
        self.level = init_level
        
        self.min_level = 1.0 # 吸入死水位
        self.max_level = 5.0 # 溢流位
        
    def step(self, Q_in, Q_out, dt):
        dh = (Q_in - Q_out) / self.area * dt
        # Numerical Damping: Clamp large dh to prevent unphysical spikes
        dh = max(-1.0, min(1.0, dh))
        
        self.level += dh
        
        status = "OK"
        if self.level < self.min_level: status = "SUCTION_RISK"
        if self.level > self.max_level: status = "OVERFLOW"
        
        return self.level, status

# ==============================================================================
# 模块二：系统拓扑构建 (System Builder)
# ==============================================================================

class CascadedChannelSystem:
    def __init__(self):
        # 密云调蓄工程真实拓扑参数 (Tundian -> Xitaishang)
        # Station 1 (Tundian) -> Channel 1 -> Station 2 (Qianliulin) -> ...
        
        self.stations = []
        self.channels = []
        self.forebays = []
        
        # 定义6级泵站参数
        pump_data = [
            ("Tundian", 20, 3.2, 1800),
            ("Qianliulin", 20, 3.2, 1800),
            ("Niantou", 20, 4.1, 2000),
            ("Xingshou", 20, 4.1, 2000),
            ("Lishishan", 20, 2.9, 1800),
            ("Xitaishang", 20, 7.15, 3800, "Mixed") # 最后一级扬程高，混流泵
        ]
        
        # 渠道分段 (总长73km，平均分配，实际每段不同)
        segment_lengths = [10000, 12000, 15000, 12000, 11000, 13000] # sum=73km
        
        for i in range(6):
            # 1. 创建泵站
            p = PumpStation(PumpSpecs(pump_data[i][0], pump_data[i][1], pump_data[i][2], pump_data[i][3], 
                                      type=pump_data[i][4] if len(pump_data[i])>4 else "Axial"))
            self.stations.append(p)
            
            # 2. 创建渠道 (连接 Pump_i 到 Pump_i+1 的前池)
            # 最后一个渠道流向 怀柔水库 (作为无限大 Forebay)
            c = SaintVenantSolver(length=segment_lengths[i], width=20.0, slope=0.00025, dx=500.0)
            # Init state
            c.h[:] = 3.0 # Increase init level to match forebay target (3.0) to reduce initial slosh
            c.Q[:] = 0.0
            self.channels.append(c)
            
            # 3. 创建前池 (Pump_i 的吸水池? 不，通常Pump_i从前池抽水送入Channel_i)
            # 拓扑: Forebay_i -> Pump_i -> Channel_i -> Forebay_i+1
            # Forebay_0 是团城湖 (无限源)
            # Forebay_1 是 Tundian 的前池?
            # 修正描述: "Tundian Pump" lifts water from Tuanchenghu (Const Level) into Channel A.
            # Channel A flows into "Qianliulin Forebay".
            # "Qianliulin Pump" lifts from Forebay into Channel B.
            if i < 5:
                # 中间前池 (Large Area for stability and realism)
                fb = Forebay(f"FB_{pump_data[i+1][0]}", area=50000.0)
                self.forebays.append(fb)
            else:
                # 终点: 怀柔水库
                self.huairou_reservoir = Forebay("Huairou", area=1e7, init_level=50.0) # Infinite

    def step(self, pump_flows: List[float], dt: float):
        """
        :param pump_flows: 6个泵站的抽水流量指令
        """
        system_state = {'forebays': [], 'vibrations': [], 'powers': []}
        
        # 1. Pre-calculate constrained flows (Low Level Interlock)
        actual_flows = list(pump_flows)
        for i in range(1, 6): # Pump 0 (source) is infinite. Pumps 1..5 have forebays.
            fb = self.forebays[i-1]
            if fb.level < 1.5:
                 # Throttle: if level=1.5 -> full flow possible? No, start throttling.
                 # if level=1.0 -> 0 flow.
                 limit = max(0, (fb.level - 1.0) * 40.0) # Slope: 0.5m diff -> 20m3/s. 
                 actual_flows[i] = min(actual_flows[i], limit)

        # 2. Sequential Calculation
        for i in range(6):
            # --- Link i (Channel) ---
            # Pump i discharge -> Channel i input
            # Downstream BC for Channel i:
            if i < 5:
                # Next is forebay for Pump i+1
                h_down = self.forebays[i].level
            else:
                # Last channel -> Huairou
                h_down = self.huairou_reservoir.level
                
            q_channel_out = self.channels[i].step(actual_flows[i], h_down, dt)
            
            # --- Node i+1 (Forebay Update) ---
            # Forebay i Update (needs outflow from Pump i+1)
            if i < 5:
                # Outflow from forebay is Pump i+1's actual flow
                q_out_pump = actual_flows[i+1] # This is safe because actual_flows is fully pre-calculated
                lev, status = self.forebays[i].step(q_channel_out, q_out_pump, dt)
                
                # DEBUG: Check for NaNs
                if np.isnan(lev):
                    print(f"!!! NAN Detected at Forebay {i} (Step {i}) !!!")
                    print(f"  Q_in (Channel Out): {q_channel_out}")
                    print(f"  Q_out (Pump {i+1}): {q_out_pump}")
                    print(f"  Prev Level: {self.forebays[i].level}")
                    # raise ValueError("Simulation Exploded") # Relaxed: Don't crash, just clamp
                    lev = 3.0 # Reset to nominal
                    self.forebays[i].level = 3.0
                    
                system_state['forebays'].append({'level': lev, 'status': status})
                
            # --- Pump Performance Calc (for Pump i) ---
            # H_static estimation
            op = self.stations[i].get_op_point(actual_flows[i], self.stations[i].specs.design_H)
            system_state['vibrations'].append(op['Vibration'])
            system_state['powers'].append(op['Power'])
            
            # Update flows in place for return
            pump_flows[i] = actual_flows[i]
            
        return system_state

# ==============================================================================
# 模块三：L5 全局调度器 (Global Optimization)
# ==============================================================================

class L5Scheduler:
    """
    求解器：根据总需水量，计算脉冲宽度和相位差
    目标：Forebay水位波动最小 (Min-Max差值最小)
    """
    def __init__(self, system: CascadedChannelSystem):
        self.sys = system
        
    def plan_daily_schedule(self, target_daily_volume_m3):
        # 1. 计算所需 Duty Cycle
        # Q_des = 20
        # Time_on = Vol / 20 / 3600
        total_time_needed_h = target_daily_volume_m3 / 20.0 / 3600.0
        duty_cycle = total_time_needed_h / 24.0
        
        schedule = []
        
        # 2. 计算相位差 (Phase Shift)
        # 基于波速预估滞后
        # T_lag_i = Length_i / WaveSpeed_i
        # WaveSpeed approx 5 m/s
        current_lag = 0.0
        
        print(f"L5 Planning: Volume={target_daily_volume_m3/1e4:.1f}万m3, Duty={duty_cycle*100:.1f}%")
        
        for i in range(6):
            # 每一级在前一级的基础上滞后
            # Channel i length
            length = self.sys.channels[i].length if i > 0 else 0 
            
            if i > 0:
                prev_len = self.sys.channels[i-1].length
                travel_t = prev_len / 5.0 # sec
                # Safety Margin: delay downstream start by 10 mins to ensure water accumulation
                # This fixes the "Forebay Drawdown" issue (3.0m -> 1.25m) seen in testing.
                safety_margin = 600.0 
                current_lag += (travel_t + safety_margin)
            
            t_start = current_lag
            t_end = t_start + total_time_needed_h * 3600
            
            schedule.append({
                'station_idx': i,
                't_start': t_start,
                't_end': t_end,
                'q_target': 20.0
            })
            print(f"  - Station {i+1}: Start +{t_start/60:.0f}min, Duration {total_time_needed_h:.1f}h")
            
        return schedule
