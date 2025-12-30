
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

# ==============================================================================
# L5 级数字孪生：密云调蓄工程全生命周期方案比选与优化系统
# ==============================================================================
# 覆盖项目建议书中的4个方案，并深度模拟已建方案(方案一)的运行痛点

class ControlMode(Enum):
    TRADITIONAL = "传统人工控制"
    L5_AUTONOMOUS = "L5级自主脉冲/联动"

@dataclass
class OperationalMetrics:
    mode_desc: str
    energy_kwh: float
    vibration_risk: str
    manual_intervention: str
    efficiency: float
    note: str

class SchemeSimulator:
    def __init__(self):
        # 基础参数
        self.target_Q = 6.0  # 痛点流量: 6 m3/s (设计流量的30%)
        self.duration = 24.0 # 模拟时长 (h)
        self.Q_design = 20.0
        
    def _calc_pump_power(self, Q, H, eta):
        if Q <= 0 or eta <= 0: return 0
        return 9.81 * Q * H / eta

    def _get_efficiency_curve(self, q_ratio, type="Axial"):
        """模拟轴流泵(Axial)和离心泵(Centrifugal)的效率衰减"""
        if type == "Axial": # 方案一多为轴流泵，马鞍区显著
            if q_ratio < 0.4: return 0.4 # 极低效率且不稳定
            return 0.85 * (1 - 2 * (q_ratio - 1.0)**2)
        else: # 管道方案多为离心泵
            return 0.80 * (1 - 0.5 * (q_ratio - 1.0)**2)

    # --------------------------------------------------------------------------
    # 方案一：京密引水渠反向输水 (已建方案 - The Built Reality)
    # --------------------------------------------------------------------------
    def simulate_scheme_1_built(self, control_mode: ControlMode):
        """
        核心痛点仿真：
        1. 低流量下扬程极低 -> 需'甩站'(Skipping)或'翻板闸憋压'(Flap Gate)
        2. 传统模式 vs L5模式
        """
        # 物理现状：在6个流量下，明渠沿程阻力极小，导致泵站静扬程不足 0.5m
        # 但水泵最小稳定扬程 H_min = 1.2m
        real_head_needed = 0.5 
        pump_min_head = 1.2
        
        if control_mode == ControlMode.TRADITIONAL:
            # === 传统痛点：人工操作 ===
            # 方式A: 甩站 (Skipping) - 直接停泵，靠上级余压自流
            # 风险：流量不可控，易发生漫堤或抽空
            
            # 方式B: 翻板闸憋压 (Flap Gate Throttling) - 最常用的无奈之举
            # 人为制造阻力，让扬程由 0.5 -> 1.2m，消耗多余能量来换取稳定
            h_operating = pump_min_head
            waste_head = h_operating - real_head_needed # 0.7m 被翻板闸浪费了
            
            eff = self._get_efficiency_curve(self.target_Q/self.Q_design, "Axial")
            power = self._calc_pump_power(self.target_Q, h_operating, eff)
            
            return OperationalMetrics(
                mode_desc="翻板闸憋压运行",
                energy_kwh=power * self.duration,
                vibration_risk="中 (靠憋压强行稳定)",
                manual_intervention="高 (需频繁调闸)",
                efficiency=eff * (real_head_needed/h_operating), # 真实系统效率极低
                note=f"痛点：人为增加 {waste_head:.1f}m 水头损失以避免喘振，能效极低。"
            )
            
        else:
            # === L5优化：PWM脉冲 + 智能甩站 ===
            # 策略：利用渠道库容。不憋压，而是全速运行一段时间，利用高效率。
            duty_cycle = self.target_Q / self.Q_design # 0.3
            
            # 运行时，流量大，扬程自然恢复到设计值附近 (e.g., 1.5m)
            h_pulse = 1.5
            eff_pulse = self._get_efficiency_curve(1.0, "Axial") # 高效区
            power_pulse = self._calc_pump_power(self.Q_design, h_pulse, eff_pulse)
            
            avg_energy = power_pulse * duty_cycle * self.duration
            
            return OperationalMetrics(
                mode_desc="L5级 脉冲/智能甩站",
                energy_kwh=avg_energy,
                vibration_risk="无 (避开马鞍区)",
                manual_intervention="零 (自主决策)",
                efficiency=eff_pulse,
                note="利用明渠库容'削峰填谷'，彻底消除翻板闸能耗。"
            )

    # --------------------------------------------------------------------------
    # 方案二：全线管道输水 (PCCP) - The Pipeline Alternative
    # --------------------------------------------------------------------------
    def simulate_scheme_2_pipe(self):
        # 刚性系统，无库容，只能变频
        # 痛点：低流量下流速慢，易淤积(如果是原水)，且无法脉冲
        q_ratio = self.target_Q / self.Q_design
        eff = self._get_efficiency_curve(q_ratio, "Centrifugal")
        # 管道阻力降低，扬程下降
        h_run = 1.5 + 0.5 * (q_ratio**2)
        power = self._calc_pump_power(self.target_Q, h_run, eff)
        
        return OperationalMetrics(
            mode_desc="连续变频运行",
            energy_kwh=power * self.duration,
            vibration_risk="低",
            manual_intervention="低",
            efficiency=eff,
            note="投资巨大，且在非设计工况下，水泵长期处于低效区，无法利用蓄能优化。"
        )

    # --------------------------------------------------------------------------
    # 方案三：明渠 + 局部管道混合 (Hybrid) - The Compromise
    # --------------------------------------------------------------------------
    def simulate_scheme_3_hybrid(self):
        # 假设前段明渠，后段管道
        # 继承了方案一的控制复杂性和方案二的部分成本
        return OperationalMetrics(
            mode_desc="分段混合控制",
            energy_kwh=2500.0, # 估算中间值
            vibration_risk="中",
            manual_intervention="极高 (需协调渠管接口)",
            efficiency=0.60,
            note="渠管衔接处的水位控制是极大的难点，易溢流。"
        )

    # --------------------------------------------------------------------------
    # 方案四：深埋隧洞 (Tunnel) - The Visionary
    # --------------------------------------------------------------------------
    def simulate_scheme_4_tunnel(self):
        # TBM施工，全线自流或少级泵站
        # 投资天价
        return OperationalMetrics(
            mode_desc="深层调水",
            energy_kwh=1500.0, # 能耗最低
            vibration_risk="无",
            manual_intervention="低",
            efficiency=0.90,
            note="虽然运行最优，但建设期风险极大，且不可逆。"
        )

    def run_full_analysis(self):
        print(f"{'='*100}")
        print(f"🚀 密云调蓄工程：L5级数字孪生全生命周期方案比选与优化分析")
        print(f"🎯 仿真场景：非设计工况 Q={self.target_Q} m3/s (痛点流量)")
        print(f"{'='*100}")
        
        # 1. 运行各方案
        s1_trad = self.simulate_scheme_1_built(ControlMode.TRADITIONAL)
        s1_l5   = self.simulate_scheme_1_built(ControlMode.L5_AUTONOMOUS)
        s2      = self.simulate_scheme_2_pipe()
        s3      = self.simulate_scheme_3_hybrid()
        s4      = self.simulate_scheme_4_tunnel()
        
        # 2. 输出对比表
        print(f"\n{'-'*100}")
        print(f"{'方案名称':<15} | {'控制模式':<15} | {'日能耗(kWh)':<12} | {'振动风险':<8} | {'真实效率':<8} | {'核心点评'}")
        print(f"{'-'*100}")
        
        print(f"{'方案一(现状)':<15} | {s1_trad.mode_desc:<15} | {s1_trad.energy_kwh:<12.0f} | {s1_trad.vibration_risk:<8} | {s1_trad.efficiency*100:.0f}%      | {s1_trad.note}")
        print(f"{'方案一(L5优化)':<13} | {s1_l5.mode_desc:<15} | {s1_l5.energy_kwh:<12.0f} | {s1_l5.vibration_risk:<8} | {s1_l5.efficiency*100:.0f}%      | {s1_l5.note}")
        print(f"{'-'*100}")
        print(f"{'方案二(管道)':<15} | {s2.mode_desc:<15} | {s2.energy_kwh:<12.0f} | {s2.vibration_risk:<8} | {s2.efficiency*100:.0f}%      | {s2.note}")
        print(f"{'方案三(混合)':<15} | {s3.mode_desc:<15} | {s3.energy_kwh:<12.0f} | {s3.vibration_risk:<8} | {s3.efficiency*100:.0f}%      | {s3.note}")
        print(f"{'方案四(隧洞)':<15} | {s4.mode_desc:<15} | {s4.energy_kwh:<12.0f} | {s4.vibration_risk:<8} | {s4.efficiency*100:.0f}%      | {s4.note}")
        
        print(f"\n💡 结论分析：")
        print(f"1. 【事前遗憾】：方案四(隧洞)运行性能最好，但因造价被否；方案二(管道)在低流量下表现平平。")
        print(f"2. 【事中痛点】：方案一(现状)在传统控制下，被迫使用'翻板闸憋压'，导致能耗激增(比L5模式高{(s1_trad.energy_kwh-s1_l5.energy_kwh)/s1_l5.energy_kwh*100:.0f}%)，且存在振动隐患。")
        print(f"3. 【事后优化】：引入L5级数字孪生后，方案一可激活'脉冲输水'潜力，其能效反超方案二，成为最具性价比的智能运行方案。")

if __name__ == "__main__":
    sim = SchemeSimulator()
    sim.run_full_analysis()
