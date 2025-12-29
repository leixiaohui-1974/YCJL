"""
冰期水力学参数数据库
==================

基于IAHR/ASCE权威文献和《水力学》冰力学章节:
1. IAHR Committee on Ice Research and Engineering
2. ASCE Journal of Hydraulic Engineering - River Ice Processes
3. HEC-RAS Ice Cover Modeling Technical Reference
4. Stefan (1891) - Ice thickness growth model
5. Belokon-Sabaneev - Composite roughness formula
6. Ashton (1986) - River ice modeling

版本: 3.3.0 - 冰期模型升级版

主要理论体系:
- 热力学模型 (Stefan方程, Ashton模型)
- 水力学模型 (复合糙率, 水力半径修正)
- 冰形成过程 (Frazil ice, Anchor ice, Border ice, Ice jam)
- 开河判据 (热力开河, 机械开河)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum, auto


class IceType(Enum):
    """冰类型"""
    FRAZIL = auto()         # 冰晶/流冰
    ANCHOR = auto()         # 锚冰
    BORDER = auto()         # 岸冰
    SHEET = auto()          # 冰盖
    JAM = auto()            # 冰塞
    SLUSH = auto()          # 冰浆


class IcePhase(Enum):
    """冰期阶段"""
    OPEN_WATER = auto()     # 无冰期
    FREEZE_UP = auto()      # 封冻期
    FROZEN = auto()         # 稳定封冻期
    BREAKUP = auto()        # 开河期
    ICE_RUN = auto()        # 流冰期


class BreakupType(Enum):
    """开河类型"""
    THERMAL = auto()        # 热力开河
    MECHANICAL = auto()     # 机械开河
    MIXED = auto()          # 混合型


# ==========================================
# 1. 冰的物理性质参数
# ==========================================
@dataclass
class IcePhysicalProperties:
    """冰的物理性质参数"""

    # 基本物性
    RHO_ICE: float = 917.0              # 冰密度 (kg/m³)
    RHO_WATER: float = 999.8            # 水密度 @0°C (kg/m³)
    LATENT_HEAT_FUSION: float = 334000  # 熔化潜热 (J/kg)
    SPECIFIC_HEAT_ICE: float = 2090     # 冰比热容 (J/(kg·K))
    SPECIFIC_HEAT_WATER: float = 4186   # 水比热容 (J/(kg·K))

    # 热传导
    THERMAL_CONDUCTIVITY_ICE: float = 2.22      # 冰热导率 (W/(m·K))
    THERMAL_CONDUCTIVITY_WATER: float = 0.58    # 水热导率 (W/(m·K))
    THERMAL_CONDUCTIVITY_SNOW: float = 0.31     # 雪热导率 (W/(m·K)) - 新雪

    # 力学性质
    YOUNGS_MODULUS_ICE: float = 9.5e9   # 冰弹性模量 (Pa)
    POISSON_RATIO_ICE: float = 0.33     # 冰泊松比
    TENSILE_STRENGTH: float = 1.0e6    # 抗拉强度 (Pa)
    FLEXURAL_STRENGTH: float = 0.8e6   # 抗弯强度 (Pa)
    COMPRESSIVE_STRENGTH: float = 5.0e6 # 抗压强度 (Pa)

    # 摩擦系数
    ICE_WATER_FRICTION: float = 0.02    # 冰-水摩擦系数
    ICE_ICE_FRICTION: float = 0.10      # 冰-冰摩擦系数
    ICE_BANK_FRICTION: float = 0.30     # 冰-岸摩擦系数


# ==========================================
# 2. Stefan方程参数 (冰厚生长模型)
# ==========================================
@dataclass
class StefanEquationParams:
    """
    Stefan方程参数

    基本方程: h = α × √(AFDD)
    其中:
    - h: 冰厚 (cm)
    - α: Stefan系数 (cm/√(°C·day))
    - AFDD: 累计冻结度日 (°C·day)
    """

    # Stefan系数 α (cm/√(°C·day))
    # 不同条件下的典型值
    ALPHA_CLEAR_ICE: float = 2.7        # 透明冰 (无雪盖)
    ALPHA_SNOW_COVERED: float = 2.0     # 有雪盖
    ALPHA_RIVER_ICE: float = 2.4        # 河冰 (水流影响)
    ALPHA_FRAZIL_ICE: float = 1.8       # 冰晶堆积

    # Michel修正系数 (考虑雪盖和气象条件)
    MICHEL_SNOW_FACTOR: float = 0.7     # 雪盖修正因子
    MICHEL_WIND_FACTOR: float = 1.1     # 风速修正因子

    # Ashton模型参数 (改进的Stefan方程)
    # 考虑气温、雪盖、风速的综合影响
    ASHTON_H_IA: float = 10.0           # 冰-气界面换热系数 (W/(m²·K))
    ASHTON_H_WI: float = 200.0          # 水-冰界面换热系数 (W/(m²·K))

    # 冰厚计算限制
    MIN_ICE_THICKNESS: float = 0.01     # 最小冰厚 (m)
    MAX_ICE_THICKNESS: float = 2.0      # 最大冰厚 (m)

    # AFDD阈值
    AFDD_ICE_START: float = 10.0        # 开始结冰的AFDD阈值 (°C·day)
    AFDD_STABLE_ICE: float = 100.0      # 稳定冰盖的AFDD阈值


# ==========================================
# 3. 复合糙率参数 (Belokon-Sabaneev公式)
# ==========================================
@dataclass
class CompositeRoughnessParams:
    """
    冰盖复合糙率参数

    Belokon-Sabaneev公式:
    n_c = ((n_b^1.5 + n_i^1.5) / 2)^(2/3)

    其中:
    - n_c: 复合曼宁糙率
    - n_b: 床面糙率
    - n_i: 冰底糙率
    """

    # 床面糙率 n_b 参考值
    N_BED_SMOOTH: float = 0.012         # 光滑床面
    N_BED_CONCRETE: float = 0.014       # 混凝土衬砌
    N_BED_GRAVEL: float = 0.025         # 砾石床面
    N_BED_NATURAL: float = 0.035        # 天然河道

    # 冰底糙率 n_i 参考值 (Nezhikhovskiy, 1964; Beltaos, 2001)
    N_ICE_SMOOTH: float = 0.008         # 光滑冰底
    N_ICE_RIPPLED: float = 0.015        # 波纹冰底
    N_ICE_ROUGH: float = 0.025          # 粗糙冰底
    N_ICE_FRAZIL: float = 0.035         # 冰晶堆积
    N_ICE_JAM: float = 0.050            # 冰塞

    # Li (2012) 综合范围
    N_COMPOSITE_MIN: float = 0.013      # 复合糙率下限
    N_COMPOSITE_MAX: float = 0.040      # 复合糙率上限

    # 冰底糙率-冰厚关系 (Li, 2012)
    # n_i = a + b × (k_s / h_i)
    # k_s: 冰底粗糙高度, h_i: 冰厚
    LI_COEF_A: float = 0.010            # 常数项
    LI_COEF_B: float = 0.018            # 比例系数

    # 冰底粗糙高度 k_s (m)
    KS_SMOOTH_ICE: float = 0.001        # 光滑冰
    KS_RIPPLED_ICE: float = 0.01        # 波纹冰
    KS_ROUGH_ICE: float = 0.05          # 粗糙冰
    KS_FRAZIL_ACCUMULATION: float = 0.10  # 冰晶堆积


# ==========================================
# 4. Frazil冰参数 (冰晶生成)
# ==========================================
@dataclass
class FrazilIceParams:
    """
    Frazil冰(冰晶)生成参数

    基于: Daly (1984), Shen & Chiang (1984)
    过冷过程数学模型: Chen et al. (2023)
    """

    # 过冷度参数
    SUPERCOOLING_ONSET: float = 0.01    # 冰晶开始生成的过冷度 (°C)
    SUPERCOOLING_MAX: float = 0.05      # 最大过冷度 (°C)
    SUPERCOOLING_RESIDUAL: float = 0.002  # 残余过冷度 (°C)

    # 冰晶特性
    FRAZIL_CRYSTAL_SIZE: float = 0.003  # 典型冰晶尺寸 (m)
    FRAZIL_POROSITY: float = 0.40       # 冰晶堆积孔隙率
    FRAZIL_RISE_VELOCITY: float = 0.01  # 冰晶上浮速度 (m/s)

    # 临界条件 (IAHR标准)
    CRITICAL_FROUDE_FRAZIL: float = 0.10    # Frazil生成临界Froude数
    CRITICAL_VELOCITY_FRAZIL: float = 0.7   # Frazil生成临界流速 (m/s)

    # 二次成核参数
    NUCLEATION_RATE_COEF: float = 1.0e6     # 成核速率系数 (1/(m³·s))
    CRYSTAL_GROWTH_RATE: float = 1.0e-6     # 晶体生长速率 (m/s/°C)

    # 冰晶浓度
    FRAZIL_CONC_INITIAL: float = 1.0e-6     # 初始浓度 (体积分数)
    FRAZIL_CONC_CRITICAL: float = 0.05      # 临界浓度 (开始聚集)
    FRAZIL_CONC_MAX: float = 0.40           # 最大浓度 (冰浆)


# ==========================================
# 5. 锚冰参数 (Anchor Ice)
# ==========================================
@dataclass
class AnchorIceParams:
    """
    锚冰参数

    基于: Altberg (1936), Qu & Doering (2007)
    """

    # 形成条件
    CRITICAL_SUPERCOOLING: float = 0.02     # 锚冰形成临界过冷度 (°C)
    CRITICAL_VELOCITY: float = 0.3          # 形成临界流速 (m/s)
    CRITICAL_DEPTH: float = 3.0             # 形成最大水深 (m)

    # 锚冰特性
    ANCHOR_ICE_POROSITY: float = 0.50       # 锚冰孔隙率
    ANCHOR_ICE_DENSITY: float = 500.0       # 锚冰表观密度 (kg/m³)

    # 生长和释放
    GROWTH_RATE_COEF: float = 0.01          # 生长速率系数 (m/day/°C)
    RELEASE_BUOYANCY_RATIO: float = 1.5     # 释放临界浮力比

    # 对水力的影响
    HYDRAULIC_RESISTANCE_FACTOR: float = 1.5  # 水力阻力增加因子
    BED_ROUGHNESS_INCREASE: float = 0.010   # 床面糙率增量


# ==========================================
# 6. 冰盖形成参数
# ==========================================
@dataclass
class IceCoverFormationParams:
    """
    冰盖形成参数

    基于: Pariset et al. (1966), Shen et al. (1997)
    """

    # 临界条件
    CRITICAL_FROUDE_COVER: float = 0.08     # 冰盖稳定临界Froude数
    CRITICAL_VELOCITY_COVER: float = 0.6    # 冰盖稳定临界流速 (m/s)
    CRITICAL_VELOCITY_UNDERTURNING: float = 0.7  # 冰块翻转临界流速

    # 边缘冰进展
    BORDER_ICE_GROWTH_RATE: float = 0.02    # 岸冰横向生长速率 (m/day/°C)
    BORDER_ICE_MAX_WIDTH: float = 10.0      # 岸冰最大宽度 (m)

    # 冰盖推进
    ICE_COVER_PROGRESSION_SPEED: float = 0.1    # 冰盖推进速度 (km/day)
    JUXTAPOSITION_COEF: float = 0.8             # 并置系数

    # 冰盖增厚 (水力增厚)
    SHOVING_STRESS_COEF: float = 0.5            # 推挤应力系数
    THICKENING_VELOCITY_THRESHOLD: float = 0.7  # 水力增厚速度阈值 (m/s)


# ==========================================
# 7. 冰塞参数 (Ice Jam)
# ==========================================
@dataclass
class IceJamParams:
    """
    冰塞参数

    基于: Pariset & Hausser (1961), Beltaos (1983)
    """

    # 冰塞几何
    EQUILIBRIUM_THICKNESS_COEF: float = 1.0     # 平衡厚度系数
    POROSITY: float = 0.40                      # 冰塞孔隙率

    # 内摩擦角
    INTERNAL_FRICTION_ANGLE: float = 45.0       # 内摩擦角 (度)
    COHESION: float = 0.0                       # 粘聚力 (Pa)

    # 侧向应力系数
    LATERAL_STRESS_COEF_PASSIVE: float = 3.0    # 被动土压力系数
    LATERAL_STRESS_COEF_ACTIVE: float = 0.33    # 主动土压力系数

    # 水力参数
    HEAD_LOSS_COEF: float = 0.5                 # 冰塞水头损失系数
    SEEPAGE_VELOCITY: float = 0.1               # 渗流速度 (m/s)

    # 稳定性
    STABILITY_FROUDE: float = 0.12              # 稳定Froude数上限
    CRITICAL_SHEAR_STRESS: float = 50.0         # 临界剪切应力 (Pa)


# ==========================================
# 8. 开河判据参数
# ==========================================
@dataclass
class BreakupCriteriaParams:
    """
    开河判据参数

    基于: Beltaos (2003), Shen & Yapa (1985)
    """

    # 热力开河判据
    MDD_THRESHOLD: float = 25.0                 # 融化度日阈值 (°C·day)
    ICE_THICKNESS_CRITICAL: float = 0.05        # 临界冰厚 (m)
    POROSITY_CRITICAL: float = 0.35             # 临界孔隙度

    # 机械开河判据
    STAGE_RISE_RATE_CRITICAL: float = 0.5       # 临界水位上涨速率 (m/day)
    DISCHARGE_INCREASE_RATIO: float = 1.5       # 流量增加比

    # 冰盖强度衰减
    STRENGTH_DECAY_COEF: float = 0.1            # 强度衰减系数 (1/day)
    COMPETENCE_CRITICAL: float = 0.3            # 临界冰盖能力 (h×σ)

    # 热力/机械开河分界
    THERMAL_BREAKUP_THRESHOLD: float = 30.0     # 热力开河AFDD阈值
    MECHANICAL_BREAKUP_VELOCITY: float = 0.5    # 机械开河临界流速 (m/s)

    # 开河推进
    BREAKUP_WAVE_SPEED: float = 5.0             # 开河波速 (km/h)
    ICE_RUN_VELOCITY: float = 1.0               # 流冰速度 (m/s)


# ==========================================
# 9. 热交换参数
# ==========================================
@dataclass
class ThermalExchangeParams:
    """热交换参数"""

    # 表面热交换
    SOLAR_RADIATION_ABSORPTION: float = 0.30    # 太阳辐射吸收率
    ICE_ALBEDO_FRESH: float = 0.70              # 新冰反照率
    ICE_ALBEDO_MELTING: float = 0.40            # 融化冰反照率
    SNOW_ALBEDO: float = 0.85                   # 雪反照率

    # 对流换热系数
    H_AIR_CALM: float = 5.0                     # 静风空气换热系数 (W/(m²·K))
    H_AIR_WINDY: float = 20.0                   # 有风空气换热系数 (W/(m²·K))
    H_WATER_ICE: float = 200.0                  # 水-冰界面换热系数 (W/(m²·K))

    # 长波辐射
    EMISSIVITY_ICE: float = 0.97                # 冰发射率
    STEFAN_BOLTZMANN: float = 5.67e-8           # Stefan-Boltzmann常数 (W/(m²·K⁴))

    # 融化参数
    MELT_RATE_DEGREE_DAY: float = 0.005         # 度日融化速率 (m/°C·day)
    BOTTOM_MELT_COEF: float = 0.8               # 底部融化系数


# ==========================================
# 10. 引绰济辽工程冰期参数
# ==========================================
@dataclass
class YCJLIcePeriodParams:
    """
    引绰济辽工程冰期运行参数

    基于工程实际条件和东北地区气候特点
    """

    # 冰期时间 (月份)
    ICE_PERIOD_START_MONTH: int = 11            # 冰期开始 (11月)
    ICE_PERIOD_END_MONTH: int = 3               # 冰期结束 (3月)
    STABLE_ICE_MONTHS: List[int] = field(default_factory=lambda: [12, 1, 2])

    # 气温参数 (基于区域气象)
    AVG_TEMP_DEC: float = -15.0                 # 12月平均气温 (°C)
    AVG_TEMP_JAN: float = -20.0                 # 1月平均气温 (°C)
    AVG_TEMP_FEB: float = -15.0                 # 2月平均气温 (°C)
    DESIGN_MIN_TEMP: float = -35.0              # 设计最低气温 (°C)

    # 冰厚设计值
    DESIGN_ICE_THICKNESS: float = 0.80          # 设计冰厚 (m)
    MAX_ICE_THICKNESS: float = 1.20             # 最大冰厚 (m)
    AFDD_TYPICAL: float = 1500.0                # 典型AFDD (°C·day)

    # 隧洞冰期参数
    TUNNEL_ICE_ROUGHNESS: float = 0.018         # 冰期隧洞糙率
    TUNNEL_NORMAL_ROUGHNESS: float = 0.014      # 正常期隧洞糙率

    # 管道冰期参数
    PIPELINE_MIN_VELOCITY: float = 0.5          # 管道最小流速 (m/s)
    PIPELINE_MIN_TEMPERATURE: float = 0.5       # 管道最小水温 (°C)

    # 调度约束
    MIN_RESERVOIR_LEVEL_ICE: float = 365.0      # 冰期最低库水位 (m)
    MAX_DISCHARGE_CHANGE_RATE: float = 0.05     # 最大流量变化率 (%/h)
    STABLE_FLOW_DURATION: float = 48.0          # 稳定流量持续时间 (h)

    # 安全裕度
    FLOW_SAFETY_FACTOR: float = 0.85            # 流量安全系数
    PRESSURE_SAFETY_MARGIN: float = 5.0         # 压力安全裕度 (m)


# ==========================================
# 综合冰期配置类
# ==========================================
@dataclass
class IceHydraulicsConfig:
    """
    冰期水力学综合配置

    整合所有冰期相关参数
    """
    VERSION: str = "3.3.0"

    Physical: IcePhysicalProperties = field(default_factory=IcePhysicalProperties)
    Stefan: StefanEquationParams = field(default_factory=StefanEquationParams)
    Roughness: CompositeRoughnessParams = field(default_factory=CompositeRoughnessParams)
    Frazil: FrazilIceParams = field(default_factory=FrazilIceParams)
    Anchor: AnchorIceParams = field(default_factory=AnchorIceParams)
    Cover: IceCoverFormationParams = field(default_factory=IceCoverFormationParams)
    Jam: IceJamParams = field(default_factory=IceJamParams)
    Breakup: BreakupCriteriaParams = field(default_factory=BreakupCriteriaParams)
    Thermal: ThermalExchangeParams = field(default_factory=ThermalExchangeParams)
    YCJL: YCJLIcePeriodParams = field(default_factory=YCJLIcePeriodParams)

    # 公式计算方法
    def stefan_ice_thickness(self, afdd: float, alpha: float = None) -> float:
        """
        Stefan方程计算冰厚

        h = α × √(AFDD)

        Args:
            afdd: 累计冻结度日 (°C·day)
            alpha: Stefan系数，默认使用河冰值

        Returns:
            冰厚 (m)
        """
        if alpha is None:
            alpha = self.Stefan.ALPHA_RIVER_ICE
        # α单位是 cm/√(°C·day)，转换为 m
        h = (alpha / 100.0) * math.sqrt(max(afdd, 0))
        return min(h, self.Stefan.MAX_ICE_THICKNESS)

    def belokon_sabaneev_roughness(self, n_bed: float, n_ice: float) -> float:
        """
        Belokon-Sabaneev复合糙率公式

        n_c = ((n_b^1.5 + n_i^1.5) / 2)^(2/3)

        Args:
            n_bed: 床面糙率
            n_ice: 冰底糙率

        Returns:
            复合糙率
        """
        term = (n_bed**1.5 + n_ice**1.5) / 2.0
        return term**(2.0/3.0)

    def ice_cover_hydraulic_radius(self, area: float, wetted_perimeter_bed: float,
                                   ice_width: float) -> float:
        """
        冰盖条件下的水力半径

        R = A / (P_bed + P_ice)

        Args:
            area: 过水断面积 (m²)
            wetted_perimeter_bed: 床面湿周 (m)
            ice_width: 冰盖宽度 (m)

        Returns:
            水力半径 (m)
        """
        total_wetted_perimeter = wetted_perimeter_bed + ice_width
        if total_wetted_perimeter > 0:
            return area / total_wetted_perimeter
        return 0.0

    def frazil_production_rate(self, supercooling: float, turbulence_intensity: float) -> float:
        """
        Frazil冰生成速率

        基于过冷度和紊动强度

        Args:
            supercooling: 过冷度 (°C, 正值)
            turbulence_intensity: 紊动强度 (0-1)

        Returns:
            生成速率 (kg/(m³·s))
        """
        if supercooling < self.Frazil.SUPERCOOLING_ONSET:
            return 0.0

        # 简化模型: Q = k × ΔT × TI × ρ_i × L
        k = self.Frazil.NUCLEATION_RATE_COEF
        rho_i = self.Physical.RHO_ICE
        L = self.Physical.LATENT_HEAT_FUSION

        rate = k * supercooling * turbulence_intensity * rho_i / L
        return rate

    def is_thermal_breakup(self, mdd: float, ice_thickness: float,
                           discharge_ratio: float) -> bool:
        """
        判断是否为热力开河

        Args:
            mdd: 融化度日 (°C·day)
            ice_thickness: 当前冰厚 (m)
            discharge_ratio: 流量增加比

        Returns:
            True=热力开河, False=机械开河
        """
        thermal_score = 0
        mechanical_score = 0

        if mdd > self.Breakup.MDD_THRESHOLD:
            thermal_score += 2
        if ice_thickness < self.Breakup.ICE_THICKNESS_CRITICAL:
            thermal_score += 1
        if discharge_ratio > self.Breakup.DISCHARGE_INCREASE_RATIO:
            mechanical_score += 2

        return thermal_score > mechanical_score

    def conveyance_reduction_factor(self, n_composite: float, n_open: float) -> float:
        """
        计算冰盖导致的输水能力折减系数

        基于HEC-RAS技术参考

        Args:
            n_composite: 复合糙率
            n_open: 无冰期糙率

        Returns:
            折减系数 (0-1)
        """
        if n_open <= 0:
            return 1.0
        # K_ice / K_open ≈ (n_open / n_composite) × (R_ice / R_open)^(2/3)
        # 假设矩形断面 R_ice ≈ 0.5 × R_open
        roughness_ratio = n_open / n_composite
        radius_factor = 0.5**(2.0/3.0)  # ≈ 0.63
        return roughness_ratio * radius_factor


# 模块级实例
IceParams = IceHydraulicsConfig()

# 便捷访问
IcePhysical = IceParams.Physical
StefanParams = IceParams.Stefan
RoughnessParams = IceParams.Roughness
FrazilParams = IceParams.Frazil
AnchorParams = IceParams.Anchor
CoverParams = IceParams.Cover
JamParams = IceParams.Jam
BreakupParams = IceParams.Breakup
ThermalParams = IceParams.Thermal
YCJLIceParams = IceParams.YCJL


# ==========================================
# 导出
# ==========================================
__all__ = [
    # 枚举
    'IceType',
    'IcePhase',
    'BreakupType',
    # 参数类
    'IcePhysicalProperties',
    'StefanEquationParams',
    'CompositeRoughnessParams',
    'FrazilIceParams',
    'AnchorIceParams',
    'IceCoverFormationParams',
    'IceJamParams',
    'BreakupCriteriaParams',
    'ThermalExchangeParams',
    'YCJLIcePeriodParams',
    'IceHydraulicsConfig',
    # 实例
    'IceParams',
    'IcePhysical',
    'StefanParams',
    'RoughnessParams',
    'FrazilParams',
    'AnchorParams',
    'CoverParams',
    'JamParams',
    'BreakupParams',
    'ThermalParams',
    'YCJLIceParams'
]
