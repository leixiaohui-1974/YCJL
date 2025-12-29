"""
冰期物理模型
============

基于IAHR/ASCE权威文献的冰期水力学仿真模型:
1. 冰厚生长模型 (Stefan方程, Ashton模型)
2. 冰盖水力学模型 (复合糙率, 过流能力折减)
3. Frazil冰生成与传输模型
4. 冰塞形成与演化模型
5. 开河过程模型

版本: 3.3.0

参考文献:
- Stefan (1891) - Ice thickness growth model
- Belokon-Sabaneev - Composite roughness formula
- Ashton (1986) - River and Lake Ice Engineering
- Shen & Chiang (1984) - Frazil ice production model
- Pariset et al. (1966) - Ice cover stability
- Beltaos (1983, 2003) - Ice jam and breakup
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum, auto
from datetime import datetime, timedelta

from ..config.ice_parameters import (
    IceType, IcePhase, BreakupType,
    IceParams, IcePhysical, StefanParams, RoughnessParams,
    FrazilParams, AnchorParams, CoverParams, JamParams,
    BreakupParams, ThermalParams, YCJLIceParams
)


# ==========================================
# 状态数据结构
# ==========================================
@dataclass
class IceState:
    """冰期状态"""
    phase: IcePhase = IcePhase.OPEN_WATER
    ice_thickness: float = 0.0          # 冰厚 (m)
    ice_cover_fraction: float = 0.0     # 冰盖覆盖率 (0-1)
    frazil_concentration: float = 0.0   # Frazil冰浓度 (体积分数)
    anchor_ice_volume: float = 0.0      # 锚冰体积 (m³/m)
    water_temperature: float = 4.0      # 水温 (°C)
    supercooling: float = 0.0           # 过冷度 (°C)
    afdd: float = 0.0                   # 累计冻结度日 (°C·day)
    mdd: float = 0.0                    # 融化度日 (°C·day)
    composite_roughness: float = 0.014  # 复合糙率
    conveyance_factor: float = 1.0      # 输水能力折减系数


@dataclass
class ChannelGeometry:
    """断面几何参数"""
    width: float              # 河宽/断面宽 (m)
    depth: float              # 水深 (m)
    area: float               # 过水面积 (m²)
    wetted_perimeter: float   # 湿周 (m)
    hydraulic_radius: float   # 水力半径 (m)
    slope: float = 0.0001     # 底坡
    bed_roughness: float = 0.025  # 床面糙率


@dataclass
class MeteoCondition:
    """气象条件"""
    air_temperature: float      # 气温 (°C)
    wind_speed: float = 2.0     # 风速 (m/s)
    solar_radiation: float = 0.0  # 太阳辐射 (W/m²)
    cloud_cover: float = 0.5    # 云量 (0-1)
    humidity: float = 0.7       # 相对湿度 (0-1)
    snow_depth: float = 0.0     # 积雪深度 (m)


# ==========================================
# 冰厚生长模型
# ==========================================
class IceThicknessModel:
    """
    冰厚生长模型

    实现Stefan方程和Ashton改进模型
    """

    def __init__(self, params: 'IceParams' = None):
        self.params = params or IceParams
        self.stefan = self.params.Stefan
        self.physical = self.params.Physical
        self.thermal = self.params.Thermal

    def stefan_growth(self, afdd: float, snow_depth: float = 0.0) -> float:
        """
        Stefan方程计算冰厚

        h = α × √(AFDD)

        考虑雪盖修正:
        h = α × β_snow × √(AFDD)

        Args:
            afdd: 累计冻结度日 (°C·day)
            snow_depth: 积雪深度 (m)

        Returns:
            冰厚 (m)
        """
        if afdd <= 0:
            return 0.0

        # 选择Stefan系数
        if snow_depth > 0.1:
            alpha = self.stefan.ALPHA_SNOW_COVERED
            # Michel雪盖修正
            beta_snow = self.stefan.MICHEL_SNOW_FACTOR
        else:
            alpha = self.stefan.ALPHA_RIVER_ICE
            beta_snow = 1.0

        # 计算冰厚 (α单位: cm/√(°C·day))
        h = (alpha / 100.0) * beta_snow * math.sqrt(afdd)

        return min(h, self.stefan.MAX_ICE_THICKNESS)

    def ashton_growth(self, current_thickness: float, air_temp: float,
                      water_temp: float, dt: float,
                      snow_depth: float = 0.0) -> float:
        """
        Ashton改进模型计算冰厚增量

        考虑:
        - 冰-气界面热交换
        - 水-冰界面热交换
        - 雪盖隔热效应

        dh/dt = (q_ia - q_wi) / (ρ_i × L)

        Args:
            current_thickness: 当前冰厚 (m)
            air_temp: 气温 (°C)
            water_temp: 水温 (°C)
            dt: 时间步长 (s)
            snow_depth: 积雪深度 (m)

        Returns:
            冰厚增量 (m)
        """
        if air_temp >= 0:
            # 融化模式
            return self._melt_rate(current_thickness, air_temp, dt)

        # 冻结模式
        k_ice = self.physical.THERMAL_CONDUCTIVITY_ICE
        k_snow = self.physical.THERMAL_CONDUCTIVITY_SNOW
        h_ia = self.stefan.ASHTON_H_IA
        h_wi = self.stefan.ASHTON_H_WI
        rho_i = self.physical.RHO_ICE
        L = self.physical.LATENT_HEAT_FUSION

        # 计算等效热阻
        R_ice = current_thickness / k_ice if current_thickness > 0 else 0
        R_snow = snow_depth / k_snow if snow_depth > 0 else 0
        R_ia = 1.0 / h_ia
        R_wi = 1.0 / h_wi

        R_total = R_ice + R_snow + R_ia + R_wi

        # 计算热通量
        T_freeze = 0.0  # 冻结点
        delta_T = T_freeze - air_temp  # 温差

        if R_total > 0:
            q = delta_T / R_total
        else:
            q = 0

        # 水-冰界面热通量
        q_wi = h_wi * (water_temp - T_freeze)

        # 净热通量用于冻结
        q_net = max(0, q - q_wi)

        # 冰厚增量
        dh = q_net * dt / (rho_i * L)

        return dh

    def _melt_rate(self, ice_thickness: float, air_temp: float, dt: float) -> float:
        """计算融化速率"""
        if air_temp <= 0 or ice_thickness <= 0:
            return 0.0

        # 度日法
        melt_rate = self.thermal.MELT_RATE_DEGREE_DAY  # m/(°C·day)
        dt_days = dt / 86400.0

        dh = -melt_rate * air_temp * dt_days

        return max(dh, -ice_thickness)  # 不能融化超过现有冰厚


# ==========================================
# 冰盖水力学模型
# ==========================================
class IceCoverHydraulics:
    """
    冰盖水力学模型

    计算冰盖条件下的:
    - 复合糙率
    - 水力半径修正
    - 过流能力折减
    """

    def __init__(self, params: 'IceParams' = None):
        self.params = params or IceParams
        self.roughness = self.params.Roughness

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
        n_c = term**(2.0/3.0)

        # 限制在合理范围
        n_c = max(n_c, self.roughness.N_COMPOSITE_MIN)
        n_c = min(n_c, self.roughness.N_COMPOSITE_MAX)

        return n_c

    def ice_roughness_from_thickness(self, ice_thickness: float,
                                     ice_type: IceType = IceType.SHEET) -> float:
        """
        根据冰厚和冰类型估算冰底糙率

        基于Li (2012): n_i = a + b × (k_s / h_i)

        Args:
            ice_thickness: 冰厚 (m)
            ice_type: 冰类型

        Returns:
            冰底糙率
        """
        if ice_thickness <= 0:
            return 0.0

        # 根据冰类型选择粗糙高度
        ks_map = {
            IceType.SHEET: self.roughness.KS_SMOOTH_ICE,
            IceType.BORDER: self.roughness.KS_RIPPLED_ICE,
            IceType.FRAZIL: self.roughness.KS_FRAZIL_ACCUMULATION,
            IceType.JAM: self.roughness.KS_FRAZIL_ACCUMULATION * 2,
            IceType.SLUSH: self.roughness.KS_FRAZIL_ACCUMULATION,
        }

        k_s = ks_map.get(ice_type, self.roughness.KS_SMOOTH_ICE)

        # Li公式
        n_i = self.roughness.LI_COEF_A + self.roughness.LI_COEF_B * (k_s / ice_thickness)

        return min(n_i, self.roughness.N_ICE_JAM)

    def hydraulic_radius_ice_cover(self, geometry: ChannelGeometry,
                                   ice_thickness: float) -> float:
        """
        冰盖条件下的水力半径

        R = A / (P_bed + P_ice)

        Args:
            geometry: 断面几何
            ice_thickness: 冰厚 (m)

        Returns:
            修正后的水力半径 (m)
        """
        # 冰盖占据的面积
        area_ice = geometry.width * ice_thickness
        area_water = geometry.area - area_ice

        if area_water <= 0:
            return 0.0

        # 总湿周 = 床面湿周 + 冰盖宽度
        total_perimeter = geometry.wetted_perimeter + geometry.width

        return area_water / total_perimeter

    def conveyance_factor(self, n_open: float, n_composite: float,
                          R_open: float, R_ice: float) -> float:
        """
        计算过流能力折减系数

        K_ice / K_open = (n_open/n_c) × (R_ice/R_open)^(2/3)

        Args:
            n_open: 无冰期糙率
            n_composite: 复合糙率
            R_open: 无冰期水力半径
            R_ice: 冰盖条件水力半径

        Returns:
            折减系数 (0-1)
        """
        if n_composite <= 0 or R_open <= 0:
            return 1.0

        roughness_ratio = n_open / n_composite
        radius_ratio = (R_ice / R_open)**(2.0/3.0) if R_ice > 0 else 0.5

        factor = roughness_ratio * radius_ratio

        return min(factor, 1.0)

    def manning_flow_ice_cover(self, geometry: ChannelGeometry,
                               ice_thickness: float,
                               n_bed: float, n_ice: float) -> float:
        """
        冰盖条件下的曼宁流量

        Q = (1/n_c) × A × R^(2/3) × S^(1/2)

        Args:
            geometry: 断面几何
            ice_thickness: 冰厚 (m)
            n_bed: 床面糙率
            n_ice: 冰底糙率

        Returns:
            流量 (m³/s)
        """
        n_c = self.belokon_sabaneev_roughness(n_bed, n_ice)
        R = self.hydraulic_radius_ice_cover(geometry, ice_thickness)

        # 冰下过水面积
        A = geometry.area - geometry.width * ice_thickness

        if A <= 0 or R <= 0:
            return 0.0

        Q = (1.0 / n_c) * A * R**(2.0/3.0) * math.sqrt(geometry.slope)

        return Q


# ==========================================
# Frazil冰模型
# ==========================================
class FrazilIceModel:
    """
    Frazil冰(冰晶)生成与传输模型

    基于Shen & Chiang (1984)和Chen et al. (2023)
    """

    def __init__(self, params: 'IceParams' = None):
        self.params = params or IceParams
        self.frazil = self.params.Frazil
        self.physical = self.params.Physical

    def check_frazil_conditions(self, velocity: float, depth: float,
                                 supercooling: float) -> bool:
        """
        检查Frazil冰生成条件

        条件:
        - 过冷度 > 阈值
        - 流速 > 临界流速 (保证紊动)
        - Froude数 > 临界值

        Args:
            velocity: 流速 (m/s)
            depth: 水深 (m)
            supercooling: 过冷度 (°C, 正值)

        Returns:
            是否满足生成条件
        """
        if supercooling < self.frazil.SUPERCOOLING_ONSET:
            return False

        if velocity < self.frazil.CRITICAL_VELOCITY_FRAZIL:
            return False

        # Froude数
        Fr = velocity / math.sqrt(9.81 * depth) if depth > 0 else 0

        if Fr < self.frazil.CRITICAL_FROUDE_FRAZIL:
            return False

        return True

    def production_rate(self, supercooling: float,
                        turbulence_intensity: float) -> float:
        """
        计算Frazil冰生成速率

        dC/dt = k × ΔT × TI

        Args:
            supercooling: 过冷度 (°C)
            turbulence_intensity: 紊动强度 (0-1)

        Returns:
            浓度增加速率 (1/s)
        """
        if supercooling < self.frazil.SUPERCOOLING_ONSET:
            return 0.0

        # 二次成核主导的生成
        k = self.frazil.NUCLEATION_RATE_COEF

        rate = k * supercooling * turbulence_intensity * 1e-10

        return rate

    def rise_and_accumulation(self, concentration: float, depth: float,
                              velocity: float, dt: float) -> Tuple[float, float]:
        """
        计算Frazil冰上浮和堆积

        Args:
            concentration: 当前浓度 (体积分数)
            depth: 水深 (m)
            velocity: 流速 (m/s)
            dt: 时间步长 (s)

        Returns:
            (剩余浓度, 堆积量 m/s)
        """
        if concentration <= 0:
            return 0.0, 0.0

        # 上浮速度
        w_r = self.frazil.FRAZIL_RISE_VELOCITY

        # 紊动抑制上浮
        if velocity > 0.5:
            suppression = min(1.0, velocity / 2.0)
        else:
            suppression = 0.0

        effective_rise = w_r * (1 - suppression)

        # 上浮量
        rise_fraction = effective_rise * dt / depth if depth > 0 else 0

        accumulated = concentration * rise_fraction
        remaining = concentration * (1 - rise_fraction)

        # 堆积速率
        accumulation_rate = accumulated * depth / dt if dt > 0 else 0

        return remaining, accumulation_rate

    def supercooling_evolution(self, T_water: float, T_air: float,
                               concentration: float,
                               heat_exchange_coef: float,
                               dt: float) -> float:
        """
        过冷度演化

        考虑:
        - 表面热损失
        - 冰晶释放潜热

        Args:
            T_water: 水温 (°C)
            T_air: 气温 (°C)
            concentration: Frazil浓度
            heat_exchange_coef: 换热系数 (W/(m²·K))
            dt: 时间步长 (s)

        Returns:
            新的水温 (°C)
        """
        rho_w = self.physical.RHO_WATER
        c_w = self.physical.SPECIFIC_HEAT_WATER
        L = self.physical.LATENT_HEAT_FUSION
        rho_i = self.physical.RHO_ICE

        # 表面热损失
        q_loss = heat_exchange_coef * (T_water - T_air)  # W/m²

        # 冰晶释放潜热 (正值表示释放热量)
        q_latent = concentration * rho_i * L  # J/m³

        # 温度变化 (简化为表面1m深度)
        depth = 1.0
        dT = (q_latent / (depth * dt) - q_loss / depth) * dt / (rho_w * c_w)

        T_new = T_water + dT

        # 保持在合理范围
        return max(T_new, -self.frazil.SUPERCOOLING_MAX)


# ==========================================
# 冰盖形成模型
# ==========================================
class IceCoverFormation:
    """
    冰盖形成过程模型

    包括:
    - 岸冰(Border ice)生长
    - 冰盖推进(Juxtaposition)
    - 冰盖增厚
    """

    def __init__(self, params: 'IceParams' = None):
        self.params = params or IceParams
        self.cover = self.params.Cover
        self.physical = self.params.Physical

    def check_stable_cover(self, velocity: float, depth: float) -> bool:
        """
        检查是否可形成稳定冰盖

        基于Pariset准则:
        - 流速 < 临界流速
        - Froude数 < 临界Froude数

        Args:
            velocity: 流速 (m/s)
            depth: 水深 (m)

        Returns:
            是否可形成稳定冰盖
        """
        if velocity > self.cover.CRITICAL_VELOCITY_COVER:
            return False

        Fr = velocity / math.sqrt(9.81 * depth) if depth > 0 else 0

        return Fr < self.cover.CRITICAL_FROUDE_COVER

    def border_ice_growth(self, current_width: float, channel_width: float,
                          afdd_rate: float, dt_days: float) -> float:
        """
        岸冰横向生长

        Args:
            current_width: 当前岸冰宽度 (m) (单侧)
            channel_width: 河道宽度 (m)
            afdd_rate: AFDD日增量 (°C)
            dt_days: 时间步长 (天)

        Returns:
            新的岸冰宽度 (m)
        """
        if afdd_rate <= 0:
            return current_width

        # 生长速率
        growth_rate = self.cover.BORDER_ICE_GROWTH_RATE  # m/(day·°C)

        growth = growth_rate * abs(afdd_rate) * dt_days

        new_width = current_width + growth

        # 限制最大宽度
        max_width = min(self.cover.BORDER_ICE_MAX_WIDTH, channel_width / 2)

        return min(new_width, max_width)

    def ice_cover_progression(self, channel_length: float,
                               current_position: float,
                               is_stable: bool, dt_days: float) -> float:
        """
        冰盖沿河道推进

        Args:
            channel_length: 河道长度 (m)
            current_position: 当前冰盖前缘位置 (m from upstream)
            is_stable: 是否满足稳定条件
            dt_days: 时间步长 (天)

        Returns:
            新的前缘位置 (m)
        """
        if not is_stable:
            return current_position

        # 推进速度 (km/day -> m/day)
        speed = self.cover.ICE_COVER_PROGRESSION_SPEED * 1000

        progress = speed * dt_days

        new_position = current_position + progress

        return min(new_position, channel_length)

    def cover_fraction(self, border_width: float, channel_width: float,
                       cover_position: float, channel_length: float) -> float:
        """
        计算冰盖覆盖率

        Args:
            border_width: 岸冰宽度 (单侧) (m)
            channel_width: 河道宽度 (m)
            cover_position: 冰盖前缘位置 (m)
            channel_length: 河道长度 (m)

        Returns:
            覆盖率 (0-1)
        """
        # 岸冰覆盖
        border_fraction = 2 * border_width / channel_width if channel_width > 0 else 0

        # 完整冰盖覆盖
        cover_fraction = cover_position / channel_length if channel_length > 0 else 0

        # 综合覆盖率
        total = max(border_fraction, cover_fraction)

        return min(total, 1.0)


# ==========================================
# 开河模型
# ==========================================
class BreakupModel:
    """
    开河过程模型

    判断热力开河vs机械开河
    模拟开河波传播
    """

    def __init__(self, params: 'IceParams' = None):
        self.params = params or IceParams
        self.breakup = self.params.Breakup
        self.thermal = self.params.Thermal
        self.physical = self.params.Physical

    def ice_strength_decay(self, initial_strength: float, mdd: float) -> float:
        """
        冰盖强度衰减

        σ(t) = σ_0 × exp(-k × MDD)

        Args:
            initial_strength: 初始强度 (Pa)
            mdd: 融化度日 (°C·day)

        Returns:
            当前强度 (Pa)
        """
        k = self.breakup.STRENGTH_DECAY_COEF

        return initial_strength * math.exp(-k * mdd / 30.0)

    def check_thermal_breakup(self, mdd: float, ice_thickness: float) -> bool:
        """
        检查热力开河条件

        Args:
            mdd: 融化度日 (°C·day)
            ice_thickness: 冰厚 (m)

        Returns:
            是否满足热力开河条件
        """
        if mdd < self.breakup.MDD_THRESHOLD:
            return False

        if ice_thickness > self.breakup.ICE_THICKNESS_CRITICAL * 2:
            return False

        return True

    def check_mechanical_breakup(self, discharge_ratio: float,
                                  stage_rise_rate: float) -> bool:
        """
        检查机械开河条件

        Args:
            discharge_ratio: 流量增加比 (Q_current / Q_base)
            stage_rise_rate: 水位上涨速率 (m/day)

        Returns:
            是否满足机械开河条件
        """
        if discharge_ratio > self.breakup.DISCHARGE_INCREASE_RATIO:
            return True

        if stage_rise_rate > self.breakup.STAGE_RISE_RATE_CRITICAL:
            return True

        return False

    def classify_breakup(self, mdd: float, ice_thickness: float,
                         discharge_ratio: float,
                         stage_rise_rate: float) -> BreakupType:
        """
        判断开河类型

        Args:
            mdd: 融化度日 (°C·day)
            ice_thickness: 冰厚 (m)
            discharge_ratio: 流量增加比
            stage_rise_rate: 水位上涨速率 (m/day)

        Returns:
            开河类型
        """
        thermal = self.check_thermal_breakup(mdd, ice_thickness)
        mechanical = self.check_mechanical_breakup(discharge_ratio, stage_rise_rate)

        if thermal and mechanical:
            return BreakupType.MIXED
        elif thermal:
            return BreakupType.THERMAL
        elif mechanical:
            return BreakupType.MECHANICAL
        else:
            return BreakupType.THERMAL  # 默认

    def breakup_wave_speed(self, channel_slope: float,
                           discharge_ratio: float) -> float:
        """
        开河波传播速度

        Args:
            channel_slope: 河道坡度
            discharge_ratio: 流量增加比

        Returns:
            传播速度 (m/s)
        """
        base_speed = self.breakup.BREAKUP_WAVE_SPEED * 1000 / 3600  # km/h -> m/s

        # 坡度和流量修正
        slope_factor = math.sqrt(channel_slope / 0.0001)
        discharge_factor = math.sqrt(discharge_ratio)

        return base_speed * slope_factor * discharge_factor


# ==========================================
# 综合冰期仿真器
# ==========================================
class IcePeriodSimulator:
    """
    综合冰期仿真器

    整合所有冰期物理过程
    """

    def __init__(self, params: 'IceParams' = None):
        self.params = params or IceParams

        # 子模型
        self.thickness_model = IceThicknessModel(params)
        self.hydraulics = IceCoverHydraulics(params)
        self.frazil_model = FrazilIceModel(params)
        self.cover_model = IceCoverFormation(params)
        self.breakup_model = BreakupModel(params)

        # 状态
        self.state = IceState()
        self.history: List[IceState] = []

    def initialize(self, initial_water_temp: float = 4.0):
        """初始化仿真"""
        self.state = IceState(
            phase=IcePhase.OPEN_WATER,
            water_temperature=initial_water_temp
        )
        self.history = []

    def step(self, geometry: ChannelGeometry, meteo: MeteoCondition,
             velocity: float, dt: float) -> IceState:
        """
        执行一个仿真步

        Args:
            geometry: 断面几何
            meteo: 气象条件
            velocity: 流速 (m/s)
            dt: 时间步长 (s)

        Returns:
            更新后的状态
        """
        dt_days = dt / 86400.0

        # 更新度日
        if meteo.air_temperature < 0:
            self.state.afdd += abs(meteo.air_temperature) * dt_days
        else:
            self.state.mdd += meteo.air_temperature * dt_days

        # 相态判断和转换
        self._update_phase(geometry, meteo, velocity, dt)

        # 根据相态更新状态
        if self.state.phase == IcePhase.OPEN_WATER:
            self._simulate_open_water(geometry, meteo, velocity, dt)
        elif self.state.phase == IcePhase.FREEZE_UP:
            self._simulate_freeze_up(geometry, meteo, velocity, dt)
        elif self.state.phase == IcePhase.FROZEN:
            self._simulate_frozen(geometry, meteo, velocity, dt)
        elif self.state.phase == IcePhase.BREAKUP:
            self._simulate_breakup(geometry, meteo, velocity, dt)

        # 更新水力学参数
        self._update_hydraulics(geometry)

        # 保存历史
        self.history.append(IceState(**self.state.__dict__))

        return self.state

    def _update_phase(self, geometry: ChannelGeometry, meteo: MeteoCondition,
                      velocity: float, dt: float):
        """更新冰期相态"""

        if self.state.phase == IcePhase.OPEN_WATER:
            # 检查是否开始封冻
            if meteo.air_temperature < 0 and self.state.afdd > StefanParams.AFDD_ICE_START:
                self.state.phase = IcePhase.FREEZE_UP

        elif self.state.phase == IcePhase.FREEZE_UP:
            # 检查是否形成稳定冰盖
            if self.state.ice_cover_fraction > 0.9:
                self.state.phase = IcePhase.FROZEN
            # 或者回到无冰期
            elif meteo.air_temperature > 5 and self.state.ice_thickness < 0.01:
                self.state.phase = IcePhase.OPEN_WATER

        elif self.state.phase == IcePhase.FROZEN:
            # 检查开河条件
            if self.state.mdd > BreakupParams.MDD_THRESHOLD * 0.5:
                self.state.phase = IcePhase.BREAKUP

        elif self.state.phase == IcePhase.BREAKUP:
            # 完全开河
            if self.state.ice_thickness < 0.01:
                self.state.phase = IcePhase.OPEN_WATER
                self.state.afdd = 0
                self.state.mdd = 0

    def _simulate_open_water(self, geometry: ChannelGeometry,
                             meteo: MeteoCondition, velocity: float, dt: float):
        """无冰期仿真"""
        # 水温演化
        if meteo.air_temperature < 4:
            cooling_rate = 0.5  # °C/day
            self.state.water_temperature -= cooling_rate * dt / 86400
            self.state.water_temperature = max(0.0, self.state.water_temperature)

        # 过冷度
        self.state.supercooling = max(0, -self.state.water_temperature)

    def _simulate_freeze_up(self, geometry: ChannelGeometry,
                            meteo: MeteoCondition, velocity: float, dt: float):
        """封冻期仿真"""
        dt_days = dt / 86400.0

        # Frazil冰生成
        if self.frazil_model.check_frazil_conditions(velocity, geometry.depth,
                                                     self.state.supercooling):
            prod_rate = self.frazil_model.production_rate(
                self.state.supercooling,
                min(1.0, velocity / 2.0)
            )
            self.state.frazil_concentration += prod_rate * dt
            self.state.frazil_concentration = min(
                self.state.frazil_concentration,
                FrazilParams.FRAZIL_CONC_MAX
            )

        # Frazil上浮堆积
        remaining, accumulation = self.frazil_model.rise_and_accumulation(
            self.state.frazil_concentration, geometry.depth, velocity, dt
        )
        self.state.frazil_concentration = remaining

        # 冰厚增长
        if self.cover_model.check_stable_cover(velocity, geometry.depth):
            # Stefan方程
            h_stefan = self.thickness_model.stefan_growth(
                self.state.afdd, meteo.snow_depth
            )
            self.state.ice_thickness = h_stefan

            # 岸冰生长计算覆盖率
            self.state.ice_cover_fraction = min(
                1.0,
                self.state.ice_thickness / 0.1 + accumulation * dt_days
            )

        # 水温保持在冰点附近
        self.state.water_temperature = 0.0
        self.state.supercooling = 0.0

    def _simulate_frozen(self, geometry: ChannelGeometry,
                         meteo: MeteoCondition, velocity: float, dt: float):
        """稳定封冻期仿真"""
        # 冰厚生长 (Ashton模型)
        dh = self.thickness_model.ashton_growth(
            self.state.ice_thickness,
            meteo.air_temperature,
            self.state.water_temperature,
            dt,
            meteo.snow_depth
        )
        self.state.ice_thickness += dh
        self.state.ice_thickness = max(0, self.state.ice_thickness)

        # 完全覆盖
        self.state.ice_cover_fraction = 1.0

    def _simulate_breakup(self, geometry: ChannelGeometry,
                          meteo: MeteoCondition, velocity: float, dt: float):
        """开河期仿真"""
        dt_days = dt / 86400.0

        # 冰厚融化
        if meteo.air_temperature > 0:
            melt = ThermalParams.MELT_RATE_DEGREE_DAY * meteo.air_temperature * dt_days
            self.state.ice_thickness -= melt
            self.state.ice_thickness = max(0, self.state.ice_thickness)

        # 覆盖率下降
        decay_rate = 0.1  # per day
        self.state.ice_cover_fraction *= (1 - decay_rate * dt_days)

    def _update_hydraulics(self, geometry: ChannelGeometry):
        """更新水力学参数"""
        if self.state.ice_thickness > 0 and self.state.ice_cover_fraction > 0.5:
            # 计算冰底糙率
            n_ice = self.hydraulics.ice_roughness_from_thickness(
                self.state.ice_thickness, IceType.SHEET
            )

            # 复合糙率
            self.state.composite_roughness = self.hydraulics.belokon_sabaneev_roughness(
                geometry.bed_roughness, n_ice
            )

            # 过流能力折减
            R_open = geometry.hydraulic_radius
            R_ice = self.hydraulics.hydraulic_radius_ice_cover(
                geometry, self.state.ice_thickness
            )

            self.state.conveyance_factor = self.hydraulics.conveyance_factor(
                geometry.bed_roughness,
                self.state.composite_roughness,
                R_open, R_ice
            )
        else:
            self.state.composite_roughness = geometry.bed_roughness
            self.state.conveyance_factor = 1.0

    def get_design_ice_thickness(self, afdd: float) -> float:
        """获取设计冰厚"""
        return self.thickness_model.stefan_growth(afdd)

    def get_roughness_increase(self) -> float:
        """获取糙率增加比例"""
        if self.state.ice_cover_fraction < 0.5:
            return 1.0
        return self.state.composite_roughness / 0.014  # 相对于正常期


# ==========================================
# 导出
# ==========================================
__all__ = [
    # 数据结构
    'IceState',
    'ChannelGeometry',
    'MeteoCondition',
    # 子模型
    'IceThicknessModel',
    'IceCoverHydraulics',
    'FrazilIceModel',
    'IceCoverFormation',
    'BreakupModel',
    # 综合仿真器
    'IcePeriodSimulator'
]
