"""
全局物理常数 (Physical Constants)
=================================

定义水利工程中使用的基本物理常数和单位换算。
这些常数在所有项目中共享使用。
"""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class PhysicsConstants:
    """
    物理常数集合 (不可变)

    包含水力学计算所需的基本物理常数。
    使用frozen=True确保常数不被意外修改。
    """
    # 基本常数
    GRAVITY: float = 9.80665                    # 重力加速度 (m/s²) - CODATA推荐值
    WATER_DENSITY_20C: float = 998.2            # 水密度 @20°C (kg/m³)
    WATER_DENSITY_4C: float = 1000.0            # 水密度 @4°C (kg/m³)
    ATMOSPHERIC_PRESSURE: float = 101325.0       # 标准大气压 (Pa)
    ATMOSPHERIC_PRESSURE_HEAD: float = 10.33     # 大气压水头 (m)

    # 水的物理性质 @20°C
    KINEMATIC_VISCOSITY: float = 1.004e-6       # 运动粘度 (m²/s)
    DYNAMIC_VISCOSITY: float = 1.002e-3         # 动力粘度 (Pa·s)
    BULK_MODULUS: float = 2.2e9                 # 体积弹性模量 (Pa)
    VAPOR_PRESSURE_HEAD: float = -9.8           # 汽化压力水头 (m, 相对大气压)
    SURFACE_TENSION: float = 0.0728             # 表面张力 (N/m)

    # 冰的物理性质
    ICE_DENSITY: float = 917.0                  # 冰密度 (kg/m³)
    ICE_LATENT_HEAT: float = 334000.0           # 冰融化潜热 (J/kg)
    ICE_THERMAL_CONDUCTIVITY: float = 2.22      # 冰导热系数 (W/(m·K))
    WATER_THERMAL_CONDUCTIVITY: float = 0.598   # 水导热系数 @20°C (W/(m·K))

    # 材料弹性模量
    CONCRETE_YOUNGS_MODULUS: float = 35e9       # 混凝土弹性模量 (Pa)
    STEEL_YOUNGS_MODULUS: float = 200e9         # 钢材弹性模量 (Pa)
    PCCP_COMPOSITE_MODULUS: float = 45e9        # PCCP复合弹性模量 (Pa, 估算)

    # 单位换算
    M3_TO_BILLION_M3: float = 1e-9              # m³ -> 亿m³
    BILLION_M3_TO_M3: float = 1e9               # 亿m³ -> m³
    KM_TO_M: float = 1000.0                     # km -> m
    M_TO_KM: float = 0.001                      # m -> km
    MW_TO_KW: float = 1000.0                    # MW -> kW
    KW_TO_MW: float = 0.001                     # kW -> MW

    # 时间常数
    HOUR_TO_SECONDS: float = 3600.0             # h -> s
    DAY_TO_SECONDS: float = 86400.0             # d -> s
    YEAR_TO_SECONDS: float = 31536000.0         # 年 -> s (365天)

    @classmethod
    def water_density(cls, temperature: float = 20.0) -> float:
        """
        根据温度计算水密度 (简化公式)

        Args:
            temperature: 水温 (°C)

        Returns:
            水密度 (kg/m³)
        """
        # 简化的温度-密度关系 (适用于0-40°C)
        # ρ = 1000 * (1 - (T-4)²/119000)
        rho = 1000.0 * (1.0 - (temperature - 4.0)**2 / 119000.0)
        return max(rho, 950.0)  # 防止负值

    @classmethod
    def kinematic_viscosity(cls, temperature: float = 20.0) -> float:
        """
        根据温度计算运动粘度 (经验公式)

        Args:
            temperature: 水温 (°C)

        Returns:
            运动粘度 (m²/s)
        """
        # Vogel方程简化
        if temperature <= 0:
            return 1.79e-6
        return 1.79e-6 / (1.0 + 0.0337 * temperature + 0.000221 * temperature**2)

    @classmethod
    def wave_speed_pccp(cls, diameter: float, wall_thickness: float,
                        youngs_modulus: float = 45e9) -> float:
        """
        计算PCCP管道压力波速

        Args:
            diameter: 内径 (m)
            wall_thickness: 壁厚 (m)
            youngs_modulus: 弹性模量 (Pa)

        Returns:
            波速 (m/s)
        """
        import math
        # Korteweg公式: a = sqrt(K/ρ / (1 + K*D/(E*e)))
        K = cls.BULK_MODULUS
        rho = cls.WATER_DENSITY_20C
        D = diameter
        e = wall_thickness
        E = youngs_modulus

        factor = 1.0 + (K * D) / (E * e)
        return math.sqrt(K / rho / factor)


# ==========================================
# 模块级常量（便捷访问）
# ==========================================
_constants = PhysicsConstants()

GRAVITY = _constants.GRAVITY
WATER_DENSITY = _constants.WATER_DENSITY_20C
ATMOSPHERIC_PRESSURE_HEAD = _constants.ATMOSPHERIC_PRESSURE_HEAD
KINEMATIC_VISCOSITY = _constants.KINEMATIC_VISCOSITY
WATER_BULK_MODULUS = _constants.BULK_MODULUS
VAPOR_PRESSURE_HEAD = _constants.VAPOR_PRESSURE_HEAD

# 冰期相关
ICE_DENSITY = _constants.ICE_DENSITY
ICE_LATENT_HEAT = _constants.ICE_LATENT_HEAT

# 材料常数
CONCRETE_E = _constants.CONCRETE_YOUNGS_MODULUS
STEEL_E = _constants.STEEL_YOUNGS_MODULUS


__all__ = [
    'PhysicsConstants',
    'GRAVITY',
    'WATER_DENSITY',
    'ATMOSPHERIC_PRESSURE_HEAD',
    'KINEMATIC_VISCOSITY',
    'WATER_BULK_MODULUS',
    'VAPOR_PRESSURE_HEAD',
    'ICE_DENSITY',
    'ICE_LATENT_HEAT',
    'CONCRETE_E',
    'STEEL_E'
]
