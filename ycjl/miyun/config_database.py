"""
密云水库调蓄工程全系统参数数据库 (Configuration Database) v1.0
===============================================================

基于:
1. 《pccp管道数据.xlsx - 泵站参数.csv》
2. 《渠道参数技术表.xlsx》
3. 《pccp管道数据.xlsx - 雁栖-溪翁庄.csv》
4. 《空气阀参数.docx》

版本说明 (v1.0):
- 物理资产全参数数据库 (Full-Fidelity Database)
- L5级瞬态仿真和精细控制参数占位
- 支持稳态运行仿真
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum, auto

try:
    from scipy.interpolate import PchipInterpolator
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ==========================================
# 辅助函数：插值器
# ==========================================
def create_interpolator(data: List[Tuple[float, float]]) -> Callable[[float], float]:
    """
    创建保形插值器

    使用PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
    保证单调性，避免过冲振荡
    """
    x = np.array([p[0] for p in data])
    y = np.array([p[1] for p in data])

    if HAS_SCIPY and len(data) >= 3:
        interpolator = PchipInterpolator(x, y)
        def interp_func(val: float) -> float:
            return float(interpolator(np.clip(val, x.min(), x.max())))
        return interp_func
    else:
        # 线性插值回退
        def linear_interp(val: float) -> float:
            return float(np.interp(val, x, y))
        return linear_interp


# ==========================================
# 1. 路由类型枚举
# ==========================================
class RouteType(Enum):
    """输水路由类型"""
    CHANNEL = "京密引水渠 (明渠)"
    PIPELINE = "PCCP/钢管 (有压)"


# ==========================================
# 2. 泵站数据库
# ==========================================
STATION_DB: Dict[str, Dict] = {
    # ==========================
    # 第一段：明渠梯级泵站 (Station 1-6)
    # ==========================
    "Tundian": {
        "id": 1,
        "name": "屯佃枢纽",
        "type": RouteType.CHANNEL,
        "pump": {
            "Q_des": 20.0,          # 设计流量 (m³/s)
            "Count": 4,             # 机组数量
            "Power": 315,           # 单机功率 (kW)
            "RPM": 245,             # 额定转速 (rpm)
            "H_des": 1.08,          # 设计扬程 (m)
            "H_max": 1.50,          # 最大扬程 (m)
            "H_min": 0.11,          # 最小扬程 (m)
            "Eff_Peak": 0.82,       # 峰值效率
            # [MISSING] 瞬态惯性数据
            "Inertia_GD2": None,    # kg*m² (缺失: 用于停泵水锤计算)
            "Hill_Chart": None      # (缺失: 用于全叶片角度效率寻优)
        },
        "gate": {
            "Type": "Sluice",       # 节制闸类型
            "Cd_Curve": None        # [MISSING] 流量系数曲线 Q = f(e, dH)
        },
        "levels": {
            "Inlet_Min": 48.38,     # 进水池最低水位 (m)
            "Inlet_Max": 49.26,     # 进水池最高水位 (m)
            "Outlet_Des": 49.68,    # 出水池设计水位 (m)
            "Outlet_Max": 49.88     # 出水池最高水位 (m)
        },
        "channel_geo": {
            "Length_km": 10.5,      # 渠道长度 (km)
            "Slope": 5.8e-5,        # 纵坡
            "Roughness": 0.020,     # 糙率
            "Bottom_W": 20.0,       # 底宽 (m)
            "Side_Slope": 2.5,      # 边坡
            "Roughness_Seasonality": None  # [MISSING] 糙率随季节变化系数
        }
    },

    "Qianliulin": {
        "id": 2,
        "name": "前柳林枢纽",
        "type": RouteType.CHANNEL,
        "pump": {
            "Q_des": 20.0,
            "Count": 4,
            "Power": 355,
            "RPM": 245,
            "H_des": 1.60,
            "H_max": 2.20,
            "H_min": 1.27,
            "Eff_Peak": 0.83,
            "Inertia_GD2": None
        },
        "gate": {"Type": "Sluice", "Cd_Curve": None},
        "levels": {
            "Inlet_Min": 48.60,
            "Inlet_Max": 49.39,
            "Outlet_Des": 50.76,
            "Outlet_Max": 50.80
        },
        "channel_geo": {
            "Length_km": 11.2,
            "Slope": 5.8e-5,
            "Roughness": 0.029,
            "Bottom_W": 20.0,
            "Side_Slope": 2.5,
            "Roughness_Seasonality": None
        }
    },

    "Niantou": {
        "id": 3,
        "name": "埝头枢纽",
        "type": RouteType.CHANNEL,
        "pump": {
            "Q_des": 20.0,
            "Count": 4,
            "Power": 400,
            "RPM": 245,
            "H_des": 2.21,
            "H_max": 2.45,
            "H_min": 2.06,
            "Eff_Peak": 0.84,
            "Inertia_GD2": None
        },
        "gate": {"Type": "Sluice", "Cd_Curve": None},
        "levels": {
            "Inlet_Min": 49.42,
            "Inlet_Max": 49.72,
            "Outlet_Des": 51.83,
            "Outlet_Max": 51.87
        },
        "channel_geo": {
            "Length_km": 9.8,
            "Slope": 6.2e-5,
            "Roughness": 0.040,
            "Bottom_W": 20.0,
            "Side_Slope": 2.5,
            "Roughness_Seasonality": None
        }
    },

    "Xingshou": {
        "id": 4,
        "name": "兴寿枢纽",
        "type": RouteType.CHANNEL,
        "pump": {
            "Q_des": 20.0,
            "Count": 4,
            "Power": 355,
            "RPM": 245,
            "H_des": 1.97,
            "H_max": 2.21,
            "H_min": 1.82,
            "Eff_Peak": 0.83,
            "Inertia_GD2": None
        },
        "gate": {"Type": "Sluice", "Cd_Curve": None},
        "levels": {
            "Inlet_Min": 50.53,
            "Inlet_Max": 50.83,
            "Outlet_Des": 52.70,
            "Outlet_Max": 52.74
        },
        "channel_geo": {
            "Length_km": 12.0,
            "Slope": 5.8e-5,
            "Roughness": 0.026,
            "Bottom_W": 20.0,
            "Side_Slope": 2.5,
            "Roughness_Seasonality": None
        }
    },

    "Lishishan": {
        "id": 5,
        "name": "李史山枢纽",
        "type": RouteType.CHANNEL,
        "pump": {
            "Q_des": 20.0,
            "Count": 4,
            "Power": 355,
            "RPM": 245,
            "H_des": 1.59,
            "H_max": 2.04,
            "H_min": 1.29,
            "Eff_Peak": 0.83,
            "Inertia_GD2": None
        },
        "gate": {"Type": "Sluice", "Cd_Curve": None},
        "levels": {
            "Inlet_Min": 51.30,
            "Inlet_Max": 51.60,
            "Outlet_Des": 53.09,
            "Outlet_Max": 53.34
        },
        "channel_geo": {
            "Length_km": 8.5,
            "Slope": 5.6e-5,
            "Roughness": 0.016,
            "Bottom_W": 20.0,
            "Side_Slope": 2.5,
            "Roughness_Seasonality": None
        }
    },

    "Xitaishang": {
        "id": 6,
        "name": "西台上枢纽",
        "type": RouteType.CHANNEL,
        "pump": {
            "Q_des": 20.0,
            "Count": 4,
            "Power": 1000,
            "RPM": 290,
            "H_des": 6.18,
            "H_max": 8.18,
            "H_min": 4.31,
            "Eff_Peak": 0.85,
            "Inertia_GD2": None
        },
        "gate": {"Type": "Sluice", "Cd_Curve": None},
        "levels": {
            "Inlet_Min": 51.92,
            "Inlet_Max": 52.92,
            "Outlet_Des": 58.81,
            "Outlet_Max": 60.10
        },
        "channel_geo": {
            "Length_km": 5.0,
            "Slope": 5.6e-5,
            "Roughness": 0.021,
            "Bottom_W": 20.0,
            "Side_Slope": 2.5,
            "Roughness_Seasonality": None
        }
    },

    # ==========================
    # 第二段：有压管路段 (Station 7-9)
    # ==========================
    "Guojiawu": {
        "id": 7,
        "name": "郭家坞枢纽",
        "type": RouteType.PIPELINE,
        "pump": {
            "Q_des": 10.0,
            "Count": 3,
            "Power": 400,
            "RPM": 290,
            "H_des": 2.54,
            "H_max": 4.23,
            "H_min": 0.89,
            "Eff_Peak": 0.80,
            "Inertia_GD2": None
        },
        "levels": {
            "Inlet_Min": 57.00,
            "Inlet_Max": 60.00,
            "Outlet_Des": 61.04,
            "Outlet_Max": 61.23
        },
        "pipe_geo": {
            "Length_m": 5000,           # 管道长度 (m)
            "Diameter_mm": 2600,        # 管径 (mm)
            "Roughness": 0.012,         # 摩阻系数
            "Static_Head": 2.5,         # 静扬程 (m)
            "Wave_Speed_a": None,       # [MISSING] 波速 (用于水锤计算)
            "Valve_Closure_Curve": None # [MISSING] 阀门关闭规律
        }
    },

    "Yanqi": {
        "id": 8,
        "name": "雁栖枢纽",
        "type": RouteType.PIPELINE,
        "pump": {
            "Q_des": 10.0,
            "Count": 3,
            "Power": 4000,
            "RPM": 495,
            "H_des": 32.55,
            "H_max": 33.85,
            "H_min": 30.9,
            "Eff_Peak": 0.86,
            "Inertia_GD2": None         # [CRITICAL MISSING] 极重要! 高扬程泵站停泵水锤关键
        },
        "levels": {
            "Inlet_Min": 59.65,
            "Inlet_Max": 60.60,
            "Outlet_Des": 92.50,
            "Outlet_Max": 93.50
        },
        "pipe_geo": {
            "Length_m": 21496.0,
            "Diameter_mm": 2600,
            "Roughness": 0.012,
            "Inlet_Elev": 52.24,        # 进水高程 (m)
            "Outlet_Elev": 88.80,       # 出水高程 (m)
            "Wave_Speed_a": None,       # [MISSING]
            "Valve_Closure_Curve": None,# [MISSING]
            "Critical_Nodes": [
                {"name": "Start", "loc": 0, "elev": 52.24},
                {"name": "AV-4", "loc": 3200, "elev": 52.21},
                {"name": "AV-16", "loc": 14500, "elev": 64.59},
                {"name": "End", "loc": 21496, "elev": 88.80}
            ]
        }
    },

    "Xiwengzhuang": {
        "id": 9,
        "name": "溪翁庄枢纽",
        "type": RouteType.PIPELINE,
        "pump": {
            "Q_des": 10.0,
            "Count": 3,
            "Power": 4000,
            "RPM": 495,
            "H_des": 52.5,
            "H_max": 59.5,
            "H_min": 38.5,
            "Eff_Peak": 0.86,
            "Inertia_GD2": None,        # [CRITICAL MISSING]
            "Suter_Curve": None         # [MISSING] 全特性曲线 (反转特性)
        },
        "levels": {
            "Inlet_Min": 91.50,
            "Inlet_Max": 93.50,
            "Outlet_Des": 145.0,
            "Outlet_Max": 155.0
        },
        "pipe_geo": {
            "Length_m": 150,
            "Diameter_mm": 2600,
            "Roughness": 0.012,
            "Static_Head": 50.0,
            "Wave_Speed_a": None        # [MISSING]
        }
    }
}


# ==========================================
# 3. 全局物理与仿真常数
# ==========================================
@dataclass
class MiyunGlobalPhysicsConfig:
    """密云水库调蓄工程全局物理常数与仿真参数"""
    # 物理常数
    G: float = 9.80665                      # 重力加速度 (m/s²)
    RHO_WATER: float = 998.2                # 水密度 (kg/m³) @20°C
    PATM_HEAD: float = 10.33                # 大气压水头 (m)
    KINEMATIC_VISCOSITY: float = 1.004e-6   # 运动粘度 (m²/s) @20°C
    BULK_MODULUS: float = 2.2e9             # 水体积弹性模量 (Pa)
    VAPOR_PRESSURE_HEAD: float = -9.8       # 汽化压力水头 (m, 相对大气压)

    # 仿真时间步长
    DT_PHYSICS: float = 0.5                 # 物理仿真步长 (s) - MOC Courant条件
    DT_SCADA: float = 1.0                   # SCADA采样周期 (s)
    DT_L1_REFLEX: float = 0.01              # L1反射层响应周期 (s)
    DT_L2_MPC: float = 60.0                 # L2 MPC采样周期 (s)
    MPC_HORIZON: int = 600                  # MPC预测时域步数 (10分钟)

    # 数值稳定性
    MIN_DEPTH: float = 0.001                # 最小水深 (m)
    MIN_FLOW: float = 1e-6                  # 最小流量 (m³/s)
    MAX_ITERATIONS: int = 100               # 最大迭代次数
    CONVERGENCE_TOL: float = 1e-6           # 收敛容差


# ==========================================
# 4. 密云水库特性曲线
# ==========================================
@dataclass
class MiyunCharacteristicCurves:
    """
    密云水库调蓄工程特性曲线数字化查找表

    所有曲线基于工程实测数据或设计文件数字化
    """

    # ---------------------------------------------------------
    # 4.1 密云水库：水位-库容-面积
    # ---------------------------------------------------------
    # 水位(m) -> 库容(m³)
    MIYUN_ZV_DATA: List[Tuple[float, float]] = field(default_factory=lambda: [
        (120.0, 0.0),               # 库底
        (130.0, 2.00e8),
        (140.0, 8.00e8),
        (145.0, 16.00e8),           # 死水位
        (150.0, 25.00e8),
        (155.0, 35.00e8),
        (157.5, 43.75e8),           # 正常蓄水位
        (160.0, 50.00e8)            # 校核洪水位
    ])

    # 水位(m) -> 水面面积(m²)
    MIYUN_ZA_DATA: List[Tuple[float, float]] = field(default_factory=lambda: [
        (120.0, 0.0),
        (130.0, 30.0e6),
        (140.0, 80.0e6),
        (145.0, 120.0e6),
        (150.0, 160.0e6),
        (155.0, 200.0e6),
        (157.5, 220.0e6),
        (160.0, 240.0e6)
    ])

    # ---------------------------------------------------------
    # 4.2 泵站效率曲线
    # ---------------------------------------------------------
    # 轴流泵效率曲线 (Q/Q_des -> Efficiency)
    PUMP_EFFICIENCY_CURVE: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.4, 0.60),
        (0.5, 0.70),
        (0.6, 0.78),
        (0.7, 0.83),
        (0.8, 0.86),
        (0.9, 0.87),
        (1.0, 0.85),                # 设计点
        (1.1, 0.82),
        (1.2, 0.75)
    ])

    # ---------------------------------------------------------
    # 4.3 明渠水力损失系数
    # ---------------------------------------------------------
    # 渠道进出口局部损失系数
    CHANNEL_LOCAL_LOSS_COEF: Dict[str, float] = field(default_factory=lambda: {
        "inlet": 0.5,               # 进口损失系数
        "outlet": 1.0,              # 出口损失系数
        "bend_90": 0.5,             # 90度弯道
        "bend_45": 0.25,            # 45度弯道
        "gate_full_open": 0.05      # 闸门全开
    })

    # ---------------------------------------------------------
    # 4.4 管道水力损失系数
    # ---------------------------------------------------------
    PIPE_LOCAL_LOSS_COEF: Dict[str, float] = field(default_factory=lambda: {
        "inlet": 0.5,
        "outlet": 1.0,
        "tee": 0.4,
        "valve_butterfly": 0.3,
        "check_valve": 1.5,
        "elbow_90": 0.3,
        "reducer": 0.2
    })

    # ---------------------------------------------------------
    # 4.5 月度运行调度限制
    # ---------------------------------------------------------
    # 月份 -> (最大输水流量, 最小输水流量) 单位: m³/s
    MONTHLY_FLOW_LIMITS: Dict[int, Tuple[float, float]] = field(default_factory=lambda: {
        1:  (15.0, 5.0),            # 冬季低流量
        2:  (15.0, 5.0),
        3:  (18.0, 8.0),            # 春季
        4:  (20.0, 10.0),
        5:  (20.0, 12.0),           # 夏季高需水期
        6:  (20.0, 12.0),
        7:  (20.0, 10.0),
        8:  (20.0, 10.0),
        9:  (18.0, 8.0),            # 秋季
        10: (18.0, 8.0),
        11: (15.0, 5.0),            # 入冬
        12: (15.0, 5.0)
    })

    def get_miyun_volume(self, level: float) -> float:
        """密云水库：水位->库容"""
        x = [p[0] for p in self.MIYUN_ZV_DATA]
        y = [p[1] for p in self.MIYUN_ZV_DATA]
        return float(np.interp(level, x, y))

    def get_miyun_area(self, level: float) -> float:
        """密云水库：水位->面积"""
        x = [p[0] for p in self.MIYUN_ZA_DATA]
        y = [p[1] for p in self.MIYUN_ZA_DATA]
        return float(np.interp(level, x, y))

    def get_miyun_level(self, volume: float) -> float:
        """密云水库：库容->水位 (反查)"""
        x = [p[1] for p in self.MIYUN_ZV_DATA]
        y = [p[0] for p in self.MIYUN_ZV_DATA]
        return float(np.interp(volume, x, y))

    def get_pump_efficiency(self, q_ratio: float) -> float:
        """泵效率：流量比->效率"""
        x = [p[0] for p in self.PUMP_EFFICIENCY_CURVE]
        y = [p[1] for p in self.PUMP_EFFICIENCY_CURVE]
        return float(np.interp(q_ratio, x, y))

    def get_monthly_flow_limits(self, month: int) -> Tuple[float, float]:
        """获取月度流量限制"""
        return self.MONTHLY_FLOW_LIMITS.get(month, (20.0, 5.0))


# ==========================================
# 5. 密云水库配置
# ==========================================
@dataclass
class MiyunReservoirConfig:
    """密云水库参数"""
    NAME: str = "密云水库"
    LOCATION: str = "北京市密云区"

    # 特征水位 (m)
    NORMAL_LEVEL: float = 157.50            # 正常蓄水位
    DEAD_LEVEL: float = 145.00              # 死水位
    FLOOD_LIMIT_LEVEL: float = 153.00       # 汛限水位
    CHECK_FLOOD_LEVEL: float = 160.00       # 校核洪水位
    DESIGN_FLOOD_LEVEL: float = 158.50      # 设计洪水位

    # 库容 (亿m³)
    TOTAL_STORAGE: float = 43.75            # 总库容
    USEFUL_STORAGE: float = 27.75           # 兴利库容
    DEAD_STORAGE: float = 16.00             # 死库容
    FLOOD_CONTROL_STORAGE: float = 6.25     # 防洪库容

    # 坝体参数
    DAM_CREST_EL: float = 162.00            # 坝顶高程 (m)
    DAM_LENGTH: float = 627.0               # 坝长 (m)
    DAM_HEIGHT: float = 66.0                # 坝高 (m)


# ==========================================
# 6. 京密引水渠配置
# ==========================================
@dataclass
class JingMiChannelConfig:
    """京密引水渠系统参数"""
    NAME: str = "京密引水渠"
    TOTAL_LENGTH_KM: float = 57.0           # 总长度 (km)

    # 设计参数
    DESIGN_FLOW: float = 20.0               # 设计流量 (m³/s)
    MAX_VELOCITY: float = 1.2               # 最大流速 (m/s)

    # 断面参数 (梯形断面)
    BOTTOM_WIDTH: float = 20.0              # 底宽 (m)
    SIDE_SLOPE: float = 2.5                 # 边坡 (m/m)
    DESIGN_DEPTH: float = 3.0               # 设计水深 (m)

    # 水力参数
    AVERAGE_SLOPE: float = 5.8e-5           # 平均纵坡
    AVERAGE_ROUGHNESS: float = 0.025        # 平均糙率

    # 泵站数量
    PUMP_STATION_COUNT: int = 6             # 明渠段泵站数


# ==========================================
# 7. 有压管道配置
# ==========================================
@dataclass
class MiyunPipelineConfig:
    """密云水库调蓄工程有压管道参数"""
    NAME: str = "PCCP/钢管输水管道"
    TOTAL_LENGTH_M: float = 26646.0         # 总长度 (m) = 5000+21496+150

    # 管道规格
    INNER_DIAMETER: float = 2.6             # 内径 (m) DN2600
    WALL_THICKNESS: float = 0.30            # 壁厚 (m)

    # 材料参数
    YOUNGS_MODULUS_CONCRETE: float = 35e9   # 混凝土弹性模量 (Pa)
    YOUNGS_MODULUS_STEEL: float = 200e9     # 钢筒弹性模量 (Pa)
    POISSON_RATIO: float = 0.2              # 泊松比

    # 水力参数
    WAVE_SPEED: float = 1000.0              # 压力波速 (m/s) 估计值
    DARCY_FRICTION: float = 0.012           # 达西摩阻系数

    # 设计压力 (m水头)
    DESIGN_PRESSURE: float = 80.0           # 设计压力
    MAX_WORKING_PRESSURE: float = 95.0      # 最大工作压力
    TEST_PRESSURE: float = 120.0            # 试验压力

    # 泵站数量
    PUMP_STATION_COUNT: int = 3             # 有压管段泵站数

    @property
    def cross_section_area(self) -> float:
        """管道断面积"""
        return math.pi * (self.INNER_DIAMETER / 2) ** 2


# ==========================================
# 8. 安全设施配置
# ==========================================
@dataclass
class MiyunSafetyConfig:
    """安全设施参数"""

    # 空气阀参数
    AIR_VALVE_COUNT: int = 50               # 空气阀数量 (估计)
    AIR_VALVE_SPACING: float = 500.0        # 平均间距 (m)

    # 压力报警阈值
    PRESSURE_ALARM_HIGH: float = 90.0       # 高压报警 (m)
    PRESSURE_ALARM_LOW: float = -3.0        # 负压报警 (m)
    PRESSURE_TRIP_HIGH: float = 95.0        # 超压停机 (m)

    # 水位报警阈值
    LEVEL_ALARM_MARGIN: float = 0.3         # 水位报警裕度 (m)

    # 负压保护临界高程点
    CRITICAL_NEGATIVE_PRESSURE_POINTS: List[Dict] = field(default_factory=lambda: [
        {"name": "AV-4", "loc_m": 3200, "elev_m": 52.21, "margin_m": 5.0},
        {"name": "AV-16", "loc_m": 14500, "elev_m": 64.59, "margin_m": 5.0}
    ])


# ==========================================
# 9. 控制参数配置
# ==========================================
@dataclass
class MiyunControlConfig:
    """控制算法参数"""

    # PID参数 (水位控制)
    PID_LEVEL_KP: float = 0.08              # 比例增益
    PID_LEVEL_KI: float = 0.003             # 积分增益
    PID_LEVEL_KD: float = 0.015             # 微分增益
    PID_INTEGRAL_LIMIT: float = 8.0         # 积分限幅

    # MPC参数
    MPC_PREDICTION_HORIZON: int = 20        # 预测时域
    MPC_CONTROL_HORIZON: int = 5            # 控制时域
    MPC_SAMPLE_TIME: float = 60.0           # 采样时间 (s)

    # 安全边界
    SAFETY_MARGIN_PRESSURE: float = 8.0     # 压力安全裕度 (m)
    SAFETY_MARGIN_LEVEL: float = 0.4        # 水位安全裕度 (m)

    # 泵站联锁
    PUMP_START_DELAY: float = 30.0          # 泵启动延时 (s)
    PUMP_STOP_DELAY: float = 60.0           # 泵停止延时 (s)
    PUMP_SEQUENCE_INTERVAL: float = 15.0    # 泵顺序启动间隔 (s)


# ==========================================
# 10. 总成配置类
# ==========================================
@dataclass
class MiyunProjectConfig:
    """
    密云水库调蓄工程全系统参数总成 (Database V1.0)

    用于生产环境部署的完整配置集
    """
    # 版本信息
    VERSION: str = "1.0.0"
    BUILD_DATE: str = "2024-12-30"
    PROJECT_NAME: str = "密云水库调蓄工程"

    # 各子系统配置
    Global: MiyunGlobalPhysicsConfig = field(default_factory=MiyunGlobalPhysicsConfig)
    Curves: MiyunCharacteristicCurves = field(default_factory=MiyunCharacteristicCurves)
    Reservoir: MiyunReservoirConfig = field(default_factory=MiyunReservoirConfig)
    Channel: JingMiChannelConfig = field(default_factory=JingMiChannelConfig)
    Pipeline: MiyunPipelineConfig = field(default_factory=MiyunPipelineConfig)
    Safety: MiyunSafetyConfig = field(default_factory=MiyunSafetyConfig)
    Control: MiyunControlConfig = field(default_factory=MiyunControlConfig)

    def validate(self) -> List[str]:
        """
        验证配置完整性和一致性

        Returns:
            错误信息列表，空列表表示验证通过
        """
        errors = []

        # 检查水位逻辑
        if self.Reservoir.DEAD_LEVEL >= self.Reservoir.NORMAL_LEVEL:
            errors.append("死水位必须低于正常蓄水位")

        if self.Reservoir.FLOOD_LIMIT_LEVEL >= self.Reservoir.CHECK_FLOOD_LEVEL:
            errors.append("汛限水位必须低于校核洪水位")

        # 检查泵站数据库完整性
        total_stations = len(STATION_DB)
        expected_stations = self.Channel.PUMP_STATION_COUNT + self.Pipeline.PUMP_STATION_COUNT
        if total_stations != expected_stations:
            errors.append(f"泵站数量不匹配: 数据库{total_stations}个, 期望{expected_stations}个")

        # 检查压力约束
        if self.Safety.PRESSURE_TRIP_HIGH <= self.Pipeline.DESIGN_PRESSURE:
            errors.append("超压停机阈值必须高于设计压力")

        return errors

    def get_summary(self) -> Dict:
        """获取配置摘要"""
        return {
            'version': self.VERSION,
            'project_name': self.PROJECT_NAME,
            'reservoir': {
                'name': self.Reservoir.NAME,
                'normal_level': self.Reservoir.NORMAL_LEVEL,
                'total_storage_billion_m3': self.Reservoir.TOTAL_STORAGE
            },
            'conveyance': {
                'channel_length_km': self.Channel.TOTAL_LENGTH_KM,
                'pipeline_length_km': self.Pipeline.TOTAL_LENGTH_M / 1000,
                'design_flow': self.Channel.DESIGN_FLOW
            },
            'pump_stations': {
                'channel_count': self.Channel.PUMP_STATION_COUNT,
                'pipeline_count': self.Pipeline.PUMP_STATION_COUNT,
                'total_count': len(STATION_DB)
            }
        }

    def get_station_by_id(self, station_id: int) -> Optional[Dict]:
        """根据ID获取泵站配置"""
        for key, data in STATION_DB.items():
            if data["id"] == station_id:
                return {key: data}
        return None

    def get_stations_by_type(self, route_type: RouteType) -> Dict[str, Dict]:
        """根据类型获取泵站列表"""
        return {k: v for k, v in STATION_DB.items() if v["type"] == route_type}


# ==========================================
# 模块级实例
# ==========================================
# 全局配置实例
MiyunParams = MiyunProjectConfig()

# 便捷访问
MiyunGlobalConfig = MiyunParams.Global
MiyunCurveDatabase = MiyunParams.Curves
MiyunReservoirCfg = MiyunParams.Reservoir
MiyunChannelCfg = MiyunParams.Channel
MiyunPipelineCfg = MiyunParams.Pipeline
MiyunSafetyCfg = MiyunParams.Safety
MiyunControlCfg = MiyunParams.Control


# ==========================================
# 导出
# ==========================================
__all__ = [
    'RouteType',
    'STATION_DB',
    'MiyunProjectConfig',
    'MiyunParams',
    'MiyunGlobalPhysicsConfig',
    'MiyunCharacteristicCurves',
    'MiyunReservoirConfig',
    'JingMiChannelConfig',
    'MiyunPipelineConfig',
    'MiyunSafetyConfig',
    'MiyunControlConfig',
    'MiyunGlobalConfig',
    'MiyunCurveDatabase',
    'MiyunReservoirCfg',
    'MiyunChannelCfg',
    'MiyunPipelineCfg',
    'MiyunSafetyCfg',
    'MiyunControlCfg',
    'create_interpolator'
]
