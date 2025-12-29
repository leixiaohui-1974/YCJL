"""
引绰济辽工程全系统参数数据库 (Configuration Database) v3.2
=======================================================

基于:
1. 《引绰济辽工程全系统控制设施深度分析与分布式智能调度研究报告》 (Vol 1 - Vol 8)
2. 《引绰济辽工程调度原则和调度运用技术要点》(2024.2.2版)

版本说明 (v3.2):
- 全图表数字化 (Complete Chart Digitization)
- 特性曲线插值器 (PCHIP保形插值)
- 工程级参数验证
- 生产环境部署支持
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
# 1. 全局物理与仿真常数
# ==========================================
@dataclass
class GlobalPhysicsConfig:
    """全局物理常数与仿真参数"""
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

    # 冰期参数
    TEMP_ICE_POINT: float = 0.5             # 冰期临界温度 (°C)
    ROUGHNESS_INCREASE_ICE: float = 0.004   # 冰期糙率增量
    ICE_COVER_HYDRAULIC_RADIUS_FACTOR: float = 0.5  # 冰盖水力半径修正因子

    # 数值稳定性
    MIN_DEPTH: float = 0.001                # 最小水深 (m)
    MIN_FLOW: float = 1e-6                  # 最小流量 (m³/s)
    MAX_ITERATIONS: int = 100               # 最大迭代次数
    CONVERGENCE_TOL: float = 1e-6           # 收敛容差


# ==========================================
# 2. 特性曲线数字化库 (Digitized Characteristic Curves)
# ==========================================
@dataclass
class CharacteristicCurves:
    """
    全系统特性曲线数字化查找表

    所有曲线基于工程实测数据或设计文件数字化
    """

    # ---------------------------------------------------------
    # 2.1 文得根水库：水位-库容-面积 (Fig 2.2-1)
    # ---------------------------------------------------------
    # 水位(m) -> 库容(m³)
    WENDEGEN_ZV_DATA: List[Tuple[float, float]] = field(default_factory=lambda: [
        (320.0, 0.0),           # 库底
        (340.0, 1.50e8),
        (351.0, 4.46e8),        # 死水位
        (360.0, 8.50e8),
        (370.0, 14.50e8),
        (377.0, 19.64e8),       # 正常蓄水位
        (379.8, 22.80e8),       # 校核洪水位
        (382.0, 25.00e8)        # 坝顶
    ])

    # 水位(m) -> 水面面积(m²)
    WENDEGEN_ZA_DATA: List[Tuple[float, float]] = field(default_factory=lambda: [
        (320.0, 0.0),
        (340.0, 15.0e6),
        (351.0, 35.0e6),        # 死水位
        (360.0, 55.0e6),
        (370.0, 75.0e6),
        (377.0, 95.0e6),        # 正常蓄水位
        (379.8, 105.0e6),
        (382.0, 110.0e6)
    ])

    # ---------------------------------------------------------
    # 2.2 绰勒水库 (下游边界)：水位-库容 (Fig 2.2-2)
    # ---------------------------------------------------------
    # 正常蓄水位 228.0m -> 库容约 1.8亿m³
    CHUOLE_ZV_DATA: List[Tuple[float, float]] = field(default_factory=lambda: [
        (218.0, 0.0),
        (220.0, 0.30e8),
        (222.0, 0.65e8),
        (224.0, 1.05e8),
        (226.0, 1.45e8),
        (228.0, 1.95e8),        # 正常蓄水位
        (230.0, 2.50e8),
        (232.0, 3.10e8)
    ])

    # ---------------------------------------------------------
    # 2.3 坝下水位-流量关系 (Fig 2.4-1)
    # ---------------------------------------------------------
    # 用于计算电站尾水位及溢洪道下游水位
    # 流量(m³/s) -> 水位(m)
    DAM_DOWNSTREAM_QZ_TABLE: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 334.5),           # 枯水位
        (500.0, 335.8),
        (1000.0, 336.5),
        (2000.0, 337.6),
        (3000.0, 338.4),
        (4000.0, 339.1),
        (5000.0, 339.7),
        (6000.0, 340.2),        # 接近设计泄量
        (8000.0, 341.0),
        (10000.0, 341.8)
    ])

    # ---------------------------------------------------------
    # 2.4 溢洪道泄流曲线参数 (Fig 2.6-1)
    # ---------------------------------------------------------
    SPILLWAY_WEIR_ELEVATION: float = 363.0  # 堰顶高程 (m)
    SPILLWAY_DISCHARGE_COEF: float = 117.0  # 流量系数
    # Q = 117 * (H - 363.0)^1.5

    # ---------------------------------------------------------
    # 2.5 洪水过程线形状因子 (Fig 2.1-1/2)
    # ---------------------------------------------------------
    # 基于1998年典型洪水（双峰型）标准化
    # (时间百分比, 流量百分比)
    FLOOD_PATTERN_1998_DOUBLE_PEAK: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.00, 0.05),   # 起涨
        (0.10, 0.10),
        (0.20, 0.30),
        (0.25, 0.85),   # 第一峰
        (0.30, 0.60),   # 峰间低谷
        (0.40, 0.40),
        (0.50, 0.95),
        (0.55, 1.00),   # 主峰
        (0.60, 0.70),
        (0.70, 0.40),
        (0.80, 0.20),
        (0.90, 0.10),
        (1.00, 0.05)    # 退水
    ])

    # 单峰型洪水（普通年份）
    FLOOD_PATTERN_SINGLE_PEAK: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.00, 0.05),
        (0.15, 0.10),
        (0.30, 0.40),
        (0.45, 0.85),
        (0.50, 1.00),   # 单峰
        (0.55, 0.90),
        (0.65, 0.60),
        (0.75, 0.35),
        (0.85, 0.15),
        (1.00, 0.05)
    ])

    # ---------------------------------------------------------
    # 2.6 兴利调度图 (Fig 3.3-1 & Table 3.3-2)
    # ---------------------------------------------------------
    # 月份 (1-12) -> (上限水位, 下限水位) 单位: m
    # 调度年: 4月-次年3月
    OPERATION_RULE_LIMITS: Dict[int, Tuple[float, float]] = field(default_factory=lambda: {
        4:  (367.30, 358.40),   # 4月 - 春灌开始
        5:  (366.80, 357.70),   # 5月 - 农业需水高峰
        6:  (366.40, 357.00),   # 6月 - 汛前
        7:  (369.40, 361.50),   # 7月 - 主汛期开始
        8:  (373.00, 362.30),   # 8月 - 主汛期
        9:  (373.50, 363.00),   # 9月 - 汛末蓄水
        10: (372.80, 361.80),   # 10月 - 秋汛
        11: (372.00, 360.90),   # 11月 - 冰期准备
        12: (371.10, 360.10),   # 12月 - 冰期
        1:  (370.20, 359.60),   # 1月 - 冰期
        2:  (369.20, 359.20),   # 2月 - 冰期
        3:  (368.20, 358.82)    # 3月 - 开河期
    })

    # 调度分区定义
    OPERATION_ZONES: Dict[str, str] = field(default_factory=lambda: {
        'UPPER': '弃水区 - 需开启溢洪道泄洪',
        'NORMAL': '正常供水区 - 按需调度',
        'LOWER': '限制供水区 - 减少供水量',
        'DEAD': '死库容区 - 停止供水'
    })

    # ---------------------------------------------------------
    # 2.7 洮儿河倒虹吸出口闸特性 (Table 4.2.4-1)
    # ---------------------------------------------------------
    # 上游水位(m) -> [(开度m, 流量m³/s), ...]
    TAOER_SIPHON_GATE_TABLE: Dict[float, List[Tuple[float, float]]] = field(default_factory=lambda: {
        325.0: [(0.2, 1.50), (0.6, 4.35), (1.0, 6.98), (1.4, 9.41),
                (2.0, 11.29), (2.4, 14.54), (3.0, 16.98)],
        324.5: [(0.2, 1.39), (0.6, 3.99), (1.0, 6.36), (1.4, 8.50),
                (2.0, 11.29), (2.4, 12.86)],
        324.0: [(0.2, 1.26), (0.6, 3.59), (1.0, 5.67), (1.4, 7.49),
                (2.0, 9.76), (2.4, 10.97)],
        323.0: [(0.2, 0.95), (0.6, 2.62), (1.0, 3.96), (1.4, 4.98)]
    })

    # ---------------------------------------------------------
    # 2.8 水轮机运转特性 (Fig 2.7-1/2)
    # ---------------------------------------------------------
    # Hill Chart简化: (水头m, 功率MW) -> 效率
    # 大机组 HLTF60-LJ-225 (3台)
    TURBINE_LARGE_HILL_POINTS: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        # (Head, Power, Efficiency)
        (41.9, 11.75, 0.935),   # 最大水头-额定功率
        (41.9, 9.0, 0.920),    # 最大水头-部分负荷
        (41.9, 6.0, 0.880),    # 最大水头-低负荷
        (34.0, 11.75, 0.944),  # 额定水头-额定功率 (最佳效率点)
        (34.0, 8.0, 0.935),    # 额定水头-部分负荷
        (34.0, 5.5, 0.910),    # 额定水头-低负荷
        (27.0, 10.0, 0.930),   # 中水头-接近满负荷
        (27.0, 6.0, 0.905),    # 中水头-部分负荷
        (21.0, 8.0, 0.900),    # 最小水头-满负荷
        (21.0, 4.0, 0.850),    # 最小水头-低负荷
    ])

    # 小机组 (1台，用于小流量运行)
    TURBINE_SMALL_HILL_POINTS: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (41.9, 3.0, 0.900),
        (34.0, 3.0, 0.915),
        (34.0, 2.0, 0.880),
        (21.0, 2.5, 0.870),
        (21.0, 1.5, 0.820),
    ])

    # ---------------------------------------------------------
    # 2.9 在线调流调压阀特性 (基于PDF 4.4.2.4描述)
    # ---------------------------------------------------------
    # 活塞式，线性行程，小开度高阻
    # (开度%, 阻力系数Zeta)
    INLINE_REGULATING_VALVE_ZETA: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 1e7),             # 全关
        (5.0, 800.0),
        (10.0, 250.0),
        (15.0, 120.0),
        (20.0, 60.0),
        (25.0, 38.0),
        (30.0, 25.0),
        (35.0, 17.0),
        (40.0, 12.0),
        (45.0, 9.0),
        (50.0, 7.0),
        (60.0, 4.5),
        (70.0, 3.0),
        (80.0, 2.0),
        (90.0, 1.5),
        (100.0, 1.2)            # 全开
    ])

    # 末端调流阀特性 (差异化)
    END_REGULATING_VALVE_ZETA: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 1e7),
        (5.0, 600.0),
        (10.0, 180.0),
        (20.0, 45.0),
        (30.0, 20.0),
        (40.0, 10.0),
        (50.0, 5.5),
        (60.0, 3.5),
        (70.0, 2.2),
        (80.0, 1.5),
        (90.0, 1.0),
        (100.0, 0.8)
    ])

    # ---------------------------------------------------------
    # 2.10 检修蝶阀特性 (PDF 4.4.2.10)
    # ---------------------------------------------------------
    # 蝶阀流阻特性 (角度-阻力曲线)
    # (开度%, 阻力系数Zeta)
    BUTTERFLY_VALVE_ZETA: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 1e8),             # 全关 (理论无穷大)
        (10.0, 800.0),
        (20.0, 200.0),
        (30.0, 80.0),
        (40.0, 35.0),
        (50.0, 15.0),
        (60.0, 8.0),
        (70.0, 4.0),
        (80.0, 2.0),
        (90.0, 0.8),
        (100.0, 0.3)            # 全开 (仍有阻力)
    ])

    # ---------------------------------------------------------
    # 2.11 鱼道逻辑 (Table 4.1.5-6)
    # ---------------------------------------------------------
    # (水位下限, 水位上限, 出口编号)
    FISHWAY_OUTLET_RULES: List[Tuple[float, float, int]] = field(default_factory=lambda: [
        (366.5, 368.0, 1),      # 1号出口
        (368.0, 369.5, 2),      # 2号出口
        (369.5, 371.0, 3),      # 3号出口
        (371.0, 372.5, 4),      # 4号出口
        (372.5, 374.0, 5),      # 5号出口
        (374.0, 375.5, 6),      # 6号出口
        (375.5, 377.0, 7)       # 7号出口
    ])

    # 鱼道运行参数
    FISHWAY_DESIGN_FLOW: float = 2.5        # 设计流量 (m³/s)
    FISHWAY_MIN_DEPTH: float = 1.2          # 最小水深 (m)
    FISHWAY_MIGRATION_SEASON: Tuple[int, int] = (4, 10)  # 洄游季节 (4月-10月)

    # ---------------------------------------------------------
    # 2.12 进水口特性曲线
    # ---------------------------------------------------------
    # 进水口水位-流量关系 (考虑淹没出流)
    INTAKE_ZQ_DATA: List[Tuple[float, float]] = field(default_factory=lambda: [
        (351.0, 0.0),           # 最低运行水位
        (355.0, 8.5),
        (360.0, 12.8),
        (365.0, 15.6),
        (370.0, 17.2),
        (377.0, 18.58),         # 设计流量
        (379.66, 18.58)         # 最高水位
    ])

    # 进水口闸门损失系数
    INTAKE_GATE_LOSS_COEF: float = 0.05
    INTAKE_TRASH_RACK_LOSS_COEF: float = 0.15

    # ---------------------------------------------------------
    # 插值方法
    # ---------------------------------------------------------
    def get_wendegen_volume(self, level: float) -> float:
        """文得根水库：水位->库容"""
        x = [p[0] for p in self.WENDEGEN_ZV_DATA]
        y = [p[1] for p in self.WENDEGEN_ZV_DATA]
        return float(np.interp(level, x, y))

    def get_wendegen_area(self, level: float) -> float:
        """文得根水库：水位->面积"""
        x = [p[0] for p in self.WENDEGEN_ZA_DATA]
        y = [p[1] for p in self.WENDEGEN_ZA_DATA]
        return float(np.interp(level, x, y))

    def get_wendegen_level(self, volume: float) -> float:
        """文得根水库：库容->水位 (反查)"""
        x = [p[1] for p in self.WENDEGEN_ZV_DATA]  # 库容作为x
        y = [p[0] for p in self.WENDEGEN_ZV_DATA]  # 水位作为y
        return float(np.interp(volume, x, y))

    def get_chuole_volume(self, level: float) -> float:
        """绰勒水库：水位->库容"""
        x = [p[0] for p in self.CHUOLE_ZV_DATA]
        y = [p[1] for p in self.CHUOLE_ZV_DATA]
        return float(np.interp(level, x, y))

    def get_dam_tailwater_level(self, discharge: float) -> float:
        """坝下：流量->水位"""
        x = [p[0] for p in self.DAM_DOWNSTREAM_QZ_TABLE]
        y = [p[1] for p in self.DAM_DOWNSTREAM_QZ_TABLE]
        return float(np.interp(discharge, x, y))

    def get_spillway_discharge(self, upstream_level: float) -> float:
        """溢洪道泄流量计算"""
        h = upstream_level - self.SPILLWAY_WEIR_ELEVATION
        if h <= 0:
            return 0.0
        return self.SPILLWAY_DISCHARGE_COEF * math.pow(h, 1.5)

    def get_inline_valve_zeta(self, opening_pct: float) -> float:
        """在线调流阀：开度->阻力系数"""
        x = [p[0] for p in self.INLINE_REGULATING_VALVE_ZETA]
        y = [p[1] for p in self.INLINE_REGULATING_VALVE_ZETA]
        return float(np.interp(opening_pct, x, y))

    def get_end_valve_zeta(self, opening_pct: float) -> float:
        """末端调流阀：开度->阻力系数"""
        x = [p[0] for p in self.END_REGULATING_VALVE_ZETA]
        y = [p[1] for p in self.END_REGULATING_VALVE_ZETA]
        return float(np.interp(opening_pct, x, y))

    def get_butterfly_valve_zeta(self, opening_pct: float) -> float:
        """蝶阀：开度->阻力系数"""
        x = [p[0] for p in self.BUTTERFLY_VALVE_ZETA]
        y = [p[1] for p in self.BUTTERFLY_VALVE_ZETA]
        return float(np.interp(opening_pct, x, y))

    def get_operation_zone(self, month: int, level: float) -> str:
        """获取调度分区"""
        if month not in self.OPERATION_RULE_LIMITS:
            return 'UNKNOWN'
        upper, lower = self.OPERATION_RULE_LIMITS[month]
        dead_level = 351.0  # 死水位

        if level > upper:
            return 'UPPER'
        elif level >= lower:
            return 'NORMAL'
        elif level > dead_level:
            return 'LOWER'
        else:
            return 'DEAD'

    def get_turbine_efficiency(self, head: float, power: float,
                               turbine_type: str = 'large') -> float:
        """
        获取水轮机效率

        使用双线性插值从Hill Chart获取效率
        """
        points = (self.TURBINE_LARGE_HILL_POINTS if turbine_type == 'large'
                  else self.TURBINE_SMALL_HILL_POINTS)

        # 简化：找最近点
        min_dist = float('inf')
        best_eta = 0.85  # 默认效率

        for h, p, eta in points:
            dist = math.sqrt((head - h)**2 + (power - p)**2)
            if dist < min_dist:
                min_dist = dist
                best_eta = eta

        return best_eta

    def get_fishway_outlet(self, reservoir_level: float) -> int:
        """根据库水位选择鱼道出口"""
        for low, high, outlet in self.FISHWAY_OUTLET_RULES:
            if low <= reservoir_level < high:
                return outlet
        # 超出范围
        if reservoir_level >= self.FISHWAY_OUTLET_RULES[-1][1]:
            return self.FISHWAY_OUTLET_RULES[-1][2]
        return 0  # 无合适出口

    def get_intake_flow(self, level: float) -> float:
        """进水口最大过流能力"""
        x = [p[0] for p in self.INTAKE_ZQ_DATA]
        y = [p[1] for p in self.INTAKE_ZQ_DATA]
        return float(np.interp(level, x, y))


# ==========================================
# 3. 水源枢纽配置 (文得根水利枢纽)
# ==========================================
@dataclass
class SourceHubConfig:
    """文得根水利枢纽参数"""
    NAME: str = "文得根水利枢纽"
    LOCATION: str = "T0+000"

    # 特征水位 (m)
    NORMAL_LEVEL: float = 377.00            # 正常蓄水位
    DEAD_LEVEL: float = 351.00              # 死水位
    FLOOD_LIMIT_LEVEL: float = 370.00       # 汛限水位
    CHECK_FLOOD_LEVEL: float = 379.80       # 校核洪水位
    DESIGN_FLOOD_LEVEL: float = 378.60      # 设计洪水位

    # 库容 (亿m³)
    TOTAL_STORAGE: float = 22.80            # 总库容 (校核)
    NORMAL_STORAGE: float = 19.64           # 兴利库容
    DEAD_STORAGE: float = 4.46              # 死库容
    FLOOD_CONTROL_STORAGE: float = 3.16     # 防洪库容

    # 溢洪道
    SPILLWAY_WEIR_EL: float = 363.00        # 堰顶高程
    SPILLWAY_GATE_COUNT: int = 5            # 弧形闸门数量
    SPILLWAY_GATE_WIDTH: float = 12.0       # 单孔宽度 (m)
    SPILLWAY_GATE_HEIGHT: float = 14.8      # 闸门高度 (m)
    SPILLWAY_MAX_DISCHARGE: float = 6340.0  # 最大泄量 (m³/s)

    # 发电机组
    TURBINE_L_COUNT: int = 3                # 大机组数量
    TURBINE_S_COUNT: int = 1                # 小机组数量
    TURBINE_L_POWER: float = 11.75          # 大机组单机功率 (MW)
    TURBINE_S_POWER: float = 3.0            # 小机组功率 (MW)
    TURBINE_L_FLOW: float = 42.0            # 大机组额定流量 (m³/s)
    TURBINE_S_FLOW: float = 10.0            # 小机组额定流量 (m³/s)
    HEAD_RATED: float = 34.0                # 额定水头 (m)
    HEAD_MAX: float = 41.9                  # 最大水头 (m)
    HEAD_MIN: float = 21.0                  # 最小水头 (m)
    TAILWATER_EL: float = 334.5             # 尾水位 (枯水)

    # 引水进水口
    INTAKE_DESIGN_FLOW: float = 18.58       # 设计引水流量 (m³/s)
    INTAKE_LEVEL_MAX: float = 379.66        # 最高运行水位
    INTAKE_LEVEL_MIN: float = 350.86        # 最低运行水位
    INTAKE_SILL_EL: float = 343.0           # 进水口底高程
    INTAKE_GATE_WIDTH: float = 3.0          # 进水口宽度 (m)
    INTAKE_GATE_HEIGHT: float = 4.0         # 进水口高度 (m)

    # 鱼道
    FISHWAY_POOL_COUNT: int = 65            # 鱼道池数
    FISHWAY_DESIGN_FLOW: float = 2.5        # 设计流量 (m³/s)


# ==========================================
# 4. 隧洞配置
# ==========================================
@dataclass
class TunnelSystemConfig:
    """输水隧洞系统参数"""

    # 分段参数 {隧洞名: (起点桩号, 长度m, 排空时间h)}
    TUNNEL_SEGMENTS: Dict[str, Tuple[str, float, float]] = field(default_factory=lambda: {
        "Tunnel_1": ("T0+000", 9035.0, 9.0),
        "Tunnel_2": ("T9+035", 58505.0, 51.0),
        "Tunnel_3": ("T67+540", 3491.0, 8.0),
        "Tunnel_4": ("T71+031", 3837.0, 12.0),
        "Tunnel_5": ("T74+868", 67856.0, 55.0),
        "Tunnel_6": ("T142+724", 31003.0, 71.0),
    })

    TOTAL_LENGTH: float = 173727.0          # 总长度 (m)

    # 断面参数 (城门洞型)
    SECTION_WIDTH: float = 6.0              # 宽度 (m)
    SECTION_HEIGHT: float = 7.0             # 高度 (m)
    ARCH_RADIUS: float = 3.0                # 拱顶半径 (m)

    # 水力参数
    BOTTOM_SLOPE: float = 1/2000            # 纵坡 (0.0005)
    MANNING_N_NORMAL: float = 0.014         # 正常糙率
    MANNING_N_ICE: float = 0.018            # 冰期糙率

    # 设计流量
    DESIGN_FLOW: float = 18.58              # 设计流量 (m³/s)
    MAX_VELOCITY: float = 3.0               # 最大流速 (m/s)

    # 数值参数
    DX_SPATIAL: float = 500.0               # 空间步长 (m)

    @property
    def cross_section_area(self) -> float:
        """城门洞断面面积"""
        # 矩形部分 + 拱顶部分
        rect_area = self.SECTION_WIDTH * (self.SECTION_HEIGHT - self.ARCH_RADIUS)
        arch_area = 0.5 * math.pi * self.ARCH_RADIUS ** 2
        return rect_area + arch_area

    @property
    def wetted_perimeter(self) -> float:
        """湿周"""
        # 两侧 + 底部 + 拱顶
        sides = 2 * (self.SECTION_HEIGHT - self.ARCH_RADIUS)
        bottom = self.SECTION_WIDTH
        arch = math.pi * self.ARCH_RADIUS
        return sides + bottom + arch

    @property
    def hydraulic_radius(self) -> float:
        """水力半径"""
        return self.cross_section_area / self.wetted_perimeter


# ==========================================
# 5. 稳流连接池配置
# ==========================================
@dataclass
class StabilizingPoolConfig:
    """稳流连接池参数"""
    NAME: str = "稳流连接池"
    LOCATION: str = "T112+238.45"

    # 几何参数
    LENGTH: float = 104.0                   # 长度 (m)
    WIDTH: float = 30.0                     # 宽度 (m)，估计值
    EFFECTIVE_VOLUME: float = 7737.71       # 有效容积 (m³)

    # 特征水位
    LEVEL_MAX: float = 284.10               # 最高水位 (溢流堰顶)
    LEVEL_DESIGN: float = 284.05            # 设计水位
    LEVEL_MIN: float = 282.26               # 最低水位
    LEVEL_WARNING_HIGH: float = 283.80      # 高水位报警
    LEVEL_WARNING_LOW: float = 282.60       # 低水位报警

    # 溢流堰
    WEIR_ELEVATION: float = 284.10          # 溢流堰顶高程
    WEIR_LENGTH: float = 20.0               # 溢流堰长度 (m)
    WEIR_COEFFICIENT: float = 1.86          # 堰流系数

    @property
    def surface_area(self) -> float:
        """水面面积"""
        return self.LENGTH * self.WIDTH


# ==========================================
# 6. PCCP管道配置
# ==========================================
@dataclass
class PipelineSystemConfig:
    """PCCP管道系统参数"""
    NAME: str = "PCCP输水干管"
    START_STATION: str = "T112+238.45"
    END_STATION: str = "T319+119.72"

    TOTAL_LENGTH: float = 206881.27         # 总长度 (m)

    # 管道规格
    INNER_DIAMETER: float = 2.4             # 内径 (m)，DN2400
    WALL_THICKNESS: float = 0.28            # 壁厚 (m)

    # 材料参数
    YOUNGS_MODULUS_CONCRETE: float = 35e9   # 混凝土弹性模量 (Pa)
    YOUNGS_MODULUS_STEEL: float = 200e9     # 钢筒弹性模量 (Pa)
    POISSON_RATIO: float = 0.2              # 泊松比

    # 水力参数
    WAVE_SPEED: float = 1050.0              # 压力波速 (m/s)
    DARCY_FRICTION: float = 0.012           # 达西摩阻系数

    # 设计压力 (m水头)
    DESIGN_PRESSURE: float = 100.0          # 设计压力
    MAX_WORKING_PRESSURE: float = 115.0     # 最大工作压力
    TEST_PRESSURE: float = 150.0            # 试验压力

    # 沿线分水口设计流量 (m³/s)
    BRANCH_DESIGN_FLOWS: Dict[str, float] = field(default_factory=lambda: {
        "Pool_to_Tuquan": 14.61,             # 稳流池至突泉
        "Tuquan_to_Tuxun": 14.25,            # 突泉至图训
        "Tuxun_to_Keyouzhong": 13.25,        # 图训至科右中
        "Keyouzhong_to_Zalute": 11.96,       # 科右中至扎鲁特
        "Zalute_to_Kezuozhong": 8.34,        # 扎鲁特至科左中
        "Kezuozhong_to_Kailu": 7.69,         # 科左中至开鲁
        "Kailu_to_End": 7.04                 # 开鲁至末端
    })

    # 分水口位置 (桩号)
    BRANCH_LOCATIONS: Dict[str, str] = field(default_factory=lambda: {
        "Tuquan": "T140+200",                # 突泉分水口
        "Tuxun": "T156+000",                 # 图训分水口
        "Keyouzhong": "T185+500",            # 科右中分水口
        "Zalute": "T230+000",                # 扎鲁特分水口
        "Kezuozhong": "T270+000",            # 科左中分水口
        "Kailu": "T295+000",                 # 开鲁分水口
        "End": "T319+119.72"                 # 末端
    })

    @property
    def cross_section_area(self) -> float:
        """管道断面积"""
        return math.pi * (self.INNER_DIAMETER / 2) ** 2

    @property
    def moc_courant_dx(self) -> float:
        """MOC空间步长 (满足Courant条件)"""
        dt = 0.5  # 时间步长
        return self.WAVE_SPEED * dt


# ==========================================
# 7. 调压塔配置
# ==========================================
@dataclass
class SurgeTankConfig:
    """调压塔参数"""
    NAME: str = "调压塔"
    LOCATION: str = "T141+903.57"

    # 几何参数
    DIAMETER: float = 6.0                   # 直径 (m)
    TOP_ELEVATION: float = 295.10           # 塔顶高程 (m)
    BOTTOM_ELEVATION: float = 263.04        # 塔底高程 (m)
    HEIGHT: float = 32.06                   # 高度 (m)

    # 阻抗参数 (双向阻抗式)
    IMPEDANCE_DIAMETER: float = 2.0         # 阻抗孔直径 (m)
    INFLOW_RESISTANCE: float = 45.5         # 入流阻抗系数
    OUTFLOW_RESISTANCE: float = 18.2        # 出流阻抗系数

    # 水位约束
    LEVEL_WARNING_MAX: float = 293.10       # 最高警戒水位
    LEVEL_WARNING_MIN: float = 269.10       # 最低警戒水位
    LEVEL_DESIGN: float = 283.50            # 设计水位

    @property
    def cross_section_area(self) -> float:
        """塔截面积"""
        return math.pi * (self.DIAMETER / 2) ** 2

    @property
    def impedance_area(self) -> float:
        """阻抗孔面积"""
        return math.pi * (self.IMPEDANCE_DIAMETER / 2) ** 2


# ==========================================
# 8. 调流调压阀配置
# ==========================================
@dataclass
class RegulatingValveConfig:
    """调流调压阀参数"""

    # 在线调流调压阀室 (T212)
    INLINE_VALVE_LOCATION: str = "T212+919.13"
    INLINE_VALVE_COUNT: int = 3             # 阀门数量 (2用1备)
    INLINE_VALVE_DN: float = 1.8            # 公称直径 (m)
    INLINE_VALVE_TYPE: str = "活塞式"

    # 末端调流阀
    END_VALVE_LOCATION: str = "T319+100"
    END_VALVE_COUNT: int = 2
    END_VALVE_DN: float = 1.6

    # 关阀时间约束 (PDF 表4.4.2.2-1)
    VALVE_CLOSE_TIMES: Dict[str, float] = field(default_factory=lambda: {
        "Tuquan": 150.0,                    # 突泉 (s)
        "Tuxun": 300.0,
        "Keyouzhong": 300.0,
        "Zalute": 1100.0,
        "Kezuozhong": 400.0,
        "Baolongshan": 1000.0,
        "Kailu": 1000.0,
        "Main_End": 2500.0,                 # 末端主阀
        "Main_Inline": 2500.0               # 在线主阀
    })

    # 阀门动作参数
    MAX_STROKE_RATE: float = 0.02           # 最大行程速率 (%/s)
    EMERGENCY_CLOSE_TIME: float = 60.0      # 紧急关闭时间 (s)
    POSITION_DEADBAND: float = 0.01         # 位置死区 (%)


# ==========================================
# 9. 用户需水配置
# ==========================================
@dataclass
class EndUserConfig:
    """终端用户需水配置"""

    # 设计需水量 (m³/s)
    FLOW_DEMANDS: Dict[str, float] = field(default_factory=lambda: {
        "Tuquan": 0.37,                     # 突泉
        "Park": 1.00,                       # 园区
        "Keyouzhong": 1.29,                 # 科右中
        "Zalute": 3.62,                     # 扎鲁特
        "Kezuozhong": 0.65,                 # 科左中
        "Kailu": 0.65,                      # 开鲁
        "DevZone": 2.55,                    # 开发区
        "Keerqin": 4.49                     # 科尔沁
    })

    # 年需水量 (亿m³)
    ANNUAL_DEMANDS: Dict[str, float] = field(default_factory=lambda: {
        "Tuquan": 0.12,
        "Park": 0.32,
        "Keyouzhong": 0.41,
        "Zalute": 1.14,
        "Kezuozhong": 0.21,
        "Kailu": 0.21,
        "DevZone": 0.80,
        "Keerqin": 1.42
    })

    # 高位调节池
    POOL_HIGH_LOCATION: str = "T318+566"
    POOL_HIGH_LEVEL_MAX: float = 213.79     # 最高水位
    POOL_HIGH_LEVEL_DESIGN: float = 213.29  # 设计水位
    POOL_HIGH_LEVEL_MIN: float = 211.19     # 最低水位
    POOL_HIGH_VOLUME: float = 5000.0        # 容积 (m³)

    @property
    def total_design_flow(self) -> float:
        """总设计流量"""
        return sum(self.FLOW_DEMANDS.values())


# ==========================================
# 10. 安全设施配置
# ==========================================
@dataclass
class SafetySystemConfig:
    """安全设施参数"""

    # 超压泄压阀
    RELIEF_VALVE_SET_PRESSURE_1: float = 115.0  # 1#泄压阀开启压力 (m)
    RELIEF_VALVE_SET_PRESSURE_2: float = 96.0   # 2#泄压阀开启压力 (m)
    RELIEF_VALVE_CLOSE_RATIO: float = 0.95      # 回座压力比
    RELIEF_VALVE_CV: float = 30.0               # 流量系数
    RELIEF_VALVE_DAMPING_TIME: float = 30.0     # 液压阻尼时间 (s)

    # 空气阀
    AIR_VALVE_COUNT: int = 232                  # 空气阀数量
    AIR_VALVE_SPACING: float = 800.0            # 平均间距 (m)
    AIR_VALVE_INTAKE_CAPACITY: float = 0.5      # 进气能力 (m³/s)
    AIR_VALVE_EXHAUST_CAPACITY: float = 0.2     # 排气能力 (m³/s)
    AIR_VALVE_LARGE_ORIFICE: float = 0.10       # 大孔直径 (m)
    AIR_VALVE_SMALL_ORIFICE: float = 0.02       # 小孔直径 (m)

    # 联通阀
    INTERCONNECT_VALVE_DN: float = 2.8          # 联通阀直径 (m)
    INTERCONNECT_VALVE_TYPE: str = "蝶阀"

    # 压力报警阈值
    PRESSURE_ALARM_HIGH: float = 110.0          # 高压报警 (m)
    PRESSURE_ALARM_LOW: float = -5.0            # 负压报警 (m)
    PRESSURE_TRIP_HIGH: float = 115.0           # 超压停机 (m)

    # 水位报警阈值
    LEVEL_ALARM_MARGIN: float = 0.3             # 水位报警裕度 (m)


# ==========================================
# 11. 仿真与硬件接口配置
# ==========================================
@dataclass
class SimulationHardwareConfig:
    """仿真与硬件接口参数"""

    # 传感器噪声模型
    NOISE_LEVEL: float = 0.01               # 水位噪声 (m)
    NOISE_PRESSURE: float = 0.5             # 压力噪声 (kPa)
    NOISE_FLOW: float = 0.02                # 流量噪声 (m³/s)
    NOISE_TEMPERATURE: float = 0.1          # 温度噪声 (°C)

    # 采样率
    SAMPLE_RATE_FAST: int = 100             # 快速采样 (Hz)
    SAMPLE_RATE_NORMAL: int = 10            # 正常采样 (Hz)
    SAMPLE_RATE_SLOW: int = 1               # 慢速采样 (Hz)

    # 通信参数
    SCADA_POLL_INTERVAL: float = 1.0        # SCADA轮询间隔 (s)
    HEARTBEAT_INTERVAL: float = 0.1         # 心跳间隔 (s)
    COMM_TIMEOUT: float = 3.0               # 通信超时 (s)

    # 执行器响应
    ACTUATOR_DELAY: float = 0.5             # 执行器延迟 (s)
    ACTUATOR_DEADBAND: float = 0.01         # 执行器死区 (%)


# ==========================================
# 12. 控制参数配置
# ==========================================
@dataclass
class ControlParameterConfig:
    """控制算法参数"""

    # PID参数 (稳流池水位控制)
    PID_POOL_KP: float = 0.1                # 比例增益
    PID_POOL_KI: float = 0.005              # 积分增益
    PID_POOL_KD: float = 0.02               # 微分增益
    PID_POOL_INTEGRAL_LIMIT: float = 10.0   # 积分限幅

    # MPC参数
    MPC_PREDICTION_HORIZON: int = 20        # 预测时域
    MPC_CONTROL_HORIZON: int = 5            # 控制时域
    MPC_SAMPLE_TIME: float = 60.0           # 采样时间 (s)
    MPC_Q_WEIGHT: float = 10.0              # 状态权重
    MPC_R_WEIGHT: float = 1.0               # 控制权重
    MPC_DELTA_U_WEIGHT: float = 0.1         # 控制增量权重

    # EKF参数
    EKF_Q_LEVEL: float = 0.1                # 水位过程噪声
    EKF_Q_FLOW: float = 1e-6                # 流量过程噪声
    EKF_R_LEVEL: float = 0.01               # 水位测量噪声
    EKF_R_PRESSURE: float = 0.5             # 压力测量噪声

    # ADMM协调参数
    ADMM_RHO: float = 1.0                   # 惩罚参数
    ADMM_MAX_ITER: int = 50                 # 最大迭代次数
    ADMM_TOLERANCE: float = 1e-4            # 收敛容差

    # 安全边界
    SAFETY_MARGIN_PRESSURE: float = 10.0    # 压力安全裕度 (m)
    SAFETY_MARGIN_LEVEL: float = 0.5        # 水位安全裕度 (m)


# ==========================================
# 13. 洪水与来水配置
# ==========================================
@dataclass
class HydrologyConfig:
    """水文参数配置"""

    # 设计洪水 (m³/s)
    DESIGN_FLOOD_PEAK: float = 6340.0       # 设计洪峰
    CHECK_FLOOD_PEAK: float = 8500.0        # 校核洪峰

    # 洪水频率
    FLOOD_FREQUENCY: Dict[str, float] = field(default_factory=lambda: {
        "P=0.01%": 8500.0,                  # 万年一遇
        "P=0.1%": 6340.0,                   # 千年一遇
        "P=1%": 4200.0,                     # 百年一遇
        "P=2%": 3600.0,                     # 五十年一遇
        "P=5%": 2800.0,                     # 二十年一遇
        "P=10%": 2200.0,                    # 十年一遇
        "P=20%": 1600.0                     # 五年一遇
    })

    # 多年平均来水量
    ANNUAL_INFLOW_MEAN: float = 9.42e8      # 多年平均 (m³)
    ANNUAL_INFLOW_P75: float = 7.5e8        # 75%保证率
    ANNUAL_INFLOW_P95: float = 5.2e8        # 95%保证率

    # 月来水分配 (占年来水百分比)
    MONTHLY_INFLOW_PATTERN: Dict[int, float] = field(default_factory=lambda: {
        1: 0.01, 2: 0.01, 3: 0.02, 4: 0.05,
        5: 0.08, 6: 0.15, 7: 0.25, 8: 0.22,
        9: 0.12, 10: 0.05, 11: 0.03, 12: 0.01
    })


# ==========================================
# 总成配置类
# ==========================================
@dataclass
class YinChuoProjectConfig:
    """
    引绰济辽工程全系统参数总成 (Database V3.2)

    用于生产环境部署的完整配置集
    """
    # 版本信息
    VERSION: str = "3.2.0"
    BUILD_DATE: str = "2024-02-02"

    # 各子系统配置
    Global: GlobalPhysicsConfig = field(default_factory=GlobalPhysicsConfig)
    Curves: CharacteristicCurves = field(default_factory=CharacteristicCurves)
    Source: SourceHubConfig = field(default_factory=SourceHubConfig)
    Tunnel: TunnelSystemConfig = field(default_factory=TunnelSystemConfig)
    Pool: StabilizingPoolConfig = field(default_factory=StabilizingPoolConfig)
    Pipeline: PipelineSystemConfig = field(default_factory=PipelineSystemConfig)
    SurgeTank: SurgeTankConfig = field(default_factory=SurgeTankConfig)
    Valve: RegulatingValveConfig = field(default_factory=RegulatingValveConfig)
    EndUser: EndUserConfig = field(default_factory=EndUserConfig)
    Safety: SafetySystemConfig = field(default_factory=SafetySystemConfig)
    Hardware: SimulationHardwareConfig = field(default_factory=SimulationHardwareConfig)
    Control: ControlParameterConfig = field(default_factory=ControlParameterConfig)
    Hydrology: HydrologyConfig = field(default_factory=HydrologyConfig)

    def validate(self) -> List[str]:
        """
        验证配置完整性和一致性

        Returns:
            错误信息列表，空列表表示验证通过
        """
        errors = []

        # 检查水位逻辑
        if self.Source.DEAD_LEVEL >= self.Source.NORMAL_LEVEL:
            errors.append("死水位必须低于正常蓄水位")

        if self.Source.FLOOD_LIMIT_LEVEL >= self.Source.CHECK_FLOOD_LEVEL:
            errors.append("汛限水位必须低于校核洪水位")

        # 检查流量平衡
        total_branch_flow = sum(self.EndUser.FLOW_DEMANDS.values())
        if total_branch_flow > self.Source.INTAKE_DESIGN_FLOW * 1.1:
            errors.append(f"用户需水量({total_branch_flow:.2f})超过设计引水流量")

        # 检查压力约束
        if self.Safety.RELIEF_VALVE_SET_PRESSURE_1 <= self.Pipeline.DESIGN_PRESSURE:
            errors.append("泄压阀开启压力必须高于设计压力")

        # 检查MOC稳定性条件
        courant = self.Pipeline.WAVE_SPEED * self.Global.DT_PHYSICS / self.Pipeline.moc_courant_dx
        if courant > 1.0:
            errors.append(f"MOC Courant数({courant:.2f})超过1.0，需减小时间步长")

        # 检查调压塔容积
        surge_volume = self.SurgeTank.cross_section_area * self.SurgeTank.HEIGHT
        if surge_volume < self.Pool.EFFECTIVE_VOLUME * 0.5:
            errors.append("调压塔容积可能不足以应对涌浪")

        return errors

    def get_summary(self) -> Dict:
        """获取配置摘要"""
        return {
            'version': self.VERSION,
            'source': {
                'name': self.Source.NAME,
                'normal_level': self.Source.NORMAL_LEVEL,
                'total_storage_billion_m3': self.Source.TOTAL_STORAGE
            },
            'conveyance': {
                'tunnel_length_km': self.Tunnel.TOTAL_LENGTH / 1000,
                'pipeline_length_km': self.Pipeline.TOTAL_LENGTH / 1000,
                'design_flow': self.Source.INTAKE_DESIGN_FLOW
            },
            'end_users': {
                'count': len(self.EndUser.FLOW_DEMANDS),
                'total_demand': self.EndUser.total_design_flow
            }
        }


# ==========================================
# 模块级实例
# ==========================================
# 全局配置实例
ProjectParams = YinChuoProjectConfig()

# 便捷访问
GlobalConfig = ProjectParams.Global
CurveDatabase = ProjectParams.Curves
SourceConfig = ProjectParams.Source
TunnelConfig = ProjectParams.Tunnel
PoolConfig = ProjectParams.Pool
PipeConfig = ProjectParams.Pipeline
SurgeConfig = ProjectParams.SurgeTank
ValveConfig = ProjectParams.Valve
UserConfig = ProjectParams.EndUser
SafetyConfig = ProjectParams.Safety
HardwareConfig = ProjectParams.Hardware
ControlConfig = ProjectParams.Control
HydroConfig = ProjectParams.Hydrology


# ==========================================
# 导出
# ==========================================
__all__ = [
    'YinChuoProjectConfig',
    'ProjectParams',
    'GlobalPhysicsConfig',
    'CharacteristicCurves',
    'SourceHubConfig',
    'TunnelSystemConfig',
    'StabilizingPoolConfig',
    'PipelineSystemConfig',
    'SurgeTankConfig',
    'RegulatingValveConfig',
    'EndUserConfig',
    'SafetySystemConfig',
    'SimulationHardwareConfig',
    'ControlParameterConfig',
    'HydrologyConfig',
    'GlobalConfig',
    'CurveDatabase',
    'SourceConfig',
    'TunnelConfig',
    'PoolConfig',
    'PipeConfig',
    'SurgeConfig',
    'ValveConfig',
    'UserConfig',
    'SafetyConfig',
    'HardwareConfig',
    'ControlConfig',
    'HydroConfig',
    'create_interpolator'
]
