"""
全局配置参数
============

包含引绰济辽工程所有物理参数、控制参数和仿真配置。
基于工程实际数据进行精确标定。
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum, auto


class SeasonMode(Enum):
    """季节模式"""
    NORMAL = auto()      # 正常运行
    ICE_PERIOD = auto()  # 冰期运行
    FLOOD = auto()       # 汛期运行
    DROUGHT = auto()     # 枯水期


class ScenarioType(Enum):
    """场景类型"""
    NORMAL = auto()              # 正常运行
    DEMAND_SURGE = auto()        # 需水突增
    DEMAND_DROP = auto()         # 需水骤降
    PIPE_BURST = auto()          # 管道爆裂
    VALVE_FAULT = auto()         # 阀门故障
    COMMUNICATION_LOSS = auto()  # 通信中断
    ICE_BLOCKAGE = auto()        # 冰塞
    POWER_OUTAGE = auto()        # 断电
    SENSOR_FAULT = auto()        # 传感器故障
    PUMP_FAILURE = auto()        # 水泵故障


@dataclass
class PhysicsConstants:
    """物理常数"""
    G: float = 9.80665              # 重力加速度 (m/s²)
    RHO: float = 998.2              # 20°C水密度 (kg/m³)
    PATM: float = 10.33             # 标准大气压水头 (m)
    KINEMATIC_VISCOSITY: float = 1.004e-6  # 运动粘度 (m²/s)
    BULK_MODULUS: float = 2.2e9     # 水体积弹性模量 (Pa)
    VAPOR_PRESSURE: float = 0.24    # 20°C水蒸汽压水头 (m)


@dataclass
class ReservoirConfig:
    """水库配置 - 文得根水利枢纽"""
    name: str = "Wendegen"
    normal_level: float = 652.0      # 正常蓄水位 (m)
    dead_level: float = 620.0        # 死水位 (m)
    flood_limit_level: float = 648.0 # 汛限水位 (m)
    design_level: float = 655.0      # 设计洪水位 (m)

    # 溢洪道 (5孔弧形闸门)
    spillway_gates: int = 5
    gate_width: float = 12.0         # 单孔宽度 (m)
    gate_max_opening: float = 10.0   # 最大开度 (m)
    gate_sill_elevation: float = 635.0  # 堰顶高程 (m)

    # 进水口
    intake_sill_elevation: float = 610.0  # 进水口底高程 (m)
    intake_gate_width: float = 6.0        # 进水口宽度 (m)

    # 库容特性 (分段线性化)
    elevation_volume_curve: Dict[float, float] = field(default_factory=lambda: {
        610.0: 0.0,
        620.0: 50e6,
        635.0: 200e6,
        648.0: 450e6,
        652.0: 550e6,
        655.0: 650e6
    })


@dataclass
class TunnelConfig:
    """隧洞配置 - 1#~6#输水隧洞"""
    name: str = "MainTunnel"
    total_length: float = 140000.0   # 总长度 (m)
    sections: int = 6                # 分段数

    # 断面参数 (城门洞型)
    width: float = 6.0               # 宽度 (m)
    height: float = 7.0              # 高度 (m)

    # 水力参数
    bottom_slope: float = 0.0005     # 底坡
    manning_n_normal: float = 0.014  # 正常糙率
    manning_n_ice: float = 0.018     # 冰期糙率

    # 数值参数
    dx: float = 500.0                # 空间步长 (m)

    @property
    def num_nodes(self) -> int:
        return int(self.total_length / self.dx) + 1

    @property
    def cross_section_area(self) -> float:
        """城门洞断面近似面积"""
        return self.width * self.height * 0.9  # 扣除拱顶


@dataclass
class PoolConfig:
    """稳流连接池配置"""
    name: str = "StabilizingPool"

    # 几何参数
    length: float = 50.0             # 长度 (m)
    width: float = 30.0              # 宽度 (m)
    depth: float = 8.0               # 深度 (m)
    bottom_elevation: float = 0.0    # 底板高程 (相对)

    # 水位约束
    design_level: float = 5.0        # 设计运行水位 (m)
    max_level: float = 7.5           # 最高水位 (溢流前)
    min_level: float = 1.5           # 最低水位 (防吸气)
    warning_high: float = 6.5        # 高水位报警
    warning_low: float = 2.0         # 低水位报警

    # 有效调节容积
    effective_volume: float = 7740.0  # m³

    @property
    def surface_area(self) -> float:
        return self.length * self.width


@dataclass
class SurgeTankConfig:
    """调压塔配置"""
    name: str = "SurgeTank"

    # 几何参数
    diameter: float = 20.0           # 直径 (m)
    height: float = 60.0             # 高度 (m)
    base_elevation: float = 10.0     # 塔底高程 (相对稳流池)

    # 阻抗孔参数 (双向阻抗式)
    impedance_diameter: float = 2.5  # 阻抗孔直径 (m)
    r_inflow: float = 45.5           # 入流阻抗系数
    r_outflow: float = 18.2          # 出流阻抗系数

    # 水位约束
    max_surge_level: float = 55.0    # 最高涌浪水位
    min_surge_level: float = 20.0    # 最低涌浪水位

    @property
    def area(self) -> float:
        return math.pi * (self.diameter / 2) ** 2

    @property
    def impedance_area(self) -> float:
        return math.pi * (self.impedance_diameter / 2) ** 2


@dataclass
class PipelineConfig:
    """PCCP管线配置"""
    name: str = "MainPipeline"

    # 几何参数
    total_length: float = 180000.0   # 总长度 (m)
    diameter: float = 2.4            # 内径 (m)
    wall_thickness: float = 0.25     # 壁厚 (m)

    # 材料参数
    youngs_modulus: float = 35e9     # PCCP弹性模量 (Pa)
    poisson_ratio: float = 0.2       # 泊松比

    # 水力参数
    wave_speed: float = 1050.0       # 压力波速 (m/s)
    darcy_f: float = 0.012           # 达西摩阻系数

    # 设计压力
    design_pressure: float = 100.0   # 设计水头 (m)
    max_pressure: float = 120.0      # 最大允许压力 (m)
    test_pressure: float = 150.0     # 试验压力 (m)

    # 分水口位置 (桩号)
    branch_positions: List[float] = field(default_factory=lambda: [
        30000.0,   # 突泉
        55000.0,   # 园区
        80000.0,   # 科右中
        110000.0,  # 扎鲁特
        140000.0,  # 科左中
        165000.0   # 开鲁
    ])

    @property
    def area(self) -> float:
        return math.pi * (self.diameter / 2) ** 2

    @property
    def num_nodes(self) -> int:
        """MOC网格节点数"""
        dx = self.wave_speed * 0.5  # dt=0.5s
        return int(self.total_length / dx) + 1


@dataclass
class ValveConfig:
    """阀门配置"""
    # T212调流调压阀室
    t212_position: float = 90000.0   # 位置 (m)
    t212_num_valves: int = 3         # 阀门数量 (2用1备)
    t212_diameter: float = 1.6       # DN1600
    t212_k_full_open: float = 0.15   # 全开阻力系数
    t212_cv_max: float = 25.0        # 最大Cv值

    # 末端调流阀
    end_position: float = 178000.0   # 位置 (m)
    end_cv_max: float = 18.5         # 最大Cv值

    # 动作速率限制
    max_rate: float = 0.02           # 最大调节速率 (%/s)
    emergency_close_time: float = 60.0  # 紧急关闭时间 (s)


@dataclass
class SafetyConfig:
    """安全设施配置"""
    # 超压泄压阀
    relief_valve_1_set_pressure: float = 115.0  # #1泄压阀开启压力 (m)
    relief_valve_2_set_pressure: float = 96.0   # #2泄压阀开启压力 (m)
    relief_valve_close_pressure: float = 0.9    # 关闭压力系数 (相对开启)
    relief_valve_cv: float = 30.0               # 泄压阀Cv值

    # 空气阀
    air_valve_count: int = 232                  # 空气阀数量
    air_valve_spacing: float = 800.0            # 平均间距 (m)
    air_valve_intake_capacity: float = 0.5      # 进气能力 (m³/s)
    air_valve_exhaust_capacity: float = 0.2     # 排气能力 (m³/s)


@dataclass
class SimulationConfig:
    """仿真配置"""
    dt: float = 0.5                  # 时间步长 (s)
    total_time: float = 86400.0      # 总仿真时间 (s) = 24h

    # 记录间隔
    record_interval: float = 60.0    # 数据记录间隔 (s)

    # 收敛参数
    max_iterations: int = 100        # 最大迭代次数
    tolerance: float = 1e-6          # 收敛容差

    # 噪声参数
    sensor_noise_std: Dict[str, float] = field(default_factory=lambda: {
        'level': 0.01,      # 水位噪声 (m)
        'pressure': 0.5,    # 压力噪声 (kPa)
        'flow': 0.05,       # 流量噪声 (m³/s)
        'temperature': 0.1  # 温度噪声 (°C)
    })


@dataclass
class ControlConfig:
    """控制配置"""
    # MPC参数
    mpc_horizon: int = 20            # 预测时域
    mpc_control_horizon: int = 5     # 控制时域
    mpc_dt: float = 60.0             # MPC采样时间 (s)

    # ADMM参数
    admm_rho: float = 1.0            # 惩罚参数
    admm_max_iter: int = 50          # 最大迭代次数
    admm_tolerance: float = 1e-4     # 收敛容差

    # 通信参数
    heartbeat_interval: float = 0.1  # 心跳间隔 (s)
    timeout: float = 0.5             # 超时阈值 (s)

    # 安全边界
    pressure_margin: float = 10.0    # 压力安全裕度 (m)
    level_margin: float = 0.5        # 水位安全裕度 (m)


class Config:
    """全局配置类"""

    # 物理常数
    physics = PhysicsConstants()

    # 组件配置
    reservoir = ReservoirConfig()
    tunnel = TunnelConfig()
    pool = PoolConfig()
    surge_tank = SurgeTankConfig()
    pipeline = PipelineConfig()
    valve = ValveConfig()
    safety = SafetyConfig()

    # 运行配置
    simulation = SimulationConfig()
    control = ControlConfig()

    # 当前模式
    season_mode: SeasonMode = SeasonMode.NORMAL

    @classmethod
    def get_manning_n(cls) -> float:
        """根据季节模式返回糙率"""
        if cls.season_mode == SeasonMode.ICE_PERIOD:
            return cls.tunnel.manning_n_ice
        return cls.tunnel.manning_n_normal

    @classmethod
    def set_ice_mode(cls, enable: bool = True):
        """设置冰期模式"""
        cls.season_mode = SeasonMode.ICE_PERIOD if enable else SeasonMode.NORMAL

    @classmethod
    def to_dict(cls) -> dict:
        """导出配置为字典"""
        return {
            'physics': cls.physics.__dict__,
            'reservoir': cls.reservoir.__dict__,
            'tunnel': cls.tunnel.__dict__,
            'pool': cls.pool.__dict__,
            'surge_tank': cls.surge_tank.__dict__,
            'pipeline': cls.pipeline.__dict__,
            'valve': cls.valve.__dict__,
            'safety': cls.safety.__dict__,
            'simulation': cls.simulation.__dict__,
            'control': cls.control.__dict__,
            'season_mode': cls.season_mode.name
        }
