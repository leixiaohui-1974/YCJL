"""
配置基类 (Base Configuration Classes)
=====================================

提供水利工程项目配置的基类和验证框架。
具体工程项目通过继承这些基类来定义自己的配置。
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum, auto


class ValidationSeverity(Enum):
    """验证结果严重程度"""
    INFO = auto()       # 信息
    WARNING = auto()    # 警告
    ERROR = auto()      # 错误
    CRITICAL = auto()   # 严重错误


@dataclass
class ValidationResult:
    """配置验证结果"""
    is_valid: bool                              # 是否通过验证
    severity: ValidationSeverity                # 严重程度
    message: str                                # 消息
    field_name: Optional[str] = None            # 相关字段名
    suggestion: Optional[str] = None            # 修复建议


class ConfigValidator:
    """
    配置验证器

    提供通用的配置验证规则
    """

    @staticmethod
    def validate_positive(value: float, name: str) -> ValidationResult:
        """验证正数"""
        if value <= 0:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"{name} 必须为正数，当前值: {value}",
                field_name=name,
                suggestion=f"将 {name} 设置为大于0的值"
            )
        return ValidationResult(is_valid=True, severity=ValidationSeverity.INFO, message="OK")

    @staticmethod
    def validate_range(value: float, min_val: float, max_val: float,
                       name: str) -> ValidationResult:
        """验证范围"""
        if value < min_val or value > max_val:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"{name} 超出范围 [{min_val}, {max_val}]，当前值: {value}",
                field_name=name,
                suggestion=f"将 {name} 设置在 [{min_val}, {max_val}] 范围内"
            )
        return ValidationResult(is_valid=True, severity=ValidationSeverity.INFO, message="OK")

    @staticmethod
    def validate_less_than(value1: float, value2: float,
                           name1: str, name2: str) -> ValidationResult:
        """验证小于关系"""
        if value1 >= value2:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"{name1} ({value1}) 必须小于 {name2} ({value2})",
                field_name=name1,
                suggestion=f"调整 {name1} 使其小于 {name2}"
            )
        return ValidationResult(is_valid=True, severity=ValidationSeverity.INFO, message="OK")

    @staticmethod
    def validate_not_none(value: Any, name: str) -> ValidationResult:
        """验证非空"""
        if value is None:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"{name} 未设置 (None)",
                field_name=name,
                suggestion=f"为 {name} 提供有效值"
            )
        return ValidationResult(is_valid=True, severity=ValidationSeverity.INFO, message="OK")


@dataclass
class BaseGlobalPhysicsConfig:
    """
    全局物理配置基类

    定义所有工程项目通用的物理常数和仿真参数
    """
    # 物理常数
    G: float = 9.80665                          # 重力加速度 (m/s²)
    RHO_WATER: float = 998.2                    # 水密度 (kg/m³) @20°C
    PATM_HEAD: float = 10.33                    # 大气压水头 (m)
    KINEMATIC_VISCOSITY: float = 1.004e-6       # 运动粘度 (m²/s) @20°C
    BULK_MODULUS: float = 2.2e9                 # 水体积弹性模量 (Pa)
    VAPOR_PRESSURE_HEAD: float = -9.8           # 汽化压力水头 (m)

    # 仿真时间步长
    DT_PHYSICS: float = 0.5                     # 物理仿真步长 (s)
    DT_SCADA: float = 1.0                       # SCADA采样周期 (s)
    DT_L1_REFLEX: float = 0.01                  # L1反射层响应周期 (s)
    DT_L2_MPC: float = 60.0                     # L2 MPC采样周期 (s)
    MPC_HORIZON: int = 600                      # MPC预测时域步数

    # 数值稳定性
    MIN_DEPTH: float = 0.001                    # 最小水深 (m)
    MIN_FLOW: float = 1e-6                      # 最小流量 (m³/s)
    MAX_ITERATIONS: int = 100                   # 最大迭代次数
    CONVERGENCE_TOL: float = 1e-6               # 收敛容差


@dataclass
class BaseReservoirConfig:
    """
    水库配置基类

    定义水库的基本参数结构
    """
    NAME: str = ""                              # 水库名称
    LOCATION: str = ""                          # 位置

    # 特征水位 (m)
    NORMAL_LEVEL: float = 0.0                   # 正常蓄水位
    DEAD_LEVEL: float = 0.0                     # 死水位
    FLOOD_LIMIT_LEVEL: float = 0.0              # 汛限水位
    CHECK_FLOOD_LEVEL: float = 0.0              # 校核洪水位
    DESIGN_FLOOD_LEVEL: float = 0.0             # 设计洪水位

    # 库容 (亿m³)
    TOTAL_STORAGE: float = 0.0                  # 总库容
    USEFUL_STORAGE: float = 0.0                 # 兴利库容
    DEAD_STORAGE: float = 0.0                   # 死库容
    FLOOD_CONTROL_STORAGE: float = 0.0          # 防洪库容

    def validate(self) -> List[ValidationResult]:
        """验证水库配置"""
        results = []

        # 验证水位关系
        results.append(ConfigValidator.validate_less_than(
            self.DEAD_LEVEL, self.NORMAL_LEVEL, "死水位", "正常蓄水位"
        ))

        if self.FLOOD_LIMIT_LEVEL > 0:
            results.append(ConfigValidator.validate_less_than(
                self.FLOOD_LIMIT_LEVEL, self.CHECK_FLOOD_LEVEL, "汛限水位", "校核洪水位"
            ))

        # 验证库容
        results.append(ConfigValidator.validate_positive(self.TOTAL_STORAGE, "总库容"))

        return [r for r in results if not r.is_valid]


@dataclass
class BasePipelineConfig:
    """
    管道配置基类

    定义有压管道的基本参数结构
    """
    NAME: str = ""                              # 管道名称
    TOTAL_LENGTH: float = 0.0                   # 总长度 (m)

    # 管道规格
    INNER_DIAMETER: float = 0.0                 # 内径 (m)
    WALL_THICKNESS: float = 0.0                 # 壁厚 (m)

    # 材料参数
    YOUNGS_MODULUS: float = 35e9                # 弹性模量 (Pa)
    POISSON_RATIO: float = 0.2                  # 泊松比

    # 水力参数
    WAVE_SPEED: float = 1000.0                  # 压力波速 (m/s)
    DARCY_FRICTION: float = 0.012               # 达西摩阻系数

    # 设计压力 (m水头)
    DESIGN_PRESSURE: float = 0.0                # 设计压力
    MAX_WORKING_PRESSURE: float = 0.0           # 最大工作压力
    TEST_PRESSURE: float = 0.0                  # 试验压力

    @property
    def cross_section_area(self) -> float:
        """管道断面积"""
        return math.pi * (self.INNER_DIAMETER / 2) ** 2

    @property
    def hydraulic_diameter(self) -> float:
        """水力直径（圆管等于内径）"""
        return self.INNER_DIAMETER

    def validate(self) -> List[ValidationResult]:
        """验证管道配置"""
        results = []
        results.append(ConfigValidator.validate_positive(self.INNER_DIAMETER, "管道内径"))
        results.append(ConfigValidator.validate_positive(self.TOTAL_LENGTH, "管道长度"))
        results.append(ConfigValidator.validate_positive(self.WAVE_SPEED, "压力波速"))
        return [r for r in results if not r.is_valid]


@dataclass
class BasePumpStationConfig:
    """
    泵站配置基类

    定义泵站的基本参数结构
    """
    NAME: str = ""                              # 泵站名称
    LOCATION: str = ""                          # 位置

    # 泵组参数
    PUMP_COUNT: int = 0                         # 机组数量
    DESIGN_FLOW: float = 0.0                    # 设计流量 (m³/s)
    DESIGN_HEAD: float = 0.0                    # 设计扬程 (m)
    POWER_RATING: float = 0.0                   # 单机功率 (kW)
    RPM: float = 0.0                            # 额定转速 (rpm)
    PEAK_EFFICIENCY: float = 0.85               # 峰值效率

    # 水位约束
    INLET_MIN_LEVEL: float = 0.0                # 进水池最低水位
    INLET_MAX_LEVEL: float = 0.0                # 进水池最高水位
    OUTLET_DESIGN_LEVEL: float = 0.0            # 出水池设计水位
    OUTLET_MAX_LEVEL: float = 0.0               # 出水池最高水位

    # 瞬态参数 (可选)
    INERTIA_GD2: Optional[float] = None         # 转动惯量 kg*m²

    def validate(self) -> List[ValidationResult]:
        """验证泵站配置"""
        results = []
        results.append(ConfigValidator.validate_positive(self.PUMP_COUNT, "机组数量"))
        results.append(ConfigValidator.validate_positive(self.DESIGN_FLOW, "设计流量"))
        results.append(ConfigValidator.validate_positive(self.DESIGN_HEAD, "设计扬程"))
        results.append(ConfigValidator.validate_range(self.PEAK_EFFICIENCY, 0.5, 1.0, "峰值效率"))
        return [r for r in results if not r.is_valid]


@dataclass
class BaseControlConfig:
    """
    控制参数配置基类
    """
    # PID参数
    PID_KP: float = 0.1                         # 比例增益
    PID_KI: float = 0.01                        # 积分增益
    PID_KD: float = 0.02                        # 微分增益
    PID_INTEGRAL_LIMIT: float = 10.0            # 积分限幅

    # MPC参数
    MPC_PREDICTION_HORIZON: int = 20            # 预测时域
    MPC_CONTROL_HORIZON: int = 5                # 控制时域
    MPC_SAMPLE_TIME: float = 60.0               # 采样时间 (s)

    # 安全边界
    SAFETY_MARGIN_PRESSURE: float = 10.0        # 压力安全裕度 (m)
    SAFETY_MARGIN_LEVEL: float = 0.5            # 水位安全裕度 (m)


@dataclass
class BaseSafetyConfig:
    """
    安全设施配置基类
    """
    # 压力报警阈值
    PRESSURE_ALARM_HIGH: float = 100.0          # 高压报警 (m)
    PRESSURE_ALARM_LOW: float = -5.0            # 负压报警 (m)
    PRESSURE_TRIP_HIGH: float = 110.0           # 超压停机 (m)

    # 水位报警阈值
    LEVEL_ALARM_MARGIN: float = 0.3             # 水位报警裕度 (m)


@dataclass
class BaseProjectConfig(ABC):
    """
    项目配置总成基类

    所有水利工程项目配置的抽象基类
    """
    # 版本信息
    VERSION: str = "1.0.0"
    BUILD_DATE: str = ""
    PROJECT_NAME: str = ""

    @abstractmethod
    def validate(self) -> List[str]:
        """
        验证配置完整性和一致性

        Returns:
            错误信息列表，空列表表示验证通过
        """
        pass

    @abstractmethod
    def get_summary(self) -> Dict:
        """
        获取配置摘要

        Returns:
            配置摘要字典
        """
        pass

    def validate_all(self) -> List[ValidationResult]:
        """
        执行完整验证

        Returns:
            所有验证结果列表
        """
        results = []
        # 子类实现具体验证
        return results

    def get_validation_report(self) -> str:
        """生成验证报告"""
        results = self.validate_all()
        if not results:
            return "配置验证通过 ✓"

        lines = ["配置验证报告:", "=" * 40]
        for r in results:
            icon = {"ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️", "CRITICAL": "🚨"}.get(
                r.severity.name, "•"
            )
            lines.append(f"{icon} [{r.severity.name}] {r.message}")
            if r.suggestion:
                lines.append(f"   建议: {r.suggestion}")
        return "\n".join(lines)


__all__ = [
    'ValidationSeverity',
    'ValidationResult',
    'ConfigValidator',
    'BaseGlobalPhysicsConfig',
    'BaseReservoirConfig',
    'BasePipelineConfig',
    'BasePumpStationConfig',
    'BaseControlConfig',
    'BaseSafetyConfig',
    'BaseProjectConfig'
]
