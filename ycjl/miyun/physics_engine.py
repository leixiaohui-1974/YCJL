"""
密云水库调蓄工程物理仿真引擎 (Physics Engine) v1.0
=================================================

L5级自主运行物理引擎，基于全参数数据库进行仿真计算

功能：
1. 扬程计算 - 基于水位和流量的精确扬程计算
2. 功率计算 - 考虑效率曲线的实际功率
3. 水力损失计算 - 明渠/管道水力损失
4. 安全诊断 - 水位约束、泵工况、负压检测
5. 系统状态评估 - 全系统健康状态
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum, auto

from .config_database import (
    RouteType, STATION_DB, MiyunParams,
    MiyunGlobalConfig, MiyunCurveDatabase
)


# ==========================================
# 1. 系统状态枚举
# ==========================================
class SystemStatus(Enum):
    """系统运行状态"""
    NORMAL = "NORMAL"           # 正常运行
    WARNING = "WARNING"         # 警告状态
    CRITICAL = "CRITICAL"       # 临界状态
    EMERGENCY = "EMERGENCY"     # 紧急状态
    SHUTDOWN = "SHUTDOWN"       # 停机状态


class PumpStatus(Enum):
    """泵运行状态"""
    RUNNING = "RUNNING"         # 运行中
    STANDBY = "STANDBY"         # 待机
    STARTING = "STARTING"       # 启动中
    STOPPING = "STOPPING"       # 停止中
    FAULT = "FAULT"             # 故障
    MAINTENANCE = "MAINTENANCE" # 检修


# ==========================================
# 2. 计算结果数据类
# ==========================================
@dataclass
class HeadCalculationResult:
    """扬程计算结果"""
    station_key: str
    station_name: str
    total_head: float           # 总扬程 (m)
    static_head: float          # 静扬程 (m)
    friction_loss: float        # 摩阻损失 (m)
    local_loss: float           # 局部损失 (m)
    power_required: float       # 所需功率 (kW)
    efficiency: float           # 效率
    status: SystemStatus        # 系统状态
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        warn_str = " | ".join(self.warnings) if self.warnings else "无"
        return (f"{self.station_name}: H={self.total_head:.2f}m, "
                f"P={self.power_required:.0f}kW, Status={self.status.value}, "
                f"Warnings: {warn_str}")


@dataclass
class SystemDiagnosisResult:
    """系统诊断结果"""
    timestamp: float
    flow_rate: float            # 系统流量 (m³/s)
    total_power: float          # 总功耗 (kW)
    total_head: float           # 总扬程 (m)
    overall_status: SystemStatus
    station_results: List[HeadCalculationResult] = field(default_factory=list)
    system_warnings: List[str] = field(default_factory=list)

    @property
    def total_power_mw(self) -> float:
        """总功耗 (MW)"""
        return self.total_power / 1000.0


# ==========================================
# 3. 仿真引擎类
# ==========================================
class MiyunSimulationEngine:
    """
    密云水库调蓄工程物理仿真引擎

    基于全参数数据库进行精确的物理仿真
    """

    def __init__(self):
        """初始化仿真引擎"""
        self.g = MiyunGlobalConfig.G
        self.rho = MiyunGlobalConfig.RHO_WATER
        self.curves = MiyunCurveDatabase

    # ---------------------------------------------------------
    # 3.1 扬程计算
    # ---------------------------------------------------------
    def calculate_pumping_head(
        self,
        station_key: str,
        Q: float,
        current_level_up: float,
        current_level_down: float
    ) -> HeadCalculationResult:
        """
        精确计算泵站扬程与功率

        Args:
            station_key: 泵站键名
            Q: 流量 (m³/s)
            current_level_up: 上游（进水池）水位 (m)
            current_level_down: 下游（出水池）水位 (m)

        Returns:
            HeadCalculationResult: 计算结果
        """
        if station_key not in STATION_DB:
            return HeadCalculationResult(
                station_key=station_key,
                station_name="Unknown",
                total_head=0.0,
                static_head=0.0,
                friction_loss=0.0,
                local_loss=0.0,
                power_required=0.0,
                efficiency=0.0,
                status=SystemStatus.CRITICAL,
                warnings=["泵站不存在"]
            )

        cfg = STATION_DB[station_key]
        p_cfg = cfg["pump"]
        l_cfg = cfg["levels"]

        # 1. 静扬程计算
        static_head = current_level_down - current_level_up

        # 2. 水力损失计算
        friction_loss = 0.0
        local_loss = 0.0

        if cfg["type"] == RouteType.PIPELINE:
            # 有压管道: Darcy-Weisbach公式
            pipe = cfg["pipe_geo"]
            D = pipe["Diameter_mm"] / 1000.0  # 转换为m
            L = pipe["Length_m"]
            f = pipe["Roughness"]

            A = math.pi * (D / 2) ** 2
            V = Q / A if A > 0 else 0

            # hf = f * L/D * V²/(2g)
            if D > 0:
                friction_loss = f * (L / D) * (V ** 2) / (2 * self.g)

            # 局部损失 (入口+出口+弯头等)
            k_local = 2.0  # 综合局部损失系数
            local_loss = k_local * (V ** 2) / (2 * self.g)

        else:
            # 明渠段: Manning公式 (损失较小，主要体现在静扬程变化)
            channel = cfg["channel_geo"]
            n = channel["Roughness"]
            S = channel["Slope"]
            B = channel["Bottom_W"]
            m = channel["Side_Slope"]

            # 估算水深 (假设梯形断面)
            # Q = A * V, V = (1/n) * R^(2/3) * S^(1/2)
            # 简化：使用设计水深估算
            h = 2.5  # 假设平均水深
            A = (B + m * h) * h
            P = B + 2 * h * math.sqrt(1 + m ** 2)
            R = A / P if P > 0 else 0

            V = Q / A if A > 0 else 0

            # 渠道沿程损失相对较小
            L = channel["Length_km"] * 1000
            if R > 0:
                friction_loss = (n ** 2 * V ** 2 * L) / (R ** (4/3))

        # 3. 总扬程
        total_head = static_head + friction_loss + local_loss

        # 4. 效率与功率计算
        q_ratio = Q / p_cfg["Q_des"] if p_cfg["Q_des"] > 0 else 1.0
        efficiency = self._get_pump_efficiency(p_cfg, q_ratio)

        # P = rho * g * Q * H / eta
        power_required = 0.0
        if efficiency > 0:
            power_required = (self.rho * self.g * Q * total_head) / (efficiency * 1000)  # kW

        # 5. 安全性诊断
        status, warnings = self._diagnose_station(
            station_key, cfg, Q, total_head,
            current_level_up, current_level_down
        )

        return HeadCalculationResult(
            station_key=station_key,
            station_name=cfg["name"],
            total_head=total_head,
            static_head=static_head,
            friction_loss=friction_loss,
            local_loss=local_loss,
            power_required=power_required,
            efficiency=efficiency,
            status=status,
            warnings=warnings
        )

    def _get_pump_efficiency(self, pump_cfg: dict, q_ratio: float) -> float:
        """获取泵效率"""
        # 优先使用泵配置中的峰值效率
        peak_eff = pump_cfg.get("Eff_Peak", 0.85)

        # 根据流量比修正效率
        if q_ratio < 0.5:
            return peak_eff * 0.7  # 小流量效率下降
        elif q_ratio < 0.8:
            return peak_eff * 0.9
        elif q_ratio <= 1.1:
            return peak_eff  # 设计点附近
        elif q_ratio <= 1.2:
            return peak_eff * 0.95
        else:
            return peak_eff * 0.85  # 大流量效率下降

    def _diagnose_station(
        self,
        station_key: str,
        cfg: dict,
        Q: float,
        total_head: float,
        level_up: float,
        level_down: float
    ) -> Tuple[SystemStatus, List[str]]:
        """
        泵站安全诊断

        Returns:
            (status, warnings)
        """
        p_cfg = cfg["pump"]
        l_cfg = cfg["levels"]
        warnings = []
        status = SystemStatus.NORMAL

        # A. 水位约束检查
        if level_up < l_cfg["Inlet_Min"]:
            warnings.append(f"进水池抽空 (当前{level_up:.2f} < {l_cfg['Inlet_Min']})")
            status = SystemStatus.CRITICAL

        if level_down > l_cfg["Outlet_Max"]:
            warnings.append(f"出水池漫堤 (当前{level_down:.2f} > {l_cfg['Outlet_Max']})")
            status = SystemStatus.CRITICAL

        # B. 泵工况检查
        if total_head < p_cfg["H_min"]:
            warnings.append(f"严重振动区 (H={total_head:.2f} < {p_cfg['H_min']})")
            if status != SystemStatus.CRITICAL:
                status = SystemStatus.WARNING

        if total_head > p_cfg["H_max"]:
            warnings.append(f"扬程过高 (H={total_head:.2f} > {p_cfg['H_max']})")
            if status != SystemStatus.CRITICAL:
                status = SystemStatus.WARNING

        # C. 流量检查
        q_ratio = Q / p_cfg["Q_des"] if p_cfg["Q_des"] > 0 else 0
        if q_ratio > 1.2:
            warnings.append(f"流量超载 (Q比={q_ratio:.2f})")
            if status != SystemStatus.CRITICAL:
                status = SystemStatus.WARNING

        # D. 管道负压检查 (针对雁栖段高点)
        if station_key == "Yanqi" and "pipe_geo" in cfg:
            pipe = cfg["pipe_geo"]
            if "Critical_Nodes" in pipe:
                for node in pipe["Critical_Nodes"]:
                    if node["name"] == "AV-16":
                        av16_elev = node["elev"]
                        # 估算水力坡度线高程
                        hgl_inlet = level_up + total_head
                        if hgl_inlet < av16_elev + 5.0:  # 留5m安全裕度
                            warnings.append(f"高点负压风险 ({node['name']})")
                            status = SystemStatus.CRITICAL

        return status, warnings

    # ---------------------------------------------------------
    # 3.2 系统级诊断
    # ---------------------------------------------------------
    def run_system_diagnosis(
        self,
        flow_scenario: float,
        timestamp: float = 0.0,
        level_conditions: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> SystemDiagnosisResult:
        """
        全系统诊断

        Args:
            flow_scenario: 输水流量场景 (m³/s)
            timestamp: 时间戳
            level_conditions: 各站水位条件 {station_key: (level_up, level_down)}

        Returns:
            SystemDiagnosisResult: 系统诊断结果
        """
        station_results = []
        total_power = 0.0
        total_head = 0.0
        system_warnings = []
        worst_status = SystemStatus.NORMAL

        for station_key, data in STATION_DB.items():
            # 确定水位边界条件
            if level_conditions and station_key in level_conditions:
                level_up, level_down = level_conditions[station_key]
            else:
                # 使用默认条件模拟
                l_cfg = data["levels"]
                # 场景模拟：低流量时水位接近Min，高流量时水位接近Max
                if flow_scenario < 10:
                    level_up = l_cfg["Inlet_Min"] + 0.2
                else:
                    level_up = l_cfg["Inlet_Max"] - 0.2
                level_down = l_cfg["Outlet_Des"]

            # 计算各站扬程
            result = self.calculate_pumping_head(
                station_key, flow_scenario, level_up, level_down
            )
            station_results.append(result)

            # 累计功率和扬程
            total_power += result.power_required
            total_head += result.total_head

            # 更新系统状态
            if result.status.value == SystemStatus.CRITICAL.value:
                worst_status = SystemStatus.CRITICAL
            elif result.status.value == SystemStatus.WARNING.value:
                if worst_status != SystemStatus.CRITICAL:
                    worst_status = SystemStatus.WARNING

            # 收集警告
            for warn in result.warnings:
                system_warnings.append(f"[{data['name']}] {warn}")

        return SystemDiagnosisResult(
            timestamp=timestamp,
            flow_rate=flow_scenario,
            total_power=total_power,
            total_head=total_head,
            overall_status=worst_status,
            station_results=station_results,
            system_warnings=system_warnings
        )

    # ---------------------------------------------------------
    # 3.3 明渠水力计算
    # ---------------------------------------------------------
    def calculate_channel_flow(
        self,
        station_key: str,
        upstream_level: float,
        downstream_level: float
    ) -> float:
        """
        明渠段自由流量计算 (Manning公式)

        Args:
            station_key: 泵站键名
            upstream_level: 上游水位 (m)
            downstream_level: 下游水位 (m)

        Returns:
            流量 (m³/s)
        """
        if station_key not in STATION_DB:
            return 0.0

        cfg = STATION_DB[station_key]
        if cfg["type"] != RouteType.CHANNEL:
            return 0.0

        channel = cfg["channel_geo"]
        n = channel["Roughness"]
        S = channel["Slope"]
        B = channel["Bottom_W"]
        m = channel["Side_Slope"]

        # 平均水深
        h = max(0.1, (upstream_level + downstream_level) / 2.0 - 50.0)  # 假设渠底高程约50m

        # 梯形断面
        A = (B + m * h) * h
        P = B + 2 * h * math.sqrt(1 + m ** 2)
        R = A / P if P > 0 else 0

        # Manning公式
        if R > 0 and n > 0:
            V = (1 / n) * (R ** (2/3)) * (S ** 0.5)
            Q = A * V
            return Q

        return 0.0

    # ---------------------------------------------------------
    # 3.4 管道水锤估算 (简化)
    # ---------------------------------------------------------
    def estimate_water_hammer(
        self,
        station_key: str,
        Q: float,
        valve_close_time: float
    ) -> Dict[str, float]:
        """
        估算水锤压力升高 (Joukowsky公式简化)

        Args:
            station_key: 泵站键名
            Q: 流量 (m³/s)
            valve_close_time: 阀门关闭时间 (s)

        Returns:
            {"delta_H": 压力升高, "max_pressure": 最大压力, "is_safe": 是否安全}
        """
        if station_key not in STATION_DB:
            return {"delta_H": 0.0, "max_pressure": 0.0, "is_safe": True}

        cfg = STATION_DB[station_key]
        if cfg["type"] != RouteType.PIPELINE:
            return {"delta_H": 0.0, "max_pressure": 0.0, "is_safe": True}

        pipe = cfg.get("pipe_geo", {})
        D = pipe.get("Diameter_mm", 2600) / 1000.0
        L = pipe.get("Length_m", 1000)

        # 波速 (如果缺失使用默认值)
        a = pipe.get("Wave_Speed_a") or 1000.0

        A = math.pi * (D / 2) ** 2
        V = Q / A if A > 0 else 0

        # 管道特征时间
        T_c = 2 * L / a

        # Joukowsky公式
        if valve_close_time < T_c:
            # 快关阀 (直接水击)
            delta_H = a * V / self.g
        else:
            # 慢关阀 (间接水击)
            delta_H = (a * V / self.g) * (T_c / valve_close_time)

        # 静态水头
        static_head = pipe.get("Static_Head", 30.0)
        max_pressure = static_head + delta_H

        # 安全判断
        max_safe = MiyunParams.Pipeline.MAX_WORKING_PRESSURE
        is_safe = max_pressure < max_safe

        return {
            "delta_H": delta_H,
            "max_pressure": max_pressure,
            "is_safe": is_safe,
            "wave_speed": a,
            "characteristic_time": T_c
        }

    # ---------------------------------------------------------
    # 3.5 效率优化建议
    # ---------------------------------------------------------
    def optimize_pump_operation(
        self,
        target_flow: float
    ) -> Dict[str, Dict]:
        """
        泵站运行优化建议

        Args:
            target_flow: 目标流量 (m³/s)

        Returns:
            各站优化建议
        """
        recommendations = {}

        for station_key, cfg in STATION_DB.items():
            p_cfg = cfg["pump"]
            Q_des = p_cfg["Q_des"]
            count = p_cfg["Count"]

            # 计算最优运行台数
            if target_flow <= 0:
                opt_count = 0
            else:
                # 每台泵最佳工况点流量
                q_per_pump = Q_des / count
                opt_count = min(count, max(1, round(target_flow / q_per_pump)))

            # 实际每台流量
            q_actual = target_flow / opt_count if opt_count > 0 else 0
            q_ratio = q_actual / (Q_des / count) if count > 0 else 0

            # 效率估算
            efficiency = self._get_pump_efficiency(p_cfg, q_ratio)

            recommendations[station_key] = {
                "station_name": cfg["name"],
                "target_flow": target_flow,
                "optimal_pump_count": opt_count,
                "flow_per_pump": q_actual,
                "load_ratio": q_ratio,
                "estimated_efficiency": efficiency,
                "status": "optimal" if 0.8 <= q_ratio <= 1.1 else "suboptimal"
            }

        return recommendations


# ==========================================
# 模块级实例
# ==========================================
SimEngine = MiyunSimulationEngine()


# ==========================================
# 导出
# ==========================================
__all__ = [
    'SystemStatus',
    'PumpStatus',
    'HeadCalculationResult',
    'SystemDiagnosisResult',
    'MiyunSimulationEngine',
    'SimEngine'
]
