"""
配置数据库测试
=============

测试配置数据库v3.2的完整性和正确性
"""

import pytest
import math
import numpy as np
from datetime import datetime

# 导入配置数据库
from ycjl.config.config_database import (
    YinChuoProjectConfig,
    ProjectParams,
    GlobalPhysicsConfig,
    CharacteristicCurves,
    SourceHubConfig,
    TunnelSystemConfig,
    StabilizingPoolConfig,
    PipelineSystemConfig,
    SurgeTankConfig,
    RegulatingValveConfig,
    EndUserConfig,
    SafetySystemConfig,
    SimulationHardwareConfig,
    ControlParameterConfig,
    HydrologyConfig,
    GlobalConfig,
    CurveDatabase,
    SourceConfig,
    create_interpolator
)


class TestGlobalConfig:
    """全局配置测试"""

    def test_physics_constants(self):
        """测试物理常数"""
        assert GlobalConfig.G == pytest.approx(9.80665, rel=1e-5)
        assert GlobalConfig.RHO_WATER == pytest.approx(998.2, rel=1e-3)
        assert GlobalConfig.PATM_HEAD == pytest.approx(10.33, rel=1e-2)

    def test_time_steps(self):
        """测试时间步长"""
        assert GlobalConfig.DT_PHYSICS == 0.5
        assert GlobalConfig.DT_SCADA == 1.0
        assert GlobalConfig.DT_L1_REFLEX == 0.01

    def test_ice_parameters(self):
        """测试冰期参数"""
        assert GlobalConfig.TEMP_ICE_POINT == 0.5
        assert GlobalConfig.ROUGHNESS_INCREASE_ICE > 0


class TestCharacteristicCurves:
    """特性曲线测试"""

    def test_wendegen_zv_data(self):
        """测试文得根水库水位-库容曲线"""
        # 检查数据点数量
        assert len(CurveDatabase.WENDEGEN_ZV_DATA) >= 5

        # 检查单调性
        levels = [p[0] for p in CurveDatabase.WENDEGEN_ZV_DATA]
        volumes = [p[1] for p in CurveDatabase.WENDEGEN_ZV_DATA]
        assert levels == sorted(levels), "水位应单调递增"
        assert volumes == sorted(volumes), "库容应单调递增"

    def test_wendegen_volume_interpolation(self):
        """测试水位-库容插值"""
        # 死水位库容
        v_dead = CurveDatabase.get_wendegen_volume(351.0)
        assert v_dead == pytest.approx(4.46e8, rel=0.01)

        # 正常蓄水位库容
        v_normal = CurveDatabase.get_wendegen_volume(377.0)
        assert v_normal == pytest.approx(19.64e8, rel=0.01)

    def test_wendegen_level_interpolation(self):
        """测试库容-水位反查"""
        # 测试可逆性
        level = 370.0
        volume = CurveDatabase.get_wendegen_volume(level)
        level_back = CurveDatabase.get_wendegen_level(volume)
        assert level_back == pytest.approx(level, abs=0.5)

    def test_spillway_discharge(self):
        """测试溢洪道泄流"""
        # 堰顶以下无泄流
        assert CurveDatabase.get_spillway_discharge(360.0) == 0.0

        # 水位超过堰顶
        q = CurveDatabase.get_spillway_discharge(370.0)
        assert q > 0
        # Q = 117 * (370-363)^1.5 = 117 * 7^1.5 ≈ 2167
        assert q == pytest.approx(2167, rel=0.05)

    def test_inline_valve_zeta(self):
        """测试在线阀流阻曲线"""
        # 全关阻力很大
        zeta_closed = CurveDatabase.get_inline_valve_zeta(0)
        assert zeta_closed > 1e6

        # 全开阻力较小
        zeta_open = CurveDatabase.get_inline_valve_zeta(100)
        assert zeta_open < 2.0

        # 单调递减
        assert CurveDatabase.get_inline_valve_zeta(50) < CurveDatabase.get_inline_valve_zeta(30)

    def test_butterfly_valve_zeta(self):
        """测试蝶阀流阻曲线"""
        zeta_open = CurveDatabase.get_butterfly_valve_zeta(100)
        # 蝶阀全开仍有阻力
        assert 0.1 < zeta_open < 1.0

    def test_operation_zone(self):
        """测试调度分区判断"""
        # 7月正常区
        zone = CurveDatabase.get_operation_zone(7, 365.0)
        assert zone == 'NORMAL'

        # 7月超蓄区
        zone = CurveDatabase.get_operation_zone(7, 372.0)
        assert zone == 'UPPER'

        # 死库容区
        zone = CurveDatabase.get_operation_zone(7, 350.0)
        assert zone == 'DEAD'

    def test_fishway_outlet(self):
        """测试鱼道出口选择"""
        outlet = CurveDatabase.get_fishway_outlet(370.0)
        assert outlet == 3  # 369.5-371.0 -> 3号出口

        outlet = CurveDatabase.get_fishway_outlet(376.0)
        assert outlet == 7  # 最高出口

    def test_turbine_efficiency(self):
        """测试水轮机效率"""
        # 额定工况效率应最高
        eta_rated = CurveDatabase.get_turbine_efficiency(34.0, 11.75, 'large')
        assert eta_rated >= 0.94

        # 偏离额定效率下降
        eta_low = CurveDatabase.get_turbine_efficiency(21.0, 4.0, 'large')
        assert eta_low < eta_rated


class TestSourceConfig:
    """水源枢纽配置测试"""

    def test_water_levels(self):
        """测试水位参数"""
        assert SourceConfig.DEAD_LEVEL < SourceConfig.FLOOD_LIMIT_LEVEL
        assert SourceConfig.FLOOD_LIMIT_LEVEL < SourceConfig.NORMAL_LEVEL
        assert SourceConfig.NORMAL_LEVEL < SourceConfig.DESIGN_FLOOD_LEVEL
        assert SourceConfig.DESIGN_FLOOD_LEVEL < SourceConfig.CHECK_FLOOD_LEVEL

    def test_storage_capacity(self):
        """测试库容"""
        # 总库容 > 兴利库容 > 死库容
        assert SourceConfig.TOTAL_STORAGE > SourceConfig.NORMAL_STORAGE
        assert SourceConfig.NORMAL_STORAGE > SourceConfig.DEAD_STORAGE

    def test_turbine_count(self):
        """测试机组配置"""
        assert SourceConfig.TURBINE_L_COUNT == 3
        assert SourceConfig.TURBINE_S_COUNT == 1

    def test_intake_design_flow(self):
        """测试进水口设计流量"""
        assert SourceConfig.INTAKE_DESIGN_FLOW == pytest.approx(18.58, rel=0.01)


class TestProjectConfigValidation:
    """项目配置验证测试"""

    def test_validate_success(self):
        """测试配置验证"""
        errors = ProjectParams.validate()
        # 应该没有严重错误
        assert len([e for e in errors if '必须' in e]) == 0

    def test_get_summary(self):
        """测试配置摘要"""
        summary = ProjectParams.get_summary()
        assert 'version' in summary
        assert 'source' in summary
        assert 'conveyance' in summary
        assert 'end_users' in summary


class TestInterpolator:
    """插值器测试"""

    def test_create_interpolator(self):
        """测试创建插值器"""
        data = [(0, 0), (1, 1), (2, 4), (3, 9)]
        interp = create_interpolator(data)

        # 边界值
        assert interp(0) == pytest.approx(0, abs=0.01)
        assert interp(3) == pytest.approx(9, abs=0.01)

        # 中间值插值
        assert 0 < interp(0.5) < 1

    def test_interpolator_boundary(self):
        """测试边界外插值"""
        data = [(0, 0), (1, 10)]
        interp = create_interpolator(data)

        # 边界外应该钳位
        assert interp(-1) == pytest.approx(0, abs=0.1)
        assert interp(2) == pytest.approx(10, abs=0.1)


class TestEndUserConfig:
    """用户需求配置测试"""

    def test_total_demand(self):
        """测试总需求"""
        from ycjl.config.config_database import UserConfig
        total = UserConfig.total_design_flow
        assert total > 0
        assert total < SourceConfig.INTAKE_DESIGN_FLOW * 1.5


class TestHydrologyConfig:
    """水文配置测试"""

    def test_flood_frequency(self):
        """测试洪水频率"""
        from ycjl.config.config_database import HydroConfig

        # 频率越低，洪峰越大
        assert HydroConfig.FLOOD_FREQUENCY['P=0.01%'] > HydroConfig.FLOOD_FREQUENCY['P=1%']
        assert HydroConfig.FLOOD_FREQUENCY['P=1%'] > HydroConfig.FLOOD_FREQUENCY['P=10%']

    def test_monthly_inflow_pattern(self):
        """测试月来水分配"""
        from ycjl.config.config_database import HydroConfig

        # 各月百分比之和应接近1
        total = sum(HydroConfig.MONTHLY_INFLOW_PATTERN.values())
        assert total == pytest.approx(1.0, rel=0.01)

        # 7-8月占比最大
        assert HydroConfig.MONTHLY_INFLOW_PATTERN[7] >= 0.2
        assert HydroConfig.MONTHLY_INFLOW_PATTERN[8] >= 0.2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
