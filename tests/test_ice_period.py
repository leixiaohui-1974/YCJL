"""
冰期模型测试
============

测试冰期物理模型和控制策略
"""

import pytest
import math
from datetime import datetime

# 冰期参数
from ycjl.config.ice_parameters import (
    IceType, IcePhase, BreakupType,
    IceParams, IcePhysical, StefanParams, RoughnessParams,
    FrazilParams, CoverParams, BreakupParams,
    IceHydraulicsConfig
)

# 冰期物理模型
from ycjl.physics.ice_model import (
    IceState, ChannelGeometry, MeteoCondition,
    IceThicknessModel, IceCoverHydraulics, FrazilIceModel,
    IceCoverFormation, BreakupModel, IcePeriodSimulator
)

# 冰期控制
from ycjl.control.ice_strategy import (
    IceOperationMode, IceAlarmLevel,
    IceOperationConstraints, IceControlDecision,
    IcePeriodController, IceFlowRateLimiter, IceMonitor
)


class TestIceParameters:
    """冰期参数测试"""

    def test_physical_properties(self):
        """测试物理性质参数"""
        assert IcePhysical.RHO_ICE == pytest.approx(917.0, rel=0.01)
        assert IcePhysical.RHO_WATER == pytest.approx(999.8, rel=0.01)
        assert IcePhysical.LATENT_HEAT_FUSION == 334000

    def test_stefan_params(self):
        """测试Stefan方程参数"""
        assert StefanParams.ALPHA_RIVER_ICE == pytest.approx(2.4, rel=0.1)
        assert StefanParams.MAX_ICE_THICKNESS == 2.0
        assert StefanParams.AFDD_ICE_START > 0

    def test_roughness_params(self):
        """测试糙率参数"""
        # 床面糙率
        assert RoughnessParams.N_BED_CONCRETE < RoughnessParams.N_BED_NATURAL

        # 冰底糙率
        assert RoughnessParams.N_ICE_SMOOTH < RoughnessParams.N_ICE_JAM

    def test_config_instance(self):
        """测试配置实例"""
        config = IceHydraulicsConfig()
        assert config.VERSION == "3.3.0"

        # 测试Stefan方程计算
        h = config.stefan_ice_thickness(400)  # 400 °C·day
        assert h > 0
        assert h < 1.0  # 应小于1m

    def test_belokon_sabaneev(self):
        """测试Belokon-Sabaneev公式"""
        config = IceHydraulicsConfig()

        # 典型值计算
        n_c = config.belokon_sabaneev_roughness(0.025, 0.015)
        assert 0.015 < n_c < 0.025  # 应在两者之间

        # 对称性
        n_c1 = config.belokon_sabaneev_roughness(0.02, 0.03)
        n_c2 = config.belokon_sabaneev_roughness(0.03, 0.02)
        assert n_c1 == pytest.approx(n_c2, rel=0.01)


class TestIceThicknessModel:
    """冰厚模型测试"""

    def setup_method(self):
        self.model = IceThicknessModel()

    def test_stefan_growth(self):
        """测试Stefan冰厚生长"""
        # AFDD = 400 °C·day, α = 2.4
        # h = 2.4/100 × √400 = 0.024 × 20 = 0.48 m
        h = self.model.stefan_growth(400)
        assert h == pytest.approx(0.48, rel=0.1)

    def test_stefan_with_snow(self):
        """测试有雪盖的冰厚"""
        h_no_snow = self.model.stefan_growth(400, snow_depth=0)
        h_with_snow = self.model.stefan_growth(400, snow_depth=0.2)

        # 有雪盖应更薄
        assert h_with_snow < h_no_snow

    def test_stefan_max_limit(self):
        """测试冰厚上限"""
        # 极端AFDD
        h = self.model.stefan_growth(10000)
        assert h <= StefanParams.MAX_ICE_THICKNESS

    def test_ashton_growth(self):
        """测试Ashton模型增量"""
        dh = self.model.ashton_growth(
            current_thickness=0.1,
            air_temp=-15,
            water_temp=0.0,
            dt=3600  # 1小时
        )
        assert dh > 0  # 应该增厚

    def test_ashton_melt(self):
        """测试Ashton模型融化"""
        dh = self.model.ashton_growth(
            current_thickness=0.5,
            air_temp=5,  # 正温
            water_temp=1.0,
            dt=3600
        )
        assert dh < 0  # 应该融化


class TestIceCoverHydraulics:
    """冰盖水力学测试"""

    def setup_method(self):
        self.hydraulics = IceCoverHydraulics()

    def test_composite_roughness_range(self):
        """测试复合糙率范围"""
        n_c = self.hydraulics.belokon_sabaneev_roughness(0.025, 0.015)
        assert RoughnessParams.N_COMPOSITE_MIN <= n_c <= RoughnessParams.N_COMPOSITE_MAX

    def test_ice_roughness_from_thickness(self):
        """测试冰厚-糙率关系"""
        # 薄冰糙率较高
        n_thin = self.hydraulics.ice_roughness_from_thickness(0.1, IceType.SHEET)
        n_thick = self.hydraulics.ice_roughness_from_thickness(0.5, IceType.SHEET)
        assert n_thin >= n_thick

        # 不同类型
        n_sheet = self.hydraulics.ice_roughness_from_thickness(0.3, IceType.SHEET)
        n_frazil = self.hydraulics.ice_roughness_from_thickness(0.3, IceType.FRAZIL)
        assert n_frazil > n_sheet

    def test_hydraulic_radius_ice_cover(self):
        """测试冰盖水力半径"""
        geometry = ChannelGeometry(
            width=10.0,
            depth=3.0,
            area=30.0,
            wetted_perimeter=16.0,  # 床面 + 两侧
            hydraulic_radius=1.875
        )

        R_ice = self.hydraulics.hydraulic_radius_ice_cover(geometry, ice_thickness=0.3)

        # 冰盖条件下水力半径应减小
        assert R_ice < geometry.hydraulic_radius

    def test_conveyance_factor(self):
        """测试过流能力折减"""
        factor = self.hydraulics.conveyance_factor(
            n_open=0.025,
            n_composite=0.030,
            R_open=2.0,
            R_ice=1.5
        )

        assert 0 < factor < 1

    def test_manning_flow(self):
        """测试曼宁流量计算"""
        geometry = ChannelGeometry(
            width=10.0,
            depth=3.0,
            area=30.0,
            wetted_perimeter=16.0,
            hydraulic_radius=1.875,
            slope=0.0005,
            bed_roughness=0.025
        )

        Q = self.hydraulics.manning_flow_ice_cover(
            geometry,
            ice_thickness=0.3,
            n_bed=0.025,
            n_ice=0.015
        )

        assert Q > 0
        # 冰盖条件应小于无冰期
        Q_open = (1/0.025) * 30 * 1.875**(2/3) * math.sqrt(0.0005)
        assert Q < Q_open


class TestFrazilIceModel:
    """Frazil冰模型测试"""

    def setup_method(self):
        self.model = FrazilIceModel()

    def test_frazil_conditions(self):
        """测试Frazil生成条件判断"""
        # 满足条件
        result = self.model.check_frazil_conditions(
            velocity=0.8,
            depth=2.0,
            supercooling=0.03
        )
        assert result is True

        # 流速不足
        result = self.model.check_frazil_conditions(
            velocity=0.3,
            depth=2.0,
            supercooling=0.03
        )
        assert result is False

        # 过冷度不足
        result = self.model.check_frazil_conditions(
            velocity=0.8,
            depth=2.0,
            supercooling=0.005
        )
        assert result is False

    def test_production_rate(self):
        """测试生成速率"""
        rate = self.model.production_rate(
            supercooling=0.03,
            turbulence_intensity=0.5
        )
        assert rate > 0

        # 无过冷无生成
        rate_zero = self.model.production_rate(
            supercooling=0.0,
            turbulence_intensity=0.5
        )
        assert rate_zero == 0


class TestIceCoverFormation:
    """冰盖形成测试"""

    def setup_method(self):
        self.model = IceCoverFormation()

    def test_stable_cover_check(self):
        """测试稳定冰盖条件"""
        # 流速低，可形成
        stable = self.model.check_stable_cover(velocity=0.4, depth=3.0)
        assert stable is True

        # 流速高，不可形成
        stable = self.model.check_stable_cover(velocity=0.8, depth=3.0)
        assert stable is False

    def test_border_ice_growth(self):
        """测试岸冰生长"""
        new_width = self.model.border_ice_growth(
            current_width=2.0,
            channel_width=20.0,
            afdd_rate=10.0,
            dt_days=1.0
        )
        assert new_width > 2.0

    def test_cover_fraction(self):
        """测试覆盖率计算"""
        fraction = self.model.cover_fraction(
            border_width=3.0,
            channel_width=10.0,
            cover_position=500,
            channel_length=1000
        )
        assert 0 <= fraction <= 1


class TestBreakupModel:
    """开河模型测试"""

    def setup_method(self):
        self.model = BreakupModel()

    def test_strength_decay(self):
        """测试冰盖强度衰减"""
        initial = 1e6  # 初始强度
        strength = self.model.ice_strength_decay(initial, mdd=50)
        assert strength < initial

    def test_thermal_breakup(self):
        """测试热力开河判断"""
        # MDD高，应为热力开河
        result = self.model.check_thermal_breakup(mdd=30, ice_thickness=0.1)
        assert result is True

        # MDD低，不是热力开河
        result = self.model.check_thermal_breakup(mdd=10, ice_thickness=0.5)
        assert result is False

    def test_mechanical_breakup(self):
        """测试机械开河判断"""
        # 流量增加大
        result = self.model.check_mechanical_breakup(
            discharge_ratio=2.0,
            stage_rise_rate=0.3
        )
        assert result is True

    def test_classify_breakup(self):
        """测试开河类型分类"""
        breakup_type = self.model.classify_breakup(
            mdd=30,
            ice_thickness=0.1,
            discharge_ratio=1.2,
            stage_rise_rate=0.2
        )
        assert breakup_type == BreakupType.THERMAL


class TestIcePeriodSimulator:
    """综合仿真器测试"""

    def setup_method(self):
        self.simulator = IcePeriodSimulator()

    def test_initialization(self):
        """测试初始化"""
        self.simulator.initialize(initial_water_temp=4.0)
        assert self.simulator.state.phase == IcePhase.OPEN_WATER
        assert self.simulator.state.water_temperature == 4.0
        assert self.simulator.state.ice_thickness == 0.0

    def test_step_simulation(self):
        """测试单步仿真"""
        self.simulator.initialize()

        geometry = ChannelGeometry(
            width=10.0,
            depth=3.0,
            area=30.0,
            wetted_perimeter=16.0,
            hydraulic_radius=1.875,
            slope=0.0005,
            bed_roughness=0.025
        )

        meteo = MeteoCondition(
            air_temperature=-15.0,
            wind_speed=3.0
        )

        state = self.simulator.step(geometry, meteo, velocity=0.5, dt=3600)

        assert state is not None
        assert isinstance(state, IceState)

    def test_freeze_up_simulation(self):
        """测试封冻过程仿真"""
        self.simulator.initialize(initial_water_temp=2.0)

        geometry = ChannelGeometry(
            width=10.0,
            depth=3.0,
            area=30.0,
            wetted_perimeter=16.0,
            hydraulic_radius=1.875,
            slope=0.0005,
            bed_roughness=0.025
        )

        # 连续模拟10天寒冷天气
        # 使用较低流速 (0.3 m/s) 以满足冰盖稳定条件
        # Froude = 0.3 / sqrt(9.81*3) ≈ 0.055 < 0.08 (临界值)
        for day in range(10):
            for hour in range(24):
                meteo = MeteoCondition(
                    air_temperature=-20.0,
                    wind_speed=2.0
                )
                state = self.simulator.step(geometry, meteo, velocity=0.3, dt=3600)

        # 应该形成冰盖
        assert state.afdd > 0
        if state.afdd > StefanParams.AFDD_ICE_START:
            assert state.ice_thickness > 0


class TestIcePeriodController:
    """冰期控制器测试"""

    def setup_method(self):
        self.controller = IcePeriodController()

    def test_detect_mode_normal(self):
        """测试正常期模式检测"""
        ice_state = IceState()
        mode = self.controller.detect_operation_mode(
            ice_state, air_temp=20, current_month=7
        )
        assert mode == IceOperationMode.NORMAL

    def test_detect_mode_pre_ice(self):
        """测试预冰期模式检测"""
        ice_state = IceState(phase=IcePhase.OPEN_WATER)
        mode = self.controller.detect_operation_mode(
            ice_state, air_temp=0, current_month=11
        )
        assert mode == IceOperationMode.PRE_ICE

    def test_detect_mode_stable_ice(self):
        """测试稳定冰期模式检测"""
        ice_state = IceState(phase=IcePhase.FROZEN, ice_thickness=0.5)
        mode = self.controller.detect_operation_mode(
            ice_state, air_temp=-15, current_month=1
        )
        assert mode == IceOperationMode.STABLE_ICE

    def test_get_constraints(self):
        """测试获取约束"""
        ice_state = IceState(
            phase=IcePhase.FROZEN,
            composite_roughness=0.022,
            conveyance_factor=0.8
        )

        constraints = self.controller.get_constraints(
            IceOperationMode.STABLE_ICE, ice_state
        )

        assert constraints.max_flow_change_rate < 10  # 冰期限制变化率
        assert constraints.capacity_factor < 1.0

    def test_generate_decision(self):
        """测试生成决策"""
        ice_state = IceState(
            phase=IcePhase.FROZEN,
            ice_thickness=0.5,
            composite_roughness=0.022,
            conveyance_factor=0.8
        )

        decision = self.controller.generate_decision(
            ice_state=ice_state,
            air_temp=-15,
            current_month=1,
            current_flow=10.0,
            flow_setpoint_request=12.0
        )

        assert isinstance(decision, IceControlDecision)
        assert decision.mode == IceOperationMode.STABLE_ICE
        assert decision.flow_setpoint <= decision.constraints.max_flow_rate


class TestIceFlowRateLimiter:
    """流量变化率限制器测试"""

    def setup_method(self):
        self.limiter = IceFlowRateLimiter(max_rate_percent_per_hour=5.0)

    def test_limit_within_range(self):
        """测试范围内变化"""
        result = self.limiter.limit(
            requested_flow=10.2,
            current_flow=10.0,
            dt_seconds=3600  # 1小时
        )
        # 2%变化在5%限制内
        assert result == 10.2

    def test_limit_exceeds_range(self):
        """测试超范围变化"""
        result = self.limiter.limit(
            requested_flow=12.0,
            current_flow=10.0,
            dt_seconds=3600
        )
        # 20%变化超过5%限制
        assert result < 12.0
        assert result > 10.0

    def test_limit_decrease(self):
        """测试下降限制"""
        result = self.limiter.limit(
            requested_flow=8.0,
            current_flow=10.0,
            dt_seconds=3600
        )
        # 20%下降超过限制
        assert result > 8.0
        assert result < 10.0


class TestIceMonitor:
    """冰期监测器测试"""

    def setup_method(self):
        self.monitor = IceMonitor()

    def test_record(self):
        """测试记录"""
        ice_state = IceState(ice_thickness=0.3)
        decision = IceControlDecision(
            mode=IceOperationMode.STABLE_ICE,
            alarm_level=IceAlarmLevel.NORMAL,
            constraints=IceOperationConstraints(
                min_flow_rate=5, max_flow_rate=15,
                max_flow_change_rate=5, min_water_level=360,
                max_water_level=375, min_velocity=0.3,
                max_velocity=2.0, roughness_factor=1.2,
                capacity_factor=0.8
            ),
            flow_setpoint=10.0,
            flow_ramp_rate=5.0,
            valve_constraints={},
            warnings=[],
            recommendations=[]
        )

        self.monitor.record(ice_state, decision, actual_flow=10.0, air_temp=-15)

        assert len(self.monitor.history) == 1
        assert self.monitor.history[0]['ice_thickness'] == 0.3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
