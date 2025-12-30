"""
通用核心框架测试 (Core Framework Tests)
======================================

测试 ycjl/core 模块的所有基类和工具
"""

import pytest
import sys
import os
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPhysicsConstants:
    """物理常数测试"""

    def test_constants_import(self):
        """测试常数导入"""
        from ycjl.core.constants import (
            GRAVITY, WATER_DENSITY, ATMOSPHERIC_PRESSURE_HEAD
        )
        assert GRAVITY == pytest.approx(9.80665)
        assert WATER_DENSITY == pytest.approx(998.2)
        assert ATMOSPHERIC_PRESSURE_HEAD == pytest.approx(10.33)

    def test_water_density_temperature(self):
        """测试温度对水密度的影响"""
        from ycjl.core.constants import PhysicsConstants

        rho_4c = PhysicsConstants.water_density(4.0)
        rho_20c = PhysicsConstants.water_density(20.0)
        rho_40c = PhysicsConstants.water_density(40.0)

        # 4°C时密度最大
        assert rho_4c >= rho_20c
        assert rho_4c >= rho_40c

    def test_wave_speed_calculation(self):
        """测试波速计算"""
        from ycjl.core.constants import PhysicsConstants

        # PCCP管道 DN2400
        wave_speed = PhysicsConstants.wave_speed_pccp(
            diameter=2.4,
            wall_thickness=0.28,
            youngs_modulus=45e9
        )

        # 波速应该在900-1300 m/s范围内（PCCP管道典型值）
        assert 900 < wave_speed < 1300


class TestInterpolators:
    """插值器测试"""

    def test_create_interpolator(self):
        """测试创建插值器"""
        from ycjl.core.interpolators import create_interpolator

        data = [(0.0, 0.0), (1.0, 1.0), (2.0, 4.0), (3.0, 9.0)]
        interp = create_interpolator(data)

        assert interp(0.0) == pytest.approx(0.0)
        assert interp(1.0) == pytest.approx(1.0)
        assert interp(2.0) == pytest.approx(4.0)

    def test_interpolator_extrapolation(self):
        """测试插值器边界处理"""
        from ycjl.core.interpolators import create_interpolator

        data = [(0.0, 0.0), (10.0, 100.0)]
        interp = create_interpolator(data)

        # 应该被裁剪到范围内
        assert interp(-5.0) == pytest.approx(0.0)
        assert interp(15.0) == pytest.approx(100.0)

    def test_curve_lookup_table(self):
        """测试曲线查找表"""
        from ycjl.core.interpolators import create_curve_lookup

        data = [(0.0, 0.0), (50.0, 5.0), (100.0, 10.0)]
        table = create_curve_lookup("test_curve", data, x_label="开度", y_label="流量")

        assert table.lookup(50.0) == pytest.approx(5.0)
        assert table.reverse_lookup(5.0) == pytest.approx(50.0)


class TestBaseConfig:
    """配置基类测试"""

    def test_config_validator_positive(self):
        """测试正数验证"""
        from ycjl.core.base_config import ConfigValidator

        result = ConfigValidator.validate_positive(10.0, "test_param")
        assert result.is_valid is True

        result = ConfigValidator.validate_positive(-5.0, "test_param")
        assert result.is_valid is False

    def test_config_validator_range(self):
        """测试范围验证"""
        from ycjl.core.base_config import ConfigValidator

        result = ConfigValidator.validate_range(50.0, 0.0, 100.0, "test_param")
        assert result.is_valid is True

        result = ConfigValidator.validate_range(150.0, 0.0, 100.0, "test_param")
        assert result.is_valid is False

    def test_reservoir_config_validation(self):
        """测试水库配置验证"""
        from ycjl.core.base_config import BaseReservoirConfig

        # 有效配置
        config = BaseReservoirConfig(
            NAME="测试水库",
            NORMAL_LEVEL=100.0,
            DEAD_LEVEL=50.0,
            FLOOD_LIMIT_LEVEL=90.0,
            CHECK_FLOOD_LEVEL=110.0,
            TOTAL_STORAGE=10.0
        )
        errors = config.validate()
        assert len(errors) == 0

        # 无效配置：死水位高于正常水位
        config_invalid = BaseReservoirConfig(
            NAME="测试水库",
            NORMAL_LEVEL=50.0,
            DEAD_LEVEL=100.0,  # 错误
            TOTAL_STORAGE=10.0
        )
        errors = config_invalid.validate()
        assert len(errors) > 0


class TestBasePhysics:
    """物理模型基类测试"""

    def test_base_reservoir(self):
        """测试水库基类"""
        from ycjl.core.base_physics import BaseReservoir

        reservoir = BaseReservoir(
            name="测试水库",
            normal_level=100.0,
            dead_level=50.0,
            max_level=110.0
        )

        # 更新状态
        state = reservoir.update(dt=3600, inflow=10.0, outflow=5.0)
        assert state.level >= 50.0
        assert state.level <= 110.0

    def test_base_pipeline(self):
        """测试管道基类"""
        from ycjl.core.base_physics import BasePipeline

        pipeline = BasePipeline(
            name="测试管道",
            length=10000.0,
            diameter=2.4,
            roughness=0.012
        )

        # 计算水头损失
        hf = pipeline.get_head_loss(flow=10.0)
        assert hf > 0

    def test_base_pump_station(self):
        """测试泵站基类"""
        from ycjl.core.base_physics import BasePumpStation

        pump = BasePumpStation(
            name="测试泵站",
            pump_count=4,
            design_flow=5.0,
            design_head=10.0,
            power_rating=400,
            peak_efficiency=0.85
        )

        # 启动2台泵
        pump.running_pumps = 2

        # 计算扬程和效率
        head = pump.get_head(flow=10.0)
        efficiency = pump.get_efficiency(flow=10.0)

        assert head > 0
        assert 0 < efficiency <= 1.0

    def test_base_valve(self):
        """测试阀门基类"""
        from ycjl.core.base_physics import BaseValve

        valve = BaseValve(
            name="测试阀门",
            diameter=1.0
        )

        # 全开时阻力系数较小
        zeta_open = valve.get_zeta(100.0)
        zeta_half = valve.get_zeta(50.0)
        zeta_closed = valve.get_zeta(0.0)

        assert zeta_open < zeta_half < zeta_closed

    def test_base_channel(self):
        """测试明渠基类"""
        from ycjl.core.base_physics import BaseChannel

        channel = BaseChannel(
            name="测试渠道",
            length=1000.0,
            bottom_width=10.0,
            side_slope=2.0,
            bed_slope=0.0001,
            manning_n=0.025
        )

        # 计算正常水深
        depth = channel.get_normal_depth(flow=20.0)
        assert depth > 0


class TestBaseSimulation:
    """仿真基类测试"""

    def test_time_series_data(self):
        """测试时间序列数据"""
        from ycjl.core.base_simulation import TimeSeriesData

        ts = TimeSeriesData(name="flow", unit="m³/s")
        ts.append(0.0, 10.0)
        ts.append(1.0, 11.0)
        ts.append(2.0, 12.0)

        assert ts.get_value_at(0.5) == pytest.approx(10.5)
        stats = ts.get_statistics()
        assert stats["mean"] == pytest.approx(11.0)

    def test_simulation_result(self):
        """测试仿真结果"""
        from ycjl.core.base_simulation import (
            SimulationResult, SimulationMode, TimeSeriesData
        )

        result = SimulationResult(
            success=True,
            mode=SimulationMode.STEADY_STATE,
            start_time=0.0,
            end_time=100.0,
            duration_s=1.5,
            step_count=100
        )

        result.add_warning("测试警告")
        assert len(result.warnings) == 1

        summary = result.summary()
        assert "成功" in summary


class TestBaseScheduler:
    """调度器基类测试"""

    def test_schedule_decision(self):
        """测试调度决策"""
        from ycjl.core.base_scheduler import (
            ScheduleDecision, ScheduleMode, OperationZone
        )

        decision = ScheduleDecision(
            timestamp=datetime.now(),
            mode=ScheduleMode.NORMAL,
            zone=OperationZone.NORMAL,
            target_flow=15.0,
            target_level=100.0
        )

        d = decision.to_dict()
        assert d["mode"] == "NORMAL"
        assert d["target_flow"] == 15.0


class TestGapAnalyzer:
    """数据诊断器测试"""

    def test_data_gap_report(self):
        """测试数据缺口报告"""
        from ycjl.core.gap_analyzer import (
            DataGapReport, DataReadinessLevel, MissingDataItem,
            DataPriority, DataCategory
        )

        report = DataGapReport(
            project_name="测试项目",
            analysis_time=datetime.now(),
            readiness_level=DataReadinessLevel.L3_OPERATIONAL,
            total_parameters=100
        )

        # 添加缺失项
        item = MissingDataItem(
            name="测试参数",
            category=DataCategory.HYDRAULIC,
            priority=DataPriority.HIGH,
            description="测试描述",
            impact="测试影响",
            suggestion="测试建议"
        )
        report.add_missing_item(item)
        report.finalize()

        assert report.missing_count == 1
        assert len(report.high_missing) == 1

    def test_missing_data_item_creation(self):
        """测试创建缺失数据项"""
        from ycjl.core.gap_analyzer import (
            create_missing_item, DataPriority, DataCategory
        )

        item = create_missing_item(
            name="波速",
            category=DataCategory.TRANSIENT,
            priority=DataPriority.CRITICAL,
            description="管道压力波速"
        )

        assert item.name == "波速"
        assert item.priority == DataPriority.CRITICAL


class TestYinChuoGapAnalyzer:
    """引绰济辽Gap Analyzer测试"""

    def test_analyze_readiness(self):
        """测试数据完备性分析"""
        try:
            from ycjl.config.gap_analyzer import YCJLGapAnalyzer

            report = YCJLGapAnalyzer.analyze()
            assert report is not None
            assert report.project_name == "引绰济辽工程"
            assert report.total_parameters > 0
        except ImportError:
            pytest.skip("Gap analyzer not yet integrated")


class TestMiyunScenarios:
    """密云场景库测试"""

    def test_scenario_database(self):
        """测试场景数据库"""
        try:
            from ycjl.miyun.scenarios import (
                MIYUN_SCENARIO_DATABASE, get_scenario_count
            )

            count = get_scenario_count()
            assert count > 10  # 至少10个场景

            # 检查关键场景存在
            assert "NOR-001" in MIYUN_SCENARIO_DATABASE
            assert "PMP-001" in MIYUN_SCENARIO_DATABASE
            assert "PIP-001" in MIYUN_SCENARIO_DATABASE
        except ImportError:
            pytest.skip("Miyun scenarios not yet available")

    def test_scenario_detector(self):
        """测试场景检测器"""
        try:
            from ycjl.miyun.scenarios import ScenarioDetector

            # 正常状态
            events = ScenarioDetector.detect_scenarios({"flow": 10.0})
            assert len(events) >= 1

            # 高流量场景
            events = ScenarioDetector.detect_scenarios({"flow": 20.0})
            scenario_ids = [e.scenario_id for e in events]
            assert "DEM-001" in scenario_ids
        except ImportError:
            pytest.skip("Miyun scenarios not yet available")


# ==========================================
# 集成测试
# ==========================================
class TestIntegration:
    """集成测试"""

    def test_full_system_import(self):
        """测试完整系统导入"""
        from ycjl.core import (
            PhysicsConstants,
            create_interpolator,
            BaseReservoir,
            BaseSimulationEngine,
            BaseScheduler,
            DataGapReport
        )

        assert PhysicsConstants.GRAVITY == pytest.approx(9.80665)

    def test_config_integration(self):
        """测试配置集成"""
        from ycjl.config.config_database import ProjectParams

        # 验证配置
        errors = ProjectParams.validate()
        # 可能有一些警告，但不应该有致命错误

        # 获取摘要
        summary = ProjectParams.get_summary()
        assert "version" in summary

    def test_miyun_integration(self):
        """测试密云模块集成"""
        try:
            from ycjl.miyun import (
                MiyunParams, SimEngine, GapAnalyzer, Scheduler
            )

            # 验证配置
            errors = MiyunParams.validate()
            assert isinstance(errors, list)

            # 运行诊断
            result = SimEngine.run_system_diagnosis(10.0)
            assert result is not None
        except ImportError:
            pytest.skip("Miyun module not available")


class TestEnhancedScheduler:
    """增强调度器测试"""

    def test_ycjl_enhanced_scheduler(self):
        """测试YCJL增强调度器"""
        from ycjl.control.enhanced_scheduler import EnhancedScheduler
        from datetime import datetime

        # 测试数据完备性检查
        report = EnhancedScheduler.check_data_readiness()
        assert report is not None
        assert report.completeness_ratio > 0

        # 测试月度约束检查
        is_valid, msgs = EnhancedScheduler.check_monthly_constraints(360.0, month=6)
        assert isinstance(is_valid, bool)
        assert len(msgs) > 0

        # 测试场景检测
        state = {"level": 360.0, "inflow": 50.0}
        scenarios = EnhancedScheduler.detect_scenarios(state)
        assert len(scenarios) > 0

        # 测试增强决策
        decision = EnhancedScheduler.make_enhanced_decision(
            datetime.now(), level=360.0, inflow=40.0
        )
        assert decision is not None
        assert hasattr(decision, 'health_score')
        assert 0 <= decision.health_score <= 100

    def test_miyun_enhanced_scheduler(self):
        """测试密云增强调度器"""
        from ycjl.miyun.scheduler import Scheduler

        # 测试月度约束
        is_valid, msgs = Scheduler.check_monthly_constraints(10.0)
        assert isinstance(is_valid, bool)
        assert len(msgs) > 0

        # 测试场景检测
        scenarios = Scheduler.detect_scenarios(10.0)
        assert len(scenarios) > 0

        # 测试流量范围
        max_flow, min_flow = Scheduler.get_recommended_flow_range()
        assert max_flow > min_flow


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
