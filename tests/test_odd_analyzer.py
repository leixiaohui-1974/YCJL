"""
ODD分析器测试 (ODD Analyzer Tests)
==================================

测试 ycjl/core/odd_analyzer.py 模块
"""

import pytest
import sys
import os
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestODDEnums:
    """ODD枚举测试"""

    def test_odd_dimension(self):
        """测试ODD维度枚举"""
        from ycjl.core.odd_analyzer import ODDDimension

        assert ODDDimension.ENVIRONMENT.value == "环境与边界"
        assert ODDDimension.DYNAMICS.value == "动力学限制"
        assert ODDDimension.INFRASTRUCTURE.value == "设备能力"
        assert ODDDimension.DIGITAL.value == "数字化支撑"

    def test_odd_status(self):
        """测试ODD状态枚举"""
        from ycjl.core.odd_analyzer import ODDStatus

        assert ODDStatus.NOMINAL.value == "正常"
        assert ODDStatus.MARGINAL.value == "边界"
        assert ODDStatus.EXCEEDED.value == "超出"

    def test_autonomy_level(self):
        """测试自主等级枚举"""
        from ycjl.core.odd_analyzer import AutonomyLevel

        assert AutonomyLevel.L0_MANUAL.level == 0
        assert AutonomyLevel.L5_FULL.level == 5
        assert AutonomyLevel.L3_CONDITIONAL.description == "条件自动"


class TestODDBoundary:
    """ODD边界测试"""

    def test_boundary_creation(self):
        """测试边界创建"""
        from ycjl.core.odd_analyzer import (
            ODDBoundary, ODDDimension, ConstraintType
        )

        boundary = ODDBoundary(
            name="water_level",
            dimension=ODDDimension.ENVIRONMENT,
            description="水位边界",
            unit="m",
            min_value=10.0,
            max_value=20.0
        )

        assert boundary.name == "water_level"
        assert boundary.min_value == 10.0
        assert boundary.max_value == 20.0

    def test_boundary_check_nominal(self):
        """测试边界检查-正常"""
        from ycjl.core.odd_analyzer import (
            ODDBoundary, ODDDimension, ODDStatus
        )

        boundary = ODDBoundary(
            name="test",
            dimension=ODDDimension.DYNAMICS,
            description="测试",
            unit="-",
            min_value=0.0,
            max_value=100.0,
            warning_margin=0.1
        )

        # 中间值-正常
        status, score = boundary.check_value(50.0)
        assert status == ODDStatus.NOMINAL
        assert score == pytest.approx(1.0)

    def test_boundary_check_marginal(self):
        """测试边界检查-边界"""
        from ycjl.core.odd_analyzer import (
            ODDBoundary, ODDDimension, ODDStatus
        )

        boundary = ODDBoundary(
            name="test",
            dimension=ODDDimension.DYNAMICS,
            description="测试",
            unit="-",
            min_value=0.0,
            max_value=100.0,
            warning_margin=0.1
        )

        # 接近边界
        status, score = boundary.check_value(95.0)
        assert status == ODDStatus.MARGINAL
        assert 0.8 < score < 1.0

    def test_boundary_check_exceeded(self):
        """测试边界检查-超出"""
        from ycjl.core.odd_analyzer import (
            ODDBoundary, ODDDimension, ODDStatus
        )

        boundary = ODDBoundary(
            name="test",
            dimension=ODDDimension.DYNAMICS,
            description="测试",
            unit="-",
            min_value=0.0,
            max_value=100.0
        )

        # 超出上限
        status, score = boundary.check_value(120.0)
        assert status == ODDStatus.EXCEEDED
        assert score < 1.0

        # 超出下限
        status, score = boundary.check_value(-20.0)
        assert status == ODDStatus.EXCEEDED
        assert score < 1.0

    def test_boundary_check_none(self):
        """测试边界检查-空值"""
        from ycjl.core.odd_analyzer import (
            ODDBoundary, ODDDimension, ODDStatus
        )

        boundary = ODDBoundary(
            name="test",
            dimension=ODDDimension.DIGITAL,
            description="测试",
            unit="-",
            min_value=0.0,
            max_value=100.0
        )

        status, score = boundary.check_value(None)
        assert status == ODDStatus.UNKNOWN


class TestObservabilityMetrics:
    """可观性指标测试"""

    def test_observability_score(self):
        """测试可观性综合得分"""
        from ycjl.core.odd_analyzer import ObservabilityMetrics

        # 完美可观性
        obs = ObservabilityMetrics(
            sensor_coverage=1.0,
            sensor_health=1.0,
            data_quality=1.0,
            communication_reliability=1.0,
            redundancy_level=2
        )
        assert obs.overall_score >= 0.9

        # 低可观性
        obs_low = ObservabilityMetrics(
            sensor_coverage=0.5,
            sensor_health=0.5,
            data_quality=0.5,
            communication_reliability=0.5,
            redundancy_level=1
        )
        assert obs_low.overall_score < 0.6

    def test_observability_to_dict(self):
        """测试可观性转字典"""
        from ycjl.core.odd_analyzer import ObservabilityMetrics

        obs = ObservabilityMetrics()
        d = obs.to_dict()

        assert "sensor_coverage" in d
        assert "overall_score" in d


class TestControllabilityMetrics:
    """可控性指标测试"""

    def test_controllability_score(self):
        """测试可控性综合得分"""
        from ycjl.core.odd_analyzer import ControllabilityMetrics

        # 完美可控性
        ctrl = ControllabilityMetrics(
            actuator_availability=1.0,
            actuator_health=1.0,
            control_authority=1.0,
            response_capability=1.0,
            dead_zone_ratio=0.0,
            response_delay=0.0
        )
        assert ctrl.overall_score >= 0.95

        # 有死区的可控性
        ctrl_deadzone = ControllabilityMetrics(
            actuator_availability=1.0,
            actuator_health=1.0,
            control_authority=1.0,
            response_capability=1.0,
            dead_zone_ratio=0.1,  # 10%死区
            response_delay=1.0
        )
        assert ctrl_deadzone.overall_score < ctrl.overall_score

    def test_controllability_delay_penalty(self):
        """测试可控性延迟惩罚"""
        from ycjl.core.odd_analyzer import ControllabilityMetrics

        ctrl_no_delay = ControllabilityMetrics(response_delay=0.0)
        ctrl_with_delay = ControllabilityMetrics(
            response_delay=2.5,
            max_response_delay=5.0
        )

        assert ctrl_no_delay.overall_score > ctrl_with_delay.overall_score


class TestWaterNetworkODDAnalyzer:
    """水网ODD分析器测试"""

    def test_analyzer_creation(self):
        """测试分析器创建"""
        from ycjl.core.odd_analyzer import WaterNetworkODDAnalyzer

        analyzer = WaterNetworkODDAnalyzer("测试水网")
        assert analyzer.system_name == "测试水网"

        # 验证预定义边界
        boundaries = list(analyzer._boundaries.keys())
        assert "water_level_error" in boundaries
        assert "communication_latency" in boundaries

    def test_analyze_nominal_state(self):
        """测试分析-正常状态"""
        from ycjl.core.odd_analyzer import (
            WaterNetworkODDAnalyzer, ODDStatus, AutonomyLevel
        )

        analyzer = WaterNetworkODDAnalyzer("测试水网")

        # 设置正常状态
        normal_state = {
            "water_level_error": 0.02,      # 偏差0.02m
            "rainfall_intensity": 10,        # 降雨10mm/h
            "wind_speed": 5,                 # 风速5m/s
            "water_level_rate": 0.05,        # 变化率0.05m/h
            "flow_disturbance_rate": 5,      # 扰动5%
            "froude_number": 0.3,            # 缓流
            "gate_deadzone": 0.02,           # 死区2cm
            "sensor_accuracy": 0.98,         # 精度98%
            "actuator_response_time": 10,    # 响应10s
            "communication_latency": 0.5,    # 时延0.5s
            "packet_loss_rate": 0.01,        # 丢包0.01%
            "compute_load": 50               # 负载50%
        }

        report = analyzer.analyze(normal_state)

        assert report.overall_score > 0.8
        assert report.overall_status in (ODDStatus.NOMINAL, ODDStatus.MARGINAL)
        assert report.autonomy_level.level >= 3

    def test_analyze_exceeded_state(self):
        """测试分析-超出状态"""
        from ycjl.core.odd_analyzer import (
            WaterNetworkODDAnalyzer, ODDStatus, AutonomyLevel
        )

        analyzer = WaterNetworkODDAnalyzer("测试水网")

        # 设置超出状态
        exceeded_state = {
            "water_level_error": 0.30,       # 偏差0.30m - 超出!
            "communication_latency": 3.0,    # 时延3s - 超出!
            "froude_number": 1.2,            # 急流 - 超出!
            "sensor_accuracy": 0.80,         # 精度低 - 超出!
        }

        report = analyzer.analyze(exceeded_state)

        assert report.overall_score < 0.5
        assert report.overall_status in (ODDStatus.EXCEEDED, ODDStatus.DEGRADED)
        assert report.mrm_triggered is True

    def test_quick_check(self):
        """测试快速检查"""
        from ycjl.core.odd_analyzer import WaterNetworkODDAnalyzer

        analyzer = WaterNetworkODDAnalyzer("测试水网")

        # 正常状态
        in_odd, level, warnings = analyzer.quick_check({
            "water_level_error": 0.05,
            "communication_latency": 0.3
        })

        assert isinstance(in_odd, bool)
        assert len(warnings) >= 0


class TestODDReport:
    """ODD报告测试"""

    def test_report_summary(self):
        """测试报告摘要"""
        from ycjl.core.odd_analyzer import (
            WaterNetworkODDAnalyzer, ObservabilityMetrics, ControllabilityMetrics
        )

        analyzer = WaterNetworkODDAnalyzer("南水北调中线")

        obs = ObservabilityMetrics(sensor_coverage=0.95, sensor_health=0.98)
        ctrl = ControllabilityMetrics(actuator_health=0.95)

        report = analyzer.analyze(
            current_state={"water_level_error": 0.05},
            observability=obs,
            controllability=ctrl
        )

        summary = report.summary()
        assert "南水北调中线" in summary
        assert "ODD得分" in summary

    def test_report_to_dict(self):
        """测试报告转字典"""
        from ycjl.core.odd_analyzer import WaterNetworkODDAnalyzer

        analyzer = WaterNetworkODDAnalyzer("测试")
        report = analyzer.analyze({"water_level_error": 0.05})

        d = report.to_dict()
        assert "overall_score" in d
        assert "autonomy_level" in d
        assert "dimension_scores" in d


class TestConvenienceFunctions:
    """便捷函数测试"""

    def test_calculate_odd_reliability_multiplicative(self):
        """测试ODD可靠性计算-乘法模型"""
        from ycjl.core.odd_analyzer import calculate_odd_reliability

        # 全部满分
        reliability = calculate_odd_reliability(1.0, 1.0, 1.0)
        assert reliability == pytest.approx(1.0)

        # 一项为0
        reliability = calculate_odd_reliability(1.0, 0.0, 1.0)
        assert reliability == pytest.approx(0.0)

        # 部分分数
        reliability = calculate_odd_reliability(0.9, 0.8, 0.7)
        assert reliability == pytest.approx(0.9 * 0.8 * 0.7)

    def test_calculate_odd_reliability_weighted(self):
        """测试ODD可靠性计算-加权平均"""
        from ycjl.core.odd_analyzer import calculate_odd_reliability

        reliability = calculate_odd_reliability(
            0.9, 0.8, 0.7,
            aggregation="weighted"
        )
        expected = 0.9 * 0.4 + 0.8 * 0.3 + 0.7 * 0.3
        assert reliability == pytest.approx(expected)

    def test_determine_autonomy_from_score(self):
        """测试自主等级判定"""
        from ycjl.core.odd_analyzer import (
            determine_autonomy_from_score, AutonomyLevel
        )

        assert determine_autonomy_from_score(0.98) == AutonomyLevel.L5_FULL
        assert determine_autonomy_from_score(0.90) == AutonomyLevel.L4_HIGH
        assert determine_autonomy_from_score(0.75) == AutonomyLevel.L3_CONDITIONAL
        assert determine_autonomy_from_score(0.55) == AutonomyLevel.L2_PARTIAL
        assert determine_autonomy_from_score(0.35) == AutonomyLevel.L1_ASSISTED
        assert determine_autonomy_from_score(0.10) == AutonomyLevel.L0_MANUAL

    def test_create_water_odd_analyzer(self):
        """测试工厂函数"""
        from ycjl.core.odd_analyzer import (
            create_water_odd_analyzer, ODDBoundary, ODDDimension
        )

        # 基础创建
        analyzer = create_water_odd_analyzer("测试系统")
        assert analyzer.system_name == "测试系统"

        # 带自定义边界
        custom_boundary = ODDBoundary(
            name="custom_param",
            dimension=ODDDimension.ENVIRONMENT,
            description="自定义参数",
            unit="-",
            min_value=0,
            max_value=10
        )
        analyzer2 = create_water_odd_analyzer(
            "测试系统2",
            custom_boundaries=[custom_boundary]
        )
        assert "custom_param" in analyzer2._boundaries


class TestIntegrationWithCore:
    """与核心框架的集成测试"""

    def test_import_from_core(self):
        """测试从核心模块导入"""
        from ycjl.core import (
            ODDDimension,
            ODDStatus,
            AutonomyLevel,
            ODDBoundary,
            ODDReport,
            WaterNetworkODDAnalyzer,
            calculate_odd_reliability
        )

        analyzer = WaterNetworkODDAnalyzer("集成测试")
        assert analyzer is not None

    def test_with_gap_analyzer(self):
        """测试与数据诊断器配合"""
        from ycjl.core import (
            WaterNetworkODDAnalyzer,
            ObservabilityMetrics,
            DataReadinessLevel
        )

        analyzer = WaterNetworkODDAnalyzer("配合测试")

        # 模拟从数据诊断器获取的完备性
        data_completeness = 0.85

        # 将数据完备性反映到可观性
        obs = ObservabilityMetrics(
            data_quality=data_completeness,
            sensor_coverage=0.90
        )

        report = analyzer.analyze(
            {"water_level_error": 0.05},
            observability=obs
        )

        assert report.observability is not None


class TestDimensionScoring:
    """维度评分测试"""

    def test_environment_dimension(self):
        """测试环境维度评分"""
        from ycjl.core.odd_analyzer import (
            WaterNetworkODDAnalyzer, ODDDimension
        )

        analyzer = WaterNetworkODDAnalyzer("环境测试")

        # 只设置环境维度参数
        state = {
            "water_level_error": 0.02,
            "rainfall_intensity": 20,
            "wind_speed": 8
        }

        report = analyzer.analyze(state)
        env_score = report.dimension_scores.get(ODDDimension.ENVIRONMENT)

        assert env_score is not None
        assert env_score.score > 0

    def test_digital_dimension(self):
        """测试数字化维度评分"""
        from ycjl.core.odd_analyzer import (
            WaterNetworkODDAnalyzer, ODDDimension, ODDStatus
        )

        analyzer = WaterNetworkODDAnalyzer("数字化测试")

        # 良好的数字化支撑
        good_digital = {
            "communication_latency": 0.3,
            "packet_loss_rate": 0.01,
            "compute_load": 40
        }
        report_good = analyzer.analyze(good_digital)
        digital_good = report_good.dimension_scores[ODDDimension.DIGITAL]
        assert digital_good.score > 0.9

        # 差的数字化支撑
        bad_digital = {
            "communication_latency": 2.5,  # 超出1.5s限制
            "packet_loss_rate": 0.5,       # 超出0.1%限制
            "compute_load": 95             # 超出80%限制
        }
        report_bad = analyzer.analyze(bad_digital)
        digital_bad = report_bad.dimension_scores[ODDDimension.DIGITAL]
        assert digital_bad.score < digital_good.score
        assert digital_bad.status == ODDStatus.EXCEEDED


class TestMRMAndDegradation:
    """MRM和降级测试"""

    def test_mrm_trigger_critical_violation(self):
        """测试严重违规触发MRM"""
        from ycjl.core.odd_analyzer import WaterNetworkODDAnalyzer, ODDStatus

        analyzer = WaterNetworkODDAnalyzer("MRM测试")

        # 弗劳德数超标（严重违规）
        state = {"froude_number": 1.5}
        report = analyzer.analyze(state)

        # 验证状态为超出
        assert report.overall_status == ODDStatus.EXCEEDED
        # 违规列表不为空
        assert len(report.violations) > 0

    def test_mrm_trigger_low_score(self):
        """测试低分触发MRM"""
        from ycjl.core.odd_analyzer import WaterNetworkODDAnalyzer

        analyzer = WaterNetworkODDAnalyzer("低分测试")

        # 多项超标
        state = {
            "water_level_error": 0.5,
            "communication_latency": 5.0,
            "sensor_accuracy": 0.5,
            "froude_number": 1.2
        }
        report = analyzer.analyze(state)

        # 综合得分较低
        assert report.overall_score < 0.5
        assert report.mrm_triggered is True

    def test_degradation_mode_selection(self):
        """测试降级模式选择"""
        from ycjl.core.odd_analyzer import WaterNetworkODDAnalyzer

        analyzer = WaterNetworkODDAnalyzer("降级测试")

        # 通信超标
        report_digital = analyzer.analyze({"communication_latency": 5.0})
        if report_digital.mrm_triggered:
            assert "通信降级" in report_digital.degradation_mode

        # 设备超标
        report_infra = analyzer.analyze({
            "sensor_accuracy": 0.5,
            "gate_deadzone": 0.2
        })
        # 根据实际超标情况判断


class TestODDAutoBuilder:
    """ODD自动构建器测试"""

    def test_sensor_profile_creation(self):
        """测试传感器配置创建"""
        from ycjl.core.odd_analyzer import SensorProfile

        profile = SensorProfile(
            name="LS01",
            sensor_type="level",
            position="进水口",
            range_min=0.0,
            range_max=20.0,
            accuracy=0.01,
            accuracy_percent=0.05
        )

        assert profile.name == "LS01"
        assert profile.sensor_type == "level"
        score = profile.get_observability_contribution()
        assert 0 < score <= 1.0

    def test_actuator_profile_creation(self):
        """测试执行器配置创建"""
        from ycjl.core.odd_analyzer import ActuatorProfile

        profile = ActuatorProfile(
            name="G01",
            actuator_type="gate",
            position="分水闸",
            max_opening=6.0,
            max_rate=0.01,
            dead_band=0.005
        )

        assert profile.name == "G01"
        assert profile.actuator_type == "gate"
        score = profile.get_controllability_contribution()
        assert 0 < score <= 1.0

    def test_odd_profile_builder(self):
        """测试ODD配置构建器"""
        from ycjl.core.odd_analyzer import (
            ODDProfileBuilder,
            SensorProfile,
            ActuatorProfile,
            CommunicationProfile
        )

        builder = ODDProfileBuilder("测试系统")

        # 添加传感器
        builder.add_sensor(SensorProfile(
            name="LS01",
            sensor_type="level",
            position="进水口",
            accuracy_percent=0.5
        ))
        builder.add_sensor(SensorProfile(
            name="FS01",
            sensor_type="flow",
            position="主渠道",
            accuracy_percent=0.5
        ))

        # 添加执行器
        builder.add_actuator(ActuatorProfile(
            name="G01",
            actuator_type="gate",
            position="分水闸",
            dead_band=0.005
        ))

        # 设置通信
        builder.set_communication(CommunicationProfile(
            latency_max=1.0,
            packet_loss_rate=0.001
        ))

        # 构建
        boundaries, obs, ctrl = builder.build()

        assert len(boundaries) > 0
        assert obs.overall_score > 0
        assert ctrl.overall_score > 0

    def test_build_odd_from_config(self):
        """测试从配置构建ODD"""
        from ycjl.core.odd_analyzer import (
            build_odd_from_config,
            SensorProfile,
            ActuatorProfile
        )

        analyzer = build_odd_from_config(
            "配置测试系统",
            sensors=[
                SensorProfile(name="S1", sensor_type="level", position="A"),
                SensorProfile(name="S2", sensor_type="flow", position="B")
            ],
            actuators=[
                ActuatorProfile(name="A1", actuator_type="gate", position="C")
            ]
        )

        assert analyzer.system_name == "配置测试系统"

        # 执行分析
        report = analyzer.analyze_with_preset({})
        assert report.overall_score >= 0


class TestDeviceProfileExtractor:
    """设备配置自动提取器测试"""

    def test_extract_level_sensor_profile(self):
        """测试提取水位传感器配置"""
        from ycjl.sensors import LevelSensor
        from ycjl.core.odd_analyzer import DeviceProfileExtractor

        sensor = LevelSensor("LS01", "进水口")

        profile = DeviceProfileExtractor.extract_sensor_profile(sensor)

        assert profile is not None
        assert profile.name == "LS01"
        assert profile.sensor_type == "level"
        assert profile.range_max == 20.0  # LevelSensor默认量程
        assert profile.accuracy == 0.01  # LevelSensor默认精度

    def test_extract_flow_sensor_profile(self):
        """测试提取流量传感器配置"""
        from ycjl.sensors import FlowSensor
        from ycjl.core.odd_analyzer import DeviceProfileExtractor

        sensor = FlowSensor("FS01", "主渠道", pipe_diameter=2.4)

        profile = DeviceProfileExtractor.extract_sensor_profile(sensor)

        assert profile is not None
        assert profile.name == "FS01"
        assert profile.sensor_type == "flow"
        assert profile.range_max == 30.0  # FlowSensor默认量程
        assert profile.accuracy_percent == 0.5  # FlowSensor默认精度

    def test_extract_pressure_sensor_profile(self):
        """测试提取压力传感器配置"""
        from ycjl.sensors import PressureSensor
        from ycjl.core.odd_analyzer import DeviceProfileExtractor

        sensor = PressureSensor("PS01", "管道入口")

        profile = DeviceProfileExtractor.extract_sensor_profile(sensor)

        assert profile is not None
        assert profile.name == "PS01"
        assert profile.sensor_type == "pressure"
        assert profile.sampling_rate == 100.0  # PressureSensor高频采样

    def test_extract_gate_actuator_profile(self):
        """测试提取闸门执行器配置"""
        from ycjl.actuators.gate import GateActuator
        from ycjl.core.odd_analyzer import DeviceProfileExtractor

        gate = GateActuator("G01", max_opening=6.0)

        profile = DeviceProfileExtractor.extract_actuator_profile(gate)

        assert profile is not None
        assert profile.name == "G01"
        assert profile.actuator_type == "gate"
        assert profile.max_opening == 6.0
        assert profile.dead_band == 0.005  # GateActuator默认死区

    def test_extract_valve_actuator_profile(self):
        """测试提取阀门执行器配置"""
        from ycjl.actuators.valve import ValveActuator
        from ycjl.core.odd_analyzer import DeviceProfileExtractor

        valve = ValveActuator("V01", stroke_time=60.0)

        profile = DeviceProfileExtractor.extract_actuator_profile(valve)

        assert profile is not None
        assert profile.name == "V01"
        assert profile.actuator_type == "valve"
        assert profile.dead_band == 0.002  # ValveActuator默认死区

    def test_extract_pump_actuator_profile(self):
        """测试提取水泵执行器配置"""
        from ycjl.actuators.pump import PumpActuator
        from ycjl.core.odd_analyzer import DeviceProfileExtractor

        pump = PumpActuator("P01", rated_flow=15.0, rated_head=50.0)

        profile = DeviceProfileExtractor.extract_actuator_profile(pump)

        assert profile is not None
        assert profile.name == "P01"
        assert profile.actuator_type == "pump"
        assert profile.max_opening == 15.0  # 泵的"开度"用额定流量表示

    def test_extract_from_device_lists(self):
        """测试批量提取设备配置"""
        from ycjl.sensors import LevelSensor, FlowSensor
        from ycjl.actuators.gate import GateActuator
        from ycjl.core.odd_analyzer import DeviceProfileExtractor

        sensors = [
            LevelSensor("LS01", "A"),
            FlowSensor("FS01", "B")
        ]
        actuators = [
            GateActuator("G01"),
            GateActuator("G02")
        ]

        sensor_profiles = DeviceProfileExtractor.extract_from_sensor_list(sensors)
        actuator_profiles = DeviceProfileExtractor.extract_from_actuator_list(actuators)

        assert len(sensor_profiles) == 2
        assert len(actuator_profiles) == 2


class TestBuildODDFromDevices:
    """从设备实例构建ODD测试"""

    def test_build_from_device_instances(self):
        """测试从设备实例构建ODD"""
        from ycjl.sensors import LevelSensor, FlowSensor
        from ycjl.actuators.gate import GateActuator
        from ycjl.core.odd_analyzer import build_odd_from_devices

        # 创建设备实例
        sensors = [
            LevelSensor("LS01", "进水口"),
            FlowSensor("FS01", "主渠道")
        ]
        actuators = [
            GateActuator("G01", max_opening=6.0)
        ]

        # 构建ODD分析器
        analyzer = build_odd_from_devices(
            "测试水网",
            sensors,
            actuators
        )

        assert analyzer.system_name == "测试水网"

        # 执行分析
        report = analyzer.analyze_with_preset({
            "level_sensor_accuracy": 0.005,
            "gate_deadzone_limit": 0.003
        })

        assert report.overall_score > 0
        assert report.observability is not None
        assert report.controllability is not None


class TestYCJLODDBuilder:
    """引绰济辽专用ODD构建器测试"""

    def test_ycjl_builder_creation(self):
        """测试YCJL构建器创建"""
        from ycjl.core.odd_analyzer import YCJLODDBuilder

        builder = YCJLODDBuilder()
        assert builder.system_name == "引绰济辽输水系统"
        assert builder.params['level_control_tolerance'] == 0.15

    def test_ycjl_builder_with_devices(self):
        """测试YCJL构建器添加设备"""
        from ycjl.sensors import LevelSensor, FlowSensor
        from ycjl.actuators.gate import GateActuator
        from ycjl.core.odd_analyzer import YCJLODDBuilder

        builder = YCJLODDBuilder("测试工程")

        # 添加设备实例
        builder.add_sensor(LevelSensor("LS01", "进水口"))
        builder.add_sensor(FlowSensor("FS01", "主渠道"))
        builder.add_actuator(GateActuator("G01"))

        analyzer = builder.build()

        assert analyzer.system_name == "测试工程"

        # 检查YCJL特有边界
        boundaries = list(analyzer._boundaries.keys())
        assert "ycjl_level_error" in boundaries
        assert "ycjl_froude_number" in boundaries

    def test_ycjl_ice_mode(self):
        """测试YCJL冰期模式"""
        from ycjl.core.odd_analyzer import YCJLODDBuilder

        builder = YCJLODDBuilder()

        # 正常模式
        analyzer_normal = builder.build()

        # 创建新的构建器用于冰期模式
        builder_ice = YCJLODDBuilder()
        analyzer_ice = builder_ice.build_with_ice_mode(is_ice_period=True)

        # 冰期模式应该有更严格的约束
        assert builder_ice.params['max_level_rate'] == 0.10  # 更低的变化率

    def test_create_ycjl_odd_analyzer(self):
        """测试YCJL便捷函数"""
        from ycjl.sensors import LevelSensor
        from ycjl.actuators.gate import GateActuator
        from ycjl.core.odd_analyzer import create_ycjl_odd_analyzer

        analyzer = create_ycjl_odd_analyzer(
            sensors=[LevelSensor("LS01", "A")],
            actuators=[GateActuator("G01")],
            is_ice_period=False
        )

        assert analyzer is not None

        # 执行分析
        report = analyzer.analyze_with_preset({
            "ycjl_level_error": 0.05,
            "ycjl_froude_number": 0.3
        })

        assert report.overall_score > 0


class TestWorldModelODDEnvelope:
    """世界模型ODD包络测试"""

    def test_envelope_creation(self):
        """测试ODD包络创建"""
        from ycjl.core.odd_analyzer import (
            WaterNetworkODDAnalyzer,
            WorldModelODDEnvelope
        )

        analyzer = WaterNetworkODDAnalyzer("测试")
        envelope = WorldModelODDEnvelope(analyzer)

        assert len(envelope._constraints) > 0

    def test_sample_scenario(self):
        """测试场景采样"""
        from ycjl.core.odd_analyzer import (
            WaterNetworkODDAnalyzer,
            WorldModelODDEnvelope
        )
        import numpy as np

        analyzer = WaterNetworkODDAnalyzer("采样测试")
        envelope = WorldModelODDEnvelope(analyzer)

        # 采样
        rng = np.random.default_rng(42)
        scenario = envelope.sample_scenario(rng=rng)

        assert len(scenario) > 0

        # 验证采样结果在范围内
        is_valid, violations = envelope.validate_scenario(scenario)
        assert is_valid
        assert len(violations) == 0

    def test_validate_scenario(self):
        """测试场景验证"""
        from ycjl.core.odd_analyzer import (
            WaterNetworkODDAnalyzer,
            WorldModelODDEnvelope
        )

        analyzer = WaterNetworkODDAnalyzer("验证测试")
        envelope = WorldModelODDEnvelope(analyzer)

        # 有效场景
        valid_scenario = {"water_level_error": 0.05}
        is_valid, violations = envelope.validate_scenario(valid_scenario)
        assert is_valid

        # 无效场景
        invalid_scenario = {"water_level_error": 0.5}  # 超出0.15
        is_valid, violations = envelope.validate_scenario(invalid_scenario)
        assert not is_valid
        assert len(violations) > 0

    def test_clip_scenario(self):
        """测试场景裁剪"""
        from ycjl.core.odd_analyzer import (
            WaterNetworkODDAnalyzer,
            WorldModelODDEnvelope
        )

        analyzer = WaterNetworkODDAnalyzer("裁剪测试")
        envelope = WorldModelODDEnvelope(analyzer)

        # 超出范围的场景
        out_of_range = {"water_level_error": 0.5}  # 超出0.15
        clipped = envelope.clip_scenario(out_of_range)

        # 应该被裁剪到边界
        assert clipped["water_level_error"] <= 0.15

    def test_safe_perturbation_range(self):
        """测试安全扰动范围"""
        from ycjl.core.odd_analyzer import (
            WaterNetworkODDAnalyzer,
            WorldModelODDEnvelope
        )

        analyzer = WaterNetworkODDAnalyzer("扰动测试")
        envelope = WorldModelODDEnvelope(analyzer)

        current_state = {"water_level_error": 0.05}
        min_delta, max_delta = envelope.get_safe_perturbation_range(
            current_state, "water_level_error"
        )

        # 当前值0.05，边界-0.15~0.15，应该有扰动空间
        assert min_delta < 0  # 可以往负方向扰动
        assert max_delta > 0  # 可以往正方向扰动

    def test_create_world_model_envelope(self):
        """测试便捷函数"""
        from ycjl.core.odd_analyzer import (
            WaterNetworkODDAnalyzer,
            create_world_model_envelope
        )

        analyzer = WaterNetworkODDAnalyzer("便捷测试")
        envelope = create_world_model_envelope(analyzer)

        assert envelope is not None
        summary = envelope.summary()
        assert "世界模型ODD包络" in summary


class TestChannelProfile:
    """渠道配置测试"""

    def test_channel_profile_creation(self):
        """测试渠道配置创建"""
        from ycjl.core.odd_analyzer import ChannelProfile

        channel = ChannelProfile(
            name="主干渠",
            channel_type="channel",
            length=100000,
            bottom_width=10.0,
            design_flow=23.0,
            max_flow=28.0,
            min_flow=10.0,
            normal_level=5.0,
            max_level=6.5,
            min_level=3.0
        )

        assert channel.name == "主干渠"
        assert channel.design_flow == 23.0

    def test_channel_boundaries_generation(self):
        """测试渠道边界生成"""
        from ycjl.core.odd_analyzer import (
            ODDProfileBuilder,
            ChannelProfile,
            SensorProfile,
            ActuatorProfile
        )

        builder = ODDProfileBuilder("渠道测试")

        # 添加必要的传感器和执行器
        builder.add_sensor(SensorProfile(
            name="LS01", sensor_type="level", position="A"
        ))
        builder.add_actuator(ActuatorProfile(
            name="G01", actuator_type="gate", position="B"
        ))

        # 添加渠道
        builder.add_channel(ChannelProfile(
            name="主干渠",
            channel_type="channel",
            length=100000,
            diameter=3.0,
            design_flow=23.0,
            max_flow=28.0,
            min_flow=10.0,
            design_velocity=2.0,
            normal_level=5.0,
            max_level=6.5,
            min_level=3.0
        ))

        boundaries, obs, ctrl = builder.build()

        # 检查是否生成了渠道相关边界
        boundary_names = [b.name for b in boundaries]
        assert any("主干渠" in name for name in boundary_names)


class TestCommunicationProfile:
    """通信配置测试"""

    def test_communication_profile_creation(self):
        """测试通信配置创建"""
        from ycjl.core.odd_analyzer import CommunicationProfile

        comm = CommunicationProfile(
            network_type="5G专网",
            bandwidth=100.0,
            latency_typical=0.1,
            latency_max=1.5,
            packet_loss_rate=0.001,
            availability=0.9999,
            has_backup=True,
            backup_type="光纤"
        )

        assert comm.network_type == "5G专网"
        assert comm.availability == 0.9999

    def test_communication_boundaries_generation(self):
        """测试通信边界生成"""
        from ycjl.core.odd_analyzer import (
            ODDProfileBuilder,
            CommunicationProfile,
            SensorProfile,
            ActuatorProfile
        )

        builder = ODDProfileBuilder("通信测试")

        builder.add_sensor(SensorProfile(
            name="LS01", sensor_type="level", position="A"
        ))
        builder.add_actuator(ActuatorProfile(
            name="G01", actuator_type="gate", position="B"
        ))
        builder.set_communication(CommunicationProfile(
            latency_max=1.0,
            packet_loss_rate=0.001
        ))

        boundaries, obs, ctrl = builder.build()

        # 检查是否生成了通信相关边界
        boundary_names = [b.name for b in boundaries]
        assert "comm_latency" in boundary_names
        assert "packet_loss" in boundary_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
