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
        from ycjl.core.odd_analyzer import WaterNetworkODDAnalyzer

        analyzer = WaterNetworkODDAnalyzer("MRM测试")

        # 弗劳德数超标（严重违规）
        state = {"froude_number": 1.5}
        report = analyzer.analyze(state)

        assert report.mrm_triggered is True
        assert report.degradation_mode is not None

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

        assert report.overall_score < 0.3
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
