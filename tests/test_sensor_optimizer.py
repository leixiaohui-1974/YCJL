"""
传感器优化模块单元测试
======================

测试传感器点位与参数优化功能
"""

import pytest
import numpy as np
from datetime import datetime

from ycjl.core.sensor_optimizer import (
    # 枚举
    SensorType,
    MeasurementPriority,
    ComponentType,
    OptimizationObjective,
    PlacementStrategy,

    # 数据类
    SensorSpec,
    MeasurementPoint,
    SensorPlacement,
    OptimizationConstraint,
    OptimizationSolution,

    # 分析器
    SensorCatalog,
    ObservabilityAnalyzer,
    CostBenefitAnalyzer,
    RobustnessAnalyzer,

    # 优化器
    WaterProjectSensorOptimizer,
    YCJLSensorOptimizer,
    create_sensor_optimizer
)


class TestSensorType:
    """传感器类型枚举测试"""

    def test_sensor_type_properties(self):
        """测试传感器类型属性"""
        assert SensorType.LEVEL.display_name == "水位"
        assert SensorType.LEVEL.unit == "m"
        assert SensorType.PRESSURE.display_name == "压力"
        assert SensorType.FLOW.display_name == "流量"

    def test_all_types_have_properties(self):
        """测试所有类型都有正确属性"""
        for st in SensorType:
            assert st.display_name != ""
            assert st.unit != ""


class TestSensorSpec:
    """传感器规格测试"""

    def test_create_sensor_spec(self):
        """测试创建传感器规格"""
        spec = SensorSpec(
            sensor_type=SensorType.LEVEL,
            name="TEST-001",
            range_min=0,
            range_max=50,
            accuracy=0.01,
            purchase_cost=5000,
            installation_cost=1000,
            maintenance_cost_annual=500,
            lifespan_years=10
        )

        assert spec.sensor_type == SensorType.LEVEL
        assert spec.range_max == 50
        assert spec.accuracy == 0.01

    def test_lifecycle_cost(self):
        """测试生命周期成本计算"""
        spec = SensorSpec(
            sensor_type=SensorType.LEVEL,
            name="TEST-001",
            purchase_cost=5000,
            installation_cost=1000,
            maintenance_cost_annual=500,
            lifespan_years=10
        )

        expected_cost = 5000 + 1000 + 500 * 10
        assert spec.total_lifecycle_cost == expected_cost
        assert spec.annual_cost == expected_cost / 10

    def test_range_suitability(self):
        """测试量程适用性检查"""
        spec = SensorSpec(
            sensor_type=SensorType.LEVEL,
            name="TEST-001",
            range_min=0,
            range_max=50
        )

        assert spec.is_suitable_for_range(5, 40) == True
        assert spec.is_suitable_for_range(0, 50) == True
        assert spec.is_suitable_for_range(-5, 40) == False
        assert spec.is_suitable_for_range(10, 60) == False

    def test_to_dict(self):
        """测试转换为字典"""
        spec = SensorSpec(
            sensor_type=SensorType.PRESSURE,
            name="PT-100"
        )

        d = spec.to_dict()
        assert d["type"] == "PRESSURE"
        assert d["name"] == "PT-100"


class TestMeasurementPoint:
    """测量点位测试"""

    def test_create_measurement_point(self):
        """测试创建测量点位"""
        point = MeasurementPoint(
            point_id="MP001",
            name="测试点位",
            component_type=ComponentType.PIPELINE,
            component_id="PIPE1",
            chainage=1000.0,
            required_measurements=[SensorType.PRESSURE, SensorType.FLOW],
            priority=MeasurementPriority.HIGH
        )

        assert point.point_id == "MP001"
        assert point.chainage == 1000.0
        assert len(point.required_measurements) == 2

    def test_priority_weight(self):
        """测试优先级权重"""
        point_critical = MeasurementPoint(
            point_id="MP001",
            name="关键点",
            component_type=ComponentType.RESERVOIR,
            component_id="RES1",
            priority=MeasurementPriority.CRITICAL
        )

        point_low = MeasurementPoint(
            point_id="MP002",
            name="低优先级",
            component_type=ComponentType.PIPELINE,
            component_id="PIPE1",
            priority=MeasurementPriority.LOW
        )

        assert point_critical.get_priority_weight() == 1.0
        assert point_low.get_priority_weight() == 0.4

    def test_safety_critical_flag(self):
        """测试安全关键标志"""
        point = MeasurementPoint(
            point_id="MP001",
            name="安全关键点",
            component_type=ComponentType.SURGE_TANK,
            component_id="ST1",
            is_safety_critical=True
        )

        assert point.is_safety_critical == True


class TestSensorPlacement:
    """传感器布置测试"""

    @pytest.fixture
    def sample_placement(self):
        """创建示例布置"""
        point = MeasurementPoint(
            point_id="MP001",
            name="测试点",
            component_type=ComponentType.PIPELINE,
            component_id="PIPE1"
        )

        spec = SensorSpec(
            sensor_type=SensorType.PRESSURE,
            name="PT-100",
            mtbf=50000,
            purchase_cost=4000,
            installation_cost=1000,
            maintenance_cost_annual=400,
            lifespan_years=10
        )

        return SensorPlacement(
            placement_id="P001",
            point=point,
            sensor_spec=spec,
            redundancy_count=2,
            redundancy_type="hot"
        )

    def test_total_cost(self, sample_placement):
        """测试总成本计算"""
        # 单个传感器总成本 = 4000 + 1000 + 400*10 = 9000
        # 冗余2个 = 18000
        assert sample_placement.total_cost == 18000

    def test_effective_mtbf(self, sample_placement):
        """测试有效MTBF（考虑冗余）"""
        # 热备冗余: MTBF * (n+1)/2 = 50000 * 3/2 = 75000
        assert sample_placement.effective_mtbf == 75000

    def test_overall_score(self, sample_placement):
        """测试综合得分计算"""
        score = sample_placement.overall_score
        assert 0 <= score <= 1


class TestOptimizationConstraint:
    """优化约束测试"""

    def test_constraint_satisfaction(self):
        """测试约束满足检查"""
        constraint = OptimizationConstraint(
            name="test",
            max_total_cost=100000,
            min_coverage=0.8,
            min_sensors=5,
            max_sensors=20
        )

        # 创建满足约束的解
        solution = OptimizationSolution(
            solution_id="S001",
            name="测试方案",
            total_cost=80000,
            coverage_rate=0.85,
            sensor_count=10,
            observability_score=0.9
        )

        satisfied, violations = constraint.is_satisfied(solution)
        assert satisfied == True
        assert len(violations) == 0

    def test_constraint_violation(self):
        """测试约束违反检测"""
        constraint = OptimizationConstraint(
            name="test",
            max_total_cost=50000,
            min_coverage=0.9
        )

        # 创建违反约束的解
        solution = OptimizationSolution(
            solution_id="S001",
            name="测试方案",
            total_cost=80000,
            coverage_rate=0.75,
            sensor_count=10,
            observability_score=0.8
        )

        satisfied, violations = constraint.is_satisfied(solution)
        assert satisfied == False
        assert len(violations) == 2  # 成本和覆盖率都违反


class TestSensorCatalog:
    """传感器目录测试"""

    def test_default_catalog(self):
        """测试默认目录初始化"""
        catalog = SensorCatalog()

        # 检查是否有预置传感器
        all_sensors = catalog.list_all()
        assert len(all_sensors) > 0

    def test_get_by_type(self):
        """测试按类型获取"""
        catalog = SensorCatalog()

        level_sensors = catalog.get_by_type(SensorType.LEVEL)
        pressure_sensors = catalog.get_by_type(SensorType.PRESSURE)

        assert len(level_sensors) > 0
        assert len(pressure_sensors) > 0
        assert all(s.sensor_type == SensorType.LEVEL for s in level_sensors)

    def test_find_suitable(self):
        """测试查找适合的传感器"""
        catalog = SensorCatalog()

        suitable = catalog.find_suitable(
            SensorType.LEVEL,
            value_range=(0, 15)
        )

        assert len(suitable) > 0
        assert all(s.range_max >= 15 for s in suitable)

    def test_add_custom_sensor(self):
        """测试添加自定义传感器"""
        catalog = SensorCatalog()

        custom = SensorSpec(
            sensor_type=SensorType.LEVEL,
            name="CUSTOM-001",
            range_min=0,
            range_max=30
        )

        catalog.add_sensor(custom)

        retrieved = catalog.get_sensor("CUSTOM-001")
        assert retrieved is not None
        assert retrieved.name == "CUSTOM-001"


class TestObservabilityAnalyzer:
    """可观测性分析器测试"""

    @pytest.fixture
    def analyzer(self):
        return ObservabilityAnalyzer()

    @pytest.fixture
    def sample_points_and_placements(self):
        """创建示例点位和布置"""
        points = [
            MeasurementPoint(
                point_id="MP001",
                name="点1",
                component_type=ComponentType.PIPELINE,
                component_id="P1",
                required_measurements=[SensorType.PRESSURE, SensorType.FLOW],
                is_safety_critical=True
            ),
            MeasurementPoint(
                point_id="MP002",
                name="点2",
                component_type=ComponentType.PIPELINE,
                component_id="P1",
                required_measurements=[SensorType.PRESSURE]
            )
        ]

        spec = SensorSpec(
            sensor_type=SensorType.PRESSURE,
            name="PT-100"
        )

        placements = [
            SensorPlacement(
                placement_id="PL001",
                point=points[0],
                sensor_spec=spec
            )
        ]

        return points, placements

    def test_coverage_analysis(self, analyzer, sample_points_and_placements):
        """测试覆盖率分析"""
        points, placements = sample_points_and_placements

        result = analyzer.analyze_coverage(points, placements)

        assert "overall_coverage" in result
        assert "critical_coverage" in result
        assert 0 <= result["overall_coverage"] <= 1

    def test_information_gain(self, analyzer):
        """测试信息增益计算"""
        point1 = MeasurementPoint(
            point_id="MP001",
            name="点1",
            component_type=ComponentType.PIPELINE,
            component_id="P1",
            chainage=0
        )

        point2 = MeasurementPoint(
            point_id="MP002",
            name="点2",
            component_type=ComponentType.PIPELINE,
            component_id="P1",
            chainage=200  # 距离200m
        )

        spec = SensorSpec(
            sensor_type=SensorType.PRESSURE,
            name="PT-100"
        )

        existing = [SensorPlacement(
            placement_id="PL001",
            point=point1,
            sensor_spec=spec
        )]

        new_placement = SensorPlacement(
            placement_id="PL002",
            point=point2,
            sensor_spec=spec
        )

        gain = analyzer.calculate_information_gain(existing, new_placement)
        assert gain > 0


class TestCostBenefitAnalyzer:
    """成本效益分析器测试"""

    @pytest.fixture
    def analyzer(self):
        return CostBenefitAnalyzer()

    @pytest.fixture
    def sample_solution(self):
        """创建示例优化方案"""
        point = MeasurementPoint(
            point_id="MP001",
            name="点1",
            component_type=ComponentType.PIPELINE,
            component_id="P1"
        )

        spec = SensorSpec(
            sensor_type=SensorType.PRESSURE,
            name="PT-100",
            purchase_cost=4000,
            installation_cost=1000,
            maintenance_cost_annual=400,
            power_consumption=5
        )

        placements = [
            SensorPlacement(
                placement_id=f"PL{i:03d}",
                point=point,
                sensor_spec=spec
            )
            for i in range(5)
        ]

        solution = OptimizationSolution(
            solution_id="S001",
            name="测试方案",
            placements=placements,
            coverage_rate=0.9,
            observability_score=0.85,
            redundancy_score=0.7
        )
        solution.update_statistics()

        return solution

    def test_roi_calculation(self, analyzer, sample_solution):
        """测试ROI计算"""
        roi = analyzer.calculate_roi(sample_solution)

        assert "initial_investment" in roi
        assert "net_present_value" in roi
        assert "internal_rate_of_return" in roi
        assert "payback_period" in roi

        assert roi["initial_investment"] > 0

    def test_compare_solutions(self, analyzer, sample_solution):
        """测试方案比较"""
        # 创建第二个方案
        solution2 = OptimizationSolution(
            solution_id="S002",
            name="方案2",
            placements=sample_solution.placements[:3],
            coverage_rate=0.75,
            observability_score=0.7,
            redundancy_score=0.5
        )
        solution2.update_statistics()

        result = analyzer.compare_solutions([sample_solution, solution2])

        assert "comparisons" in result
        assert len(result["comparisons"]) == 2


class TestRobustnessAnalyzer:
    """鲁棒性分析器测试"""

    @pytest.fixture
    def analyzer(self):
        return RobustnessAnalyzer()

    @pytest.fixture
    def sample_placements(self):
        """创建示例布置"""
        placements = []

        for i in range(3):
            point = MeasurementPoint(
                point_id=f"MP{i:03d}",
                name=f"点{i+1}",
                component_type=ComponentType.PIPELINE,
                component_id="P1",
                is_safety_critical=(i == 0)
            )

            spec = SensorSpec(
                sensor_type=SensorType.PRESSURE,
                name="PT-100",
                mtbf=50000
            )

            placements.append(SensorPlacement(
                placement_id=f"PL{i:03d}",
                point=point,
                sensor_spec=spec,
                redundancy_count=2 if i == 0 else 1,
                redundancy_type="hot" if i == 0 else "none"
            ))

        return placements

    def test_redundancy_analysis(self, analyzer, sample_placements):
        """测试冗余分析"""
        result = analyzer.analyze_redundancy(sample_placements)

        assert "average_redundancy" in result
        assert "single_point_failures" in result
        assert result["average_redundancy"] > 1  # 因为有一个点是2冗余

    def test_availability_analysis(self, analyzer, sample_placements):
        """测试可用性分析"""
        result = analyzer.analyze_availability(sample_placements)

        assert "system_availability" in result
        assert "critical_availability" in result
        assert 0 < result["system_availability"] <= 1

    def test_failure_simulation(self, analyzer, sample_placements):
        """测试故障模拟"""
        result = analyzer.simulate_failures(sample_placements, n_simulations=100)

        assert "system_failure_rate" in result
        assert "robustness_score" in result
        assert 0 <= result["system_failure_rate"] <= 1


class TestWaterProjectOptimizer:
    """通用水利工程优化器测试"""

    @pytest.fixture
    def optimizer(self):
        """创建优化器"""
        params = {
            "components": [
                {
                    "type": "RESERVOIR",
                    "id": "RES1",
                    "name": "上游水库",
                    "dead_level": 100,
                    "normal_level": 150
                },
                {
                    "type": "PIPELINE",
                    "id": "PIPE1",
                    "name": "输水管道",
                    "length": 10000,
                    "start_chainage": 0,
                    "design_pressure": 80,
                    "design_flow": 20
                },
                {
                    "type": "PUMP_STATION",
                    "id": "PS1",
                    "name": "加压泵站"
                }
            ]
        }

        return WaterProjectSensorOptimizer("测试工程", params)

    def test_initialization(self, optimizer):
        """测试初始化"""
        assert optimizer.project_name == "测试工程"
        assert len(optimizer._measurement_points) > 0

    def test_optimize(self, optimizer):
        """测试优化执行"""
        optimizer.set_constraints(OptimizationConstraint(
            name="test",
            max_total_cost=1000000
        ))

        solution = optimizer.optimize()

        assert solution is not None
        assert solution.sensor_count > 0
        assert solution.coverage_rate > 0

    def test_different_objectives(self, optimizer):
        """测试不同优化目标"""
        objectives = [
            OptimizationObjective.COVERAGE,
            OptimizationObjective.COST,
            OptimizationObjective.BALANCED
        ]

        for obj in objectives:
            optimizer.set_objective(obj)
            solution = optimizer.optimize()
            assert solution is not None

    def test_generate_report(self, optimizer):
        """测试报告生成"""
        solution = optimizer.optimize()
        report = optimizer.generate_report(solution)

        assert "优化方案" in report
        assert "覆盖率" in report
        assert "成本" in report


class TestYCJLOptimizer:
    """YCJL专用优化器测试"""

    @pytest.fixture
    def optimizer(self):
        return YCJLSensorOptimizer()

    def test_initialization(self, optimizer):
        """测试初始化"""
        assert optimizer.project_name == "引绰济辽输水工程"
        assert optimizer.ice_period_enabled == True
        assert len(optimizer._measurement_points) > 0

    def test_ice_sensors_added(self, optimizer):
        """测试冰期传感器添加"""
        ice_sensors = optimizer.sensor_catalog.get_by_type(SensorType.ICE_THICKNESS)
        assert len(ice_sensors) > 0

    def test_measurement_points_structure(self, optimizer):
        """测试测量点位结构"""
        points = optimizer._measurement_points

        # 检查是否包含关键点位
        point_ids = [p.point_id for p in points]

        assert any("WDG" in pid for pid in point_ids)  # 文得根水库
        assert any("TUN" in pid for pid in point_ids)  # 隧洞
        assert any("PWR" in pid for pid in point_ids)  # 电站
        assert any("SRG" in pid for pid in point_ids)  # 调压井

    def test_ice_monitoring_points(self, optimizer):
        """测试冰期监测点"""
        points = optimizer._measurement_points

        ice_points = [p for p in points if "ICE" in p.point_id]
        assert len(ice_points) > 0

        for p in ice_points:
            assert SensorType.ICE_THICKNESS in p.required_measurements
            assert p.has_ice_risk == True

    def test_optimize_normal(self, optimizer):
        """测试常规优化"""
        solution = optimizer.optimize_for_normal_operation()

        assert solution is not None
        assert "常规" in solution.name
        assert solution.sensor_count > 0

    def test_optimize_ice_period(self, optimizer):
        """测试冰期优化"""
        solution = optimizer.optimize_for_ice_period()

        assert solution is not None
        assert "冰期" in solution.name

    def test_safety_critical_points(self, optimizer):
        """测试安全关键点标识"""
        points = optimizer._measurement_points

        critical_points = [p for p in points if p.is_safety_critical]
        assert len(critical_points) > 0

        # 水库水位应该是安全关键的
        reservoir_points = [p for p in critical_points
                          if p.component_type == ComponentType.RESERVOIR]
        assert len(reservoir_points) > 0


class TestFactoryFunction:
    """工厂函数测试"""

    def test_create_ycjl_optimizer(self):
        """测试创建YCJL优化器"""
        optimizer = create_sensor_optimizer("ycjl", "引绰济辽")

        assert isinstance(optimizer, YCJLSensorOptimizer)

    def test_create_generic_optimizer(self):
        """测试创建通用优化器"""
        params = {
            "components": [
                {"type": "RESERVOIR", "id": "R1", "name": "水库"}
            ]
        }

        optimizer = create_sensor_optimizer("generic", "测试工程", params)

        assert isinstance(optimizer, WaterProjectSensorOptimizer)

    def test_invalid_type(self):
        """测试无效类型"""
        with pytest.raises(ValueError):
            create_sensor_optimizer("invalid_type", "测试")

    def test_missing_params(self):
        """测试缺少参数"""
        with pytest.raises(ValueError):
            create_sensor_optimizer("generic", "测试")


class TestOptimizationSolution:
    """优化方案测试"""

    def test_update_statistics(self):
        """测试统计信息更新"""
        point = MeasurementPoint(
            point_id="MP001",
            name="点1",
            component_type=ComponentType.PIPELINE,
            component_id="P1"
        )

        spec1 = SensorSpec(
            sensor_type=SensorType.PRESSURE,
            name="PT-100",
            purchase_cost=4000,
            installation_cost=1000,
            maintenance_cost_annual=400,
            lifespan_years=10
        )

        spec2 = SensorSpec(
            sensor_type=SensorType.FLOW,
            name="FM-100",
            purchase_cost=20000,
            installation_cost=5000,
            maintenance_cost_annual=2000,
            lifespan_years=10
        )

        placements = [
            SensorPlacement(
                placement_id="PL001",
                point=point,
                sensor_spec=spec1,
                redundancy_count=2
            ),
            SensorPlacement(
                placement_id="PL002",
                point=point,
                sensor_spec=spec2,
                redundancy_count=1
            )
        ]

        solution = OptimizationSolution(
            solution_id="S001",
            name="测试方案",
            placements=placements
        )

        solution.update_statistics()

        assert solution.sensor_count == 3  # 2 + 1
        assert solution.sensor_by_type[SensorType.PRESSURE] == 2
        assert solution.sensor_by_type[SensorType.FLOW] == 1

    def test_summary(self):
        """测试摘要生成"""
        solution = OptimizationSolution(
            solution_id="S001",
            name="测试方案",
            coverage_rate=0.9,
            observability_score=0.85,
            redundancy_score=0.7,
            robustness_score=0.8,
            total_cost=100000,
            annual_cost=15000,
            sensor_count=10,
            covered_points=8,
            total_points=10
        )

        summary = solution.summary()

        assert "测试方案" in summary
        assert "90.0%" in summary  # 覆盖率
        assert "100,000" in summary  # 成本

    def test_to_dict(self):
        """测试转换为字典"""
        solution = OptimizationSolution(
            solution_id="S001",
            name="测试方案",
            coverage_rate=0.9
        )

        d = solution.to_dict()

        assert d["id"] == "S001"
        assert d["coverage_rate"] == 0.9


class TestIntegration:
    """集成测试"""

    def test_full_optimization_workflow(self):
        """测试完整优化流程"""
        # 1. 创建优化器
        optimizer = YCJLSensorOptimizer()

        # 2. 设置约束
        constraints = OptimizationConstraint(
            name="standard",
            max_total_cost=3000000,
            min_coverage=0.8,
            min_observability=0.7
        )
        optimizer.set_constraints(constraints)

        # 3. 设置目标
        optimizer.set_objective(OptimizationObjective.BALANCED)

        # 4. 执行优化
        solution = optimizer.optimize()

        # 5. 验证结果
        assert solution.sensor_count > 0
        assert solution.coverage_rate >= 0.5  # 合理范围
        assert solution.total_cost > 0

        # 6. 生成报告
        report = optimizer.generate_report(solution)
        assert len(report) > 0

    def test_comparison_of_strategies(self):
        """测试不同策略比较"""
        optimizer = YCJLSensorOptimizer()

        # 成本优先
        optimizer.set_objective(OptimizationObjective.COST)
        solution_cost = optimizer.optimize()

        # 覆盖率优先
        optimizer.set_objective(OptimizationObjective.COVERAGE)
        solution_coverage = optimizer.optimize()

        # 验证不同策略产生不同结果
        # 成本优先应该更便宜
        assert solution_cost.objective_value != solution_coverage.objective_value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
