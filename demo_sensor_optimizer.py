#!/usr/bin/env python3
"""
传感器点位与参数优化模块演示
============================

演示如何使用传感器优化模块进行：
1. 引绰济辽工程的传感器布置优化
2. 通用水利工程的传感器配置
3. 不同优化目标的比较
4. 成本效益分析
"""

from ycjl.core.sensor_optimizer import (
    # 枚举
    SensorType,
    OptimizationObjective,
    OptimizationConstraint,
    MeasurementPriority,
    ComponentType,

    # 优化器
    YCJLSensorOptimizer,
    WaterProjectSensorOptimizer,
    create_sensor_optimizer,

    # 分析器
    SensorCatalog,
    CostBenefitAnalyzer
)


def demo_ycjl_optimization():
    """演示引绰济辽工程传感器优化"""
    print("=" * 70)
    print("演示1: 引绰济辽工程传感器优化")
    print("=" * 70)

    # 创建YCJL专用优化器
    optimizer = YCJLSensorOptimizer()

    print(f"\n项目: {optimizer.project_name}")
    print(f"测量点位数: {len(optimizer._measurement_points)}")
    print(f"冰期模式: {'启用' if optimizer.ice_period_enabled else '禁用'}")

    # 显示部分测量点
    print("\n关键测量点位:")
    for point in optimizer._measurement_points[:5]:
        measurements = [m.display_name for m in point.required_measurements]
        print(f"  - {point.name}: {', '.join(measurements)}")

    # 常规运行优化
    print("\n\n--- 常规运行优化 ---")
    solution_normal = optimizer.optimize_for_normal_operation()
    print(solution_normal.summary())

    # 冰期运行优化
    print("\n\n--- 冰期运行优化 ---")
    solution_ice = optimizer.optimize_for_ice_period()
    print(solution_ice.summary())

    return solution_normal, solution_ice


def demo_generic_optimization():
    """演示通用水利工程传感器优化"""
    print("\n" + "=" * 70)
    print("演示2: 通用水利工程传感器优化")
    print("=" * 70)

    # 定义工程参数
    project_params = {
        "components": [
            {
                "type": "RESERVOIR",
                "id": "RES-01",
                "name": "上游水库",
                "dead_level": 520,
                "normal_level": 580,
                "total_storage": 5.0
            },
            {
                "type": "PIPELINE",
                "id": "PIPE-01",
                "name": "输水管道1段",
                "length": 15000,
                "start_chainage": 0,
                "design_pressure": 100,
                "design_flow": 25
            },
            {
                "type": "PUMP_STATION",
                "id": "PS-01",
                "name": "中间加压泵站"
            },
            {
                "type": "PIPELINE",
                "id": "PIPE-02",
                "name": "输水管道2段",
                "length": 20000,
                "start_chainage": 15000,
                "design_pressure": 80,
                "design_flow": 25
            },
            {
                "type": "SURGE_TANK",
                "id": "ST-01",
                "name": "调压井"
            },
            {
                "type": "VALVE",
                "id": "VLV-01",
                "name": "出口调节阀"
            }
        ]
    }

    # 创建优化器
    optimizer = create_sensor_optimizer(
        project_type="generic",
        project_name="某长距离调水工程",
        project_params=project_params
    )

    print(f"\n项目: {optimizer.project_name}")
    print(f"自动生成测量点位数: {len(optimizer._measurement_points)}")

    # 设置约束
    constraints = OptimizationConstraint(
        name="预算约束",
        max_total_cost=2000000,  # 200万预算
        min_coverage=0.85,
        min_observability=0.8,
        min_system_availability=0.98
    )
    optimizer.set_constraints(constraints)

    # 执行优化
    solution = optimizer.optimize()
    print(solution.summary())

    # 生成详细报告
    print("\n--- 详细报告 ---")
    report = optimizer.generate_report(solution)
    print(report)

    return solution


def demo_objective_comparison():
    """演示不同优化目标的比较"""
    print("\n" + "=" * 70)
    print("演示3: 不同优化目标比较")
    print("=" * 70)

    objectives = [
        (OptimizationObjective.COST, "成本优先"),
        (OptimizationObjective.COVERAGE, "覆盖率优先"),
        (OptimizationObjective.ROBUSTNESS, "鲁棒性优先"),
        (OptimizationObjective.BALANCED, "多目标平衡")
    ]

    solutions = []

    for obj, name in objectives:
        optimizer = YCJLSensorOptimizer()
        optimizer.set_objective(obj)
        optimizer.set_constraints(OptimizationConstraint(
            name=name,
            max_total_cost=5000000
        ))

        solution = optimizer.optimize()
        solutions.append((name, solution))

        print(f"\n【{name}】")
        print(f"  传感器数: {solution.sensor_count}")
        print(f"  总投资: ¥{solution.total_cost:,.0f}")
        print(f"  覆盖率: {solution.coverage_rate:.1%}")
        print(f"  可观测性: {solution.observability_score:.1%}")
        print(f"  鲁棒性: {solution.robustness_score:.1%}")

    # 成本效益比较
    print("\n\n--- 成本效益比较 ---")
    analyzer = CostBenefitAnalyzer()
    comparison = analyzer.compare_solutions([s for _, s in solutions])

    print("\n方案排名 (按NPV):")
    for i, comp in enumerate(comparison["comparisons"], 1):
        print(f"  {i}. {comp['name']}: NPV=¥{comp['npv']:,.0f}, IRR={comp['irr']:.1%}")

    return solutions


def demo_sensor_catalog():
    """演示传感器目录功能"""
    print("\n" + "=" * 70)
    print("演示4: 传感器产品目录")
    print("=" * 70)

    catalog = SensorCatalog()

    print("\n可用传感器列表:")
    for sensor_type in SensorType:
        sensors = catalog.get_by_type(sensor_type)
        if sensors:
            print(f"\n【{sensor_type.display_name}传感器】")
            for s in sensors:
                print(f"  - {s.name}: 量程{s.range_min}~{s.range_max}{sensor_type.unit}, "
                      f"精度±{s.accuracy}, 成本¥{s.total_lifecycle_cost:.0f}")

    # 查找适合的传感器
    print("\n\n查找适合量程0-15m的水位传感器:")
    suitable = catalog.find_suitable(SensorType.LEVEL, (0, 15))
    for s in suitable:
        print(f"  - {s.name}: 量程{s.range_min}~{s.range_max}m")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print(" 传感器点位与参数优化模块演示")
    print(" YCJL Sensor Optimization Module Demo")
    print("=" * 70)

    # 演示1: YCJL工程优化
    demo_ycjl_optimization()

    # 演示2: 通用工程优化
    demo_generic_optimization()

    # 演示3: 目标比较
    demo_objective_comparison()

    # 演示4: 传感器目录
    demo_sensor_catalog()

    print("\n" + "=" * 70)
    print(" 演示完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
