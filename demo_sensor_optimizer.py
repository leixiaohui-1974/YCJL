#!/usr/bin/env python3
"""
传感器点位与参数优化模块演示 (v2.0)
===================================

演示如何使用传感器优化模块进行：
1. 引绰济辽工程的传感器布置优化
2. 通用水利工程的传感器配置
3. 不同优化目标的比较
4. 高级优化算法比较（贪心/遗传/粒子群/多算法集成）
5. 可视化报告生成（TEXT/HTML/Markdown/JSON）
6. 传感器产品目录查询
"""

from ycjl.core.sensor_optimizer import (
    # 枚举
    SensorType,
    OptimizationObjective,
    OptimizationConstraint,
    MeasurementPriority,
    ComponentType,
    OptimizationAlgorithm,
    ReportFormat,

    # 优化器
    YCJLSensorOptimizer,
    WaterProjectSensorOptimizer,
    create_sensor_optimizer,

    # 高级优化算法
    GeneticSensorOptimizer,
    PSOSensorOptimizer,
    MultiAlgorithmOptimizer,
    GeneticAlgorithmConfig,
    PSOConfig,

    # 分析器
    SensorCatalog,
    CostBenefitAnalyzer,

    # 报告生成器
    SensorOptimizationReporter
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


def demo_advanced_algorithms():
    """演示高级优化算法"""
    print("\n" + "=" * 70)
    print("演示4: 高级优化算法比较")
    print("=" * 70)

    # 创建基础优化器
    base_optimizer = YCJLSensorOptimizer()
    constraints = OptimizationConstraint(
        name="算法比较约束",
        max_total_cost=3000000,
        min_coverage=0.9
    )
    base_optimizer.set_constraints(constraints)

    print("\n比较不同优化算法的性能...")

    # 1. 贪心算法（基准）
    print("\n【贪心算法】")
    solution_greedy = base_optimizer.optimize()
    print(f"  传感器数: {solution_greedy.sensor_count}")
    print(f"  总投资: ¥{solution_greedy.total_cost:,.0f}")
    print(f"  覆盖率: {solution_greedy.coverage_rate:.1%}")
    print(f"  优化耗时: {solution_greedy.optimization_time:.3f}s")

    # 2. 遗传算法
    print("\n【遗传算法 (GA)】")
    ga_config = GeneticAlgorithmConfig(
        population_size=30,
        generations=50,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    ga_optimizer = GeneticSensorOptimizer(base_optimizer, ga_config)
    solution_ga = ga_optimizer.optimize()
    print(f"  传感器数: {solution_ga.sensor_count}")
    print(f"  总投资: ¥{solution_ga.total_cost:,.0f}")
    print(f"  覆盖率: {solution_ga.coverage_rate:.1%}")
    print(f"  优化耗时: {solution_ga.optimization_time:.3f}s")

    # 3. 粒子群算法
    print("\n【粒子群算法 (PSO)】")
    pso_config = PSOConfig(
        swarm_size=25,
        iterations=50,
        w=0.7,
        c1=1.5,
        c2=1.5
    )
    pso_optimizer = PSOSensorOptimizer(base_optimizer, pso_config)
    solution_pso = pso_optimizer.optimize()
    print(f"  传感器数: {solution_pso.sensor_count}")
    print(f"  总投资: ¥{solution_pso.total_cost:,.0f}")
    print(f"  覆盖率: {solution_pso.coverage_rate:.1%}")
    print(f"  优化耗时: {solution_pso.optimization_time:.3f}s")

    # 4. 多算法集成
    print("\n【多算法集成优化】")
    multi_optimizer = MultiAlgorithmOptimizer(base_optimizer)
    all_results = multi_optimizer.optimize_all()
    solution_multi = multi_optimizer.get_best_solution()
    print(f"  比较算法数: {len(all_results)}")
    print(f"  最佳方案: {solution_multi.name}")
    print(f"  传感器数: {solution_multi.sensor_count}")
    print(f"  总投资: ¥{solution_multi.total_cost:,.0f}")
    print(f"  覆盖率: {solution_multi.coverage_rate:.1%}")
    print(f"  优化耗时: {solution_multi.optimization_time:.3f}s")

    # 比较汇总
    print("\n\n--- 算法性能比较汇总 ---")
    print(f"{'算法':<15} {'传感器数':<10} {'总投资':<15} {'覆盖率':<10} {'耗时(s)':<10}")
    print("-" * 60)
    for name, sol in [
        ("贪心算法", solution_greedy),
        ("遗传算法", solution_ga),
        ("粒子群算法", solution_pso),
        ("多算法集成", solution_multi)
    ]:
        print(f"{name:<15} {sol.sensor_count:<10} ¥{sol.total_cost:>12,.0f} "
              f"{sol.coverage_rate:<10.1%} {sol.optimization_time:<10.3f}")

    return solution_multi


def demo_report_generation():
    """演示报告生成功能"""
    print("\n" + "=" * 70)
    print("演示5: 可视化报告生成")
    print("=" * 70)

    # 创建优化方案
    optimizer = YCJLSensorOptimizer()
    solution = optimizer.optimize()

    # 创建报告生成器
    reporter = SensorOptimizationReporter(solution, optimizer)

    # 生成Markdown报告
    print("\n【Markdown报告预览】")
    md_report = reporter.generate_markdown_report()
    # 只显示前30行
    lines = md_report.split('\n')[:30]
    print('\n'.join(lines))
    print("... (更多内容省略)")

    # 生成JSON报告
    print("\n\n【JSON报告结构】")
    import json
    json_report = reporter.generate_json_report()
    json_data = json.loads(json_report)
    print(f"  方案ID: {json_data['id']}")
    print(f"  方案名称: {json_data['name']}")
    print(f"  生成时间: {json_data['timestamp']}")
    print(f"  评估指标:")
    print(f"    - 覆盖率: {json_data['coverage_rate']:.1%}")
    print(f"    - 可观测性: {json_data['observability_score']:.1%}")
    print(f"    - 冗余度: {json_data['redundancy_score']:.1%}")
    print(f"    - 总投资: ¥{json_data['total_cost']:,.0f}")
    print(f"    - 传感器数: {json_data['sensor_count']}")

    # 保存HTML报告
    print("\n\n【HTML报告生成】")
    html_path = "/tmp/sensor_optimization_report.html"
    reporter.save_report(html_path, ReportFormat.HTML)
    print(f"  HTML报告已保存到: {html_path}")

    # 保存Markdown报告
    md_path = "/tmp/sensor_optimization_report.md"
    reporter.save_report(md_path, ReportFormat.MARKDOWN)
    print(f"  Markdown报告已保存到: {md_path}")

    return solution


def demo_sensor_catalog():
    """演示传感器目录功能"""
    print("\n" + "=" * 70)
    print("演示6: 传感器产品目录")
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
    print(" 传感器点位与参数优化模块演示 (v2.0)")
    print(" YCJL Sensor Optimization Module Demo")
    print("=" * 70)

    # 演示1: YCJL工程优化
    demo_ycjl_optimization()

    # 演示2: 通用工程优化
    demo_generic_optimization()

    # 演示3: 目标比较
    demo_objective_comparison()

    # 演示4: 高级优化算法 (新增)
    demo_advanced_algorithms()

    # 演示5: 报告生成 (新增)
    demo_report_generation()

    # 演示6: 传感器目录
    demo_sensor_catalog()

    print("\n" + "=" * 70)
    print(" 演示完成")
    print("=" * 70)
    print("\n生成的报告文件:")
    print("  - HTML报告: /tmp/sensor_optimization_report.html")
    print("  - Markdown报告: /tmp/sensor_optimization_report.md")


if __name__ == "__main__":
    main()
