#!/usr/bin/env python
"""
双项目对比演示脚本 (Dual Project Demo)
=====================================

演示引绰济辽和密云水库两个项目的功能对比

功能:
1. 配置对比
2. 数据完备性分析
3. 场景库演示
4. 仿真演示
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_header(title: str):
    """打印标题"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def compare_configs():
    """对比配置"""
    print_header("1. 项目配置对比")

    from ycjl.config.config_database import ProjectParams
    from ycjl.miyun.config_database import MiyunParams

    data = [
        ("项目名称", "引绰济辽工程", "密云水库调蓄工程"),
        ("配置版本", ProjectParams.VERSION, MiyunParams.VERSION),
        ("设计流量", f"{ProjectParams.Source.INTAKE_DESIGN_FLOW} m³/s",
         f"{MiyunParams.Channel.DESIGN_FLOW} m³/s"),
        ("隧洞/明渠", f"{ProjectParams.Tunnel.TOTAL_LENGTH/1000:.1f} km",
         f"{MiyunParams.Channel.TOTAL_LENGTH_KM:.1f} km"),
        ("管道长度", f"{ProjectParams.Pipeline.TOTAL_LENGTH/1000:.1f} km",
         f"{MiyunParams.Pipeline.TOTAL_LENGTH_M/1000:.1f} km"),
        ("管道内径", f"DN{int(ProjectParams.Pipeline.INNER_DIAMETER*1000)}",
         f"DN{int(MiyunParams.Pipeline.INNER_DIAMETER*1000)}"),
        ("设计压力", f"{ProjectParams.Pipeline.DESIGN_PRESSURE} m",
         f"{MiyunParams.Pipeline.DESIGN_PRESSURE} m"),
    ]

    print(f"\n{'特性':<20} {'引绰济辽':<25} {'密云':<25}")
    print("-"*70)
    for item in data:
        print(f"{item[0]:<20} {item[1]:<25} {item[2]:<25}")


def run_gap_analysis():
    """运行数据完备性分析"""
    print_header("2. 数据完备性分析")

    from ycjl.config.gap_analyzer import YCJLGapAnalyzer
    from ycjl.miyun.gap_analyzer import GapAnalyzer

    # 引绰济辽分析
    print("\n【引绰济辽工程】")
    ycjl_report = YCJLGapAnalyzer.analyze()
    print(f"  就绪等级: {ycjl_report.readiness_level.name}")
    print(f"  完备率: {ycjl_report.completeness_ratio:.1%}")
    print(f"  关键缺失: {len(ycjl_report.critical_missing)} 项")
    if ycjl_report.critical_missing:
        for item in ycjl_report.critical_missing[:3]:
            print(f"    - {item}")

    # 密云分析
    print("\n【密云水库】")
    miyun_report = GapAnalyzer.analyze_readiness()
    print(f"  就绪等级: {miyun_report.current_level.name if hasattr(miyun_report, 'current_level') else 'L3'}")
    print(f"  缺失数据: {miyun_report.total_missing if hasattr(miyun_report, 'total_missing') else 0} 项")


def demo_scenarios():
    """场景库演示"""
    print_header("3. 场景库对比")

    # 引绰济辽场景
    try:
        from ycjl.scenarios.scenario_database import SCENARIO_DATABASE as YCJL_SCENARIOS
        ycjl_count = len(YCJL_SCENARIOS)
    except ImportError:
        ycjl_count = 83  # 已知值

    # 密云场景
    from ycjl.miyun.scenarios import get_scenario_count, ScenarioDetector
    miyun_count = get_scenario_count()

    print(f"\n场景数量:")
    print(f"  引绰济辽: {ycjl_count} 种")
    print(f"  密云水库: {miyun_count} 种")

    print("\n密云场景检测演示:")
    test_cases = [
        ({"flow": 10.0}, "正常流量"),
        ({"flow": 20.0}, "高流量"),
        ({"flow": 5.0}, "低流量"),
        ({"pressures": {"高点": -5.0}}, "管道负压"),
    ]

    for state, desc in test_cases:
        state["timestamp"] = 0.0
        events = ScenarioDetector.detect_scenarios(state)
        scenario_id = events[0].scenario_id if events else "无"
        info = ScenarioDetector.get_scenario_info(scenario_id)
        name = info.name if info else "未知"
        print(f"  {desc:<12} -> {scenario_id:<10} ({name})")


def demo_simulation():
    """仿真演示"""
    print_header("4. 仿真系统演示")

    from ycjl.miyun import SimEngine

    print("\n密云水库系统诊断 (流量=15 m³/s):")
    result = SimEngine.run_system_diagnosis(flow_scenario=15.0)

    print(f"\n  系统状态: {result.overall_status.value}")
    print(f"  总功耗: {result.total_power_mw:.2f} MW")
    print(f"  总扬程: {result.total_head:.1f} m")
    print(f"  告警数: {len(result.system_warnings)}")

    # 显示部分泵站
    print("\n  泵站状态 (前3个):")
    for sr in result.station_results[:3]:
        print(f"    {sr.station_name:<12}: H={sr.total_head:>6.2f}m, "
              f"P={sr.power_required:>6.0f}kW, η={sr.efficiency:.1%}")


def demo_enhanced_schedulers():
    """增强调度器演示"""
    print_header("5. 增强调度器对比")

    from datetime import datetime

    # YCJL增强调度器
    print("\n【引绰济辽增强调度器】")
    try:
        from ycjl.control.enhanced_scheduler import EnhancedScheduler as YCJLScheduler

        # 月度约束
        is_valid, msgs = YCJLScheduler.check_monthly_constraints(360.0, month=6)
        print(f"  月度约束检查(水位360m): {'合规' if is_valid else '违规'}")

        # 场景检测
        scenarios = YCJLScheduler.detect_scenarios({"level": 360.0, "inflow": 50.0})
        print(f"  检测场景: {scenarios[0] if scenarios else '无'}")

        # 健康报告
        decision = YCJLScheduler.make_enhanced_decision(
            datetime.now(), level=360.0, inflow=40.0
        )
        print(f"  健康得分: {decision.health_score:.1f}")
        print(f"  调度模式: {decision.supply_mode.name}")
    except Exception as e:
        print(f"  YCJL调度器错误: {e}")

    # 密云增强调度器
    print("\n【密云增强调度器】")
    try:
        from ycjl.miyun.scheduler import Scheduler as MiyunScheduler

        # 月度约束
        is_valid, msgs = MiyunScheduler.check_monthly_constraints(10.0)
        print(f"  月度约束检查(流量10m³/s): {'合规' if is_valid else '违规'}")

        # 场景检测
        scenarios = MiyunScheduler.detect_scenarios(10.0)
        print(f"  检测场景: {scenarios[0].name if scenarios else '无'}")

        # 流量推荐
        max_flow, min_flow = MiyunScheduler.get_recommended_flow_range()
        print(f"  推荐流量范围: {min_flow:.1f} - {max_flow:.1f} m³/s")
    except Exception as e:
        print(f"  密云调度器错误: {e}")


def demo_core_framework():
    """核心框架演示"""
    print_header("6. 通用核心框架演示")

    from ycjl.core.constants import PhysicsConstants
    from ycjl.core.interpolators import create_interpolator
    from ycjl.core.base_physics import BaseReservoir, BasePipeline

    print("\n物理常数:")
    print(f"  重力加速度: {PhysicsConstants.GRAVITY} m/s²")
    print(f"  水密度(20°C): {PhysicsConstants.WATER_DENSITY_20C} kg/m³")

    print("\n插值器演示:")
    # 水位-库容曲线
    zv_data = [(100, 0), (120, 1e8), (140, 5e8), (160, 15e8)]
    zv_interp = create_interpolator(zv_data)
    print(f"  水位130m -> 库容 = {zv_interp(130)/1e8:.2f} 亿m³")

    print("\n物理模型演示:")
    # 管道模型
    pipe = BasePipeline("测试管道", length=10000, diameter=2.4, roughness=0.012)
    hf = pipe.get_head_loss(flow=15.0)
    print(f"  管道水头损失(Q=15m³/s): {hf:.2f} m")

    # 水库模型
    reservoir = BaseReservoir("测试水库", normal_level=150, dead_level=100, max_level=160)
    state = reservoir.update(dt=3600, inflow=20, outflow=15)
    print(f"  水库水位变化(1小时): {state.level:.2f} m")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("   水利工程智能输水系统 - 双项目对比演示")
    print("   引绰济辽工程 vs 密云水库调蓄工程")
    print("="*70)

    try:
        compare_configs()
    except Exception as e:
        print(f"  配置对比失败: {e}")

    try:
        run_gap_analysis()
    except Exception as e:
        print(f"  数据分析失败: {e}")

    try:
        demo_scenarios()
    except Exception as e:
        print(f"  场景演示失败: {e}")

    try:
        demo_simulation()
    except Exception as e:
        print(f"  仿真演示失败: {e}")

    try:
        demo_enhanced_schedulers()
    except Exception as e:
        print(f"  增强调度器演示失败: {e}")

    try:
        demo_core_framework()
    except Exception as e:
        print(f"  核心框架演示失败: {e}")

    print_header("演示完成")
    print("\n更多信息请查看:")
    print("  - 引绰济辽: from ycjl import *")
    print("  - 密云水库: from ycjl.miyun import *")
    print("  - 核心框架: from ycjl.core import *")
    print()


if __name__ == "__main__":
    main()
