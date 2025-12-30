#!/usr/bin/env python
"""
水利工程智能输水系统命令行工具 (CLI)
====================================

用法:
    python cli.py [命令] [选项]

命令:
    version     显示版本信息
    status      显示系统状态
    diagnose    运行数据诊断
    simulate    运行仿真
    schedule    生成调度决策
    compare     对比两个项目
"""

import sys
import argparse
from datetime import datetime


def cmd_version(args):
    """显示版本信息"""
    import ycjl
    import ycjl.core

    print("水利工程智能输水系统 (YCJL)")
    print("=" * 40)
    print(f"项目版本: {ycjl.__version__}")
    print(f"核心框架: {ycjl.core.__version__}")
    print(f"作者: {ycjl.__author__}")


def cmd_status(args):
    """显示系统状态"""
    project = args.project

    if project == "ycjl":
        from ycjl.control.enhanced_scheduler import EnhancedScheduler
        print("\n引绰济辽工程系统状态")
        print("=" * 40)

        # 数据完备性
        report = EnhancedScheduler.check_data_readiness()
        print(f"数据就绪: {report.readiness_level.name}")
        print(f"完备率: {report.completeness_ratio:.1%}")

        # 月度约束
        is_valid, msgs = EnhancedScheduler.check_monthly_constraints(360.0)
        print(f"约束状态: {'合规' if is_valid else '违规'}")

    elif project == "miyun":
        from ycjl.miyun.scheduler import Scheduler
        print("\n密云水库调蓄工程系统状态")
        print("=" * 40)

        # 数据诊断
        from ycjl.miyun.gap_analyzer import GapAnalyzer
        report = GapAnalyzer.analyze_readiness()
        print(f"数据就绪: {report.current_level.name}")

        # 流量范围
        max_f, min_f = Scheduler.get_recommended_flow_range()
        print(f"推荐流量: {min_f:.1f} - {max_f:.1f} m³/s")


def cmd_diagnose(args):
    """运行数据诊断"""
    project = args.project

    if project == "ycjl":
        from ycjl.config.gap_analyzer import YCJLGapAnalyzer
        print("\n引绰济辽工程数据诊断")
        print("=" * 40)
        YCJLGapAnalyzer.print_report()

    elif project == "miyun":
        from ycjl.miyun.gap_analyzer import GapAnalyzer
        print("\n密云水库调蓄工程数据诊断")
        print("=" * 40)
        GapAnalyzer.print_report()


def cmd_simulate(args):
    """运行仿真"""
    project = args.project
    flow = args.flow

    if project == "miyun":
        from ycjl.miyun import SimEngine
        print(f"\n密云水库仿真 (流量={flow} m³/s)")
        print("=" * 40)

        result = SimEngine.run_system_diagnosis(flow)
        print(f"系统状态: {result.overall_status.value}")
        print(f"总功耗: {result.total_power_mw:.2f} MW")
        print(f"总扬程: {result.total_head:.1f} m")
        print(f"告警数: {len(result.system_warnings)}")

        if result.system_warnings:
            print("\n告警:")
            for w in result.system_warnings[:5]:
                print(f"  - {w}")
    else:
        print("YCJL仿真功能开发中...")


def cmd_schedule(args):
    """生成调度决策"""
    project = args.project

    if project == "ycjl":
        from ycjl.control.enhanced_scheduler import EnhancedScheduler
        print("\n引绰济辽工程调度决策")
        print("=" * 40)

        decision = EnhancedScheduler.make_enhanced_decision(
            datetime.now(),
            level=args.level,
            inflow=args.inflow
        )
        print(f"调度分区: {decision.zone.name}")
        print(f"供水模式: {decision.supply_mode.name}")
        print(f"健康得分: {decision.health_score:.1f}")
        print(f"检测场景: {decision.detected_scenarios}")

    elif project == "miyun":
        from ycjl.miyun.scheduler import Scheduler
        print("\n密云水库调度决策")
        print("=" * 40)

        decision = Scheduler.generate_schedule(args.flow)
        print(f"调度模式: {decision.mode.value}")
        print(f"目标流量: {decision.target_flow} m³/s")
        print(f"预估功耗: {decision.estimated_power/1000:.2f} MW")


def cmd_compare(args):
    """对比两个项目"""
    import demo_comparison
    demo_comparison.main()


def main():
    parser = argparse.ArgumentParser(
        description="水利工程智能输水系统命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # version
    subparsers.add_parser("version", help="显示版本信息")

    # status
    p_status = subparsers.add_parser("status", help="显示系统状态")
    p_status.add_argument("-p", "--project", choices=["ycjl", "miyun"],
                          default="ycjl", help="项目名称")

    # diagnose
    p_diag = subparsers.add_parser("diagnose", help="运行数据诊断")
    p_diag.add_argument("-p", "--project", choices=["ycjl", "miyun"],
                        default="ycjl", help="项目名称")

    # simulate
    p_sim = subparsers.add_parser("simulate", help="运行仿真")
    p_sim.add_argument("-p", "--project", choices=["ycjl", "miyun"],
                       default="miyun", help="项目名称")
    p_sim.add_argument("-f", "--flow", type=float, default=10.0,
                       help="流量 (m³/s)")

    # schedule
    p_sched = subparsers.add_parser("schedule", help="生成调度决策")
    p_sched.add_argument("-p", "--project", choices=["ycjl", "miyun"],
                         default="ycjl", help="项目名称")
    p_sched.add_argument("-l", "--level", type=float, default=360.0,
                         help="水位 (m)")
    p_sched.add_argument("-i", "--inflow", type=float, default=40.0,
                         help="入库流量 (m³/s)")
    p_sched.add_argument("-f", "--flow", type=float, default=10.0,
                         help="目标流量 (m³/s)")

    # compare
    subparsers.add_parser("compare", help="对比两个项目")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "version": cmd_version,
        "status": cmd_status,
        "diagnose": cmd_diagnose,
        "simulate": cmd_simulate,
        "schedule": cmd_schedule,
        "compare": cmd_compare,
    }

    try:
        commands[args.command](args)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
