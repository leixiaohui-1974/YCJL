#!/usr/bin/env python3
"""
引绰济辽智能输水系统命令行接口
==============================

用法:
    python -m ycjl.cli run [--duration SECONDS] [--scenario SCENARIO]
    python -m ycjl.cli test [--scenario SCENARIO]
    python -m ycjl.cli analyze [--input FILE] [--output FILE]
    python -m ycjl.cli status
"""

import argparse
import sys
import json
from typing import Optional


def cmd_run(args):
    """运行仿真"""
    from .simulation.runner import SimulationRunner, SimulationConfig
    from .simulation.scenario_injector import InjectionEvent
    from .config.settings import ScenarioType

    print(f"启动仿真...")
    print(f"  时长: {args.duration}s")
    print(f"  时间步长: {args.dt}s")

    config = SimulationConfig(
        duration=args.duration,
        dt=args.dt,
        enable_control=not args.no_control,
        enable_detection=not args.no_detection,
        verbose=args.verbose,
        log_interval=args.log_interval
    )

    runner = SimulationRunner(config)

    # 场景事件
    events = []
    if args.scenario and args.scenario != 'normal':
        scenario_type = getattr(ScenarioType, args.scenario.upper(), ScenarioType.NORMAL)
        events.append(InjectionEvent(
            time=args.scenario_time,
            scenario=scenario_type,
            duration=args.scenario_duration,
            severity=0.5,
            parameters={}
        ))
        print(f"  场景: {args.scenario} @ t={args.scenario_time}s")

    # 运行
    result = runner.run(scenario_events=events)

    # 输出结果
    print("\n" + "="*50)
    print("仿真完成!")
    print("="*50)
    print(f"总步数: {result.steps}")
    print(f"运行时间: {result.duration:.2f}s")
    print(f"控制动作数: {result.control_actions}")
    print(f"安全干预数: {result.safety_interventions}")

    print("\n性能指标:")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\n场景历史:")
    for time, scenario in result.scenario_history[:10]:
        print(f"  t={time:.1f}s: {scenario.name}")

    # 保存结果
    if args.output:
        output_data = {
            'success': result.success,
            'steps': result.steps,
            'duration': result.duration,
            'metrics': result.metrics,
            'scenario_history': [
                {'time': t, 'scenario': s.name}
                for t, s in result.scenario_history
            ]
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n结果已保存到: {args.output}")

    return 0 if result.success else 1


def cmd_test(args):
    """运行场景测试"""
    from .simulation.runner import run_scenario_test
    from .config.settings import ScenarioType

    scenario_type = getattr(ScenarioType, args.scenario.upper(), ScenarioType.NORMAL)

    print(f"运行场景测试: {args.scenario}")
    print(f"  时长: {args.duration}s")

    result = run_scenario_test(
        scenario_type,
        duration=args.duration
    )

    print("\n" + "="*50)
    print("测试完成!")
    print("="*50)
    print(f"成功: {'是' if result.success else '否'}")
    print(f"安全干预: {result.safety_interventions}")

    if result.errors:
        print("\n错误:")
        for err in result.errors:
            print(f"  - {err}")

    return 0 if result.success else 1


def cmd_analyze(args):
    """分析数据"""
    from .analysis.logger import DataLogger
    from .analysis.reporter import ReportGenerator

    print(f"分析数据...")

    if args.input:
        print(f"  输入文件: {args.input}")
        logger = DataLogger()
        logger.load_from_file(args.input)
    else:
        print("  使用模拟数据...")
        logger = DataLogger()
        # 生成一些模拟数据
        import numpy as np
        for i in range(1000):
            logger.log_timeseries({
                'pool_level': 5.0 + np.sin(i * 0.1) * 0.5 + np.random.randn() * 0.1,
                'pipe_flow': 10.0 + np.random.randn() * 0.5,
                'pipe_pressure': 50.0 + np.random.randn() * 2
            }, float(i))

    # 生成报告
    generator = ReportGenerator(logger)
    report = generator.generate_report()

    print("\n" + "="*50)
    print(f"报告: {report.title}")
    print("="*50)
    print(f"摘要: {report.summary}")

    for section in report.sections:
        print(f"\n## {section.title}")
        # 简化输出
        lines = section.content.split('\n')
        for line in lines[:10]:
            if line.strip():
                print(f"  {line.strip()}")
        if len(lines) > 10:
            print(f"  ... (省略 {len(lines)-10} 行)")

    # 导出
    if args.output:
        if args.output.endswith('.json'):
            generator.export_json(report, args.output)
        elif args.output.endswith('.html'):
            generator.export_html(report, args.output)
        else:
            generator.export_markdown(report, args.output)
        print(f"\n报告已保存到: {args.output}")

    return 0


def cmd_status(args):
    """显示系统状态"""
    print("引绰济辽智能输水系统")
    print("="*40)

    try:
        from . import __version__
        print(f"版本: {__version__}")
    except:
        print("版本: 未知")

    print("\n已安装模块:")
    modules = [
        ('config', '配置模块'),
        ('physics', '物理仿真'),
        ('sensors', '传感器'),
        ('actuators', '执行器'),
        ('estimation', '状态估计'),
        ('models', '降阶模型'),
        ('agents', '智能体'),
        ('control', '控制算法'),
        ('scenarios', '场景识别'),
        ('simulation', '仿真引擎'),
        ('analysis', '数据分析')
    ]

    for module, desc in modules:
        try:
            __import__(f'ycjl.{module}')
            status = '✓'
        except ImportError:
            status = '✗'
        print(f"  [{status}] {module}: {desc}")

    return 0


def cmd_demo(args):
    """运行演示"""
    print("运行系统演示...")
    print("="*50)

    # 导入演示脚本
    import subprocess
    import os

    demo_path = os.path.join(os.path.dirname(__file__), '..', 'demo.py')
    if os.path.exists(demo_path):
        subprocess.run([sys.executable, demo_path])
    else:
        print("演示脚本未找到")
        return 1

    return 0


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description='引绰济辽智能输水系统命令行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  ycjl run --duration 3600 --scenario demand_surge
  ycjl test --scenario pipe_burst
  ycjl analyze --input data.json --output report.html
  ycjl status
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # run 命令
    run_parser = subparsers.add_parser('run', help='运行仿真')
    run_parser.add_argument('--duration', type=float, default=3600,
                            help='仿真时长(秒)')
    run_parser.add_argument('--dt', type=float, default=1.0,
                            help='时间步长(秒)')
    run_parser.add_argument('--scenario', type=str, default='normal',
                            choices=['normal', 'demand_surge', 'pipe_burst', 'ice_period', 'power_failure'],
                            help='场景类型')
    run_parser.add_argument('--scenario-time', type=float, default=300,
                            help='场景触发时间')
    run_parser.add_argument('--scenario-duration', type=float, default=1800,
                            help='场景持续时间')
    run_parser.add_argument('--no-control', action='store_true',
                            help='禁用控制')
    run_parser.add_argument('--no-detection', action='store_true',
                            help='禁用场景检测')
    run_parser.add_argument('--verbose', action='store_true',
                            help='详细输出')
    run_parser.add_argument('--log-interval', type=float, default=60,
                            help='日志间隔')
    run_parser.add_argument('--output', '-o', type=str,
                            help='输出文件')
    run_parser.set_defaults(func=cmd_run)

    # test 命令
    test_parser = subparsers.add_parser('test', help='运行场景测试')
    test_parser.add_argument('--scenario', type=str, default='normal',
                             choices=['normal', 'demand_surge', 'pipe_burst', 'ice_period', 'power_failure'],
                             help='测试场景')
    test_parser.add_argument('--duration', type=float, default=600,
                             help='测试时长')
    test_parser.set_defaults(func=cmd_test)

    # analyze 命令
    analyze_parser = subparsers.add_parser('analyze', help='分析数据')
    analyze_parser.add_argument('--input', '-i', type=str,
                                help='输入数据文件')
    analyze_parser.add_argument('--output', '-o', type=str,
                                help='输出报告文件')
    analyze_parser.set_defaults(func=cmd_analyze)

    # status 命令
    status_parser = subparsers.add_parser('status', help='显示系统状态')
    status_parser.set_defaults(func=cmd_status)

    # demo 命令
    demo_parser = subparsers.add_parser('demo', help='运行演示')
    demo_parser.set_defaults(func=cmd_demo)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
