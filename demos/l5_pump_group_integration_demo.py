#!/usr/bin/env python3
"""
L5泵站群集成系统演示程序
========================

演示L5自主系统与泵站群智能体的深度融合:
1. 态势感知集成 - 泵站群状态纳入L5态势
2. 决策规划集成 - 经济优化融入L5决策
3. 执行控制集成 - 安全规则优先执行
4. 统一决策框架 - 安全与经济目标融合

使用方法:
    python demos/l5_pump_group_integration_demo.py
"""

import sys
import os
import time
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ycjl.agents.l5_pump_group_integration import (
    create_l5_pump_group_system,
    L5PumpGroupSystem,
    IntegrationMode,
    PumpGroupSituation,
)
from ycjl.agents.pump_group_agents import PumpStatus


def print_header(title: str, char: str = "="):
    """打印标题"""
    width = 80
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def demo_system_architecture(system: L5PumpGroupSystem):
    """演示系统架构"""
    print_header("1. L5泵站群集成系统架构")

    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │                   L5PumpGroupSystem                          │
    ├─────────────────────────────────────────────────────────────┤
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
    │  │  态势感知    │──│  决策规划    │──│     执行控制        │  │
    │  │  Awareness  │  │  Planning   │  │    Execution        │  │
    │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
    │        │               │                    │                │
    │        ▼               ▼                    ▼                │
    │  ┌─────────────────────────────────────────────────────────┐│
    │  │              泵站群智能体层 (PumpGroup Layer)            ││
    │  │  ┌───────────┐  ┌───────────┐  ┌───────────────────┐    ││
    │  │  │ L1 Safety │  │L3 Economic│  │   Coordinator     │    ││
    │  │  │ 安全规则  │  │ 经济优化  │  │   协调融合        │    ││
    │  │  └───────────┘  └───────────┘  └───────────────────┘    ││
    │  └─────────────────────────────────────────────────────────┘│
    │        │               │                    │                │
    │        ▼               ▼                    ▼                │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
    │  │  协调管理    │  │  学习优化    │  │   泵站群状态       │  │
    │  │ Coordination│  │  Learning   │  │  PumpGroupState    │  │
    │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
    """)

    print_section("1.1 系统组件")

    print(f"\n  [L5核心智能体]")
    agents = system.get_agent_status()
    for role, status in agents.items():
        print(f"    - {status['name']}: {status['status']}")

    print(f"\n  [泵站群智能体]")
    print(f"    - 安全智能体: {system.pump_system.safety_agent.agent_id}")
    print(f"      规则数: {len(system.pump_system.safety_agent.rules)}")
    print(f"    - 经济智能体: {system.pump_system.economic_agent.agent_id}")
    print(f"      优化周期: {system.pump_system.economic_agent.optimization_interval/3600:.0f}小时")
    print(f"    - 协调智能体: {system.pump_system.coordinator.agent_id}")

    print(f"\n  [泵站群配置]")
    print(f"    - 泵站数量: {len(system.pump_system.pump_group_state.stations)}")
    for sid, station in system.pump_system.pump_group_state.stations.items():
        print(f"      - {station.station_name}: {station.total_pump_count}台泵")


def demo_situation_awareness(system: L5PumpGroupSystem):
    """演示态势感知集成"""
    print_header("2. 态势感知集成演示")

    print_section("2.1 模拟传感器数据")

    # 模拟传感器数据
    sensor_data = {
        'flow_rate': 15.0,
        'pressure': 0.5,
        'water_level': 3.0,
        'valve_position': 0.8,
        'hour': 14,
        'target_flow': 15.0,
        'stations': {
            'tundian': {
                'forebay_level': 3.2,
                'total_flow': 4.0,
                'total_power': 800,
                'pumps': {
                    'P1': {'flow': 2.0, 'power': 400, 'efficiency': 0.82, 'vibration': 2.5},
                    'P2': {'flow': 2.0, 'power': 400, 'efficiency': 0.80, 'vibration': 2.8},
                }
            },
            'qianliulin': {
                'forebay_level': 2.8,
                'total_flow': 3.5,
                'total_power': 700,
            }
        }
    }

    print(f"\n  传感器数据:")
    print(f"    - 总流量: {sensor_data['flow_rate']} m³/s")
    print(f"    - 压力: {sensor_data['pressure']} MPa")
    print(f"    - 当前时刻: {sensor_data['hour']}:00")

    print_section("2.2 执行处理周期")

    # 执行处理周期
    result = system.process_cycle(sensor_data)

    print(f"\n  周期执行时间: {result['cycle_time']*1000:.2f} ms")

    # 显示态势感知结果
    print_section("2.3 态势感知结果")

    pump_situation = result['pump_situation']
    print(f"\n  [泵站群态势]")
    print(f"    - 总泵站数: {pump_situation['total_stations']}")
    print(f"    - 总泵数: {pump_situation['total_pumps']}")
    print(f"    - 运行泵数: {pump_situation['running_pumps']}")
    print(f"    - 故障泵数: {pump_situation['fault_pumps']}")
    print(f"    - 平均前池水位: {pump_situation['average_forebay_level']:.2f} m")
    print(f"    - 安全风险等级: {pump_situation['safety_risk_level']:.1f}")
    print(f"    - 经济效率: {pump_situation['economic_efficiency']:.2f}")
    print(f"    - 整体健康度: {pump_situation['overall_health']:.2f}")

    integrated = result['integrated_situation']
    print(f"\n  [融合态势]")
    print(f"    - 综合风险等级: {integrated['integrated_risk']:.1f}")
    print(f"    - 活跃安全规则: {len(integrated['active_safety_rules'])}")
    print(f"    - 活跃场景数: {integrated['active_scenarios']}")


def demo_decision_planning(system: L5PumpGroupSystem):
    """演示决策规划集成"""
    print_header("3. 决策规划集成演示")

    print_section("3.1 不同时段的经济优化决策")

    hours_to_test = [3, 12, 19]  # 谷时、平时、峰时

    for hour in hours_to_test:
        sensor_data = {
            'hour': hour,
            'target_flow': 15.0,
            'flow_rate': 15.0,
        }

        result = system.process_cycle(sensor_data)

        price = system.pump_system.economic_agent._get_current_price(hour)
        period = "谷时" if hour in [0,1,2,3,4,5,6] else ("峰时" if hour in [8,9,10,11,18,19,20,21] else "平时")

        print(f"\n  [{hour:02d}:00 - {period}] 电价: {price:.2f} 元/kWh")
        print(f"    - 安全决策数: {result['safety_decisions']}")
        print(f"    - 经济决策数: {result['economic_decisions']}")
        print(f"    - 总决策数: {result['total_decisions']}")
        print(f"    - 批准决策数: {result['approved_decisions']}")

        # 获取优化建议
        opt_advice = system.pump_planning.optimize_for_price(hour)
        print(f"    - 策略: {opt_advice['strategy']}")
        print(f"    - 建议: {opt_advice['recommendation']}")


def demo_safety_priority(system: L5PumpGroupSystem):
    """演示安全优先机制"""
    print_header("4. 安全优先机制演示")

    print_section("4.1 正常运行场景")

    # 设置正常状态
    for station in system.pump_system.pump_group_state.stations.values():
        station.forebay_level = 3.0
        station.running_pump_count = 2

    sensor_data = {'hour': 12, 'target_flow': 15.0}
    result = system.process_cycle(sensor_data)

    print(f"\n  前池水位: 3.0m (正常)")
    print(f"  安全覆盖触发: {'否' if result['safety_decisions'] == 0 else '是'}")
    print(f"  安全决策数: {result['safety_decisions']}")

    print_section("4.2 低水位报警场景")

    # 设置低水位
    system.pump_system.pump_group_state.stations['tundian'].forebay_level = 1.2

    result = system.process_cycle(sensor_data)

    print(f"\n  屯佃泵站前池水位: 1.2m (低水位报警)")
    print(f"  安全覆盖触发: {'是' if result['safety_decisions'] > 0 else '否'}")
    print(f"  安全决策数: {result['safety_decisions']}")
    active_rules = system.pump_system.safety_agent.active_rules
    if active_rules:
        print(f"  触发的安全规则:")
        for rule in active_rules:
            print(f"    - {rule}")

    print_section("4.3 临界低水位场景")

    # 设置临界低水位
    system.pump_system.pump_group_state.stations['tundian'].forebay_level = 0.5

    result = system.process_cycle(sensor_data)

    print(f"\n  屯佃泵站前池水位: 0.5m (临界低水位)")
    print(f"  安全覆盖触发: {'是' if result['safety_decisions'] > 0 else '否'}")
    print(f"  安全决策数: {result['safety_decisions']}")
    active_rules = system.pump_system.safety_agent.active_rules
    if active_rules:
        print(f"  触发的安全规则:")
        for rule in active_rules:
            print(f"    - {rule}")
        print(f"  预期动作: 紧急停止所有泵")

    # 恢复正常
    for station in system.pump_system.pump_group_state.stations.values():
        station.forebay_level = 3.0


def demo_integrated_cycle(system: L5PumpGroupSystem):
    """演示完整集成周期"""
    print_header("5. 24小时完整集成仿真")

    print_section("5.1 仿真设置")
    print(f"\n  - 仿真时长: 24小时")
    print(f"  - 目标流量: 15.0 m³/s")
    print(f"  - 初始前池水位: 3.0m")

    # 重置系统
    system.pump_system.reset()
    system.integrated_cycle_count = 0
    system.safety_override_count = 0
    system.economic_decisions_count = 0

    for station in system.pump_system.pump_group_state.stations.values():
        station.forebay_level = 3.0
        station.running_pump_count = 2

    print_section("5.2 仿真执行")

    hourly_results = []

    for hour in range(24):
        # 模拟前池水位波动
        for station in system.pump_system.pump_group_state.stations.values():
            level_change = np.random.uniform(-0.2, 0.2)
            station.forebay_level = np.clip(
                station.forebay_level + level_change,
                1.5, 4.5
            )

        sensor_data = {
            'hour': hour,
            'target_flow': 15.0,
            'flow_rate': 15.0,
        }

        result = system.process_cycle(sensor_data)

        hourly_results.append({
            'hour': hour,
            'cycle_time': result['cycle_time'],
            'safety_decisions': result['safety_decisions'],
            'economic_decisions': result['economic_decisions'],
            'approved': result['approved_decisions'],
            'risk': result['integrated_situation']['integrated_risk'],
        })

        # 每6小时输出一次
        if hour % 6 == 0 or hour == 23:
            price = system.pump_system.economic_agent._get_current_price(hour)
            print(f"  {hour:02d}:00 | 电价:{price:.2f} | "
                  f"安全决策:{result['safety_decisions']} | "
                  f"经济决策:{result['economic_decisions']} | "
                  f"风险:{result['integrated_situation']['integrated_risk']:.1f}")

    print_section("5.3 仿真统计")

    summary = system.get_integrated_state_summary()

    print(f"\n  [系统统计]")
    print(f"    - 总周期数: {summary['integrated_cycles']}")
    print(f"    - 安全覆盖次数: {summary['safety_overrides']}")
    print(f"    - 经济决策总数: {summary['economic_decisions']}")
    print(f"    - 系统健康度: {summary['system_health']:.2%}")
    print(f"    - 置信度: {summary['confidence']:.2%}")

    print(f"\n  [智能体状态]")
    agents = system.get_agent_status()
    for role, status in agents.items():
        print(f"    - {role}: 周期={status['cycle_count']}, 错误={status['error_count']}")

    # 统计每小时的安全决策和经济决策
    total_safety = sum(r['safety_decisions'] for r in hourly_results)
    total_economic = sum(r['economic_decisions'] for r in hourly_results)
    avg_risk = sum(r['risk'] for r in hourly_results) / len(hourly_results)

    print(f"\n  [决策统计]")
    print(f"    - 安全决策总数: {total_safety}")
    print(f"    - 经济决策总数: {total_economic}")
    print(f"    - 平均风险等级: {avg_risk:.2f}")


def demo_optimization_report(system: L5PumpGroupSystem):
    """演示优化报告"""
    print_header("6. 泵站群优化报告")

    report = system.get_pump_optimization_report()

    print_section("6.1 经济优化摘要")
    opt_summary = report['optimization_summary']
    print(f"\n  - 调度计划时长: {opt_summary['schedule_length']} 小时")
    print(f"  - 24小时总成本: {opt_summary['total_24h_cost']:.0f} 元")
    print(f"  - 能耗成本: {opt_summary['energy_cost']:.0f} 元")
    print(f"  - 启停成本: {opt_summary['start_stop_cost']:.0f} 元")

    print_section("6.2 前6小时调度计划")
    if report['schedule']:
        print(f"\n  {'时刻':<6} {'电价':<8} {'建议方案'}")
        print(f"  {'-'*40}")
        for hour_data in report['schedule'][:6]:
            hour = hour_data['hour']
            price = hour_data['electricity_price']
            stations_summary = []
            for sid, sdata in hour_data.get('stations', {}).items():
                stations_summary.append(f"{sid[:6]}:{sdata['optimal_pump_count']}台")
            print(f"  {hour:02d}:00  {price:.2f}元   {', '.join(stations_summary[:3])}")

    print_section("6.3 安全规则状态")
    safety = report['safety_rules']
    print(f"\n  - 总规则数: {safety['rule_count']}")
    print(f"  - 触发过的规则: {safety['triggered_count']}")
    if safety['active_rules']:
        print(f"  - 当前活跃规则: {', '.join(safety['active_rules'])}")

    print_section("6.4 协调状态")
    coord = report['coordination']
    print(f"\n  - 协调模式: {coord['mode']}")
    print(f"  - 级联调度数: {coord['cascade_schedule_length']}")


def main():
    """主函数"""
    print("\n")
    print("=" * 80)
    print("  L5泵站群集成系统演示")
    print("  L5 Pump Group Integration System Demo")
    print("=" * 80)
    print("\n  版本: v1.0")
    print("  功能: L5自主系统 + 泵站群智能体 深度融合")
    print("  目标: 统一决策框架，安全与经济目标融合")

    # 创建集成系统
    print_header("0. 创建L5泵站群集成系统")
    system = create_l5_pump_group_system()
    print(f"\n  系统创建成功!")
    print(f"  - 集成模式: {system.integration_mode.name}")
    print(f"  - 自主等级: {system.state.autonomy_level.name}")
    print(f"  - 运行模式: {system.state.operation_mode.name}")

    # 1. 系统架构演示
    demo_system_architecture(system)

    # 2. 态势感知集成演示
    demo_situation_awareness(system)

    # 3. 决策规划集成演示
    demo_decision_planning(system)

    # 4. 安全优先机制演示
    demo_safety_priority(system)

    # 5. 完整集成周期演示
    demo_integrated_cycle(system)

    # 6. 优化报告演示
    demo_optimization_report(system)

    # 总结
    print_header("总结", "=")
    print("""
  L5泵站群集成系统已完成深度融合:

  [态势感知集成]
    - L5核心态势 + 泵站群态势 → 融合态势
    - 风险评估: 核心风险 + 安全风险 → 综合风险

  [决策规划集成]
    - L5核心决策 + 泵站群经济决策 → 统一决策
    - 峰谷电价优化自动融入决策流程

  [执行控制集成]
    - 安全检查: 所有决策需通过安全审核
    - 安全优先: L1安全决策 > L3经济决策

  [统一决策框架]
    - 优先级排序: 安全(0) > 战术(5) > 经济(5)
    - 冲突仲裁: 安全规则自动覆盖经济目标

  系统版本: v3.9.0
""")


if __name__ == "__main__":
    main()
