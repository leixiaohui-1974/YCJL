#!/usr/bin/env python3
"""
泵站群多智能体系统演示程序
==========================

演示内容:
1. 泵站群安全智能体 (L1层) - 前池水位保护、启停约束、汽蚀保护等
2. 泵站群经济优化智能体 (L3层) - 峰谷电价优化、最优台数选择、设备寿命均衡
3. 泵站群协调智能体 - 安全与经济目标融合、级联调度

使用方法:
    python demos/pump_group_multi_agent_demo.py
"""

import sys
import os
import time
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ycjl.agents.pump_group_agents import (
    create_pump_group_system,
    PumpGroupMultiAgentSystem,
    PumpStatus,
    PumpState,
    StationState,
)


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


def demo_basic_system():
    """演示基本系统功能"""
    print_header("1. 泵站群多智能体系统基本功能演示")

    # 创建系统（使用密云工程配置）
    print("\n[1.1] 创建泵站群多智能体系统...")
    system = create_pump_group_system()

    print(f"\n系统初始化完成:")
    print(f"  - 泵站数量: {len(system.pump_group_state.stations)}")
    for sid, station in system.pump_group_state.stations.items():
        print(f"    - {station.station_name}: {station.total_pump_count}台泵")

    # 显示智能体信息
    print(f"\n[1.2] 多智能体架构:")
    print(f"  - L1 安全智能体: {system.safety_agent.agent_id}")
    print(f"    - 安全规则数: {len(system.safety_agent.rules)}")
    print(f"  - L3 经济智能体: {system.economic_agent.agent_id}")
    print(f"    - 优化周期: {system.economic_agent.optimization_interval/3600:.0f}小时")
    print(f"  - 协调智能体: {system.coordinator.agent_id}")

    return system


def demo_safety_agent(system: PumpGroupMultiAgentSystem):
    """演示安全智能体功能"""
    print_header("2. 泵站群安全智能体 (L1层) 演示")

    safety_agent = system.safety_agent

    # 显示安全规则
    print_section("2.1 安全规则库")
    print(f"\n共 {len(safety_agent.rules)} 条安全规则:")

    # 按类别分组显示
    from ycjl.agents.reflex_agent import RuleCategory
    for category in RuleCategory:
        rules_in_category = [r for r in safety_agent.rules.values()
                            if r.category == category]
        if rules_in_category:
            print(f"\n  [{category.name}] ({len(rules_in_category)} 条)")
            for rule in rules_in_category:
                print(f"    - {rule.rule_id}: {rule.description}")

    # 显示安全参数
    print_section("2.2 安全参数配置")
    params = safety_agent.params
    print(f"\n  启停约束:")
    print(f"    - 最小启动间隔: {params['min_start_interval']}s")
    print(f"    - 最小停机间隔: {params['min_stop_interval']}s")
    print(f"    - 最小运行时间: {params['min_run_time']}s")

    print(f"\n  前池水位保护:")
    print(f"    - 临界低水位: {params['forebay_critical_low']}m")
    print(f"    - 报警低水位: {params['forebay_alarm_low']}m")
    print(f"    - 报警高水位: {params['forebay_alarm_high']}m")
    print(f"    - 临界高水位: {params['forebay_critical_high']}m")

    print(f"\n  振动保护:")
    print(f"    - 振动报警值: {params['vibration_alarm']} mm/s")
    print(f"    - 振动跳闸值: {params['vibration_trip']} mm/s")

    print(f"\n  温度保护:")
    print(f"    - 轴承温度报警: {params['bearing_temp_alarm']}°C")
    print(f"    - 轴承温度跳闸: {params['bearing_temp_trip']}°C")

    # 模拟安全场景
    print_section("2.3 安全场景模拟")

    # 场景1: 正常运行
    print("\n  [场景1] 正常运行 (前池水位3.0m)")
    for station in system.pump_group_state.stations.values():
        station.forebay_level = 3.0
        station.running_pump_count = 2

    result = system.step({'hour': 12, 'target_flow': 10.0})
    print(f"    协调模式: {result.get('coordination_mode', 'normal')}")
    print(f"    触发规则: {safety_agent.active_rules if safety_agent.active_rules else '无'}")

    # 场景2: 前池低水位报警
    print("\n  [场景2] 前池低水位报警 (水位1.2m)")
    system.pump_group_state.stations['tundian'].forebay_level = 1.2

    result = system.step({'hour': 12, 'target_flow': 10.0})
    print(f"    协调模式: {result.get('coordination_mode', 'normal')}")
    print(f"    触发规则: {safety_agent.active_rules if safety_agent.active_rules else '无'}")
    print(f"    动作数: {len(result.get('actions', []))}")

    # 场景3: 前池临界低水位
    print("\n  [场景3] 前池临界低水位 (水位0.5m) - 紧急停泵")
    system.pump_group_state.stations['tundian'].forebay_level = 0.5

    result = system.step({'hour': 12, 'target_flow': 10.0})
    print(f"    协调模式: {result.get('coordination_mode', 'normal')}")
    print(f"    触发规则: {safety_agent.active_rules if safety_agent.active_rules else '无'}")
    print(f"    动作数: {len(result.get('actions', []))}")
    if result.get('actions'):
        for action in result['actions']:
            print(f"      - {action['actuator']}: {action['type']} = {action['value']}")

    # 恢复正常
    for station in system.pump_group_state.stations.values():
        station.forebay_level = 3.0


def demo_economic_agent(system: PumpGroupMultiAgentSystem):
    """演示经济优化智能体功能"""
    print_header("3. 泵站群经济优化智能体 (L3层) 演示")

    economic_agent = system.economic_agent

    # 显示电价曲线
    print_section("3.1 24小时电价曲线")
    print("\n  时段电价:")
    price_curve = economic_agent.price_curve

    # 按时段显示
    valley_hours = economic_agent.params['valley_hours']
    peak_hours = economic_agent.params['peak_hours']

    print(f"\n  谷时 (00:00-06:00): {economic_agent.params['valley_price']:.2f} 元/kWh")
    print(f"  平时 (其他时段):    {economic_agent.params['flat_price']:.2f} 元/kWh")
    print(f"  峰时 (08-12,18-22): {economic_agent.params['peak_price']:.2f} 元/kWh")

    # 显示电价曲线图示
    print("\n  电价曲线:")
    print("  " + "─" * 50)
    for h in range(24):
        price = price_curve[h]
        bar_len = int((price - 0.3) / 0.1)
        period = "谷" if h in valley_hours else ("峰" if h in peak_hours else "平")
        print(f"  {h:02d}:00 [{period}] {price:.2f} | {'█' * bar_len}")

    # 显示优化参数
    print_section("3.2 经济优化参数")
    params = economic_agent.params
    print(f"\n  启停成本:")
    print(f"    - 启动成本: {params['start_cost']:.0f} 元/次")
    print(f"    - 停机成本: {params['stop_cost']:.0f} 元/次")

    print(f"\n  优化权重:")
    print(f"    - 能耗成本: {params['energy_cost_weight']:.0%}")
    print(f"    - 启停成本: {params['start_stop_weight']:.0%}")
    print(f"    - 效率权重: {params['efficiency_weight']:.0%}")
    print(f"    - 寿命均衡: {params['lifetime_weight']:.0%}")

    # 运行优化
    print_section("3.3 24小时调度优化")

    # 触发优化
    economic_agent.last_optimization_time = 0  # 强制重新优化
    system.step({'hour': 12, 'target_flow': 15.0})

    schedule = economic_agent.optimal_schedule
    if schedule:
        print(f"\n  优化结果 (目标流量: 15.0 m³/s):")
        print(f"\n  {'时刻':<6} {'电价':<8} {'建议台数':<10} {'预估功率':<12} {'能耗成本':<10}")
        print(f"  {'-'*50}")

        total_cost = 0.0
        for hour_data in schedule[:6]:  # 显示前6小时
            hour = hour_data['hour']
            price = hour_data['electricity_price']

            # 汇总各站
            total_pumps = 0
            total_power = 0.0
            total_hourly_cost = 0.0

            for station_id, station_data in hour_data['stations'].items():
                total_pumps += station_data['optimal_pump_count']
                total_power += station_data['expected_power']
                total_hourly_cost += station_data['energy_cost']

            total_cost += total_hourly_cost
            print(f"  {hour:02d}:00  {price:.2f}元    {total_pumps}台       "
                  f"{total_power:.0f}kW      {total_hourly_cost:.0f}元")

        print(f"  {'-'*50}")
        print(f"  ... (共{len(schedule)}小时调度计划)")

        # 成本汇总
        costs = economic_agent._calculate_total_cost(schedule)
        print(f"\n  24小时成本预估:")
        print(f"    - 能耗成本: {costs['energy_cost']:.0f} 元")
        print(f"    - 启停成本: {costs['start_stop_cost']:.0f} 元")
        print(f"    - 总成本:   {costs['total_cost']:.0f} 元")


def demo_coordination(system: PumpGroupMultiAgentSystem):
    """演示协调智能体功能"""
    print_header("4. 泵站群协调智能体演示")

    coordinator = system.coordinator

    print_section("4.1 安全与经济目标融合")
    print("\n  协调原则:")
    print("    1. 安全优先 - L1层决策优先于L3层")
    print("    2. 经济服从安全 - 安全约束下最优经济运行")
    print("    3. 冲突仲裁 - 同一执行器的冲突动作取高优先级")

    print_section("4.2 级联泵站协调")

    # 显示波传播时间
    print("\n  波传播时间配置:")
    for (src, dst), travel_time in system.pump_group_state.wave_propagation_time.items():
        src_name = system.pump_group_state.stations[src].station_name
        dst_name = system.pump_group_state.stations[dst].station_name
        print(f"    {src_name} -> {dst_name}: {travel_time/60:.1f} 分钟")

    print_section("4.3 协调决策模拟")

    # 模拟不同场景
    scenarios = [
        {'name': '正常运行', 'forebay': 3.0, 'hour': 12},
        {'name': '谷时经济运行', 'forebay': 3.0, 'hour': 3},
        {'name': '峰时节能运行', 'forebay': 3.0, 'hour': 19},
        {'name': '安全优先', 'forebay': 1.0, 'hour': 12},
    ]

    for scenario in scenarios:
        print(f"\n  [场景: {scenario['name']}]")
        print(f"    - 前池水位: {scenario['forebay']}m, 时刻: {scenario['hour']:02d}:00")

        # 设置状态
        for station in system.pump_group_state.stations.values():
            station.forebay_level = scenario['forebay']

        # 执行决策
        result = system.step({
            'hour': scenario['hour'],
            'target_flow': 15.0
        })

        mode = result.get('coordination_mode', 'normal')
        actions = result.get('actions', [])
        safety_rules = system.safety_agent.active_rules

        print(f"    - 协调模式: {mode}")
        print(f"    - 活跃安全规则: {len(safety_rules)}")
        print(f"    - 产生动作数: {len(actions)}")

        if mode == 'safety_override':
            print(f"    - [!] 安全覆盖生效!")


def demo_lifetime_balance(system: PumpGroupMultiAgentSystem):
    """演示设备寿命均衡功能"""
    print_header("5. 设备寿命均衡演示")

    economic_agent = system.economic_agent

    # 模拟不同运行时长的泵
    print_section("5.1 设置泵运行时长")

    station = list(system.pump_group_state.stations.values())[0]
    run_hours = [1000, 2500, 1500, 2000]

    for i, (pump_id, pump) in enumerate(station.pumps.items()):
        pump.run_hours = run_hours[i] if i < len(run_hours) else 1500
        print(f"  {pump_id}: {pump.run_hours} 小时")

    # 计算寿命均衡因子
    print_section("5.2 寿命均衡因子计算")

    factors = economic_agent._calculate_lifetime_balance_factor(station)
    avg_hours = sum(p.run_hours for p in station.pumps.values()) / len(station.pumps)

    print(f"\n  平均运行时长: {avg_hours:.0f} 小时")
    print(f"\n  {'泵ID':<8} {'运行时长':<12} {'均衡因子':<12} {'优先级'}")
    print(f"  {'-'*45}")

    for pump_id, factor in sorted(factors.items(), key=lambda x: x[1], reverse=True):
        pump = station.pumps[pump_id]
        priority = "高" if factor > 1.2 else ("低" if factor < 0.8 else "中")
        print(f"  {pump_id:<8} {pump.run_hours:<12.0f} {factor:<12.2f} {priority}")

    print(f"\n  说明: 均衡因子>1表示优先启动，<1表示优先停止")


def demo_full_simulation(system: PumpGroupMultiAgentSystem):
    """完整24小时仿真演示"""
    print_header("6. 24小时完整仿真演示")

    print_section("6.1 仿真设置")
    print("\n  - 仿真时长: 24小时")
    print("  - 目标流量: 15 m³/s")
    print("  - 电价模式: 峰谷电价")
    print("  - 初始前池水位: 3.0m")

    # 重置系统
    system.reset()
    for station in system.pump_group_state.stations.values():
        station.forebay_level = 3.0
        station.running_pump_count = 2

    print_section("6.2 仿真执行")

    hourly_stats = []

    for hour in range(24):
        # 模拟前池水位波动
        for station in system.pump_group_state.stations.values():
            # 简单的水位模型: 随机波动
            level_change = np.random.uniform(-0.2, 0.2)
            station.forebay_level = np.clip(
                station.forebay_level + level_change,
                1.5, 4.5
            )

        # 执行系统步进
        result = system.step({
            'hour': hour,
            'target_flow': 15.0
        })

        # 记录统计
        price = system.economic_agent._get_current_price(hour)
        hourly_stats.append({
            'hour': hour,
            'price': price,
            'mode': result.get('coordination_mode', 'normal'),
            'actions': len(result.get('actions', [])),
            'safety_rules': len(system.safety_agent.active_rules)
        })

        # 每6小时输出一次
        if hour % 6 == 0 or hour == 23:
            avg_level = np.mean([s.forebay_level for s in system.pump_group_state.stations.values()])
            print(f"  {hour:02d}:00 | 电价:{price:.2f} | 前池:{avg_level:.2f}m | "
                  f"模式:{result.get('coordination_mode', 'normal')[:8]:<8} | "
                  f"动作:{len(result.get('actions', []))}")

    print_section("6.3 仿真统计")

    status = system.get_system_status()

    print(f"\n  总周期数: {status['cycle_count']}")
    print(f"  总动作数: {status['total_actions']}")
    print(f"  安全覆盖次数: {status['safety_overrides']}")

    # 安全规则触发统计
    safety_status = status['safety_status']
    triggered_rules = [r for r in system.safety_agent.rules.values()
                      if r.trigger_count > 0]
    print(f"\n  安全规则触发统计:")
    if triggered_rules:
        for rule in sorted(triggered_rules, key=lambda x: x.trigger_count, reverse=True)[:5]:
            print(f"    - {rule.rule_id}: {rule.trigger_count}次")
    else:
        print(f"    - 无规则触发 (系统运行平稳)")

    # 经济优化摘要
    economic_status = status['economic_status']
    print(f"\n  经济优化摘要:")
    print(f"    - 24小时总成本: {economic_status.get('total_24h_cost', 0):.0f} 元")
    print(f"    - 能耗成本: {economic_status.get('energy_cost', 0):.0f} 元")
    print(f"    - 启停成本: {economic_status.get('start_stop_cost', 0):.0f} 元")


def main():
    """主函数"""
    print("\n")
    print("=" * 80)
    print("  泵站群多智能体系统 - 安全与经济运行演示")
    print("  Pump Group Multi-Agent System Demo")
    print("=" * 80)
    print("\n  版本: v1.0")
    print("  功能: 安全智能体(L1) + 经济智能体(L3) + 协调智能体")
    print("  目标: 融合到多智能体体系，实现泵站群安全与经济运行")

    # 1. 基本功能演示
    system = demo_basic_system()

    # 2. 安全智能体演示
    demo_safety_agent(system)

    # 3. 经济优化智能体演示
    demo_economic_agent(system)

    # 4. 协调智能体演示
    demo_coordination(system)

    # 5. 设备寿命均衡演示
    demo_lifetime_balance(system)

    # 6. 完整仿真演示
    demo_full_simulation(system)

    # 结论
    print_header("总结", "=")
    print("\n  泵站群多智能体系统已完整实现并融合到现有体系:")
    print()
    print("  [L1层 - 安全智能体]")
    print("    - 13条安全规则 (紧急/保护/联锁/限幅)")
    print("    - 前池水位保护、启停约束、汽蚀/过载/振动/温度保护")
    print("    - 毫秒级响应，硬约束执行")
    print()
    print("  [L3层 - 经济智能体]")
    print("    - 峰谷电价优化 (谷时0.35/平时0.55/峰时0.85 元/kWh)")
    print("    - 最优运行台数选择 (基于效率曲线)")
    print("    - 启停成本考虑 (50元/次启动, 30元/次停机)")
    print("    - 设备寿命均衡 (运行时长均匀分配)")
    print()
    print("  [协调智能体]")
    print("    - 安全优先原则 (L1覆盖L3)")
    print("    - 级联泵站协调 (波传播时间预测)")
    print("    - 冲突仲裁机制")
    print()
    print("  系统版本: v3.8.0")
    print()


if __name__ == "__main__":
    main()
