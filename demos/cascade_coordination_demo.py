#!/usr/bin/env python3
"""
级联泵站协调调度演示程序
========================

演示增强的级联泵站协调调度算法:
1. 波传播预测 - 浅水波速估算
2. 前池水位控制 - 物质平衡计算
3. 智能接力调度 - 上下游配合
4. 协调指令生成 - 自动调度

使用方法:
    python demos/cascade_coordination_demo.py
"""

import sys
import os
import time
import math

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ycjl.agents.cascade_coordination import (
    create_cascade_coordinator,
    CascadeCoordinator,
    CanalSection,
    Forebay,
    CoordinationStrategy,
    WaveType,
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


def demo_wave_propagation(coordinator: CascadeCoordinator):
    """演示波传播预测"""
    print_header("1. 波传播预测演示")

    print_section("1.1 渠道网络参数")

    wave_info = coordinator.get_wave_propagation_info()

    print(f"\n  [渠道段参数]")
    for canal_id, info in wave_info['canal_sections'].items():
        print(f"    {canal_id}:")
        print(f"      - 长度: {info['length']:.0f} m")
        print(f"      - 波速: {info['wave_celerity']:.2f} m/s")
        print(f"      - 传播时间: {info['travel_time']:.0f} s ({info['travel_time']/60:.1f} min)")

    print(f"\n  [站间传播时间]")
    for route, info in wave_info['estimated_travel_times'].items():
        print(f"    {route}:")
        print(f"      - 距离: {info['distance']:.0f} m")
        print(f"      - 传播时间: {info['travel_time_min']:.1f} min")

    print_section("1.2 浅水波速计算原理")
    print("""
  浅水波速公式: c = sqrt(g * h)

  其中:
    c = 波速 (m/s)
    g = 重力加速度 (9.81 m/s²)
    h = 水深 (m)

  示例计算:
    水深 h = 2.5 m
    波速 c = sqrt(9.81 * 2.5) = 4.95 m/s

  考虑流速影响:
    顺流: 有效波速 = c + v
    逆流: 有效波速 = c - v
""")

    print_section("1.3 波传播模拟")

    # 模拟上游泵站启动
    print(f"\n  [模拟场景] 屯佃泵站启动泵组，流量增加 5 m³/s")

    commands = coordinator.on_upstream_action(
        station_id='tundian',
        action='start_pump',
        flow_change=5.0
    )

    print(f"\n  [波传播预测]")
    for wave_id, wave in coordinator.wave_predictor.active_waves.items():
        print(f"    波ID: {wave_id}")
        print(f"    类型: {wave.wave_type.name}")
        print(f"    振幅: {wave.amplitude} m³/s")
        print(f"\n    预测到达时间:")
        for station, arrival in wave.arrival_times.items():
            relative_time = arrival - wave.start_time
            print(f"      {station}: +{relative_time/60:.1f} min")

    print(f"\n  [生成的协调指令] 共 {len(commands)} 条")
    for cmd in commands[:3]:  # 显示前3条
        execute_delay = cmd.execute_time - time.time()
        print(f"    - {cmd.target_station}: {cmd.action}")
        print(f"      目标值: {cmd.target_value:.2f} m³/s")
        print(f"      执行时间: +{execute_delay/60:.1f} min")
        print(f"      原因: {cmd.reason}")


def demo_forebay_control(coordinator: CascadeCoordinator):
    """演示前池水位控制"""
    print_header("2. 前池水位控制演示")

    print_section("2.1 物质平衡原理")
    print("""
  前池水位变化公式: dH/dt = (Qin - Qout) / A

  其中:
    dH/dt = 水位变化率 (m/s)
    Qin = 入流量 (m³/s)
    Qout = 出流量 (m³/s)
    A = 前池面积 (m²)

  示例计算:
    入流量 Qin = 5.0 m³/s
    出流量 Qout = 4.5 m³/s
    前池面积 A = 2000 m²
    水位变化率 = (5.0 - 4.5) / 2000 = 0.00025 m/s = 0.9 m/h
""")

    print_section("2.2 前池状态")

    for station_id, controller in coordinator.forebay_controllers.items():
        forebay = controller.forebay
        print(f"\n  [{station_id}]")
        print(f"    - 面积: {forebay.area} m²")
        print(f"    - 水位范围: {forebay.min_level} ~ {forebay.max_level} m")
        print(f"    - 报警范围: {forebay.alarm_low} ~ {forebay.alarm_high} m")
        print(f"    - 当前水位: {forebay.level} m")
        print(f"    - 当前蓄量: {forebay.storage_volume:.0f} m³")

    print_section("2.3 水位预测模拟")

    # 选择一个前池进行模拟
    station_id = 'tundian'
    controller = coordinator.forebay_controllers[station_id]
    forebay = controller.forebay

    scenarios = [
        {'inflow': 5.0, 'outflow': 5.0, 'name': '平衡状态'},
        {'inflow': 6.0, 'outflow': 5.0, 'name': '入流大于出流'},
        {'inflow': 4.0, 'outflow': 5.0, 'name': '出流大于入流'},
    ]

    print(f"\n  [屯佃泵站前池水位预测] 初始水位: {forebay.level} m")

    for scenario in scenarios:
        predicted_1h = controller.predict_level(
            forebay.level,
            scenario['inflow'],
            scenario['outflow'],
            3600  # 1小时
        )
        predicted_4h = controller.predict_level(
            forebay.level,
            scenario['inflow'],
            scenario['outflow'],
            14400  # 4小时
        )

        print(f"\n    场景: {scenario['name']}")
        print(f"      入流: {scenario['inflow']} m³/s, 出流: {scenario['outflow']} m³/s")
        print(f"      1小时后水位: {predicted_1h:.2f} m")
        print(f"      4小时后水位: {predicted_4h:.2f} m")

    print_section("2.4 缓冲时间估算")

    print(f"\n  [各站前池缓冲时间估算] (入流5m³/s, 出流4.5m³/s)")

    for station_id, controller in coordinator.forebay_controllers.items():
        forebay = controller.forebay
        time_to_high, time_to_low = controller.estimate_buffer_time(
            forebay.level, 5.0, 4.5
        )

        print(f"\n    {station_id}:")
        print(f"      当前水位: {forebay.level} m")
        if time_to_high > 0:
            print(f"      到高水位报警: {time_to_high/3600:.1f} 小时")
        else:
            print(f"      到高水位报警: 不会到达 (出流>入流)")
        if time_to_low > 0:
            print(f"      到低水位报警: {time_to_low/3600:.1f} 小时")
        else:
            print(f"      到低水位报警: 不会到达 (入流>出流)")


def demo_relay_coordination(coordinator: CascadeCoordinator):
    """演示智能接力调度"""
    print_header("3. 智能接力调度演示")

    print_section("3.1 接力调度原理")
    print("""
  智能接力策略:
  1. 上游泵站启动/停止产生流量变化
  2. 变化以波形式向下游传播
  3. 下游泵站提前调整，配合波到达
  4. 保持前池水位稳定

  时序控制:
  - 提前量: 考虑泵启动时间 (60s)
  - 安全裕度: 流量多预留10%
  - 协调时域: 1小时滚动优化
""")

    print_section("3.2 级联调度模拟")

    # 重置协调器
    coordinator.command_queue.clear()
    coordinator.wave_predictor.active_waves.clear()

    # 模拟一系列操作
    operations = [
        ('tundian', 'start_pump', 5.0, '启动2台泵'),
        ('tundian', 'adjust_flow', 2.0, '增加流量'),
    ]

    for station_id, action, flow_change, desc in operations:
        print(f"\n  [操作] {station_id}: {desc} (流量变化 {flow_change:+.1f} m³/s)")

        commands = coordinator.on_upstream_action(
            station_id=station_id,
            action=action,
            flow_change=flow_change
        )

        if commands:
            print(f"    生成 {len(commands)} 条协调指令:")
            for cmd in commands[:3]:
                execute_delay = cmd.execute_time - time.time()
                print(f"      → {cmd.target_station}: {cmd.action} @ +{execute_delay/60:.1f}min")

    print_section("3.3 协调状态")

    status = coordinator.get_coordination_status()
    print(f"\n  协调策略: {status['strategy']}")
    print(f"  站点数量: {status['station_count']}")
    print(f"  站点顺序: {' → '.join(status['station_order'])}")
    print(f"  活跃波数: {status['active_waves']}")
    print(f"  待执行指令: {status['pending_commands']}")


def demo_optimal_schedule(coordinator: CascadeCoordinator):
    """演示最优调度计划"""
    print_header("4. 最优调度计划演示")

    print_section("4.1 4小时调度计划")

    schedule = coordinator.get_optimal_schedule(target_flow=15.0, horizon_hours=4)

    for hour_data in schedule:
        print(f"\n  [第 {hour_data['hour']} 小时]")
        for station_id, station_data in hour_data['stations'].items():
            status_icon = ""
            if station_data['status'] == 'high_risk':
                status_icon = " [!高水位风险]"
            elif station_data['status'] == 'low_risk':
                status_icon = " [!低水位风险]"

            print(f"    {station_id}: 目标流量 {station_data['target_flow']:.2f} m³/s"
                  f" | 前池 {station_data['forebay_level']:.2f}m{status_icon}")


def demo_coordination_summary():
    """演示总结"""
    print_header("总结", "=")
    print("""
  级联泵站协调调度算法已实现:

  [波传播预测]
    - 浅水波速: c = sqrt(g*h) ≈ 5 m/s (h=2.5m)
    - 传播时间: 12km渠道 ≈ 40分钟
    - 波衰减: 按距离指数衰减

  [前池水位控制]
    - 物质平衡: dH/dt = (Qin - Qout) / A
    - PID控制: 维持目标水位
    - 缓冲估算: 预测高/低水位时间

  [智能接力调度]
    - 提前量: 60秒 (泵启动时间)
    - 安全裕度: 流量多10%
    - 协调指令: 自动生成下游动作

  [协调策略]
    - RELAY: 智能接力 (上游脉冲，下游配合)
    - PARALLEL: 并行运行 (同步调整)
    - SEQUENTIAL: 顺序调整 (逐级调整)
    - ADAPTIVE: 自适应 (根据工况选择)

  系统版本: v3.10.0
""")


def main():
    """主函数"""
    print("\n")
    print("=" * 80)
    print("  级联泵站协调调度演示")
    print("  Cascade Pump Station Coordination Demo")
    print("=" * 80)
    print("\n  版本: v1.0")
    print("  功能: 波传播预测 + 前池水位控制 + 智能接力调度")

    # 创建级联协调器
    print_header("0. 创建级联协调器")
    coordinator = create_cascade_coordinator()
    print(f"\n  协调器创建成功!")
    print(f"  - 站点数量: {len(coordinator.station_order)}")
    print(f"  - 站点顺序: {' → '.join(coordinator.station_order)}")

    # 1. 波传播预测
    demo_wave_propagation(coordinator)

    # 2. 前池水位控制
    demo_forebay_control(coordinator)

    # 3. 智能接力调度
    demo_relay_coordination(coordinator)

    # 4. 最优调度计划
    demo_optimal_schedule(coordinator)

    # 总结
    demo_coordination_summary()


if __name__ == "__main__":
    main()
