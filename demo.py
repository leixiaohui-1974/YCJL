#!/usr/bin/env python3
"""
引绰济辽智能输水系统演示
========================

演示全系统仿真能力:
1. 正常运行
2. 需水激增响应
3. 爆管检测与隔离
4. 冰期自适应运行
"""

import sys
sys.path.insert(0, '/home/user/YCJL')

from ycjl import (
    SimulationRunner,
    SimulationConfig,
    run_scenario_test,
    ScenarioType,
    WaterTransferPlant,
    MultiAgentSystem
)


def demo_normal_operation():
    """演示正常运行"""
    print("\n" + "="*60)
    print("演示1: 正常运行仿真")
    print("="*60)

    config = SimulationConfig(
        duration=300,  # 5分钟
        dt=1.0,
        enable_control=True,
        enable_detection=True,
        verbose=True,
        log_interval=60
    )

    runner = SimulationRunner(config)
    result = runner.run()

    print(f"\n仿真完成:")
    print(f"  - 总步数: {result.steps}")
    print(f"  - 控制动作数: {result.control_actions}")
    print(f"  - 平均流量: {result.metrics.get('flow_mean', 0):.2f} m³/s")
    print(f"  - 池水位均值: {result.metrics.get('pool_level_mean', 0):.2f} m")

    return result


def demo_demand_surge():
    """演示需水激增响应"""
    print("\n" + "="*60)
    print("演示2: 需水激增场景")
    print("="*60)

    result = run_scenario_test(
        ScenarioType.DEMAND_SURGE,
        duration=600,
        surge_factor=1.8
    )

    print(f"\n场景历史:")
    for time, scenario in result.scenario_history[:5]:
        print(f"  t={time:.0f}s: {scenario.name}")

    print(f"\n场景持续时间:")
    for scenario, duration in result.scenario_durations.items():
        if duration > 0:
            print(f"  {scenario.name}: {duration:.0f}s")

    return result


def demo_pipe_burst():
    """演示爆管检测与响应"""
    print("\n" + "="*60)
    print("演示3: 管道破裂场景")
    print("="*60)

    result = run_scenario_test(
        ScenarioType.PIPE_BURST,
        duration=600,
        leak_rate=0.15,
        location=0.5
    )

    print(f"\n安全干预次数: {result.safety_interventions}")
    print(f"控制动作总数: {result.control_actions}")

    return result


def demo_multi_agent():
    """演示多智能体协调"""
    print("\n" + "="*60)
    print("演示4: 多智能体系统")
    print("="*60)

    mas = MultiAgentSystem()
    plant = WaterTransferPlant()

    print("\n运行多智能体控制循环...")

    for i in range(30):
        # 获取系统状态
        state = plant.get_state_dict()

        # 多智能体决策
        result = mas.step(state)

        # 提取控制动作
        actions = result.get('all_actions', [])
        control = {a['actuator']: a['value'] for a in actions if 'actuator' in a}

        # 执行控制
        plant.step(1.0, control)

        if i % 10 == 0:
            status = mas.get_system_status()
            print(f"  步 {i}: L1规则={len(status['l1_active_rules'])}, "
                  f"场景={status['l3_scenario']}")

    print("\n系统状态:")
    status = mas.get_system_status()
    print(f"  - 循环次数: {status['cycle_count']}")
    print(f"  - ADMM状态: {status['admm_status']}")
    print(f"  - 通信消息数: {status['comm_stats']['total_messages']}")


def main():
    """主函数"""
    print("\n" + "#"*60)
    print("#  引绰济辽智能输水系统 - 仿真演示")
    print("#"*60)

    # 演示1: 正常运行
    demo_normal_operation()

    # 演示2: 需水激增
    demo_demand_surge()

    # 演示3: 爆管场景
    demo_pipe_burst()

    # 演示4: 多智能体
    demo_multi_agent()

    print("\n" + "="*60)
    print("所有演示完成!")
    print("="*60)


if __name__ == '__main__':
    main()
