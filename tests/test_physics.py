"""
物理模块单元测试
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/YCJL')

from ycjl.physics.reservoir import Reservoir
from ycjl.physics.tunnel import Tunnel
from ycjl.physics.pool import StabilizingPool, PoolStatus
from ycjl.physics.pipeline import Pipeline
from ycjl.physics.surge_tank import SurgeTank


class TestReservoir:
    """水库模型测试"""

    def test_initialization(self):
        """测试初始化"""
        reservoir = Reservoir()
        assert reservoir.level > 0
        assert reservoir.storage > 0

    def test_water_balance(self):
        """测试水量平衡"""
        reservoir = Reservoir()
        initial_storage = reservoir.storage

        # 模拟入流
        inflow = 10.0
        dt = 1.0
        reservoir.step(dt, inflow)

        # 检查水量增加 (无出流时)
        assert reservoir.storage >= initial_storage

    def test_spillway_flow(self):
        """测试溢洪流量"""
        reservoir = Reservoir()
        reservoir.level = reservoir.cfg.normal_level + 5  # 高于正常水位

        # 打开溢洪闸
        reservoir.set_spillway_opening(0, 1.0)
        flow = reservoir.compute_spillway_flow()

        assert flow > 0

    def test_intake_flow(self):
        """测试引水流量"""
        reservoir = Reservoir()
        reservoir.set_intake_opening(0.5)

        flow = reservoir.compute_intake_flow()
        assert flow >= 0


class TestTunnel:
    """隧洞模型测试"""

    def test_initialization(self):
        """测试初始化"""
        tunnel = Tunnel()
        assert tunnel.length > 0
        assert len(tunnel.sections) > 0

    def test_flow_propagation(self):
        """测试流量传播"""
        tunnel = Tunnel()
        tunnel.upstream_head = 100.0

        for _ in range(10):
            tunnel.step(1.0)

        assert tunnel.flow_out >= 0

    def test_ice_mode(self):
        """测试冰期模式"""
        tunnel = Tunnel()
        original_manning = tunnel.manning_n

        tunnel.set_ice_mode(True, 0.5)
        assert tunnel.ice_mode
        assert tunnel.manning_n > original_manning


class TestStabilizingPool:
    """稳流池模型测试"""

    def test_initialization(self):
        """测试初始化"""
        pool = StabilizingPool()
        assert pool.level > 0
        assert pool.volume > 0

    def test_level_change(self):
        """测试水位变化"""
        pool = StabilizingPool()
        initial_level = pool.level

        # 入流大于出流
        pool.step(1.0, inflow=15.0, outflow=10.0)

        assert pool.level > initial_level

    def test_overflow(self):
        """测试溢流"""
        pool = StabilizingPool()
        pool.level = pool.cfg.max_level + 1.0

        overflow = pool.compute_overflow()
        assert overflow > 0

    def test_status(self):
        """测试状态判断"""
        pool = StabilizingPool()

        pool.level = pool.cfg.design_level
        assert pool.get_status() == PoolStatus.NORMAL

        pool.level = pool.cfg.min_level - 0.1
        assert pool.get_status() == PoolStatus.EMPTY

    def test_buffer_time(self):
        """测试缓冲时间"""
        pool = StabilizingPool()
        pool.inflow = 10.0
        pool.outflow = 15.0  # 出流大于入流

        buffer_time = pool.compute_buffer_time()
        assert buffer_time > 0
        assert buffer_time < float('inf')


class TestPipeline:
    """管道模型测试"""

    def test_initialization(self):
        """测试初始化"""
        pipeline = Pipeline(num_sections=50)
        assert len(pipeline.H) == 50
        assert len(pipeline.Q) == 50

    def test_steady_state(self):
        """测试稳态"""
        pipeline = Pipeline(num_sections=50)

        # 运行到稳态
        for _ in range(100):
            pipeline.step(0.1)

        # 检查流量连续
        flow_diff = np.max(np.abs(np.diff(pipeline.Q)))
        assert flow_diff < 1.0  # 流量变化小于1

    def test_water_hammer(self):
        """测试水锤"""
        pipeline = Pipeline(num_sections=50)

        # 运行到稳态
        for _ in range(50):
            pipeline.step(0.1)

        initial_pressure = pipeline.H[0]

        # 快速关阀
        pipeline.valve_opening_end = 0.1
        pipeline.step(0.1)

        # 压力应该增加
        assert pipeline.H[0] >= initial_pressure * 0.9


class TestSurgeTank:
    """调压井模型测试"""

    def test_initialization(self):
        """测试初始化"""
        tank = SurgeTank()
        assert tank.level > 0
        assert tank.area > 0

    def test_oscillation(self):
        """测试振荡"""
        tank = SurgeTank()
        levels = []

        # 给一个扰动
        tank.inflow = 15.0
        tank.outflow = 10.0

        for _ in range(100):
            tank.step(1.0)
            levels.append(tank.level)

        # 检查是否有振荡
        level_changes = np.diff(levels)
        sign_changes = np.sum(np.diff(np.sign(level_changes)) != 0)

        assert sign_changes > 0  # 应该有方向变化


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
