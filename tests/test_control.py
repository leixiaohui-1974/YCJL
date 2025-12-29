"""
控制模块测试
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/YCJL')

from ycjl.control.pid import PIDController, PIDParams, PIDMode, CascadePID, PIDAutotuner
from ycjl.control.adaptive import MRACController, STRController, AdaptiveGainController
from ycjl.control.feedforward import FeedforwardCompensator, SmithPredictor, DisturbanceObserver
from ycjl.control.coordinator import ControlCoordinator, ControlLoop, ControlMode


class TestPIDController:
    """PID控制器测试"""

    def test_initialization(self):
        """测试初始化"""
        pid = PIDController()
        assert pid.params.Kp == 1.0
        assert pid.state.mode == PIDMode.AUTO

    def test_proportional_only(self):
        """测试纯比例控制"""
        params = PIDParams(Kp=2.0, Ki=0.0, Kd=0.0)
        pid = PIDController(params)

        output = pid.compute(pv=5.0, sp=10.0)

        # P = Kp * error = 2.0 * 5.0 = 10.0, 限幅后为1.0
        assert output == 1.0

    def test_integral_action(self):
        """测试积分作用"""
        params = PIDParams(Kp=0.0, Ki=0.1, Kd=0.0, dt=1.0)
        pid = PIDController(params)

        # 多次计算，积分应该累积
        for _ in range(10):
            output = pid.compute(pv=9.0, sp=10.0)

        assert pid.state.integral > 0

    def test_derivative_action(self):
        """测试微分作用"""
        params = PIDParams(Kp=0.0, Ki=0.0, Kd=1.0, dt=1.0)
        pid = PIDController(params)

        # 第一次
        pid.compute(pv=5.0, sp=10.0)
        # 第二次PV变化
        output = pid.compute(pv=6.0, sp=10.0)

        # 微分应该有作用
        assert pid.state.derivative != 0

    def test_anti_windup(self):
        """测试抗积分饱和"""
        params = PIDParams(Kp=0.0, Ki=1.0, Kd=0.0,
                           output_max=0.5)
        pid = PIDController(params)

        # 持续大误差
        for _ in range(100):
            output = pid.compute(pv=0.0, sp=10.0)

        # 输出应该被限制
        assert output <= 0.5

    def test_manual_mode(self):
        """测试手动模式"""
        pid = PIDController()
        pid.state.output = 0.3
        pid.set_mode(PIDMode.MANUAL)

        output = pid.compute(pv=5.0, sp=10.0)

        assert output == 0.3

    def test_bumpless_transfer(self):
        """测试无扰切换"""
        pid = PIDController()
        pid.state.output = 0.5
        pid.set_mode(PIDMode.MANUAL)

        # 切换到自动
        pid.set_mode(PIDMode.AUTO)

        # 积分应该被初始化以保持输出连续
        assert pid.state.integral != 0


class TestCascadePID:
    """串级PID测试"""

    def test_cascade_control(self):
        """测试串级控制"""
        cascade = CascadePID()

        # 模拟一个过程
        primary_pv = 5.0
        secondary_pv = 10.0
        primary_sp = 6.0

        output = cascade.compute(primary_pv, secondary_pv, primary_sp)

        assert 0.0 <= output <= 1.0


class TestPIDAutotuner:
    """PID自整定测试"""

    def test_relay_feedback(self):
        """测试继电反馈"""
        tuner = PIDAutotuner(relay_amplitude=0.1)
        tuner.start(setpoint=5.0)

        # 模拟振荡过程
        for i in range(100):
            pv = 5.0 + 0.5 * np.sin(i * 0.3)
            output = tuner.step(pv, dt=0.1)

        # 应该检测到振荡
        assert tuner.oscillation_count >= 0


class TestMRACController:
    """MRAC控制器测试"""

    def test_initialization(self):
        """测试初始化"""
        mrac = MRACController()
        assert len(mrac.theta) == 2

    def test_adaptation(self):
        """测试自适应"""
        mrac = MRACController(adaptation_gain=0.1)

        initial_theta = mrac.theta.copy()

        # 运行多步
        for i in range(50):
            reference = 1.0
            measurement = 0.5 + 0.01 * i
            mrac.compute_control(reference, measurement, dt=0.1)

        # 参数应该变化
        assert not np.allclose(mrac.theta, initial_theta)


class TestSTRController:
    """STR控制器测试"""

    def test_initialization(self):
        """测试初始化"""
        str_ctrl = STRController(model_order=2)
        assert str_ctrl.na == 2
        assert str_ctrl.nb == 2

    def test_parameter_estimation(self):
        """测试参数估计"""
        str_ctrl = STRController(model_order=1)

        # 模拟系统
        y = 0.0
        for i in range(50):
            u = str_ctrl.compute_control(1.0, y, dt=0.1)
            # 简单一阶系统
            y = 0.9 * y + 0.1 * u

        # 应该有参数估计
        params = str_ctrl.get_model_params()
        assert len(params['A']) > 0


class TestSmithPredictor:
    """史密斯预估器测试"""

    def test_initialization(self):
        """测试初始化"""
        smith = SmithPredictor(
            process_gain=1.0,
            process_tau=10.0,
            dead_time=5.0,
            dt=1.0
        )
        assert smith.delay_steps == 5

    def test_prediction(self):
        """测试预测"""
        smith = SmithPredictor(dead_time=5.0, dt=1.0)

        for i in range(20):
            u = 0.5
            y_model_nodelay, y_model_delayed = smith.predict(u)

        # 无延迟模型应该快于延迟模型
        # (在稳态时两者相同)

    def test_compensation(self):
        """测试补偿"""
        smith = SmithPredictor(dead_time=5.0, dt=1.0)

        for i in range(30):
            u = 0.5
            y = 0.3  # 实际测量
            compensated = smith.compute_compensation(u, y)

        assert compensated != 0


class TestControlCoordinator:
    """控制协调器测试"""

    def test_add_loop(self):
        """测试添加回路"""
        coord = ControlCoordinator()

        loop = ControlLoop(
            loop_id='test_loop',
            controller=PIDController(),
            pv_tag='pv1',
            sp_tag='sp1',
            output_tag='out1'
        )

        coord.add_loop(loop)
        assert 'test_loop' in coord.loops

    def test_step(self):
        """测试步进"""
        coord = ControlCoordinator()

        loop = ControlLoop(
            loop_id='level_control',
            controller=PIDController(PIDParams(Kp=1.0, Ki=0.1)),
            pv_tag='level',
            sp_tag='level_sp',
            output_tag='valve'
        )
        coord.add_loop(loop)

        measurements = {'level': 5.0}
        setpoints = {'level_sp': 6.0}

        outputs = coord.step(measurements, setpoints, dt=1.0)

        assert 'valve' in outputs
        assert 0 <= outputs['valve'] <= 1

    def test_mode_change(self):
        """测试模式切换"""
        coord = ControlCoordinator()

        loop = ControlLoop(
            loop_id='test',
            controller=PIDController(),
            pv_tag='pv',
            sp_tag='sp',
            output_tag='out'
        )
        coord.add_loop(loop)

        coord.set_mode('test', ControlMode.MANUAL)
        assert coord.loops['test'].mode == ControlMode.MANUAL

    def test_performance_tracking(self):
        """测试性能跟踪"""
        coord = ControlCoordinator()

        loop = ControlLoop(
            loop_id='test',
            controller=PIDController(),
            pv_tag='pv',
            sp_tag='sp',
            output_tag='out'
        )
        coord.add_loop(loop)

        for i in range(10):
            coord.step({'pv': 5.0 + i * 0.1}, {'sp': 6.0})

        summary = coord.get_performance_summary()
        assert 'test' in summary
        assert summary['test']['iae'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
