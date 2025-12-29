"""
PID控制器
=========

工业级PID控制实现:
- 增量式/位置式PID
- 抗积分饱和
- 微分滤波
- 死区处理
- 自动整定
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque


class PIDMode(Enum):
    """PID模式"""
    MANUAL = auto()
    AUTO = auto()
    CASCADE = auto()


class AntiWindup(Enum):
    """抗积分饱和方法"""
    NONE = auto()
    CLAMPING = auto()      # 限幅
    BACK_CALCULATION = auto()  # 反算
    CONDITIONAL = auto()    # 条件积分


@dataclass
class PIDParams:
    """PID参数"""
    Kp: float = 1.0        # 比例增益
    Ki: float = 0.1        # 积分增益
    Kd: float = 0.01       # 微分增益

    # 输出限制
    output_min: float = 0.0
    output_max: float = 1.0

    # 积分限制
    integral_min: float = -10.0
    integral_max: float = 10.0

    # 微分滤波系数
    derivative_filter: float = 0.1

    # 死区
    deadband: float = 0.0

    # 采样周期
    dt: float = 1.0

    # 抗饱和
    anti_windup: AntiWindup = AntiWindup.BACK_CALCULATION
    Kb: float = 1.0  # 反算增益


@dataclass
class PIDState:
    """PID状态"""
    setpoint: float = 0.0
    process_value: float = 0.0
    error: float = 0.0
    output: float = 0.0

    integral: float = 0.0
    derivative: float = 0.0
    last_error: float = 0.0
    last_pv: float = 0.0
    last_output: float = 0.0

    is_saturated: bool = False
    mode: PIDMode = PIDMode.AUTO


class PIDController:
    """
    工业级PID控制器

    特性:
    - 增量式/位置式可选
    - 多种抗积分饱和策略
    - 微分前馈/微分反馈
    - 无扰切换
    - 手动/自动/串级模式
    """

    def __init__(self, params: PIDParams = None, name: str = "PID"):
        self.params = params or PIDParams()
        self.name = name

        # 状态
        self.state = PIDState()

        # 微分滤波器状态
        self.filtered_derivative = 0.0

        # 历史记录
        self.history: deque = deque(maxlen=1000)

        # 增量式模式
        self.incremental_mode = False

        # 微分作用于误差还是PV
        self.derivative_on_pv = True  # 推荐: 对PV微分避免设定点突变

    def set_params(self, Kp: float = None, Ki: float = None, Kd: float = None):
        """设置PID参数"""
        if Kp is not None:
            self.params.Kp = Kp
        if Ki is not None:
            self.params.Ki = Ki
        if Kd is not None:
            self.params.Kd = Kd

    def set_setpoint(self, sp: float):
        """设置设定点"""
        self.state.setpoint = sp

    def set_mode(self, mode: PIDMode):
        """设置控制模式"""
        if mode != self.state.mode:
            # 无扰切换
            if mode == PIDMode.AUTO:
                # 手动->自动: 初始化积分项
                self.state.integral = self.state.output / max(self.params.Ki, 1e-6)
            self.state.mode = mode

    def reset(self):
        """重置控制器"""
        self.state = PIDState()
        self.filtered_derivative = 0.0
        self.history.clear()

    def compute(self, pv: float, sp: float = None, dt: float = None) -> float:
        """
        计算控制输出

        Parameters:
            pv: 过程变量(测量值)
            sp: 设定点(可选,使用已设置的值)
            dt: 采样周期(可选)

        Returns:
            控制输出
        """
        if sp is not None:
            self.state.setpoint = sp
        if dt is not None:
            self.params.dt = dt

        # 更新状态
        self.state.process_value = pv

        # 计算误差
        error = self.state.setpoint - pv

        # 死区处理
        if abs(error) < self.params.deadband:
            error = 0.0

        self.state.error = error

        # 手动模式
        if self.state.mode == PIDMode.MANUAL:
            return self.state.output

        # 比例项
        P = self.params.Kp * error

        # 积分项 (带抗饱和)
        I = self._compute_integral(error)

        # 微分项 (带滤波)
        D = self._compute_derivative(pv, error)

        # 计算输出
        output_raw = P + I + D

        # 输出限幅
        output = np.clip(output_raw, self.params.output_min, self.params.output_max)

        # 检查饱和
        self.state.is_saturated = (output != output_raw)

        # 抗饱和反算
        if self.params.anti_windup == AntiWindup.BACK_CALCULATION:
            if self.state.is_saturated:
                self.state.integral += self.params.Kb * (output - output_raw)

        # 更新状态
        self.state.output = output
        self.state.last_error = error
        self.state.last_pv = pv
        self.state.last_output = output

        # 记录历史
        self._record_history()

        return output

    def _compute_integral(self, error: float) -> float:
        """计算积分项"""
        dt = self.params.dt

        # 条件积分: 饱和时不积分
        if self.params.anti_windup == AntiWindup.CONDITIONAL:
            if self.state.is_saturated:
                return self.state.integral

        # 积分累加
        self.state.integral += self.params.Ki * error * dt

        # 限幅抗饱和
        if self.params.anti_windup == AntiWindup.CLAMPING:
            self.state.integral = np.clip(
                self.state.integral,
                self.params.integral_min,
                self.params.integral_max
            )

        return self.state.integral

    def _compute_derivative(self, pv: float, error: float) -> float:
        """计算微分项"""
        dt = self.params.dt

        if self.derivative_on_pv:
            # 对PV微分 (避免设定点突变导致的微分冲击)
            d_input = -(pv - self.state.last_pv) / dt
        else:
            # 对误差微分
            d_input = (error - self.state.last_error) / dt

        # 一阶低通滤波
        alpha = self.params.derivative_filter
        self.filtered_derivative = alpha * d_input + (1 - alpha) * self.filtered_derivative

        self.state.derivative = self.params.Kd * self.filtered_derivative

        return self.state.derivative

    def _record_history(self):
        """记录历史"""
        import time
        self.history.append({
            'timestamp': time.time(),
            'sp': self.state.setpoint,
            'pv': self.state.process_value,
            'error': self.state.error,
            'output': self.state.output,
            'integral': self.state.integral,
            'derivative': self.state.derivative
        })

    def get_status(self) -> Dict:
        """获取控制器状态"""
        return {
            'name': self.name,
            'mode': self.state.mode.name,
            'setpoint': self.state.setpoint,
            'pv': self.state.process_value,
            'error': self.state.error,
            'output': self.state.output,
            'is_saturated': self.state.is_saturated,
            'params': {
                'Kp': self.params.Kp,
                'Ki': self.params.Ki,
                'Kd': self.params.Kd
            }
        }


class CascadePID:
    """
    串级PID控制器

    主回路(外环) -> 副回路(内环)
    """

    def __init__(self,
                 primary_params: PIDParams = None,
                 secondary_params: PIDParams = None,
                 name: str = "CascadePID"):
        self.name = name

        # 主回路 (慢)
        self.primary = PIDController(
            primary_params or PIDParams(Kp=1.0, Ki=0.05, Kd=0.0),
            name=f"{name}_primary"
        )

        # 副回路 (快)
        self.secondary = PIDController(
            secondary_params or PIDParams(Kp=2.0, Ki=0.2, Kd=0.01),
            name=f"{name}_secondary"
        )

        # 副回路设定点限制
        self.secondary_sp_min = 0.0
        self.secondary_sp_max = 100.0

    def compute(self, primary_pv: float, secondary_pv: float,
                primary_sp: float, dt: float = 1.0) -> float:
        """
        计算串级控制输出

        Parameters:
            primary_pv: 主回路过程变量
            secondary_pv: 副回路过程变量
            primary_sp: 主回路设定点
            dt: 采样周期

        Returns:
            最终控制输出
        """
        # 主回路计算 -> 副回路设定点
        secondary_sp = self.primary.compute(primary_pv, primary_sp, dt)

        # 限制副回路设定点
        secondary_sp = np.clip(secondary_sp,
                               self.secondary_sp_min,
                               self.secondary_sp_max)

        # 副回路计算 -> 最终输出
        output = self.secondary.compute(secondary_pv, secondary_sp, dt)

        return output

    def set_mode(self, mode: PIDMode):
        """设置模式"""
        self.primary.set_mode(mode)
        self.secondary.set_mode(mode)

    def reset(self):
        """重置"""
        self.primary.reset()
        self.secondary.reset()


class PIDAutotuner:
    """
    PID自动整定器

    基于继电反馈法 (Relay Feedback)
    """

    def __init__(self, relay_amplitude: float = 0.1):
        self.relay_amplitude = relay_amplitude

        # 状态
        self.is_tuning = False
        self.oscillation_count = 0
        self.required_oscillations = 4

        # 记录
        self.peaks: List[Tuple[float, float]] = []  # (time, value)
        self.valleys: List[Tuple[float, float]] = []

        # 结果
        self.ultimate_gain = 0.0
        self.ultimate_period = 0.0

        # 上一个值
        self.last_pv = 0.0
        self.last_output = 0.0
        self.setpoint = 0.0
        self.time = 0.0

    def start(self, setpoint: float):
        """开始整定"""
        self.is_tuning = True
        self.oscillation_count = 0
        self.peaks.clear()
        self.valleys.clear()
        self.setpoint = setpoint
        self.time = 0.0

    def step(self, pv: float, dt: float = 1.0) -> float:
        """
        执行一步整定

        Parameters:
            pv: 过程变量
            dt: 时间步长

        Returns:
            控制输出
        """
        if not self.is_tuning:
            return 0.0

        self.time += dt

        # 继电反馈
        error = self.setpoint - pv

        if error > 0:
            output = self.relay_amplitude
        else:
            output = -self.relay_amplitude

        # 检测振荡
        self._detect_oscillation(pv)

        # 检查是否完成
        if self.oscillation_count >= self.required_oscillations:
            self._compute_parameters()
            self.is_tuning = False

        self.last_pv = pv
        self.last_output = output

        return output + 0.5  # 偏置到0.5

    def _detect_oscillation(self, pv: float):
        """检测振荡"""
        # 检测峰值
        if len(self.peaks) == 0 or (pv > self.last_pv and self.last_pv < self.peaks[-1][1]):
            if pv > self.setpoint:
                self.peaks.append((self.time, pv))

        # 检测谷值
        if len(self.valleys) == 0 or (pv < self.last_pv and self.last_pv > self.valleys[-1][1]):
            if pv < self.setpoint:
                self.valleys.append((self.time, pv))

        # 计数
        self.oscillation_count = min(len(self.peaks), len(self.valleys))

    def _compute_parameters(self):
        """计算整定参数"""
        if len(self.peaks) < 2 or len(self.valleys) < 2:
            return

        # 计算振幅
        peak_values = [p[1] for p in self.peaks[-3:]]
        valley_values = [v[1] for v in self.valleys[-3:]]
        amplitude = (np.mean(peak_values) - np.mean(valley_values)) / 2

        # 计算周期
        peak_times = [p[0] for p in self.peaks[-3:]]
        periods = np.diff(peak_times)
        self.ultimate_period = np.mean(periods)

        # 计算临界增益
        self.ultimate_gain = 4 * self.relay_amplitude / (np.pi * amplitude)

    def get_zn_params(self, method: str = "classic") -> PIDParams:
        """
        获取Ziegler-Nichols整定参数

        Parameters:
            method: "classic", "pessen", "some_overshoot", "no_overshoot"

        Returns:
            PIDParams
        """
        Ku = self.ultimate_gain
        Tu = self.ultimate_period

        if method == "classic":
            Kp = 0.6 * Ku
            Ki = 1.2 * Ku / Tu
            Kd = 0.075 * Ku * Tu
        elif method == "pessen":
            Kp = 0.7 * Ku
            Ki = 1.75 * Ku / Tu
            Kd = 0.105 * Ku * Tu
        elif method == "some_overshoot":
            Kp = 0.33 * Ku
            Ki = 0.66 * Ku / Tu
            Kd = 0.11 * Ku * Tu
        elif method == "no_overshoot":
            Kp = 0.2 * Ku
            Ki = 0.4 * Ku / Tu
            Kd = 0.066 * Ku * Tu
        else:
            Kp = 0.6 * Ku
            Ki = 1.2 * Ku / Tu
            Kd = 0.075 * Ku * Tu

        return PIDParams(Kp=Kp, Ki=Ki, Kd=Kd)

    def get_status(self) -> Dict:
        """获取整定状态"""
        return {
            'is_tuning': self.is_tuning,
            'oscillation_count': self.oscillation_count,
            'required_oscillations': self.required_oscillations,
            'ultimate_gain': self.ultimate_gain,
            'ultimate_period': self.ultimate_period
        }
