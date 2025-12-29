"""
控制协调器
==========

统一管理多个控制回路:
- 模式切换
- 冲突仲裁
- 性能监控
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import time

from .pid import PIDController, PIDParams, PIDMode


class ControlMode(Enum):
    """控制模式"""
    MANUAL = auto()         # 手动控制
    LOCAL_AUTO = auto()     # 本地自动
    REMOTE_AUTO = auto()    # 远程自动
    CASCADE = auto()        # 串级控制
    RATIO = auto()          # 比值控制
    SEQUENCE = auto()       # 顺序控制
    EMERGENCY = auto()      # 应急控制


@dataclass
class ControlLoop:
    """控制回路配置"""
    loop_id: str
    controller: PIDController
    pv_tag: str             # 过程变量标签
    sp_tag: str             # 设定点标签
    output_tag: str         # 输出标签
    mode: ControlMode = ControlMode.LOCAL_AUTO

    # 约束
    output_min: float = 0.0
    output_max: float = 1.0
    rate_limit: float = 0.1  # 输出变化率限制

    # 优先级
    priority: int = 1

    # 状态
    is_active: bool = True
    last_output: float = 0.0


@dataclass
class ControlPerformance:
    """控制性能指标"""
    loop_id: str
    iae: float = 0.0        # 积分绝对误差
    ise: float = 0.0        # 积分平方误差
    itae: float = 0.0       # 积分时间绝对误差
    overshoot: float = 0.0  # 超调量
    settling_time: float = 0.0  # 调节时间
    rise_time: float = 0.0  # 上升时间


class ControlCoordinator:
    """
    控制协调器

    功能:
    - 多回路管理
    - 模式切换
    - 冲突仲裁
    - 性能监控
    """

    def __init__(self):
        # 控制回路
        self.loops: Dict[str, ControlLoop] = {}

        # 全局模式
        self.global_mode = ControlMode.LOCAL_AUTO

        # 性能记录
        self.performance: Dict[str, ControlPerformance] = {}

        # 历史
        self.history: deque = deque(maxlen=1000)

        # 约束检查器
        self.constraint_checkers: Dict[str, Callable] = {}

        # 仲裁规则
        self.arbitration_rules: List[Callable] = []

        # 时间
        self.time = 0.0
        self.dt = 1.0

    def add_loop(self, loop: ControlLoop):
        """添加控制回路"""
        self.loops[loop.loop_id] = loop
        self.performance[loop.loop_id] = ControlPerformance(loop.loop_id)

    def remove_loop(self, loop_id: str):
        """移除控制回路"""
        if loop_id in self.loops:
            del self.loops[loop_id]
            del self.performance[loop_id]

    def set_mode(self, loop_id: str = None, mode: ControlMode = None):
        """
        设置控制模式

        Parameters:
            loop_id: 回路ID (None表示全局)
            mode: 控制模式
        """
        if loop_id is None:
            self.global_mode = mode
            for loop in self.loops.values():
                loop.mode = mode
        else:
            if loop_id in self.loops:
                self.loops[loop_id].mode = mode

    def add_constraint_checker(self, name: str, checker: Callable):
        """添加约束检查器"""
        self.constraint_checkers[name] = checker

    def add_arbitration_rule(self, rule: Callable):
        """添加仲裁规则"""
        self.arbitration_rules.append(rule)

    def step(self, measurements: Dict[str, float],
             setpoints: Dict[str, float],
             dt: float = None) -> Dict[str, float]:
        """
        执行一步控制

        Parameters:
            measurements: 测量值 {tag: value}
            setpoints: 设定值 {tag: value}
            dt: 采样周期

        Returns:
            控制输出 {tag: value}
        """
        if dt is not None:
            self.dt = dt

        self.time += self.dt
        outputs = {}

        # 遍历所有回路
        for loop_id, loop in self.loops.items():
            if not loop.is_active:
                continue

            # 获取PV和SP
            pv = measurements.get(loop.pv_tag, 0.0)
            sp = setpoints.get(loop.sp_tag, 0.0)

            # 计算控制
            if loop.mode == ControlMode.MANUAL:
                output = loop.last_output
            elif loop.mode == ControlMode.EMERGENCY:
                output = self._emergency_control(loop, pv, sp)
            else:
                output = loop.controller.compute(pv, sp, self.dt)

            # 应用约束
            output = self._apply_constraints(loop, output)

            # 冲突仲裁
            output = self._arbitrate(loop_id, output, outputs)

            # 更新
            loop.last_output = output
            outputs[loop.output_tag] = output

            # 更新性能指标
            self._update_performance(loop_id, sp, pv)

        # 记录历史
        self._record_history(measurements, setpoints, outputs)

        return outputs

    def _apply_constraints(self, loop: ControlLoop, output: float) -> float:
        """应用约束"""
        # 输出限幅
        output = np.clip(output, loop.output_min, loop.output_max)

        # 变化率限制
        delta = output - loop.last_output
        max_delta = loop.rate_limit * self.dt
        if abs(delta) > max_delta:
            output = loop.last_output + np.sign(delta) * max_delta

        # 自定义约束
        for name, checker in self.constraint_checkers.items():
            try:
                output = checker(loop, output)
            except Exception:
                pass

        return output

    def _arbitrate(self, loop_id: str, output: float,
                   existing_outputs: Dict[str, float]) -> float:
        """冲突仲裁"""
        loop = self.loops[loop_id]

        # 检查是否有冲突
        for rule in self.arbitration_rules:
            try:
                output = rule(loop, output, existing_outputs)
            except Exception:
                pass

        return output

    def _emergency_control(self, loop: ControlLoop,
                           pv: float, sp: float) -> float:
        """应急控制"""
        # 简单的比例控制
        error = sp - pv
        output = 0.5 + 0.5 * np.sign(error) * min(abs(error), 1.0)
        return output

    def _update_performance(self, loop_id: str, sp: float, pv: float):
        """更新性能指标"""
        perf = self.performance[loop_id]

        error = abs(sp - pv)

        # IAE
        perf.iae += error * self.dt

        # ISE
        perf.ise += error**2 * self.dt

        # ITAE
        perf.itae += self.time * error * self.dt

    def _record_history(self, measurements: Dict, setpoints: Dict,
                        outputs: Dict):
        """记录历史"""
        self.history.append({
            'time': self.time,
            'measurements': measurements.copy(),
            'setpoints': setpoints.copy(),
            'outputs': outputs.copy()
        })

    def get_loop_status(self, loop_id: str = None) -> Dict:
        """获取回路状态"""
        if loop_id:
            loop = self.loops.get(loop_id)
            if loop:
                return {
                    'loop_id': loop.loop_id,
                    'mode': loop.mode.name,
                    'is_active': loop.is_active,
                    'last_output': loop.last_output,
                    'controller_status': loop.controller.get_status(),
                    'performance': {
                        'iae': self.performance[loop_id].iae,
                        'ise': self.performance[loop_id].ise
                    }
                }
        else:
            return {
                loop_id: self.get_loop_status(loop_id)
                for loop_id in self.loops
            }

    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        return {
            loop_id: {
                'iae': perf.iae,
                'ise': perf.ise,
                'itae': perf.itae
            }
            for loop_id, perf in self.performance.items()
        }

    def reset(self):
        """重置"""
        for loop in self.loops.values():
            loop.controller.reset()
            loop.last_output = 0.0

        for perf in self.performance.values():
            perf.iae = 0.0
            perf.ise = 0.0
            perf.itae = 0.0

        self.history.clear()
        self.time = 0.0


class SequenceController:
    """
    顺序控制器

    用于启停操作等顺序控制
    """

    def __init__(self):
        self.steps: List[Dict] = []
        self.current_step = 0
        self.is_running = False
        self.step_start_time = 0.0

        # 状态
        self.status = "idle"
        self.error_message = ""

    def add_step(self, name: str, action: Callable,
                 condition: Callable = None,
                 timeout: float = 60.0):
        """
        添加步骤

        Parameters:
            name: 步骤名称
            action: 动作函数
            condition: 完成条件
            timeout: 超时时间
        """
        self.steps.append({
            'name': name,
            'action': action,
            'condition': condition,
            'timeout': timeout
        })

    def start(self):
        """开始顺序控制"""
        if self.steps:
            self.is_running = True
            self.current_step = 0
            self.status = "running"
            self.step_start_time = time.time()

    def stop(self):
        """停止顺序控制"""
        self.is_running = False
        self.status = "stopped"

    def step(self, state: Dict) -> Optional[Dict]:
        """
        执行一步

        Parameters:
            state: 当前系统状态

        Returns:
            控制动作 (如果有)
        """
        if not self.is_running or self.current_step >= len(self.steps):
            return None

        current = self.steps[self.current_step]
        elapsed = time.time() - self.step_start_time

        # 检查超时
        if elapsed > current['timeout']:
            self.error_message = f"Step '{current['name']}' timeout"
            self.status = "error"
            self.is_running = False
            return None

        # 执行动作
        action_result = current['action'](state)

        # 检查完成条件
        condition = current.get('condition')
        if condition is None or condition(state):
            # 进入下一步
            self.current_step += 1
            self.step_start_time = time.time()

            if self.current_step >= len(self.steps):
                self.status = "completed"
                self.is_running = False

        return action_result

    def get_status(self) -> Dict:
        """获取状态"""
        return {
            'status': self.status,
            'current_step': self.current_step,
            'total_steps': len(self.steps),
            'current_step_name': self.steps[self.current_step]['name']
                if self.current_step < len(self.steps) else None,
            'error_message': self.error_message
        }

    def reset(self):
        """重置"""
        self.current_step = 0
        self.is_running = False
        self.status = "idle"
        self.error_message = ""


class InterlockManager:
    """
    联锁管理器

    管理安全联锁逻辑
    """

    def __init__(self):
        self.interlocks: Dict[str, Dict] = {}
        self.active_interlocks: List[str] = []

    def add_interlock(self, name: str,
                      condition: Callable[[Dict], bool],
                      action: Callable[[Dict], Dict],
                      priority: int = 0):
        """
        添加联锁

        Parameters:
            name: 联锁名称
            condition: 触发条件
            action: 联锁动作
            priority: 优先级
        """
        self.interlocks[name] = {
            'condition': condition,
            'action': action,
            'priority': priority
        }

    def check_interlocks(self, state: Dict) -> Tuple[bool, Dict]:
        """
        检查联锁

        Parameters:
            state: 系统状态

        Returns:
            (是否有联锁触发, 联锁动作)
        """
        self.active_interlocks = []
        combined_action = {}

        # 按优先级排序
        sorted_interlocks = sorted(
            self.interlocks.items(),
            key=lambda x: x[1]['priority']
        )

        for name, interlock in sorted_interlocks:
            try:
                if interlock['condition'](state):
                    self.active_interlocks.append(name)
                    action = interlock['action'](state)
                    combined_action.update(action)
            except Exception:
                pass

        return len(self.active_interlocks) > 0, combined_action

    def get_active_interlocks(self) -> List[str]:
        """获取活跃联锁列表"""
        return self.active_interlocks.copy()
