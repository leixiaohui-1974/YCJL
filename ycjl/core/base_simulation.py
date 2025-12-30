"""
仿真引擎基类 (Base Simulation Engine)
=====================================

提供水利工程仿真引擎的抽象基类，支持：
- 稳态仿真
- 瞬态仿真
- 多组件耦合仿真
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any
from enum import Enum, auto
import numpy as np

from .base_physics import BaseHydraulicComponent, HydraulicState


class SimulationMode(Enum):
    """仿真模式"""
    STEADY_STATE = auto()       # 稳态
    TRANSIENT = auto()          # 瞬态
    QUASI_STEADY = auto()       # 准稳态


class SimulationStatus(Enum):
    """仿真状态"""
    IDLE = auto()               # 空闲
    RUNNING = auto()            # 运行中
    PAUSED = auto()             # 暂停
    COMPLETED = auto()          # 完成
    FAILED = auto()             # 失败
    ABORTED = auto()            # 中止


@dataclass
class TimeSeriesData:
    """时间序列数据"""
    name: str                                   # 变量名
    unit: str = ""                              # 单位
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    values: np.ndarray = field(default_factory=lambda: np.array([]))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def append(self, t: float, v: float):
        """追加数据点"""
        self.times = np.append(self.times, t)
        self.values = np.append(self.values, v)

    def get_value_at(self, t: float) -> float:
        """获取指定时间的值（线性插值）"""
        if len(self.times) == 0:
            return 0.0
        return float(np.interp(t, self.times, self.values))

    def get_statistics(self) -> Dict[str, float]:
        """获取统计信息"""
        if len(self.values) == 0:
            return {"min": 0, "max": 0, "mean": 0, "std": 0}
        return {
            "min": float(np.min(self.values)),
            "max": float(np.max(self.values)),
            "mean": float(np.mean(self.values)),
            "std": float(np.std(self.values))
        }


@dataclass
class SimulationResult:
    """仿真结果"""
    success: bool                               # 是否成功
    mode: SimulationMode                        # 仿真模式
    start_time: float                           # 仿真起始时间
    end_time: float                             # 仿真结束时间
    duration_s: float                           # 实际耗时 (s)
    step_count: int                             # 步数

    # 时间序列结果
    time_series: Dict[str, TimeSeriesData] = field(default_factory=dict)

    # 最终状态
    final_states: Dict[str, HydraulicState] = field(default_factory=dict)

    # 诊断信息
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    performance: Dict[str, float] = field(default_factory=dict)

    def get_series(self, name: str) -> Optional[TimeSeriesData]:
        """获取时间序列"""
        return self.time_series.get(name)

    def get_final_state(self, component_name: str) -> Optional[HydraulicState]:
        """获取组件最终状态"""
        return self.final_states.get(component_name)

    def add_warning(self, msg: str):
        """添加警告"""
        self.warnings.append(msg)

    def add_error(self, msg: str):
        """添加错误"""
        self.errors.append(msg)

    def summary(self) -> str:
        """生成摘要"""
        lines = [
            f"仿真结果摘要",
            f"=" * 40,
            f"状态: {'成功 ✓' if self.success else '失败 ✗'}",
            f"模式: {self.mode.name}",
            f"时间范围: {self.start_time:.1f}s - {self.end_time:.1f}s",
            f"步数: {self.step_count}",
            f"耗时: {self.duration_s:.3f}s"
        ]

        if self.time_series:
            lines.append(f"\n记录的变量: {len(self.time_series)}")
            for name, ts in list(self.time_series.items())[:5]:
                stats = ts.get_statistics()
                lines.append(f"  - {name}: {stats['mean']:.2f} [{stats['min']:.2f}, {stats['max']:.2f}]")

        if self.warnings:
            lines.append(f"\n警告 ({len(self.warnings)}):")
            for w in self.warnings[:3]:
                lines.append(f"  ⚠️ {w}")

        if self.errors:
            lines.append(f"\n错误 ({len(self.errors)}):")
            for e in self.errors[:3]:
                lines.append(f"  ❌ {e}")

        return "\n".join(lines)


class BaseSimulationEngine(ABC):
    """
    仿真引擎抽象基类

    提供仿真运行的通用框架
    """

    def __init__(self, name: str = "SimulationEngine"):
        self.name = name
        self.status = SimulationStatus.IDLE
        self.mode = SimulationMode.STEADY_STATE
        self.components: Dict[str, BaseHydraulicComponent] = {}
        self._current_time = 0.0
        self._dt = 1.0
        self._result: Optional[SimulationResult] = None

        # 回调函数
        self._step_callback: Optional[Callable[[float, Dict], None]] = None
        self._progress_callback: Optional[Callable[[float], None]] = None

    def add_component(self, component: BaseHydraulicComponent):
        """添加组件"""
        self.components[component.name] = component

    def remove_component(self, name: str):
        """移除组件"""
        if name in self.components:
            del self.components[name]

    def get_component(self, name: str) -> Optional[BaseHydraulicComponent]:
        """获取组件"""
        return self.components.get(name)

    def set_step_callback(self, callback: Callable[[float, Dict], None]):
        """设置步进回调"""
        self._step_callback = callback

    def set_progress_callback(self, callback: Callable[[float], None]):
        """设置进度回调"""
        self._progress_callback = callback

    @abstractmethod
    def setup(self, **kwargs) -> bool:
        """
        设置仿真参数

        Returns:
            是否设置成功
        """
        pass

    @abstractmethod
    def step(self, dt: float) -> Dict[str, HydraulicState]:
        """
        执行单步仿真

        Args:
            dt: 时间步长

        Returns:
            各组件状态
        """
        pass

    def run(self, duration: float, dt: float = 1.0,
            mode: SimulationMode = SimulationMode.STEADY_STATE,
            **kwargs) -> SimulationResult:
        """
        运行仿真

        Args:
            duration: 仿真时长 (s)
            dt: 时间步长 (s)
            mode: 仿真模式
            **kwargs: 其他参数

        Returns:
            仿真结果
        """
        self.mode = mode
        self._dt = dt
        self.status = SimulationStatus.RUNNING
        wall_time_start = time.time()

        # 初始化结果
        result = SimulationResult(
            success=True,
            mode=mode,
            start_time=self._current_time,
            end_time=self._current_time + duration,
            duration_s=0.0,
            step_count=0
        )

        # 初始化时间序列
        for comp_name in self.components:
            result.time_series[f"{comp_name}_flow"] = TimeSeriesData(
                name=f"{comp_name}_flow", unit="m³/s"
            )
            result.time_series[f"{comp_name}_level"] = TimeSeriesData(
                name=f"{comp_name}_level", unit="m"
            )
            result.time_series[f"{comp_name}_pressure"] = TimeSeriesData(
                name=f"{comp_name}_pressure", unit="m"
            )

        # 设置
        if not self.setup(**kwargs):
            result.success = False
            result.add_error("仿真设置失败")
            self.status = SimulationStatus.FAILED
            return result

        # 主循环
        try:
            end_time = self._current_time + duration
            step_count = 0
            total_steps = int(duration / dt)

            while self._current_time < end_time and self.status == SimulationStatus.RUNNING:
                # 执行单步
                states = self.step(dt)
                step_count += 1

                # 记录时间序列
                for comp_name, state in states.items():
                    if f"{comp_name}_flow" in result.time_series:
                        result.time_series[f"{comp_name}_flow"].append(
                            self._current_time, state.flow
                        )
                    if f"{comp_name}_level" in result.time_series:
                        result.time_series[f"{comp_name}_level"].append(
                            self._current_time, state.level
                        )
                    if f"{comp_name}_pressure" in result.time_series:
                        result.time_series[f"{comp_name}_pressure"].append(
                            self._current_time, state.pressure_head
                        )

                # 回调
                if self._step_callback:
                    self._step_callback(self._current_time, states)

                if self._progress_callback and total_steps > 0:
                    progress = step_count / total_steps
                    self._progress_callback(progress)

                self._current_time += dt
                result.step_count = step_count

            # 记录最终状态
            for comp_name, comp in self.components.items():
                result.final_states[comp_name] = comp.state

            self.status = SimulationStatus.COMPLETED

        except Exception as e:
            result.success = False
            result.add_error(f"仿真运行错误: {str(e)}")
            self.status = SimulationStatus.FAILED

        # 计算耗时
        result.duration_s = time.time() - wall_time_start
        result.performance["steps_per_second"] = result.step_count / max(result.duration_s, 0.001)
        result.performance["real_time_factor"] = duration / max(result.duration_s, 0.001)

        self._result = result
        return result

    def pause(self):
        """暂停仿真"""
        if self.status == SimulationStatus.RUNNING:
            self.status = SimulationStatus.PAUSED

    def resume(self):
        """恢复仿真"""
        if self.status == SimulationStatus.PAUSED:
            self.status = SimulationStatus.RUNNING

    def abort(self):
        """中止仿真"""
        self.status = SimulationStatus.ABORTED

    def reset(self):
        """重置仿真"""
        self._current_time = 0.0
        self.status = SimulationStatus.IDLE
        for comp in self.components.values():
            comp.clear_history()

    def get_result(self) -> Optional[SimulationResult]:
        """获取最近一次仿真结果"""
        return self._result

    @property
    def current_time(self) -> float:
        """当前仿真时间"""
        return self._current_time


__all__ = [
    'SimulationMode',
    'SimulationStatus',
    'TimeSeriesData',
    'SimulationResult',
    'BaseSimulationEngine'
]
