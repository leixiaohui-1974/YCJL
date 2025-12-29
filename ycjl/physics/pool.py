"""
稳流连接池模型
==============

核心解耦节点，作为系统级"电容":
- 积分环节特性: dH/dt = (Q_in - Q_out) / A
- 水位-面积变化关系
- 溢流堰与死水位限幅
- 冰期有效库容变化
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto

from ..config.settings import Config


class PoolStatus(Enum):
    """稳流池状态"""
    NORMAL = auto()       # 正常运行
    HIGH_LEVEL = auto()   # 高水位报警
    LOW_LEVEL = auto()    # 低水位报警
    OVERFLOW = auto()     # 溢流
    EMPTY = auto()        # 抽空危险


@dataclass
class PoolState:
    """稳流池状态"""
    level: float          # 水位 (m)
    volume: float         # 容积 (m³)
    inflow: float         # 入流 (m³/s)
    outflow: float        # 出流 (m³/s)
    overflow: float       # 溢流 (m³/s)
    status: PoolStatus    # 运行状态
    buffer_time: float    # 剩余缓冲时间 (s)


class StabilizingPool:
    """
    稳流连接池仿真模型

    物理特性:
    - 积分环节 (水量守恒)
    - 非线性边界 (溢流堰、死水位)
    - 变面积池体
    """

    def __init__(self, config: Optional[Config] = None):
        self.cfg = config.pool if config else Config.pool
        self.physics = Config.physics

        # 几何参数
        self.length = self.cfg.length
        self.width = self.cfg.width
        self.base_area = self.length * self.width

        # 状态变量
        self.level = self.cfg.design_level
        self.volume = self._level_to_volume(self.level)

        # 流量
        self.inflow = 0.0
        self.outflow = 0.0
        self.overflow = 0.0

        # 冰期参数
        self.ice_thickness = 0.0  # 冰盖厚度 (m)
        self.ice_mode = False

        # 溢流堰参数
        self.weir_length = self.width  # 堰长
        self.weir_coeff = 0.42         # 堰流系数

        # 历史记录
        self.history = []

    def _level_to_volume(self, level: float) -> float:
        """水位转容积"""
        # 考虑侧壁微斜 (简化为线性)
        avg_area = self.base_area * (1 + 0.02 * level)
        return avg_area * level

    def _volume_to_level(self, volume: float) -> float:
        """容积转水位"""
        # 迭代求解
        level = volume / self.base_area
        for _ in range(5):
            area = self.base_area * (1 + 0.02 * level)
            level = volume / area
        return max(level, 0)

    def get_effective_area(self) -> float:
        """获取有效水面面积（考虑冰盖）"""
        if self.ice_mode and self.ice_thickness > 0:
            # 冰盖减少有效容积
            return self.base_area * (1 - 0.1 * self.ice_thickness)
        return self.base_area * (1 + 0.02 * self.level)

    def compute_overflow(self) -> float:
        """计算溢流量"""
        if self.level <= self.cfg.max_level:
            return 0.0

        # 堰流公式: Q = C * L * H^1.5
        H = self.level - self.cfg.max_level
        Q_overflow = self.weir_coeff * self.weir_length * (H ** 1.5) * \
                     np.sqrt(2 * self.physics.G)
        return Q_overflow

    def get_status(self) -> PoolStatus:
        """获取当前状态"""
        if self.level >= self.cfg.max_level:
            return PoolStatus.OVERFLOW
        elif self.level >= self.cfg.warning_high:
            return PoolStatus.HIGH_LEVEL
        elif self.level <= self.cfg.min_level:
            return PoolStatus.EMPTY
        elif self.level <= self.cfg.warning_low:
            return PoolStatus.LOW_LEVEL
        else:
            return PoolStatus.NORMAL

    def compute_buffer_time(self) -> float:
        """
        计算缓冲时间

        在当前流量差下，水位到达极限的时间
        """
        net_flow = self.inflow - self.outflow
        area = self.get_effective_area()

        if abs(net_flow) < 0.001:
            return float('inf')

        if net_flow > 0:
            # 水位上升，计算到溢流的时间
            delta_h = self.cfg.max_level - self.level
            return delta_h * area / net_flow
        else:
            # 水位下降，计算到死水位的时间
            delta_h = self.level - self.cfg.min_level
            return delta_h * area / abs(net_flow)

    def step(self, dt: float, inflow: float = None,
             outflow: float = None) -> PoolState:
        """
        推进一个时间步

        Parameters:
            dt: 时间步长 (s)
            inflow: 入流 (m³/s)
            outflow: 出流 (m³/s)

        Returns:
            PoolState: 当前状态
        """
        if inflow is not None:
            self.inflow = inflow
        if outflow is not None:
            self.outflow = outflow

        # 计算溢流
        self.overflow = self.compute_overflow()

        # 净流量
        net_flow = self.inflow - self.outflow - self.overflow

        # 有效面积
        area = self.get_effective_area()

        # 水位变化 (积分)
        dH = net_flow * dt / area
        self.level += dH

        # 物理约束
        # 下限: 死水位（触发紧急关闸）
        if self.level < self.cfg.min_level:
            self.level = self.cfg.min_level
            # 实际上此时outflow应该被强制限制，这里只是记录

        # 上限: 溢流保护
        if self.level > self.cfg.max_level + 0.5:
            self.level = self.cfg.max_level + 0.5
            # 溢流量增加
            self.overflow = self.compute_overflow()

        # 更新容积
        self.volume = self._level_to_volume(self.level)

        state = PoolState(
            level=self.level,
            volume=self.volume,
            inflow=self.inflow,
            outflow=self.outflow,
            overflow=self.overflow,
            status=self.get_status(),
            buffer_time=self.compute_buffer_time()
        )

        self.history.append(state)
        return state

    def set_ice_mode(self, enable: bool, thickness: float = 0.5):
        """设置冰期模式"""
        self.ice_mode = enable
        if enable:
            self.ice_thickness = thickness
        else:
            self.ice_thickness = 0.0

    def get_available_capacity(self) -> Tuple[float, float]:
        """
        获取可用调节容量

        Returns:
            (上调空间, 下调空间) in m³
        """
        area = self.get_effective_area()
        up_space = (self.cfg.max_level - self.level) * area
        down_space = (self.level - self.cfg.min_level) * area
        return max(up_space, 0), max(down_space, 0)

    def reset(self, level: float = None):
        """重置状态"""
        self.level = level if level else self.cfg.design_level
        self.volume = self._level_to_volume(self.level)
        self.inflow = 0.0
        self.outflow = 0.0
        self.overflow = 0.0
        self.ice_mode = False
        self.ice_thickness = 0.0
        self.history.clear()

    def get_state_dict(self) -> dict:
        """获取状态字典"""
        return {
            'level': self.level,
            'volume': self.volume,
            'inflow': self.inflow,
            'outflow': self.outflow,
            'overflow': self.overflow,
            'status': self.get_status().name,
            'buffer_time': self.compute_buffer_time(),
            'ice_mode': self.ice_mode
        }
