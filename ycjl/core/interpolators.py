"""
插值器工厂 (Interpolator Factory)
=================================

提供各类插值器的创建和管理，支持：
- 一维插值：线性、PCHIP（保形）
- 二维插值：双线性、薄板样条
- 曲线查找表：用于特性曲线

设计目标：
- 在有scipy时使用高精度插值
- 无scipy时自动回退到numpy基础插值
- 统一的接口便于切换
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional, Dict, Union
from enum import Enum, auto

try:
    from scipy.interpolate import PchipInterpolator, interp1d, RectBivariateSpline
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class InterpolatorType(Enum):
    """插值器类型"""
    LINEAR = auto()             # 线性插值
    PCHIP = auto()              # 保形分段三次Hermite插值
    CUBIC = auto()              # 三次样条
    NEAREST = auto()            # 最近邻
    BILINEAR = auto()           # 双线性
    THIN_PLATE = auto()         # 薄板样条


@dataclass
class CurveLookupTable:
    """
    曲线查找表

    用于存储和查询工程特性曲线（如水位-库容、开度-流阻等）
    """
    name: str                                   # 曲线名称
    x_data: np.ndarray                          # X轴数据
    y_data: np.ndarray                          # Y轴数据
    x_label: str = "x"                          # X轴标签
    y_label: str = "y"                          # Y轴标签
    x_unit: str = ""                            # X轴单位
    y_unit: str = ""                            # Y轴单位
    interpolator: Optional[Callable] = None     # 插值函数
    inverse_interpolator: Optional[Callable] = None  # 反查插值函数

    def __post_init__(self):
        """初始化插值器"""
        if self.interpolator is None:
            self.interpolator = create_interpolator(
                list(zip(self.x_data, self.y_data))
            )
        # 创建反向查找（如果Y单调）
        if self.inverse_interpolator is None and self._is_monotonic(self.y_data):
            self.inverse_interpolator = create_interpolator(
                list(zip(self.y_data, self.x_data))
            )

    @staticmethod
    def _is_monotonic(arr: np.ndarray) -> bool:
        """检查数组是否单调"""
        diff = np.diff(arr)
        return np.all(diff >= 0) or np.all(diff <= 0)

    def lookup(self, x: float) -> float:
        """正向查找"""
        if self.interpolator is None:
            return float(np.interp(x, self.x_data, self.y_data))
        return self.interpolator(x)

    def reverse_lookup(self, y: float) -> float:
        """反向查找"""
        if self.inverse_interpolator is None:
            return float(np.interp(y, self.y_data, self.x_data))
        return self.inverse_interpolator(y)

    def get_range(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """获取数据范围"""
        return (
            (float(self.x_data.min()), float(self.x_data.max())),
            (float(self.y_data.min()), float(self.y_data.max()))
        )


class InterpolatorFactory:
    """
    插值器工厂

    提供统一的插值器创建接口，自动处理scipy可用性
    """

    @staticmethod
    def create_1d(data: List[Tuple[float, float]],
                  method: InterpolatorType = InterpolatorType.PCHIP,
                  extrapolate: bool = False) -> Callable[[float], float]:
        """
        创建一维插值器

        Args:
            data: 数据点列表 [(x1, y1), (x2, y2), ...]
            method: 插值方法
            extrapolate: 是否允许外推

        Returns:
            插值函数 f(x) -> y
        """
        x = np.array([p[0] for p in data])
        y = np.array([p[1] for p in data])

        if not HAS_SCIPY or len(data) < 3:
            # 回退到线性插值
            def linear_interp(val: float) -> float:
                if not extrapolate:
                    val = np.clip(val, x.min(), x.max())
                return float(np.interp(val, x, y))
            return linear_interp

        if method == InterpolatorType.LINEAR:
            interpolator = interp1d(x, y, kind='linear',
                                    bounds_error=not extrapolate,
                                    fill_value='extrapolate' if extrapolate else (y[0], y[-1]))
        elif method == InterpolatorType.PCHIP:
            interpolator = PchipInterpolator(x, y, extrapolate=extrapolate)
        elif method == InterpolatorType.CUBIC:
            interpolator = interp1d(x, y, kind='cubic',
                                    bounds_error=not extrapolate,
                                    fill_value='extrapolate' if extrapolate else (y[0], y[-1]))
        elif method == InterpolatorType.NEAREST:
            interpolator = interp1d(x, y, kind='nearest',
                                    bounds_error=not extrapolate,
                                    fill_value='extrapolate' if extrapolate else (y[0], y[-1]))
        else:
            interpolator = PchipInterpolator(x, y, extrapolate=extrapolate)

        def interp_func(val: float) -> float:
            if not extrapolate:
                val = np.clip(val, x.min(), x.max())
            return float(interpolator(val))

        return interp_func

    @staticmethod
    def create_2d(x_data: np.ndarray, y_data: np.ndarray, z_data: np.ndarray,
                  method: InterpolatorType = InterpolatorType.BILINEAR) -> Callable[[float, float], float]:
        """
        创建二维插值器

        Args:
            x_data: X轴数据 (1D)
            y_data: Y轴数据 (1D)
            z_data: Z值矩阵 (2D, shape = (len(y_data), len(x_data)))
            method: 插值方法

        Returns:
            插值函数 f(x, y) -> z
        """
        if not HAS_SCIPY:
            # 回退到简单的双线性插值
            def bilinear_simple(x: float, y: float) -> float:
                ix = np.searchsorted(x_data, x) - 1
                iy = np.searchsorted(y_data, y) - 1
                ix = max(0, min(ix, len(x_data) - 2))
                iy = max(0, min(iy, len(y_data) - 2))

                x0, x1 = x_data[ix], x_data[ix + 1]
                y0, y1 = y_data[iy], y_data[iy + 1]

                if x1 == x0:
                    tx = 0.0
                else:
                    tx = (x - x0) / (x1 - x0)
                if y1 == y0:
                    ty = 0.0
                else:
                    ty = (y - y0) / (y1 - y0)

                z00 = z_data[iy, ix]
                z01 = z_data[iy, ix + 1]
                z10 = z_data[iy + 1, ix]
                z11 = z_data[iy + 1, ix + 1]

                z0 = z00 * (1 - tx) + z01 * tx
                z1 = z10 * (1 - tx) + z11 * tx
                return float(z0 * (1 - ty) + z1 * ty)

            return bilinear_simple

        # 使用scipy的RectBivariateSpline
        kx = ky = 1 if method == InterpolatorType.BILINEAR else 3
        spline = RectBivariateSpline(y_data, x_data, z_data, kx=kx, ky=ky)

        def interp_2d(x: float, y: float) -> float:
            x = np.clip(x, x_data.min(), x_data.max())
            y = np.clip(y, y_data.min(), y_data.max())
            return float(spline(y, x)[0, 0])

        return interp_2d

    @staticmethod
    def create_lookup_table(name: str,
                            data: List[Tuple[float, float]],
                            x_label: str = "x",
                            y_label: str = "y",
                            x_unit: str = "",
                            y_unit: str = "") -> CurveLookupTable:
        """
        创建曲线查找表

        Args:
            name: 曲线名称
            data: 数据点列表
            x_label, y_label: 轴标签
            x_unit, y_unit: 单位

        Returns:
            CurveLookupTable对象
        """
        x = np.array([p[0] for p in data])
        y = np.array([p[1] for p in data])
        return CurveLookupTable(
            name=name,
            x_data=x,
            y_data=y,
            x_label=x_label,
            y_label=y_label,
            x_unit=x_unit,
            y_unit=y_unit
        )


# ==========================================
# 便捷函数
# ==========================================
def create_interpolator(data: List[Tuple[float, float]],
                        method: InterpolatorType = InterpolatorType.PCHIP) -> Callable[[float], float]:
    """
    创建保形插值器（便捷函数）

    使用PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
    保证单调性，避免过冲振荡

    Args:
        data: 数据点列表 [(x1, y1), (x2, y2), ...]
        method: 插值方法，默认PCHIP

    Returns:
        插值函数
    """
    return InterpolatorFactory.create_1d(data, method)


def create_bilinear_interpolator(x_data: np.ndarray,
                                  y_data: np.ndarray,
                                  z_data: np.ndarray) -> Callable[[float, float], float]:
    """
    创建双线性插值器（便捷函数）

    Args:
        x_data: X轴数据
        y_data: Y轴数据
        z_data: Z值矩阵

    Returns:
        双线性插值函数
    """
    return InterpolatorFactory.create_2d(x_data, y_data, z_data, InterpolatorType.BILINEAR)


def create_curve_lookup(name: str,
                        data: List[Tuple[float, float]],
                        **kwargs) -> CurveLookupTable:
    """
    创建曲线查找表（便捷函数）

    Args:
        name: 曲线名称
        data: 数据点列表
        **kwargs: 其他参数

    Returns:
        CurveLookupTable对象
    """
    return InterpolatorFactory.create_lookup_table(name, data, **kwargs)


__all__ = [
    'InterpolatorType',
    'InterpolatorFactory',
    'CurveLookupTable',
    'create_interpolator',
    'create_bilinear_interpolator',
    'create_curve_lookup',
    'HAS_SCIPY'
]
