"""
数据分析器
==========

性能分析、趋势分析、异常检测
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from scipy import stats, signal


@dataclass
class PerformanceMetrics:
    """性能指标"""
    # 误差指标
    mae: float = 0.0        # 平均绝对误差
    mse: float = 0.0        # 均方误差
    rmse: float = 0.0       # 均方根误差
    max_error: float = 0.0  # 最大误差

    # 控制性能
    iae: float = 0.0        # 积分绝对误差
    ise: float = 0.0        # 积分平方误差
    itae: float = 0.0       # 积分时间绝对误差

    # 响应特性
    rise_time: float = 0.0      # 上升时间
    settling_time: float = 0.0  # 调节时间
    overshoot: float = 0.0      # 超调量
    undershoot: float = 0.0     # 下冲量

    # 稳定性
    variance: float = 0.0       # 方差
    std_dev: float = 0.0        # 标准差


@dataclass
class TrendResult:
    """趋势分析结果"""
    slope: float            # 斜率
    intercept: float        # 截距
    r_squared: float        # 决定系数
    p_value: float          # p值
    trend_type: str         # 趋势类型: 'increasing', 'decreasing', 'stable'
    confidence: float       # 置信度


@dataclass
class AnomalyResult:
    """异常检测结果"""
    timestamp: float
    value: float
    expected: float
    deviation: float
    severity: float         # 0-1
    anomaly_type: str       # 类型


class PerformanceAnalyzer:
    """
    性能分析器

    计算控制系统性能指标
    """

    def __init__(self, settling_threshold: float = 0.02):
        """
        Parameters:
            settling_threshold: 调节时间阈值 (相对误差)
        """
        self.settling_threshold = settling_threshold

    def analyze(self, time: np.ndarray, setpoint: np.ndarray,
                measurement: np.ndarray) -> PerformanceMetrics:
        """
        分析控制性能

        Parameters:
            time: 时间序列
            setpoint: 设定值序列
            measurement: 测量值序列

        Returns:
            PerformanceMetrics
        """
        if len(time) < 2:
            return PerformanceMetrics()

        error = setpoint - measurement
        dt = np.mean(np.diff(time))

        metrics = PerformanceMetrics()

        # 误差指标
        metrics.mae = np.mean(np.abs(error))
        metrics.mse = np.mean(error ** 2)
        metrics.rmse = np.sqrt(metrics.mse)
        metrics.max_error = np.max(np.abs(error))

        # 积分指标
        metrics.iae = np.sum(np.abs(error)) * dt
        metrics.ise = np.sum(error ** 2) * dt
        metrics.itae = np.sum(time * np.abs(error)) * dt

        # 响应特性
        self._analyze_response(time, setpoint, measurement, metrics)

        # 稳定性
        metrics.variance = np.var(measurement)
        metrics.std_dev = np.std(measurement)

        return metrics

    def _analyze_response(self, time: np.ndarray, setpoint: np.ndarray,
                          measurement: np.ndarray, metrics: PerformanceMetrics):
        """分析响应特性"""
        if len(setpoint) < 2:
            return

        # 检测阶跃变化
        sp_changes = np.where(np.abs(np.diff(setpoint)) > 0.1)[0]

        if len(sp_changes) == 0:
            return

        # 分析最后一次阶跃响应
        step_idx = sp_changes[-1]
        if step_idx + 10 >= len(time):
            return

        # 阶跃幅度
        sp_before = setpoint[step_idx]
        sp_after = setpoint[step_idx + 1]
        step_amplitude = sp_after - sp_before

        if abs(step_amplitude) < 0.01:
            return

        # 响应段
        response = measurement[step_idx:]
        response_time = time[step_idx:] - time[step_idx]

        # 上升时间 (10%-90%)
        target_10 = sp_before + 0.1 * step_amplitude
        target_90 = sp_before + 0.9 * step_amplitude

        try:
            if step_amplitude > 0:
                idx_10 = np.where(response >= target_10)[0]
                idx_90 = np.where(response >= target_90)[0]
            else:
                idx_10 = np.where(response <= target_10)[0]
                idx_90 = np.where(response <= target_90)[0]

            if len(idx_10) > 0 and len(idx_90) > 0:
                metrics.rise_time = response_time[idx_90[0]] - response_time[idx_10[0]]
        except:
            pass

        # 超调量
        if step_amplitude > 0:
            peak = np.max(response)
            metrics.overshoot = max(0, (peak - sp_after) / step_amplitude * 100)
        else:
            trough = np.min(response)
            metrics.overshoot = max(0, (sp_after - trough) / abs(step_amplitude) * 100)

        # 调节时间
        error_band = abs(step_amplitude) * self.settling_threshold

        for i in range(len(response) - 1, -1, -1):
            if abs(response[i] - sp_after) > error_band:
                if i + 1 < len(response_time):
                    metrics.settling_time = response_time[i + 1]
                break

    def compare_controllers(self, results: Dict[str, PerformanceMetrics]) -> Dict:
        """
        比较多个控制器性能

        Parameters:
            results: {controller_name: metrics}

        Returns:
            比较结果
        """
        comparison = {
            'rankings': {},
            'best': {},
            'summary': {}
        }

        # 各指标排名
        metrics_to_compare = ['iae', 'ise', 'rise_time', 'settling_time', 'overshoot']

        for metric in metrics_to_compare:
            values = {name: getattr(m, metric) for name, m in results.items()}
            sorted_names = sorted(values.keys(), key=lambda x: values[x])
            comparison['rankings'][metric] = sorted_names
            comparison['best'][metric] = sorted_names[0]

        return comparison


class TrendAnalyzer:
    """
    趋势分析器

    线性趋势、周期性、变化点检测
    """

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def analyze_linear_trend(self, time: np.ndarray,
                             values: np.ndarray) -> TrendResult:
        """
        线性趋势分析

        Parameters:
            time: 时间序列
            values: 值序列

        Returns:
            TrendResult
        """
        if len(time) < 3:
            return TrendResult(0, 0, 0, 1, 'stable', 0)

        # 线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(time, values)

        r_squared = r_value ** 2

        # 判断趋势类型
        if p_value < self.significance_level:
            if slope > 0:
                trend_type = 'increasing'
            else:
                trend_type = 'decreasing'
            confidence = 1 - p_value
        else:
            trend_type = 'stable'
            confidence = p_value

        return TrendResult(
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            p_value=p_value,
            trend_type=trend_type,
            confidence=confidence
        )

    def detect_seasonality(self, values: np.ndarray,
                           max_period: int = None) -> Dict:
        """
        检测周期性

        Parameters:
            values: 值序列
            max_period: 最大周期

        Returns:
            周期性分析结果
        """
        if len(values) < 10:
            return {'has_seasonality': False}

        if max_period is None:
            max_period = len(values) // 2

        # 自相关分析
        autocorr = np.correlate(values - np.mean(values),
                                values - np.mean(values), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]

        # 找峰值
        peaks = []
        for i in range(1, min(max_period, len(autocorr) - 1)):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                if autocorr[i] > 0.3:  # 阈值
                    peaks.append((i, autocorr[i]))

        if peaks:
            # 最强周期
            peaks.sort(key=lambda x: x[1], reverse=True)
            main_period = peaks[0][0]
            strength = peaks[0][1]

            return {
                'has_seasonality': True,
                'main_period': main_period,
                'strength': strength,
                'all_periods': peaks[:5]
            }
        else:
            return {'has_seasonality': False}

    def detect_change_points(self, values: np.ndarray,
                             penalty: float = 1.0) -> List[int]:
        """
        检测变化点

        基于CUSUM算法

        Parameters:
            values: 值序列
            penalty: 惩罚参数

        Returns:
            变化点索引列表
        """
        if len(values) < 10:
            return []

        mean_val = np.mean(values)
        std_val = np.std(values)

        if std_val < 1e-6:
            return []

        # 标准化
        normalized = (values - mean_val) / std_val

        # CUSUM
        cusum_pos = np.zeros(len(values))
        cusum_neg = np.zeros(len(values))

        for i in range(1, len(values)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + normalized[i] - penalty)
            cusum_neg[i] = max(0, cusum_neg[i-1] - normalized[i] - penalty)

        # 检测阈值
        threshold = 4.0

        change_points = []
        for i in range(len(values)):
            if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
                change_points.append(i)
                # 重置
                cusum_pos[i] = 0
                cusum_neg[i] = 0

        return change_points


class AnomalyAnalyzer:
    """
    异常检测分析器

    多种异常检测方法
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)

        # 统计量
        self.mean = 0.0
        self.std = 1.0
        self.ewma = 0.0

    def update(self, value: float, timestamp: float = None) -> Optional[AnomalyResult]:
        """
        更新并检测异常

        Parameters:
            value: 新值
            timestamp: 时间戳

        Returns:
            异常结果 (如果检测到异常)
        """
        if timestamp is None:
            timestamp = len(self.history)

        self.history.append(value)

        if len(self.history) < 10:
            return None

        # 更新统计量
        data = np.array(self.history)
        self.mean = np.mean(data)
        self.std = np.std(data)
        if self.std < 1e-6:
            self.std = 1.0

        # EWMA更新
        alpha = 2 / (self.window_size + 1)
        self.ewma = alpha * value + (1 - alpha) * self.ewma

        # 检测异常
        z_score = abs(value - self.mean) / self.std

        if z_score > 3:
            severity = min((z_score - 3) / 3, 1.0)

            anomaly_type = self._classify_anomaly(value, data)

            return AnomalyResult(
                timestamp=timestamp,
                value=value,
                expected=self.mean,
                deviation=value - self.mean,
                severity=severity,
                anomaly_type=anomaly_type
            )

        return None

    def _classify_anomaly(self, value: float, history: np.ndarray) -> str:
        """分类异常类型"""
        mean = np.mean(history)

        if value > mean:
            # 检查是否是突变
            if len(history) > 2:
                diff = value - history[-2]
                avg_diff = np.mean(np.abs(np.diff(history)))
                if abs(diff) > 5 * avg_diff:
                    return 'spike_up'
            return 'high_value'
        else:
            if len(history) > 2:
                diff = value - history[-2]
                avg_diff = np.mean(np.abs(np.diff(history)))
                if abs(diff) > 5 * avg_diff:
                    return 'spike_down'
            return 'low_value'

    def batch_detect(self, time: np.ndarray, values: np.ndarray,
                     method: str = 'zscore') -> List[AnomalyResult]:
        """
        批量异常检测

        Parameters:
            time: 时间序列
            values: 值序列
            method: 检测方法 ('zscore', 'iqr', 'isolation')

        Returns:
            异常列表
        """
        anomalies = []

        if method == 'zscore':
            mean = np.mean(values)
            std = np.std(values)
            if std < 1e-6:
                return []

            z_scores = np.abs((values - mean) / std)

            for i, z in enumerate(z_scores):
                if z > 3:
                    anomalies.append(AnomalyResult(
                        timestamp=time[i],
                        value=values[i],
                        expected=mean,
                        deviation=values[i] - mean,
                        severity=min((z - 3) / 3, 1.0),
                        anomaly_type='zscore_outlier'
                    ))

        elif method == 'iqr':
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            for i, v in enumerate(values):
                if v < lower or v > upper:
                    severity = abs(v - np.median(values)) / (iqr + 1e-6)
                    severity = min(severity / 3, 1.0)

                    anomalies.append(AnomalyResult(
                        timestamp=time[i],
                        value=v,
                        expected=np.median(values),
                        deviation=v - np.median(values),
                        severity=severity,
                        anomaly_type='iqr_outlier'
                    ))

        return anomalies

    def reset(self):
        """重置"""
        self.history.clear()
        self.mean = 0.0
        self.std = 1.0
        self.ewma = 0.0
