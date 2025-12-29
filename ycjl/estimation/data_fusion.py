"""
多源数据融合
============

融合多种传感器数据:
- 加权平均融合
- 卡尔曼融合
- 异常检测与剔除
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque

from ..config.settings import Config


@dataclass
class FusedData:
    """融合数据"""
    value: float           # 融合值
    variance: float        # 方差
    confidence: float      # 置信度
    sources: List[str]     # 数据源
    is_valid: bool         # 是否有效
    outlier_detected: bool # 是否检测到异常值


class DataFusion:
    """
    多源数据融合器

    支持:
    - 加权最小方差融合
    - 动态权重调整
    - 异常值检测与剔除
    """

    def __init__(self):
        # 传感器数据
        self.sensors: Dict[str, Dict] = {}

        # 融合权重
        self.weights: Dict[str, float] = {}

        # 历史数据 (用于异常检测)
        self.history: Dict[str, deque] = {}
        self.history_size = 50

        # 异常检测参数
        self.outlier_threshold = 3.0  # 标准差倍数

    def register_sensor(self, name: str, variance: float = 0.1,
                        weight: float = 1.0):
        """
        注册传感器

        Parameters:
            name: 传感器名称
            variance: 测量方差
            weight: 初始权重
        """
        self.sensors[name] = {
            'variance': variance,
            'last_value': None,
            'last_quality': 1.0
        }
        self.weights[name] = weight
        self.history[name] = deque(maxlen=self.history_size)

    def update_sensor(self, name: str, value: float, quality: float = 1.0):
        """
        更新传感器数据

        Parameters:
            name: 传感器名称
            value: 测量值
            quality: 数据质量 (0~1)
        """
        if name not in self.sensors:
            self.register_sensor(name)

        self.sensors[name]['last_value'] = value
        self.sensors[name]['last_quality'] = quality
        self.history[name].append(value)

    def detect_outlier(self, name: str, value: float) -> bool:
        """
        检测异常值

        基于历史数据的统计特性
        """
        if name not in self.history or len(self.history[name]) < 10:
            return False

        history = list(self.history[name])
        mean = np.mean(history)
        std = np.std(history)

        if std < 0.001:
            return False

        z_score = abs(value - mean) / std
        return z_score > self.outlier_threshold

    def compute_weights(self) -> Dict[str, float]:
        """
        计算融合权重

        基于方差和质量的最优加权
        """
        weights = {}
        total_inverse_var = 0.0

        for name, sensor in self.sensors.items():
            if sensor['last_value'] is None:
                continue

            var = sensor['variance']
            quality = sensor['last_quality']

            # 考虑质量的有效方差
            effective_var = var / max(quality, 0.1)

            # 异常惩罚
            if self.detect_outlier(name, sensor['last_value']):
                effective_var *= 10.0

            inverse_var = 1.0 / max(effective_var, 1e-6)
            weights[name] = inverse_var
            total_inverse_var += inverse_var

        # 归一化
        if total_inverse_var > 0:
            for name in weights:
                weights[name] /= total_inverse_var

        return weights

    def fuse(self) -> FusedData:
        """
        执行数据融合

        Returns:
            FusedData: 融合结果
        """
        weights = self.compute_weights()

        if not weights:
            return FusedData(
                value=0.0,
                variance=float('inf'),
                confidence=0.0,
                sources=[],
                is_valid=False,
                outlier_detected=False
            )

        # 加权平均
        fused_value = 0.0
        sources = []
        outlier_detected = False

        for name, weight in weights.items():
            value = self.sensors[name]['last_value']
            if value is not None:
                if self.detect_outlier(name, value):
                    outlier_detected = True
                    continue  # 剔除异常值

                fused_value += weight * value
                sources.append(name)

        # 融合方差
        total_weight = sum(weights[s] for s in sources)
        if total_weight > 0:
            fused_value /= total_weight

        # 计算融合方差
        fused_variance = 0.0
        for name in sources:
            fused_variance += (weights[name] ** 2) * self.sensors[name]['variance']

        # 置信度
        confidence = min(len(sources) / max(len(self.sensors), 1), 1.0)
        if outlier_detected:
            confidence *= 0.8

        return FusedData(
            value=fused_value,
            variance=fused_variance,
            confidence=confidence,
            sources=sources,
            is_valid=len(sources) > 0,
            outlier_detected=outlier_detected
        )

    def fuse_with_model(self, model_prediction: float,
                        model_variance: float) -> FusedData:
        """
        融合传感器数据与模型预测

        Parameters:
            model_prediction: 模型预测值
            model_variance: 模型预测方差

        Returns:
            FusedData: 融合结果
        """
        # 先融合传感器数据
        sensor_fusion = self.fuse()

        if not sensor_fusion.is_valid:
            return FusedData(
                value=model_prediction,
                variance=model_variance,
                confidence=0.5,
                sources=['model'],
                is_valid=True,
                outlier_detected=False
            )

        # 卡尔曼融合
        # x_fused = x_model + K * (x_sensor - x_model)
        # K = P_model / (P_model + P_sensor)

        K = model_variance / (model_variance + sensor_fusion.variance + 1e-6)
        fused_value = model_prediction + K * (sensor_fusion.value - model_prediction)
        fused_variance = (1 - K) * model_variance

        sources = sensor_fusion.sources + ['model']
        confidence = 0.5 + 0.5 * sensor_fusion.confidence

        return FusedData(
            value=fused_value,
            variance=fused_variance,
            confidence=confidence,
            sources=sources,
            is_valid=True,
            outlier_detected=sensor_fusion.outlier_detected
        )

    def get_sensor_health(self) -> Dict[str, float]:
        """
        获取传感器健康状态

        Returns:
            各传感器健康度 (0~1)
        """
        health = {}

        for name, sensor in self.sensors.items():
            if sensor['last_value'] is None:
                health[name] = 0.0
                continue

            # 基于质量
            quality = sensor['last_quality']

            # 基于异常检测
            if self.detect_outlier(name, sensor['last_value']):
                quality *= 0.5

            # 基于历史稳定性
            if name in self.history and len(self.history[name]) > 10:
                std = np.std(list(self.history[name]))
                stability = 1.0 / (1.0 + std)
                quality *= stability

            health[name] = quality

        return health

    def reset(self):
        """重置"""
        for name in self.sensors:
            self.sensors[name]['last_value'] = None
            self.sensors[name]['last_quality'] = 1.0

        for name in self.history:
            self.history[name].clear()
