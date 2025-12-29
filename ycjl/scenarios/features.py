"""
特征提取器
==========

从多源数据中提取用于场景识别的特征:
- 统计特征
- 时序特征
- 频域特征
- 关联特征
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto


class FeatureType(Enum):
    """特征类型"""
    STATISTICAL = auto()    # 统计特征
    TEMPORAL = auto()       # 时序特征
    FREQUENCY = auto()      # 频域特征
    CORRELATION = auto()    # 关联特征
    DERIVATIVE = auto()     # 导数特征


@dataclass
class FeatureSet:
    """特征集"""
    timestamp: float
    features: Dict[str, float]
    feature_types: Dict[str, FeatureType]
    confidence: float = 1.0
    is_valid: bool = True


class FeatureExtractor:
    """
    多模态特征提取器

    从传感器数据中提取场景识别特征
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

        # 数据缓冲
        self.buffers: Dict[str, deque] = {}
        self.timestamps: deque = deque(maxlen=window_size)

        # 特征配置
        self.feature_configs = self._init_feature_configs()

        # 标准化参数
        self.mean_values: Dict[str, float] = {}
        self.std_values: Dict[str, float] = {}
        self.is_calibrated = False

    def _init_feature_configs(self) -> Dict:
        """初始化特征配置"""
        return {
            # 水位相关
            'pool_level': {
                'statistical': True,
                'temporal': True,
                'frequency': False
            },
            # 流量相关
            'tunnel_flow': {
                'statistical': True,
                'temporal': True,
                'frequency': True
            },
            'pipe_flow': {
                'statistical': True,
                'temporal': True,
                'frequency': True
            },
            # 压力相关
            'pipe_pressure': {
                'statistical': True,
                'temporal': True,
                'frequency': True
            },
            # 温度相关
            'water_temperature': {
                'statistical': True,
                'temporal': False,
                'frequency': False
            }
        }

    def update(self, data: Dict[str, float], timestamp: float):
        """更新数据缓冲"""
        self.timestamps.append(timestamp)

        for key, value in data.items():
            if key not in self.buffers:
                self.buffers[key] = deque(maxlen=self.window_size)
            self.buffers[key].append(value)

    def calibrate(self, data_history: List[Dict[str, float]]):
        """校准标准化参数"""
        if not data_history:
            return

        # 汇总数据
        all_data: Dict[str, List[float]] = {}
        for data in data_history:
            for key, value in data.items():
                if key not in all_data:
                    all_data[key] = []
                all_data[key].append(value)

        # 计算统计量
        for key, values in all_data.items():
            self.mean_values[key] = np.mean(values)
            self.std_values[key] = np.std(values)
            if self.std_values[key] < 1e-6:
                self.std_values[key] = 1.0

        self.is_calibrated = True

    def extract_statistical_features(self, key: str) -> Dict[str, float]:
        """提取统计特征"""
        features = {}

        if key not in self.buffers or len(self.buffers[key]) < 5:
            return features

        data = np.array(self.buffers[key])

        # 基本统计量
        features[f'{key}_mean'] = float(np.mean(data))
        features[f'{key}_std'] = float(np.std(data))
        features[f'{key}_min'] = float(np.min(data))
        features[f'{key}_max'] = float(np.max(data))
        features[f'{key}_range'] = float(np.max(data) - np.min(data))

        # 分位数
        features[f'{key}_q25'] = float(np.percentile(data, 25))
        features[f'{key}_q75'] = float(np.percentile(data, 75))
        features[f'{key}_iqr'] = features[f'{key}_q75'] - features[f'{key}_q25']

        # 偏度和峰度
        if len(data) > 10:
            mean = np.mean(data)
            std = np.std(data)
            if std > 1e-6:
                features[f'{key}_skewness'] = float(
                    np.mean(((data - mean) / std) ** 3)
                )
                features[f'{key}_kurtosis'] = float(
                    np.mean(((data - mean) / std) ** 4) - 3
                )

        return features

    def extract_temporal_features(self, key: str) -> Dict[str, float]:
        """提取时序特征"""
        features = {}

        if key not in self.buffers or len(self.buffers[key]) < 10:
            return features

        data = np.array(self.buffers[key])

        # 趋势
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        features[f'{key}_trend'] = float(coeffs[0])

        # 变化率
        diff = np.diff(data)
        features[f'{key}_mean_change'] = float(np.mean(diff))
        features[f'{key}_max_change'] = float(np.max(np.abs(diff)))
        features[f'{key}_change_std'] = float(np.std(diff))

        # 二阶变化率 (加速度)
        if len(data) > 10:
            diff2 = np.diff(diff)
            features[f'{key}_acceleration'] = float(np.mean(diff2))

        # 过零率
        zero_crossings = np.sum(np.abs(np.diff(np.sign(data - np.mean(data)))) > 0)
        features[f'{key}_zero_crossing_rate'] = float(zero_crossings / len(data))

        # 自相关
        if len(data) > 20:
            autocorr = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            features[f'{key}_autocorr_lag1'] = float(autocorr[1]) if len(autocorr) > 1 else 0
            features[f'{key}_autocorr_lag5'] = float(autocorr[5]) if len(autocorr) > 5 else 0

        return features

    def extract_frequency_features(self, key: str) -> Dict[str, float]:
        """提取频域特征"""
        features = {}

        if key not in self.buffers or len(self.buffers[key]) < 32:
            return features

        data = np.array(self.buffers[key])

        # FFT
        fft = np.fft.fft(data - np.mean(data))
        freqs = np.fft.fftfreq(len(data))

        # 幅值谱
        magnitude = np.abs(fft[:len(fft)//2])
        freqs = freqs[:len(freqs)//2]

        # 主频率
        if len(magnitude) > 0:
            max_idx = np.argmax(magnitude)
            features[f'{key}_dominant_freq'] = float(abs(freqs[max_idx]))
            features[f'{key}_dominant_magnitude'] = float(magnitude[max_idx])

        # 频谱能量分布
        total_energy = np.sum(magnitude ** 2)
        if total_energy > 0:
            # 低频能量比例 (前1/4)
            low_freq_energy = np.sum(magnitude[:len(magnitude)//4] ** 2)
            features[f'{key}_low_freq_ratio'] = float(low_freq_energy / total_energy)

            # 频谱熵
            normalized_mag = magnitude / np.sum(magnitude)
            normalized_mag = normalized_mag[normalized_mag > 0]
            features[f'{key}_spectral_entropy'] = float(
                -np.sum(normalized_mag * np.log2(normalized_mag))
            )

        return features

    def extract_correlation_features(self) -> Dict[str, float]:
        """提取关联特征"""
        features = {}

        # 可用的数据对
        pairs = [
            ('tunnel_flow', 'pipe_flow'),
            ('pool_level', 'pipe_flow'),
            ('pipe_pressure', 'pipe_flow'),
            ('tunnel_flow', 'pool_level')
        ]

        for key1, key2 in pairs:
            if key1 in self.buffers and key2 in self.buffers:
                data1 = np.array(self.buffers[key1])
                data2 = np.array(self.buffers[key2])

                min_len = min(len(data1), len(data2))
                if min_len < 10:
                    continue

                data1 = data1[-min_len:]
                data2 = data2[-min_len:]

                # 相关系数
                corr = np.corrcoef(data1, data2)[0, 1]
                if not np.isnan(corr):
                    features[f'corr_{key1}_{key2}'] = float(corr)

                # 互信息估计 (简化)
                # 基于联合直方图
                try:
                    hist_2d, _, _ = np.histogram2d(data1, data2, bins=10)
                    pxy = hist_2d / np.sum(hist_2d)
                    px = np.sum(pxy, axis=1)
                    py = np.sum(pxy, axis=0)

                    mi = 0.0
                    for i in range(pxy.shape[0]):
                        for j in range(pxy.shape[1]):
                            if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                                mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
                    features[f'mi_{key1}_{key2}'] = float(mi)
                except:
                    pass

        return features

    def extract_derivative_features(self) -> Dict[str, float]:
        """提取导数相关特征"""
        features = {}

        # 流量差 (入流 - 出流)
        if 'tunnel_flow' in self.buffers and 'pipe_flow' in self.buffers:
            tunnel = np.array(self.buffers['tunnel_flow'])
            pipe = np.array(self.buffers['pipe_flow'])
            min_len = min(len(tunnel), len(pipe))

            if min_len > 5:
                diff = tunnel[-min_len:] - pipe[-min_len:]
                features['flow_imbalance_mean'] = float(np.mean(diff))
                features['flow_imbalance_std'] = float(np.std(diff))
                features['flow_imbalance_trend'] = float(
                    np.polyfit(np.arange(min_len), diff, 1)[0]
                )

        # 压力梯度
        if 'pressure_upstream' in self.buffers and 'pressure_downstream' in self.buffers:
            p_up = np.array(self.buffers['pressure_upstream'])
            p_down = np.array(self.buffers['pressure_downstream'])
            min_len = min(len(p_up), len(p_down))

            if min_len > 5:
                gradient = p_up[-min_len:] - p_down[-min_len:]
                features['pressure_gradient_mean'] = float(np.mean(gradient))
                features['pressure_gradient_std'] = float(np.std(gradient))

        return features

    def extract_all(self) -> FeatureSet:
        """提取所有特征"""
        import time

        all_features: Dict[str, float] = {}
        all_types: Dict[str, FeatureType] = {}

        # 按配置提取各变量特征
        for key, config in self.feature_configs.items():
            if key not in self.buffers:
                continue

            if config.get('statistical', False):
                stat_features = self.extract_statistical_features(key)
                all_features.update(stat_features)
                for k in stat_features:
                    all_types[k] = FeatureType.STATISTICAL

            if config.get('temporal', False):
                temp_features = self.extract_temporal_features(key)
                all_features.update(temp_features)
                for k in temp_features:
                    all_types[k] = FeatureType.TEMPORAL

            if config.get('frequency', False):
                freq_features = self.extract_frequency_features(key)
                all_features.update(freq_features)
                for k in freq_features:
                    all_types[k] = FeatureType.FREQUENCY

        # 关联特征
        corr_features = self.extract_correlation_features()
        all_features.update(corr_features)
        for k in corr_features:
            all_types[k] = FeatureType.CORRELATION

        # 导数特征
        deriv_features = self.extract_derivative_features()
        all_features.update(deriv_features)
        for k in deriv_features:
            all_types[k] = FeatureType.DERIVATIVE

        # 标准化
        if self.is_calibrated:
            all_features = self._normalize_features(all_features)

        # 置信度 (基于数据量)
        min_buffer_len = min(
            (len(b) for b in self.buffers.values()),
            default=0
        )
        confidence = min(min_buffer_len / self.window_size, 1.0)

        return FeatureSet(
            timestamp=time.time(),
            features=all_features,
            feature_types=all_types,
            confidence=confidence,
            is_valid=len(all_features) > 0
        )

    def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """标准化特征"""
        normalized = {}
        for key, value in features.items():
            # 查找对应的原始变量
            base_key = key.split('_')[0] if '_' in key else key

            if base_key in self.mean_values and base_key in self.std_values:
                normalized[key] = (value - self.mean_values[base_key]) / self.std_values[base_key]
            else:
                normalized[key] = value

        return normalized

    def get_feature_vector(self, feature_names: List[str] = None) -> np.ndarray:
        """获取特征向量"""
        feature_set = self.extract_all()

        if feature_names is None:
            feature_names = sorted(feature_set.features.keys())

        vector = np.zeros(len(feature_names))
        for i, name in enumerate(feature_names):
            vector[i] = feature_set.features.get(name, 0.0)

        return vector

    def reset(self):
        """重置缓冲"""
        for buf in self.buffers.values():
            buf.clear()
        self.timestamps.clear()
