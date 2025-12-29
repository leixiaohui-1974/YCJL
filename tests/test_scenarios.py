"""
场景识别与处理测试
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/YCJL')

from ycjl.scenarios.features import FeatureExtractor, FeatureType
from ycjl.scenarios.classifier import ScenarioClassifier, RuleBasedClassifier
from ycjl.scenarios.detector import ScenarioDetector, AlertLevel
from ycjl.scenarios.scenarios import (
    ScenarioManager, DemandSurgeHandler, PipeBurstHandler,
    IcePeriodHandler, PowerFailureHandler
)
from ycjl.config.settings import ScenarioType


class TestFeatureExtractor:
    """特征提取器测试"""

    def test_initialization(self):
        """测试初始化"""
        extractor = FeatureExtractor(window_size=50)
        assert extractor.window_size == 50

    def test_update(self):
        """测试数据更新"""
        extractor = FeatureExtractor()

        for i in range(20):
            data = {
                'pipe_flow': 10.0 + np.random.randn() * 0.1,
                'pipe_pressure': 50.0 + np.random.randn() * 0.5
            }
            extractor.update(data, float(i))

        assert 'pipe_flow' in extractor.buffers
        assert len(extractor.buffers['pipe_flow']) == 20

    def test_statistical_features(self):
        """测试统计特征"""
        extractor = FeatureExtractor()

        for i in range(50):
            extractor.update({'pipe_flow': 10.0 + np.sin(i * 0.1)}, float(i))

        features = extractor.extract_statistical_features('pipe_flow')

        assert 'pipe_flow_mean' in features
        assert 'pipe_flow_std' in features
        assert 'pipe_flow_min' in features
        assert 'pipe_flow_max' in features

    def test_temporal_features(self):
        """测试时序特征"""
        extractor = FeatureExtractor()

        # 上升趋势
        for i in range(50):
            extractor.update({'pipe_flow': 10.0 + i * 0.1}, float(i))

        features = extractor.extract_temporal_features('pipe_flow')

        assert 'pipe_flow_trend' in features
        assert features['pipe_flow_trend'] > 0  # 应该有正趋势

    def test_frequency_features(self):
        """测试频域特征"""
        extractor = FeatureExtractor()

        # 周期信号
        for i in range(64):
            extractor.update({
                'pipe_pressure': 50.0 + 5.0 * np.sin(2 * np.pi * i / 10)
            }, float(i))

        features = extractor.extract_frequency_features('pipe_pressure')

        assert 'pipe_pressure_dominant_freq' in features


class TestScenarioClassifier:
    """场景分类器测试"""

    def test_initialization(self):
        """测试初始化"""
        classifier = ScenarioClassifier()
        assert classifier.current_scenario == ScenarioType.NORMAL

    def test_normal_classification(self):
        """测试正常场景分类"""
        classifier = ScenarioClassifier()

        # 正常特征
        from ycjl.scenarios.features import FeatureSet
        features = FeatureSet(
            timestamp=0,
            features={
                'pipe_flow_std': 0.2,
                'pipe_pressure_std': 1.0,
                'pool_level_trend': 0.001,
                'flow_imbalance_mean': 0.1
            },
            feature_types={},
            confidence=1.0,
            is_valid=True
        )

        result = classifier.classify(features)

        # 应该识别为正常
        assert result.scenario == ScenarioType.NORMAL or \
               result.probabilities[ScenarioType.NORMAL] > 0.5

    def test_demand_surge_classification(self):
        """测试需水激增分类"""
        classifier = ScenarioClassifier()

        from ycjl.scenarios.features import FeatureSet
        features = FeatureSet(
            timestamp=0,
            features={
                'pipe_flow_trend': 0.3,
                'pool_level_trend': -0.1,
                'pipe_flow_mean': 15.0,
                'pipe_pressure_trend': -0.05
            },
            feature_types={},
            confidence=1.0,
            is_valid=True
        )

        result = classifier.classify(features)

        # 应该有较高的需水激增概率
        assert result.probabilities.get(ScenarioType.DEMAND_SURGE, 0) > 0.3 or \
               result.scenario == ScenarioType.DEMAND_SURGE


class TestScenarioDetector:
    """场景检测器测试"""

    def test_initialization(self):
        """测试初始化"""
        detector = ScenarioDetector()
        assert detector.current_scenario == ScenarioType.NORMAL

    def test_update_and_detect(self):
        """测试更新和检测"""
        detector = ScenarioDetector()

        # 正常数据
        for i in range(50):
            measurements = {
                'pipe_flow': 10.0 + np.random.randn() * 0.1,
                'pipe_pressure': 50.0 + np.random.randn() * 0.5,
                'pool_level': 5.0 + np.random.randn() * 0.05
            }
            detector.update(measurements, float(i))

        result = detector.detect()

        assert result.scenario is not None
        assert result.confidence >= 0

    def test_anomaly_detection(self):
        """测试异常检测"""
        detector = ScenarioDetector()

        # 先填充正常数据
        for i in range(30):
            detector.update({
                'pipe_flow': 10.0,
                'pipe_pressure': 50.0
            }, float(i))
            detector.detect()

        # 注入异常
        for i in range(20):
            detector.update({
                'pipe_flow': 15.0,  # 流量突增
                'pipe_pressure': 40.0  # 压力下降
            }, float(30 + i))
            detector.detect()

        # 检查异常分数
        anomaly_score = detector.get_anomaly_score()
        assert anomaly_score >= 0


class TestScenarioHandlers:
    """场景处理器测试"""

    def test_demand_surge_handler(self):
        """测试需水激增处理器"""
        handler = DemandSurgeHandler()

        # 正常状态 - 不应触发
        state = {'pipe_flow': 10.0, 'flow_baseline': 10.0}
        satisfied, confidence = handler.evaluate_entry_conditions(state)
        assert not satisfied

        # 激增状态 - 应触发
        state = {
            'pipe_flow': 16.0,
            'flow_baseline': 10.0,
            'pool_level_trend': -0.1,
            'pressure_trend': -0.2
        }
        satisfied, confidence = handler.evaluate_entry_conditions(state)
        assert satisfied

    def test_pipe_burst_handler(self):
        """测试爆管处理器"""
        handler = PipeBurstHandler()

        # 爆管状态
        state = {
            'flow_rate_of_change': 5.0,
            'pressure_trend': -1.0,
            'flow_imbalance': 3.0
        }
        satisfied, confidence = handler.evaluate_entry_conditions(state)
        assert satisfied

        # 生成响应
        handler.activate(0)
        response = handler.generate_response(state)

        assert len(response.control_recommendations) > 0
        assert len(response.notifications) > 0

    def test_ice_period_handler(self):
        """测试冰期处理器"""
        handler = IcePeriodHandler()

        # 冰期状态
        state = {
            'water_temperature': 2.0,
            'friction_increase': 0.2,
            'is_winter': True
        }
        satisfied, confidence = handler.evaluate_entry_conditions(state)
        assert satisfied

    def test_scenario_manager(self):
        """测试场景管理器"""
        manager = ScenarioManager()

        # 初始应该是正常场景
        assert manager.get_current_scenario() == ScenarioType.NORMAL

        # 评估场景
        state = {
            'pipe_flow': 16.0,
            'flow_baseline': 10.0,
            'pool_level_trend': -0.1,
            'pressure_trend': -0.2
        }
        scenario, confidence = manager.evaluate_scenario(state)

        # 可能识别为需水激增
        assert scenario in [ScenarioType.NORMAL, ScenarioType.DEMAND_SURGE]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
