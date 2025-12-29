"""
场景识别模块
============

多模态数据融合场景识别:
- 特征提取
- 场景分类
- 状态转移检测
- 自适应响应
"""

from .detector import ScenarioDetector, DetectionResult
from .classifier import ScenarioClassifier, ClassificationResult
from .features import FeatureExtractor, FeatureSet
from .scenarios import (
    ScenarioHandler,
    NormalScenarioHandler,
    DemandSurgeHandler,
    PipeBurstHandler,
    IcePeriodHandler,
    PowerFailureHandler
)

__all__ = [
    'ScenarioDetector',
    'DetectionResult',
    'ScenarioClassifier',
    'ClassificationResult',
    'FeatureExtractor',
    'FeatureSet',
    'ScenarioHandler',
    'NormalScenarioHandler',
    'DemandSurgeHandler',
    'PipeBurstHandler',
    'IcePeriodHandler',
    'PowerFailureHandler'
]
