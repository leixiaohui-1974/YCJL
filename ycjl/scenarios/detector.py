"""
场景检测器
==========

综合特征提取和分类器,实现端到端场景检测:
- 在线检测
- 异常检测
- 预警生成
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import time

from ..config.settings import ScenarioType
from .features import FeatureExtractor, FeatureSet
from .classifier import ScenarioClassifier, ClassificationResult


class AlertLevel(Enum):
    """报警级别"""
    INFO = auto()
    WARNING = auto()
    ALARM = auto()
    CRITICAL = auto()


@dataclass
class DetectionResult:
    """检测结果"""
    timestamp: float
    scenario: ScenarioType
    confidence: float
    alert_level: AlertLevel
    message: str
    features: Dict[str, float]
    evidence: Dict[str, float]
    recommendations: List[str]


@dataclass
class ScenarioTransition:
    """场景转移记录"""
    timestamp: float
    from_scenario: ScenarioType
    to_scenario: ScenarioType
    confidence: float
    trigger_features: Dict[str, float]


class ScenarioDetector:
    """
    场景检测器

    功能:
    - 实时场景识别
    - 异常早期预警
    - 趋势预测
    - 建议生成
    """

    def __init__(self, window_size: int = 100):
        # 特征提取器
        self.feature_extractor = FeatureExtractor(window_size)

        # 分类器
        self.classifier = ScenarioClassifier()

        # 当前状态
        self.current_scenario = ScenarioType.NORMAL
        self.scenario_start_time = time.time()
        self.scenario_duration = 0.0

        # 历史
        self.detection_history: deque = deque(maxlen=1000)
        self.transition_history: List[ScenarioTransition] = []

        # 报警阈值
        self.alert_thresholds = {
            ScenarioType.NORMAL: 0.8,
            ScenarioType.DEMAND_SURGE: 0.6,
            ScenarioType.PIPE_BURST: 0.4,
            ScenarioType.ICE_PERIOD: 0.7,
            ScenarioType.POWER_FAILURE: 0.5
        }

        # 场景报警级别
        self.scenario_alert_levels = {
            ScenarioType.NORMAL: AlertLevel.INFO,
            ScenarioType.DEMAND_SURGE: AlertLevel.WARNING,
            ScenarioType.PIPE_BURST: AlertLevel.CRITICAL,
            ScenarioType.ICE_PERIOD: AlertLevel.ALARM,
            ScenarioType.POWER_FAILURE: AlertLevel.CRITICAL
        }

        # 建议模板
        self.recommendation_templates = self._init_recommendations()

        # 统计
        self.detection_count = 0
        self.scenario_counts: Dict[ScenarioType, int] = {s: 0 for s in ScenarioType}

    def _init_recommendations(self) -> Dict[ScenarioType, List[str]]:
        """初始化建议模板"""
        return {
            ScenarioType.NORMAL: [
                "系统运行正常，保持当前控制策略"
            ],
            ScenarioType.DEMAND_SURGE: [
                "检测到需水激增，建议增大进水闸门开度",
                "监控稳流池水位，防止过度下降",
                "检查末端用户是否有异常用水",
                "准备启用备用水源"
            ],
            ScenarioType.PIPE_BURST: [
                "疑似管道破裂！立即启动应急程序",
                "分段关闭阀门，隔离故障区域",
                "派遣巡检人员现场确认",
                "通知相关调度部门"
            ],
            ScenarioType.ICE_PERIOD: [
                "进入冰期运行模式",
                "降低流速，防止冰塞",
                "加强水温监测",
                "准备融冰设备"
            ],
            ScenarioType.POWER_FAILURE: [
                "检测到电力故障！启动应急电源",
                "切换到备用控制模式",
                "关闭非必要设备",
                "监控水锤压力"
            ]
        }

    def update(self, measurements: Dict[str, float], timestamp: float = None):
        """
        更新检测器

        Parameters:
            measurements: 测量数据
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = time.time()

        self.feature_extractor.update(measurements, timestamp)

    def detect(self) -> DetectionResult:
        """
        执行场景检测

        Returns:
            DetectionResult: 检测结果
        """
        current_time = time.time()

        # 提取特征
        feature_set = self.feature_extractor.extract_all()

        # 分类
        classification = self.classifier.classify(feature_set)

        # 检查场景转移
        if classification.scenario != self.current_scenario:
            if classification.confidence > self.alert_thresholds.get(
                classification.scenario, 0.5
            ):
                self._record_transition(
                    self.current_scenario,
                    classification.scenario,
                    classification.confidence,
                    feature_set.features
                )
                self.current_scenario = classification.scenario
                self.scenario_start_time = current_time

        # 更新持续时间
        self.scenario_duration = current_time - self.scenario_start_time

        # 确定报警级别
        alert_level = self._determine_alert_level(classification)

        # 生成消息
        message = self._generate_message(classification, alert_level)

        # 获取建议
        recommendations = self._get_recommendations(classification.scenario)

        result = DetectionResult(
            timestamp=current_time,
            scenario=classification.scenario,
            confidence=classification.confidence,
            alert_level=alert_level,
            message=message,
            features=feature_set.features,
            evidence=classification.evidence,
            recommendations=recommendations
        )

        # 记录
        self.detection_history.append(result)
        self.detection_count += 1
        self.scenario_counts[classification.scenario] += 1

        return result

    def _record_transition(self, from_scenario: ScenarioType,
                           to_scenario: ScenarioType,
                           confidence: float,
                           features: Dict[str, float]):
        """记录场景转移"""
        transition = ScenarioTransition(
            timestamp=time.time(),
            from_scenario=from_scenario,
            to_scenario=to_scenario,
            confidence=confidence,
            trigger_features=features.copy()
        )
        self.transition_history.append(transition)

    def _determine_alert_level(self, classification: ClassificationResult) -> AlertLevel:
        """确定报警级别"""
        base_level = self.scenario_alert_levels.get(
            classification.scenario,
            AlertLevel.INFO
        )

        # 根据置信度调整
        if classification.confidence < 0.5:
            # 降低级别
            if base_level == AlertLevel.CRITICAL:
                return AlertLevel.ALARM
            elif base_level == AlertLevel.ALARM:
                return AlertLevel.WARNING

        # 根据持续时间调整
        if self.scenario_duration > 300:  # 超过5分钟
            if base_level == AlertLevel.WARNING:
                return AlertLevel.ALARM

        return base_level

    def _generate_message(self, classification: ClassificationResult,
                          alert_level: AlertLevel) -> str:
        """生成消息"""
        scenario_names = {
            ScenarioType.NORMAL: "正常运行",
            ScenarioType.DEMAND_SURGE: "需水激增",
            ScenarioType.PIPE_BURST: "管道破裂",
            ScenarioType.ICE_PERIOD: "冰期运行",
            ScenarioType.POWER_FAILURE: "电力故障"
        }

        level_prefixes = {
            AlertLevel.INFO: "[信息]",
            AlertLevel.WARNING: "[警告]",
            AlertLevel.ALARM: "[报警]",
            AlertLevel.CRITICAL: "[紧急]"
        }

        scenario_name = scenario_names.get(classification.scenario, "未知")
        prefix = level_prefixes.get(alert_level, "")

        message = f"{prefix} 检测到{scenario_name}场景 (置信度: {classification.confidence:.2f})"

        if classification.evidence:
            evidence_str = ", ".join([
                f"{k}={v:.2f}" for k, v in list(classification.evidence.items())[:3]
            ])
            message += f" | 证据: {evidence_str}"

        return message

    def _get_recommendations(self, scenario: ScenarioType) -> List[str]:
        """获取建议"""
        return self.recommendation_templates.get(scenario, [])

    def predict_scenario(self, horizon: int = 10) -> Dict[ScenarioType, float]:
        """
        预测未来场景

        基于当前趋势预测

        Parameters:
            horizon: 预测步数

        Returns:
            各场景概率
        """
        if len(self.detection_history) < 5:
            return {self.current_scenario: 1.0}

        # 分析历史趋势
        recent = list(self.detection_history)[-20:]

        # 统计各场景出现频率
        freq: Dict[ScenarioType, int] = {}
        for r in recent:
            freq[r.scenario] = freq.get(r.scenario, 0) + 1

        # 考虑趋势
        # 如果最近的场景与当前不同，增加转移概率
        if len(recent) >= 3:
            recent_scenarios = [r.scenario for r in recent[-3:]]
            if recent_scenarios[-1] != self.current_scenario:
                freq[recent_scenarios[-1]] = freq.get(recent_scenarios[-1], 0) + 5

        # 归一化
        total = sum(freq.values())
        probs = {s: f / total for s, f in freq.items()}

        return probs

    def get_anomaly_score(self) -> float:
        """
        获取异常分数

        Returns:
            异常分数 (0~1, 越高越异常)
        """
        if not self.detection_history:
            return 0.0

        recent = list(self.detection_history)[-10:]

        # 基于场景异常程度
        scenario_anomaly = {
            ScenarioType.NORMAL: 0.0,
            ScenarioType.DEMAND_SURGE: 0.4,
            ScenarioType.PIPE_BURST: 1.0,
            ScenarioType.ICE_PERIOD: 0.3,
            ScenarioType.POWER_FAILURE: 0.9
        }

        # 加权平均
        scores = [
            scenario_anomaly.get(r.scenario, 0.5) * r.confidence
            for r in recent
        ]

        return np.mean(scores)

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'detection_count': self.detection_count,
            'current_scenario': self.current_scenario.name,
            'scenario_duration': self.scenario_duration,
            'scenario_counts': {s.name: c for s, c in self.scenario_counts.items()},
            'transition_count': len(self.transition_history),
            'anomaly_score': self.get_anomaly_score()
        }

    def get_recent_transitions(self, n: int = 5) -> List[Dict]:
        """获取最近的场景转移"""
        recent = self.transition_history[-n:]
        return [
            {
                'timestamp': t.timestamp,
                'from': t.from_scenario.name,
                'to': t.to_scenario.name,
                'confidence': t.confidence
            }
            for t in recent
        ]

    def calibrate(self, normal_data: List[Dict[str, float]]):
        """
        校准检测器

        使用正常运行数据校准特征提取器

        Parameters:
            normal_data: 正常运行数据
        """
        self.feature_extractor.calibrate(normal_data)

    def reset(self):
        """重置"""
        self.feature_extractor.reset()
        self.classifier.reset()
        self.current_scenario = ScenarioType.NORMAL
        self.scenario_start_time = time.time()
        self.scenario_duration = 0.0
        self.detection_history.clear()
        self.transition_history.clear()
        self.detection_count = 0
        self.scenario_counts = {s: 0 for s in ScenarioType}
