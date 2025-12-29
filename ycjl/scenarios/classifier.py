"""
场景分类器
==========

基于提取的特征进行场景分类:
- 规则分类器
- 统计分类器
- 融合判决
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque

from ..config.settings import ScenarioType
from .features import FeatureSet, FeatureType


@dataclass
class ClassificationResult:
    """分类结果"""
    scenario: ScenarioType
    confidence: float
    probabilities: Dict[ScenarioType, float]
    evidence: Dict[str, float]
    timestamp: float


class RuleBasedClassifier:
    """
    规则分类器

    基于专家规则的场景判别
    """

    def __init__(self):
        self.rules: Dict[ScenarioType, List[callable]] = {}
        self.weights: Dict[ScenarioType, List[float]] = {}
        self._init_rules()

    def _init_rules(self):
        """初始化规则"""

        # 正常场景规则
        self.rules[ScenarioType.NORMAL] = [
            # 流量稳定
            lambda f: f.get('pipe_flow_std', 1.0) < 0.5,
            # 压力稳定
            lambda f: f.get('pipe_pressure_std', 5.0) < 3.0,
            # 无趋势
            lambda f: abs(f.get('pool_level_trend', 0)) < 0.01,
            # 流量平衡
            lambda f: abs(f.get('flow_imbalance_mean', 0)) < 1.0
        ]
        self.weights[ScenarioType.NORMAL] = [0.3, 0.3, 0.2, 0.2]

        # 需水激增场景
        self.rules[ScenarioType.DEMAND_SURGE] = [
            # 出流增加
            lambda f: f.get('pipe_flow_trend', 0) > 0.1,
            # 水位下降
            lambda f: f.get('pool_level_trend', 0) < -0.05,
            # 流量增大
            lambda f: f.get('pipe_flow_mean', 10) > 12.0,
            # 压力下降
            lambda f: f.get('pipe_pressure_trend', 0) < 0
        ]
        self.weights[ScenarioType.DEMAND_SURGE] = [0.35, 0.25, 0.25, 0.15]

        # 爆管场景
        self.rules[ScenarioType.PIPE_BURST] = [
            # 流量突变
            lambda f: f.get('pipe_flow_max_change', 0) > 2.0,
            # 压力骤降
            lambda f: f.get('pipe_pressure_trend', 0) < -0.5,
            # 流量不平衡
            lambda f: abs(f.get('flow_imbalance_mean', 0)) > 2.0,
            # 高频扰动
            lambda f: f.get('pipe_pressure_spectral_entropy', 0) > 2.0
        ]
        self.weights[ScenarioType.PIPE_BURST] = [0.3, 0.3, 0.25, 0.15]

        # 冰期场景
        self.rules[ScenarioType.ICE_PERIOD] = [
            # 低温
            lambda f: f.get('water_temperature_mean', 10) < 4.0,
            # 流阻增大 (压力梯度增大)
            lambda f: f.get('pressure_gradient_mean', 0) > 5.0,
            # 流速降低
            lambda f: f.get('pipe_flow_mean', 10) < 8.0,
            # 变化缓慢
            lambda f: f.get('pipe_flow_change_std', 1) < 0.1
        ]
        self.weights[ScenarioType.ICE_PERIOD] = [0.4, 0.25, 0.2, 0.15]

        # 电力故障场景
        self.rules[ScenarioType.POWER_FAILURE] = [
            # 流量骤降
            lambda f: f.get('pipe_flow_min_change', 0) < -3.0,
            # 压力波动
            lambda f: f.get('pipe_pressure_std', 0) > 10.0,
            # 瞬态振荡
            lambda f: f.get('pipe_pressure_zero_crossing_rate', 0) > 0.3
        ]
        self.weights[ScenarioType.POWER_FAILURE] = [0.4, 0.35, 0.25]

    def classify(self, features: Dict[str, float]) -> Dict[ScenarioType, float]:
        """分类"""
        scores: Dict[ScenarioType, float] = {}

        for scenario, rules in self.rules.items():
            weights = self.weights.get(scenario, [1.0] * len(rules))
            score = 0.0
            total_weight = 0.0

            for rule, weight in zip(rules, weights):
                try:
                    if rule(features):
                        score += weight
                except Exception:
                    pass
                total_weight += weight

            scores[scenario] = score / total_weight if total_weight > 0 else 0.0

        return scores


class StatisticalClassifier:
    """
    统计分类器

    基于概率模型的场景分类 (简化的朴素贝叶斯)
    """

    def __init__(self):
        # 各场景的特征分布参数 (均值, 标准差)
        self.feature_distributions: Dict[ScenarioType, Dict[str, Tuple[float, float]]] = {}
        self.prior: Dict[ScenarioType, float] = {}
        self._init_distributions()

    def _init_distributions(self):
        """初始化特征分布"""
        # 基于经验设定的分布参数

        # 正常场景
        self.feature_distributions[ScenarioType.NORMAL] = {
            'pipe_flow_std': (0.3, 0.1),
            'pipe_pressure_std': (2.0, 0.5),
            'pool_level_trend': (0.0, 0.005),
            'flow_imbalance_mean': (0.0, 0.5)
        }

        # 需水激增
        self.feature_distributions[ScenarioType.DEMAND_SURGE] = {
            'pipe_flow_std': (0.8, 0.2),
            'pipe_pressure_std': (4.0, 1.0),
            'pool_level_trend': (-0.1, 0.03),
            'pipe_flow_trend': (0.2, 0.05)
        }

        # 爆管
        self.feature_distributions[ScenarioType.PIPE_BURST] = {
            'pipe_flow_max_change': (5.0, 1.0),
            'pipe_pressure_trend': (-1.0, 0.3),
            'flow_imbalance_mean': (3.0, 1.0)
        }

        # 冰期
        self.feature_distributions[ScenarioType.ICE_PERIOD] = {
            'water_temperature_mean': (2.0, 1.0),
            'pipe_flow_mean': (7.0, 1.0),
            'pipe_flow_change_std': (0.05, 0.02)
        }

        # 先验概率
        self.prior = {
            ScenarioType.NORMAL: 0.7,
            ScenarioType.DEMAND_SURGE: 0.1,
            ScenarioType.PIPE_BURST: 0.02,
            ScenarioType.ICE_PERIOD: 0.15,
            ScenarioType.POWER_FAILURE: 0.03
        }

    def _gaussian_likelihood(self, x: float, mean: float, std: float) -> float:
        """高斯似然"""
        return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

    def classify(self, features: Dict[str, float]) -> Dict[ScenarioType, float]:
        """分类"""
        log_posteriors: Dict[ScenarioType, float] = {}

        for scenario, distributions in self.feature_distributions.items():
            # 对数似然
            log_likelihood = 0.0

            for feature_name, (mean, std) in distributions.items():
                if feature_name in features:
                    x = features[feature_name]
                    likelihood = self._gaussian_likelihood(x, mean, std)
                    log_likelihood += np.log(max(likelihood, 1e-10))

            # 加上先验
            log_prior = np.log(self.prior.get(scenario, 0.1))
            log_posteriors[scenario] = log_likelihood + log_prior

        # 转换为概率
        max_log = max(log_posteriors.values())
        posteriors = {}
        total = 0.0

        for scenario, log_p in log_posteriors.items():
            p = np.exp(log_p - max_log)
            posteriors[scenario] = p
            total += p

        # 归一化
        for scenario in posteriors:
            posteriors[scenario] /= total

        return posteriors


class ScenarioClassifier:
    """
    融合分类器

    组合多种分类方法进行决策
    """

    def __init__(self):
        self.rule_classifier = RuleBasedClassifier()
        self.stat_classifier = StatisticalClassifier()

        # 融合权重
        self.fusion_weights = {
            'rule': 0.6,
            'statistical': 0.4
        }

        # 置信度阈值
        self.confidence_threshold = 0.5

        # 历史 (用于平滑)
        self.history: deque = deque(maxlen=10)

        # 状态转移约束
        self.valid_transitions = {
            ScenarioType.NORMAL: [
                ScenarioType.NORMAL,
                ScenarioType.DEMAND_SURGE,
                ScenarioType.PIPE_BURST,
                ScenarioType.ICE_PERIOD,
                ScenarioType.POWER_FAILURE
            ],
            ScenarioType.DEMAND_SURGE: [
                ScenarioType.DEMAND_SURGE,
                ScenarioType.NORMAL,
                ScenarioType.PIPE_BURST
            ],
            ScenarioType.PIPE_BURST: [
                ScenarioType.PIPE_BURST,
                ScenarioType.NORMAL
            ],
            ScenarioType.ICE_PERIOD: [
                ScenarioType.ICE_PERIOD,
                ScenarioType.NORMAL
            ],
            ScenarioType.POWER_FAILURE: [
                ScenarioType.POWER_FAILURE,
                ScenarioType.NORMAL
            ]
        }

        # 当前场景
        self.current_scenario = ScenarioType.NORMAL

    def classify(self, feature_set: FeatureSet) -> ClassificationResult:
        """
        执行分类

        Parameters:
            feature_set: 特征集

        Returns:
            ClassificationResult: 分类结果
        """
        import time

        features = feature_set.features

        # 规则分类
        rule_scores = self.rule_classifier.classify(features)

        # 统计分类
        stat_probs = self.stat_classifier.classify(features)

        # 融合
        fused_probs = self._fuse_results(rule_scores, stat_probs)

        # 时序平滑
        smoothed_probs = self._temporal_smooth(fused_probs)

        # 应用状态转移约束
        constrained_probs = self._apply_transition_constraints(smoothed_probs)

        # 选择最优场景
        best_scenario = max(constrained_probs, key=constrained_probs.get)
        confidence = constrained_probs[best_scenario]

        # 收集证据
        evidence = self._collect_evidence(features, best_scenario)

        # 更新当前场景
        if confidence > self.confidence_threshold:
            self.current_scenario = best_scenario

        result = ClassificationResult(
            scenario=best_scenario,
            confidence=confidence,
            probabilities=constrained_probs,
            evidence=evidence,
            timestamp=time.time()
        )

        self.history.append(result)

        return result

    def _fuse_results(self, rule_scores: Dict[ScenarioType, float],
                      stat_probs: Dict[ScenarioType, float]) -> Dict[ScenarioType, float]:
        """融合多分类器结果"""
        fused = {}
        all_scenarios = set(rule_scores.keys()) | set(stat_probs.keys())

        for scenario in all_scenarios:
            rule_score = rule_scores.get(scenario, 0)
            stat_prob = stat_probs.get(scenario, 0)

            fused[scenario] = (
                self.fusion_weights['rule'] * rule_score +
                self.fusion_weights['statistical'] * stat_prob
            )

        # 归一化
        total = sum(fused.values())
        if total > 0:
            for scenario in fused:
                fused[scenario] /= total

        return fused

    def _temporal_smooth(self, probs: Dict[ScenarioType, float]) -> Dict[ScenarioType, float]:
        """时序平滑"""
        if len(self.history) < 3:
            return probs

        # 指数移动平均
        alpha = 0.7  # 当前权重

        smoothed = {}
        for scenario in probs:
            current = probs[scenario]

            # 历史平均
            history_avg = np.mean([
                r.probabilities.get(scenario, 0)
                for r in self.history
            ])

            smoothed[scenario] = alpha * current + (1 - alpha) * history_avg

        return smoothed

    def _apply_transition_constraints(self, probs: Dict[ScenarioType, float]) -> Dict[ScenarioType, float]:
        """应用状态转移约束"""
        valid = self.valid_transitions.get(self.current_scenario, list(probs.keys()))

        constrained = {}
        for scenario, prob in probs.items():
            if scenario in valid:
                constrained[scenario] = prob
            else:
                # 降低无效转移的概率
                constrained[scenario] = prob * 0.1

        # 归一化
        total = sum(constrained.values())
        if total > 0:
            for scenario in constrained:
                constrained[scenario] /= total

        return constrained

    def _collect_evidence(self, features: Dict[str, float],
                          scenario: ScenarioType) -> Dict[str, float]:
        """收集证据"""
        evidence = {}

        # 根据场景收集关键特征作为证据
        if scenario == ScenarioType.DEMAND_SURGE:
            evidence['pipe_flow_trend'] = features.get('pipe_flow_trend', 0)
            evidence['pool_level_trend'] = features.get('pool_level_trend', 0)

        elif scenario == ScenarioType.PIPE_BURST:
            evidence['pipe_flow_max_change'] = features.get('pipe_flow_max_change', 0)
            evidence['pipe_pressure_trend'] = features.get('pipe_pressure_trend', 0)
            evidence['flow_imbalance'] = features.get('flow_imbalance_mean', 0)

        elif scenario == ScenarioType.ICE_PERIOD:
            evidence['water_temperature'] = features.get('water_temperature_mean', 10)
            evidence['flow_reduction'] = 10 - features.get('pipe_flow_mean', 10)

        elif scenario == ScenarioType.POWER_FAILURE:
            evidence['flow_drop'] = features.get('pipe_flow_min_change', 0)
            evidence['pressure_oscillation'] = features.get('pipe_pressure_std', 0)

        return evidence

    def get_current_scenario(self) -> ScenarioType:
        """获取当前场景"""
        return self.current_scenario

    def force_scenario(self, scenario: ScenarioType):
        """强制设置场景"""
        self.current_scenario = scenario

    def reset(self):
        """重置"""
        self.history.clear()
        self.current_scenario = ScenarioType.NORMAL
