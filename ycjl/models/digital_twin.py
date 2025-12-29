"""
数字孪生系统
============

实时同步物理系统状态:
- 模型-实测对比
- 参数自适应
- 预测性分析
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum, auto

from ..config.settings import Config
from ..estimation.ekf import ExtendedKalmanFilter
from ..estimation.parameter_id import ManningEstimator, FrictionEstimator
from .reduced_order import ReducedOrderModel, TunnelReducedModel, PipelineReducedModel


class TwinSyncStatus(Enum):
    """同步状态"""
    SYNCHRONIZED = auto()
    DRIFTING = auto()
    DIVERGED = auto()
    UPDATING = auto()


@dataclass
class TwinState:
    """数字孪生状态"""
    physical_state: Dict          # 物理状态
    model_state: Dict             # 模型状态
    residual: Dict                # 残差
    sync_status: TwinSyncStatus   # 同步状态
    confidence: float             # 置信度
    parameters: Dict              # 辨识参数


class DigitalTwin:
    """
    数字孪生系统

    功能:
    - 实时状态估计
    - 参数在线辨识
    - 故障预测
    - 优化决策支持
    """

    def __init__(self):
        self.cfg = Config

        # 子模型
        self.tunnel_model = TunnelReducedModel(num_modes=4)
        self.pipeline_model = PipelineReducedModel(num_segments=4)

        # 状态估计器
        self.ekf = ExtendedKalmanFilter(state_dim=4, meas_dim=2)

        # 参数辨识器
        self.manning_estimator = ManningEstimator()
        self.friction_estimator = FrictionEstimator()

        # 物理状态 (从传感器获取)
        self.physical_state: Dict = {}

        # 模型状态
        self.model_state: Dict = {
            'tunnel_flow': 10.0,
            'pool_level': 5.0,
            'pipe_pressure': 50.0,
            'pipe_flow': 10.0
        }

        # 残差阈值
        self.residual_thresholds = {
            'flow': 1.0,      # m³/s
            'level': 0.3,     # m
            'pressure': 5.0   # m
        }

        # 同步状态
        self.sync_status = TwinSyncStatus.SYNCHRONIZED
        self.sync_confidence = 1.0

        # 历史
        self.history: List[TwinState] = []

    def update_physical_state(self, measurements: Dict):
        """
        更新物理状态

        Parameters:
            measurements: 传感器测量值
        """
        self.physical_state = measurements.copy()

    def step(self, dt: float, control_inputs: Dict) -> TwinState:
        """
        推进一个时间步

        Parameters:
            dt: 时间步长
            control_inputs: 控制输入

        Returns:
            TwinState: 当前状态
        """
        # 1. 模型预测
        self._predict_model(dt, control_inputs)

        # 2. 计算残差
        residuals = self._compute_residuals()

        # 3. 状态同化
        self._assimilate_state(residuals, dt)

        # 4. 参数辨识
        self._update_parameters(dt)

        # 5. 评估同步状态
        self._evaluate_sync_status(residuals)

        # 构建状态
        state = TwinState(
            physical_state=self.physical_state.copy(),
            model_state=self.model_state.copy(),
            residual=residuals,
            sync_status=self.sync_status,
            confidence=self.sync_confidence,
            parameters={
                'manning_n': self.manning_estimator.value,
                'darcy_f': self.friction_estimator.value
            }
        )

        self.history.append(state)
        return state

    def _predict_model(self, dt: float, inputs: Dict):
        """模型预测"""
        # 隧洞模型
        u_tunnel = np.array([
            inputs.get('source_flow', 10.0),
            inputs.get('pool_level', 5.0)
        ])
        y_tunnel = self.tunnel_model.step(u_tunnel, dt)
        self.model_state['tunnel_flow'] = y_tunnel[0]

        # 管道模型
        u_pipe = np.array([
            inputs.get('valve_mid', 0.5),
            inputs.get('valve_end', 1.0)
        ])
        y_pipe = self.pipeline_model.step(u_pipe, dt)
        self.model_state['pipe_flow'] = y_pipe[0]
        self.model_state['pipe_pressure'] = y_pipe[1] * 50 + 50  # 缩放

    def _compute_residuals(self) -> Dict:
        """计算模型-实测残差"""
        residuals = {}

        # 流量残差
        if 'tunnel_flow' in self.physical_state:
            residuals['tunnel_flow'] = (
                self.physical_state['tunnel_flow'] -
                self.model_state['tunnel_flow']
            )

        # 水位残差
        if 'pool_level' in self.physical_state:
            residuals['pool_level'] = (
                self.physical_state['pool_level'] -
                self.model_state.get('pool_level', 5.0)
            )

        # 压力残差
        if 'pipe_pressure' in self.physical_state:
            residuals['pipe_pressure'] = (
                self.physical_state['pipe_pressure'] -
                self.model_state['pipe_pressure']
            )

        # 管道流量残差
        if 'pipe_flow' in self.physical_state:
            residuals['pipe_flow'] = (
                self.physical_state['pipe_flow'] -
                self.model_state['pipe_flow']
            )

        return residuals

    def _assimilate_state(self, residuals: Dict, dt: float):
        """状态同化"""
        # 使用简单的比例校正
        alpha = 0.1  # 校正增益

        if 'tunnel_flow' in residuals:
            self.model_state['tunnel_flow'] += alpha * residuals['tunnel_flow']

        if 'pipe_flow' in residuals:
            self.model_state['pipe_flow'] += alpha * residuals['pipe_flow']

        if 'pipe_pressure' in residuals:
            self.model_state['pipe_pressure'] += alpha * residuals['pipe_pressure']

    def _update_parameters(self, dt: float):
        """更新参数估计"""
        # 曼宁糙率
        if 'tunnel_flow' in self.physical_state and 'tunnel_depth' in self.physical_state:
            manning_meas = {
                'flow': self.physical_state['tunnel_flow'],
                'depth': self.physical_state.get('tunnel_depth', 3.5),
                'slope': Config.tunnel.bottom_slope
            }
            self.manning_estimator.update(manning_meas, dt)

        # 摩阻系数
        if 'pipe_flow' in self.physical_state and 'H_up' in self.physical_state:
            friction_meas = {
                'flow': self.physical_state['pipe_flow'],
                'H_up': self.physical_state.get('H_up', 60.0),
                'H_down': self.physical_state.get('H_down', 50.0),
                'length': Config.pipeline.total_length / 2
            }
            self.friction_estimator.update(friction_meas, dt)

    def _evaluate_sync_status(self, residuals: Dict):
        """评估同步状态"""
        max_normalized_residual = 0.0

        for key, value in residuals.items():
            threshold = self.residual_thresholds.get(
                key.split('_')[-1],  # 提取类型
                1.0
            )
            normalized = abs(value) / threshold
            max_normalized_residual = max(max_normalized_residual, normalized)

        # 更新同步状态
        if max_normalized_residual < 0.5:
            self.sync_status = TwinSyncStatus.SYNCHRONIZED
            self.sync_confidence = 1.0 - max_normalized_residual
        elif max_normalized_residual < 1.0:
            self.sync_status = TwinSyncStatus.DRIFTING
            self.sync_confidence = 1.0 - max_normalized_residual
        else:
            self.sync_status = TwinSyncStatus.DIVERGED
            self.sync_confidence = max(0.0, 1.0 - max_normalized_residual * 0.5)

    def predict_future(self, horizon: float, dt: float,
                       future_inputs: Dict) -> List[Dict]:
        """
        预测未来状态

        Parameters:
            horizon: 预测时域 (s)
            dt: 时间步长
            future_inputs: 未来控制输入

        Returns:
            预测状态序列
        """
        predictions = []
        steps = int(horizon / dt)

        # 保存当前状态
        saved_tunnel_state = self.tunnel_model.x.copy()
        saved_pipe_state = self.pipeline_model.x.copy()
        saved_model_state = self.model_state.copy()

        for k in range(steps):
            self._predict_model(dt, future_inputs)
            predictions.append(self.model_state.copy())

        # 恢复状态
        self.tunnel_model.x = saved_tunnel_state
        self.pipeline_model.x = saved_pipe_state
        self.model_state = saved_model_state

        return predictions

    def detect_anomaly(self) -> Tuple[bool, str, float]:
        """
        检测异常

        Returns:
            (是否异常, 异常类型, 严重程度)
        """
        if len(self.history) < 10:
            return False, "", 0.0

        recent_residuals = [s.residual for s in self.history[-10:]]

        # 检查残差趋势
        for key in ['tunnel_flow', 'pipe_flow', 'pipe_pressure']:
            values = [r.get(key, 0) for r in recent_residuals]
            if len(values) < 5:
                continue

            # 均值和趋势
            mean_residual = np.mean(values)
            trend = np.polyfit(range(len(values)), values, 1)[0]

            # 突增检测
            if abs(mean_residual) > self.residual_thresholds.get(key.split('_')[-1], 1.0):
                severity = abs(mean_residual) / self.residual_thresholds.get(key.split('_')[-1], 1.0)
                return True, f"{key}_deviation", severity

            # 趋势检测
            if abs(trend) > 0.1:
                return True, f"{key}_trending", abs(trend)

        return False, "", 0.0

    def get_model_fidelity(self) -> Dict[str, float]:
        """
        获取模型保真度

        Returns:
            各变量的保真度 (0~1)
        """
        fidelity = {}

        if len(self.history) < 20:
            return {'overall': 0.5}

        recent = self.history[-20:]

        for key in ['tunnel_flow', 'pipe_flow', 'pipe_pressure']:
            residuals = [abs(s.residual.get(key, 0)) for s in recent]
            mean_residual = np.mean(residuals)
            threshold = self.residual_thresholds.get(key.split('_')[-1], 1.0)
            fidelity[key] = max(0, 1.0 - mean_residual / threshold)

        fidelity['overall'] = np.mean(list(fidelity.values()))
        return fidelity

    def reset(self):
        """重置"""
        self.tunnel_model.reset()
        self.pipeline_model.reset()
        self.manning_estimator.reset()
        self.friction_estimator.reset()
        self.physical_state.clear()
        self.model_state = {
            'tunnel_flow': 10.0,
            'pool_level': 5.0,
            'pipe_pressure': 50.0,
            'pipe_flow': 10.0
        }
        self.sync_status = TwinSyncStatus.SYNCHRONIZED
        self.sync_confidence = 1.0
        self.history.clear()
