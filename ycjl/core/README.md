# 水利工程智能输水系统核心框架 (Core Framework)

版本: 2.0.0

## 概述

本核心框架提供可复用的基础设施，支持多个水利工程项目的L5级自主运行系统开发。

## 已支持的项目

- **引绰济辽工程** - 超大型跨流域调水工程
- **密云水库调蓄工程** - 城市供水泵站系统

## 模块结构

```
ycjl/core/
├── __init__.py          # 模块入口
├── constants.py         # 物理常数
├── interpolators.py     # 插值器工厂
├── base_config.py       # 配置基类
├── base_physics.py      # 物理模型基类
├── base_simulation.py   # 仿真引擎基类
├── base_scheduler.py    # 调度器基类 (含增强接口v2.0)
├── gap_analyzer.py      # 数据完备性诊断器
└── README.md           # 本文档
```

## 快速开始

```python
from ycjl.core import (
    # 物理常数
    PhysicsConstants, GRAVITY, WATER_DENSITY,

    # 插值器
    create_interpolator, create_bilinear_interpolator,

    # 物理模型基类
    BaseReservoir, BasePipeline, BasePumpStation,

    # 调度器接口
    IEnhancedScheduler, HealthLevel, calculate_health_score,

    # 数据诊断
    DataGapReport, DataReadinessLevel
)

# 创建插值器
zv_curve = create_interpolator([(100, 0), (150, 1e8), (200, 5e8)])
volume = zv_curve(175)  # 水位175m对应的库容

# 计算健康得分
score = calculate_health_score(
    data_completeness=0.93,
    constraint_violations=1,
    critical_scenarios=0
)
print(f"系统健康得分: {score:.1f}")
```

## 核心组件

### 1. 物理常数 (constants.py)

```python
from ycjl.core import PhysicsConstants

g = PhysicsConstants.GRAVITY              # 9.80665 m/s²
rho = PhysicsConstants.WATER_DENSITY_20C  # 998.2 kg/m³
K = PhysicsConstants.BULK_MODULUS         # 2.2e9 Pa
```

### 2. 插值器工厂 (interpolators.py)

支持PCHIP（保单调）、线性、双线性插值：

```python
from ycjl.core import create_interpolator, InterpolatorType

# 创建PCHIP插值器（默认，保单调性）
interp = create_interpolator(data_points, method=InterpolatorType.PCHIP)

# 创建双线性插值器（用于Hill Chart等）
from ycjl.core import create_bilinear_interpolator
hill_chart = create_bilinear_interpolator(x_vals, y_vals, z_matrix)
```

### 3. 物理模型基类 (base_physics.py)

```python
from ycjl.core import BaseReservoir, BasePipeline, BasePumpStation

# 水库模型
reservoir = BaseReservoir("测试水库", normal_level=150, dead_level=100)
state = reservoir.update(dt=3600, inflow=20, outflow=15)

# 管道模型
pipe = BasePipeline("输水管道", length=10000, diameter=2.4)
head_loss = pipe.get_head_loss(flow=15.0)
```

### 4. 增强调度器接口 (base_scheduler.py v2.0)

```python
from ycjl.core import IEnhancedScheduler, HealthLevel

class MyScheduler(IEnhancedScheduler):
    def check_monthly_constraints(self, value, month=None):
        # 实现月度约束检查
        pass

    def detect_scenarios(self, state):
        # 实现场景检测
        pass

    def get_scenario_response(self, scenario_id):
        # 返回场景响应措施
        pass

    def check_data_readiness(self):
        # 检查数据完备性
        pass
```

### 5. 数据完备性诊断 (gap_analyzer.py)

```python
from ycjl.core import BaseGapAnalyzer, DataReadinessLevel

class MyGapAnalyzer(BaseGapAnalyzer):
    def analyze(self):
        # 分析数据完备性
        report = DataGapReport(...)
        return report
```

## 数据就绪等级

| 等级 | 名称 | 说明 |
|------|------|------|
| L0 | 不可用 | 关键数据缺失 |
| L1 | 最小可用 | 支持手动控制 |
| L2 | 部分可用 | 支持辅助决策 |
| L3 | 可运行 | 支持自动控制 |
| L4 | 已优化 | 支持自适应控制 |
| L5 | 全自主 | 支持L5级无人驾驶 |

## 健康等级

| 得分 | 等级 | 说明 |
|------|------|------|
| 90-100 | 优秀 | 系统运行最佳 |
| 75-89 | 良好 | 正常运行 |
| 60-74 | 一般 | 需要关注 |
| 40-59 | 警告 | 需要干预 |
| 0-39 | 严重 | 紧急处理 |

## 扩展新项目

1. 创建项目配置类，继承 `BaseProjectConfig`
2. 创建物理模型，继承对应基类
3. 实现 `IEnhancedScheduler` 接口
4. 实现 `BaseGapAnalyzer` 进行数据诊断

## 许可证

内部使用
