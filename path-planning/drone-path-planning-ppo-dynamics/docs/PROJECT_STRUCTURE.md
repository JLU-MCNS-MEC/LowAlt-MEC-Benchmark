# 项目结构说明

## 目录结构

```
drone-path-planning-ppo-dynamics/
├── core/                          # 核心功能模块
│   ├── __init__.py               # 模块初始化
│   ├── environment.py           # 无人机路径规划环境（动力学模型）
│   └── ppo_agent.py              # PPO算法实现
│
├── scripts/                       # 脚本文件
│   ├── __init__.py               # 模块初始化
│   ├── train.py                  # 训练脚本
│   └── test.py                   # 测试脚本
│
├── utils/                         # 工具脚本
│   ├── __init__.py               # 模块初始化
│   ├── analyze_scenario.py       # 场景分析脚本
│   ├── plot_scenario_reward.py   # 场景奖励绘图脚本
│   └── eval_plot_episodes.py     # 评估和可视化脚本
│
├── docs/                          # 文档目录
│   ├── analysis/                 # 分析文档
│   │   ├── ACTION_OBSERVATION_REWARD_ANALYSIS.md
│   │   ├── TRAINING_RESULT_ANALYSIS.md
│   │   └── ...
│   ├── fixes/                    # 修复方案文档
│   ├── improvements/             # 训练改进文档
│   ├── performance/             # 性能分析文档
│   ├── PROJECT_STRUCTURE.md     # 项目结构说明（本文件）
│   ├── PROJECT_SUMMARY.md       # 项目总结
│   └── ...
│
├── models/                        # 保存的模型
├── plots/                         # 训练曲线和轨迹图
│
├── requirements.txt              # 依赖包
└── README.md                     # 项目说明
```

## 文件说明

### 核心模块 (`core/`)

- **`environment.py`**: 无人机路径规划环境
  - 实现动力学模型
  - 定义观测空间（17维）
  - 定义动作空间（4维：thrust, roll_torque, pitch_torque, yaw_torque）
  - 实现奖励函数（12个组件）

- **`ppo_agent.py`**: PPO算法实现
  - Actor-Critic网络
  - PPO更新逻辑
  - 模型保存和加载

### 脚本文件 (`scripts/`)

- **`train.py`**: 训练脚本
  - 训练循环
  - Curriculum Learning
  - 训练曲线绘制
  - 模型保存

- **`test.py`**: 测试脚本
  - 模型加载
  - 测试评估
  - 轨迹可视化

### 工具脚本 (`utils/`)

- **`analyze_scenario.py`**: 场景分析
  - 分析特定场景的训练效果
  - 场景难度评估

- **`plot_scenario_reward.py`**: 场景奖励绘图
  - 绘制场景奖励曲线
  - 场景对比分析

- **`eval_plot_episodes.py`**: 评估和可视化
  - Episode评估
  - 轨迹可视化

### 文档 (`docs/`)

- **`analysis/`**: 分析文档
  - 动作、观测、奖励设计分析
  - 训练结果分析
  - 问题诊断文档

- **`fixes/`**: 修复方案文档
  - 已实施的修复
  - 修复效果分析

- **`improvements/`**: 训练改进文档
  - 训练改进措施
  - 改进效果评估

- **`performance/`**: 性能分析文档
  - 性能瓶颈分析
  - 优化建议

## 使用说明

### 导入模块

由于文件已按功能分类，导入时需要指定路径：

```python
# 在scripts/或utils/中的文件
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.environment import DronePathPlanningEnv
from core.ppo_agent import PPO
```

### 运行脚本

```bash
# 训练
python scripts/train.py

# 测试
python scripts/test.py

# 分析场景
python utils/analyze_scenario.py

# 绘制场景奖励
python utils/plot_scenario_reward.py
```

## 文件组织原则

1. **核心功能分离**: 核心算法和环境实现放在`core/`
2. **脚本分类**: 训练/测试脚本放在`scripts/`，工具脚本放在`utils/`
3. **文档集中**: 所有文档放在`docs/`，按类型分类
4. **模块化设计**: 每个目录都有`__init__.py`，便于导入
