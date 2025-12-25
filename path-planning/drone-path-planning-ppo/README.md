# Drone Path Planning with PPO

一个基于PPO（Proximal Policy Optimization）算法的无人机路径规划强化学习系统。

> **注意**: 本项目是 `Test-Project` 下的一个子工程。完整的项目结构请查看父目录的 `README.md`。

## 项目概述

一个面向低空边缘计算场景的模块化 benchmark 与测试平台，用于无人机路径规划任务的强化学习训练与评估。

本项目实现了一个基于PPO（Proximal Policy Optimization）算法的无人机路径规划系统。系统会随机生成目标点，无人机需要学习如何规划路径到达目标点。

### 主要特性

- ✅ 基于PPO算法的强化学习实现（支持连续动作空间）
- ✅ 连续速度控制（vx, vy）而非离散网格移动
- ✅ 基于成功率的Curriculum Learning（场景切换）
- ✅ 动态速度控制（根据距离调整速度上限）
- ✅ 完整的训练和测试流程
- ✅ 训练曲线和轨迹可视化
- ✅ 模型保存和加载功能

## 项目结构

```
drone-path-planning-ppo/
├── environment.py          # 无人机路径规划环境
├── ppo_agent.py            # PPO算法实现
├── train.py                # 训练脚本
├── test.py                 # 测试脚本
├── analyze_scenario.py     # 场景分析脚本
├── plot_scenario_reward.py # 场景奖励绘图脚本
├── eval_plot_episodes.py   # 评估和可视化脚本
├── requirements.txt        # 依赖包
├── README.md              # 项目说明（本文件）
├── PROJECT_SUMMARY.md      # 项目总结
├── PROJECT_STRUCTURE.md    # 项目结构说明
│
├── docs/                  # 详细文档（见 docs/README.md）
│   ├── analysis/          # 问题分析文档
│   ├── fixes/             # 修复方案文档
│   └── improvements/      # 训练改进文档
│
├── models/                # 保存的模型
└── plots/                 # 训练曲线和轨迹图
```

> 详细的项目结构说明请查看 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 激活conda环境（如果需要）
conda activate pytorch-venv

# 开始训练
python train.py
```

**训练参数**（可在`train.py`中修改）：
- `num_episodes`: 训练episode数（默认8000）
- `max_steps`: 每个episode最大步数（默认60）
- `update_frequency`: 每N个episode更新一次策略（默认10）
- `use_curriculum`: 是否使用curriculum learning（默认True）
- `target_success_rate`: 场景切换的成功率阈值（默认80%）
- `scenario_success_rate_window`: 计算成功率的窗口大小（默认50）

### 3. 测试模型

```bash
python test.py [模型路径]
```

如果不指定模型路径，默认使用 `models/ppo_model_final.pth`。

## 环境配置

### 环境参数

- **world_size**: 1000m × 1000m
- **max_steps**: 60步
- **max_speed**: 20.0 m/s
- **dt**: 1.0s
- **arrival_threshold**: 20.0m

### 状态空间（5维）

- `Δx`: 相对位置x（归一化）
- `Δy`: 相对位置y（归一化）
- `vx/max_speed`: 归一化速度x
- `vy/max_speed`: 归一化速度y
- `distance/world_size`: 归一化距离

### 动作空间（连续2维）

- `vx`: x方向速度 [-max_speed, max_speed]
- `vy`: y方向速度 [-max_speed, max_speed]

**动态速度限制**：
- 距离 < 20m: 速度限制为30%（6 m/s）
- 距离 20-50m: 速度限制为50%（10 m/s）
- 距离 > 50m: 使用全速（20 m/s）

### 奖励函数

奖励函数包含以下组件：

1. **Arrival Reward**: 成功到达目标时获得500.0
2. **Success Funnel Reward**: 距离<150m时，根据距离给予引导奖励（0-15 per step）
3. **Precision Reward**: 距离<50m时，给予精确导航奖励（0-20 per step）
4. **Progress Reward**: 根据距离改善给予奖励（主要信号）
5. **Distance Guidance**: 基于绝对距离的持续引导（小权重）
6. **Step Penalty**: 渐进式步数惩罚，鼓励快速到达
7. **Smoothness Penalty**: 速度变化惩罚（近距离时减少）

**重要特性**：
- 奖励基于progress条件：只有真正接近target时才获得full reward
- 防止在徘徊时获得大量reward

## 算法配置

### PPO超参数

- **lr_actor**: 3e-4
- **lr_critic**: 5e-4
- **gamma**: 0.99
- **gae_lambda**: 0.95
- **eps_clip**: 0.2
- **k_epochs**: 40
- **value_clip**: True（启用value clipping稳定critic学习）

### 网络结构

- **Hidden Dimension**: 256
- **Activation**: ReLU
- **Actor输出**: Tanh激活，限制在[-1, 1]
- **初始化**: Orthogonal初始化

## Curriculum Learning

系统实现了基于成功率的Curriculum Learning：

- **机制**: 当当前场景的成功率≥80%时，自动切换到下一个场景
- **评估窗口**: 使用最近50个episode计算成功率
- **优势**: 自适应学习，快速学习者可以更快切换，慢速学习者有更多时间

## 训练输出

训练过程中会生成：

1. **训练曲线** (`plots/training_curves.png`):
   - Episode reward曲线
   - Episode length曲线
   - Success rate曲线

2. **Loss曲线** (`plots/loss_curves.png`):
   - Actor loss
   - Critic loss

3. **轨迹分析** (`plots/trajectories_analysis.png`):
   - 采样episode的轨迹可视化

4. **模型文件** (`models/`):
   - `ppo_model_final.pth`: 最终模型
   - `ppo_model_episode_{N}.pth`: 检查点模型

## 文档索引

详细文档请参考 [docs/README.md](docs/README.md) 和 [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)

### 主要文档分类

1. **环境与算法文档**
   - `ENVIRONMENT_SUMMARY.md`: 环境详细说明
   - `REFACTORING_SUMMARY.md`: 代码重构总结

2. **问题分析与修复**
   - `TRAINING_RESULT_ANALYSIS.md`: 训练结果分析
   - `CLOSE_TARGET_ISSUES_ANALYSIS.md`: 近距离目标问题分析
   - `REWARD_ACCUMULATION_ISSUE.md`: Reward累积问题分析
   - `REWARD_SCALE_ISSUE.md`: Reward scale问题分析

3. **修复实施文档**
   - `FIXES_APPLIED.md`: 早期修复总结
   - `CLOSE_TARGET_FIXES.md`: 近距离问题修复
   - `REWARD_ACCUMULATION_FIX.md`: Reward累积问题修复
   - `REWARD_SCALE_FIX.md`: Reward scale修复

4. **训练改进文档**
   - `TRAINING_IMPROVEMENTS.md`: 训练改进措施
   - `IMPROVEMENTS_APPLIED.md`: 已应用的改进
   - `SUCCESS_RATE_ANALYSIS.md`: 成功率分析
   - `CURRICULUM_LEARNING_UPDATE.md`: Curriculum Learning更新

5. **性能分析**
   - `PERFORMANCE_ANALYSIS.md`: 性能瓶颈分析
   - `POSITION_STRATEGY_ANALYSIS.md`: 位置策略分析

## 常见问题

### Q: Critic loss很大怎么办？
A: 已通过降低critic loss权重（0.5→0.1）和降低arrival reward（1000→500）解决。如果仍然很大，可以进一步降低权重到0.05。

### Q: 训练时成功率一直很低？
A: 检查以下几点：
1. 是否启用了curriculum learning
2. 场景切换的成功率阈值是否合理（默认80%）
3. 奖励信号是否正常（检查训练曲线）

### Q: 测试时reward很大但没到达target？
A: 已通过基于progress的条件奖励修复。确保使用最新版本的环境代码。

## 版本历史

### 最新版本（当前）

**主要改进**：
- ✅ 基于成功率的Curriculum Learning
- ✅ 动态速度控制（根据距离调整）
- ✅ 增强的近距离奖励信号
- ✅ 基于progress的条件奖励（防止徘徊）
- ✅ Value clipping稳定critic学习
- ✅ 优化的reward scale平衡

**关键修复**：
- 修复了reward累积问题（失败episode reward反而更高）
- 修复了critic loss过大的问题
- 修复了近距离犹豫和错过target的问题

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

[根据项目实际情况填写]
