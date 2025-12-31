# Multi-Drone Path Planning with PPO

一个基于PPO（Proximal Policy Optimization）算法的多无人机路径规划强化学习系统。

> **注意**: 本项目是 `Test-Project` 下的一个子工程，基于单无人机版本 `drone-path-planning-ppo` 扩展而来。完整的项目结构请查看父目录的 `README.md`。

## 项目概述

本项目实现了一个基于PPO算法的多无人机路径规划系统。系统支持多架无人机，每架无人机有对应的目标点，无人机需要学习如何规划路径到达各自的目标点。

### 主要特性

- ✅ 支持多架无人机同时规划路径（默认3架，可配置）
- ✅ 基于PPO算法的强化学习实现（支持连续动作空间）
- ✅ 共享策略网络（所有无人机使用同一个策略，适合同质智能体）
- ✅ 连续速度控制（vx, vy）而非离散网格移动
- ✅ 基于成功率的Curriculum Learning（场景切换）
- ✅ 动态速度控制（根据距离调整速度上限）
- ✅ 完整的训练和测试流程
- ✅ 训练曲线和轨迹可视化
- ✅ 模型保存和加载功能

## 项目结构

```
multi-drone-path-planning-ppo/
├── environment.py          # 多无人机路径规划环境
├── ppo_agent.py            # PPO算法实现（与单无人机版本共享）
├── train.py                # 训练脚本
├── test.py                 # 测试脚本
├── requirements.txt        # 依赖包
└── README.md              # 项目说明（本文件）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 激活conda环境（如果需要）
conda activate pytorch-venv

# 开始训练（默认3架无人机）
python train.py
```

**训练参数**（可在`train.py`中修改）：
- `num_drones`: 无人机数量（默认3）
- `num_episodes`: 训练episode数（默认8000）
- `max_steps`: 每个episode最大步数（默认60）
- `update_frequency`: 每N个episode更新一次策略（默认10）
- `use_curriculum`: 是否使用curriculum learning（默认True）
- `target_success_rate`: 场景切换的成功率阈值（默认80%）
- `scenario_success_rate_window`: 计算成功率的窗口大小（默认50）

### 3. 测试模型

```bash
python test.py [模型路径] [无人机数量]
```

如果不指定模型路径，默认使用 `models/ppo_model_final.pth`。
如果不指定无人机数量，默认使用3架。

示例：
```bash
# 使用默认模型和3架无人机
python test.py

# 使用指定模型和5架无人机
python test.py models/ppo_model_final.pth 5
```

## 环境配置

### 环境参数

- **world_size**: 1000m × 1000m
- **max_steps**: 60步
- **max_speed**: 20.0 m/s
- **dt**: 1.0s
- **arrival_threshold**: 20.0m
- **num_drones**: 无人机数量（默认3，可配置）

### 状态空间（每架无人机7维）

对于每架无人机，观察空间包括：
- `Δx`: 相对位置x（归一化）
- `Δy`: 相对位置y（归一化）
- `direction_x`: 单位方向向量x分量
- `direction_y`: 单位方向向量y分量
- `vx/max_speed`: 归一化速度x
- `vy/max_speed`: 归一化速度y
- `distance/world_size`: 归一化距离

总观察空间形状：`(num_drones, 7)`

### 动作空间（每架无人机2维）

对于每架无人机，动作空间包括：
- `vx`: x方向速度 [-max_speed, max_speed]
- `vy`: y方向速度 [-max_speed, max_speed]

总动作空间形状：`(num_drones, 2)`

**动态速度限制**：
- 距离 < 20m: 速度限制为30%（6 m/s）
- 距离 20-50m: 速度限制为50%（10 m/s）
- 距离 > 50m: 使用全速（20 m/s）

### 奖励函数

奖励函数与单无人机版本相同，但针对每架无人机独立计算：
1. **Arrival Reward**: 成功到达目标时获得500.0
2. **Success Funnel Reward**: 距离<150m时，根据距离给予引导奖励（0-15 per step）
3. **Precision Reward**: 距离<50m时，给予精确导航奖励（0-20 per step）
4. **Progress Reward**: 根据距离改善给予奖励（主要信号）
5. **Distance Guidance**: 基于绝对距离的持续引导（小权重）
6. **Step Penalty**: 渐进式步数惩罚，鼓励快速到达
7. **Smoothness Penalty**: 速度变化惩罚（近距离时减少）

**多无人机奖励聚合**：
- Episode总奖励 = 所有无人机奖励的平均值
- 成功判定：所有无人机都到达目标点

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

### 共享策略

所有无人机使用同一个策略网络（共享参数），这是适合同质智能体的常见做法：
- 减少参数数量，提高训练效率
- 利用多无人机经验，加速学习
- 适合所有无人机具有相同能力的情况

## Curriculum Learning

系统实现了基于成功率的Curriculum Learning：

- **机制**: 当当前场景的成功率≥80%时，自动切换到下一个场景
- **评估窗口**: 使用最近50个episode计算成功率
- **成功判定**: 所有无人机都到达目标点才算成功
- **优势**: 自适应学习，快速学习者可以更快切换，慢速学习者有更多时间

## 训练输出

训练过程中会生成：

1. **训练曲线** (`plots/training_curves.png`):
   - Episode reward曲线
   - Episode length曲线
   - Success rate曲线（所有无人机都到达的成功率）

2. **Loss曲线** (`plots/loss_curves.png`):
   - Actor loss
   - Critic loss

3. **模型文件** (`models/`):
   - `ppo_model_final.pth`: 最终模型
   - `ppo_model_episode_{N}.pth`: 检查点模型

## 测试输出

测试时会生成：

1. **轨迹可视化** (`plots/test_episode_{N}.png`):
   - 每架无人机的轨迹
   - 起始位置、结束位置、目标位置
   - 目标区域圆圈

2. **统计信息**:
   - 所有无人机都到达的成功率
   - 每架无人机单独的成功率
   - 平均奖励和步数

## 与单无人机版本的区别

1. **环境**：
   - 支持多架无人机和多个目标点
   - 每架无人机有独立的起始位置和目标位置
   - 观察和动作空间扩展到多无人机

2. **训练**：
   - 使用共享策略网络（所有无人机共享参数）
   - 奖励聚合为平均值
   - 成功判定需要所有无人机都到达

3. **测试**：
   - 可视化所有无人机的轨迹
   - 统计每架无人机的表现

## 常见问题

### Q: 如何修改无人机数量？

A: 在 `train.py` 中修改 `num_drones` 参数，或在训练函数调用时传入：
```python
train(num_drones=5, ...)
```

### Q: 所有无人机必须都到达才算成功吗？

A: 是的，当前实现中，只有当所有无人机都到达各自的目标点时，episode才算成功。这是为了鼓励所有无人机都能完成任务。

### Q: 可以使用不同的策略网络吗？

A: 当前实现使用共享策略（所有无人机使用同一个网络）。如果需要独立策略，需要修改训练脚本，为每架无人机创建独立的PPO代理。

### Q: 训练时成功率一直很低？

A: 检查以下几点：
1. 是否启用了curriculum learning
2. 场景切换的成功率阈值是否合理（默认80%）
3. 奖励信号是否正常（检查训练曲线）
4. 无人机数量是否过多（过多可能导致训练困难）

## 版本历史

### 当前版本

**主要特性**：
- ✅ 多无人机路径规划支持
- ✅ 共享策略网络
- ✅ 基于成功率的Curriculum Learning
- ✅ 动态速度控制
- ✅ 完整的训练和测试流程

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

[根据项目实际情况填写]

