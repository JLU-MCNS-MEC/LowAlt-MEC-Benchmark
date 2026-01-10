# Multi-Drone Path Planning with PPO (Dynamics Model)

一个基于PPO（Proximal Policy Optimization）算法和动力学模型的多无人机路径规划强化学习系统。

> **注意**: 本项目是 `Test-Project` 下的一个子工程。完整的项目结构请查看父目录的 `README.md`。
> 
> **与 `drone-path-planning-ppo-dynamics` 的区别**: 本工程支持多架无人机同时规划路径，每架无人机有对应的目标点，使用动力学模型控制。
> 
> **与 `multi-drone-path-planning-ppo` 的区别**: 本工程使用动力学模型控制无人机，动作空间为 [thrust, roll_torque, pitch_torque, yaw_torque]（4维），而不是直接的速度控制 [vx, vy]（2维）。

## 项目概述

一个面向低空边缘计算场景的模块化 benchmark 与测试平台，用于多无人机路径规划任务的强化学习训练与评估。

本项目实现了一个基于PPO算法和动力学模型的多无人机路径规划系统。系统支持多架无人机（默认3架，可配置1-9架），每架无人机有对应的目标点，需要通过控制推力、滚转力矩、俯仰力矩和偏航力矩来规划路径到达各自的目标点。

### 主要特性

- ✅ **多无人机支持**：支持1-9架无人机同时规划路径（默认3架）
- ✅ 基于PPO算法的强化学习实现（支持连续动作空间）
- ✅ **共享策略网络**：所有无人机使用同一个策略，适合同质智能体
- ✅ **动力学模型控制**：使用推力、滚转力矩、俯仰力矩、偏航力矩控制，而非直接速度控制
- ✅ 完整的姿态动力学：包含roll, pitch, yaw姿态和角速度
- ✅ 基于成功率的Curriculum Learning（场景切换）
- ✅ 完整的训练和测试流程
- ✅ 训练曲线和轨迹可视化（支持多无人机轨迹）
- ✅ 模型保存和加载功能

## 项目结构

```
drone-path-planning-ppo-dynamics/
├── core/                          # 核心功能模块
│   ├── environment.py             # 无人机路径规划环境（动力学模型）
│   └── ppo_agent.py               # PPO算法实现
│
├── scripts/                       # 脚本文件
│   ├── train.py                  # 训练脚本
│   └── test.py                   # 测试脚本
│
├── utils/                         # 工具脚本
│   ├── analyze_scenario.py       # 场景分析脚本
│   ├── plot_scenario_reward.py   # 场景奖励绘图脚本
│   └── eval_plot_episodes.py     # 评估和可视化脚本
│
├── docs/                          # 文档目录
│   ├── analysis/                  # 问题分析文档
│   │   ├── ACTION_OBSERVATION_REWARD_ANALYSIS.md
│   │   ├── TRAINING_RESULT_ANALYSIS.md
│   │   └── ...
│   ├── fixes/                    # 修复方案文档
│   ├── improvements/             # 训练改进文档
│   └── ...
│
├── models/                        # 保存的模型
├── plots/                         # 训练曲线和轨迹图
│
├── requirements.txt               # 依赖包
└── README.md                     # 项目说明（本文件）
```

> 详细的项目结构说明请查看 [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)

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
python scripts/train.py
```

**训练参数**（可在`scripts/train.py`中修改）：
- `num_drones`: 无人机数量（默认3，范围1-9）
- `num_episodes`: 训练episode数（默认4000）
- `max_steps`: 每个episode最大步数（默认600）
- `update_frequency`: 每N个episode更新一次策略（默认10）
- `use_curriculum`: 是否使用curriculum learning（默认True）
- `target_success_rate`: 场景切换的成功率阈值（默认50%）
- `scenario_success_rate_window`: 计算成功率的窗口大小（默认30）

### 3. 测试模型

```bash
# 测试模型（默认3架无人机）
python scripts/test.py [模型路径] [无人机数量]

# 示例：使用5架无人机测试
python scripts/test.py models/ppo_model_final.pth 5
```

如果不指定模型路径，默认使用 `models/ppo_model_final.pth`。
如果不指定无人机数量，默认使用3架。

## 环境配置

### 环境参数

- **world_size**: 1000m × 1000m
- **max_steps**: 600步
- **max_speed**: 20.0 m/s
- **dt**: 0.1s（更精细的控制，适合动力学模型）
- **arrival_threshold**: 20.0m

### 动力学参数

- **m**: 0.2 kg（质量）
- **Ixx, Iyy, Izz**: 1.0 kg·m²（转动惯量）
- **g**: 9.81 m/s²（重力加速度）
- **max_thrust**: 5.0 N（最大推力）
- **max_torque**: 1.0 N·m（最大力矩，从2.0减小以提供更平滑的控制）

### 观测空间（Observation Space）- 每个无人机17维

观测空间为每个无人机提供完整的动力学状态信息，帮助Agent理解当前状态并做出决策。总观测空间形状为 `(num_drones, 17)`。

#### 位置与方向信息（4维）
- **`Δx`**: 相对位置x（归一化到world_size，范围约[-1, 1]）
- **`Δy`**: 相对位置y（归一化到world_size，范围约[-1, 1]）
- **`direction_x`**: 单位方向向量x分量（指向目标的方向，范围[-1, 1]）
- **`direction_y`**: 单位方向向量y分量（指向目标的方向，范围[-1, 1]）

#### 速度信息（2维）
- **`vx/max_speed`**: 归一化速度x（范围约[-1, 1]）
- **`vy/max_speed`**: 归一化速度y（范围约[-1, 1]）

#### 距离信息（1维）
- **`distance/world_size`**: 归一化距离（使用最大可能距离（对角线）归一化，范围[0, 1]）

#### 姿态信息（3维）
- **`roll/(π/4)`**: 归一化滚转角（使用π/4=45°作为典型范围，范围约[-2, 2]）
- **`pitch/(π/4)`**: 归一化俯仰角（使用π/4=45°作为典型范围，范围约[-2, 2]）
- **`yaw/π`**: 归一化偏航角（范围约[-1, 1]）

#### 角速度信息（3维）
- **`roll_vel/max_angular_vel`**: 归一化滚转角速度（max_angular_vel=3.0 rad/s，范围约[-2, 2]）
- **`pitch_vel/max_angular_vel`**: 归一化俯仰角速度（max_angular_vel=3.0 rad/s，范围约[-2, 2]）
- **`yaw_vel/max_angular_vel`**: 归一化偏航角速度（max_angular_vel=3.0 rad/s，范围约[-2, 2]）

#### 动作历史（4维）
- **`prev_thrust`**: 上一时刻的推力（范围[-1, 1]）
- **`prev_roll_torque`**: 上一时刻的滚转力矩（范围[-1, 1]）
- **`prev_pitch_torque`**: 上一时刻的俯仰力矩（范围[-1, 1]）
- **`prev_yaw_torque`**: 上一时刻的偏航力矩（范围[-1, 1]）

**设计说明**：
- 动作历史帮助Agent理解动作的影响，提高学习效率
- 姿态归一化使用π/4而非π，使归一化后的值在合理范围内
- 所有观测值都经过归一化，便于神经网络学习

### 动作空间（Action Space）- 每个无人机连续4维

动作空间使用动力学控制，Agent直接控制每架无人机的推力和力矩。总动作空间形状为 `(num_drones, 4)`。

#### 动作组成
1. **`thrust`**: 推力
   - 输入范围: [-1, 1]
   - 实际范围: [0, max_thrust] = [0, 5.0] N
   - 映射公式: `thrust = (action[0] + 1.0) / 2.0 * max_thrust`
   - 作用: 提供向上的推力，通过姿态（roll, pitch）转换为水平加速度

2. **`roll_torque`**: 滚转力矩
   - 输入范围: [-1, 1]
   - 实际范围: [-max_torque, max_torque] = [-1.0, 1.0] N·m
   - 映射公式: `roll_torque = action[1] * max_torque`
   - 作用: 控制无人机绕x轴旋转（左右倾斜）

3. **`pitch_torque`**: 俯仰力矩
   - 输入范围: [-1, 1]
   - 实际范围: [-max_torque, max_torque] = [-1.0, 1.0] N·m
   - 映射公式: `pitch_torque = action[2] * max_torque`
   - 作用: 控制无人机绕y轴旋转（前后倾斜）

4. **`yaw_torque`**: 偏航力矩
   - 输入范围: [-1, 1]
   - 实际范围: [-max_torque, max_torque] = [-1.0, 1.0] N·m
   - 映射公式: `yaw_torque = action[3] * max_torque`
   - 作用: 控制无人机绕z轴旋转（水平旋转）

#### 动力学更新流程

```
动作输入 [thrust, roll_torque, pitch_torque, yaw_torque]
    ↓
1. 角速度更新: ω += τ * dt / I
   - roll_vel += roll_torque * dt / Ixx
   - pitch_vel += pitch_torque * dt / Iyy
   - yaw_vel += yaw_torque * dt / Izz
   - 添加阻尼: ω *= 0.95 (每步)
    ↓
2. 姿态更新: θ += ω * dt
   - roll += roll_vel * dt
   - pitch += pitch_vel * dt
   - yaw += yaw_vel * dt
    ↓
3. 加速度计算: 通过旋转矩阵转换
   - 计算3D旋转矩阵 R (ZYX Euler angles)
   - 推力向量在机体坐标系: [0, 0, thrust]
   - 转换到世界坐标系: thrust_world = R @ [0, 0, thrust]
   - 水平加速度: acc = [thrust_world[0], thrust_world[1]] / m
    ↓
4. 速度更新: v += a * dt
   - 限制最大速度: |v| ≤ max_speed
    ↓
5. 位置更新: pos += v * dt
```

**设计特点**：
- 完整的4自由度控制（推力 + 3个力矩）
- 物理真实的动力学模型
- 时间步长dt=0.1s，提供精细控制

### 奖励函数（Reward Function）

奖励函数设计用于引导Agent学习稳定、高效的飞行策略，包含12个组件：

#### 主要目标奖励

1. **Arrival Reward** (到达奖励)
   - **值**: 500.0
   - **触发条件**: 距离 < arrival_threshold (20m)
   - **作用**: 鼓励Agent到达目标

2. **Progress Reward** (距离改善奖励) - **主要信号**
   - **公式**: `progress_weight * (progress / 100.0)`
   - **progress**: `prev_distance - current_distance`
   - **权重**: 
     - 近距离 (<50m): `progress_weight * 2.0` = 400.0
     - 远距离 (>800m): `progress_weight * 1.5` = 300.0
     - 中等距离: `progress_weight` = 200.0
   - **作用**: 主要的学习信号，鼓励向目标移动

#### 引导奖励

3. **Success Funnel Reward** (成功漏斗奖励)
   - **触发条件**: 距离 < 150m
   - **公式**: `15 * ((150 - distance) / 150)²`
   - **范围**: 0-15 per step
   - **条件**: 如果progress < -1.0（远离），奖励减少到30%
   - **作用**: 在接近目标时提供引导

4. **Precision Reward** (精确导航奖励)
   - **触发条件**: 距离 < 50m
   - **公式**: `20.0 * ((50 - distance) / 50)^1.5`
   - **范围**: 0-20 per step
   - **条件**: 如果progress < -0.5（远离），奖励减少到30%
   - **作用**: 在近距离时鼓励精确控制

5. **Distance Guidance** (距离引导)
   - **公式**: `-0.5 * (distance / world_size)`
   - **范围**: 约[-0.5, 0]
   - **作用**: 持续的距离信号，帮助保持方向

6. **Initial Direction Reward** (初始方向奖励)
   - **触发条件**: 前5步
   - **公式**: 
     - 方向对齐 > 30%: `5.0 * alignment`
     - 方向对齐 < -30%: `-3.0 * abs(alignment)`
   - **作用**: 在开始时引导正确的方向

#### 效率与稳定性奖励

7. **Path Efficiency Reward** (路径效率奖励)
   - **公式**: `efficiency_weight * (ideal_distance / actual_path_length)`
   - **efficiency_weight**: `2.0 * (1.0 - distance / initial_distance)`
   - **作用**: 鼓励直线路径，减少绕行

8. **Attitude Stability Reward** (姿态稳定性奖励) ⭐ **新增**
   - **公式**: `-1.0 * (abs(roll) + abs(pitch)) / (π/4)`
   - **范围**: 约[-1.0, 0]
   - **作用**: 鼓励保持水平姿态（roll, pitch接近0）
   - **权重**: 已增加到-1.0（10倍），与其他奖励匹配

9. **Attitude Angle Limit Penalty** (姿态角度限制惩罚) ⭐ **新增**
   - **触发条件**: roll或pitch > 30° (π/6)
   - **公式**: `-5.0 * (abs(angle) - 30°) / (π/6)`
   - **作用**: 惩罚过大倾斜，确保安全飞行
   - **权重**: 已增加到-5.0（5倍）

10. **Attitude Smoothness Reward** (姿态平滑性奖励) ⭐ **新增**
    - **公式**: `-0.5 * min(attitude_change², (π/4)²) / (π/4)²`
    - **作用**: 惩罚快速姿态变化，鼓励平滑控制
    - **权重**: 已增加到-0.5（10倍）

#### 惩罚项

11. **Step Penalty** (步数惩罚)
    - **公式**: `-step_penalty * (1.0 + step_count * 0.01)`
    - **范围**: 约[-0.01, -0.016] per step（渐进式）
    - **作用**: 鼓励快速到达，避免长时间徘徊

12. **Smoothness Penalty** (平滑性惩罚)
    - **公式**: 
      - 近距离: `-0.00005 * velocity_change² - 0.0001 * action_change²`
      - 远距离: `-0.0001 * velocity_change² - 0.0002 * action_change²`
    - **作用**: 惩罚速度和动作的突变

13. **Boundary Penalty** (边界惩罚)
    - **公式**: `-boundary_penalty_weight * penetration_depth`
    - **权重**: 5.0（目标接近边界时减半）
    - **作用**: 防止越界

#### 奖励设计特点

- **多目标平衡**: 同时考虑到达目标、飞行稳定性、路径效率
- **条件奖励**: Success funnel和Precision reward基于progress条件，防止徘徊
- **姿态引导**: 新增的姿态相关奖励确保稳定、安全的飞行
- **渐进式惩罚**: Step penalty随步数增加，鼓励快速到达

#### 典型奖励值范围

| 场景 | 典型奖励范围 |
|------|------------|
| 快速到达（20步） | 500 (arrival) - 0.2 (step) ≈ **500** |
| 正常接近 | 200 (progress) + 10 (funnel) - 1.0 (attitude) ≈ **200** |
| 徘徊60步 | 15 (funnel) * 60 - 0.6 (step) - 60 (attitude) ≈ **-45** |
| 过度倾斜 | -5.0 (angle penalty) - 1.0 (stability) ≈ **-6** |

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

详细文档请参考 [docs/README.md](docs/README.md)

### 主要文档分类

#### 分析文档 (`docs/analysis/`)
- **`ACTION_OBSERVATION_REWARD_ANALYSIS.md`**: 动作、观测与奖励设计分析
- **`TRAINING_RESULT_ANALYSIS.md`**: 训练结果分析与改进建议

#### 其他文档 (`docs/`)
- **`PROJECT_STRUCTURE.md`**: 项目结构说明
- **`PROJECT_SUMMARY.md`**: 项目总结
- **`ENVIRONMENT_SUMMARY.md`**: 环境详细说明
- **`LEARNING_GUIDE.md`**: 学习指南

详细文档请查看 [docs/](docs/) 目录

## 与单无人机版本的区别

1. **环境**：
   - 支持多架无人机和多个目标点
   - 每架无人机有独立的起始位置和目标位置
   - 观察和动作空间扩展到多无人机：`(num_drones, 17)` 和 `(num_drones, 4)`

2. **训练**：
   - 使用共享策略网络（所有无人机共享参数）
   - 奖励聚合为平均值
   - 成功判定需要所有无人机都到达

3. **测试**：
   - 可视化所有无人机的轨迹
   - 统计每架无人机的表现

## 常见问题

### Q: 如何修改无人机数量？
A: 在 `scripts/train.py` 中修改 `num_drones` 参数，或在训练函数调用时传入：
```python
train(num_drones=5, ...)
```

### Q: 所有无人机必须都到达才算成功吗？
A: 是的，当前实现中，只有当所有无人机都到达各自的目标点时，episode才算成功。这是为了鼓励所有无人机都能完成任务。

### Q: 可以使用不同的策略网络吗？
A: 当前实现使用共享策略（所有无人机使用同一个网络）。如果需要独立策略，需要修改训练脚本，为每架无人机创建独立的PPO代理。

### Q: Critic loss很大怎么办？
A: 已通过降低critic loss权重（0.5→0.1）和降低arrival reward（1000→500）解决。如果仍然很大，可以进一步降低权重到0.05。

### Q: 训练时成功率一直很低？
A: 检查以下几点：
1. 是否启用了curriculum learning
2. 场景切换的成功率阈值是否合理（默认50%）
3. 奖励信号是否正常（检查训练曲线）
4. 无人机数量是否过多（过多可能导致训练困难）

### Q: 测试时reward很大但没到达target？
A: 已通过基于progress的条件奖励修复。确保使用最新版本的环境代码。

## 版本历史

### 最新版本（当前）

**主要改进**：
- ✅ **多无人机支持**: 支持1-9架无人机同时规划路径（默认3架）
- ✅ **动力学模型控制**: 使用推力、滚转力矩、俯仰力矩、偏航力矩控制
- ✅ **完整的姿态动力学**: 包含roll, pitch, yaw姿态和角速度
- ✅ **共享策略网络**: 所有无人机使用同一个策略，适合同质智能体
- ✅ **姿态相关奖励**: 添加姿态稳定性、角度限制、平滑性奖励
- ✅ **精细时间步长**: dt=0.1s，提供更精细的控制
- ✅ **动作历史**: 观测空间包含上一时刻的动作
- ✅ **基于成功率的Curriculum Learning**: 自适应场景切换
- ✅ **优化的奖励函数**: 12个组件，平衡多目标

**关键修复**：
- 增加姿态奖励权重（10倍），使姿态控制更有效
- 减小时间步长（0.1s），提高控制精度
- 增强姿态角度限制惩罚（5倍），确保安全飞行
- 添加yaw_torque控制，完整4自由度控制
- 多无人机位置均匀分布，避免重叠

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

[根据项目实际情况填写]
