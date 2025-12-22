# LowAlt-MEC-Benchmark
一个面向低空边缘计算场景的模块化 benchmark 与测试平台，旨在对 状态表征、动作空间、奖励函数及策略结构等关键设计选择 进行系统化、可复现的评测与对比。 该项目可用于快速验证不同设计方案在多种低空 MEC 场景下的可学性、稳定性、泛化性与系统性能表现。

## 无人机路径规划 - PPO算法实现

本项目实现了一个基于PPO（Proximal Policy Optimization）算法的无人机路径规划系统。系统会随机生成目标点，无人机需要学习如何规划路径到达目标点。

### 项目结构

```
LowAlt-MEC-Benchmark/
├── environment.py      # 无人机路径规划环境
├── ppo_agent.py        # PPO算法实现
├── train.py            # 训练脚本
├── test.py             # 测试脚本
├── requirements.txt    # 依赖包
└── README.md          # 项目说明
```

### 环境说明

**状态空间**（7维）：
- 无人机当前位置（归一化）
- 目标位置（归一化）
- 相对位置（归一化）
- 距离目标的距离（归一化）

**动作空间**（连续2维）：
- 速度控制 (vx, vy)：x方向和y方向的速度分量
- 速度范围：[-max_speed, max_speed]，默认max_speed=2.0
- 位置更新：pos = pos + velocity * dt，默认dt=0.1

**奖励函数**：
- 距离奖励：距离目标越近，奖励越高
- 到达奖励：成功到达目标时给予大额奖励
- 改善奖励：距离目标的改善给予奖励
- 步数惩罚：鼓励快速到达目标

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练模型

```bash
python train.py
```

训练参数可以在 `train.py` 中修改：
- `num_episodes`: 训练episode数（默认2000）
- `max_steps`: 每个episode最大步数（默认500）
- `update_frequency`: 每N个episode更新一次策略（默认20）

训练过程中会：
- 每100个episode打印训练统计信息
- 每500个episode保存模型到 `models/` 目录
- 训练结束后保存训练曲线到 `plots/` 目录

### 测试模型

```bash
python test.py [模型路径]
```

如果不指定模型路径，默认使用 `models/ppo_model_final.pth`。

测试会：
- 运行指定数量的测试episode
- 显示每个episode的轨迹可视化
- 统计成功率和平均性能

### 使用示例

1. **训练新模型**：
```bash
conda activate pytorch-venv
python train.py
```

2. **测试训练好的模型**：
```bash
python test.py models/ppo_model_final.pth
```

### 主要特性

- ✅ 基于PPO算法的强化学习实现（支持连续动作空间）
- ✅ 连续速度控制（vx, vy）而非离散网格移动
- ✅ 随机目标点生成
- ✅ 完整的训练和测试流程
- ✅ 训练曲线可视化
- ✅ 轨迹可视化
- ✅ 模型保存和加载功能
