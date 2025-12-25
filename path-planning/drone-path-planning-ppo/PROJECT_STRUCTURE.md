# 项目结构说明

本文档说明 `drone-path-planning-ppo` 项目的目录结构和文件组织。

## 目录结构

```
drone-path-planning-ppo/
├── README.md                    # 项目主文档
├── PROJECT_SUMMARY.md           # 项目总结
├── requirements.txt             # Python依赖
│
├── environment.py              # 无人机路径规划环境实现
├── ppo_agent.py                # PPO算法实现（Actor-Critic网络）
├── train.py                    # 训练脚本（主入口）
├── test.py                     # 测试脚本
│
├── analyze_scenario.py         # 场景分析脚本
├── plot_scenario_reward.py     # 场景奖励绘图脚本
├── eval_plot_episodes.py       # 评估和可视化脚本
│
├── docs/                       # 详细文档目录
│   ├── README.md               # 文档索引
│   ├── DOCUMENTATION.md        # 文档索引（详细版）
│   ├── ENVIRONMENT_SUMMARY.md  # 环境总结
│   │
│   ├── analysis/               # 问题分析文档
│   │   ├── TRAINING_RESULT_ANALYSIS.md
│   │   ├── CLOSE_TARGET_ISSUES_ANALYSIS.md
│   │   ├── REWARD_ACCUMULATION_ISSUE.md
│   │   └── ...
│   │
│   ├── fixes/                  # 修复方案文档
│   │   ├── FIXES_APPLIED.md
│   │   ├── CLOSE_TARGET_FIXES.md
│   │   ├── REWARD_ACCUMULATION_FIX.md
│   │   └── ...
│   │
│   └── improvements/           # 训练改进文档
│       ├── TRAINING_IMPROVEMENTS.md
│       ├── CURRICULUM_LEARNING_UPDATE.md
│       └── ...
│
├── models/                     # 保存的模型文件
│   ├── ppo_model_final.pth     # 最终模型
│   └── ppo_model_scenario_*.pth  # 场景特定模型
│
└── plots/                      # 训练结果可视化
    ├── training_curves.png     # 训练曲线
    ├── reward_curve.png        # 奖励曲线
    ├── loss_curves.png         # 损失曲线
    ├── trajectories_*.png      # 轨迹图
    └── test_episode_*.png      # 测试episode可视化
```

## 核心文件说明

### 环境与算法

- **`environment.py`**: 
  - 实现 `DronePathPlanningEnv` 类
  - 定义观察空间、动作空间、奖励函数
  - 实现动态速度控制和边界处理

- **`ppo_agent.py`**: 
  - 实现 `PPO` 类
  - Actor-Critic网络结构
  - PPO算法更新逻辑
  - 支持值函数裁剪（value clipping）

### 训练与测试

- **`train.py`**: 
  - 主训练脚本
  - 实现Curriculum Learning
  - 场景切换逻辑
  - 模型保存和可视化

- **`test.py`**: 
  - 模型测试脚本
  - 评估成功率、平均奖励等指标
  - 生成测试轨迹可视化

### 分析工具

- **`analyze_scenario.py`**: 
  - 分析特定场景的训练表现
  - 诊断场景失败原因
  - 提供修复建议

- **`plot_scenario_reward.py`**: 
  - 绘制单个场景的奖励曲线
  - 分析场景内的学习过程

- **`eval_plot_episodes.py`**: 
  - 评估模型并生成可视化
  - 批量测试episode

## 文档组织

### docs/ 目录

文档按类别组织：

1. **analysis/**: 问题分析文档
   - 记录训练过程中发现的问题
   - 详细的问题诊断过程
   - 问题原因分析

2. **fixes/**: 修复方案文档
   - 针对问题的修复方案
   - 修复实施细节
   - 修复效果评估

3. **improvements/**: 训练改进文档
   - 训练过程的优化措施
   - 算法改进和参数调整
   - 性能提升记录

### 文档阅读建议

1. **新手入门**: 
   - 先阅读 `README.md`
   - 查看 `docs/ENVIRONMENT_SUMMARY.md` 了解环境
   - 阅读 `docs/fixes/FIXES_APPLIED.md` 了解主要修复

2. **问题诊断**: 
   - 查看 `docs/analysis/` 目录下的相关分析文档
   - 参考 `docs/fixes/` 目录下的修复方案

3. **理解实现**: 
   - 查看 `docs/improvements/` 了解训练改进
   - 阅读代码注释和文档

## 输出文件

### models/ 目录

保存训练好的模型：
- `ppo_model_final.pth`: 最终训练模型
- `ppo_model_scenario_N_episode_M.pth`: 特定场景的检查点

### plots/ 目录

训练和测试的可视化结果：
- 训练曲线：奖励、成功率、episode长度
- 损失曲线：actor loss、critic loss
- 轨迹图：无人机路径、目标点位置
- 测试结果：测试episode的可视化

## 使用流程

### 1. 训练模型

```bash
python train.py
```

训练过程会：
- 自动保存模型到 `models/`
- 生成训练曲线到 `plots/`
- 记录场景切换信息

### 2. 测试模型

```bash
python test.py
```

测试过程会：
- 加载模型
- 运行多个测试episode
- 生成测试结果和可视化

### 3. 分析场景

```bash
python analyze_scenario.py <scenario_num>
```

分析特定场景：
- 加载场景信息
- 测试场景表现
- 诊断问题并提供建议

### 4. 绘制场景奖励

```bash
python plot_scenario_reward.py
```

绘制单个场景的奖励曲线。

## 依赖管理

### requirements.txt

包含所有Python依赖：
- PyTorch
- Gymnasium
- NumPy
- Matplotlib
- tqdm

安装依赖：
```bash
pip install -r requirements.txt
```

## 配置说明

主要配置在代码中：
- `train.py`: 训练参数（episode数、学习率等）
- `environment.py`: 环境参数（世界大小、速度限制等）
- `ppo_agent.py`: PPO参数（网络结构、更新频率等）

建议通过修改代码中的参数来调整配置。

---

**最后更新**: 2024年12月

