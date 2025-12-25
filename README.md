# Test-Project

本目录包含多个子工程项目，每个子工程都是独立的模块化组件。

## 子工程列表

### 1. drone-path-planning-ppo

**基于PPO算法的无人机路径规划强化学习系统**

一个面向低空边缘计算场景的模块化 benchmark 与测试平台，用于无人机路径规划任务的强化学习训练与评估。

#### 主要特性

- ✅ 基于PPO算法的强化学习实现（支持连续动作空间）
- ✅ 连续速度控制（vx, vy）而非离散网格移动
- ✅ 基于成功率的Curriculum Learning（场景切换）
- ✅ 动态速度控制（根据距离调整速度上限）
- ✅ 完整的训练和测试流程
- ✅ 训练曲线和轨迹可视化
- ✅ 模型保存和加载功能

#### 技术栈

- **算法**: PPO (Proximal Policy Optimization)
- **框架**: PyTorch
- **环境**: Gymnasium (OpenAI Gym)
- **可视化**: Matplotlib

#### 快速开始

```bash
cd drone-path-planning-ppo
pip install -r requirements.txt
conda activate pytorch-venv  # 如果需要
python train.py
```

#### 详细文档

请查看 `drone-path-planning-ppo/README.md` 和 `drone-path-planning-ppo/docs/` 目录获取完整文档。

---

## 项目结构

```
Test-Project/
├── README.md                    # 本文件（项目总览）
│
├── drone-path-planning-ppo/     # 无人机路径规划子工程
│   ├── README.md                # 子工程说明
│   ├── environment.py           # 环境实现
│   ├── ppo_agent.py             # PPO算法实现
│   ├── train.py                  # 训练脚本
│   ├── test.py                   # 测试脚本
│   ├── docs/                     # 详细文档
│   ├── models/                   # 保存的模型
│   └── plots/                    # 训练曲线和可视化
│
└── [其他子工程...]              # 未来可添加更多子工程
```

## 添加新子工程

如需添加新的子工程：

1. 在 `Test-Project/` 目录下创建新的子工程目录
2. 在新目录中实现子工程功能
3. 在本 `README.md` 中添加子工程说明
4. 确保子工程有独立的 `README.md` 和文档

## 开发规范

### 目录命名

- 使用小写字母和连字符：`sub-project-name`
- 名称应清晰描述子工程功能

### 文档要求

每个子工程应包含：
- `README.md`: 子工程说明、快速开始、API文档
- `requirements.txt`: Python依赖（如适用）
- `docs/`: 详细文档（如适用）

### 代码组织

- 保持子工程独立性
- 避免跨子工程的直接依赖
- 使用清晰的模块化设计

---

**最后更新**: 2024年12月

