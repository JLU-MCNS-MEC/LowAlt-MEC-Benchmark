# 场景Reward曲线绘制脚本使用说明

## 功能概述

`plot_scenario_reward.py` 脚本用于绘制一个场景/episode中从起点到目标点过程中获得的reward曲线图。

## 主要功能

1. **Step Reward曲线**: 显示每一步获得的reward
2. **Cumulative Reward曲线**: 显示累积reward的变化
3. **Distance变化曲线**: 显示到目标距离的变化
4. **Reward vs Distance对比**: 双y轴显示reward和距离的关系
5. **轨迹可视化**: 显示agent的移动轨迹

## 使用方法

### 基本用法

```bash
# 使用随机策略（不加载模型）
python plot_scenario_reward.py

# 指定episode编号
python plot_scenario_reward.py --episode 100

# 指定起点和目标
python plot_scenario_reward.py --start 600 350 --target 150 950

# 加载模型进行测试
python plot_scenario_reward.py --model models/ppo_model_final.pth --episode 1
```

### 完整参数

```bash
python plot_scenario_reward.py \
    --episode 100 \
    --start 600 350 \
    --target 150 950 \
    --model models/ppo_model_final.pth \
    --max-steps 60 \
    --plot-dir plots \
    --world-size 1000
```

### 参数说明

- `--episode`: Episode编号（用于标题和文件名）
- `--start`: 起点位置 [x, y]，例如 `--start 600 350`
- `--target`: 目标位置 [x, y]，例如 `--target 150 950`
- `--model`: 模型文件路径（.pth文件），如果提供则使用模型策略，否则使用随机策略
- `--max-steps`: 最大步数（默认60）
- `--plot-dir`: 保存图片的目录（默认'plots'）
- `--world-size`: 世界大小（默认1000）

## 输出内容

### 1. 控制台输出

脚本会输出：
- 场景信息（起点、目标、初始距离）
- Episode摘要（总步数、总reward、最终距离、是否到达目标）
- 详细统计（mean、std、min、max reward等）

### 2. 图片输出

保存到 `plots/scenario_reward_analysis_ep{episode_num}.png`，包含：

#### 子图1: Step Reward曲线
- 每一步的reward值
- 平均reward线
- y=0参考线

#### 子图2: Cumulative Reward曲线
- 累积reward的变化
- 最终total reward标记

#### 子图3: Distance变化曲线
- 到目标距离的变化
- Arrival threshold线（绿色虚线）

#### 子图4: Reward vs Distance对比
- 双y轴显示reward和距离
- 便于观察reward和距离的关系

#### 子图5: 轨迹可视化
- Agent的移动轨迹
- 起点（绿色圆点）
- 终点（红色圆点）
- 目标（红色星号）
- 目标区域（红色圆圈）
- 关键步骤的距离标注

## 使用示例

### 示例1: 分析特定场景

```bash
# 分析起点(600, 350)到目标(150, 950)的场景
python plot_scenario_reward.py \
    --episode 831 \
    --start 600 350 \
    --target 150 950 \
    --model models/ppo_model_final.pth
```

### 示例2: 测试模型性能

```bash
# 使用训练好的模型测试随机场景
python plot_scenario_reward.py \
    --episode 1 \
    --model models/ppo_model_final.pth
```

### 示例3: 分析失败场景

```bash
# 分析一个已知失败的场景
python plot_scenario_reward.py \
    --episode 832 \
    --start 600 350 \
    --target 150 950 \
    --model models/ppo_model_final.pth
```

## 在Python中使用

也可以作为模块导入使用：

```python
from plot_scenario_reward import plot_scenario_reward

# 绘制reward曲线
fig, data = plot_scenario_reward(
    episode_num=831,
    start_pos=[600, 350],
    target_pos=[150, 950],
    model_path='models/ppo_model_final.pth',
    max_steps=60,
    plot_dir='plots'
)

# 访问数据
step_rewards = data['step_rewards']
cumulative_rewards = data['cumulative_rewards']
distances = data['distances']
```

## 输出文件

- **文件名**: `scenario_reward_analysis_ep{episode_num}.png`
- **位置**: `plots/` 目录
- **分辨率**: 300 DPI
- **格式**: PNG

## 注意事项

1. **模型路径**: 如果提供模型路径，确保文件存在且格式正确
2. **观察空间**: 脚本会自动适配当前的观察空间维度（7维，包含方向向量）
3. **随机策略**: 如果不提供模型，将使用随机策略（用于演示）
4. **数据记录**: 脚本会记录每一步的reward、距离、位置等信息

## 分析建议

### 1. 检查初始方向
- 查看前几步的reward
- 如果初始reward为负，可能方向错误

### 2. 检查reward趋势
- 观察reward是否逐渐增加
- 如果reward一直很低，可能策略有问题

### 3. 检查距离变化
- 观察距离是否持续减小
- 如果距离先减小后增大，可能绕弯了

### 4. 检查轨迹
- 查看轨迹是否直接
- 如果轨迹绕弯，可能需要优化路径效率奖励

## 故障排除

### 问题1: 模型加载失败
**解决**: 检查模型路径是否正确，模型文件是否存在

### 问题2: 观察空间维度不匹配
**解决**: 确保模型是用相同版本的environment训练的（7维观察空间）

### 问题3: 图片保存失败
**解决**: 检查plot_dir目录是否存在，是否有写入权限

