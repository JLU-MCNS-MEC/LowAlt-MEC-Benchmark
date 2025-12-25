# 场景切换跟踪更新

## 修改概述

将原本随机保存episode的代码改为只保存场景切换时的episode，用于诊断场景切换后reward下降的问题。

## 主要修改

### 1. 删除随机采样 ✅

**之前**:
```python
# Randomly select 10 episodes to track
sampled_episodes = sorted(random.sample(range(num_episodes), min(10, num_episodes)))
```

**之后**:
```python
# Track episodes around scenario switches
scenario_switch_episodes = []  # Store episode numbers when scenario switches
step_rewards_tracking = {}  # Will be populated when scenario switches
```

### 2. 添加场景切换跟踪函数 ✅

**新增函数**:
```python
def track_scenario_switch_episodes(switch_episode, episodes_before=3, episodes_after=5):
    """Track episodes around scenario switch for diagnosis"""
    # 跟踪切换前3个、切换时、切换后5个episode
```

**功能**:
- 跟踪场景切换前后的episode
- 默认：切换前3个 + 切换时1个 + 切换后5个 = 总共9个episode
- 可以诊断场景切换时的问题

### 3. 场景切换时记录信息 ✅

**新增记录**:
- 场景切换的episode编号
- 旧场景信息（起点、目标、距离、成功率、episode数）
- 新场景信息（起点、目标、距离）
- 距离变化（绝对值和百分比）

**输出示例**:
```
Episode 150: Scenario 1 completed!
  Final success rate: 82.5%
  Episodes in scenario: 150
  Old scenario distance: 350.5m
  Starting scenario 2
  Start: (234.5, 567.8)
  Target: (789.1, 123.4)
  New scenario distance: 420.3m
  Distance change: +69.8m (+19.9%)
  Tracking episodes: [147, 148, 149, 150, 151, 152, 153, 154, 155] (total 9 episodes)
```

### 4. 可视化增强 ✅

**轨迹图标题增强**:
- 场景切换的episode标记为 `[SCENARIO SWITCH]`
- 切换前的episode标记为 `[Before Switch {episode}]`
- 切换后的episode标记为 `[After Switch {episode}]`

**统计信息增强**:
- 显示场景切换信息
- 显示每个场景的距离

## 诊断能力

### 可以诊断的问题

1. **场景难度突然增加**
   - 通过距离变化百分比判断
   - 如果新场景距离比旧场景大很多（如+50%），可能导致reward下降

2. **策略适应问题**
   - 观察切换后前几个episode的表现
   - 检查是否快速适应还是需要很长时间

3. **Reward下降模式**
   - 对比切换前后的reward
   - 分析reward下降的原因

### 使用方法

1. **运行训练**:
   ```bash
   python train.py
   ```

2. **查看输出**:
   - 训练过程中会打印场景切换信息
   - 包括距离变化、跟踪的episode列表

3. **分析结果**:
   - 查看 `plots/trajectories_analysis.png`
   - 查看 `plots/step_rewards_summary.png`
   - 关注标记为 `[SCENARIO SWITCH]` 的episode

## 可能发现的问题

### 问题1: 距离突然增加
**现象**: 新场景距离比旧场景大很多（>30%）
**影响**: Agent需要更多时间适应，reward下降
**解决**: 实现渐进式场景难度（见SCENARIO_SWITCH_ISSUE.md）

### 问题2: 策略过拟合
**现象**: 切换后前几个episode表现很差
**影响**: Agent需要重新学习
**解决**: 提高切换条件（成功率阈值、评估窗口）

### 问题3: 场景分布不均
**现象**: 某些场景距离很大，某些很小
**影响**: 训练不稳定
**解决**: 限制场景距离范围

## 下一步

根据跟踪结果，可以：
1. 分析场景切换时的距离变化
2. 观察agent的适应速度
3. 调整场景生成策略
4. 优化切换条件

## 技术细节

### 跟踪的Episode数量

- **切换前**: 3个episode（默认）
- **切换时**: 1个episode
- **切换后**: 5个episode（默认）
- **总计**: 9个episode per场景切换

### 存储的信息

- `scenario_switch_episodes`: 场景切换的episode列表
- `scenario_info`: 每个场景的信息（起点、目标、距离等）
- `step_rewards_tracking`: 每个跟踪episode的step-by-step reward
- `trajectories_tracking`: 每个跟踪episode的轨迹
- `target_positions`: 每个跟踪episode的目标位置
- `reached_targets`: 每个跟踪episode是否到达目标

## 总结

这些修改使得我们可以：
1. ✅ 只保存场景切换相关的episode（而不是随机采样）
2. ✅ 诊断场景切换时的问题
3. ✅ 分析距离变化和reward下降的关系
4. ✅ 观察agent的适应过程

通过这种方式，可以更清楚地看到场景切换时发生了什么，从而找出reward下降的根本原因。

