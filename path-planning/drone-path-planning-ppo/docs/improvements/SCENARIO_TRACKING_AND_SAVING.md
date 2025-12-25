# 场景训练记录和针对性保存功能

## 功能概述

实现了记录每个场景的训练episode，并对训练时间特别长的场景进行针对性保存的功能。

## 主要功能

### 1. 场景训练记录 ✅

**记录内容**:
- 每个场景的episode范围（开始和结束episode）
- 场景配置（起点、目标、距离）
- 最终成功率
- 总训练episode数

**数据结构**:
```python
scenario_episodes = {
    scenario_num: {
        'start_episode': int,
        'end_episode': int,
        'start_pos': [x, y],
        'target_pos': [x, y],
        'distance': float,
        'final_success_rate': float,
        'total_episodes': int
    }
}
```

### 2. 长训练场景识别 ✅

**阈值**: 200个episode（可配置）

**识别条件**:
- 场景训练episode数 >= 200
- 自动标记为"长训练场景"
- 在场景切换时自动保存检查点

**保存内容**:
- 模型检查点: `ppo_model_scenario_{N}_episode_{M}.pth`
- 场景信息记录在摘要文件中

### 3. 场景训练摘要 ✅

**保存位置**: `models/scenario_training_summary.txt`

**内容**:
- 所有场景的训练记录
- 长训练场景的详细信息
- 每个场景的配置和性能

**格式示例**:
```
================================================================================
Scenario Training Summary
================================================================================

Total scenarios: 5
Long-training scenarios (>= 200 episodes): 2
  Scenario numbers: [2, 4]

================================================================================

Scenario 1:
  Episode range: 0 - 150
  Total episodes: 150
  Start position: (234.5, 567.8)
  Target position: (789.1, 123.4)
  Distance: 456.7m
  Final success rate: 82.5%

Scenario 2:
  Episode range: 150 - 380
  Total episodes: 230
  Start position: (123.4, 567.8)
  Target position: (890.1, 234.5)
  Distance: 789.2m
  Final success rate: 80.1%
  ⚠️  LONG TRAINING SCENARIO
  Checkpoint: ppo_model_scenario_2_episode_380.pth

...
```

## 实现细节

### 1. 场景开始记录

**位置**: Episode 0 或场景切换时

```python
scenario_episodes[scenario_count] = {
    'start_episode': episode,
    'end_episode': None,  # 将在场景完成时设置
    'start_pos': current_fixed_start.copy(),
    'target_pos': current_fixed_target.copy(),
    'distance': np.linalg.norm(current_fixed_target - current_fixed_start)
}
```

### 2. 场景完成记录

**位置**: 场景切换时

```python
# 更新场景信息
scenario_episodes[scenario_count]['end_episode'] = episode
scenario_episodes[scenario_count]['final_success_rate'] = current_scenario_success_rate
scenario_episodes[scenario_count]['total_episodes'] = episodes_in_scenario

# 检查是否长训练场景
if episodes_in_scenario >= long_training_threshold:
    long_training_scenarios.append(scenario_count)
    # 保存检查点
    checkpoint_path = f'ppo_model_scenario_{scenario_count}_episode_{episode}.pth'
    agent.save(checkpoint_path)
```

### 3. 最终场景记录

**位置**: 训练结束时

```python
# 完成最终场景记录
if scenario_count in scenario_episodes:
    scenario_episodes[scenario_count]['end_episode'] = num_episodes - 1
    # 检查是否长训练场景
    if final_episodes >= long_training_threshold:
        # 保存检查点
```

## 使用场景

### 1. 诊断困难场景

**用途**: 识别哪些场景特别困难，需要更多训练

**方法**:
1. 查看 `scenario_training_summary.txt`
2. 找到长训练场景（>= 200 episodes）
3. 分析这些场景的特征（距离、位置等）

### 2. 分析场景难度

**用途**: 了解场景难度分布

**方法**:
1. 查看每个场景的训练episode数
2. 对比不同场景的距离和训练时间
3. 找出难度模式

### 3. 模型检查点恢复

**用途**: 从特定场景的检查点恢复训练

**方法**:
1. 找到长训练场景的检查点
2. 加载检查点: `agent.load('models/ppo_model_scenario_2_episode_380.pth')`
3. 继续训练或分析

## 配置参数

### 长训练阈值

**当前设置**: `long_training_threshold = 200`

**调整方法**:
```python
long_training_threshold = 200  # 在train()函数中修改
```

**建议值**:
- **200**: 标准阈值，适合大多数情况
- **150**: 更敏感，捕获更多困难场景
- **300**: 更严格，只捕获非常困难的场景

## 输出文件

### 1. 场景训练摘要

**文件**: `models/scenario_training_summary.txt`

**内容**:
- 所有场景的完整记录
- 长训练场景的详细信息
- 检查点文件路径

### 2. 模型检查点

**文件**: `models/ppo_model_scenario_{N}_episode_{M}.pth`

**保存时机**:
- 场景切换时，如果该场景训练时间 >= 阈值
- 训练结束时，如果最终场景训练时间 >= 阈值

## 分析建议

### 1. 识别困难场景模式

查看长训练场景的共同特征：
- 距离是否特别大？
- 起点和目标位置是否有特殊模式？
- 成功率是否特别低？

### 2. 优化场景生成

根据长训练场景的特征，优化场景生成策略：
- 如果距离大的场景困难，限制最大距离
- 如果某些位置组合困难，避免这些组合

### 3. 调整训练策略

根据长训练场景调整：
- 增加这些场景的训练时间
- 调整奖励函数
- 优化网络结构

## 总结

这个功能提供了：
1. ✅ **完整的场景记录**: 每个场景的训练episode范围
2. ✅ **困难场景识别**: 自动识别训练时间长的场景
3. ✅ **针对性保存**: 自动保存困难场景的检查点
4. ✅ **详细摘要**: 完整的场景训练报告

通过这些信息，可以更好地理解训练过程，识别问题场景，并优化训练策略。

