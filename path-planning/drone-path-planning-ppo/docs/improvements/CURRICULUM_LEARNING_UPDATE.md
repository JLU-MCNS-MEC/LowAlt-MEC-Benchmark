# Curriculum Learning 更新 - 基于成功率的场景切换

## 修改日期
2024年（当前日期）

## 修改概述

将curriculum learning的场景切换逻辑从**基于episode数量**改为**基于成功率**。现在系统会在当前场景的成功率达到80%后自动切换到下一个场景。

## 主要修改

### 1. 场景切换逻辑改变

**之前**: 基于episode数量固定切换
- Stage 1: 每500个episode切换
- Stage 2: 每100个episode切换
- Stage 3: 每50个episode切换
- Stage 4: 每10个episode切换
- Stage 5: 每个episode切换（完全随机）

**现在**: 基于成功率动态切换
- 当当前场景的成功率 >= 80% 时，切换到下一个场景
- 需要至少50个episode来评估成功率（可配置）
- 每个场景独立跟踪成功率

### 2. 新增参数

在`train()`函数中添加了两个新参数：

```python
def train(
    ...
    use_curriculum=True,
    target_success_rate=80.0,  # 切换场景的目标成功率 (%)
    scenario_success_rate_window=50  # 计算成功率的窗口大小（episode数）
):
```

**参数说明**:
- `target_success_rate`: 目标成功率阈值，默认80.0%
- `scenario_success_rate_window`: 计算成功率时使用的最近N个episode，默认50

### 3. 实现细节

#### 3.1 场景成功率跟踪

```python
# 为每个场景单独跟踪成功率
scenario_successes = []  # 存储当前场景的成功/失败记录
scenario_count = 0  # 场景计数器
```

#### 3.2 场景切换条件

```python
# 计算当前场景成功率
current_scenario_success_rate = np.mean(scenario_successes) * 100

# 切换条件：
# 1. 在当前场景中至少运行了N个episode（窗口大小）
# 2. 成功率 >= 目标成功率（默认80%）
if (episodes_in_current_scenario >= scenario_success_rate_window and 
    current_scenario_success_rate >= target_success_rate):
    # 切换到下一个场景
```

#### 3.3 场景切换时的行为

- 生成新的随机起点和目标点
- 重置场景成功率跟踪（`scenario_successes = []`）
- 更新场景计数器
- 打印场景切换信息

### 4. 输出信息增强

#### 4.1 训练开始时的信息

```
Using curriculum learning for position strategy
  Target success rate: 80.0%
  Success rate window: 50 episodes
```

#### 4.2 场景切换时的信息

```
Episode 150: Scenario 1 completed!
  Final success rate: 82.5%
  Episodes in scenario: 150
  Starting scenario 2
  Start: (234.5, 567.8)
  Target: (789.1, 123.4)
  Distance: 456.7m
```

#### 4.3 定期输出（每100个episode）

```
Episode 200
  Average reward: 45.23
  Global success rate: 65.0%
  Current scenario success rate: 78.5% (50 episodes)
  Scenario 2, Episodes in scenario: 50
  Average steps: 45.2
  ...
```

#### 4.4 训练结束时的总结

```
Training completed!
Final global success rate: 75.0%
Final average reward: 50.12

Curriculum Learning Summary:
  Total scenarios completed: 5
  Final scenario (5) success rate: 85.0%
  Episodes in final scenario: 120
```

## 优势

### 1. 自适应学习
- 不再固定切换时间，根据实际学习效果动态调整
- 快速学习者可以更快切换到新场景
- 慢速学习者有更多时间掌握当前场景

### 2. 更稳定的学习
- 确保每个场景都达到一定成功率后再切换
- 避免过早切换到困难场景导致学习失败
- 提供更平滑的难度曲线

### 3. 更好的监控
- 可以清楚看到每个场景的学习进度
- 了解agent在不同场景下的表现
- 便于调试和优化

## 使用示例

### 默认配置（80%成功率，50个episode窗口）

```python
train(
    num_episodes=8000,
    use_curriculum=True  # 使用默认的80%阈值和50个episode窗口
)
```

### 自定义配置

```python
train(
    num_episodes=8000,
    use_curriculum=True,
    target_success_rate=90.0,  # 更严格：需要90%成功率
    scenario_success_rate_window=100  # 更大的窗口：使用最近100个episode
)
```

### 更宽松的配置（适合快速测试）

```python
train(
    num_episodes=4000,
    use_curriculum=True,
    target_success_rate=70.0,  # 更宽松：70%即可切换
    scenario_success_rate_window=30  # 更小的窗口：更快评估
)
```

## 注意事项

### 1. 窗口大小选择
- **太小**（如20）：可能受噪声影响，不够稳定
- **太大**（如100）：需要更长时间才能切换，可能过于保守
- **推荐**：50-100个episode

### 2. 成功率阈值选择
- **太低**（如60%）：可能过早切换，导致学习不稳定
- **太高**（如95%）：可能长时间停留在简单场景，学习效率低
- **推荐**：75%-85%

### 3. 场景数量
- 场景数量不再固定，取决于学习速度
- 快速学习者可能经历更多场景
- 慢速学习者可能经历较少场景

### 4. 训练时间
- 总训练时间可能因场景切换而有所不同
- 如果某个场景一直无法达到80%成功率，可能长时间停留
- 建议设置最大episode数限制

## 与旧版本的兼容性

- 如果`use_curriculum=False`，行为与之前完全相同（每个episode随机位置）
- 新参数都有默认值，不影响现有代码
- `get_position_change_frequency()`函数仍然存在，但不再使用

## 未来可能的改进

1. **自适应阈值**: 根据全局成功率动态调整目标成功率
2. **难度递增**: 根据场景编号逐渐增加目标距离
3. **场景难度评估**: 根据场景难度（如距离）调整成功率阈值
4. **最小/最大场景时间**: 设置每个场景的最小/最大episode数限制

## 测试建议

1. **短期测试**: 使用较小的窗口（30）和较低的成功率（70%）快速验证
2. **完整训练**: 使用默认配置（50窗口，80%成功率）进行完整训练
3. **监控指标**: 关注场景切换频率和每个场景的学习曲线

