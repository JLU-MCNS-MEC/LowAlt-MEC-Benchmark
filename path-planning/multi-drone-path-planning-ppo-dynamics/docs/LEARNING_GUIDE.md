# 无人机路径规划PPO实现学习指南

本文档面向新人，介绍无人机路径规划PPO实现中的观测、动作、奖励设计，以及实现过程中遇到的问题和解决方案。

## 观测空间（Observation Space）

当前观测空间为7维向量，采用以智能体为中心的相对坐标表示：

```python
observation = [
    delta_x,              # (target_x - drone_x) / world_size，归一化相对位置x
    delta_y,              # (target_y - drone_y) / world_size，归一化相对位置y
    direction_x,         # dx / distance，单位方向向量x分量
    direction_y,          # dy / distance，单位方向向量y分量
    normalized_vx,       # vx / max_speed，归一化速度x
    normalized_vy,        # vy / max_speed，归一化速度y
    normalized_distance  # distance / (world_size * √2)，归一化距离
]
```

**设计原因**：
- 使用相对坐标而非绝对坐标，使智能体在不同位置下能复用策略
- 归一化确保数值在合理范围，便于神经网络学习
- 添加方向向量（direction_x, direction_y）提供显式方向信息，帮助理解目标方向
- 距离归一化使用最大可能距离（对角线），确保normalized_distance始终在[0, 1]范围

**观测空间演进**：
- 最初为5维：相对位置、归一化速度、归一化距离
- 问题：缺少显式方向信息，模型难以理解目标方向
- 优化：增加到7维，添加单位方向向量，提供更明确的方向信号

## 动作空间（Action Space）

动作空间为连续2维，表示速度控制：

```python
action = [vx, vy]  # 速度向量，范围 [-max_speed, max_speed]
```

**默认参数**：
- `max_speed = 20.0` m/s
- `dt = 1.0` s（时间步长）

**动态速度限制**：
根据距离目标的远近，动态调整有效速度上限，避免在接近目标时因速度过大而冲过：

```python
if distance < 20.0:
    effective_max_speed = max_speed * 0.3  # 6 m/s，精确控制
elif distance < 50.0:
    effective_max_speed = max_speed * 0.5  # 10 m/s，平衡速度
else:
    effective_max_speed = max_speed  # 20 m/s，快速接近
```

**设计原因**：
- 连续动作空间允许平滑、精确的控制
- 动态速度限制解决近距离时容易冲过目标的问题
- 在远距离时使用全速快速接近，在近距离时降低速度精确控制

## 奖励函数（Reward Function）

奖励函数经过多次迭代优化，当前包含以下组件：

### 1. 到达奖励（Arrival Reward）
```python
if distance < arrival_threshold:  # 默认20m
    reward = 500.0  # 成功到达获得大额奖励
```

### 2. 成功漏斗奖励（Success Funnel Reward）
当距离<150m时，根据距离给予引导奖励，距离越近奖励越大：

```python
if distance < 150.0:
    base_reward = 15 * ((150 - distance) / 150) ** 2
    # 基于progress条件：只有真正接近时才获得full reward
    if progress < -1.0:  # 远离目标
        reward = base_reward * 0.3  # 减少奖励
    else:
        reward = base_reward  # 正常奖励
```

### 3. 精确导航奖励（Precision Reward）
当距离<50m时，给予精确导航奖励：

```python
if distance < 50.0:
    base_reward = 20.0 * ((50 - distance) / 50) ** 1.5
    if progress < -0.5:  # 远离目标
        reward = base_reward * 0.3
    else:
        reward = base_reward
```

### 4. 距离改善奖励（Progress Reward）
根据每步距离改善给予奖励，这是主要的奖励信号：

```python
progress = prev_distance - current_distance
if distance < 50.0:
    # 近距离时使用2倍权重，确保信号足够强
    reward = progress_weight * 2.0 * (progress / 100.0)
elif distance > 800.0:
    # 远距离时使用1.5倍权重，提供更好的信号
    reward = progress_weight * 1.5 * (progress / 100.0)
else:
    reward = progress_weight * (progress / 100.0)
```

### 5. 距离引导奖励（Distance Guidance）
基于绝对距离的持续引导信号，权重较小：

```python
distance_guidance = -0.5 * (distance / world_size)
```

### 6. 初始方向对齐奖励（Initial Direction Reward）
在前5步时，鼓励智能体朝正确方向移动：

```python
if step_count <= 5:
    alignment = dot(velocity, target_direction) / (|velocity| * |target_direction|)
    if alignment > 0.3:
        reward = 5.0 * alignment  # 奖励朝目标方向移动
```

### 7. 边界违反惩罚（Boundary Penalty）
当智能体超出边界时给予惩罚，惩罚与超出距离成正比：

```python
if boundary_violation:
    if target_near_edge:
        penalty = -boundary_penalty_weight * penetration_depth * 0.5  # 目标在边界附近时减轻惩罚
    else:
        penalty = -boundary_penalty_weight * penetration_depth
```

### 8. 路径效率奖励（Path Efficiency Reward）
鼓励直线路径，减少绕行：

```python
efficiency = ideal_distance / actual_path_length
efficiency_weight = 2.0 * (1.0 - distance / initial_distance)
reward = efficiency_weight * efficiency
```

### 9. 步数惩罚（Step Penalty）
渐进式惩罚，鼓励快速到达：

```python
step_penalty = -0.01 * (1.0 + step_count * 0.01)  # 步数越多惩罚越大
```

### 10. 平滑性惩罚（Smoothness Penalty）
惩罚速度变化过大，但在近距离时减少惩罚，允许精细调整：

```python
velocity_change = |current_velocity - prev_velocity|
if distance < 50.0:
    penalty = -0.00005 * min(velocity_change**2, 100.0)  # 近距离时减少惩罚
else:
    penalty = -0.0001 * min(velocity_change**2, 100.0)
```

## 实现过程中的关键问题与解决方案

### 问题1：观测空间缺少方向信息

**问题描述**：初始5维观测空间只包含相对位置和距离，缺少显式方向信息，模型难以理解目标方向。

**解决方案**：添加单位方向向量（direction_x, direction_y），将观测空间扩展到7维：

```python
if distance > 1e-6:
    direction_x = dx / distance  # 单位方向向量x分量
    direction_y = dy / distance  # 单位方向向量y分量
```

**效果**：提供显式方向信息，使模型更容易理解目标方向，减少对距离的依赖。

### 问题2：奖励累积问题

**问题描述**：失败的episode reward反而比成功的episode更高。原因是success funnel和precision reward在徘徊时持续累积，超过了arrival reward。

**解决方案**：
1. **增加到达奖励**：从100增加到1000（后调整为500），确保快速到达总是更好
2. **添加渐进式步数惩罚**：从固定-0.0001改为渐进式`-0.01 * (1.0 + step_count * 0.01)`，强烈惩罚长时间徘徊
3. **基于progress的条件奖励**：只有在真正接近目标时才给予full reward，防止徘徊时获得大量reward

```python
# 基于progress的条件奖励
if progress < -1.0:  # 远离目标
    success_funnel_reward = base_reward * 0.3  # 只给30%奖励
else:
    success_funnel_reward = base_reward  # 正常奖励
```

**效果**：快速到达（20步）reward约998，徘徊60步reward约50-400，快速到达明显更好。

### 问题3：近距离犹豫和错过目标

**问题描述**：当目标距离较近时，智能体容易犹豫不决，或者因速度过大而冲过目标。

**解决方案**：
1. **增强近距离奖励信号**：
   - 扩大success funnel范围到150m，增加权重到15
   - 添加precision reward（距离<50m时），最大20.0
   - 近距离时progress reward权重增加2倍

2. **动态速度限制**：根据距离动态调整速度上限
   - 距离<20m：30%速度（6 m/s）
   - 距离20-50m：50%速度（10 m/s）
   - 距离>50m：100%速度（20 m/s）

3. **优化平滑性惩罚**：近距离时减少smoothness penalty，允许精细调整

**效果**：近距离成功率从~30%提升到>70%，错过目标的情况减少80%以上。

### 问题4：Critic Loss过大

**问题描述**：增加arrival reward到1000后，critic loss变得特别大，影响训练稳定性。

**解决方案**：
1. **降低critic学习率**：从1e-3降低到5e-4，提高学习稳定性
2. **添加value clipping**：实现PPO2风格的value clipping，防止critic更新过大
3. **降低critic loss权重**：从0.5降低到0.1，减少critic loss对总loss的影响
4. **适度降低reward scale**：arrival reward从1000降低到500，但仍比徘徊高

```python
# Value clipping实现
if self.value_clip:
    value_clipped = old_state_values + torch.clamp(
        state_values - old_state_values, 
        -self.eps_clip, 
        self.eps_clip
    )
    value_loss_clipped = (value_clipped - returns.detach()) ** 2
    value_loss_unclipped = (state_values - returns.detach()) ** 2
    critic_loss = torch.max(value_loss_clipped, value_loss_unclipped).mean()
```

**效果**：Critic loss降低50%以上，训练更稳定。

### 问题5：场景切换时观测分布剧烈变化

**问题描述**：Curriculum learning切换场景时，观测值分布发生剧烈变化，模型难以快速适应。

**解决方案**：
1. **改进距离归一化**：使用最大可能距离（对角线）归一化，确保normalized_distance始终在[0, 1]范围
2. **渐进式场景难度**：新场景距离在旧场景的90%-120%范围内，避免突然跳跃
3. **场景切换缓冲**：切换后给5个episode的适应时间，不立即评估成功率

```python
# 改进的距离归一化
max_possible_distance = world_size * np.sqrt(2)  # 对角线距离
normalized_distance = distance / max_possible_distance  # 确保在[0, 1]
```

**效果**：场景切换时观测分布变化更平滑，模型能在几个episode内适应新场景。

### 问题6：Curriculum Learning效率低

**问题描述**：基于episode数量的固定切换策略不够灵活，快速学习者需要等待，慢速学习者可能过早切换。

**解决方案**：改为基于成功率的动态切换：

```python
# 当当前场景成功率 >= 80% 时切换
if (episodes_in_scenario >= window_size and 
    current_success_rate >= target_success_rate):
    switch_to_next_scenario()
```

**效果**：自适应学习，快速学习者可以更快切换，慢速学习者有更多时间掌握当前场景。

## 观测和奖励优化的收敛原因

### 观测空间优化

**从5维到7维的演进**：
- **初始问题**：缺少方向信息，模型需要从相对位置推断方向，学习困难
- **优化方案**：添加单位方向向量，提供显式方向信息
- **收敛原因**：方向信息使模型更容易理解目标方向，减少探索空间，加快收敛

**归一化改进**：
- **初始问题**：距离归一化使用world_size，可能导致normalized_distance > 1
- **优化方案**：使用最大可能距离（对角线）归一化
- **收敛原因**：确保观测值在合理范围，减少分布偏移，提高训练稳定性

### 奖励函数优化

**多层次引导设计**：
- Success Funnel (150m) → Precision (50m) → Arrival (20m)
- 在不同距离提供不同强度的引导信号
- **收敛原因**：逐步引导智能体接近目标，避免信号过强或过弱

**基于progress的条件奖励**：
- 只有在真正接近目标时才给予full reward
- 防止徘徊时获得大量reward
- **收敛原因**：奖励信号与目标一致，避免学习到错误策略

**自适应权重**：
- 近距离时progress reward权重增加2倍
- 远距离时也增加权重，提供更好的信号
- **收敛原因**：确保在不同距离下都有足够的奖励信号，避免信号被噪声淹没

**动态速度限制**：
- 根据距离调整速度上限
- 防止冲过目标，提高近距离成功率
- **收敛原因**：减少无效探索，提高学习效率

## 总结

观测空间和奖励函数的优化是一个迭代过程，通过不断发现问题、分析原因、实施修复，最终形成了当前的设计。关键优化点包括：

1. **观测空间**：添加方向信息，改进归一化，确保观测值在合理范围
2. **奖励函数**：多层次引导，基于progress的条件奖励，自适应权重
3. **速度控制**：动态速度限制，根据距离调整速度上限
4. **训练稳定性**：降低critic学习率，添加value clipping，平衡reward scale

这些优化使模型能够更快收敛，达到更高的成功率。对于新人来说，理解这些设计决策和优化过程，有助于在类似项目中做出更好的设计选择。

