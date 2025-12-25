# 直线路径和边缘问题修复

## 修复内容

### 1. 添加路径效率奖励 ✅ **关键修复**

**问题**: Agent选择绕弯而不是直线路径

**修复**:
- 添加路径效率奖励，鼓励直线路径
- 效率 = 理想距离（直线） / 实际路径长度
- 权重随距离减小（接近target时权重降低）

**实现**:
```python
# 在reset()中初始化
self.initial_pos = self.drone_pos.copy()
self.initial_distance = self.prev_distance
self.total_path_length = 0.0

# 在step()中更新
step_distance = np.linalg.norm(new_pos - self.drone_pos)
self.total_path_length += step_distance

# 在_compute_reward()中计算
path_efficiency = self.initial_distance / max(self.total_path_length, self.initial_distance)
efficiency_weight = 2.0 * (1.0 - distance / self.initial_distance)
path_efficiency_reward = efficiency_weight * path_efficiency
```

**效果**:
- 直线路径获得更高奖励
- 绕弯路径被惩罚
- Agent会学习选择更直接的路径

### 2. 改进边缘处理 ✅ **关键修复**

**问题**: Target点放置在场景边缘很容易失败

**修复**:
1. **减少边缘时的边界penalty**: 当target在边缘时，边界penalty减少50%
2. **避免生成边缘target**: 场景生成时，优先选择距离边缘至少30m的位置
3. **检测边缘target**: 当target在边缘50m内时，使用特殊处理

**实现**:
```python
# 检测target是否在边缘
target_near_edge = (
    self.target_pos[0] < 50.0 or self.target_pos[0] > (self.world_size - 50.0) or
    self.target_pos[1] < 50.0 or self.target_pos[1] > (self.world_size - 50.0)
)

# 减少边缘时的边界penalty
if boundary_violation:
    if target_near_edge:
        boundary_penalty_reward = -self.boundary_penalty_weight * boundary_penalty * 0.5
    else:
        boundary_penalty_reward = -self.boundary_penalty_weight * boundary_penalty
```

**场景生成改进**:
```python
edge_margin = 30.0  # 保持target至少30m远离边缘

# 尝试生成不在边缘的target
for _ in range(max_attempts):
    candidate_pos = ...
    if (edge_margin <= candidate_pos[0] <= world_size - edge_margin and
        edge_margin <= candidate_pos[1] <= world_size - edge_margin):
        target_pos = candidate_pos
        break
```

**效果**:
- 边缘target更容易到达
- 减少边界penalty对边缘target的影响
- 场景生成时优先选择非边缘位置

### 3. 修复No Data问题 ✅ **重要修复**

**问题**: 绘图中有一些是"No data"

**修复**:
1. 确保所有跟踪的episode都有初始数据
2. 在episode开始时记录初始位置和目标
3. 使用`.get()`方法安全访问字典

**实现**:
```python
# 在episode开始时
if episode in trajectories_tracking:
    trajectories_tracking[episode].append(env.drone_pos.copy())
    if target_positions.get(episode) is None:
        target_positions[episode] = env.target_pos.copy()
    if episode not in reached_targets:
        reached_targets[episode] = False

# 在step循环中
if episode in step_rewards_tracking:
    step_rewards_tracking[episode].append(reward)
    trajectories_tracking[episode].append(env.drone_pos.copy())
    if target_positions.get(episode) is None:
        target_positions[episode] = env.target_pos.copy()
```

**效果**:
- 所有跟踪的episode都有完整数据
- 不再出现"No data"的情况
- 绘图更完整

## 预期效果

### 1. 路径更直接
- Agent会学习选择直线路径
- 减少不必要的绕弯
- 提高路径效率

### 2. 边缘target更容易到达
- 边缘target的成功率提高
- 边界penalty不再过度惩罚边缘接近
- 场景生成时优先选择非边缘位置

### 3. 数据完整性
- 所有跟踪的episode都有完整数据
- 绘图不再出现"No data"
- 更好的诊断能力

## 技术细节

### 路径效率奖励权重

**公式**:
```
efficiency_weight = 2.0 * (1.0 - distance / initial_distance)
```

**特点**:
- 距离远时权重高（鼓励直线路径）
- 距离近时权重低（接近target时路径效率不重要）
- 最大权重2.0，确保有足够信号

### 边缘检测阈值

- **边缘检测**: 50m（target距离边界<50m视为边缘）
- **场景生成安全距离**: 30m（生成target时保持至少30m距离边界）

### 边界Penalty减少

- **正常情况**: 100% penalty
- **边缘target**: 50% penalty（减少一半）

## 总结

这些修复解决了三个关键问题：
1. ✅ **绕弯问题**: 通过路径效率奖励鼓励直线路径
2. ✅ **边缘失败**: 通过减少边缘penalty和改进场景生成
3. ✅ **No data问题**: 通过改进数据跟踪逻辑

预期这些修复能够显著提高训练效果和成功率。

