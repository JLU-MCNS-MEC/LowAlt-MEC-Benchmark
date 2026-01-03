# 观察空间在场景切换时的变化分析

## 当前观察空间

### 观察向量（5维）
```python
observation = [
    delta_x,              # (target_x - drone_x) / world_size
    delta_y,              # (target_y - drone_y) / world_size
    normalized_vx,        # vx / max_speed
    normalized_vy,        # vy / max_speed
    normalized_distance   # distance / world_size
]
```

## 场景切换时的变化分析

### 1. 初始观察值的变化 ⚠️ **严重**

**场景切换时**:
- 旧场景：起点(100, 200)，目标(400, 500)，距离≈424m
- 新场景：起点(50, 50)，目标(900, 950)，距离≈1273m

**观察值变化**:
- `delta_x`: 从 0.3 变到 0.85（变化 183%）
- `delta_y`: 从 0.3 变到 0.9（变化 200%）
- `normalized_distance`: 从 0.424 变到 1.273（变化 200%）
- `normalized_vx/vy`: 从 0 开始（相同）

**问题**:
- `normalized_distance`可能超过1.0（如果距离>world_size，虽然理论上不会）
- 观察值的分布发生剧烈变化
- 模型可能从未见过这么大的normalized_distance值

### 2. 观察值分布不连续 ⚠️ **严重**

**问题**:
- 旧场景中，normalized_distance主要在[0.3, 0.5]范围
- 新场景中，normalized_distance可能在[0.5, 1.0]范围
- 模型在旧场景中学习的策略可能不适用于新场景的观察分布

### 3. 相对位置变化大 ⚠️ **中等**

**问题**:
- `delta_x`和`delta_y`的绝对值可能很大
- 如果新场景距离很大，这些值可能接近±1.0
- 模型可能不熟悉这些极端值

## 快速适配方案

### 方案1: 改进观察空间归一化 ✅ **推荐**

**问题**: 当前normalized_distance = distance / world_size，但distance可能接近world_size*√2

**改进**:
```python
# 使用最大可能距离归一化
max_possible_distance = world_size * np.sqrt(2)  # 对角线距离
normalized_distance = distance / max_possible_distance  # 确保在[0, 1]
```

**效果**:
- 确保normalized_distance始终在[0, 1]范围
- 更稳定的观察分布

### 方案2: 渐进式场景难度 ✅ **关键**

**问题**: 场景切换时距离可能突然增加很多

**改进**:
```python
def generate_next_scenario(old_distance, world_size, difficulty_increase=0.1):
    """Generate next scenario with gradual difficulty increase"""
    # 新场景距离在[old_distance * 0.9, old_distance * 1.2]范围内
    min_distance = old_distance * 0.9
    max_distance = old_distance * 1.2
    
    # 生成满足距离要求的位置
    target_distance = np.random.uniform(min_distance, max_distance)
    # ... 生成位置
```

**效果**:
- 场景难度逐步增加，不会突然跳跃
- 模型更容易适应

### 方案3: 观察空间增强 ✅ **可选**

**添加方向信息**:
```python
observation = [
    delta_x,
    delta_y,
    normalized_vx,
    normalized_vy,
    normalized_distance,
    direction_x,  # cos(angle) = dx / distance
    direction_y   # sin(angle) = dy / distance
]
```

**效果**:
- 提供方向信息，帮助模型理解目标方向
- 减少对距离的依赖

### 方案4: 观察空间归一化改进 ✅ **推荐**

**使用对数归一化**:
```python
# 对于距离，使用对数归一化，减少大距离的影响
normalized_distance = np.log1p(distance) / np.log1p(max_possible_distance)
```

**效果**:
- 大距离和小距离的差异被压缩
- 更稳定的观察分布

### 方案5: 场景切换缓冲 ✅ **重要**

**问题**: 场景切换后立即评估可能不公平

**改进**:
```python
# 场景切换后，给N个episode的适应时间
adaptation_episodes = 10
# 在适应时间内，不计算成功率，或者降低成功率要求
```

**效果**:
- 给模型时间适应新场景
- 避免过早切换回旧场景

### 方案6: 观察空间标准化 ✅ **高级**

**使用运行统计标准化**:
```python
# 维护观察值的运行均值和方差
running_mean = np.zeros(5)
running_std = np.ones(5)

# 标准化观察
normalized_obs = (observation - running_mean) / (running_std + 1e-8)
```

**效果**:
- 自动适应观察分布
- 减少分布偏移的影响

## 推荐实施顺序

### 优先级1: 渐进式场景难度 ✅ **必须**
- 防止场景难度突然跳跃
- 最直接有效

### 优先级2: 改进距离归一化 ✅ **重要**
- 使用max_possible_distance归一化
- 确保观察值在合理范围

### 优先级3: 场景切换缓冲 ✅ **重要**
- 给模型适应时间
- 避免过早评估

### 优先级4: 观察空间增强 ✅ **可选**
- 添加方向信息
- 提高模型理解能力

## 预期效果

实施后：
- **观察分布更稳定**: 场景切换时观察值变化更平滑
- **快速适应**: 模型能在几个episode内适应新场景
- **更稳定的训练**: 减少reward波动

