# 动作、观测与奖励设计分析

## 一、当前设计概览

### 1.1 动作空间（Action Space）
- **维度**: 3维连续动作
- **动作**: `[thrust, roll_torque, pitch_torque]`
  - `thrust`: [-1, 1] → [0, max_thrust] (推力)
  - `roll_torque`: [-1, 1] → [-max_torque, max_torque] (滚转力矩)
  - `pitch_torque`: [-1, 1] → [-max_torque, max_torque] (俯仰力矩)
- **特点**: 没有yaw_torque控制

### 1.2 观测空间（Observation Space）
- **维度**: 13维
- **组成**:
  1. 位置信息 (4维): `Δx, Δy, direction_x, direction_y`
  2. 速度信息 (2维): `vx/max_speed, vy/max_speed`
  3. 距离信息 (1维): `distance/world_size`
  4. 姿态信息 (3维): `roll/π, pitch/π, yaw/π`
  5. 角速度信息 (3维): `roll_vel/max_angular_vel, pitch_vel/max_angular_vel, yaw_vel/max_angular_vel`

### 1.3 奖励函数（Reward Function）
- **主要组件**:
  1. Arrival reward: 500.0 (到达目标)
  2. Success funnel reward: 0-15 (距离<150m)
  3. Precision reward: 0-20 (距离<50m)
  4. Progress reward: 基于距离改善 (主要信号)
  5. Distance guidance: -0.5 * (distance/world_size)
  6. Initial direction reward: 前5步的方向对齐奖励
  7. Boundary penalty: 边界违反惩罚
  8. Path efficiency: 路径效率奖励
  9. Step penalty: 步数惩罚
  10. Smoothness penalty: 速度和动作变化惩罚

## 二、设计匹配度分析

### 2.1 ✅ 匹配良好的方面

#### 1. 观测空间包含动力学状态
- **姿态信息完整**: roll, pitch, yaw都在观测中
- **角速度信息完整**: roll_vel, pitch_vel, yaw_vel都在观测中
- **速度信息**: vx, vy用于计算奖励和引导

#### 2. 动作到状态的物理映射清晰
```
thrust → 通过旋转矩阵 → 加速度 → 速度 → 位置
roll_torque → roll_vel → roll → 影响推力方向
pitch_torque → pitch_vel → pitch → 影响推力方向
```

#### 3. 奖励函数关注主要目标
- 距离改善是主要奖励信号
- 速度方向对齐有初始引导

### 2.2 ⚠️ 存在的问题

#### 问题1: 奖励函数与动作空间不匹配

**问题描述**:
- 奖励函数主要基于**位置和速度**（距离、progress、方向对齐）
- 但动作空间是**动力学控制**（thrust, roll_torque, pitch_torque）
- **缺少对姿态控制的直接奖励信号**

**影响**:
- Agent可能学习到"通过快速改变姿态来快速改变速度"的策略
- 但奖励函数不直接奖励"好的姿态控制"
- 可能导致不稳定的飞行（频繁的姿态变化）

**示例**:
```python
# 当前奖励主要看这些：
progress_reward = progress_weight * (progress / 100.0)  # 只看距离改善
initial_direction_reward = 5.0 * alignment  # 只看速度方向

# 但没有奖励：
# - 姿态稳定性（roll, pitch接近0）
# - 平滑的姿态变化
# - 合理的姿态角度（避免过大倾斜）
```

#### 问题2: Yaw控制缺失

**问题描述**:
- 动作空间没有`yaw_torque`
- Yaw是根据速度方向自动更新的（代码333-343行）
- 但观测空间包含`yaw`和`yaw_vel`

**影响**:
- Agent无法主动控制yaw
- Yaw的自动更新可能与Agent的意图不一致
- 观测中的yaw信息对Agent来说可能是"噪声"

**代码位置**:
```python
# environment.py 333-343行
if np.linalg.norm(self.drone_vel) > 0.1:
    desired_yaw = np.arctan2(self.drone_vel[1], self.drone_vel[0])
    yaw_error = desired_yaw - yaw
    self.drone_attitude[2] += 0.1 * yaw_error  # 自动更新yaw
```

#### 问题3: 观测归一化可能不合理

**问题描述**:
- 姿态归一化: `roll/π, pitch/π, yaw/π`
- 角速度归一化: `roll_vel/5.0, pitch_vel/5.0, yaw_vel/5.0`
- 但实际物理范围可能不同

**影响**:
- 如果实际roll/pitch很少超过±π/4，归一化后范围是[-0.25, 0.25]
- 如果实际角速度很少超过±2 rad/s，归一化后范围是[-0.4, 0.4]
- 这些值可能太小，Agent难以学习

#### 问题4: 奖励函数对姿态不敏感

**问题描述**:
- Smoothness penalty只惩罚速度和动作变化
- 但没有直接惩罚**姿态变化**或**姿态角度过大**

**影响**:
- Agent可能学习到"快速倾斜然后快速恢复"的策略
- 这在实际无人机中是不安全的
- 奖励函数没有引导Agent保持稳定姿态

#### 问题5: 动力学参数可能不合理

**问题描述**:
- `max_thrust = 5.0 N`, `m = 0.2 kg` → 最大加速度 = 25 m/s²
- `max_torque = 2.0 N·m`, `Ixx = Iyy = 1.0 kg·m²` → 最大角加速度 = 2 rad/s²
- `dt = 1.0s` → 每步可能产生很大的变化

**影响**:
- 在1秒内，角速度可能变化±2 rad/s
- 姿态可能变化±2 rad（约115度）
- 这可能导致非常不稳定的控制

## 三、改进建议

### 3.1 奖励函数改进

#### 建议1: 添加姿态稳定性奖励
```python
# 奖励接近水平姿态（roll, pitch接近0）
attitude_stability_reward = -0.1 * (abs(roll) + abs(pitch)) / np.pi
```

#### 建议2: 添加姿态平滑性奖励
```python
# 惩罚姿态变化过大
attitude_change = abs(roll - prev_roll) + abs(pitch - prev_pitch)
attitude_smoothness_reward = -0.05 * attitude_change / np.pi
```

#### 建议3: 添加姿态角度限制惩罚
```python
# 惩罚过大倾斜（不安全）
max_safe_angle = np.pi / 6  # 30度
if abs(roll) > max_safe_angle or abs(pitch) > max_safe_angle:
    attitude_penalty = -1.0 * (abs(roll) + abs(pitch) - 2 * max_safe_angle) / np.pi
```

### 3.2 动作空间改进

#### 建议1: 添加yaw_torque控制
```python
# 动作空间改为4维
action_space = [thrust, roll_torque, pitch_torque, yaw_torque]
```

#### 建议2: 限制动作变化率
```python
# 限制每步动作变化，避免突变
action_change = action - prev_action
max_action_change = 0.5
action = prev_action + np.clip(action_change, -max_action_change, max_action_change)
```

### 3.3 观测空间改进

#### 建议1: 添加动作历史
```python
# 添加上一时刻的动作，帮助Agent理解动作的影响
observation = [..., prev_thrust, prev_roll_torque, prev_pitch_torque]
```

#### 建议2: 改进归一化
```python
# 使用更合理的归一化范围
# 例如：如果roll/pitch通常在±π/4范围内，归一化到[-1, 1]
normalized_roll = roll / (np.pi / 4)  # 而不是 roll / π
```

### 3.4 动力学参数调整

#### 建议1: 减小时间步长
```python
dt = 0.1  # 从1.0改为0.1，更精细的控制
max_steps = 600  # 保持总时长60秒
```

#### 建议2: 调整动力学参数
```python
# 减小最大力矩，使控制更平滑
max_torque = 1.0  # 从2.0改为1.0

# 或增加阻尼
angular_vel_damping = 0.9  # 从0.95改为0.9
```

## 四、优先级建议

### 高优先级（必须修复）
1. **添加姿态稳定性奖励** - 引导Agent学习稳定飞行
2. **添加姿态角度限制惩罚** - 确保安全飞行
3. **减小时间步长或调整动力学参数** - 避免过度控制

### 中优先级（建议修复）
1. **添加yaw_torque控制** - 给Agent完整控制能力
2. **改进观测归一化** - 提高学习效率
3. **添加动作历史到观测** - 帮助理解动作影响

### 低优先级（可选优化）
1. **添加姿态平滑性奖励** - 进一步优化飞行质量
2. **限制动作变化率** - 避免突变

## 五、已实现的改进

### ✅ 已完成的改进（2024年更新）

#### 1. 添加yaw_torque控制
- **实现**: 动作空间从3维扩展到4维，添加了`yaw_torque`
- **代码位置**: `environment.py` 第103-108行
- **效果**: Agent现在可以主动控制yaw，不再依赖自动更新

#### 2. 添加姿态稳定性奖励
- **实现**: 在奖励函数中添加了`attitude_stability_reward`
- **公式**: `-0.1 * (abs(roll) + abs(pitch)) / (π/4)`
- **效果**: 引导Agent学习保持水平姿态

#### 3. 添加姿态角度限制惩罚
- **实现**: 在奖励函数中添加了`attitude_angle_penalty`
- **阈值**: 30度（π/6）
- **效果**: 惩罚过大倾斜，确保安全飞行

#### 4. 添加姿态平滑性奖励
- **实现**: 在奖励函数中添加了`attitude_smoothness_reward`
- **公式**: `-0.05 * min(attitude_change², (π/4)²) / (π/4)²`
- **效果**: 惩罚快速姿态变化，鼓励平滑控制

#### 5. 改进观测归一化
- **实现**: 姿态归一化从`roll/π`改为`roll/(π/4)`
- **效果**: 更合理的归一化范围，提高学习效率
- **角速度归一化**: 从5.0 rad/s改为3.0 rad/s

#### 6. 添加动作历史到观测
- **实现**: 观测空间从13维扩展到17维，添加了4个上一时刻的动作
- **效果**: 帮助Agent理解动作的影响

#### 7. 调整动力学参数
- **实现**: `max_torque`从2.0 N·m减小到1.0 N·m
- **效果**: 提供更平滑的控制，减少过度控制

#### 8. 移除yaw自动更新
- **实现**: 删除了基于速度方向的yaw自动更新代码
- **效果**: yaw完全由Agent通过yaw_torque控制

### 📊 改进前后对比

| 特性 | 改进前 | 改进后 |
|------|--------|--------|
| 动作维度 | 3维 | 4维（+yaw_torque） |
| 观测维度 | 13维 | 17维（+动作历史） |
| 姿态归一化 | roll/π | roll/(π/4) |
| 最大力矩 | 2.0 N·m | 1.0 N·m |
| 姿态奖励 | ❌ 无 | ✅ 稳定性+角度限制+平滑性 |
| Yaw控制 | 自动更新 | Agent控制 |

## 六、总结

### 当前设计的优点
- ✅ 观测空间包含完整的动力学状态
- ✅ 动作到状态的物理映射清晰
- ✅ 奖励函数关注主要目标（到达目标）
- ✅ **新增**: 奖励函数包含姿态相关奖励
- ✅ **新增**: 完整的4维动作控制
- ✅ **新增**: 动作历史在观测中

### 已解决的主要问题
- ✅ 奖励函数与动作空间匹配（添加了姿态相关奖励）
- ✅ Yaw控制完整（添加了yaw_torque）
- ✅ 姿态稳定性引导（添加了相关奖励）
- ✅ 观测归一化改进（使用更合理的范围）
- ✅ 动作历史添加（帮助理解动作影响）

### 可能的进一步优化方向
1. **时间步长**: 如果控制仍然粗糙，可以考虑减小dt到0.1s
2. **奖励权重调优**: 根据训练效果调整姿态奖励的权重
3. **动作变化率限制**: 如果出现动作突变，可以添加变化率限制

这些改进使动作设计、观测空间和奖励函数更好地匹配，应该能显著提高训练效率和最终性能。

