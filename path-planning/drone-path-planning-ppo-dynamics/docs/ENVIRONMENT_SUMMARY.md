# Environment Configuration Summary

## Overview
This document summarizes the observation space, action space, and reward function settings for the drone path planning environment.

---

## 1. Observation Space (State Space)

### Dimensions: 7

The observation is a 7-dimensional vector containing normalized information about the drone's position and the target:

| Index | Component | Description | Normalization | Range |
|-------|-----------|-------------|---------------|-------|
| 0 | `normalized_drone_x` | Drone X position | `(x / world_size) * 2 - 1` | [-1, 1] |
| 1 | `normalized_drone_y` | Drone Y position | `(y / world_size) * 2 - 1` | [-1, 1] |
| 2 | `normalized_target_x` | Target X position | `(target_x / world_size) * 2 - 1` | [-1, 1] |
| 3 | `normalized_target_y` | Target Y position | `(target_y / world_size) * 2 - 1` | [-1, 1] |
| 4 | `normalized_dx` | Relative X position | `dx / world_size` | ~[-1, 1] |
| 5 | `normalized_dy` | Relative Y position | `dy / world_size` | ~[-1, 1] |
| 6 | `normalized_distance` | Distance to target | `distance / (world_size * √2)` | [0, 1] |

### Observation Calculation:
```python
dx = target_x - drone_x
dy = target_y - drone_y
distance = sqrt(dx² + dy²)

observation = [
    normalized_drone_x,      # [-1, 1]
    normalized_drone_y,      # [-1, 1]
    normalized_target_x,     # [-1, 1]
    normalized_target_y,     # [-1, 1]
    normalized_dx,           # ~[-1, 1]
    normalized_dy,           # ~[-1, 1]
    normalized_distance      # [0, 1]
]
```

### Space Definition:
- **Type**: `Box`
- **Shape**: `(7,)`
- **Dtype**: `float32`
- **Low**: `-inf`
- **High**: `inf`

---

## 2. Action Space

### Dimensions: 2

The action space is continuous, representing velocity control in X and Y directions:

| Index | Component | Description | Range |
|-------|-----------|-------------|-------|
| 0 | `vx` | Velocity in X direction | [-max_speed, max_speed] |
| 1 | `vy` | Velocity in Y direction | [-max_speed, max_speed] |

### Default Parameters:
- **max_speed**: `2.0` (default)
- **dt**: `0.1` (time step)

### Action Processing:
1. **Speed Limiting**: If the magnitude of velocity exceeds `max_speed`, it is normalized:
   ```python
   speed = sqrt(vx² + vy²)
   if speed > max_speed:
       vx = vx / speed * max_speed
       vy = vy / speed * max_speed
   ```

2. **Position Update**:
   ```python
   new_pos = current_pos + velocity * dt
   ```

3. **Boundary Handling**: 
   - If position exceeds boundaries, velocity in that direction is set to 0
   - Position is clipped to [0, world_size] for both X and Y

### Space Definition:
- **Type**: `Box`
- **Shape**: `(2,)`
- **Dtype**: `float32`
- **Low**: `-max_speed` (default: -2.0)
- **High**: `max_speed` (default: 2.0)

---

## 3. Reward Function

The reward function consists of multiple components:

### 3.1 Arrival Reward
**Condition**: `distance < 2.0`  
**Reward**: `+500.0`  
**Note**: If the drone reaches within 2.0 units of the target, it receives a large positive reward and the episode terminates.

### 3.2 Distance Reward (Base Reward)
**Formula**: `distance_reward = -normalized_distance * 10`  
**Range**: Approximately `[-10, 0]`  
**Description**: 
- Normalized distance: `distance / (world_size * √2)`
- Closer to target → higher (less negative) reward
- Provides a baseline reward signal

### 3.3 Improvement Reward
**Formula**:
```python
distance_improvement = prev_distance - current_distance

if distance_improvement > 0:
    improvement_reward = (distance_improvement / world_size) * 50
else:
    improvement_reward = (distance_improvement / world_size) * 20
```

**Description**:
- **Positive improvement**: Reward proportional to distance reduction (×50 multiplier)
- **Negative improvement** (moving away): Penalty proportional to distance increase (×20 multiplier)
- Encourages the drone to move toward the target

### 3.4 Boundary Violation Penalty
**Condition**: `boundary_violation == True`  
**Formula**: `boundary_penalty_reward = -boundary_penalty * 5.0`  
**Description**:
- `boundary_penalty` is the distance the drone exceeded the boundary
- Penalty is proportional to how far outside the boundary
- Scale factor: 5.0
- Encourages the drone to stay within bounds [0, world_size]

### 3.5 Step Penalty
**Reward**: `-0.001` per step  
**Description**: Small constant penalty to encourage efficient path planning (reaching target quickly)

### Total Reward Formula:
```python
if distance < 2.0:
    total_reward = 500.0  # Arrival reward
else:
    total_reward = distance_reward + improvement_reward + boundary_penalty_reward + step_penalty
```

### Reward Components Summary:

| Component | Formula | Typical Range | Purpose |
|-----------|---------|---------------|---------|
| Arrival | +500.0 (if distance < 2.0) | 500.0 | Large reward for success |
| Distance | -normalized_distance × 10 | [-10, 0] | Baseline distance signal |
| Improvement | ±(Δdistance / world_size) × {50 or 20} | Variable | Encourage movement toward target |
| Boundary | -boundary_penalty × 5.0 | ≤ 0 | Penalize boundary violations |
| Step | -0.001 | -0.001 | Encourage efficiency |

---

## 4. Environment Parameters

### Default Settings:
- **world_size**: `100` (square area side length)
- **max_steps**: `500` (maximum steps per episode)
- **max_speed**: `2.0` (maximum velocity magnitude)
- **dt**: `0.1` (time step)

### Episode Termination:
- **Success**: `distance < 2.0` → `terminated = True`
- **Timeout**: `step_count >= max_steps` → `truncated = True`

### Initialization:
- Drone initial position: Random uniform in [0, world_size]²
- Target position: Random uniform in [0, world_size]²
- Minimum initial distance: 5.0 units (ensures target is not too close to start)

---

## 5. Info Dictionary

The `info` dictionary returned by `step()` contains:

| Key | Type | Description |
|-----|------|-------------|
| `distance` | float | Current distance to target |
| `reached_target` | bool | Whether target was reached (distance < 2.0) |
| `boundary_violation` | bool | Whether boundary was violated in this step |
| `boundary_penalty` | float | Magnitude of boundary violation (distance exceeded) |

---

## 6. Key Design Choices

### Observation Normalization:
- All position values are normalized to facilitate learning
- Relative positions and distances provide directional information

### Continuous Action Space:
- Allows smooth, precise control
- Speed limiting prevents unrealistic movements
- Boundary handling ensures valid states

### Reward Shaping:
- **Sparse + Dense**: Large arrival reward + dense distance/improvement signals
- **Balanced**: Multiple components prevent over-optimization of single objective
- **Boundary Awareness**: Explicit penalty for boundary violations

### Training Considerations:
- Normalized observations help with neural network training
- Reward scaling balances different objectives
- Step penalty encourages efficiency without being too harsh

