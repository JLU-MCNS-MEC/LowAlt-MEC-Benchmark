"""
无人机路径规划环境
随机生成目标点，无人机需要规划路径到达目标点
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DronePathPlanningEnv(gym.Env):
    """
    Drone path planning environment with agent-centric observations
    
    Observation space (5D):
    - Relative position: Δx, Δy (normalized)
    - Normalized velocity: vx/max_speed, vy/max_speed
    - Normalized distance: distance/world_size
    
    Action space:
    - Continuous velocity control (vx, vy) ∈ [-max_speed, max_speed]
    """
    
    def __init__(self, world_size=1000, max_steps=60, max_speed=20.0, dt=1.0, 
                 arrival_threshold=20.0, arrival_reward=100.0,  # Scaled for 1000m world
                 progress_weight=100.0,  # Increased for stronger progress signal
                 step_penalty=0.01,  # Increased from 0.001
                 boundary_penalty_weight=5.0, smoothness_weight=0.1,
                 fixed_start_pos=None, fixed_target_pos=None):
        """
        Initialize environment
        
        Args:
            world_size: World size (square area side length)
            max_steps: Maximum steps per episode
            max_speed: Maximum velocity magnitude
            dt: Time step
            arrival_threshold: Distance threshold for success (default: 2.0)
            arrival_reward: Reward for reaching target (default: 100.0)
            progress_weight: Weight for distance progress reward (default: 50.0)
            step_penalty: Penalty per step (default: 0.001)
            boundary_penalty_weight: Weight for boundary violation penalty (default: 5.0)
            smoothness_weight: Weight for smoothness penalty (default: 0.1)
            fixed_start_pos: Fixed starting position [x, y]. If None, uses random position
            fixed_target_pos: Fixed target position [x, y]. If None, uses random position
        """
        super().__init__()
        
        self.world_size = world_size
        self.max_steps = max_steps
        self.max_speed = max_speed
        self.dt = dt
        self.arrival_threshold = arrival_threshold
        self.arrival_reward = arrival_reward
        self.progress_weight = progress_weight
        self.step_penalty = step_penalty
        self.boundary_penalty_weight = boundary_penalty_weight
        self.smoothness_weight = smoothness_weight
        
        # Fixed positions (if provided)
        if fixed_start_pos is not None:
            self.fixed_start_pos = np.array(fixed_start_pos, dtype=np.float32)
        else:
            # Default fixed position: center-left of the world
            self.fixed_start_pos = np.array([world_size * 0.1, world_size * 0.5], dtype=np.float32)
        
        if fixed_target_pos is not None:
            self.fixed_target_pos = np.array(fixed_target_pos, dtype=np.float32)
        else:
            # Default fixed position: center-right of the world
            self.fixed_target_pos = np.array([world_size * 0.9, world_size * 0.5], dtype=np.float32)
        
        # Ensure fixed positions are within bounds
        self.fixed_start_pos = np.clip(self.fixed_start_pos, 0, world_size)
        self.fixed_target_pos = np.clip(self.fixed_target_pos, 0, world_size)
        
        # Action space: continuous velocity (vx, vy)
        self.action_space = spaces.Box(
            low=-max_speed,
            high=max_speed,
            shape=(2,),
            dtype=np.float32
        )
        
        # Observation space: [Δx, Δy, vx/max_speed, vy/max_speed, distance/world_size]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float32
        )
        
        # Store previous velocity for smoothness penalty
        self.prev_velocity = np.array([0.0, 0.0], dtype=np.float32)
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment - uses fixed positions"""
        super().reset(seed=seed)
        
        # Use fixed positions (same for every episode)
        self.drone_pos = self.fixed_start_pos.copy()
        self.target_pos = self.fixed_target_pos.copy()
        
        self.step_count = 0
        self.prev_distance = np.linalg.norm(self.target_pos - self.drone_pos)
        self.prev_velocity = np.array([0.0, 0.0], dtype=np.float32)
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def _get_observation(self, velocity=None):
        """
        Get agent-centric observation
        
        Observation: [Δx, Δy, vx/max_speed, vy/max_speed, distance/world_size]
        """
        # Relative position (normalized)
        dx = self.target_pos[0] - self.drone_pos[0]
        dy = self.target_pos[1] - self.drone_pos[1]
        distance = np.linalg.norm([dx, dy])
        
        delta_x = dx / self.world_size
        delta_y = dy / self.world_size
        
        # Normalized velocity (use current velocity if provided, else previous)
        if velocity is None:
            velocity = self.prev_velocity
        normalized_vx = velocity[0] / self.max_speed
        normalized_vy = velocity[1] / self.max_speed
        
        # Normalized distance
        normalized_distance = distance / self.world_size
        
        observation = np.array([
            delta_x,
            delta_y,
            normalized_vx,
            normalized_vy,
            normalized_distance
        ], dtype=np.float32)
        
        return observation
    
    def step(self, action):
        """Execute one step"""
        # action is [vx, vy] velocity vector (expected in [-1, 1] range from actor)
        # Scale to actual velocity range [-max_speed, max_speed]
        vx, vy = action[0] * self.max_speed, action[1] * self.max_speed
        
        # Limit speed magnitude (safety check)
        speed = np.sqrt(vx**2 + vy**2)
        if speed > self.max_speed:
            vx = vx / speed * self.max_speed
            vy = vy / speed * self.max_speed
        
        # Store velocity before update (for smoothness penalty)
        current_velocity = np.array([vx, vy], dtype=np.float32)
        
        # Update position: pos = pos + velocity * dt
        # Actions always take effect, we don't zero velocity at boundaries
        velocity = np.array([vx, vy], dtype=np.float32)
        new_pos = self.drone_pos + velocity * self.dt
        
        # Check boundary violation and calculate penetration depth
        boundary_violation = False
        boundary_penalty = 0.0
        
        # Calculate how far outside boundaries (penetration depth)
        if new_pos[0] < 0:
            boundary_violation = True
            boundary_penalty += abs(new_pos[0])
        elif new_pos[0] > self.world_size:
            boundary_violation = True
            boundary_penalty += (new_pos[0] - self.world_size)
            
        if new_pos[1] < 0:
            boundary_violation = True
            boundary_penalty += abs(new_pos[1])
        elif new_pos[1] > self.world_size:
            boundary_violation = True
            boundary_penalty += (new_pos[1] - self.world_size)
        
        # Clamp position to world bounds (but velocity remains unchanged)
        new_pos[0] = np.clip(new_pos[0], 0, self.world_size)
        new_pos[1] = np.clip(new_pos[1], 0, self.world_size)
        
        self.drone_pos = new_pos
        self.step_count += 1
        
        # Calculate distance (before updating prev_distance)
        distance = np.linalg.norm(self.target_pos - self.drone_pos)

        # Remove speed adjustment when close to target - let agent learn naturally
        # The previous logic was causing issues: it would slow down too much near target

        # Check termination conditions
        reached_target = distance < self.arrival_threshold
        truncated = self.step_count >= self.max_steps
        terminated = reached_target
        
        # Calculate reward using current prev_distance (before update)
        if not reached_target:
            reward = self._compute_reward(distance, boundary_violation, boundary_penalty, current_velocity)
        else:
            reward = self.arrival_reward
        
        # Get observation with current velocity
        observation = self._get_observation(current_velocity)
        
        info = {
            'distance': distance,
            'reached_target': reached_target,
            'boundary_violation': boundary_violation,
            'boundary_penalty': boundary_penalty
        }
        
        # Update previous values AFTER reward calculation
        self.prev_distance = distance
        self.prev_velocity = current_velocity.copy()
        
        return observation, reward, terminated, truncated, info
    
    def _compute_reward(self, distance, boundary_violation=False, boundary_penalty=0.0, velocity=None):
        """
        Compute reward with refactored components
        
        Reward components:
        1. Arrival reward (if reached target)
        2. Distance progress reward (main shaping signal)
        3. Boundary violation penalty
        4. Step penalty
        5. Smoothness penalty (optional)
        """
        # Arrival reward (if reached target) - scaled for larger world
        success_radius = self.world_size * 0.2  # 20% of world size
        w_funnel = 3
        if distance < success_radius:
            success_funnel_reward = w_funnel * ((success_radius - distance) / success_radius) ** 2
        else:
            success_funnel_reward = 0.0

        # Distance progress reward (main shaping signal)
        # Use absolute progress instead of normalized to provide stronger signal
        progress = self.prev_distance - distance
        # Scale progress reward: 1m progress = progress_weight reward
        # This provides much stronger signal than normalization
        progress_reward = self.progress_weight * (progress / 100.0)  # Scale by 100m instead of world_size    
        
        # Boundary violation penalty (proportional to penetration depth)
        if boundary_violation:
            boundary_penalty_reward = -self.boundary_penalty_weight * boundary_penalty
        else:
            boundary_penalty_reward = 0.0
        
        # Step penalty - removed to encourage exploration
        # Only apply minimal penalty to prevent infinite episodes
        step_penalty_reward = -self.step_penalty * 0.01
        
        # Smoothness penalty (penalize large velocity changes) - further reduced
        # Only penalize very large changes to allow exploration
        velocity_change = np.linalg.norm(velocity - self.prev_velocity)
        smoothness_reward = -0.001 * min(velocity_change**2, 100.0)  # Cap penalty
        
        total_reward = success_funnel_reward + progress_reward + boundary_penalty_reward + step_penalty_reward + smoothness_reward

        return total_reward
    
    def render(self):
        """可视化环境（可选）"""
        pass

