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
    
    def __init__(self, world_size=100, max_steps=500, max_speed=2.0, dt=0.1, 
                 arrival_threshold=2.0, arrival_reward=50.0,  # Reduced from 100.0
                 progress_weight=100.0,  # Increased from 50.0 for stronger signal
                 step_penalty=0.01,  # Increased from 0.001
                 boundary_penalty_weight=5.0, smoothness_weight=0.1):
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
        """Reset environment"""
        super().reset(seed=seed)
        
        # Random initial drone position
        self.drone_pos = np.random.uniform(
            [0, 0], 
            [self.world_size, self.world_size]
        ).astype(np.float32)
        
        # Random target position
        self.target_pos = np.random.uniform(
            [0, 0], 
            [self.world_size, self.world_size]
        ).astype(np.float32)
        
        # Ensure target is not too close to start
        while np.linalg.norm(self.target_pos - self.drone_pos) < 5:
            self.target_pos = np.random.uniform(
                [0, 0], 
                [self.world_size, self.world_size]
            ).astype(np.float32)
        
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
        # action is [vx, vy] velocity vector
        vx, vy = action[0], action[1]
        
        # Limit speed magnitude
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
        
        # Calculate reward using current prev_distance (before update)
        reward = self._compute_reward(distance, boundary_violation, boundary_penalty, current_velocity)
        
        # Check termination conditions
        reached_target = distance < self.arrival_threshold
        truncated = self.step_count >= self.max_steps
        terminated = reached_target
        
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
        # Arrival reward
        if distance < self.arrival_threshold:
            return self.arrival_reward
        
        # Distance progress reward (main shaping signal)
        # Normalize progress by world_size for consistent scaling
        progress = self.prev_distance - distance
        normalized_progress = progress / self.world_size
        progress_reward = self.progress_weight * normalized_progress
        
        # Add small distance shaping reward (encourages getting closer)
        distance_shaping = -0.1 * (distance / self.world_size)
        
        # Boundary violation penalty (proportional to penetration depth)
        if boundary_violation:
            boundary_penalty_reward = -self.boundary_penalty_weight * boundary_penalty
        else:
            boundary_penalty_reward = 0.0
        
        # Step penalty
        step_penalty_reward = -self.step_penalty
        
        # Smoothness penalty (penalize large velocity changes)
        smoothness_penalty_reward = 0.0
        if velocity is not None and self.smoothness_weight > 0:
            velocity_change = np.linalg.norm(velocity - self.prev_velocity)
            smoothness_penalty_reward = -self.smoothness_weight * (velocity_change ** 2)
        
        total_reward = progress_reward + distance_shaping + boundary_penalty_reward + step_penalty_reward + smoothness_penalty_reward
        
        return total_reward
    
    def render(self):
        """可视化环境（可选）"""
        pass

