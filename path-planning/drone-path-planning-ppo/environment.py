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
                 arrival_threshold=20.0, arrival_reward=500.0,  # Balanced: high enough to prefer arrival, but not too large for stable training
                 progress_weight=200.0,  # Further increased for stronger progress signal
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
        self.use_fixed_positions = fixed_start_pos is not None and fixed_target_pos is not None
        
        if self.use_fixed_positions:
            self.fixed_start_pos = np.array(fixed_start_pos, dtype=np.float32)
            self.fixed_target_pos = np.array(fixed_target_pos, dtype=np.float32)
            # Ensure fixed positions are within bounds
            self.fixed_start_pos = np.clip(self.fixed_start_pos, 0, world_size)
            self.fixed_target_pos = np.clip(self.fixed_target_pos, 0, world_size)
        else:
            # Will use random positions in reset()
            self.fixed_start_pos = None
            self.fixed_target_pos = None
        
        # Action space: continuous velocity (vx, vy)
        self.action_space = spaces.Box(
            low=-max_speed,
            high=max_speed,
            shape=(2,),
            dtype=np.float32
        )
        
        # Observation space: [Δx, Δy, direction_x, direction_y, vx/max_speed, vy/max_speed, distance/world_size]
        # Added direction_x and direction_y (unit direction vector) for explicit direction information
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),  # Increased from 5 to 7
            dtype=np.float32
        )
        
        # Store previous velocity for smoothness penalty
        self.prev_velocity = np.array([0.0, 0.0], dtype=np.float32)
        
        self.reset()
    
    def set_fixed_positions(self, start_pos, target_pos):
        """
        Dynamically set fixed positions (for curriculum learning)
        
        Args:
            start_pos: [x, y] start position
            target_pos: [x, y] target position
        """
        self.fixed_start_pos = np.array(start_pos, dtype=np.float32)
        self.fixed_target_pos = np.array(target_pos, dtype=np.float32)
        # Ensure positions are within bounds
        self.fixed_start_pos = np.clip(self.fixed_start_pos, 0, self.world_size)
        self.fixed_target_pos = np.clip(self.fixed_target_pos, 0, self.world_size)
        self.use_fixed_positions = True
    
    def set_random_positions(self):
        """Enable random positions for each episode"""
        self.use_fixed_positions = False
        self.fixed_start_pos = None
        self.fixed_target_pos = None
    
    def reset(self, seed=None, options=None):
        """Reset environment - uses fixed positions if set, otherwise random positions"""
        super().reset(seed=seed)
        
        if self.use_fixed_positions:
            # Use fixed positions (same for every episode)
            self.drone_pos = self.fixed_start_pos.copy()
            self.target_pos = self.fixed_target_pos.copy()
        else:
            # Use random positions for each episode
            # Random initial drone position
            self.drone_pos = np.random.uniform(
                [0, 0], 
                [self.world_size, self.world_size]
            ).astype(np.float32)
            
            # Random target position - ensure it's not too close to start
            min_distance = self.world_size * 0.05  # 5% of world size
            max_attempts = 10
            for _ in range(max_attempts):
                self.target_pos = np.random.uniform(
                    [0, 0], 
                    [self.world_size, self.world_size]
                ).astype(np.float32)
                if np.linalg.norm(self.target_pos - self.drone_pos) >= min_distance:
                    break
        
        self.step_count = 0
        self.prev_distance = np.linalg.norm(self.target_pos - self.drone_pos)
        self.prev_velocity = np.array([0.0, 0.0], dtype=np.float32)
        
        # Track initial position and ideal path for path efficiency reward
        self.initial_pos = self.drone_pos.copy()
        self.initial_distance = self.prev_distance
        self.total_path_length = 0.0  # Track total path length
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def _get_observation(self, velocity=None):
        """
        Get agent-centric observation
        
        Observation: [Δx, Δy, direction_x, direction_y, vx/max_speed, vy/max_speed, distance/world_size]
        Added direction_x and direction_y (unit direction vector) for explicit direction information
        """
        # Relative position (normalized)
        dx = self.target_pos[0] - self.drone_pos[0]
        dy = self.target_pos[1] - self.drone_pos[1]
        distance = np.linalg.norm([dx, dy])
        
        delta_x = dx / self.world_size
        delta_y = dy / self.world_size
        
        # Unit direction vector (explicit direction information)
        # This makes direction information more explicit and easier to learn
        if distance > 1e-6:  # Avoid division by zero
            direction_x = dx / distance  # Unit direction vector x component [-1, 1]
            direction_y = dy / distance  # Unit direction vector y component [-1, 1]
        else:
            direction_x = 0.0
            direction_y = 0.0
        
        # Normalized velocity (use current velocity if provided, else previous)
        if velocity is None:
            velocity = self.prev_velocity
        normalized_vx = velocity[0] / self.max_speed
        normalized_vy = velocity[1] / self.max_speed
        
        # Normalized distance - use max possible distance (diagonal) for normalization
        # This ensures normalized_distance is always in [0, 1] range, even for large distances
        max_possible_distance = self.world_size * np.sqrt(2)  # Diagonal distance
        normalized_distance = distance / max_possible_distance
        
        observation = np.array([
            delta_x,
            delta_y,
            direction_x,        # Unit direction vector x component
            direction_y,        # Unit direction vector y component
            normalized_vx,
            normalized_vy,
            normalized_distance
        ], dtype=np.float32)
        
        return observation
    
    def step(self, action):
        """Execute one step"""
        # Calculate distance before processing action (for dynamic speed limit)
        distance_before = np.linalg.norm(self.target_pos - self.drone_pos)
        
        # Dynamic speed limit based on distance to target
        # When close to target, reduce max speed for more precise control
        if distance_before < 20.0:
            # Very close: use 30% of max speed for precise control
            effective_max_speed = self.max_speed * 0.3  # 6 m/s
        elif distance_before < 50.0:
            # Close: use 50% of max speed
            effective_max_speed = self.max_speed * 0.5  # 10 m/s
        else:
            # Far: use full max speed
            effective_max_speed = self.max_speed
        
        # action is [vx, vy] velocity vector (expected in [-1, 1] range from actor)
        # Scale to actual velocity range [-effective_max_speed, effective_max_speed]
        vx, vy = action[0] * effective_max_speed, action[1] * effective_max_speed
        
        # Limit speed magnitude (safety check)
        speed = np.sqrt(vx**2 + vy**2)
        if speed > effective_max_speed:
            vx = vx / speed * effective_max_speed
            vy = vy / speed * effective_max_speed
        
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
        
        # Calculate step distance for path length tracking
        step_distance = np.linalg.norm(new_pos - self.drone_pos)
        self.total_path_length += step_distance
        
        self.drone_pos = new_pos
        self.step_count += 1
        
        # Calculate distance (before updating prev_distance)
        distance = np.linalg.norm(self.target_pos - self.drone_pos)
        
        # Check if target is near edge (within 50m of any boundary)
        target_near_edge = (
            self.target_pos[0] < 50.0 or self.target_pos[0] > (self.world_size - 50.0) or
            self.target_pos[1] < 50.0 or self.target_pos[1] > (self.world_size - 50.0)
        )

        # Remove speed adjustment when close to target - let agent learn naturally
        # The previous logic was causing issues: it would slow down too much near target

        # Check termination conditions
        reached_target = distance < self.arrival_threshold
        truncated = self.step_count >= self.max_steps
        terminated = reached_target
        
        # Calculate reward using current prev_distance (before update)
        if not reached_target:
            reward = self._compute_reward(distance, boundary_violation, boundary_penalty, current_velocity, target_near_edge)
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
    
    def _compute_reward(self, distance, boundary_violation=False, boundary_penalty=0.0, velocity=None, target_near_edge=False):
        """
        Compute reward with refactored components
        
        Reward components:
        1. Arrival reward (if reached target)
        2. Distance progress reward (main shaping signal)
        3. Close-range precision navigation reward (when distance < 50m)
        4. Success funnel reward (when distance < 150m)
        5. Distance guidance reward (small weight, always active)
        6. Path efficiency reward (encourage straight paths)
        7. Boundary violation penalty (reduced when target near edge)
        8. Step penalty
        9. Smoothness penalty (reduced when close to target)
        """
        # Calculate progress first (needed for funnel and precision rewards)
        progress = self.prev_distance - distance
        
        # Success funnel reward - simplified for easier learning
        # Give reward when close to target, with reduced penalty for no progress
        success_radius = self.arrival_threshold * 7.5  # 7.5x arrival threshold (150m) for earlier guidance
        w_funnel = 15  # Increased weight for better guidance
        if distance < success_radius:
            base_funnel_reward = w_funnel * ((success_radius - distance) / success_radius) ** 2
            # Simplified: only reduce if moving away significantly
            if progress < -1.0:  # Only penalize if moving away by more than 1m
                # Moving away: reduce funnel reward
                success_funnel_reward = base_funnel_reward * 0.3  # Less harsh penalty
            else:
                # Any progress or staying: full funnel reward
                success_funnel_reward = base_funnel_reward
        else:
            success_funnel_reward = 0.0

        # Close-range precision navigation reward (when distance < 50m)
        # Strong reward to encourage precise navigation when close to target
        # Simplified: only penalize if moving away significantly
        precision_radius = 50.0  # 50m threshold
        if distance < precision_radius:
            base_precision_reward = 20.0 * ((precision_radius - distance) / precision_radius) ** 1.5
            # Simplified: only reduce if moving away significantly
            if progress < -0.5:  # Only penalize if moving away by more than 0.5m
                # Moving away: reduce precision reward
                precision_reward = base_precision_reward * 0.3  # Less harsh penalty
            else:
                # Any progress or staying: full precision reward
                precision_reward = base_precision_reward
        else:
            precision_reward = 0.0

        # Distance progress reward (main shaping signal)
        # Use adaptive weight: stronger when close to target OR when very far (to help with long distances)
        if distance < 50.0:
            # When close, use stronger weight for progress
            close_progress_weight = self.progress_weight * 2.0  # 2x weight when close
            progress_reward = close_progress_weight * (progress / 100.0)
        elif distance > 800.0:
            # When very far, also use stronger weight to provide better signal
            far_progress_weight = self.progress_weight * 1.5  # 1.5x weight when very far
            progress_reward = far_progress_weight * (progress / 100.0)
        else:
            # Normal weight when medium distance
            progress_reward = self.progress_weight * (progress / 100.0)
        
        # Distance guidance reward (small weight, always active)
        # Provides constant guidance signal based on absolute distance
        # Helps agent maintain direction, especially when close
        distance_guidance = -0.5 * (distance / self.world_size)  # Small negative reward based on distance
        
        # Initial direction alignment reward (first few steps)
        # Encourages agent to move in the correct direction from the start
        # Extended to first 5 steps for stronger initial guidance
        initial_direction_reward = 0.0
        if self.step_count <= 5 and velocity is not None:
            # Calculate target direction
            dx = self.target_pos[0] - self.drone_pos[0]
            dy = self.target_pos[1] - self.drone_pos[1]
            target_direction = np.array([dx, dy])
            movement_direction = np.array([velocity[0], velocity[1]])
            
            # Calculate direction alignment (cosine similarity)
            if np.linalg.norm(movement_direction) > 0.1 and np.linalg.norm(target_direction) > 1e-6:
                alignment = np.dot(movement_direction, target_direction) / (
                    np.linalg.norm(movement_direction) * np.linalg.norm(target_direction)
                )
                # Reward high alignment (moving towards target) - increased weight
                if alignment > 0.3:  # Lower threshold (30%) for easier reward
                    initial_direction_reward = 5.0 * alignment  # Increased from 3.0 to 5.0
                elif alignment < -0.3:  # Penalty for moving away
                    initial_direction_reward = -3.0 * abs(alignment)  # Increased penalty
        
        # Boundary violation penalty (proportional to penetration depth)
        # Reduce penalty when target is near edge (agent needs to approach edge)
        if boundary_violation:
            if target_near_edge:
                # Reduce boundary penalty when target is near edge (50% penalty)
                boundary_penalty_reward = -self.boundary_penalty_weight * boundary_penalty * 0.5
            else:
                boundary_penalty_reward = -self.boundary_penalty_weight * boundary_penalty
        else:
            boundary_penalty_reward = 0.0
        
        # Path efficiency reward - encourage straight paths
        # Efficiency = ideal_distance / actual_path_length
        # Ideal path is straight line from start to target
        if self.total_path_length > 0:
            path_efficiency = self.initial_distance / max(self.total_path_length, self.initial_distance)
            # Reward efficiency, but weight decreases as we get closer (less important when close)
            efficiency_weight = 2.0 * (1.0 - distance / self.initial_distance) if self.initial_distance > 0 else 0.0
            path_efficiency_reward = efficiency_weight * path_efficiency
        else:
            path_efficiency_reward = 0.0
        
        # Step penalty - increased to discourage long episodes and wandering
        # Penalty increases with episode length to encourage quick arrival
        step_penalty_reward = -self.step_penalty * (1.0 + self.step_count * 0.01)  # Progressive penalty
        
        # Smoothness penalty - reduced or removed when close to target
        # When close, allow more aggressive adjustments
        velocity_change = np.linalg.norm(velocity - self.prev_velocity)
        if distance < 50.0:
            # Reduced smoothness penalty when close to target
            smoothness_reward = -0.00005 * min(velocity_change**2, 100.0)  # Half penalty when close
        else:
            # Normal smoothness penalty when far
            smoothness_reward = -0.0001 * min(velocity_change**2, 100.0)
        
        total_reward = (success_funnel_reward + precision_reward + progress_reward + 
                       distance_guidance + boundary_penalty_reward + 
                       path_efficiency_reward + initial_direction_reward +
                       step_penalty_reward + smoothness_reward)

        return total_reward
    
    def render(self):
        """可视化环境（可选）"""
        pass

