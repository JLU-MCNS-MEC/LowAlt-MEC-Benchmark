"""
多无人机路径规划环境
支持多架无人机，每架无人机有对应的目标点
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MultiDronePathPlanningEnv(gym.Env):
    """
    Multi-drone path planning environment
    
    Observation space for each drone (7D):
    - Relative position: Δx, Δy (normalized)
    - Direction: direction_x, direction_y (unit vector)
    - Normalized velocity: vx/max_speed, vy/max_speed
    - Normalized distance: distance/world_size
    
    Action space for each drone:
    - Continuous velocity control (vx, vy) ∈ [-max_speed, max_speed]
    """
    
    def __init__(self, num_drones=3, world_size=1000, max_steps=60, max_speed=20.0, dt=1.0, 
                 arrival_threshold=20.0, arrival_reward=500.0,
                 progress_weight=200.0, step_penalty=0.01,
                 boundary_penalty_weight=5.0, smoothness_weight=0.1,
                 fixed_start_positions=None, fixed_target_positions=None):
        """
        Initialize multi-drone environment
        
        Args:
            num_drones: Number of drones (must be between 1 and 9, inclusive)
            world_size: World size (square area side length)
            max_steps: Maximum steps per episode
            max_speed: Maximum velocity magnitude
            dt: Time step
            arrival_threshold: Distance threshold for success
            arrival_reward: Reward for reaching target
            progress_weight: Weight for distance progress reward
            step_penalty: Penalty per step
            boundary_penalty_weight: Weight for boundary violation penalty
            smoothness_weight: Weight for smoothness penalty
            fixed_start_positions: List of fixed starting positions [[x1, y1], [x2, y2], ...]. If None, uses random positions
            fixed_target_positions: List of fixed target positions [[x1, y1], [x2, y2], ...]. If None, uses random positions
        """
        super().__init__()
        
        # Validate num_drones: must be between 1 and 9
        if not isinstance(num_drones, int) or num_drones < 1 or num_drones > 9:
            raise ValueError(f"num_drones must be an integer between 1 and 9 (inclusive), got {num_drones}")
        
        self.num_drones = num_drones
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
        self.use_fixed_positions = (
            fixed_start_positions is not None and 
            fixed_target_positions is not None and
            len(fixed_start_positions) == num_drones and
            len(fixed_target_positions) == num_drones
        )
        
        if self.use_fixed_positions:
            self.fixed_start_positions = np.array(fixed_start_positions, dtype=np.float32)
            self.fixed_target_positions = np.array(fixed_target_positions, dtype=np.float32)
            # Ensure fixed positions are within bounds
            self.fixed_start_positions = np.clip(self.fixed_start_positions, 0, world_size)
            self.fixed_target_positions = np.clip(self.fixed_target_positions, 0, world_size)
        else:
            self.fixed_start_positions = None
            self.fixed_target_positions = None
        
        # Action space: continuous velocity for each drone (vx, vy)
        # Shape: (num_drones, 2)
        self.action_space = spaces.Box(
            low=-max_speed,
            high=max_speed,
            shape=(num_drones, 2),
            dtype=np.float32
        )
        
        # Observation space: for each drone (7D)
        # Shape: (num_drones, 7)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_drones, 7),
            dtype=np.float32
        )
        
        # Store previous velocities for smoothness penalty
        self.prev_velocities = np.zeros((num_drones, 2), dtype=np.float32)
        
        self.reset()
    
    def set_fixed_positions(self, start_positions, target_positions):
        """
        Dynamically set fixed positions (for curriculum learning)
        
        Args:
            start_positions: List of [x, y] start positions for each drone
            target_positions: List of [x, y] target positions for each drone
        """
        if len(start_positions) != self.num_drones or len(target_positions) != self.num_drones:
            raise ValueError(f"Number of positions must match num_drones ({self.num_drones})")
        
        self.fixed_start_positions = np.array(start_positions, dtype=np.float32)
        self.fixed_target_positions = np.array(target_positions, dtype=np.float32)
        # Ensure positions are within bounds
        self.fixed_start_positions = np.clip(self.fixed_start_positions, 0, self.world_size)
        self.fixed_target_positions = np.clip(self.fixed_target_positions, 0, self.world_size)
        self.use_fixed_positions = True
    
    def set_random_positions(self):
        """Enable random positions for each episode"""
        self.use_fixed_positions = False
        self.fixed_start_positions = None
        self.fixed_target_positions = None
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        if self.use_fixed_positions:
            # Use fixed positions
            self.drone_positions = self.fixed_start_positions.copy()
            self.target_positions = self.fixed_target_positions.copy()
        else:
            # Use evenly distributed positions for each episode
            edge_margin = 50.0
            available_size = self.world_size - 2 * edge_margin
            
            # Calculate grid layout for even distribution
            # Try to create a roughly square grid
            grid_size = int(np.ceil(np.sqrt(self.num_drones)))
            
            self.drone_positions = np.zeros((self.num_drones, 2), dtype=np.float32)
            self.target_positions = np.zeros((self.num_drones, 2), dtype=np.float32)
            
            # Generate evenly distributed start positions
            for i in range(self.num_drones):
                # Use grid layout with some randomization
                row = i // grid_size
                col = i % grid_size
                
                # Calculate grid cell size
                cell_width = available_size / grid_size
                cell_height = available_size / grid_size
                
                # Add small random offset within cell for variety
                offset_range = min(cell_width, cell_height) * 0.3  # 30% of cell size
                offset_x = np.random.uniform(-offset_range, offset_range)
                offset_y = np.random.uniform(-offset_range, offset_range)
                
                start_x = edge_margin + col * cell_width + cell_width / 2 + offset_x
                start_y = edge_margin + row * cell_height + cell_height / 2 + offset_y
                
                # Ensure within bounds
                start_x = np.clip(start_x, edge_margin, self.world_size - edge_margin)
                start_y = np.clip(start_y, edge_margin, self.world_size - edge_margin)
                
                self.drone_positions[i] = np.array([start_x, start_y], dtype=np.float32)
            
            # Generate evenly distributed target positions (different from starts)
            min_distance = self.world_size * 0.05  # 5% of world size
            max_attempts = 20
            
            for i in range(self.num_drones):
                # Use opposite grid position or nearby grid for target
                # This ensures targets are also evenly distributed
                target_row = (i + grid_size // 2) % grid_size
                target_col = (i + 1) % grid_size
                
                cell_width = available_size / grid_size
                cell_height = available_size / grid_size
                
                # Add random offset
                offset_range = min(cell_width, cell_height) * 0.3
                offset_x = np.random.uniform(-offset_range, offset_range)
                offset_y = np.random.uniform(-offset_range, offset_range)
                
                target_x = edge_margin + target_col * cell_width + cell_width / 2 + offset_x
                target_y = edge_margin + target_row * cell_height + cell_height / 2 + offset_y
                
                # Ensure within bounds
                target_x = np.clip(target_x, edge_margin, self.world_size - edge_margin)
                target_y = np.clip(target_y, edge_margin, self.world_size - edge_margin)
                
                target_pos = np.array([target_x, target_y], dtype=np.float32)
                
                # Verify minimum distance
                if np.linalg.norm(target_pos - self.drone_positions[i]) >= min_distance:
                    self.target_positions[i] = target_pos
                else:
                    # If too close, place at minimum distance in opposite direction
                    direction = target_pos - self.drone_positions[i]
                    if np.linalg.norm(direction) < 1e-6:
                        # Random direction if same position
                        angle = np.random.uniform(0, 2 * np.pi)
                        direction = np.array([np.cos(angle), np.sin(angle)])
                    direction = direction / np.linalg.norm(direction)
                    self.target_positions[i] = self.drone_positions[i] + direction * min_distance
                    self.target_positions[i] = np.clip(self.target_positions[i], edge_margin, self.world_size - edge_margin)
        
        self.step_count = 0
        self.prev_distances = np.array([
            np.linalg.norm(self.target_positions[i] - self.drone_positions[i])
            for i in range(self.num_drones)
        ], dtype=np.float32)
        self.prev_velocities = np.zeros((self.num_drones, 2), dtype=np.float32)
        
        # Track initial positions and distances for each drone
        self.initial_positions = self.drone_positions.copy()
        self.initial_distances = self.prev_distances.copy()
        self.total_path_lengths = np.zeros(self.num_drones, dtype=np.float32)
        
        # Track which drones have reached their targets (frozen after success)
        self.drones_reached = np.zeros(self.num_drones, dtype=bool)
        
        # Track minimum distances reached for each drone (for "near success but failed" penalty)
        self.min_distances_reached = self.prev_distances.copy()
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def _get_observation(self, velocities=None):
        """
        Get observations for all drones
        
        Observation for each drone: [Δx, Δy, direction_x, direction_y, vx/max_speed, vy/max_speed, distance/world_size]
        """
        if velocities is None:
            velocities = self.prev_velocities
        
        observations = []
        max_possible_distance = self.world_size * np.sqrt(2)  # Diagonal distance
        
        for i in range(self.num_drones):
            # Relative position (normalized)
            dx = self.target_positions[i, 0] - self.drone_positions[i, 0]
            dy = self.target_positions[i, 1] - self.drone_positions[i, 1]
            distance = np.linalg.norm([dx, dy])
            
            delta_x = dx / self.world_size
            delta_y = dy / self.world_size
            
            # Unit direction vector
            if distance > 1e-6:
                direction_x = dx / distance
                direction_y = dy / distance
            else:
                direction_x = 0.0
                direction_y = 0.0
            
            # Normalized velocity
            normalized_vx = velocities[i, 0] / self.max_speed
            normalized_vy = velocities[i, 1] / self.max_speed
            
            # Normalized distance
            normalized_distance = distance / max_possible_distance
            
            observation = np.array([
                delta_x,
                delta_y,
                direction_x,
                direction_y,
                normalized_vx,
                normalized_vy,
                normalized_distance
            ], dtype=np.float32)
            
            observations.append(observation)
        
        return np.array(observations, dtype=np.float32)
    
    def step(self, actions):
        """
        Execute one step for all drones
        
        Args:
            actions: Array of shape (num_drones, 2) containing [vx, vy] for each drone
        
        Returns:
            observation: Array of shape (num_drones, 7)
            reward: Scalar (average reward across all drones)
            terminated: Boolean (True if all drones reached targets)
            truncated: Boolean (True if max_steps reached)
            info: Dictionary with additional information
        """
        # Process actions for each drone
        current_velocities = np.zeros((self.num_drones, 2), dtype=np.float32)
        all_rewards = []
        # Track if all active (non-frozen) drones reached
        active_drones_reached = True
        
        for i in range(self.num_drones):
            # If drone already reached target, freeze it (no position update, no reward calculation)
            if self.drones_reached[i]:
                # Keep previous velocity (zero) and previous position
                current_velocities[i] = np.array([0.0, 0.0], dtype=np.float32)
                # Give zero reward for already-reached drones
                all_rewards.append(0.0)
                continue
            
            # Calculate distance before processing action (for dynamic speed limit)
            distance_before = np.linalg.norm(self.target_positions[i] - self.drone_positions[i])
            
            # Update minimum distance reached (for "near success but failed" penalty)
            if distance_before < self.min_distances_reached[i]:
                self.min_distances_reached[i] = distance_before
            
            # Dynamic speed limit based on distance to target
            if distance_before < 20.0:
                effective_max_speed = self.max_speed * 0.3  # 6 m/s
            elif distance_before < 50.0:
                effective_max_speed = self.max_speed * 0.5  # 10 m/s
            else:
                effective_max_speed = self.max_speed
            
            # Scale action to actual velocity range
            vx, vy = actions[i, 0] * effective_max_speed, actions[i, 1] * effective_max_speed
            
            # Limit speed magnitude
            speed = np.sqrt(vx**2 + vy**2)
            if speed > effective_max_speed:
                vx = vx / speed * effective_max_speed
                vy = vy / speed * effective_max_speed
            
            current_velocities[i] = np.array([vx, vy], dtype=np.float32)
            
            # Update position
            velocity = np.array([vx, vy], dtype=np.float32)
            new_pos = self.drone_positions[i] + velocity * self.dt
            
            # Check boundary violation
            boundary_violation = False
            boundary_penalty = 0.0
            
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
            
            # Clamp position to world bounds
            new_pos[0] = np.clip(new_pos[0], 0, self.world_size)
            new_pos[1] = np.clip(new_pos[1], 0, self.world_size)
            
            # Calculate step distance for path length tracking
            step_distance = np.linalg.norm(new_pos - self.drone_positions[i])
            self.total_path_lengths[i] += step_distance
            
            self.drone_positions[i] = new_pos
            
            # Calculate distance
            distance = np.linalg.norm(self.target_positions[i] - self.drone_positions[i])
            
            # Check if target is near edge
            target_near_edge = (
                self.target_positions[i, 0] < 50.0 or self.target_positions[i, 0] > (self.world_size - 50.0) or
                self.target_positions[i, 1] < 50.0 or self.target_positions[i, 1] > (self.world_size - 50.0)
            )
            
            # Check termination conditions
            reached_target = distance < self.arrival_threshold
            if reached_target:
                # Mark drone as reached and freeze it
                self.drones_reached[i] = True
                reward = self.arrival_reward
            else:
                # This active drone hasn't reached yet
                active_drones_reached = False
                # Check if in success region (close to target) - disable all penalties
                in_success_region = distance < (self.arrival_threshold * 3.0)  # 60m radius
                reward = self._compute_reward(
                    i, distance, boundary_violation, boundary_penalty, 
                    current_velocities[i], target_near_edge, in_success_region
                )
            
            all_rewards.append(reward)
            
            # Update previous values
            self.prev_distances[i] = distance
            self.prev_velocities[i] = current_velocities[i].copy()
        
        self.step_count += 1
        
        # Get observation with current velocities
        observation = self._get_observation(current_velocities)
        
        # Overall termination: all drones reached targets
        all_reached = np.all(self.drones_reached)
        terminated = all_reached
        truncated = self.step_count >= self.max_steps
        
        # Average reward across all drones
        total_reward = np.mean(all_rewards)
        
        # Add episode-end rewards/penalties
        if terminated or truncated:
            # 1. Success ratio reward (even if not all succeeded)
            num_reached = np.sum(self.drones_reached)
            success_ratio = num_reached / self.num_drones
            # Reward proportional to success ratio (even partial success gets reward)
            success_ratio_reward = 100.0 * success_ratio  # Max 100.0 if all succeed
            total_reward += success_ratio_reward
            
            # 2. Penalty for "near success but failed" episodes
            if not all_reached:
                # Check if any drone got very close but failed
                near_success_threshold = self.arrival_threshold * 2.0  # 40m
                near_success_penalty = 0.0
                for i in range(self.num_drones):
                    if not self.drones_reached[i]:
                        # If drone got close but didn't succeed, add penalty
                        if self.min_distances_reached[i] < near_success_threshold:
                            # Penalty increases with how close it got
                            closeness = (near_success_threshold - self.min_distances_reached[i]) / near_success_threshold
                            near_success_penalty += -50.0 * closeness  # Max -50 per drone
                total_reward += near_success_penalty
        
        # Calculate success info
        num_reached = np.sum(self.drones_reached)
        final_distances = [np.linalg.norm(self.target_positions[i] - self.drone_positions[i]) 
                          for i in range(self.num_drones)]
        
        info = {
            'num_reached': num_reached,
            'all_reached': all_reached,
            'individual_rewards': all_rewards,
            'individual_distances': final_distances,
            'success_ratio': num_reached / self.num_drones if terminated or truncated else 0.0
        }
        
        return observation, total_reward, terminated, truncated, info
    
    def _compute_reward(self, drone_idx, distance, boundary_violation=False, boundary_penalty=0.0, 
                       velocity=None, target_near_edge=False, in_success_region=False):
        """
        Compute reward for a single drone
        
        Args:
            in_success_region: If True, disable all penalties (drone is close to target)
        """
        # Calculate progress
        progress = self.prev_distances[drone_idx] - distance
        
        # 3. Progress reward decay in late episode
        # Decay factor: 1.0 at start, 0.3 at max_steps
        progress_decay = 1.0 - 0.7 * (self.step_count / self.max_steps)
        progress_decay = max(progress_decay, 0.3)  # Minimum 30% of progress reward
        
        # Success funnel reward
        success_radius = self.arrival_threshold * 7.5  # 150m
        w_funnel = 15
        if distance < success_radius:
            base_funnel_reward = w_funnel * ((success_radius - distance) / success_radius) ** 2
            if progress < -1.0:
                success_funnel_reward = base_funnel_reward * 0.3
            else:
                success_funnel_reward = base_funnel_reward
        else:
            success_funnel_reward = 0.0

        # Close-range precision navigation reward
        precision_radius = 50.0
        if distance < precision_radius:
            base_precision_reward = 20.0 * ((precision_radius - distance) / precision_radius) ** 1.5
            if progress < -0.5:
                precision_reward = base_precision_reward * 0.3
            else:
                precision_reward = base_precision_reward
        else:
            precision_reward = 0.0

        # Distance progress reward (with late-episode decay)
        if distance < 50.0:
            close_progress_weight = self.progress_weight * 2.0
            progress_reward = close_progress_weight * (progress / 100.0) * progress_decay
        elif distance > 800.0:
            far_progress_weight = self.progress_weight * 1.5
            progress_reward = far_progress_weight * (progress / 100.0) * progress_decay
        else:
            progress_reward = self.progress_weight * (progress / 100.0) * progress_decay
        
        # Distance guidance reward
        distance_guidance = -0.5 * (distance / self.world_size)
        
        # Initial direction alignment reward
        initial_direction_reward = 0.0
        if self.step_count <= 5 and velocity is not None:
            dx = self.target_positions[drone_idx, 0] - self.drone_positions[drone_idx, 0]
            dy = self.target_positions[drone_idx, 1] - self.drone_positions[drone_idx, 1]
            target_direction = np.array([dx, dy])
            movement_direction = np.array([velocity[0], velocity[1]])
            
            if np.linalg.norm(movement_direction) > 0.1 and np.linalg.norm(target_direction) > 1e-6:
                alignment = np.dot(movement_direction, target_direction) / (
                    np.linalg.norm(movement_direction) * np.linalg.norm(target_direction)
                )
                if alignment > 0.3:
                    initial_direction_reward = 5.0 * alignment
                elif alignment < -0.3:
                    initial_direction_reward = -3.0 * abs(alignment)
        
        # 5. Disable all penalties in success region
        if in_success_region:
            # In success region: only keep positive rewards, disable all penalties
            boundary_penalty_reward = 0.0
            step_penalty_reward = 0.0
            smoothness_reward = 0.0
            distance_guidance = 0.0  # Also disable distance guidance penalty
        else:
            # Boundary violation penalty
            if boundary_violation:
                if target_near_edge:
                    boundary_penalty_reward = -self.boundary_penalty_weight * boundary_penalty * 0.5
                else:
                    boundary_penalty_reward = -self.boundary_penalty_weight * boundary_penalty
            else:
                boundary_penalty_reward = 0.0
            
            # Step penalty
            step_penalty_reward = -self.step_penalty * (1.0 + self.step_count * 0.01)
            
            # Smoothness penalty
            velocity_change = np.linalg.norm(velocity - self.prev_velocities[drone_idx])
            if distance < 50.0:
                smoothness_reward = -0.00005 * min(velocity_change**2, 100.0)
            else:
                smoothness_reward = -0.0001 * min(velocity_change**2, 100.0)
        
        # Path efficiency reward (always positive, keep it)
        if self.total_path_lengths[drone_idx] > 0:
            path_efficiency = self.initial_distances[drone_idx] / max(
                self.total_path_lengths[drone_idx], self.initial_distances[drone_idx]
            )
            efficiency_weight = 2.0 * (1.0 - distance / self.initial_distances[drone_idx]) if self.initial_distances[drone_idx] > 0 else 0.0
            path_efficiency_reward = efficiency_weight * path_efficiency
        else:
            path_efficiency_reward = 0.0
        
        total_reward = (success_funnel_reward + precision_reward + progress_reward + 
                       distance_guidance + boundary_penalty_reward + 
                       path_efficiency_reward + initial_direction_reward +
                       step_penalty_reward + smoothness_reward)

        return total_reward
    
    def render(self):
        """可视化环境（可选）"""
        pass

