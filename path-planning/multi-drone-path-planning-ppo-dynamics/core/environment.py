"""
多无人机路径规划环境（使用动力学模型）
支持多架无人机，每架无人机有对应的目标点
使用动力学模型控制：推力、滚转力矩、俯仰力矩、偏航力矩
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from math import cos, sin


class MultiDronePathPlanningEnv(gym.Env):
    """
    Multi-drone path planning environment with dynamics model
    
    Observation space for each drone (17D):
    - Relative position: Δx, Δy (normalized)
    - Direction: direction_x, direction_y (unit vector)
    - Normalized velocity: vx/max_speed, vy/max_speed
    - Normalized distance: distance/world_size
    - Attitude: roll, pitch, yaw (normalized)
    - Angular velocity: roll_vel, pitch_vel, yaw_vel (normalized)
    - Previous action: prev_thrust, prev_roll_torque, prev_pitch_torque, prev_yaw_torque
    
    Action space for each drone:
    - Continuous dynamics control: [thrust, roll_torque, pitch_torque, yaw_torque]
    """
    
    def __init__(self, num_drones=3, world_size=1000, max_steps=400, max_speed=20.0, dt=0.1,  # Reduced from 600 to 400 for better performance 
                 arrival_threshold=20.0, arrival_reward=500.0,
                 progress_weight=300.0,  # Increased from 200.0 to 300.0 for stronger progress signal
                 step_penalty=0.01,
                 boundary_penalty_weight=5.0, smoothness_weight=0.1,
                 fixed_start_positions=None, fixed_target_positions=None,
                 # Dynamics parameters
                 m=0.2,  # Mass (kg)
                 Ixx=1.0,  # Moment of inertia around x-axis
                 Iyy=1.0,  # Moment of inertia around y-axis
                 Izz=1.0,  # Moment of inertia around z-axis
                 g=9.81,  # Gravity
                 max_thrust=5.0,  # Maximum thrust (N)
                 max_torque=1.0):  # Maximum torque (N·m) - Reduced from 2.0 for smoother control
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
            m: Mass of the drone (kg)
            Ixx, Iyy, Izz: Moments of inertia (kg·m²)
            g: Gravity acceleration (m/s²)
            max_thrust: Maximum thrust (N)
            max_torque: Maximum torque (N·m)
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
        
        # Dynamics parameters
        self.m = m
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.g = g
        self.max_thrust = max_thrust
        self.max_torque = max_torque
        
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
        
        # Action space: [thrust, roll_torque, pitch_torque, yaw_torque] for each drone
        # Actions are expected in [-1, 1] range, will be scaled to [0, max_thrust] and [-max_torque, max_torque]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(num_drones, 4),  # (num_drones, 4) for dynamics control
            dtype=np.float32
        )
        
        # Observation space: [Δx, Δy, direction_x, direction_y, vx/max_speed, vy/max_speed, 
        #                     distance/world_size, roll, pitch, yaw, roll_vel, pitch_vel, yaw_vel,
        #                     prev_thrust, prev_roll_torque, prev_pitch_torque, prev_yaw_torque] for each drone
        # Normalized attitude and angular velocities, plus previous action
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_drones, 17),  # (num_drones, 17) for each drone
            dtype=np.float32
        )
        
        # Store previous velocity, action, and attitude for smoothness penalty (for each drone)
        self.prev_velocities = np.zeros((num_drones, 2), dtype=np.float32)
        self.prev_actions = np.zeros((num_drones, 4), dtype=np.float32)  # Added yaw_torque
        self.prev_attitudes = np.zeros((num_drones, 3), dtype=np.float32)
        
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
        """Reset environment - uses fixed positions if set, otherwise evenly distributed positions"""
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
            grid_size = int(np.ceil(np.sqrt(self.num_drones)))
            
            self.drone_positions = np.zeros((self.num_drones, 2), dtype=np.float32)
            self.target_positions = np.zeros((self.num_drones, 2), dtype=np.float32)
            
            # Generate evenly distributed start positions
            for i in range(self.num_drones):
                row = i // grid_size
                col = i % grid_size
                
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
        self.prev_actions = np.zeros((self.num_drones, 4), dtype=np.float32)
        self.prev_attitudes = np.zeros((self.num_drones, 3), dtype=np.float32)
        
        # Initialize dynamics state for each drone
        # Position: x, y (z is fixed at 0 for 2D)
        self.drone_velocities = np.zeros((self.num_drones, 2), dtype=np.float32)  # vx, vy for each drone
        self.drone_attitudes = np.zeros((self.num_drones, 3), dtype=np.float32)  # roll, pitch, yaw for each drone
        self.drone_angular_velocities = np.zeros((self.num_drones, 3), dtype=np.float32)  # roll_vel, pitch_vel, yaw_vel for each drone
        
        # Track initial positions and ideal paths for path efficiency reward
        self.initial_positions = self.drone_positions.copy()
        self.initial_distances = self.prev_distances.copy()
        self.total_path_lengths = np.zeros(self.num_drones, dtype=np.float32)  # Track total path length for each drone
        
        # Track which drones have reached their targets (frozen after success)
        self.drones_reached = np.zeros(self.num_drones, dtype=bool)
        
        # Track minimum distances reached for each drone (for "near success but failed" penalty)
        self.min_distances_reached = self.prev_distances.copy()
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def _get_observation(self, velocities=None):
        """
        Get observations for all drones with dynamics state
        
        Observation for each drone: [Δx, Δy, direction_x, direction_y, vx/max_speed, vy/max_speed, 
                     distance/world_size, roll, pitch, yaw, roll_vel, pitch_vel, yaw_vel,
                     prev_thrust, prev_roll_torque, prev_pitch_torque, prev_yaw_torque]
        """
        if velocities is None:
            velocities = self.drone_velocities
        
        observations = []
        max_possible_distance = self.world_size * np.sqrt(2)  # Diagonal distance
        max_attitude_angle = np.pi / 4  # 45 degrees
        max_angular_vel = 3.0  # rad/s
        
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
            
            # Normalized attitude
            normalized_roll = np.clip(self.drone_attitudes[i, 0] / max_attitude_angle, -2.0, 2.0)
            normalized_pitch = np.clip(self.drone_attitudes[i, 1] / max_attitude_angle, -2.0, 2.0)
            normalized_yaw = self.drone_attitudes[i, 2] / np.pi
            
            # Normalized angular velocities
            normalized_roll_vel = np.clip(self.drone_angular_velocities[i, 0] / max_angular_vel, -2.0, 2.0)
            normalized_pitch_vel = np.clip(self.drone_angular_velocities[i, 1] / max_angular_vel, -2.0, 2.0)
            normalized_yaw_vel = np.clip(self.drone_angular_velocities[i, 2] / max_angular_vel, -2.0, 2.0)
            
            # Normalized previous action (for action history)
            normalized_prev_thrust = self.prev_actions[i, 0]  # Already in [-1, 1] range
            normalized_prev_roll_torque = self.prev_actions[i, 1]  # Already in [-1, 1] range
            normalized_prev_pitch_torque = self.prev_actions[i, 2]  # Already in [-1, 1] range
            normalized_prev_yaw_torque = self.prev_actions[i, 3]  # Already in [-1, 1] range
            
            observation = np.array([
                delta_x,
                delta_y,
                direction_x,
                direction_y,
                normalized_vx,
                normalized_vy,
                normalized_distance,
                normalized_roll,
                normalized_pitch,
                normalized_yaw,
                normalized_roll_vel,
                normalized_pitch_vel,
                normalized_yaw_vel,
                normalized_prev_thrust,
                normalized_prev_roll_torque,
                normalized_prev_pitch_torque,
                normalized_prev_yaw_torque
            ], dtype=np.float32)
            
            observations.append(observation)
        
        return np.array(observations, dtype=np.float32)
    
    def _rotation_matrix(self, roll, pitch, yaw):
        """
        Calculate ZYX rotation matrix for 2D projection
        Returns 2x2 rotation matrix for x-y plane
        """
        # For 2D movement, we mainly use yaw (rotation around z-axis)
        # Roll and pitch affect the thrust direction in 3D, but for 2D we simplify
        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)
        
        # 2D rotation matrix (yaw only for horizontal plane)
        R_2d = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])
        return R_2d
    
    def step(self, actions):
        """
        Execute one step for all drones using dynamics model
        
        Args:
            actions: Array of shape (num_drones, 4) containing [thrust, roll_torque, pitch_torque, yaw_torque] for each drone
        
        Returns:
            observation: Array of shape (num_drones, 17)
            reward: Scalar (average reward across all drones)
            terminated: Boolean (True if all drones reached targets)
            truncated: Boolean (True if max_steps reached)
            info: Dictionary with additional information
        """
        # Process actions for each drone
        all_rewards = []
        active_drones_reached = True
        
        for i in range(self.num_drones):
            # If drone already reached target, freeze it (no position update, no reward calculation)
            if self.drones_reached[i]:
                # Keep previous velocity (zero) and previous position
                # Give zero reward for already-reached drones
                all_rewards.append(0.0)
                continue
            
            # action is [thrust, roll_torque, pitch_torque, yaw_torque] in [-1, 1] range
            # Scale to actual control inputs
            thrust = (actions[i, 0] + 1.0) / 2.0 * self.max_thrust  # Map [-1, 1] to [0, max_thrust]
            roll_torque = actions[i, 1] * self.max_torque  # Map [-1, 1] to [-max_torque, max_torque]
            pitch_torque = actions[i, 2] * self.max_torque  # Map [-1, 1] to [-max_torque, max_torque]
            yaw_torque = actions[i, 3] * self.max_torque  # Map [-1, 1] to [-max_torque, max_torque]
            
            # Store action for smoothness penalty (keep in [-1, 1] range for observation)
            current_action = np.array([actions[i, 0], actions[i, 1], actions[i, 2], actions[i, 3]], dtype=np.float32)
            
            # Update angular velocities: ω += τ * dt / I
            self.drone_angular_velocities[i, 0] += roll_torque * self.dt / self.Ixx  # roll_vel
            self.drone_angular_velocities[i, 1] += pitch_torque * self.dt / self.Iyy  # pitch_vel
            self.drone_angular_velocities[i, 2] += yaw_torque * self.dt / self.Izz  # yaw_vel
            # Add damping to angular velocities for stability
            self.drone_angular_velocities[i, 0] *= 0.95  # Damping factor
            self.drone_angular_velocities[i, 1] *= 0.95
            self.drone_angular_velocities[i, 2] *= 0.95
            
            # Update attitude: θ += ω * dt
            self.drone_attitudes[i, 0] += self.drone_angular_velocities[i, 0] * self.dt  # roll
            self.drone_attitudes[i, 1] += self.drone_angular_velocities[i, 1] * self.dt  # pitch
            self.drone_attitudes[i, 2] += self.drone_angular_velocities[i, 2] * self.dt  # yaw
            # Normalize yaw to [-π, π]
            while self.drone_attitudes[i, 2] > np.pi:
                self.drone_attitudes[i, 2] -= 2 * np.pi
            while self.drone_attitudes[i, 2] < -np.pi:
                self.drone_attitudes[i, 2] += 2 * np.pi
            
            # Calculate acceleration from thrust using rotation matrix
            roll = self.drone_attitudes[i, 0]
            pitch = self.drone_attitudes[i, 1]
            yaw = self.drone_attitudes[i, 2]
            
            # Calculate 3D rotation matrix (ZYX Euler angles)
            cos_r, sin_r = cos(roll), sin(roll)
            cos_p, sin_p = cos(pitch), sin(pitch)
            cos_y, sin_y = cos(yaw), sin(yaw)
            
            R_x = np.array([
                [cos_y * cos_p, -sin_y * cos_r + cos_y * sin_p * sin_r, sin_y * sin_r + cos_y * sin_p * cos_r],
                [sin_y * cos_p, cos_y * cos_r + sin_y * sin_p * sin_r, -cos_y * sin_r + sin_y * sin_p * cos_r],
                [-sin_p, cos_p * sin_r, cos_p * cos_r]
            ])
            
            # Thrust vector in body frame (pointing up in z-direction)
            thrust_body = np.array([0.0, 0.0, thrust])
            
            # Rotate to world frame
            thrust_world = R_x @ thrust_body
            
            # Acceleration = thrust_world / m (only horizontal components)
            acc = np.array([thrust_world[0] / self.m, thrust_world[1] / self.m])
            
            # Update velocity: v += a * dt
            self.drone_velocities[i] += acc * self.dt
            
            # Limit velocity magnitude (safety check)
            speed = np.linalg.norm(self.drone_velocities[i])
            if speed > self.max_speed:
                self.drone_velocities[i] = self.drone_velocities[i] / speed * self.max_speed
            
            # Update position: pos += v * dt
            new_pos = self.drone_positions[i] + self.drone_velocities[i] * self.dt
            
            # Check boundary violation and calculate penetration depth
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
            
            # Calculate distance (before updating prev_distance)
            distance = np.linalg.norm(self.target_positions[i] - self.drone_positions[i])
            
            # Update minimum distance reached
            if distance < self.min_distances_reached[i]:
                self.min_distances_reached[i] = distance
            
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
                    self.drone_velocities[i], target_near_edge, current_action, in_success_region
                )
            
            all_rewards.append(reward)
            
            # Update previous values AFTER reward calculation
            self.prev_distances[i] = distance
            self.prev_velocities[i] = self.drone_velocities[i].copy()
            self.prev_actions[i] = current_action.copy()
            self.prev_attitudes[i] = self.drone_attitudes[i].copy()
        
        self.step_count += 1
        
        # Get observation with current velocities
        observation = self._get_observation(self.drone_velocities)
        
        # Overall termination: all drones reached targets
        all_reached = np.all(self.drones_reached)
        terminated = all_reached
        truncated = self.step_count >= self.max_steps
        
        # Average reward across all drones
        total_reward = np.mean(all_rewards)
        
        # Add episode-end rewards/penalties
        if terminated or truncated:
            # Success ratio reward (even if not all succeeded)
            # Use nonlinear function to strongly encourage all drones to reach
            num_reached = np.sum(self.drones_reached)
            success_ratio = num_reached / self.num_drones
            # Nonlinear: square function to encourage all drones reaching
            # All reach: 200.0, 2/3 reach: 200.0 * (2/3)² ≈ 88.9, 1/3 reach: 200.0 * (1/3)² ≈ 22.2
            success_ratio_reward = 200.0 * (success_ratio ** 2)  # Increased from 100.0 and use square
            total_reward += success_ratio_reward
            
            # Penalty for "near success but failed" episodes
            if not all_reached:
                near_success_threshold = self.arrival_threshold * 2.0  # 40m
                near_success_penalty = 0.0
                for i in range(self.num_drones):
                    if not self.drones_reached[i]:
                        if self.min_distances_reached[i] < near_success_threshold:
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
                       velocity=None, target_near_edge=False, action=None, in_success_region=False):
        """
        Compute reward for a single drone with refactored components
        
        Args:
            drone_idx: Index of the drone
            in_success_region: If True, disable all penalties (drone is close to target)
        
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
        10. Attitude stability reward (encourage stable flight)
        11. Attitude angle limit penalty (penalize excessive tilt)
        12. Attitude smoothness reward (penalize rapid attitude changes)
        """
        # Calculate progress first (needed for funnel and precision rewards)
        progress = self.prev_distances[drone_idx] - distance
        
        # Progress reward decay in late episode
        progress_decay = 1.0 - 0.7 * (self.step_count / self.max_steps)
        progress_decay = max(progress_decay, 0.3)  # Minimum 30% of progress reward
        
        # Success funnel reward - simplified for easier learning
        # Give reward when close to target, with reduced penalty for no progress
        success_radius = self.arrival_threshold * 7.5  # 7.5x arrival threshold (150m) for earlier guidance
        w_funnel = 15  # Increased weight for better guidance
        if distance < success_radius:
            base_funnel_reward = w_funnel * ((success_radius - distance) / success_radius) ** 2
            # Enhanced: reduce reward based on progress to prevent wandering
            if progress < -1.0:  # Moving away significantly
                success_funnel_reward = base_funnel_reward * 0.1  # Strong penalty for moving away
            elif progress < 0.0:  # No progress or small negative
                success_funnel_reward = base_funnel_reward * 0.3  # Reduced reward for no progress
            elif progress < 1.0:  # Small progress
                success_funnel_reward = base_funnel_reward * 0.7  # Partial reward for small progress
            else:
                # Good progress: full funnel reward
                success_funnel_reward = base_funnel_reward
        else:
            success_funnel_reward = 0.0

        # Close-range precision navigation reward (when distance < 50m)
        # Strong reward to encourage precise navigation when close to target
        # Enhanced: reduce reward based on progress to prevent wandering
        precision_radius = 50.0  # 50m threshold
        if distance < precision_radius:
            base_precision_reward = 20.0 * ((precision_radius - distance) / precision_radius) ** 1.5
            # Enhanced: reduce reward based on progress
            if progress < -0.5:  # Moving away significantly
                precision_reward = base_precision_reward * 0.1  # Strong penalty for moving away
            elif progress < 0.0:  # No progress
                precision_reward = base_precision_reward * 0.3  # Reduced reward for no progress
            elif progress < 0.5:  # Small progress
                precision_reward = base_precision_reward * 0.7  # Partial reward for small progress
            else:
                # Good progress: full precision reward
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
                    initial_direction_reward = 10.0 * alignment  # Increased from 5.0 to 10.0
                elif alignment < -0.3:
                    initial_direction_reward = -5.0 * abs(alignment)  # Increased from -3.0 to -5.0
        
        # Disable all penalties in success region
        if in_success_region:
            boundary_penalty_reward = 0.0
            step_penalty_reward = 0.0
            smoothness_reward = 0.0
            distance_guidance = 0.0
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
                smoothness_reward = -0.00005 * min(velocity_change**2, 100.0) - 0.0001 * min(np.linalg.norm(action - self.prev_actions[drone_idx])**2, 10.0)
            else:
                smoothness_reward = -0.0001 * min(velocity_change**2, 100.0) - 0.0002 * min(np.linalg.norm(action - self.prev_actions[drone_idx])**2, 10.0)
        
        # Path efficiency reward (always positive, keep it)
        if self.total_path_lengths[drone_idx] > 0:
            path_efficiency = self.initial_distances[drone_idx] / max(
                self.total_path_lengths[drone_idx], self.initial_distances[drone_idx]
            )
            efficiency_weight = 2.0 * (1.0 - distance / self.initial_distances[drone_idx]) if self.initial_distances[drone_idx] > 0 else 0.0
            path_efficiency_reward = efficiency_weight * path_efficiency
        else:
            path_efficiency_reward = 0.0
        
        # Attitude stability reward
        roll = self.drone_attitudes[drone_idx, 0]
        pitch = self.drone_attitudes[drone_idx, 1]
        attitude_stability_reward = -1.0 * (abs(roll) + abs(pitch)) / (np.pi / 4)
        
        # Attitude angle limit penalty
        max_safe_angle = np.pi / 6  # 30 degrees
        attitude_angle_penalty = 0.0
        if abs(roll) > max_safe_angle:
            attitude_angle_penalty -= 5.0 * (abs(roll) - max_safe_angle) / (np.pi / 6)
        if abs(pitch) > max_safe_angle:
            attitude_angle_penalty -= 5.0 * (abs(pitch) - max_safe_angle) / (np.pi / 6)
        
        # Attitude smoothness reward
        attitude_change = np.linalg.norm(self.drone_attitudes[drone_idx] - self.prev_attitudes[drone_idx])
        attitude_smoothness_reward = -0.5 * min(attitude_change**2, (np.pi / 4)**2) / (np.pi / 4)**2
        
        total_reward = (success_funnel_reward + precision_reward + progress_reward + 
                       distance_guidance + boundary_penalty_reward + 
                       path_efficiency_reward + initial_direction_reward +
                       step_penalty_reward + smoothness_reward +
                       attitude_stability_reward + attitude_angle_penalty + 
                       attitude_smoothness_reward)

        return total_reward
    
    def render(self):
        """可视化环境（可选）"""
        pass

