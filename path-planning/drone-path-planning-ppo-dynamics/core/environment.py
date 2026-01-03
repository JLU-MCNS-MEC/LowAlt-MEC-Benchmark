"""
无人机路径规划环境（使用动力学模型）
随机生成目标点，无人机需要规划路径到达目标点
使用动力学模型控制：推力、滚转力矩、俯仰力矩
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from math import cos, sin


class DronePathPlanningEnv(gym.Env):
    """
    Drone path planning environment with dynamics model
    
    Observation space:
    - Relative position: Δx, Δy (normalized)
    - Direction: direction_x, direction_y (unit vector)
    - Normalized velocity: vx/max_speed, vy/max_speed
    - Normalized distance: distance/world_size
    - Attitude: roll, pitch, yaw (normalized)
    - Angular velocity: roll_vel, pitch_vel, yaw_vel (normalized)
    
    Action space:
    - Continuous dynamics control: [thrust, roll_torque, pitch_torque]
    """
    
    def __init__(self, world_size=1000, max_steps=600, max_speed=20.0, dt=0.1, 
                 arrival_threshold=20.0, arrival_reward=500.0,
                 progress_weight=200.0,
                 step_penalty=0.01,
                 boundary_penalty_weight=5.0, smoothness_weight=0.1,
                 fixed_start_pos=None, fixed_target_pos=None,
                 # Dynamics parameters
                 m=0.2,  # Mass (kg)
                 Ixx=1.0,  # Moment of inertia around x-axis
                 Iyy=1.0,  # Moment of inertia around y-axis
                 Izz=1.0,  # Moment of inertia around z-axis
                 g=9.81,  # Gravity
                 max_thrust=5.0,  # Maximum thrust (N)
                 max_torque=1.0):  # Maximum torque (N·m) - Reduced from 2.0 for smoother control
        """
        Initialize environment
        
        Args:
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
            fixed_start_pos: Fixed starting position [x, y]. If None, uses random position
            fixed_target_pos: Fixed target position [x, y]. If None, uses random position
            m: Mass of the drone (kg)
            Ixx, Iyy, Izz: Moments of inertia (kg·m²)
            g: Gravity acceleration (m/s²)
            max_thrust: Maximum thrust (N)
            max_torque: Maximum torque (N·m)
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
        
        # Dynamics parameters
        self.m = m
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.g = g
        self.max_thrust = max_thrust
        self.max_torque = max_torque
        
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
        
        # Action space: [thrust, roll_torque, pitch_torque, yaw_torque]
        # Actions are expected in [-1, 1] range, will be scaled to [0, max_thrust] and [-max_torque, max_torque]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),  # Added yaw_torque
            dtype=np.float32
        )
        
        # Observation space: [Δx, Δy, direction_x, direction_y, vx/max_speed, vy/max_speed, 
        #                     distance/world_size, roll, pitch, yaw, roll_vel, pitch_vel, yaw_vel,
        #                     prev_thrust, prev_roll_torque, prev_pitch_torque, prev_yaw_torque]
        # Normalized attitude and angular velocities, plus previous action
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(17,),  # 13 original + 4 previous actions
            dtype=np.float32
        )
        
        # Store previous velocity, action, and attitude for smoothness penalty
        self.prev_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Added yaw_torque
        self.prev_attitude = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
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
        self.prev_action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Added yaw_torque
        self.prev_attitude = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Initialize dynamics state
        # Position: x, y (z is fixed at 0 for 2D)
        self.drone_vel = np.array([0.0, 0.0], dtype=np.float32)  # vx, vy
        self.drone_attitude = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # roll, pitch, yaw
        self.drone_angular_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # roll_vel, pitch_vel, yaw_vel
        
        # Track initial position and ideal path for path efficiency reward
        self.initial_pos = self.drone_pos.copy()
        self.initial_distance = self.prev_distance
        self.total_path_length = 0.0  # Track total path length
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def _get_observation(self, velocity=None):
        """
        Get agent-centric observation with dynamics state
        
        Observation: [Δx, Δy, direction_x, direction_y, vx/max_speed, vy/max_speed, 
                     distance/world_size, roll, pitch, yaw, roll_vel, pitch_vel, yaw_vel]
        """
        # Relative position (normalized)
        dx = self.target_pos[0] - self.drone_pos[0]
        dy = self.target_pos[1] - self.drone_pos[1]
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
        if velocity is None:
            velocity = self.drone_vel
        normalized_vx = velocity[0] / self.max_speed
        normalized_vy = velocity[1] / self.max_speed
        
        # Normalized distance
        max_possible_distance = self.world_size * np.sqrt(2)
        normalized_distance = distance / max_possible_distance
        
        # Normalized attitude (use π/4 as typical range instead of π for better scaling)
        # This assumes roll/pitch typically stay within ±45 degrees
        max_attitude_angle = np.pi / 4  # 45 degrees
        normalized_roll = np.clip(self.drone_attitude[0] / max_attitude_angle, -2.0, 2.0)
        normalized_pitch = np.clip(self.drone_attitude[1] / max_attitude_angle, -2.0, 2.0)
        # Yaw can be in full range [-π, π]
        normalized_yaw = self.drone_attitude[2] / np.pi
        
        # Normalized angular velocities (normalize by max expected angular velocity)
        max_angular_vel = 3.0  # rad/s (reduced from 5.0 for better scaling)
        normalized_roll_vel = np.clip(self.drone_angular_vel[0] / max_angular_vel, -2.0, 2.0)
        normalized_pitch_vel = np.clip(self.drone_angular_vel[1] / max_angular_vel, -2.0, 2.0)
        normalized_yaw_vel = np.clip(self.drone_angular_vel[2] / max_angular_vel, -2.0, 2.0)
        
        # Normalized previous action (for action history)
        normalized_prev_thrust = self.prev_action[0]  # Already in [-1, 1] range
        normalized_prev_roll_torque = self.prev_action[1]  # Already in [-1, 1] range
        normalized_prev_pitch_torque = self.prev_action[2]  # Already in [-1, 1] range
        normalized_prev_yaw_torque = self.prev_action[3]  # Already in [-1, 1] range
        
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
        
        return observation
    
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
    
    def step(self, action):
        """Execute one step using dynamics model"""
        # action is [thrust, roll_torque, pitch_torque, yaw_torque] in [-1, 1] range
        # Scale to actual control inputs
        thrust = (action[0] + 1.0) / 2.0 * self.max_thrust  # Map [-1, 1] to [0, max_thrust]
        roll_torque = action[1] * self.max_torque  # Map [-1, 1] to [-max_torque, max_torque]
        pitch_torque = action[2] * self.max_torque  # Map [-1, 1] to [-max_torque, max_torque]
        yaw_torque = action[3] * self.max_torque  # Map [-1, 1] to [-max_torque, max_torque]
        
        # Store action for smoothness penalty (keep in [-1, 1] range for observation)
        current_action = np.array([action[0], action[1], action[2], action[3]], dtype=np.float32)
        
        # Update angular velocities: ω += τ * dt / I
        self.drone_angular_vel[0] += roll_torque * self.dt / self.Ixx  # roll_vel
        self.drone_angular_vel[1] += pitch_torque * self.dt / self.Iyy  # pitch_vel
        self.drone_angular_vel[2] += yaw_torque * self.dt / self.Izz  # yaw_vel
        # Add damping to angular velocities for stability
        self.drone_angular_vel[0] *= 0.95  # Damping factor
        self.drone_angular_vel[1] *= 0.95
        self.drone_angular_vel[2] *= 0.95
        
        # Update attitude: θ += ω * dt
        self.drone_attitude[0] += self.drone_angular_vel[0] * self.dt  # roll
        self.drone_attitude[1] += self.drone_angular_vel[1] * self.dt  # pitch
        self.drone_attitude[2] += self.drone_angular_vel[2] * self.dt  # yaw
        # Normalize yaw to [-π, π]
        while self.drone_attitude[2] > np.pi:
            self.drone_attitude[2] -= 2 * np.pi
        while self.drone_attitude[2] < -np.pi:
            self.drone_attitude[2] += 2 * np.pi
        
        # Calculate acceleration from thrust using rotation matrix
        # Similar to 3D model: acc = (R * [0, 0, thrust] - [0, 0, mg]) / m
        # For 2D, we project to x-y plane
        roll = self.drone_attitude[0]
        pitch = self.drone_attitude[1]
        yaw = self.drone_attitude[2]
        
        # Calculate 3D rotation matrix (ZYX Euler angles)
        # R = R_z(yaw) * R_y(pitch) * R_x(roll)
        cos_r, sin_r = cos(roll), sin(roll)
        cos_p, sin_p = cos(pitch), sin(pitch)
        cos_y, sin_y = cos(yaw), sin(yaw)
        
        # Rotation matrix elements for x-y projection
        # The thrust vector in body frame is [0, 0, thrust]
        # After rotation, we get horizontal components
        # R[0,2] and R[1,2] give the x and y components of the z-axis after rotation
        R_x = np.array([
            [cos_y * cos_p, -sin_y * cos_r + cos_y * sin_p * sin_r, sin_y * sin_r + cos_y * sin_p * cos_r],
            [sin_y * cos_p, cos_y * cos_r + sin_y * sin_p * sin_r, -cos_y * sin_r + sin_y * sin_p * cos_r],
            [-sin_p, cos_p * sin_r, cos_p * cos_r]
        ])
        
        # Thrust vector in body frame (pointing up in z-direction)
        thrust_body = np.array([0.0, 0.0, thrust])
        
        # Rotate to world frame
        thrust_world = R_x @ thrust_body
        
        # Gravity acts downward (in -z direction), but for 2D we assume it's balanced
        # So acceleration = thrust_world / m (only horizontal components)
        acc = np.array([thrust_world[0] / self.m, thrust_world[1] / self.m])
        
        # Update velocity: v += a * dt
        self.drone_vel += acc * self.dt
        
        # Limit velocity magnitude (safety check)
        speed = np.linalg.norm(self.drone_vel)
        if speed > self.max_speed:
            self.drone_vel = self.drone_vel / speed * self.max_speed
        
        # Update position: pos += v * dt
        new_pos = self.drone_pos + self.drone_vel * self.dt
        
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
        
        # Clamp position to world bounds
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

        # Check termination conditions
        reached_target = distance < self.arrival_threshold
        truncated = self.step_count >= self.max_steps
        terminated = reached_target
        
        # Calculate reward using current prev_distance (before update)
        if not reached_target:
            reward = self._compute_reward(distance, boundary_violation, boundary_penalty, 
                                        self.drone_vel, target_near_edge, current_action)
        else:
            reward = self.arrival_reward
        
        # Get observation
        observation = self._get_observation()
        
        info = {
            'distance': distance,
            'reached_target': reached_target,
            'boundary_violation': boundary_violation,
            'boundary_penalty': boundary_penalty
        }
        
        # Update previous values AFTER reward calculation
        self.prev_distance = distance
        self.prev_velocity = self.drone_vel.copy()
        self.prev_action = current_action.copy()
        self.prev_attitude = self.drone_attitude.copy()
        
        return observation, reward, terminated, truncated, info
    
    def _compute_reward(self, distance, boundary_violation=False, boundary_penalty=0.0, velocity=None, target_near_edge=False, action=None):
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
        10. Attitude stability reward (NEW: encourage stable flight)
        11. Attitude angle limit penalty (NEW: penalize excessive tilt)
        12. Attitude smoothness reward (NEW: penalize rapid attitude changes)
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
        # Penalize large changes in velocity and action
        if velocity is not None:
            velocity_change = np.linalg.norm(velocity - self.prev_velocity)
        else:
            velocity_change = 0.0
        
        if action is not None:
            action_change = np.linalg.norm(action - self.prev_action)
        else:
            action_change = 0.0
        
        if distance < 50.0:
            # Reduced smoothness penalty when close to target
            smoothness_reward = -0.00005 * min(velocity_change**2, 100.0) - 0.0001 * min(action_change**2, 10.0)
        else:
            # Normal smoothness penalty when far
            smoothness_reward = -0.0001 * min(velocity_change**2, 100.0) - 0.0002 * min(action_change**2, 10.0)
        
        # Attitude stability reward: encourage near-horizontal attitude (roll, pitch close to 0)
        # This helps agent learn stable flight
        # Increased weight from -0.1 to -1.0 (10x) to match the scale of other rewards
        roll = self.drone_attitude[0]
        pitch = self.drone_attitude[1]
        attitude_stability_reward = -1.0 * (abs(roll) + abs(pitch)) / (np.pi / 4)  # Normalize by 45 degrees
        
        # Attitude angle limit penalty: penalize excessive tilt (unsafe flight)
        # Penalize if roll or pitch exceeds 30 degrees (π/6)
        # Increased penalty from -1.0 to -5.0 (5x) to ensure safe flight
        max_safe_angle = np.pi / 6  # 30 degrees
        attitude_angle_penalty = 0.0
        if abs(roll) > max_safe_angle:
            attitude_angle_penalty -= 5.0 * (abs(roll) - max_safe_angle) / (np.pi / 6)
        if abs(pitch) > max_safe_angle:
            attitude_angle_penalty -= 5.0 * (abs(pitch) - max_safe_angle) / (np.pi / 6)
        
        # Attitude smoothness reward: penalize rapid attitude changes
        # Increased weight from -0.05 to -0.5 (10x) to match the scale of other rewards
        attitude_change = np.linalg.norm(self.drone_attitude - self.prev_attitude)
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

