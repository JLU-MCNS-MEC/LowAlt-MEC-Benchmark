"""
Training script: Train drone path planning using PPO algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import os
import random

# Configure font support
try:
    import matplotlib.font_manager as fm
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 
                     'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'STHeiti', 'Arial Unicode MS']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    font_found = False
    for font in chinese_fonts:
        if font in available_fonts:
            matplotlib.rcParams['font.sans-serif'] = [font] + matplotlib.rcParams['font.sans-serif']
            font_found = True
            break
    if not font_found:
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
except Exception:
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.environment import DronePathPlanningEnv
from core.ppo_agent import PPO


def train(
    num_episodes=4000,
    max_steps=600,  # 60s / 0.1s = 600 steps (dt=0.1 for finer control)
    update_frequency=10,  # Increased frequency (was 20)
    plot_dir='plots',
    model_dir='models',
    use_curriculum=True,  # Use curriculum learning for position strategy
    target_success_rate=80.0,  # Switch scenario when success rate >= this value (%)
    scenario_success_rate_window=50  # Calculate success rate over last N episodes
):
    """
    Train PPO agent
    
    Args:
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        update_frequency: Update policy every N episodes
        plot_dir: Directory to save plots
        model_dir: Directory to save models
        use_curriculum: Whether to use curriculum learning (switch scenarios based on success rate)
        target_success_rate: Target success rate (%) to switch to next scenario (default: 80.0)
        scenario_success_rate_window: Number of episodes to calculate success rate over (default: 50)
    """
    # Create directories
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create environment
    # For curriculum learning, we'll change position strategy during training
    env = DronePathPlanningEnv(world_size=1000, max_steps=max_steps)
    state_dim = env.observation_space.shape[0]
    # Continuous action space: action_dim is the dimension of action space (thrust, roll_torque, pitch_torque, yaw_torque = 4)
    action_dim = env.action_space.shape[0]
    
    # Curriculum learning: position change based on success rate
    current_fixed_start = None
    current_fixed_target = None
    position_change_episode = 0
    # Track success rate for current scenario
    scenario_successes = []  # Track successes for current scenario
    scenario_count = 0  # Track number of scenarios
    
    # Create PPO agent with improved hyperparameters
    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=3e-4,      # Standard learning rate for actor
        lr_critic=5e-4,     # Reduced from 1e-3 for more stable value learning
        gamma=0.99,
        gae_lambda=0.95,
        eps_clip=0.2,
        k_epochs=40,        # Increased from 10 for better policy updates
        value_clip=True     # Enable value clipping for stable critic learning
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    success_rates = []
    recent_successes = []
    actor_losses = []
    critic_losses = []
    
    # Track failed episodes at scenario switches (only first failed episode per scenario)
    scenario_switch_episodes = []  # Store episode numbers when scenario switches
    step_rewards_tracking = {}  # Will be populated for first failed episode in each scenario
    trajectories_tracking = {}  # Store (x, y) positions
    target_positions = {}  # Store target positions
    reached_targets = {}  # Track if reached target
    scenario_info = {}  # Store scenario information (start, target, distance)
    scenario_failed_episode = {}  # Track first failed episode for each scenario: {scenario_num: episode_num}
    
    # Track each scenario's training episodes and save long-training scenarios
    scenario_episodes = {}  # Track episode range for each scenario: {scenario_num: [start_ep, end_ep]}
    long_training_threshold = 200  # Episodes threshold for "long training" scenarios
    long_training_scenarios = []  # List of scenarios that took too long to train
    
    def mark_scenario_for_tracking(scenario_num, first_episode):
        """Mark the first episode of a new scenario for tracking (will save if it fails)"""
        if scenario_num not in scenario_failed_episode:
            # Initialize tracking for this episode
            step_rewards_tracking[first_episode] = []
            trajectories_tracking[first_episode] = []
            target_positions[first_episode] = None
            reached_targets[first_episode] = False
            scenario_failed_episode[scenario_num] = first_episode
    
    print("Starting training...")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    if use_curriculum:
        print("Using curriculum learning for position strategy")
        print(f"  Target success rate: {target_success_rate}%")
        print(f"  Success rate window: {scenario_success_rate_window} episodes")
        print("  Tracking first episode of each scenario (only save if failed)")
    
    for episode in tqdm(range(num_episodes), desc="Training progress"):
        # Curriculum learning: update position strategy based on success rate
        if use_curriculum:
            # Initialize first scenario - use fixed, easier scenario for initial learning
            if episode == 0:
                # First scenario: use a fixed, moderate distance (450m) to make learning easier
                first_scenario_distance = 450.0  # Fixed moderate distance
                current_fixed_start = np.array([200.0, 200.0], dtype=np.float32)
                # Place target at fixed distance (diagonal direction)
                current_fixed_target = current_fixed_start + np.array(
                    [first_scenario_distance * 0.6, first_scenario_distance * 0.8], 
                    dtype=np.float32
                )
                current_fixed_target = np.clip(current_fixed_target, 50, env.world_size - 50)
                
                env.set_fixed_positions(current_fixed_start, current_fixed_target)
                scenario_count = 1
                scenario_successes = []
                position_change_episode = episode
                # Mark first episode of first scenario for tracking (will save if it fails)
                mark_scenario_for_tracking(scenario_count, 0)
                # Record scenario start
                scenario_episodes[scenario_count] = {
                    'start_episode': episode,
                    'end_episode': None,  # Will be set when scenario completes
                    'start_pos': current_fixed_start.copy(),
                    'target_pos': current_fixed_target.copy(),
                    'distance': np.linalg.norm(current_fixed_target - current_fixed_start)
                }
                print(f"\nEpisode {episode}: Starting scenario {scenario_count} (EASIER FIRST SCENARIO)")
                print(f"  Start: ({current_fixed_start[0]:.1f}, {current_fixed_start[1]:.1f})")
                print(f"  Target: ({current_fixed_target[0]:.1f}, {current_fixed_target[1]:.1f})")
                print(f"  Distance: {np.linalg.norm(current_fixed_target - current_fixed_start):.1f}m")
            else:
                # Calculate current scenario success rate
                if len(scenario_successes) >= scenario_success_rate_window:
                    # Keep only last N episodes
                    scenario_successes = scenario_successes[-scenario_success_rate_window:]
                
                current_scenario_success_rate = 0.0
                if len(scenario_successes) > 0:
                    current_scenario_success_rate = np.mean(scenario_successes) * 100
                
                # Check if we should switch to next scenario
                # Adjust evaluation window based on current scenario distance
                current_distance = np.linalg.norm(current_fixed_target - current_fixed_start)
                if current_distance > 800:
                    current_window = 60  # More episodes for very long distances
                elif current_distance > 600:
                    current_window = 45  # More episodes for long distances
                else:
                    current_window = scenario_success_rate_window  # Standard for medium distances
                
                min_episodes_in_scenario = current_window  # Need at least N episodes to evaluate
                episodes_in_current_scenario = episode - position_change_episode
                
                # Add adaptation period after scenario switch (don't evaluate immediately)
                adaptation_period = 5  # Give 5 episodes to adapt to new scenario
                can_evaluate = episodes_in_current_scenario >= (min_episodes_in_scenario + adaptation_period)
                
                # Check if scenario is too difficult and needs reset
                # If training for too long with very low success rate, reduce difficulty
                scenario_reset_threshold = 300  # Episodes threshold for reset
                scenario_reset_success_rate = 20.0  # Success rate threshold for reset
                should_reset_scenario = (
                    episodes_in_current_scenario >= scenario_reset_threshold and
                    current_scenario_success_rate < scenario_reset_success_rate and
                    scenario_count > 1  # Don't reset first scenario
                )
                
                if should_reset_scenario:
                    print(f"\n⚠️  Scenario {scenario_count} is too difficult!")
                    print(f"   Episodes: {episodes_in_current_scenario}, Success rate: {current_scenario_success_rate:.1f}%")
                    print(f"   Resetting to easier scenario (reducing distance by 20%)")
                    
                    # Reduce distance by 20% and generate new scenario
                    old_distance = np.linalg.norm(current_fixed_target - current_fixed_start)
                    reduced_distance = old_distance * 0.8
                    
                    # Generate new scenario with reduced distance
                    current_fixed_start, current_fixed_target = generate_random_positions(
                        env.world_size,
                        old_distance=reduced_distance,
                        difficulty_increase=0.0  # No increase, just use reduced distance
                    )
                    env.set_fixed_positions(current_fixed_start, current_fixed_target)
                    position_change_episode = episode
                    scenario_successes = []  # Reset for new scenario
                    
                    new_distance = np.linalg.norm(current_fixed_target - current_fixed_start)
                    print(f"   New scenario distance: {new_distance:.1f}m (reduced from {old_distance:.1f}m)")
                    continue
                
                if (can_evaluate and 
                    current_scenario_success_rate >= target_success_rate):
                    # Complete current scenario and record its information
                    old_distance = np.linalg.norm(current_fixed_target - current_fixed_start)
                    episodes_in_scenario = episodes_in_current_scenario
                    
                    # Update scenario episodes record
                    if scenario_count in scenario_episodes:
                        scenario_episodes[scenario_count]['end_episode'] = episode
                        scenario_episodes[scenario_count]['final_success_rate'] = current_scenario_success_rate
                        scenario_episodes[scenario_count]['total_episodes'] = episodes_in_scenario
                    
                    # Check if this scenario took too long
                    if episodes_in_scenario >= long_training_threshold:
                        long_training_scenarios.append(scenario_count)
                        print(f"\n⚠️  WARNING: Scenario {scenario_count} took {episodes_in_scenario} episodes (>= {long_training_threshold})")
                        print(f"   This scenario will be saved for analysis")
                    
                    # Mark scenario switch
                    scenario_switch_episodes.append(episode)
                    
                    # Store old scenario info
                    scenario_info[scenario_count - 1] = {
                        'start': current_fixed_start.copy(),
                        'target': current_fixed_target.copy(),
                        'distance': old_distance,
                        'final_success_rate': current_scenario_success_rate,
                        'episodes': episodes_in_scenario
                    }
                    
                    # Save model checkpoint for long-training scenarios
                    if scenario_count in long_training_scenarios:
                        checkpoint_path = os.path.join(model_dir, f'ppo_model_scenario_{scenario_count}_episode_{episode}.pth')
                        agent.save(checkpoint_path)
                        print(f"   💾 Saved checkpoint: {checkpoint_path}")
                    
                    # Switch to next scenario with gradual difficulty increase
                    scenario_count += 1
                    
                    # Dynamic difficulty increase based on current distance
                    # Longer distances get smaller increases to avoid excessive difficulty
                    if old_distance > 800:
                        difficulty_increase = 0.05  # 5% for very long distances
                    elif old_distance > 600:
                        difficulty_increase = 0.10  # 10% for long distances
                    else:
                        difficulty_increase = 0.15  # 15% for short-medium distances
                    
                    current_fixed_start, current_fixed_target = generate_random_positions(
                        env.world_size, 
                        old_distance=old_distance,
                        difficulty_increase=difficulty_increase
                    )
                    
                    # Record new scenario start
                    new_distance = np.linalg.norm(current_fixed_target - current_fixed_start)
                    
                    # Force distance limit: if distance exceeds 600m, regenerate with reduced distance
                    max_allowed_distance = env.world_size * 0.6  # 600m hard limit
                    if new_distance > max_allowed_distance:
                        print(f"  ⚠️  Generated distance {new_distance:.1f}m exceeds limit {max_allowed_distance:.1f}m")
                        print(f"     Regenerating with reduced distance...")
                        # Regenerate with distance capped at limit
                        reduced_old_distance = min(old_distance, max_allowed_distance * 0.9)  # Use 90% of limit
                        current_fixed_start, current_fixed_target = generate_random_positions(
                            env.world_size,
                            old_distance=reduced_old_distance,
                            difficulty_increase=0.0  # No increase, just use reduced distance
                        )
                        new_distance = np.linalg.norm(current_fixed_target - current_fixed_start)
                        print(f"     New distance: {new_distance:.1f}m")
                    
                    env.set_fixed_positions(current_fixed_start, current_fixed_target)
                    position_change_episode = episode
                    scenario_successes = []  # Reset for new scenario
                    scenario_episodes[scenario_count] = {
                        'start_episode': episode,
                        'end_episode': None,  # Will be set when scenario completes
                        'start_pos': current_fixed_start.copy(),
                        'target_pos': current_fixed_target.copy(),
                        'distance': new_distance
                    }
                    
                    distance_change = new_distance - old_distance
                    distance_change_pct = (distance_change / old_distance * 100) if old_distance > 0 else 0
                    
                    print(f"\nEpisode {episode}: Scenario {scenario_count-1} completed!")
                    print(f"  Final success rate: {current_scenario_success_rate:.1f}%")
                    print(f"  Episodes in scenario: {episodes_in_scenario}")
                    print(f"  Old scenario distance: {old_distance:.1f}m")
                    print(f"  Starting scenario {scenario_count}")
                    print(f"  Start: ({current_fixed_start[0]:.1f}, {current_fixed_start[1]:.1f})")
                    print(f"  Target: ({current_fixed_target[0]:.1f}, {current_fixed_target[1]:.1f})")
                    print(f"  New scenario distance: {new_distance:.1f}m")
                    # Mark first episode of new scenario for tracking (will save if it fails)
                    first_episode_of_new_scenario = episode + 1  # Next episode is the first of new scenario
                    if first_episode_of_new_scenario < num_episodes:
                        mark_scenario_for_tracking(scenario_count, first_episode_of_new_scenario)
                    
                    print(f"  Distance change: {distance_change:+.1f}m ({distance_change_pct:+.1f}%)")
                    print(f"  Will track first episode of new scenario (episode {first_episode_of_new_scenario}) if it fails")
        else:
            # No curriculum: use random positions every episode
            env.set_random_positions()
        
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        reached_target = False
        
        # Track initial position and target for tracked episodes (scenario switches)
        if episode in trajectories_tracking:
            trajectories_tracking[episode].append(env.drone_pos.copy())
            if target_positions.get(episode) is None:
                target_positions[episode] = env.target_pos.copy()
            if episode not in reached_targets:
                reached_targets[episode] = False
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Store reward
            is_terminal = terminated or truncated
            agent.store_reward(reward, is_terminal)
            
            # Track step-by-step reward and trajectory for tracked episodes (first failed episode per scenario)
            if episode in step_rewards_tracking:
                step_rewards_tracking[episode].append(reward)
                trajectories_tracking[episode].append(env.drone_pos.copy())
                # Ensure target position is recorded
                if target_positions[episode] is None:
                    target_positions[episode] = env.target_pos.copy()
            
            episode_reward += reward
            episode_length += 1
            
            if info.get('reached_target', False):
                reached_target = True
                if episode in reached_targets:
                    reached_targets[episode] = True
            
            if terminated or truncated:
                break
            
            state = next_state
        
        # Record statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        recent_successes.append(1 if reached_target else 0)
        
        # Track success for current scenario (if using curriculum)
        # Skip adaptation period episodes from success rate calculation
        if use_curriculum:
            episodes_in_scenario = episode - position_change_episode + 1
            adaptation_period = 5
            if episodes_in_scenario > adaptation_period:
                # Only count episodes after adaptation period
                scenario_successes.append(1 if reached_target else 0)
        
        # Calculate success rate for last 100 episodes (global)
        if len(recent_successes) > 100:
            recent_successes.pop(0)
        success_rate = np.mean(recent_successes) * 100
        success_rates.append(success_rate)
        
        # Update policy
        if (episode + 1) % update_frequency == 0:
            loss_info = agent.update()
            # Record loss statistics
            actor_losses.append(loss_info['actor_loss'])
            critic_losses.append(loss_info['critic_loss'])
            
            if (episode + 1) % 100 == 0:
                print(f"\nEpisode {episode + 1}")
                print(f"  Average reward: {np.mean(episode_rewards[-update_frequency:]):.2f}")
                print(f"  Global success rate: {success_rate:.1f}%")
                if use_curriculum and len(scenario_successes) > 0:
                    current_scenario_rate = np.mean(scenario_successes) * 100
                    print(f"  Current scenario success rate: {current_scenario_rate:.1f}% ({len(scenario_successes)} episodes)")
                    print(f"  Scenario {scenario_count}, Episodes in scenario: {episode - position_change_episode + 1}")
                print(f"  Average steps: {np.mean(episode_lengths[-update_frequency:]):.1f}")
                print(f"  Actor loss: {loss_info['actor_loss']:.4f}")
                print(f"  Critic loss: {loss_info['critic_loss']:.4f}")
                print(f"  Total loss: {loss_info['loss']:.4f}")
    
    # Complete final scenario record
    if use_curriculum and scenario_count in scenario_episodes:
        scenario_episodes[scenario_count]['end_episode'] = num_episodes - 1
        if len(scenario_successes) > 0:
            final_scenario_rate = np.mean(scenario_successes) * 100
            scenario_episodes[scenario_count]['final_success_rate'] = final_scenario_rate
        scenario_episodes[scenario_count]['total_episodes'] = num_episodes - 1 - position_change_episode
        
        # Check if final scenario took too long
        final_episodes = scenario_episodes[scenario_count]['total_episodes']
        if final_episodes >= long_training_threshold and scenario_count not in long_training_scenarios:
            long_training_scenarios.append(scenario_count)
            # Save checkpoint for final long-training scenario
            checkpoint_path = os.path.join(model_dir, f'ppo_model_scenario_{scenario_count}_episode_{num_episodes-1}.pth')
            agent.save(checkpoint_path)
            print(f"\n⚠️  Final scenario {scenario_count} took {final_episodes} episodes (>= {long_training_threshold})")
            print(f"   💾 Saved checkpoint: {checkpoint_path}")
    
    # Plot training curves
    plot_training_curves(episode_rewards, episode_lengths, success_rates, plot_dir)
    # Plot separate reward curve
    plot_reward_curve(episode_rewards, plot_dir)
    # Plot loss curves
    if len(actor_losses) > 0:
        plot_loss_curves(actor_losses, critic_losses, plot_dir)
    # Plot step-by-step rewards and trajectories for scenario switch episodes
    # Only plot failed episodes (not successful ones)
    if len(scenario_switch_episodes) > 0:
        # Get all tracked episodes (around scenario switches)
        tracked_episodes = sorted(step_rewards_tracking.keys())
        # Filter to only failed episodes
        failed_episodes = [ep for ep in tracked_episodes if not reached_targets.get(ep, False)]
        print(f"\nTracking {len(tracked_episodes)} episodes around {len(scenario_switch_episodes)} scenario switches")
        print(f"  Total tracked: {len(tracked_episodes)}, Failed: {len(failed_episodes)}, Successful: {len(tracked_episodes) - len(failed_episodes)}")
        print(f"  Only plotting {len(failed_episodes)} failed episodes")
        print(f"Scenario switch episodes: {scenario_switch_episodes}")
        
        if len(failed_episodes) > 0:
            plot_step_rewards(step_rewards_tracking, episode_rewards, failed_episodes, plot_dir, reached_targets)
            plot_trajectories(trajectories_tracking, target_positions, reached_targets, failed_episodes, plot_dir, env.world_size, scenario_info, scenario_switch_episodes)
        else:
            print("  No failed episodes to plot")
    else:
        print("\nNo scenario switches detected, skipping trajectory analysis")
    
    # Save final model
    final_model_path = os.path.join(model_dir, 'ppo_model_final.pth')
    agent.save(final_model_path)
    
    # Save checkpoint model (for potential resuming)
    checkpoint_model_path = os.path.join(model_dir, f'ppo_model_episode_{num_episodes}.pth')
    agent.save(checkpoint_model_path)
    
    # Save scenario training summary
    if use_curriculum:
        save_scenario_training_summary(scenario_episodes, long_training_scenarios, model_dir, plot_dir)
    
    print("\nTraining completed!")
    print(f"Final global success rate: {success_rates[-1]:.1f}%")
    print(f"Final average reward: {np.mean(episode_rewards[-100:]):.2f}")
    if use_curriculum:
        print(f"\nCurriculum Learning Summary:")
        print(f"  Total scenarios completed: {scenario_count}")
        if len(scenario_successes) > 0:
            final_scenario_rate = np.mean(scenario_successes) * 100
            print(f"  Final scenario ({scenario_count}) success rate: {final_scenario_rate:.1f}%")
            print(f"  Episodes in final scenario: {len(scenario_successes)}")
        print(f"  Long-training scenarios (>= {long_training_threshold} episodes): {len(long_training_scenarios)}")
        if len(long_training_scenarios) > 0:
            print(f"    Scenario numbers: {long_training_scenarios}")
            print(f"    Checkpoints saved in: {model_dir}")
    if len(actor_losses) > 0:
        print(f"Final actor loss: {np.mean(actor_losses[-10:]):.4f}")
        print(f"Final critic loss: {np.mean(critic_losses[-10:]):.4f}")
    print(f"\nModel saved to: {final_model_path}")


def get_position_change_frequency(episode, total_episodes):
    """
    Curriculum learning: gradually increase position change frequency
    
    Stage 1 (0-25%): Fixed positions (change every 500 episodes)
    Stage 2 (25-50%): Change every 100 episodes
    Stage 3 (50-75%): Change every 50 episodes
    Stage 4 (75-90%): Change every 10 episodes
    Stage 5 (90-100%): Change every episode (fully random)
    """
    progress = episode / total_episodes
    
    if progress < 0.25:
        return 500  # Stage 1: Fixed positions, learn basic strategy
    elif progress < 0.50:
        return 100  # Stage 2: Moderate diversity
    elif progress < 0.75:
        return 50   # Stage 3: More diversity
    elif progress < 0.90:
        return 10   # Stage 4: High diversity
    else:
        return 1    # Stage 5: Fully random (every episode)


def generate_random_positions(world_size, min_distance_ratio=0.3, old_distance=None, difficulty_increase=0.15):
    """
    Generate random start and target positions with minimum distance
    If old_distance is provided, generate next scenario with gradual difficulty increase
    
    Args:
        world_size: Size of the world
        min_distance_ratio: Minimum distance as ratio of world_size
        old_distance: Previous scenario distance (for gradual difficulty)
        difficulty_increase: Maximum difficulty increase ratio (default: 15%)
    
    Returns:
        start_pos, target_pos: Random positions with minimum distance
    """
    # If old_distance is provided, use gradual difficulty increase
    if old_distance is not None and old_distance > 0:
        # New scenario distance in [old_distance * 0.95, old_distance * (1 + difficulty_increase)]
        # Slight decrease allowed (5%) to allow some variation, but mostly increase
        min_distance = old_distance * 0.95
        max_distance = old_distance * (1.0 + difficulty_increase)
        # Ensure within reasonable bounds
        min_distance = max(min_distance, world_size * min_distance_ratio)
        # Limit maximum distance to 60% of world size to avoid excessively difficult scenarios
        max_reasonable_distance = world_size * 0.6  # 600m maximum - HARD LIMIT
        max_distance = min(max_distance, world_size * np.sqrt(2) * 0.9, max_reasonable_distance)  # Max 90% of diagonal or 600m
        # Ensure target_distance does not exceed limit
        target_distance = np.random.uniform(min_distance, max_distance)
        target_distance = min(target_distance, max_reasonable_distance)  # Enforce hard limit
    else:
        # First scenario or no old_distance: use standard generation
        min_distance = world_size * min_distance_ratio
        max_distance = world_size * np.sqrt(2) * 0.9  # Max 90% of diagonal
        target_distance = None  # Will use min_distance as constraint
    
    max_attempts = 50  # Increased attempts for distance-constrained generation
    
    # Avoid edge positions for start (keep at least 50m from edge)
    edge_margin_start = 50.0
    for _ in range(max_attempts):
        start_pos = np.random.uniform(
            [edge_margin_start, edge_margin_start], 
            [world_size - edge_margin_start, world_size - edge_margin_start]
        ).astype(np.float32)
        
        if target_distance is not None:
            # Generate target at approximately target_distance from start
            # Use polar coordinates approach
            # Try multiple angles to avoid edge targets
            max_attempts = 20
            target_pos = None
            edge_margin = 30.0  # Keep target at least 30m from edge
            max_reasonable_distance = world_size * 0.6  # 600m hard limit
            
            for _ in range(max_attempts):
                angle = np.random.uniform(0, 2 * np.pi)
                candidate_pos = start_pos + np.array([
                    target_distance * np.cos(angle),
                    target_distance * np.sin(angle)
                ], dtype=np.float32)
                # Check if candidate is within safe bounds (not too close to edge)
                if (edge_margin <= candidate_pos[0] <= world_size - edge_margin and
                    edge_margin <= candidate_pos[1] <= world_size - edge_margin):
                    target_pos = candidate_pos
                    break
            
            # If no suitable position found, clip but prefer center
            if target_pos is None:
                angle = np.random.uniform(0, 2 * np.pi)
                target_pos = start_pos + np.array([
                    target_distance * np.cos(angle),
                    target_distance * np.sin(angle)
                ], dtype=np.float32)
                # Clip to safe bounds (with margin)
                target_pos = np.clip(target_pos, edge_margin, world_size - edge_margin)
            
            # Final check: ensure distance is within limit (enforce hard limit)
            actual_distance = np.linalg.norm(target_pos - start_pos)
            max_reasonable_distance = world_size * 0.6  # 600m hard limit
            if actual_distance > max_reasonable_distance:
                # Adjust target to enforce distance limit
                direction_vec = (target_pos - start_pos) / actual_distance
                target_pos = start_pos + direction_vec * max_reasonable_distance
                target_pos = np.clip(target_pos, edge_margin, world_size - edge_margin)
        else:
            # Standard random generation
            target_pos = np.random.uniform(
                [edge_margin, edge_margin], 
                [world_size - edge_margin, world_size - edge_margin]
            ).astype(np.float32)
        
        distance = np.linalg.norm(target_pos - start_pos)
        
        # Enforce maximum distance limit (hard check for all cases)
        max_reasonable_distance = world_size * 0.6  # 600m hard limit
        if distance > max_reasonable_distance:
            # Adjust target to enforce distance limit
            direction_vec = (target_pos - start_pos) / distance
            target_pos = start_pos + direction_vec * max_reasonable_distance
            target_pos = np.clip(target_pos, edge_margin, world_size - edge_margin)
            distance = max_reasonable_distance
        
        # Check if distance meets requirements
        if target_distance is not None:
            # For distance-constrained: check if close enough to target
            if abs(distance - target_distance) / target_distance < 0.1:  # Within 10%
                return start_pos, target_pos
        else:
            # Standard: check if >= min_distance
            if distance >= min_distance:
                return start_pos, target_pos
    
    # Fallback: use default positions if can't find suitable ones
    if old_distance is not None:
        # Use old distance with slight increase
        fallback_distance = old_distance * 1.05
        start_pos = np.array([world_size * 0.1, world_size * 0.5], dtype=np.float32)
        target_pos = start_pos + np.array([fallback_distance, 0], dtype=np.float32)
        target_pos = np.clip(target_pos, 0, world_size)
        return start_pos, target_pos
    else:
        return np.array([world_size * 0.1, world_size * 0.5], dtype=np.float32), \
               np.array([world_size * 0.9, world_size * 0.5], dtype=np.float32)


def save_scenario_training_summary(scenario_episodes, long_training_scenarios, model_dir, plot_dir):
    """
    Save scenario training summary to file
    
    Args:
        scenario_episodes: dict mapping scenario number to episode info
        long_training_scenarios: list of scenario numbers that took long to train
        model_dir: directory to save summary
        plot_dir: directory to save plots
    """
    summary_path = os.path.join(model_dir, 'scenario_training_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Scenario Training Summary\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total scenarios: {len(scenario_episodes)}\n")
        f.write(f"Long-training scenarios (>= 200 episodes): {len(long_training_scenarios)}\n")
        if len(long_training_scenarios) > 0:
            f.write(f"  Scenario numbers: {long_training_scenarios}\n")
        f.write("\n" + "="*80 + "\n\n")
        
        for scenario_num in sorted(scenario_episodes.keys()):
            info = scenario_episodes[scenario_num]
            f.write(f"Scenario {scenario_num}:\n")
            f.write(f"  Episode range: {info['start_episode']} - {info.get('end_episode', 'ongoing')}\n")
            if 'total_episodes' in info:
                f.write(f"  Total episodes: {info['total_episodes']}\n")
            f.write(f"  Start position: ({info['start_pos'][0]:.1f}, {info['start_pos'][1]:.1f})\n")
            f.write(f"  Target position: ({info['target_pos'][0]:.1f}, {info['target_pos'][1]:.1f})\n")
            f.write(f"  Distance: {info['distance']:.1f}m\n")
            if 'final_success_rate' in info:
                f.write(f"  Final success rate: {info['final_success_rate']:.1f}%\n")
            if scenario_num in long_training_scenarios:
                f.write(f"  ⚠️  LONG TRAINING SCENARIO\n")
                if 'end_episode' in info:
                    f.write(f"  Checkpoint: ppo_model_scenario_{scenario_num}_episode_{info['end_episode']}.pth\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("Long-Training Scenarios Details\n")
        f.write("="*80 + "\n\n")
        
        if len(long_training_scenarios) > 0:
            for scenario_num in long_training_scenarios:
                if scenario_num in scenario_episodes:
                    info = scenario_episodes[scenario_num]
                    f.write(f"Scenario {scenario_num} (Long Training):\n")
                    f.write(f"  Episodes: {info.get('total_episodes', 'N/A')}\n")
                    f.write(f"  Distance: {info['distance']:.1f}m\n")
                    f.write(f"  Success rate: {info.get('final_success_rate', 'N/A'):.1f}%\n")
                    if 'end_episode' in info:
                        f.write(f"  Checkpoint saved: ppo_model_scenario_{scenario_num}_episode_{info['end_episode']}.pth\n")
                    f.write("\n")
        else:
            f.write("No long-training scenarios found.\n")
    
    print(f"Scenario training summary saved to: {summary_path}")


def smooth(data, window=100):
    """Smoothing function"""
    if len(data) < window:
        return data
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2)
        smoothed.append(np.mean(data[start:end]))
    return smoothed


def plot_training_curves(episode_rewards, episode_lengths, success_rates, plot_dir):
    """Plot training curves"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    episodes = range(1, len(episode_rewards) + 1)
    
    # Reward curve
    axes[0].plot(episodes, smooth(episode_rewards), label='Smoothed Reward', alpha=0.7, linewidth=2, color='blue')
    axes[0].plot(episodes, episode_rewards, alpha=0.2, label='Raw Reward', color='lightblue')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Reward Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Steps curve
    axes[1].plot(episodes, smooth(episode_lengths), label='Smoothed Steps', alpha=0.7, linewidth=2, color='orange')
    axes[1].plot(episodes, episode_lengths, alpha=0.2, label='Raw Steps', color='lightcoral')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps')
    axes[1].set_title('Episode Length Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Success rate curve
    axes[2].plot(episodes, success_rates, label='Success Rate', color='green', linewidth=2)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Success Rate (%)')
    axes[2].set_title('Success Rate Curve (Last 100 Episodes)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/training_curves.png", dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {plot_dir}/training_curves.png")


def plot_reward_curve(episode_rewards, plot_dir):
    """Plot separate reward curve"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    episodes = range(1, len(episode_rewards) + 1)
    
    # Plot smoothed and raw rewards
    smoothed_rewards = smooth(episode_rewards)
    ax.plot(episodes, smoothed_rewards, label='Smoothed Reward', alpha=0.9, linewidth=2, color='#2E86AB')
    ax.plot(episodes, episode_rewards, alpha=0.15, label='Raw Reward', color='#A23B72')
    
    # Add statistics
    mean_reward = np.mean(episode_rewards)
    final_mean = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else mean_reward
    max_reward = np.max(episode_rewards)
    
    # Add mean lines
    ax.axhline(y=mean_reward, color='red', linestyle='--', alpha=0.5, label=f'Overall Mean: {mean_reward:.2f}')
    if len(episode_rewards) >= 100:
        ax.axhline(y=final_mean, color='green', linestyle='--', alpha=0.5, label=f'Recent Mean (last 100): {final_mean:.2f}')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Reward Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add text information
    textstr = f'Total Episodes: {len(episode_rewards)}\nMax Reward: {max_reward:.2f}\nMean Reward: {mean_reward:.2f}'
    if len(episode_rewards) >= 100:
        textstr += f'\nRecent Mean: {final_mean:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/reward_curve.png", dpi=300, bbox_inches='tight')
    print(f"Reward curve saved to: {plot_dir}/reward_curve.png")


def plot_loss_curves(actor_losses, critic_losses, plot_dir):
    """Plot loss curves"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Convert to update indices (each update happens every update_frequency episodes)
    update_indices = range(1, len(actor_losses) + 1)
    
    # Actor loss curve
    axes[0].plot(update_indices, actor_losses, label='Actor Loss', alpha=0.8, linewidth=2, color='#E63946')
    if len(actor_losses) > 10:
        smoothed_actor = smooth(actor_losses)
        axes[0].plot(update_indices, smoothed_actor, label='Smoothed Actor Loss', alpha=0.6, linewidth=2, color='#FF6B6B', linestyle='--')
    axes[0].set_xlabel('Update Step', fontsize=12)
    axes[0].set_ylabel('Actor Loss', fontsize=12)
    axes[0].set_title('Actor Loss Curve', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Critic loss curve
    axes[1].plot(update_indices, critic_losses, label='Critic Loss', alpha=0.8, linewidth=2, color='#457B9D')
    if len(critic_losses) > 10:
        smoothed_critic = smooth(critic_losses)
        axes[1].plot(update_indices, smoothed_critic, label='Smoothed Critic Loss', alpha=0.6, linewidth=2, color='#6C9BD2', linestyle='--')
    axes[1].set_xlabel('Update Step', fontsize=12)
    axes[1].set_ylabel('Critic Loss', fontsize=12)
    axes[1].set_title('Critic Loss Curve', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Add statistics text
    actor_mean = np.mean(actor_losses)
    critic_mean = np.mean(critic_losses)
    textstr = f'Actor Loss - Mean: {actor_mean:.4f}, Min: {np.min(actor_losses):.4f}, Max: {np.max(actor_losses):.4f}\n'
    textstr += f'Critic Loss - Mean: {critic_mean:.4f}, Min: {np.min(critic_losses):.4f}, Max: {np.max(critic_losses):.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes[0].text(0.02, 0.98, textstr, transform=axes[0].transAxes, fontsize=9,
                 verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/loss_curves.png", dpi=300, bbox_inches='tight')
    print(f"Loss curves saved to: {plot_dir}/loss_curves.png")


def plot_step_rewards(step_rewards_tracking, episode_rewards, sampled_episodes, plot_dir, reached_targets=None):
    """
    Plot step-by-step rewards for sampled episodes (only failed episodes)
    
    Args:
        step_rewards_tracking: dict mapping episode number to list of step rewards
        episode_rewards: list of total episode rewards
        sampled_episodes: list of sampled episode numbers (should already be filtered to failed only)
        plot_dir: directory to save plots
        reached_targets: dict mapping episode number to whether target was reached (optional, for additional filtering)
    """
    # Filter to only failed episodes if reached_targets is provided
    if reached_targets is not None:
        sampled_episodes = [ep for ep in sampled_episodes if not reached_targets.get(ep, False)]
    
    if len(sampled_episodes) == 0:
        print("No failed episodes to plot for step rewards")
        return
    
    # Create figure with subplots
    num_episodes = len(sampled_episodes)
    cols = 3
    rows = (num_episodes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if num_episodes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Color map for different episodes
    colors = plt.cm.tab10(np.linspace(0, 1, num_episodes))
    
    for idx, episode_num in enumerate(sampled_episodes):
        ax = axes[idx]
        step_rewards = step_rewards_tracking[episode_num]
        
        if len(step_rewards) == 0:
            ax.text(0.5, 0.5, f'Episode {episode_num}\nNo data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Episode {episode_num}', fontsize=12, fontweight='bold')
            continue
        
        steps = range(1, len(step_rewards) + 1)
        cumulative_rewards = np.cumsum(step_rewards)
        
        # Plot step rewards
        ax.plot(steps, step_rewards, alpha=0.6, linewidth=1.5, color=colors[idx], 
               label='Step Reward', marker='o', markersize=2)
        
        # Plot cumulative reward on secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(steps, cumulative_rewards, alpha=0.8, linewidth=2, 
                color='red', linestyle='--', label='Cumulative Reward')
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        
        # Labels and title
        ax.set_xlabel('Step', fontsize=10)
        ax.set_ylabel('Step Reward', fontsize=10, color=colors[idx])
        ax2.set_ylabel('Cumulative Reward', fontsize=10, color='red')
        
        total_reward = episode_rewards[episode_num] if episode_num < len(episode_rewards) else sum(step_rewards)
        ax.set_title(f'Episode {episode_num} (Total: {total_reward:.2f})', 
                    fontsize=12, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='y', labelcolor=colors[idx])
        ax2.tick_params(axis='y', labelcolor='red')
    
    # Hide unused subplots
    for idx in range(num_episodes, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/step_rewards_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Step-by-step reward analysis saved to: {plot_dir}/step_rewards_analysis.png")
    
    # Create summary plot showing all episodes together
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Step rewards overlaid
    for idx, episode_num in enumerate(sampled_episodes):
        step_rewards = step_rewards_tracking[episode_num]
        if len(step_rewards) > 0:
            steps = range(1, len(step_rewards) + 1)
            ax1.plot(steps, step_rewards, alpha=0.5, linewidth=1.5, 
                    color=colors[idx], label=f'Episode {episode_num}', marker='o', markersize=2)
    
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Step Reward', fontsize=12)
    ax1.set_title('Step-by-Step Rewards for Sampled Episodes', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative rewards
    for idx, episode_num in enumerate(sampled_episodes):
        step_rewards = step_rewards_tracking[episode_num]
        if len(step_rewards) > 0:
            steps = range(1, len(step_rewards) + 1)
            cumulative_rewards = np.cumsum(step_rewards)
            ax2.plot(steps, cumulative_rewards, alpha=0.7, linewidth=2, 
                    color=colors[idx], label=f'Episode {episode_num}')
    
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Cumulative Reward', fontsize=12)
    ax2.set_title('Cumulative Rewards for Sampled Episodes', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = "Statistics:\n"
    for episode_num in sampled_episodes:
        step_rewards = step_rewards_tracking[episode_num]
        if len(step_rewards) > 0:
            mean_reward = np.mean(step_rewards)
            std_reward = np.std(step_rewards)
            min_reward = np.min(step_rewards)
            max_reward = np.max(step_rewards)
            total_reward = sum(step_rewards)
            stats_text += f"Ep {episode_num}: Mean={mean_reward:.3f}, Std={std_reward:.3f}, "
            stats_text += f"Range=[{min_reward:.3f}, {max_reward:.3f}], Total={total_reward:.2f}\n"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=8,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/step_rewards_summary.png", dpi=300, bbox_inches='tight')
    print(f"Step-by-step reward summary saved to: {plot_dir}/step_rewards_summary.png")
    
    # Print analysis to console
    print("\n" + "="*60)
    print("Step-by-Step Reward Analysis")
    print("="*60)
    for episode_num in sampled_episodes:
        step_rewards = step_rewards_tracking[episode_num]
        if len(step_rewards) > 0:
            print(f"\nEpisode {episode_num}:")
            print(f"  Total steps: {len(step_rewards)}")
            print(f"  Total reward: {sum(step_rewards):.2f}")
            print(f"  Mean step reward: {np.mean(step_rewards):.3f}")
            print(f"  Std step reward: {np.std(step_rewards):.3f}")
            print(f"  Min step reward: {np.min(step_rewards):.3f}")
            print(f"  Max step reward: {np.max(step_rewards):.3f}")
            print(f"  Positive rewards: {sum(1 for r in step_rewards if r > 0)}/{len(step_rewards)}")
            print(f"  Negative rewards: {sum(1 for r in step_rewards if r < 0)}/{len(step_rewards)}")
    print("="*60)


def plot_trajectories(trajectories_tracking, target_positions, reached_targets, sampled_episodes, plot_dir, world_size, scenario_info=None, scenario_switch_episodes=None):
    """
    Plot trajectories for sampled episodes (only failed episodes)
    
    Args:
        trajectories_tracking: dict mapping episode number to list of (x, y) positions
        target_positions: dict mapping episode number to target (x, y) position
        reached_targets: dict mapping episode number to whether target was reached
        sampled_episodes: list of sampled episode numbers (should already be filtered to failed only)
        plot_dir: directory to save plots
        world_size: size of the world (for axis limits)
        scenario_info: dict with scenario information (optional)
        scenario_switch_episodes: list of episode numbers where scenario switches (optional)
    """
    # Filter to only failed episodes
    sampled_episodes = [ep for ep in sampled_episodes if not reached_targets.get(ep, False)]
    
    if len(sampled_episodes) == 0:
        print("No failed episodes to plot for trajectories")
        return
    
    num_episodes = len(sampled_episodes)
    cols = 3
    rows = (num_episodes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if num_episodes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Color map for different episodes
    colors = plt.cm.tab10(np.linspace(0, 1, num_episodes))
    
    for idx, episode_num in enumerate(sampled_episodes):
        ax = axes[idx]
        trajectory = trajectories_tracking[episode_num]
        target_pos = target_positions[episode_num]
        reached = reached_targets.get(episode_num, False)
        
        if len(trajectory) == 0 or target_pos is None:
            ax.text(0.5, 0.5, f'Episode {episode_num}\nNo data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Episode {episode_num}', fontsize=12, fontweight='bold')
            continue
        
        trajectory = np.array(trajectory)
        
        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.7, linewidth=2, 
               color=colors[idx], label='Trajectory', marker='o', markersize=3)
        
        # Plot start position
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=12, 
               label='Start', markeredgecolor='black', markeredgewidth=1.5)
        
        # Plot end position
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=12, 
               label='End', markeredgecolor='black', markeredgewidth=1.5)
        
        # Plot target position
        ax.plot(target_pos[0], target_pos[1], 'r*', markersize=20, 
               label='Target', markeredgecolor='black', markeredgewidth=1)
        
        # Draw target area circle
        arrival_threshold = world_size * 0.02  # 2% of world size
        circle = plt.Circle(target_pos, arrival_threshold, color='r', alpha=0.2, 
                           label='Target Area')
        ax.add_patch(circle)
        
        # Set axis limits
        ax.set_xlim([0, world_size])
        ax.set_ylim([0, world_size])
        ax.set_xlabel('X Coordinate (m)', fontsize=10)
        ax.set_ylabel('Y Coordinate (m)', fontsize=10)
        
        # Calculate final distance
        final_distance = np.linalg.norm(trajectory[-1] - target_pos)
        status = "Success" if reached else "Failed"
        
        # Add scenario switch indicator
        title = f'Episode {episode_num} - {status}\nFinal Distance: {final_distance:.1f}m'
        if scenario_switch_episodes and episode_num in scenario_switch_episodes:
            title += '\n[SCENARIO SWITCH]'
        elif scenario_switch_episodes:
            # Check if this is near a switch
            for switch_ep in scenario_switch_episodes:
                if abs(episode_num - switch_ep) <= 3:
                    if episode_num < switch_ep:
                        title += f'\n[Before Switch {switch_ep}]'
                    else:
                        title += f'\n[After Switch {switch_ep}]'
                    break
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    # Hide unused subplots
    for idx in range(num_episodes, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/trajectories_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Trajectory analysis saved to: {plot_dir}/trajectories_analysis.png")
    
    # Create summary plot showing all trajectories together
    fig2, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    for idx, episode_num in enumerate(sampled_episodes):
        trajectory = trajectories_tracking[episode_num]
        target_pos = target_positions[episode_num]
        reached = reached_targets.get(episode_num, False)
        
        if len(trajectory) == 0 or target_pos is None:
            continue
        
        trajectory = np.array(trajectory)
        
        # Plot trajectory (all should be failed since we filtered)
        ax.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.5, linewidth=1.5, 
               color=colors[idx], linestyle='--', 
               label=f'Ep {episode_num} ✗', marker='o', markersize=2)
        
        # Plot start position
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'o', markersize=8, 
               color=colors[idx], markeredgecolor='black', markeredgewidth=1)
        
        # Plot end position (always 'x' for failed)
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'x', markersize=10, 
               color=colors[idx], markeredgecolor='black', markeredgewidth=1.5)
        
        # Plot target position
        ax.plot(target_pos[0], target_pos[1], '*', markersize=15, 
               color=colors[idx], markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlim([0, world_size])
    ax.set_ylim([0, world_size])
    ax.set_xlabel('X Coordinate (m)', fontsize=12)
    ax.set_ylabel('Y Coordinate (m)', fontsize=12)
    ax.set_title('Failed Episodes Trajectories (Only)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Add statistics (all should be failed since we filtered)
    failed_count = sum(1 for ep in sampled_episodes if not reached_targets.get(ep, False))
    stats_text = f"Failed Episodes: {failed_count}/{num_episodes}\n"
    stats_text += "Note: Only failed episodes are plotted\n"
    stats_text += "Legend: ✗ = Failed\n"
    stats_text += "Markers: o = Start, x = End, * = Target"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/trajectories_summary.png", dpi=300, bbox_inches='tight')
    print(f"Trajectory summary saved to: {plot_dir}/trajectories_summary.png")
    
    # Print trajectory statistics
    print("\n" + "="*60)
    print("Trajectory Analysis")
    print("="*60)
    for episode_num in sampled_episodes:
        trajectory = trajectories_tracking[episode_num]
        target_pos = target_positions[episode_num]
        reached = reached_targets.get(episode_num, False)
        
        if len(trajectory) > 0 and target_pos is not None:
            trajectory = np.array(trajectory)
            final_distance = np.linalg.norm(trajectory[-1] - target_pos)
            total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
            initial_distance = np.linalg.norm(trajectory[0] - target_pos)
            
            print(f"\nEpisode {episode_num}:")
            print(f"  Status: {'Success' if reached else 'Failed'}")
            print(f"  Steps: {len(trajectory)}")
            print(f"  Initial distance: {initial_distance:.2f}m")
            print(f"  Final distance: {final_distance:.2f}m")
            print(f"  Total path length: {total_distance:.2f}m")
            print(f"  Efficiency: {initial_distance/total_distance*100:.1f}%" if total_distance > 0 else "  Efficiency: N/A")
    print("="*60)


if __name__ == "__main__":
    train(
        num_episodes=4000,  # Increased for more training
        max_steps=600,  # 60s / 0.1s = 600 steps (dt=0.1 for finer control)
        update_frequency=10,  # More frequent updates for faster learning
        target_success_rate=50.0,  # Lower threshold for first scenario (was 80.0)
        scenario_success_rate_window=30  # Smaller window for faster evaluation (was 50)
    )

