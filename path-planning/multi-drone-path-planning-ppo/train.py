"""
Training script: Train multi-drone path planning using PPO algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import os

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

from environment import MultiDronePathPlanningEnv
from ppo_agent import PPO


def train(
    num_drones=3,
    num_episodes=4000,
    max_steps=60,
    update_frequency=10,
    plot_dir='plots',
    model_dir='models',
    use_curriculum=True,
    target_success_rate=80.0,
    scenario_success_rate_window=50
):
    """
    Train PPO agent for multi-drone path planning
    
    Args:
        num_drones: Number of drones (must be between 1 and 9, inclusive)
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        update_frequency: Update policy every N episodes
        plot_dir: Directory to save plots
        model_dir: Directory to save models
        use_curriculum: Whether to use curriculum learning
        target_success_rate: Target success rate (%) to switch to next scenario
        scenario_success_rate_window: Number of episodes to calculate success rate over
    """
    # Validate num_drones
    if not isinstance(num_drones, int) or num_drones < 1 or num_drones > 9:
        raise ValueError(f"num_drones must be an integer between 1 and 9 (inclusive), got {num_drones}")
    
    # Create directories
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create environment
    env = MultiDronePathPlanningEnv(
        num_drones=num_drones,
        world_size=1000,
        max_steps=max_steps
    )
    
    # State and action dimensions (per drone)
    state_dim = env.observation_space.shape[1]  # 7D per drone
    action_dim = env.action_space.shape[1]  # 2D per drone (vx, vy)
    
    # Curriculum learning
    current_fixed_starts = None
    current_fixed_targets = None
    position_change_episode = 0
    scenario_successes = []
    scenario_count = 0
    current_scenario_episodes = []  # Track episodes in current scenario
    
    # Create PPO agent (shared policy for all drones)
    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=3e-4,
        lr_critic=5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        eps_clip=0.2,
        k_epochs=40,
        value_clip=True
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    success_rates = []
    recent_successes = []
    actor_losses = []
    critic_losses = []
    all_drones_success_rates = []  # Track success rate for all drones
    
    # Track scenarios with low success rates for trajectory plotting
    # Instead of random episodes, track episodes from scenarios with low success rates
    scenario_tracking = {}  # {scenario_num: {'start_ep': int, 'end_ep': int, 'success_rate': float, 'episodes': list}}
    low_success_threshold = 40.0  # Track scenarios with success rate below 40%
    num_trajectory_plots = 5  # Number of low-success scenarios to plot
    
    # Track trajectories for low-success scenarios
    trajectories_tracking = {}  # {episode: [trajectories per drone]}
    target_positions_tracking = {}  # {episode: target_positions}
    reached_targets_tracking = {}  # {episode: bool}
    scenario_episode_mapping = {}  # {episode: scenario_num}
    
    print("Starting multi-drone training...")
    print(f"Number of drones: {num_drones}")
    print(f"State dimension (per drone): {state_dim}, Action dimension (per drone): {action_dim}")
    print(f"Will track and plot episodes from scenarios with success rate < {low_success_threshold}%")
    if use_curriculum:
        print("Using curriculum learning for position strategy")
        print(f"  Target success rate: {target_success_rate}%")
        print(f"  Success rate window: {scenario_success_rate_window} episodes")
    
    for episode in tqdm(range(num_episodes), desc="Training progress"):
        # Curriculum learning: update position strategy based on success rate
        if use_curriculum:
            if episode == 0:
                # First scenario: use fixed, moderate distances
                first_scenario_distance = 450.0
                current_fixed_starts = []
                current_fixed_targets = []
                
                # Generate evenly distributed positions
                edge_margin = 50.0
                available_size = env.world_size - 2 * edge_margin
                grid_size = int(np.ceil(np.sqrt(num_drones)))
                
                for i in range(num_drones):
                    # Evenly distributed start positions using grid
                    row = i // grid_size
                    col = i % grid_size
                    cell_width = available_size / grid_size
                    cell_height = available_size / grid_size
                    
                    start_x = edge_margin + col * cell_width + cell_width / 2
                    start_y = edge_margin + row * cell_height + cell_height / 2
                    start_pos = np.array([start_x, start_y], dtype=np.float32)
                    
                    # Place target at fixed distance, but in different grid position for even distribution
                    # Use opposite or shifted grid position
                    target_row = (i + grid_size // 2) % grid_size
                    target_col = (i + 1) % grid_size
                    
                    target_x = edge_margin + target_col * cell_width + cell_width / 2
                    target_y = edge_margin + target_row * cell_height + cell_height / 2
                    target_pos = np.array([target_x, target_y], dtype=np.float32)
                    
                    # Adjust target to be at approximately first_scenario_distance from start
                    direction = target_pos - start_pos
                    if np.linalg.norm(direction) > 1e-6:
                        direction = direction / np.linalg.norm(direction)
                    else:
                        # Random direction if same position
                        angle = 2 * np.pi * i / num_drones
                        direction = np.array([np.cos(angle), np.sin(angle)])
                    
                    target_pos = start_pos + direction * first_scenario_distance
                    target_pos = np.clip(target_pos, edge_margin, env.world_size - edge_margin)
                    
                    current_fixed_starts.append(start_pos)
                    current_fixed_targets.append(target_pos)
                
                current_fixed_starts = np.array(current_fixed_starts)
                current_fixed_targets = np.array(current_fixed_targets)
                
                env.set_fixed_positions(current_fixed_starts.tolist(), current_fixed_targets.tolist())
                scenario_count = 1
                scenario_successes = []
                position_change_episode = episode
                current_scenario_episodes = []
                
                print(f"\nEpisode {episode}: Starting scenario {scenario_count}")
                print(f"  Number of drones: {num_drones}")
                avg_distance = np.mean([
                    np.linalg.norm(current_fixed_targets[i] - current_fixed_starts[i])
                    for i in range(num_drones)
                ])
                print(f"  Average distance: {avg_distance:.1f}m")
            else:
                # Calculate current scenario success rate
                if len(scenario_successes) >= scenario_success_rate_window:
                    scenario_successes = scenario_successes[-scenario_success_rate_window:]
                
                current_scenario_success_rate = 0.0
                if len(scenario_successes) > 0:
                    current_scenario_success_rate = np.mean(scenario_successes) * 100
                
                # Check if we should switch to next scenario
                min_episodes_in_scenario = scenario_success_rate_window
                episodes_in_current_scenario = episode - position_change_episode
                adaptation_period = 5
                can_evaluate = episodes_in_current_scenario >= (min_episodes_in_scenario + adaptation_period)
                
                if can_evaluate and current_scenario_success_rate >= target_success_rate:
                    # Record scenario information before switching
                    completed_scenario = scenario_count
                    scenario_start_ep = position_change_episode
                    scenario_end_ep = episode - 1
                    
                    # Check if this scenario had low success rate
                    if current_scenario_success_rate < low_success_threshold:
                        scenario_tracking[completed_scenario] = {
                            'start_ep': scenario_start_ep,
                            'end_ep': scenario_end_ep,
                            'success_rate': current_scenario_success_rate,
                            'episodes': current_scenario_episodes.copy()
                        }
                        print(f"\n⚠️  Scenario {completed_scenario} has low success rate: {current_scenario_success_rate:.1f}%")
                        print(f"   Will track episodes from this scenario for plotting")
                    
                    # Switch to next scenario
                    scenario_count += 1
                    
                    # Generate new positions with increased difficulty
                    avg_old_distance = np.mean([
                        np.linalg.norm(current_fixed_targets[i] - current_fixed_starts[i])
                        for i in range(num_drones)
                    ])
                    
                    difficulty_increase = 0.15
                    if avg_old_distance > 800:
                        difficulty_increase = 0.05
                    elif avg_old_distance > 600:
                        difficulty_increase = 0.10
                    
                    current_fixed_starts, current_fixed_targets = generate_random_positions(
                        env.world_size,
                        num_drones,
                        old_distance=avg_old_distance,
                        difficulty_increase=difficulty_increase
                    )
                    
                    env.set_fixed_positions(current_fixed_starts.tolist(), current_fixed_targets.tolist())
                    position_change_episode = episode
                    scenario_successes = []
                    current_scenario_episodes = []
                    
                    avg_new_distance = np.mean([
                        np.linalg.norm(current_fixed_targets[i] - current_fixed_starts[i])
                        for i in range(num_drones)
                    ])
                    
                    print(f"\nEpisode {episode}: Scenario {completed_scenario} completed!")
                    print(f"  Final success rate: {current_scenario_success_rate:.1f}%")
                    print(f"  Starting scenario {scenario_count}")
                    print(f"  Average distance: {avg_new_distance:.1f}m (was {avg_old_distance:.1f}m)")
        else:
            # No curriculum: use random positions every episode
            env.set_random_positions()
        
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        all_reached = False
        
        # Track current scenario episodes
        if use_curriculum:
            current_scenario_episodes.append(episode)
            scenario_episode_mapping[episode] = scenario_count
        
        # Check if we should track this episode (from low-success scenarios)
        # We'll track episodes during the scenario, and later select which ones to plot
        should_track = False
        if use_curriculum:
            # Track all episodes in current scenario for potential plotting
            # We'll filter later based on scenario success rate
            should_track = True  # Track all episodes, filter later when plotting
        
        # Initialize tracking for this episode if needed
        if should_track:
            trajectories_tracking[episode] = [[] for _ in range(num_drones)]
            target_positions_tracking[episode] = env.target_positions.copy()
            reached_targets_tracking[episode] = False
        
        # Track initial positions for tracked episodes
        if should_track:
            for i in range(num_drones):
                trajectories_tracking[episode][i].append(env.drone_positions[i].copy())
        
        for step in range(max_steps):
            # Select actions for all drones (shared policy)
            # Flatten observations for agent (process each drone independently)
            actions = []
            for i in range(num_drones):
                action = agent.select_action(state[i])
                actions.append(action)
            actions = np.array(actions)  # Shape: (num_drones, 2)
            
            # Execute actions
            next_state, reward, terminated, truncated, info = env.step(actions)
            
            # Track trajectories for low-success scenario episodes
            if should_track:
                for i in range(num_drones):
                    trajectories_tracking[episode][i].append(env.drone_positions[i].copy())
            
            # Store rewards for each drone (use individual rewards)
            individual_rewards = info.get('individual_rewards', [reward] * num_drones)
            is_terminal = terminated or truncated
            
            # Store experience for each drone
            for i in range(num_drones):
                agent.store_reward(individual_rewards[i], is_terminal)
            
            episode_reward += reward
            episode_length += 1
            
            if info.get('all_reached', False):
                all_reached = True
                if should_track:
                    reached_targets_tracking[episode] = True
            
            if terminated or truncated:
                break
            
            state = next_state
        
        # Record statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        recent_successes.append(1 if all_reached else 0)
        
        # Track success for current scenario
        if use_curriculum:
            episodes_in_scenario = episode - position_change_episode + 1
            adaptation_period = 5
            if episodes_in_scenario > adaptation_period:
                scenario_successes.append(1 if all_reached else 0)
        
        # Calculate success rate
        if len(recent_successes) > 100:
            recent_successes.pop(0)
        success_rate = np.mean(recent_successes) * 100
        success_rates.append(success_rate)
        
        # Update policy
        if (episode + 1) % update_frequency == 0:
            loss_info = agent.update()
            actor_losses.append(loss_info['actor_loss'])
            critic_losses.append(loss_info['critic_loss'])
            
            if (episode + 1) % 100 == 0:
                print(f"\nEpisode {episode + 1}")
                print(f"  Average reward: {np.mean(episode_rewards[-update_frequency:]):.2f}")
                print(f"  Global success rate (all drones): {success_rate:.1f}%")
                if use_curriculum and len(scenario_successes) > 0:
                    current_scenario_rate = np.mean(scenario_successes) * 100
                    print(f"  Current scenario success rate: {current_scenario_rate:.1f}% ({len(scenario_successes)} episodes)")
                    print(f"  Scenario {scenario_count}, Episodes in scenario: {episode - position_change_episode + 1}")
                print(f"  Average steps: {np.mean(episode_lengths[-update_frequency:]):.1f}")
                print(f"  Actor loss: {loss_info['actor_loss']:.4f}")
                print(f"  Critic loss: {loss_info['critic_loss']:.4f}")
    
    # Plot training curves
    plot_training_curves(episode_rewards, episode_lengths, success_rates, plot_dir, num_drones)
    if len(actor_losses) > 0:
        plot_loss_curves(actor_losses, critic_losses, plot_dir)
    
    # Handle final scenario if it has low success rate
    if use_curriculum and len(scenario_successes) > 0:
        final_scenario_rate = np.mean(scenario_successes) * 100
        if final_scenario_rate < low_success_threshold:
            # Final scenario also has low success rate, track it
            scenario_tracking[scenario_count] = {
                'start_ep': position_change_episode,
                'end_ep': num_episodes - 1,
                'success_rate': final_scenario_rate,
                'episodes': current_scenario_episodes.copy()
            }
            print(f"\n⚠️  Final scenario {scenario_count} has low success rate: {final_scenario_rate:.1f}%")
    
    # Select episodes from low-success scenarios for plotting
    episodes_to_plot = []
    if len(scenario_tracking) > 0:
        # Sort scenarios by success rate (lowest first)
        sorted_scenarios = sorted(scenario_tracking.items(), key=lambda x: x[1]['success_rate'])
        
        # Select up to num_trajectory_plots scenarios
        selected_scenarios = sorted_scenarios[:num_trajectory_plots]
        
        # For each selected scenario, pick 1 episode (prefer failed episodes)
        for scenario_num, scenario_info in selected_scenarios:
            scenario_episodes = scenario_info['episodes']
            if len(scenario_episodes) > 0:
                # Try to find a failed episode first
                failed_episode = None
                for ep in scenario_episodes:
                    if ep in reached_targets_tracking and not reached_targets_tracking[ep]:
                        failed_episode = ep
                        break
                
                # If no failed episode, use the last episode
                episode_to_plot = failed_episode if failed_episode is not None else scenario_episodes[-1]
                
                if episode_to_plot in trajectories_tracking:
                    episodes_to_plot.append(episode_to_plot)
        
        print(f"\n📊 Low-Success Scenario Analysis:")
        print(f"  Found {len(scenario_tracking)} scenarios with success rate < {low_success_threshold}%")
        print(f"  Selected {len(episodes_to_plot)} episodes from {len(selected_scenarios)} scenarios for plotting")
        for ep in episodes_to_plot:
            scenario_num = scenario_episode_mapping.get(ep, 'unknown')
            success_rate = scenario_tracking.get(scenario_num, {}).get('success_rate', 0.0)
            reached = reached_targets_tracking.get(ep, False)
            status = "Failed" if not reached else "Success"
            print(f"    Episode {ep+1} (Scenario {scenario_num}, Success rate: {success_rate:.1f}%, Status: {status})")
    else:
        print(f"\n📊 No scenarios found with success rate < {low_success_threshold}%")
    
    # Plot trajectories from low-success scenarios (only failed episodes, max 5)
    if len(episodes_to_plot) > 0:
        # Filter to only failed episodes and limit to 5
        failed_episodes = [ep for ep in episodes_to_plot if not reached_targets_tracking.get(ep, False)]
        if len(failed_episodes) == 0:
            # If no failed episodes, use the selected ones (but they should mostly be failed)
            failed_episodes = episodes_to_plot[:num_trajectory_plots]
        else:
            failed_episodes = failed_episodes[:num_trajectory_plots]
        
        if len(failed_episodes) > 0:
            print(f"\n  Plotting {len(failed_episodes)} failed episodes from low-success scenarios")
            plot_training_trajectories(trajectories_tracking, target_positions_tracking, 
                                       reached_targets_tracking, failed_episodes, 
                                       plot_dir, env.world_size, num_drones, scenario_tracking, scenario_episode_mapping)
        else:
            print("  No failed episodes to plot")
    else:
        print("  No trajectories to plot (no low-success scenarios found)")
    
    # Save final model
    final_model_path = os.path.join(model_dir, 'ppo_model_final.pth')
    agent.save(final_model_path)
    
    checkpoint_model_path = os.path.join(model_dir, f'ppo_model_episode_{num_episodes}.pth')
    agent.save(checkpoint_model_path)
    
    print("\nTraining completed!")
    if len(success_rates) > 0:
        print(f"Final global success rate: {success_rates[-1]:.1f}%")
    else:
        print("Final global success rate: N/A (no episodes completed)")
    
    if len(episode_rewards) > 0:
        if len(episode_rewards) >= 100:
            print(f"Final average reward: {np.mean(episode_rewards[-100:]):.2f}")
        else:
            print(f"Final average reward: {np.mean(episode_rewards):.2f}")
    else:
        print("Final average reward: N/A (no episodes completed)")
    
    if use_curriculum:
        print(f"\nCurriculum Learning Summary:")
        print(f"  Total scenarios completed: {scenario_count}")
        if len(scenario_successes) > 0:
            final_scenario_rate = np.mean(scenario_successes) * 100
            print(f"  Final scenario ({scenario_count}) success rate: {final_scenario_rate:.1f}%")
        else:
            print(f"  Final scenario ({scenario_count}) success rate: N/A")
    
    if len(actor_losses) > 0:
        if len(actor_losses) >= 10:
            print(f"Final actor loss: {np.mean(actor_losses[-10:]):.4f}")
            print(f"Final critic loss: {np.mean(critic_losses[-10:]):.4f}")
        else:
            print(f"Final actor loss: {np.mean(actor_losses):.4f}")
            print(f"Final critic loss: {np.mean(critic_losses):.4f}")
    
    print(f"\nModel saved to: {final_model_path}")


def generate_random_positions(world_size, num_drones, min_distance_ratio=0.3, old_distance=None, difficulty_increase=0.15):
    """
    Generate random start and target positions for multiple drones
    
    Args:
        world_size: Size of the world
        num_drones: Number of drones
        min_distance_ratio: Minimum distance as ratio of world_size
        old_distance: Previous scenario average distance
        difficulty_increase: Maximum difficulty increase ratio
    
    Returns:
        start_positions, target_positions: Arrays of shape (num_drones, 2)
    """
    if old_distance is not None and old_distance > 0:
        min_distance = old_distance * 0.95
        max_distance = old_distance * (1.0 + difficulty_increase)
        min_distance = max(min_distance, world_size * min_distance_ratio)
        max_reasonable_distance = world_size * 0.6
        max_distance = min(max_distance, world_size * np.sqrt(2) * 0.9, max_reasonable_distance)
        target_distance = np.random.uniform(min_distance, max_distance)
        target_distance = min(target_distance, max_reasonable_distance)
    else:
        min_distance = world_size * min_distance_ratio
        max_distance = world_size * np.sqrt(2) * 0.9
        target_distance = None
    
    start_positions = []
    target_positions = []
    edge_margin = 50.0
    available_size = world_size - 2 * edge_margin
    
    # Calculate grid layout for even distribution
    grid_size = int(np.ceil(np.sqrt(num_drones)))
    
    for i in range(num_drones):
        # Evenly distributed start positions using grid
        row = i // grid_size
        col = i % grid_size
        cell_width = available_size / grid_size
        cell_height = available_size / grid_size
        
        # Add small random offset within cell for variety
        offset_range = min(cell_width, cell_height) * 0.2  # 20% of cell size
        offset_x = np.random.uniform(-offset_range, offset_range)
        offset_y = np.random.uniform(-offset_range, offset_range)
        
        start_x = edge_margin + col * cell_width + cell_width / 2 + offset_x
        start_y = edge_margin + row * cell_height + cell_height / 2 + offset_y
        
        # Ensure within bounds
        start_x = np.clip(start_x, edge_margin, world_size - edge_margin)
        start_y = np.clip(start_y, edge_margin, world_size - edge_margin)
        start_pos = np.array([start_x, start_y], dtype=np.float32)
        
        # Generate target position in different grid cell for even distribution
        if target_distance is not None:
            # Use shifted grid position for target
            target_row = (i + grid_size // 2) % grid_size
            target_col = (i + 1) % grid_size
            
            target_x = edge_margin + target_col * cell_width + cell_width / 2
            target_y = edge_margin + target_row * cell_height + cell_height / 2
            target_pos = np.array([target_x, target_y], dtype=np.float32)
            
            # Adjust to be at approximately target_distance from start
            direction = target_pos - start_pos
            if np.linalg.norm(direction) > 1e-6:
                direction = direction / np.linalg.norm(direction)
            else:
                # Random direction if same position
                angle = np.random.uniform(0, 2 * np.pi)
                direction = np.array([np.cos(angle), np.sin(angle)])
            
            target_pos = start_pos + direction * target_distance
            target_pos = np.clip(target_pos, edge_margin, world_size - edge_margin)
            
            # Enforce distance limit
            actual_distance = np.linalg.norm(target_pos - start_pos)
            max_reasonable_distance = world_size * 0.6
            if actual_distance > max_reasonable_distance:
                direction_vec = (target_pos - start_pos) / actual_distance
                target_pos = start_pos + direction_vec * max_reasonable_distance
                target_pos = np.clip(target_pos, edge_margin, world_size - edge_margin)
        else:
            # Standard: use different grid position
            target_row = (i + grid_size // 2) % grid_size
            target_col = (i + 1) % grid_size
            
            offset_range = min(cell_width, cell_height) * 0.2
            offset_x = np.random.uniform(-offset_range, offset_range)
            offset_y = np.random.uniform(-offset_range, offset_range)
            
            target_x = edge_margin + target_col * cell_width + cell_width / 2 + offset_x
            target_y = edge_margin + target_row * cell_height + cell_height / 2 + offset_y
            
            target_x = np.clip(target_x, edge_margin, world_size - edge_margin)
            target_y = np.clip(target_y, edge_margin, world_size - edge_margin)
            target_pos = np.array([target_x, target_y], dtype=np.float32)
        
        start_positions.append(start_pos)
        target_positions.append(target_pos)
    
    return np.array(start_positions), np.array(target_positions)


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


def plot_training_curves(episode_rewards, episode_lengths, success_rates, plot_dir, num_drones):
    """Plot training curves"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    episodes = range(1, len(episode_rewards) + 1)
    
    # Reward curve
    axes[0].plot(episodes, smooth(episode_rewards), label='Smoothed Reward', alpha=0.7, linewidth=2, color='blue')
    axes[0].plot(episodes, episode_rewards, alpha=0.2, label='Raw Reward', color='lightblue')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title(f'Training Reward Curve ({num_drones} Drones)')
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
    axes[2].plot(episodes, success_rates, label='Success Rate (All Drones)', color='green', linewidth=2)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Success Rate (%)')
    axes[2].set_title('Success Rate Curve (Last 100 Episodes)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/training_curves.png", dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {plot_dir}/training_curves.png")


def plot_loss_curves(actor_losses, critic_losses, plot_dir):
    """Plot loss curves"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
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
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/loss_curves.png", dpi=300, bbox_inches='tight')
    print(f"Loss curves saved to: {plot_dir}/loss_curves.png")


def plot_training_trajectories(trajectories_tracking, target_positions_tracking, 
                               reached_targets_tracking, episodes_to_track, 
                               plot_dir, world_size, num_drones, 
                               scenario_tracking=None, scenario_episode_mapping=None):
    """
    Plot trajectories for episodes from low-success scenarios
    Each episode is saved as a separate image file
    
    Args:
        trajectories_tracking: dict mapping episode number to list of trajectories (one per drone)
        target_positions_tracking: dict mapping episode number to target positions array
        reached_targets_tracking: dict mapping episode number to whether all drones reached
        episodes_to_track: list of episode numbers to plot
        plot_dir: directory to save plots
        world_size: size of the world
        num_drones: number of drones
        scenario_tracking: dict with scenario information (optional)
        scenario_episode_mapping: dict mapping episode to scenario number (optional)
    """
    if len(episodes_to_track) == 0:
        return
    
    # Color map for different drones
    drone_colors = plt.cm.tab10(np.linspace(0, 1, num_drones))
    
    # Plot each episode separately
    for episode_num in sorted(episodes_to_track):
        trajectories = trajectories_tracking.get(episode_num)
        target_positions = target_positions_tracking.get(episode_num)
        all_reached = reached_targets_tracking.get(episode_num, False)
        
        if target_positions is None or trajectories is None or len(trajectories) == 0:
            print(f"  Skipping episode {episode_num + 1}: No data")
            continue
        
        # Create a new figure for this episode
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # Plot trajectories for all drones
        for i in range(num_drones):
            if len(trajectories[i]) == 0:
                continue
            
            trajectory = np.array(trajectories[i])
            target_pos = target_positions[i]
            
            # Plot trajectory
            ax.plot(trajectory[:, 0], trajectory[:, 1], '-', linewidth=2, 
                   color=drone_colors[i], label=f'Drone {i+1}', alpha=0.7, 
                   marker='o', markersize=3)
            
            # Plot start position
            ax.plot(trajectory[0, 0], trajectory[0, 1], 'o', markersize=10, 
                   color=drone_colors[i], markeredgecolor='black', markeredgewidth=1.5,
                   label=f'Start {i+1}' if i == 0 else '')
            
            # Plot end position
            ax.plot(trajectory[-1, 0], trajectory[-1, 1], 's', markersize=10, 
                   color=drone_colors[i], markeredgecolor='black', markeredgewidth=1.5,
                   label=f'End {i+1}' if i == 0 else '')
            
            # Plot target position
            ax.plot(target_pos[0], target_pos[1], '*', markersize=20, 
                   color=drone_colors[i], markeredgecolor='black', markeredgewidth=1,
                   label=f'Target {i+1}' if i == 0 else '')
            
            # Draw target area circle
            circle = plt.Circle(target_pos, 20.0, color=drone_colors[i], alpha=0.2)
            ax.add_patch(circle)
        
        # Set axis limits
        ax.set_xlim([0, world_size])
        ax.set_ylim([0, world_size])
        ax.set_xlabel('X Coordinate (m)', fontsize=12)
        ax.set_ylabel('Y Coordinate (m)', fontsize=12)
        
        # Calculate final distances
        final_distances = []
        for i in range(num_drones):
            if len(trajectories[i]) > 0 and target_positions is not None:
                trajectory = np.array(trajectories[i])
                final_distance = np.linalg.norm(trajectory[-1] - target_positions[i])
                final_distances.append(final_distance)
        
        status = "Success (All Reached)" if all_reached else "Partial/Failed"
        avg_distance = np.mean(final_distances) if len(final_distances) > 0 else 0.0
        
        # Add scenario information to title
        title = f'Episode {episode_num + 1} - {status}\nAverage Distance: {avg_distance:.1f}m'
        if scenario_episode_mapping and scenario_tracking:
            scenario_num = scenario_episode_mapping.get(episode_num)
            if scenario_num and scenario_num in scenario_tracking:
                scenario_info = scenario_tracking[scenario_num]
                success_rate = scenario_info.get('success_rate', 0.0)
                title += f'\nScenario {scenario_num} (Success Rate: {success_rate:.1f}%)'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Save individual episode plot
        plt.tight_layout()
        episode_filename = f"{plot_dir}/training_trajectory_episode_{episode_num + 1}.png"
        plt.savefig(episode_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {episode_filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train multi-drone path planning with PPO')
    parser.add_argument('--num_drones', type=int, default=3, 
                       help='Number of drones (must be between 1 and 9, default: 3)')
    parser.add_argument('--num_episodes', type=int, default=3000,
                       help='Number of training episodes (default: 8000)')
    parser.add_argument('--max_steps', type=int, default=60,
                       help='Maximum steps per episode (default: 60)')
    parser.add_argument('--update_frequency', type=int, default=10,
                       help='Update policy every N episodes (default: 10)')
    parser.add_argument('--target_success_rate', type=float, default=80.0,
                       help='Target success rate to switch scenario (default: 80.0)')
    parser.add_argument('--scenario_success_rate_window', type=int, default=50,
                       help='Window size for success rate calculation (default: 50)')
    
    args = parser.parse_args()
    
    # Validate num_drones
    if args.num_drones < 1 or args.num_drones > 9:
        parser.error(f"num_drones must be between 1 and 9 (inclusive), got {args.num_drones}")
    
    train(
        num_drones=args.num_drones,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        update_frequency=args.update_frequency,
        target_success_rate=args.target_success_rate,
        scenario_success_rate_window=args.scenario_success_rate_window
    )

