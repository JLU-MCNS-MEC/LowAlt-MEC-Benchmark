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

from environment import DronePathPlanningEnv
from ppo_agent import PPO


def train(
    num_episodes=2000,
    max_steps=60,  # 60s / 1s = 60 steps
    update_frequency=10,  # Increased frequency (was 20)
    plot_dir='plots'
):
    """
    Train PPO agent
    
    Args:
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        update_frequency: Update policy every N episodes
        plot_dir: Directory to save plots
    """
    # Create directories
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create environment
    env = DronePathPlanningEnv(world_size=1000, max_steps=max_steps)
    state_dim = env.observation_space.shape[0]
    # Continuous action space: action_dim is the dimension of action space (vx, vy = 2)
    action_dim = env.action_space.shape[0]
    
    # Create PPO agent with improved hyperparameters
    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=3e-4,      # Reduced from 1e-3 for stability
        lr_critic=5e-4,    # Reduced from 1e-3 for stability
        gamma=0.99,
        gae_lambda=0.95,
        eps_clip=0.2,
        k_epochs=10
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    success_rates = []
    recent_successes = []
    actor_losses = []
    critic_losses = []
    
    # Randomly select 10 episodes to track step-by-step rewards and trajectories
    random.seed(42)  # For reproducibility
    sampled_episodes = sorted(random.sample(range(num_episodes), min(10, num_episodes)))
    step_rewards_tracking = {ep: [] for ep in sampled_episodes}
    trajectories_tracking = {ep: [] for ep in sampled_episodes}  # Store (x, y) positions
    target_positions = {ep: None for ep in sampled_episodes}  # Store target positions
    reached_targets = {ep: False for ep in sampled_episodes}  # Track if reached target
    
    print("Starting training...")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    print(f"Sampled episodes for step-by-step reward analysis: {sampled_episodes}")
    
    for episode in tqdm(range(num_episodes), desc="Training progress"):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        reached_target = False
        
        # Track initial position and target for sampled episodes
        if episode in sampled_episodes:
            trajectories_tracking[episode].append(env.drone_pos.copy())
            target_positions[episode] = env.target_pos.copy()
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Store reward
            is_terminal = terminated or truncated
            agent.store_reward(reward, is_terminal)
            
            # Track step-by-step reward and trajectory for sampled episodes
            if episode in sampled_episodes:
                step_rewards_tracking[episode].append(reward)
                trajectories_tracking[episode].append(env.drone_pos.copy())
            
            episode_reward += reward
            episode_length += 1
            
            if info.get('reached_target', False):
                reached_target = True
                if episode in sampled_episodes:
                    reached_targets[episode] = True
            
            if terminated or truncated:
                break
            
            state = next_state
        
        # Record statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        recent_successes.append(1 if reached_target else 0)
        
        # Calculate success rate for last 100 episodes
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
                print(f"  Success rate: {success_rate:.1f}%")
                print(f"  Average steps: {np.mean(episode_lengths[-update_frequency:]):.1f}")
                print(f"  Actor loss: {loss_info['actor_loss']:.4f}")
                print(f"  Critic loss: {loss_info['critic_loss']:.4f}")
                print(f"  Total loss: {loss_info['loss']:.4f}")
    
    # Plot training curves
    plot_training_curves(episode_rewards, episode_lengths, success_rates, plot_dir)
    # Plot separate reward curve
    plot_reward_curve(episode_rewards, plot_dir)
    # Plot loss curves
    if len(actor_losses) > 0:
        plot_loss_curves(actor_losses, critic_losses, plot_dir)
    # Plot step-by-step rewards and trajectories for sampled episodes
    plot_step_rewards(step_rewards_tracking, episode_rewards, sampled_episodes, plot_dir)
    plot_trajectories(trajectories_tracking, target_positions, reached_targets, sampled_episodes, plot_dir, env.world_size)
    
    print("\nTraining completed!")
    print(f"Final success rate: {success_rates[-1]:.1f}%")
    print(f"Final average reward: {np.mean(episode_rewards[-100:]):.2f}")
    if len(actor_losses) > 0:
        print(f"Final actor loss: {np.mean(actor_losses[-10:]):.4f}")
        print(f"Final critic loss: {np.mean(critic_losses[-10:]):.4f}")


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


def plot_step_rewards(step_rewards_tracking, episode_rewards, sampled_episodes, plot_dir):
    """
    Plot step-by-step rewards for sampled episodes
    
    Args:
        step_rewards_tracking: dict mapping episode number to list of step rewards
        episode_rewards: list of total episode rewards
        sampled_episodes: list of sampled episode numbers
        plot_dir: directory to save plots
    """
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


def plot_trajectories(trajectories_tracking, target_positions, reached_targets, sampled_episodes, plot_dir, world_size):
    """
    Plot trajectories for sampled episodes
    
    Args:
        trajectories_tracking: dict mapping episode number to list of (x, y) positions
        target_positions: dict mapping episode number to target (x, y) position
        reached_targets: dict mapping episode number to whether target was reached
        sampled_episodes: list of sampled episode numbers
        plot_dir: directory to save plots
        world_size: size of the world (for axis limits)
    """
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
        reached = reached_targets[episode_num]
        
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
        ax.set_title(f'Episode {episode_num} - {status}\nFinal Distance: {final_distance:.1f}m', 
                    fontsize=12, fontweight='bold')
        
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
        reached = reached_targets[episode_num]
        
        if len(trajectory) == 0 or target_pos is None:
            continue
        
        trajectory = np.array(trajectory)
        
        # Plot trajectory with different style based on success
        linestyle = '-' if reached else '--'
        linewidth = 2 if reached else 1.5
        alpha = 0.8 if reached else 0.5
        
        ax.plot(trajectory[:, 0], trajectory[:, 1], alpha=alpha, linewidth=linewidth, 
               color=colors[idx], linestyle=linestyle, 
               label=f'Ep {episode_num} {"✓" if reached else "✗"}', marker='o', markersize=2)
        
        # Plot start position
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'o', markersize=8, 
               color=colors[idx], markeredgecolor='black', markeredgewidth=1)
        
        # Plot end position
        marker = 's' if reached else 'x'
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], marker=marker, markersize=10, 
               color=colors[idx], markeredgecolor='black', markeredgewidth=1.5)
        
        # Plot target position
        ax.plot(target_pos[0], target_pos[1], '*', markersize=15, 
               color=colors[idx], markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlim([0, world_size])
    ax.set_ylim([0, world_size])
    ax.set_xlabel('X Coordinate (m)', fontsize=12)
    ax.set_ylabel('Y Coordinate (m)', fontsize=12)
    ax.set_title('All Sampled Episodes Trajectories', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Add statistics
    success_count = sum(1 for ep in sampled_episodes if reached_targets[ep])
    stats_text = f"Success Rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)\n"
    stats_text += "Legend: ✓ = Success, ✗ = Failed\n"
    stats_text += "Markers: o = Start, s/x = End, * = Target"
    
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
        reached = reached_targets[episode_num]
        
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
        num_episodes=2000,
        max_steps=60,  # 60s / 1s = 60 steps
        update_frequency=20
    )

