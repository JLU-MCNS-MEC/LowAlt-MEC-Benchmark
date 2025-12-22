"""
Training script: Train drone path planning using PPO algorithm
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

from environment import DronePathPlanningEnv
from ppo_agent import PPO


def train(
    num_episodes=2000,
    max_steps=500,
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
    env = DronePathPlanningEnv(world_size=100, max_steps=max_steps)
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
    
    print("Starting training...")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    for episode in tqdm(range(num_episodes), desc="Training progress"):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        reached_target = False
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Store reward
            is_terminal = terminated or truncated
            agent.store_reward(reward, is_terminal)
            
            episode_reward += reward
            episode_length += 1
            
            if info.get('reached_target', False):
                reached_target = True
            
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


if __name__ == "__main__":
    train(
        num_episodes=2000,
        max_steps=500,
        update_frequency=20
    )

