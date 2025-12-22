"""
Test script: Test trained model
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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


def test(model_path, num_episodes=10, render=True):
    """
    Test trained model
    
    Args:
        model_path: Path to model file
        num_episodes: Number of test episodes
        render: Whether to visualize
    """
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Create environment
    env = DronePathPlanningEnv(world_size=100, max_steps=500)
    state_dim = env.observation_space.shape[0]
    # Continuous action space: action_dim is the dimension of action space (vx, vy = 2)
    action_dim = env.action_space.shape[0]
    
    # Create PPO agent and load model
    agent = PPO(state_dim=state_dim, action_dim=action_dim)
    agent.load(model_path)
    
    success_count = 0
    total_rewards = []
    total_steps = []
    
    print(f"Starting test with {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        trajectory = [env.drone_pos.copy()]
        reached_target = False
        
        for step in range(500):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            trajectory.append(env.drone_pos.copy())
            episode_reward += reward
            episode_steps += 1
            
            if info.get('reached_target', False):
                reached_target = True
                success_count += 1
            
            if terminated or truncated:
                break
            
            state = next_state
        
        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)
        
        print(f"\nEpisode {episode + 1}:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Steps: {episode_steps}")
        print(f"  Reached target: {'Yes' if reached_target else 'No'}")
        print(f"  Final distance: {np.linalg.norm(env.target_pos - env.drone_pos):.2f}")
        
        # Visualize trajectory
        if render:
            plot_trajectory(trajectory, env.target_pos, episode + 1, reached_target)
    
    # Print statistics
    print("\n" + "="*50)
    print("Test Results:")
    print(f"  Success rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    print(f"  Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Average steps: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")


def plot_trajectory(trajectory, target_pos, episode, reached_target):
    """Plot trajectory"""
    trajectory = np.array(trajectory)
    
    plt.figure(figsize=(8, 8))
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Trajectory', alpha=0.7)
    plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
    plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End')
    plt.plot(target_pos[0], target_pos[1], 'r*', markersize=15, label='Target')
    
    # Draw target area
    circle = plt.Circle(target_pos, 2.0, color='r', alpha=0.2, label='Target Area')
    plt.gca().add_patch(circle)
    
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    status = "Success" if reached_target else "Failed"
    plt.title(f'Episode {episode} - {status}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'plots/test_episode_{episode}.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "models/ppo_model_final.pth"
    
    test(model_path, num_episodes=10, render=True)

