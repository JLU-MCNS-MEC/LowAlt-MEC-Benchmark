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

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.environment import MultiDronePathPlanningEnv
from core.ppo_agent import PPO


def test(model_path, num_drones=3, num_episodes=10, render=True):
    """
    Test trained model for multi-drone path planning
    
    Args:
        model_path: Path to model file
        num_drones: Number of drones (must be between 1 and 9, inclusive)
        num_episodes: Number of test episodes
        render: Whether to visualize
    """
    # Validate num_drones
    if not isinstance(num_drones, int) or num_drones < 1 or num_drones > 9:
        raise ValueError(f"num_drones must be an integer between 1 and 9 (inclusive), got {num_drones}")
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Create environment with random positions (no fixed positions)
    env = MultiDronePathPlanningEnv(
        num_drones=num_drones,
        world_size=1000, 
        max_steps=600,  # 60s / 0.1s = 600 steps (dt=0.1 for finer control)
        fixed_start_positions=None,  # Use random start positions
        fixed_target_positions=None  # Use random target positions
    )
    # State and action dimensions (per drone)
    state_dim = env.observation_space.shape[1]  # 17D per drone
    action_dim = env.action_space.shape[1]  # 4D per drone
    
    # Create PPO agent and load model
    agent = PPO(state_dim=state_dim, action_dim=action_dim)
    agent.load(model_path)
    
    success_count = 0
    total_rewards = []
    total_steps = []
    individual_success_counts = [0] * num_drones
    
    print(f"Starting test with {num_episodes} episodes, {num_drones} drones...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        trajectories = [[] for _ in range(num_drones)]
        all_reached = False
        
        # Record initial positions
        for i in range(num_drones):
            trajectories[i].append(env.drone_positions[i].copy())
        
        for step in range(env.max_steps):
            # Select actions for all drones
            actions = []
            for i in range(num_drones):
                action = agent.select_action(state[i])
                actions.append(action)
            actions = np.array(actions)
            
            next_state, reward, terminated, truncated, info = env.step(actions)
            
            # Record trajectories
            for i in range(num_drones):
                trajectories[i].append(env.drone_positions[i].copy())
            
            episode_reward += reward
            episode_steps += 1
            
            if info.get('all_reached', False):
                all_reached = True
                success_count += 1
            
            # Track individual drone successes
            individual_distances = info.get('individual_distances', [])
            for i in range(num_drones):
                if i < len(individual_distances) and individual_distances[i] < env.arrival_threshold:
                    individual_success_counts[i] += 1
            
            if terminated or truncated:
                break
            
            state = next_state
        
        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)
        
        print(f"\nEpisode {episode + 1}:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Steps: {episode_steps}")
        print(f"  All drones reached: {'Yes' if all_reached else 'No'}")
        individual_distances = info.get('individual_distances', [])
        for i in range(num_drones):
            if i < len(individual_distances):
                print(f"  Drone {i+1}: Distance {individual_distances[i]:.2f}m")
        
        # Visualize trajectories
        if render:
            plot_trajectories(trajectories, env.target_positions, episode + 1, all_reached, num_drones)
    
    # Print statistics
    print("\n" + "="*50)
    print("Test Results:")
    print(f"  Success rate (all reached): {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    for i in range(num_drones):
        print(f"  Drone {i+1} success rate: {individual_success_counts[i]}/{num_episodes} ({individual_success_counts[i]/num_episodes*100:.1f}%)")
    print(f"  Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Average steps: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")


def plot_trajectories(trajectories, target_positions, episode, all_reached, num_drones):
    """Plot trajectories for all drones"""
    colors = plt.cm.tab10(np.linspace(0, 1, num_drones))
    
    plt.figure(figsize=(10, 10))
    
    for i in range(num_drones):
        trajectory = np.array(trajectories[i])
        color = colors[i]
        
        plt.plot(trajectory[:, 0], trajectory[:, 1], '-', linewidth=2, 
                label=f'Drone {i+1}', alpha=0.7, color=color)
        plt.plot(trajectory[0, 0], trajectory[0, 1], 'o', markersize=8, 
                color=color, markeredgecolor='black', markeredgewidth=1)
        plt.plot(trajectory[-1, 0], trajectory[-1, 1], 's', markersize=8, 
                color=color, markeredgecolor='black', markeredgewidth=1)
        plt.plot(target_positions[i, 0], target_positions[i, 1], '*', 
                markersize=12, color=color, markeredgecolor='black', markeredgewidth=0.5)
        
        # Draw target area
        circle = plt.Circle(target_positions[i], 20.0, color=color, 
                          alpha=0.15, linestyle='--', linewidth=1)
        plt.gca().add_patch(circle)
    
    plt.xlim([0, 1000])
    plt.ylim([0, 1000])
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    status = "Success" if all_reached else "Failed"
    plt.title(f'Episode {episode} - {status} ({num_drones} drones)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, loc='upper right')
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
    
    num_drones = 3  # Number of drones (must be between 1 and 9, inclusive)
    if len(sys.argv) > 2:
        num_drones = int(sys.argv[2])
    
    test(model_path, num_drones=num_drones, num_episodes=10, render=True)

