"""
Test script: Test trained multi-drone model
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

from environment import MultiDronePathPlanningEnv
from ppo_agent import PPO


def test(model_path, num_drones=3, num_episodes=10, render=True, num_plots=5):
    """
    Test trained multi-drone model
    
    Args:
        model_path: Path to model file
        num_drones: Number of drones (must be between 1 and 9, inclusive)
        num_episodes: Number of test episodes
        render: Whether to visualize
        num_plots: Number of random episodes to plot (default: 5)
    """
    # Validate num_drones
    if not isinstance(num_drones, int) or num_drones < 1 or num_drones > 9:
        raise ValueError(f"num_drones must be an integer between 1 and 9 (inclusive), got {num_drones}")
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Create environment with random positions
    env = MultiDronePathPlanningEnv(
        num_drones=num_drones,
        world_size=1000,
        max_steps=60,
        fixed_start_positions=None,
        fixed_target_positions=None
    )
    
    state_dim = env.observation_space.shape[1]  # 7D per drone
    action_dim = env.action_space.shape[1]  # 2D per drone
    
    # Create PPO agent and load model
    agent = PPO(state_dim=state_dim, action_dim=action_dim)
    agent.load(model_path)
    
    success_count = 0
    total_rewards = []
    total_steps = []
    individual_success_counts = [0] * num_drones
    
    # Randomly select episodes to plot
    if render and num_episodes > num_plots:
        episodes_to_plot = sorted(np.random.choice(num_episodes, size=num_plots, replace=False))
        print(f"Starting test with {num_episodes} episodes ({num_drones} drones)...")
        print(f"Will randomly plot {num_plots} episodes: {[ep+1 for ep in episodes_to_plot]}")
    else:
        episodes_to_plot = list(range(min(num_episodes, num_plots)))
        print(f"Starting test with {num_episodes} episodes ({num_drones} drones)...")
        if render:
            print(f"Will plot all {len(episodes_to_plot)} episodes")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        trajectories = [[] for _ in range(num_drones)]
        all_reached = False
        
        # Record initial positions
        for i in range(num_drones):
            trajectories[i].append(env.drone_positions[i].copy())
        
        for step in range(60):
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
                reached = individual_distances[i] < env.arrival_threshold
                print(f"  Drone {i+1}: distance={individual_distances[i]:.2f}m, reached={'Yes' if reached else 'No'}")
        
        # Visualize trajectories (only for selected episodes)
        if render and episode in episodes_to_plot:
            plot_trajectories(trajectories, env.target_positions, episode + 1, all_reached, num_drones)
    
    # Print statistics
    print("\n" + "="*50)
    print("Test Results:")
    print(f"  All drones success rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    print(f"  Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Average steps: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
    print(f"\n  Individual drone success rates:")
    for i in range(num_drones):
        print(f"    Drone {i+1}: {individual_success_counts[i]}/{num_episodes} ({individual_success_counts[i]/num_episodes*100:.1f}%)")


def plot_trajectories(trajectories, target_positions, episode, all_reached, num_drones):
    """Plot trajectories for all drones"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Color map for different drones
    colors = plt.cm.tab10(np.linspace(0, 1, num_drones))
    
    for i in range(num_drones):
        if len(trajectories[i]) == 0:
            continue
        
        trajectory = np.array(trajectories[i])
        target_pos = target_positions[i]
        
        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], '-', linewidth=2, 
               color=colors[i], label=f'Drone {i+1}', alpha=0.7, marker='o', markersize=3)
        
        # Plot start position
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'o', markersize=10, 
               color=colors[i], markeredgecolor='black', markeredgewidth=1.5, label=f'Start {i+1}' if i == 0 else '')
        
        # Plot end position
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 's', markersize=10, 
               color=colors[i], markeredgecolor='black', markeredgewidth=1.5, label=f'End {i+1}' if i == 0 else '')
        
        # Plot target position
        ax.plot(target_pos[0], target_pos[1], '*', markersize=20, 
               color=colors[i], markeredgecolor='black', markeredgewidth=1, label=f'Target {i+1}' if i == 0 else '')
        
        # Draw target area circle
        circle = plt.Circle(target_pos, 20.0, color=colors[i], alpha=0.2)
        ax.add_patch(circle)
    
    ax.set_xlim([0, 1000])
    ax.set_ylim([0, 1000])
    ax.set_xlabel('X Coordinate (m)', fontsize=12)
    ax.set_ylabel('Y Coordinate (m)', fontsize=12)
    status = "Success (All Reached)" if all_reached else "Partial/Failed"
    ax.set_title(f'Episode {episode} - {status} ({num_drones} Drones)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(f'plots/test_episode_{episode}.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained multi-drone path planning model')
    parser.add_argument('--model_path', type=str, default='models/ppo_model_final.pth',
                       help='Path to model file (default: models/ppo_model_final.pth)')
    parser.add_argument('--num_drones', type=int, default=3,
                       help='Number of drones (must be between 1 and 9, default: 3)')
    parser.add_argument('--num_episodes', type=int, default=20,
                       help='Number of test episodes (default: 20)')
    parser.add_argument('--num_plots', type=int, default=5,
                       help='Number of random episodes to plot (default: 5)')
    parser.add_argument('--no_render', action='store_true',
                       help='Disable trajectory visualization')
    
    args = parser.parse_args()
    
    # Validate num_drones
    if args.num_drones < 1 or args.num_drones > 9:
        parser.error(f"num_drones must be between 1 and 9 (inclusive), got {args.num_drones}")
    
    test(args.model_path, num_drones=args.num_drones, 
         num_episodes=args.num_episodes, render=not args.no_render, 
         num_plots=args.num_plots)

