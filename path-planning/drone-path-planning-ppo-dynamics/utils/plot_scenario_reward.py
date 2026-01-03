"""
绘制场景中从起点到目标点的reward曲线图
可以用于分析单个场景/episode的reward变化
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.environment import DronePathPlanningEnv
from core.ppo_agent import PPO
import torch


def plot_scenario_reward(episode_num=None, start_pos=None, target_pos=None, 
                        model_path=None, max_steps=60, plot_dir='plots', 
                        world_size=1000):
    """
    绘制一个场景/episode的reward曲线图
    
    Args:
        episode_num: Episode编号（用于标题）
        start_pos: 起点位置 [x, y]，如果None则随机生成
        target_pos: 目标位置 [x, y]，如果None则随机生成
        model_path: 模型路径，如果提供则加载模型进行测试
        max_steps: 最大步数
        plot_dir: 保存图片的目录
        world_size: 世界大小
    """
    # 创建环境
    env = DronePathPlanningEnv(
        world_size=world_size,
        max_steps=max_steps,
        fixed_start_pos=start_pos,
        fixed_target_pos=target_pos
    )
    
    # 如果提供了模型路径，加载模型
    agent = None
    if model_path and os.path.exists(model_path):
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = PPO(state_dim, action_dim)
        agent.load(model_path)
        print(f"Loaded model from: {model_path}")
    else:
        # 使用随机策略（用于演示）
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = PPO(state_dim, action_dim)
        print("Using random policy (no model loaded)")
    
    # 重置环境
    state, _ = env.reset()
    
    # 记录数据
    step_rewards = []
    cumulative_rewards = []
    distances = []
    positions = []
    step_numbers = []
    
    # 记录初始信息
    initial_pos = env.drone_pos.copy()
    target_pos = env.target_pos.copy()
    initial_distance = np.linalg.norm(target_pos - initial_pos)
    
    print(f"\nScenario Information:")
    print(f"  Start position: ({initial_pos[0]:.1f}, {initial_pos[1]:.1f})")
    print(f"  Target position: ({target_pos[0]:.1f}, {target_pos[1]:.1f})")
    print(f"  Initial distance: {initial_distance:.1f}m")
    print(f"  Max steps: {max_steps}")
    print(f"\nRunning episode...")
    
    # 运行episode
    total_reward = 0
    reached_target = False
    
    for step in range(max_steps):
        # 选择动作
        action = agent.select_action(state)
        
        # 执行动作
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # 记录数据
        step_rewards.append(reward)
        total_reward += reward
        cumulative_rewards.append(total_reward)
        distances.append(info['distance'])
        positions.append(env.drone_pos.copy())
        step_numbers.append(step + 1)
        
        # 检查是否到达目标
        if info.get('reached_target', False):
            reached_target = True
            print(f"  Step {step + 1}: Reached target! Distance: {info['distance']:.2f}m")
            break
        
        if terminated or truncated:
            break
        
        state = next_state
    
    # 计算最终统计
    final_distance = distances[-1] if len(distances) > 0 else initial_distance
    final_pos = positions[-1] if len(positions) > 0 else initial_pos
    
    print(f"\nEpisode Summary:")
    print(f"  Total steps: {len(step_rewards)}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final distance: {final_distance:.2f}m")
    print(f"  Final position: ({final_pos[0]:.1f}, {final_pos[1]:.1f})")
    print(f"  Reached target: {reached_target}")
    
    # 创建图表
    os.makedirs(plot_dir, exist_ok=True)
    
    # 创建主图：包含多个子图
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Step Reward曲线
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(step_numbers, step_rewards, 'o-', linewidth=2, markersize=4, 
            color='#2E86AB', label='Step Reward', alpha=0.8)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Step Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Step Reward Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # 添加统计信息
    mean_reward = np.mean(step_rewards)
    max_reward = np.max(step_rewards)
    min_reward = np.min(step_rewards)
    ax1.axhline(y=mean_reward, color='red', linestyle=':', alpha=0.7, 
                label=f'Mean: {mean_reward:.3f}')
    
    # 2. Cumulative Reward曲线
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(step_numbers, cumulative_rewards, '-', linewidth=2.5, 
            color='#A23B72', label='Cumulative Reward')
    ax2.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Reward', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Reward Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # 添加最终reward标记
    ax2.plot(len(step_numbers), total_reward, 'ro', markersize=10, 
            label=f'Total: {total_reward:.2f}')
    
    # 3. Distance变化曲线
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(step_numbers, distances, '-', linewidth=2, color='#F18F01', 
            label='Distance to Target')
    ax3.axhline(y=env.arrival_threshold, color='green', linestyle='--', 
                alpha=0.7, label=f'Arrival Threshold ({env.arrival_threshold}m)')
    ax3.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Distance (m)', fontsize=12, fontweight='bold')
    ax3.set_title('Distance to Target Over Time', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.invert_yaxis()  # 距离越小越好，所以反转y轴
    
    # 4. Reward和Distance对比（双y轴）
    ax4 = fig.add_subplot(gs[1, 1])
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(step_numbers, step_rewards, 'o-', linewidth=2, 
                    markersize=4, color='#2E86AB', label='Step Reward', alpha=0.8)
    line2 = ax4_twin.plot(step_numbers, distances, '-', linewidth=2, 
                         color='#F18F01', label='Distance', alpha=0.8)
    
    ax4.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Step Reward', fontsize=12, fontweight='bold', color='#2E86AB')
    ax4_twin.set_ylabel('Distance (m)', fontsize=12, fontweight='bold', color='#F18F01')
    ax4.set_title('Reward vs Distance', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='y', labelcolor='#2E86AB')
    ax4_twin.tick_params(axis='y', labelcolor='#F18F01')
    ax4_twin.invert_yaxis()
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper right', fontsize=10)
    
    # 5. 轨迹图
    ax5 = fig.add_subplot(gs[2, :])
    positions_array = np.array(positions)
    
    # 绘制轨迹
    ax5.plot(positions_array[:, 0], positions_array[:, 1], 'o-', 
            linewidth=2, markersize=4, color='#2E86AB', alpha=0.7, label='Trajectory')
    
    # 标记起点
    ax5.plot(initial_pos[0], initial_pos[1], 'go', markersize=15, 
            markeredgecolor='black', markeredgewidth=2, label='Start', zorder=5)
    
    # 标记终点
    ax5.plot(final_pos[0], final_pos[1], 'ro', markersize=15, 
            markeredgecolor='black', markeredgewidth=2, label='End', zorder=5)
    
    # 标记目标
    ax5.plot(target_pos[0], target_pos[1], 'r*', markersize=20, 
            markeredgecolor='black', markeredgewidth=1, label='Target', zorder=5)
    
    # 绘制目标区域
    circle = plt.Circle(target_pos, env.arrival_threshold, color='r', 
                       alpha=0.2, label='Target Area')
    ax5.add_patch(circle)
    
    # 添加距离标注
    for i in [0, len(positions)//2, len(positions)-1]:
        if i < len(positions):
            ax5.annotate(f'Step {i+1}\n{distances[i]:.1f}m', 
                        xy=(positions[i][0], positions[i][1]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=8, bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='yellow', alpha=0.7))
    
    ax5.set_xlim([0, world_size])
    ax5.set_ylim([0, world_size])
    ax5.set_xlabel('X Coordinate (m)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Y Coordinate (m)', fontsize=12, fontweight='bold')
    ax5.set_title('Trajectory Visualization', fontsize=14, fontweight='bold')
    ax5.legend(loc='best', fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal', adjustable='box')
    
    # 添加总体信息
    title = f'Scenario Reward Analysis'
    if episode_num is not None:
        title += f' - Episode {episode_num}'
    title += f'\nStart: ({initial_pos[0]:.1f}, {initial_pos[1]:.1f}) | '
    title += f'Target: ({target_pos[0]:.1f}, {target_pos[1]:.1f}) | '
    title += f'Distance: {initial_distance:.1f}m | '
    title += f'Status: {"SUCCESS" if reached_target else "FAILED"}'
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    # 保存图片
    filename = f'scenario_reward_analysis'
    if episode_num is not None:
        filename += f'_ep{episode_num}'
    filename += '.png'
    filepath = os.path.join(plot_dir, filename)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {filepath}")
    
    # 打印详细统计
    print("\n" + "="*60)
    print("Reward Statistics")
    print("="*60)
    print(f"Total steps: {len(step_rewards)}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Mean step reward: {mean_reward:.3f}")
    print(f"Std step reward: {np.std(step_rewards):.3f}")
    print(f"Min step reward: {min_reward:.3f}")
    print(f"Max step reward: {max_reward:.3f}")
    print(f"Positive rewards: {sum(1 for r in step_rewards if r > 0)}/{len(step_rewards)}")
    print(f"Negative rewards: {sum(1 for r in step_rewards if r < 0)}/{len(step_rewards)}")
    print(f"Initial distance: {initial_distance:.2f}m")
    print(f"Final distance: {final_distance:.2f}m")
    print(f"Distance improvement: {initial_distance - final_distance:.2f}m")
    print(f"Reached target: {reached_target}")
    print("="*60)
    
    return fig, {
        'step_rewards': step_rewards,
        'cumulative_rewards': cumulative_rewards,
        'distances': distances,
        'positions': positions,
        'total_reward': total_reward,
        'reached_target': reached_target
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot reward curve for a scenario')
    parser.add_argument('--episode', type=int, default=None, 
                       help='Episode number (for title)')
    parser.add_argument('--start', type=float, nargs=2, default=None,
                       help='Start position [x, y]')
    parser.add_argument('--target', type=float, nargs=2, default=None,
                       help='Target position [x, y]')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (.pth)')
    parser.add_argument('--max-steps', type=int, default=60,
                       help='Maximum steps per episode')
    parser.add_argument('--plot-dir', type=str, default='plots',
                       help='Directory to save plots')
    parser.add_argument('--world-size', type=float, default=1000,
                       help='World size')
    
    args = parser.parse_args()
    
    # 运行分析
    plot_scenario_reward(
        episode_num=args.episode,
        start_pos=args.start,
        target_pos=args.target,
        model_path=args.model,
        max_steps=args.max_steps,
        plot_dir=args.plot_dir,
        world_size=args.world_size
    )
    
    plt.show()

