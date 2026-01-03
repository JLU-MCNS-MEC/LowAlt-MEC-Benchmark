"""
分析特定场景训练失败的原因
"""

import os
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.environment import DronePathPlanningEnv
from core.ppo_agent import PPO
import torch


def analyze_scenario(scenario_num, model_path=None, num_test_episodes=20, max_steps=60):
    """
    分析特定场景的训练失败原因
    
    Args:
        scenario_num: 场景编号
        model_path: 模型路径（可选）
        num_test_episodes: 测试episode数
        max_steps: 最大步数
    """
    # 读取场景信息（如果存在）
    summary_path = 'models/scenario_training_summary.txt'
    scenario_info = None
    
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 查找场景信息
            if f'Scenario {scenario_num}:' in content:
                print(f"Found scenario {scenario_num} in summary file")
                # 解析场景信息
                lines = content.split('\n')
                in_scenario = False
                for i, line in enumerate(lines):
                    if f'Scenario {scenario_num}:' in line:
                        in_scenario = True
                        scenario_info = {}
                    elif in_scenario:
                        if line.strip().startswith('Episode range:'):
                            parts = line.split(':')[1].strip().split(' - ')
                            scenario_info['start_ep'] = int(parts[0])
                            scenario_info['end_ep'] = int(parts[1]) if parts[1] != 'ongoing' else None
                        elif line.strip().startswith('Total episodes:'):
                            scenario_info['total_episodes'] = int(line.split(':')[1].strip())
                        elif line.strip().startswith('Start position:'):
                            pos_str = line.split(':')[1].strip()
                            pos_str = pos_str.replace('(', '').replace(')', '')
                            scenario_info['start_pos'] = [float(x) for x in pos_str.split(',')]
                        elif line.strip().startswith('Target position:'):
                            pos_str = line.split(':')[1].strip()
                            pos_str = pos_str.replace('(', '').replace(')', '')
                            scenario_info['target_pos'] = [float(x) for x in pos_str.split(',')]
                        elif line.strip().startswith('Distance:'):
                            scenario_info['distance'] = float(line.split(':')[1].strip().replace('m', ''))
                        elif line.strip().startswith('Final success rate:'):
                            scenario_info['success_rate'] = float(line.split(':')[1].strip().replace('%', ''))
                        elif line.strip() == '' and scenario_info:
                            break
    
    print(f"\n{'='*80}")
    print(f"Scenario {scenario_num} Analysis")
    print(f"{'='*80}\n")
    
    if scenario_info:
        print("Scenario Information:")
        print(f"  Start position: {scenario_info.get('start_pos', 'N/A')}")
        print(f"  Target position: {scenario_info.get('target_pos', 'N/A')}")
        print(f"  Distance: {scenario_info.get('distance', 'N/A'):.1f}m")
        print(f"  Total episodes: {scenario_info.get('total_episodes', 'N/A')}")
        print(f"  Final success rate: {scenario_info.get('success_rate', 'N/A'):.1f}%")
        print()
        
        # 分析场景特征
        start_pos = np.array(scenario_info['start_pos'])
        target_pos = np.array(scenario_info['target_pos'])
        distance = scenario_info['distance']
        
        # 检查是否在边缘
        world_size = 1000
        start_near_edge = (start_pos[0] < 50 or start_pos[0] > world_size - 50 or
                          start_pos[1] < 50 or start_pos[1] > world_size - 50)
        target_near_edge = (target_pos[0] < 50 or target_pos[0] > world_size - 50 or
                           target_pos[1] < 50 or target_pos[1] > world_size - 50)
        
        print("Scenario Characteristics:")
        print(f"  Distance: {distance:.1f}m")
        if distance > 800:
            print(f"  ⚠️  VERY LONG DISTANCE (>800m) - May be too difficult")
        elif distance > 600:
            print(f"  ⚠️  Long distance (>600m) - May be challenging")
        else:
            print(f"  ✓ Moderate distance - Should be manageable")
        
        print(f"  Start near edge: {start_near_edge}")
        print(f"  Target near edge: {target_near_edge}")
        if target_near_edge:
            print(f"  ⚠️  Target near edge - May cause issues")
        
        # 计算方向
        dx = target_pos[0] - start_pos[0]
        dy = target_pos[1] - start_pos[1]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        print(f"  Direction: {angle:.1f}° (dx={dx:.1f}, dy={dy:.1f})")
        
        # 检查是否是对角线方向（可能更难）
        if abs(abs(angle) - 45) < 10 or abs(abs(angle) - 135) < 10:
            print(f"  ⚠️  Diagonal direction - May be more challenging")
        
    else:
        print("⚠️  Scenario information not found in summary file")
        print("   Using default test scenario")
        start_pos = np.array([200.0, 200.0])
        target_pos = np.array([700.0, 700.0])
        distance = np.linalg.norm(target_pos - start_pos)
    
    # 测试场景
    print(f"\n{'='*80}")
    print("Testing Scenario Performance")
    print(f"{'='*80}\n")
    
    env = DronePathPlanningEnv(
        world_size=1000,
        max_steps=max_steps,
        fixed_start_pos=start_pos.tolist() if 'start_pos' in locals() else None,
        fixed_target_pos=target_pos.tolist() if 'target_pos' in locals() else None
    )
    
    # 加载模型（如果提供）
    agent = None
    if model_path and os.path.exists(model_path):
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = PPO(state_dim, action_dim)
        agent.load(model_path)
        print(f"Loaded model from: {model_path}")
    else:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = PPO(state_dim, action_dim)
        print("Using random policy (no model loaded)")
    
    # 运行测试episodes
    successes = 0
    episode_rewards = []
    episode_lengths = []
    final_distances = []
    
    for ep in range(num_test_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if info.get('reached_target', False):
                successes += 1
                break
            
            if terminated or truncated:
                break
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        final_distances.append(info['distance'])
    
    # 分析结果
    success_rate = successes / num_test_episodes * 100
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    avg_final_distance = np.mean(final_distances)
    min_final_distance = np.min(final_distances)
    
    print(f"Test Results ({num_test_episodes} episodes):")
    print(f"  Success rate: {success_rate:.1f}% ({successes}/{num_test_episodes})")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Average episode length: {avg_length:.1f} steps")
    print(f"  Average final distance: {avg_final_distance:.1f}m")
    print(f"  Minimum final distance: {min_final_distance:.1f}m")
    
    # 诊断问题
    print(f"\n{'='*80}")
    print("Problem Diagnosis")
    print(f"{'='*80}\n")
    
    issues = []
    recommendations = []
    
    if success_rate < 50:
        issues.append("Low success rate")
        if distance > 700:
            recommendations.append("Distance too large - Consider reducing difficulty increase")
        if target_near_edge:
            recommendations.append("Target near edge - May need better edge handling")
        if avg_final_distance > distance * 0.5:
            recommendations.append("Agent not making progress - May need stronger reward signal")
    
    if avg_reward < 0:
        issues.append("Negative average reward")
        recommendations.append("Reward function may be too harsh - Consider adjusting penalties")
    
    if min_final_distance > 100:
        issues.append("Never got close to target")
        recommendations.append("Initial direction may be wrong - Check direction reward")
        recommendations.append("May need stronger progress reward")
    
    if avg_length >= max_steps * 0.9:
        issues.append("Episodes often timeout")
        recommendations.append("Agent too slow - May need better speed control")
    
    if issues:
        print("Identified Issues:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
        print("\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("✓ No major issues identified")
    
    return {
        'scenario_info': scenario_info,
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_final_distance': avg_final_distance,
        'issues': issues,
        'recommendations': recommendations
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze scenario training failure')
    parser.add_argument('--scenario', type=int, required=True,
                       help='Scenario number to analyze')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Number of test episodes')
    
    args = parser.parse_args()
    
    analyze_scenario(
        scenario_num=args.scenario,
        model_path=args.model,
        num_test_episodes=args.episodes
    )

