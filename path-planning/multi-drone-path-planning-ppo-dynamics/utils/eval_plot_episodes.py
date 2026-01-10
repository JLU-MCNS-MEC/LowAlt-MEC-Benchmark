import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.environment import MultiDronePathPlanningEnv
from core.ppo_agent import PPO


def run_and_plot(num_episodes=10, model_path='models/ppo_model_final.pth', out_dir='plots/eval'):
    os.makedirs(out_dir, exist_ok=True)

    env = DronePathPlanningEnv(world_size=1000, max_steps=60)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ppo = PPO(state_dim, action_dim)
    model_loaded = False
    if os.path.exists(model_path):
        try:
            ppo.load(model_path)
            model_loaded = True
            print(f"Loaded policy from {model_path}")
        except Exception as e:
            print(f"Failed to load model {model_path}: {e}")
            model_loaded = False
    else:
        print(f"Model file not found at {model_path}, using random policy.")

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        truncated = False
        traj = []
        rewards = []

        step_idx = 0
        while not done and not truncated and step_idx < env.max_steps:
            # get action
            if model_loaded:
                state_tensor = torch.FloatTensor(obs).to(ppo.device)
                with torch.no_grad():
                    action, _, _ = ppo.policy_old.act(state_tensor)
            else:
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)

            # record position and reward
            traj.append(env.drone_pos.copy())
            rewards.append(float(reward))

            step_idx += 1

        total_reward = sum(rewards)
        print(f"Episode {ep}: steps={len(rewards)}, total_reward={total_reward:.4f}")
        for i, r in enumerate(rewards):
            print(f"  step {i}: reward={r:.6f}")

        # Convert trajectory to array
        traj = np.array(traj)

        # Plot trajectory + rewards
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: trajectory
        ax = axes[0]
        if traj.size > 0:
            ax.plot(traj[:, 0], traj[:, 1], '-o', markersize=3, linewidth=1)
            ax.scatter(traj[0, 0], traj[0, 1], c='green', label='start')
            ax.scatter(env.target_pos[0], env.target_pos[1], c='red', label='target')
        ax.set_title(f'Trajectory Episode {ep}')
        ax.set_xlim(0, env.world_size)
        ax.set_ylim(0, env.world_size)
        ax.set_aspect('equal')
        ax.legend()

        # Right: per-step rewards
        ax2 = axes[1]
        ax2.plot(np.arange(len(rewards)), rewards, '-o')
        ax2.set_title('Per-step reward')
        ax2.set_xlabel('step')
        ax2.set_ylabel('reward')

        plt.tight_layout()
        out_path = os.path.join(out_dir, f'eval_episode_{ep}.png')
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved plot to {out_path}\n")


if __name__ == '__main__':
    run_and_plot()
