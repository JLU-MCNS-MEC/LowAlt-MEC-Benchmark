"""
PPO算法实现（支持连续动作空间）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


class ActorCritic(nn.Module):
    """
    Actor-Critic网络（连续动作空间）
    Actor: 输出动作的均值和标准差（用于Normal分布）
    Critic: 输出状态价值
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):  # Increased from 128
        super(ActorCritic, self).__init__()
        
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # 共享特征提取层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor头：输出动作均值
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Actor头：输出动作标准差的对数
        self.actor_logstd = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic头：输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """
        前向传播
        
        Args:
            state: [batch_size, state_dim]
        
        Returns:
            action_mean: [batch_size, action_dim]
            action_logstd: [batch_size, action_dim]
            state_value: [batch_size, 1]
        """
        features = self.feature(state)
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd(features)
        # 限制log_std的范围
        action_logstd = torch.clamp(action_logstd, self.log_std_min, self.log_std_max)
        state_value = self.critic(features)
        return action_mean, action_logstd, state_value
    
    def act(self, state):
        """
        根据状态选择动作
        
        Args:
            state: [state_dim] or [batch_size, state_dim]
        
        Returns:
            action: [action_dim] or [batch_size, action_dim] 选择的动作
            action_logprob: 动作的对数概率
            state_value: 状态价值
        """
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        action_mean, action_logstd, state_value = self.forward(state)
        action_std = torch.exp(action_logstd)
        
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)  # 对动作维度求和
        
        # 如果是单样本，返回numpy数组
        if action.shape[0] == 1:
            return action.squeeze(0).cpu().numpy(), action_logprob, state_value
        else:
            return action.cpu().numpy(), action_logprob, state_value
    
    def evaluate(self, state, action):
        """
        评估状态和动作
        
        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
        
        Returns:
            action_logprob: [batch_size]
            state_value: [batch_size, 1]
            dist_entropy: 分布熵
        """
        action_mean, action_logstd, state_value = self.forward(state)
        action_std = torch.exp(action_logstd)
        
        dist = Normal(action_mean, action_std)
        action_logprob = dist.log_prob(action).sum(dim=-1)  # 对动作维度求和
        dist_entropy = dist.entropy().sum(dim=-1)  # 对动作维度求和
        
        return action_logprob, state_value, dist_entropy


class PPO:
    """
    PPO算法实现
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        eps_clip=0.2,
        k_epochs=10,
        device=None
    ):
        """
        初始化PPO
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度（连续动作空间为2，表示vx, vy）
            lr_actor: Actor学习率
            lr_critic: Critic学习率
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
            eps_clip: PPO裁剪参数
            k_epochs: 每次更新的epoch数
            device: 设备（CPU或GPU）
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 使用不同的学习率
        self.optimizer = optim.Adam([
            {'params': self.policy.actor_mean.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor_logstd.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
            {'params': self.policy.feature.parameters(), 'lr': lr_actor}
        ])
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        # 经验缓冲区
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'is_terminals': [],
            'logprobs': []
        }
    
    def select_action(self, state):
        """
        选择动作
        
        Args:
            state: numpy array, 状态
        
        Returns:
            action: numpy array, 选择的动作 [vx, vy]
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            action, action_logprob, state_value = self.policy_old.act(state_tensor)
        
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        # action_logprob可能是tensor或标量
        if isinstance(action_logprob, torch.Tensor):
            self.buffer['logprobs'].append(action_logprob.item())
        else:
            self.buffer['logprobs'].append(action_logprob)
        
        return action
    
    def store_reward(self, reward, is_terminal):
        """存储奖励和终止标志"""
        self.buffer['rewards'].append(reward)
        self.buffer['is_terminals'].append(is_terminal)
    
    def update(self):
        """
        更新策略（使用GAE计算优势函数）
        
        Returns:
            loss_info: 损失信息字典
        """
        # 转换为tensor
        old_states = torch.FloatTensor(np.array(self.buffer['states'])).to(self.device)
        old_actions = torch.FloatTensor(np.array(self.buffer['actions'])).to(self.device)
        old_logprobs = torch.FloatTensor(self.buffer['logprobs']).to(self.device)
        rewards = torch.FloatTensor(self.buffer['rewards']).to(self.device)
        dones = torch.FloatTensor(self.buffer['is_terminals']).to(self.device)
        
        # 计算旧策略的价值估计
        with torch.no_grad():
            _, old_state_values, _ = self.policy_old.evaluate(old_states, old_actions)
            old_state_values = old_state_values.squeeze()
        
        # 计算下一个状态的价值（用于TD误差）
        next_values = torch.zeros_like(old_state_values)
        for i in range(len(old_state_values) - 1):
            if not dones[i]:
                next_values[i] = old_state_values[i + 1]
        
        # 计算TD误差: δ_t = r_t + γV(s_{t+1}) - V(s_t)
        td_errors = rewards + self.gamma * next_values * (1 - dones) - old_state_values
        
        # 使用GAE计算优势函数: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        advantages = torch.zeros_like(td_errors)
        last_gae = 0.0
        for t in reversed(range(len(td_errors))):
            if dones[t]:
                last_gae = td_errors[t]
            else:
                last_gae = td_errors[t] + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae
        
        # 计算returns（用于critic损失）
        returns = advantages + old_state_values
        
        # 归一化优势函数（而不是奖励）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 更新策略
        total_loss = 0
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for _ in range(self.k_epochs):
            # 评估当前策略
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = state_values.squeeze()
            
            # 计算比率
            ratios = torch.exp(logprobs - old_logprobs)
            
            # 计算裁剪的surrogate损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic损失（使用returns而不是rewards）
            critic_loss = nn.MSELoss()(state_values, returns.detach())
            
            # 总损失（调整权重）
            loss = actor_loss + 0.5 * critic_loss - 0.02 * dist_entropy.mean()
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            # 累计统计
            total_loss += loss.item()
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += dist_entropy.mean().item()
        
        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 清空缓冲区
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'is_terminals': [],
            'logprobs': []
        }
        
        # 返回统计信息
        loss_info = {
            'loss': total_loss / self.k_epochs,
            'actor_loss': total_actor_loss / self.k_epochs,
            'critic_loss': total_critic_loss / self.k_epochs,
            'entropy': total_entropy / self.k_epochs
        }
        
        return loss_info
    
    def save(self, filepath):
        """保存模型"""
        torch.save(self.policy.state_dict(), filepath)
        print(f"模型已保存到: {filepath}")
    
    def load(self, filepath):
        """加载模型"""
        self.policy.load_state_dict(torch.load(filepath, map_location=self.device))
        self.policy_old.load_state_dict(self.policy.state_dict())
        print(f"模型已从 {filepath} 加载")

