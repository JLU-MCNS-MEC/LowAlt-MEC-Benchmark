"""
PPO算法实现（支持连续动作空间）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


def orthogonal_init(layer, gain=1.0):
    """Orthogonal initialization for better exploration"""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0)


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
        
        # Actor头：输出动作均值（使用tanh限制输出范围到[-1, 1]，然后乘以max_speed）
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Limit output to [-1, 1] range
        )
        
        # Actor头：输出动作标准差的对数
        # Use learnable parameter instead of network output for better initialization
        # Initialize to -0.5 so std ≈ 0.6, providing more exploration initially
        # This helps agent explore more in early training
        self.actor_logstd = nn.Parameter(torch.full((action_dim,), -0.5))
        
        # Critic头：输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize network weights for better exploration
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization"""
        # Initialize feature layers with standard gain
        for layer in self.feature:
            if isinstance(layer, nn.Linear):
                orthogonal_init(layer, gain=np.sqrt(2))
        
        # Initialize actor mean layers with moderate gain to encourage exploration
        for layer in self.actor_mean:
            if isinstance(layer, nn.Linear):
                orthogonal_init(layer, gain=0.1)  # Increased gain for better initial exploration
        
        # Initialize critic layers
        for layer in self.critic:
            if isinstance(layer, nn.Linear):
                orthogonal_init(layer, gain=np.sqrt(2))
    
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
        # Use learnable parameter, expand to batch size
        action_logstd = self.actor_logstd.unsqueeze(0).expand(action_mean.size(0), -1)
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
            action: [action_dim] or [batch_size, action_dim] 选择的动作（已缩放到[-max_speed, max_speed]）
            action_logprob: 动作的对数概率
            state_value: 状态价值
        """
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        action_mean, action_logstd, state_value = self.forward(state)
        # action_mean is already in [-1, 1] range due to Tanh
        # Scale to action space range (will be done in environment, but we can also do it here)
        # For now, let environment handle the scaling based on max_speed
        
        action_std = torch.exp(action_logstd)
        
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        # Allow slight overflow for exploration - environment will handle max_speed limit
        # Only clip extreme values to prevent numerical issues
        action = torch.clamp(action, -2.0, 2.0)  # Allow 2x range for exploration
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
            action: [batch_size, action_dim] (expected in [-1, 1] range)
        
        Returns:
            action_logprob: [batch_size]
            state_value: [batch_size, 1]
            dist_entropy: 分布熵
        """
        action_mean, action_logstd, state_value = self.forward(state)
        action_std = torch.exp(action_logstd)
        
        # Allow slight overflow for exploration - environment will handle scaling
        # Only clip extreme values to prevent numerical issues
        action = torch.clamp(action, -2.0, 2.0)  # Allow 2x range for exploration
        
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
        device=None,
        value_clip=False
    ):
        """
        初始化PPO
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度（连续动作空间为4，表示thrust, roll_torque, pitch_torque, yaw_torque）
            lr_actor: Actor学习率
            lr_critic: Critic学习率
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
            eps_clip: PPO裁剪参数
            k_epochs: 每次更新的epoch数
            device: 设备（CPU或GPU）
            value_clip: 是否使用value clipping来稳定critic学习
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 使用不同的学习率
        self.optimizer = optim.Adam([
            {'params': self.policy.actor_mean.parameters(), 'lr': lr_actor},
            {'params': [self.policy.actor_logstd], 'lr': lr_actor},  # Learnable parameter
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
            {'params': self.policy.feature.parameters(), 'lr': lr_actor}
        ])
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.value_clip = value_clip
        
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
            action: numpy array, 选择的动作 [thrust, roll_torque, pitch_torque, yaw_torque]
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
        
        # 计算下一个状态的价值（用于TD误差）- 向量化实现
        next_values = torch.zeros_like(old_state_values)
        # 使用roll操作和mask来向量化
        next_values[:-1] = old_state_values[1:] * (1 - dones[:-1])
        
        # 计算TD误差: δ_t = r_t + γV(s_{t+1}) - V(s_t)
        td_errors = rewards + self.gamma * next_values * (1 - dones) - old_state_values
        
        # 使用GAE计算优势函数: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        # 向量化实现：从后向前累积
        advantages = torch.zeros_like(td_errors)
        last_gae = 0.0
        # 使用向量化操作，但需要处理done标志
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
            if self.value_clip:
                # Value clipping for stable critic learning (similar to PPO2)
                # Clip value estimates to prevent large updates
                value_clipped = old_state_values + torch.clamp(
                    state_values - old_state_values, 
                    -self.eps_clip, 
                    self.eps_clip
                )
                value_loss_clipped = (value_clipped - returns.detach()) ** 2
                value_loss_unclipped = (state_values - returns.detach()) ** 2
                critic_loss = torch.max(value_loss_clipped, value_loss_unclipped).mean()
            else:
                # Standard MSE loss
                critic_loss = nn.MSELoss()(state_values, returns.detach())
            
            # 总损失（调整权重）
            # Reduced critic loss weight due to large reward scale
            # Increased entropy bonus to encourage exploration
            loss = actor_loss + 0.1 * critic_loss - 0.05 * dist_entropy.mean()  # Reduced critic weight from 0.5 to 0.1
            
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

