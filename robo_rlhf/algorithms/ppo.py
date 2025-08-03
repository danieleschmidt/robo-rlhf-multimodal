"""
Proximal Policy Optimization (PPO) for policy learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class PPOTrainer:
    """PPO trainer for robotic policies."""
    
    def __init__(
        self,
        policy: nn.Module,
        reward_model: Optional[nn.Module] = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cuda"
    ):
        """Initialize PPO trainer."""
        self.policy = policy.to(device)
        self.reward_model = reward_model.to(device) if reward_model else None
        self.device = device
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Value function
        hidden_dim = 256
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        num_epochs: int = 10,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """Update policy using PPO."""
        stats = {"policy_loss": 0, "value_loss": 0, "entropy": 0}
        
        num_samples = states.shape[0]
        indices = np.arange(num_samples)
        
        for _ in range(num_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                new_actions, new_log_probs = self.policy(
                    batch_states[:, :3],  # RGB (mock)
                    batch_states[:, 3:],   # Proprioception (mock)
                )
                
                # PPO loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                policy_loss = -torch.min(
                    ratio * batch_advantages,
                    clipped_ratio * batch_advantages
                ).mean()
                
                # Entropy bonus
                entropy = -(new_log_probs * torch.exp(new_log_probs)).mean()
                
                # Total loss
                loss = policy_loss - self.entropy_coef * entropy
                
                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Update statistics
                stats["policy_loss"] += policy_loss.item()
                stats["entropy"] += entropy.item()
        
        # Average statistics
        num_updates = num_epochs * (num_samples // batch_size)
        for key in stats:
            stats[key] /= num_updates
        
        return stats