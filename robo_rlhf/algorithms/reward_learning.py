"""
Reward learning from human preferences.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, Optional


class RewardModel(nn.Module):
    """Base reward model for preference learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        """Initialize reward model."""
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict reward for input features."""
        return self.network(x)


class BradleyTerryModel(RewardModel):
    """Bradley-Terry model for preference learning."""
    
    def compute_preference_probability(
        self,
        reward_a: torch.Tensor,
        reward_b: torch.Tensor
    ) -> torch.Tensor:
        """Compute probability that A is preferred over B."""
        return torch.sigmoid(reward_a - reward_b)


class RewardLearner:
    """Trainer for reward models from preferences."""
    
    def __init__(
        self,
        model_type: str = "bradley_terry",
        input_dim: int = 6,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        device: str = "cuda"
    ):
        """Initialize reward learner."""
        if model_type == "bradley_terry":
            self.model = BradleyTerryModel(input_dim, hidden_dim).to(device)
        else:
            self.model = RewardModel(input_dim, hidden_dim).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.device = device
    
    def compute_preference_loss(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        labels: torch.Tensor,
        confidence: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute preference learning loss."""
        reward_a = self.model(features_a)
        reward_b = self.model(features_b)
        
        # Bradley-Terry loss
        probs = torch.sigmoid(reward_a - reward_b)
        loss = F.binary_cross_entropy(probs.squeeze(), labels)
        
        # Weight by confidence if provided
        if confidence is not None:
            loss = (loss * confidence).mean()
        
        return loss, probs