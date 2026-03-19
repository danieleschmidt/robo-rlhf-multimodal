"""
RewardModel: learns a scalar reward from human preference pairs.

Theory
------
We use the Bradley-Terry model. Given two observations A and B, the
probability that a human prefers A over B is:

    P(A ≻ B) = σ(r(A) - r(B))

where r(·) is the learned scalar reward and σ is the sigmoid function.

Training minimises the binary cross-entropy:

    L = -[ p · log σ(r_a - r_b) + (1-p) · log σ(r_b - r_a) ]

where p=1 means A is preferred, p=0 means B is preferred.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from robo_rlhf.encoder import MultimodalEncoder, FUSED_DIM
from robo_rlhf.observation import RobotObservation


class RewardModel(nn.Module):
    """
    Scalar reward model trained on (obs_a, obs_b, preference) triples.

    Parameters
    ----------
    encoder : MultimodalEncoder
        Shared encoder; may be pre-trained or trained jointly.
    hidden_dim : int
        Hidden size of the reward head MLP.
    """

    def __init__(
        self,
        encoder: MultimodalEncoder | None = None,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else MultimodalEncoder()
        fused_dim = self.encoder.out_dim
        self.reward_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reward(self, images: torch.Tensor, proprios: torch.Tensor) -> torch.Tensor:
        """
        Compute scalar reward for a batch.

        Parameters
        ----------
        images   : (B, 3, 64, 64)
        proprios : (B, PROPRIO_DIM)

        Returns
        -------
        r : (B,)
        """
        z = self.encoder(images, proprios)
        return self.reward_head(z).squeeze(-1)

    def reward_from_obs(self, obs: RobotObservation) -> torch.Tensor:
        """Reward for a single RobotObservation → scalar tensor."""
        z = self.encoder.encode_obs(obs)
        return self.reward_head(z).squeeze()

    def preference_logit(
        self,
        images_a: torch.Tensor,
        proprios_a: torch.Tensor,
        images_b: torch.Tensor,
        proprios_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Logit for the event 'A is preferred over B'.

        Returns
        -------
        logit : (B,)   — positive means A predicted preferred
        """
        r_a = self.reward(images_a, proprios_a)
        r_b = self.reward(images_b, proprios_b)
        return r_a - r_b

    def preference_prob(
        self,
        images_a: torch.Tensor,
        proprios_a: torch.Tensor,
        images_b: torch.Tensor,
        proprios_b: torch.Tensor,
    ) -> torch.Tensor:
        """P(A ≻ B) ∈ (0, 1) for each pair in the batch."""
        return torch.sigmoid(
            self.preference_logit(images_a, proprios_a, images_b, proprios_b)
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def bradley_terry_loss(
        r_a: torch.Tensor,
        r_b: torch.Tensor,
        preference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bradley-Terry binary cross-entropy loss.

        Parameters
        ----------
        r_a, r_b   : (B,) scalar rewards for obs A and obs B
        preference : (B,) float — 1.0 if A preferred, 0.0 if B preferred

        Returns
        -------
        loss : scalar
        """
        logit = r_a - r_b          # (B,)
        return F.binary_cross_entropy_with_logits(logit, preference)

    def forward(
        self,
        images_a: torch.Tensor,
        proprios_a: torch.Tensor,
        images_b: torch.Tensor,
        proprios_b: torch.Tensor,
        preference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Bradley-Terry loss for a batch of preference pairs.

        Parameters
        ----------
        images_a, proprios_a : tensors for observation A
        images_b, proprios_b : tensors for observation B
        preference           : (B,) — 1.0 if A preferred, 0.0 if B preferred

        Returns
        -------
        loss : scalar
        """
        r_a = self.reward(images_a, proprios_a)
        r_b = self.reward(images_b, proprios_b)
        return self.bradley_terry_loss(r_a, r_b, preference)
