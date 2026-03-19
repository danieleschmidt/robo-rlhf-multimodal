"""
RLHFTrainer: trains the reward model and fine-tunes a policy via PPO.

Two-stage pipeline
------------------
Stage 1 — Reward learning
    Optimise RewardModel on human preference pairs using Bradley-Terry loss.

Stage 2 — Policy fine-tuning (PPO)
    A lightweight policy (MLP over the fused embedding) collects trajectories
    in a synthetic environment, receives rewards from the frozen reward model,
    and is updated with PPO.

The PPO implementation here is self-contained and minimal — no stable-baselines
or RL library dependency.  It is correct enough to demonstrate reward-shaping
from learned human preferences on synthetic observations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from robo_rlhf.encoder import MultimodalEncoder, FUSED_DIM
from robo_rlhf.observation import RobotObservation, PROPRIO_DIM
from robo_rlhf.preference_dataset import PreferenceDataset
from robo_rlhf.reward_model import RewardModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight policy network
# ---------------------------------------------------------------------------

class RobotPolicy(nn.Module):
    """
    Simple Gaussian policy over the fused observation embedding.

    Action space: continuous, same dimensionality as proprioception
    (i.e., joint torques / velocities).
    """

    def __init__(
        self,
        fused_dim: int = FUSED_DIM,
        action_dim: int = PROPRIO_DIM,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        z : (B, fused_dim) — encoded observation

        Returns
        -------
        mean : (B, action_dim)
        std  : (B, action_dim)  — broadcast from learned log_std
        """
        h = self.trunk(z)
        mean = self.mean_head(h)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def sample(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and log-prob."""
        mean, std = self(z)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)  # (B,)
        return action, log_prob

    def log_prob(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean, std = self(z)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(-1)  # (B,)


# ---------------------------------------------------------------------------
# Trainer config
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    # Reward model training
    reward_lr: float = 3e-4
    reward_epochs: int = 10
    reward_batch_size: int = 32

    # PPO
    ppo_lr: float = 1e-4
    ppo_epochs: int = 5              # epochs per PPO update
    ppo_rollout_steps: int = 128     # steps before each PPO update
    ppo_updates: int = 10            # total PPO update rounds
    ppo_clip_eps: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95

    device: str = "cpu"


# ---------------------------------------------------------------------------
# RLHFTrainer
# ---------------------------------------------------------------------------

class RLHFTrainer:
    """
    Orchestrates reward model training + PPO policy fine-tuning.

    Parameters
    ----------
    config : TrainerConfig
    encoder : optional pre-built MultimodalEncoder; if None, creates one.
    """

    def __init__(
        self,
        config: TrainerConfig | None = None,
        encoder: MultimodalEncoder | None = None,
    ) -> None:
        self.cfg = config or TrainerConfig()
        self.device = torch.device(self.cfg.device)

        self.encoder = (encoder or MultimodalEncoder()).to(self.device)
        self.reward_model = RewardModel(encoder=self.encoder).to(self.device)
        self.policy = RobotPolicy(fused_dim=self.encoder.out_dim).to(self.device)

        # Value network (critic for PPO)
        self.value_net = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        ).to(self.device)

        self.reward_opt = Adam(self.reward_model.parameters(), lr=self.cfg.reward_lr)
        self.policy_opt = Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=self.cfg.ppo_lr,
        )

    # ------------------------------------------------------------------
    # Stage 1: Reward learning
    # ------------------------------------------------------------------

    def train_reward_model(
        self, dataset: PreferenceDataset
    ) -> List[float]:
        """
        Train the reward model on preference pairs.

        Returns
        -------
        loss history (one value per epoch)
        """
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.reward_batch_size,
            shuffle=True,
            collate_fn=PreferenceDataset.collate_fn,
        )

        self.reward_model.train()
        history: List[float] = []

        for epoch in range(1, self.cfg.reward_epochs + 1):
            epoch_loss = 0.0
            for img_a, prop_a, img_b, prop_b, pref in loader:
                img_a  = img_a.to(self.device)
                prop_a = prop_a.to(self.device)
                img_b  = img_b.to(self.device)
                prop_b = prop_b.to(self.device)
                pref   = pref.to(self.device)

                self.reward_opt.zero_grad()
                loss = self.reward_model(img_a, prop_a, img_b, prop_b, pref)
                loss.backward()
                self.reward_opt.step()
                epoch_loss += loss.item()

            avg = epoch_loss / len(loader)
            history.append(avg)
            logger.info(f"Reward epoch {epoch}/{self.cfg.reward_epochs}  loss={avg:.4f}")

        self.reward_model.eval()
        return history

    # ------------------------------------------------------------------
    # Stage 2: PPO fine-tuning with learned reward
    # ------------------------------------------------------------------

    def _collect_rollout(self) -> dict:
        """
        Collect a rollout in a *synthetic* environment.

        The "environment" simply generates random observations each step;
        the reward comes from the learned reward model.  This isolates the
        RLHF training loop from any real simulator dependency.
        """
        steps = self.cfg.ppo_rollout_steps

        imgs    = torch.zeros(steps, 3, 64, 64, device=self.device)
        props   = torch.zeros(steps, PROPRIO_DIM, device=self.device)
        actions = torch.zeros(steps, PROPRIO_DIM, device=self.device)
        log_ps  = torch.zeros(steps, device=self.device)
        rewards = torch.zeros(steps, device=self.device)
        values  = torch.zeros(steps, device=self.device)

        self.encoder.eval()
        self.reward_model.eval()
        self.policy.eval()
        self.value_net.eval()

        with torch.no_grad():
            for t in range(steps):
                obs = RobotObservation.random(device=self.device)
                img = obs.image.unsqueeze(0)
                prop = obs.proprioception.unsqueeze(0)
                z = self.encoder(img, prop)

                action, log_p = self.policy.sample(z)
                r = self.reward_model.reward(img, prop)
                v = self.value_net(z).squeeze(-1)

                imgs[t]    = img[0]
                props[t]   = prop[0]
                actions[t] = action[0]
                log_ps[t]  = log_p[0]
                rewards[t] = r[0]
                values[t]  = v[0]

        # GAE returns + advantages
        returns, advantages = self._gae(rewards, values)
        return dict(
            imgs=imgs, props=props, actions=actions,
            log_ps=log_ps, returns=returns, advantages=advantages,
            rewards=rewards,
        )

    def _gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generalised Advantage Estimation."""
        T = len(rewards)
        returns    = torch.zeros(T, device=self.device)
        advantages = torch.zeros(T, device=self.device)
        gae = 0.0
        next_val = 0.0

        for t in reversed(range(T)):
            delta = rewards[t] + self.cfg.gamma * next_val - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_val = values[t].item()

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def ppo_update(self, rollout: dict) -> dict:
        """Run PPO epochs on the collected rollout."""
        imgs       = rollout["imgs"]
        props      = rollout["props"]
        actions    = rollout["actions"]
        old_log_ps = rollout["log_ps"]
        returns    = rollout["returns"]
        advantages = rollout["advantages"]

        policy_losses, value_losses, entropy_losses = [], [], []

        self.encoder.train()
        self.policy.train()
        self.value_net.train()

        for _ in range(self.cfg.ppo_epochs):
            z = self.encoder(imgs, props)

            new_log_ps = self.policy.log_prob(z, actions)
            mean, std  = self.policy(z)
            entropy    = torch.distributions.Normal(mean, std).entropy().sum(-1).mean()

            ratio = (new_log_ps - old_log_ps).exp()
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - self.cfg.ppo_clip_eps, 1 + self.cfg.ppo_clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            values = self.value_net(z).squeeze(-1)
            value_loss = F.mse_loss(values, returns)

            loss = (
                policy_loss
                + self.cfg.ppo_value_coef * value_loss
                - self.cfg.ppo_entropy_coef * entropy
            )

            self.policy_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value_net.parameters()), 0.5
            )
            self.policy_opt.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy.item())

        return dict(
            policy_loss=sum(policy_losses) / len(policy_losses),
            value_loss=sum(value_losses) / len(value_losses),
            entropy=sum(entropy_losses) / len(entropy_losses),
        )

    def train_policy(self) -> List[dict]:
        """
        Fine-tune the policy with PPO using the learned reward.

        Returns
        -------
        list of per-update metrics dicts
        """
        metrics = []
        for update in range(1, self.cfg.ppo_updates + 1):
            rollout = self._collect_rollout()
            m = self.ppo_update(rollout)
            mean_reward = rollout["rewards"].mean().item()
            m["mean_reward"] = mean_reward
            metrics.append(m)
            logger.info(
                f"PPO update {update}/{self.cfg.ppo_updates}  "
                f"policy_loss={m['policy_loss']:.4f}  "
                f"value_loss={m['value_loss']:.4f}  "
                f"mean_reward={mean_reward:.4f}"
            )
        return metrics

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def evaluate_reward_model(self, dataset: PreferenceDataset) -> float:
        """
        Compute preference prediction accuracy on a dataset.

        Returns fraction of pairs where the model correctly predicts
        which observation is preferred.
        """
        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=PreferenceDataset.collate_fn,
        )
        self.reward_model.eval()
        correct = total = 0

        with torch.no_grad():
            for img_a, prop_a, img_b, prop_b, pref in loader:
                img_a  = img_a.to(self.device)
                prop_a = prop_a.to(self.device)
                img_b  = img_b.to(self.device)
                prop_b = prop_b.to(self.device)
                pref   = pref.to(self.device)

                prob = self.reward_model.preference_prob(img_a, prop_a, img_b, prop_b)
                predicted = (prob >= 0.5).float()
                correct += (predicted == pref).sum().item()
                total   += pref.size(0)

        return correct / total if total > 0 else 0.0
