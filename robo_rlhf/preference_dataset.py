"""
PreferenceDataset: stores (obs_a, obs_b, preference) tuples.

Preference label convention
---------------------------
  1.0  →  obs_a is preferred
  0.0  →  obs_b is preferred

Synthetic preference generation
--------------------------------
We define a ground-truth proxy reward and use it to generate consistent
labels. The proxy here is a simple rule: prefer the observation whose
proprioception has a lower L2 norm (simulates a "less stressed" posture)
and whose image mean intensity is higher (simulates "brighter / more open"
scenes). This is arbitrary but deterministic and suitable for demos.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from robo_rlhf.observation import RobotObservation


@dataclass
class PreferencePair:
    obs_a: RobotObservation
    obs_b: RobotObservation
    preference: float  # 1.0 = A preferred, 0.0 = B preferred

    def to_tensors(self) -> Tuple[
        torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
        torch.Tensor,
    ]:
        """Unpack into (image_a, proprio_a, image_b, proprio_b, pref)."""
        return (
            self.obs_a.image,
            self.obs_a.proprioception,
            self.obs_b.image,
            self.obs_b.proprioception,
            torch.tensor(self.preference, dtype=torch.float32),
        )


def _proxy_reward(obs: RobotObservation) -> float:
    """
    Deterministic proxy reward for synthetic label generation.

    Higher is better:
      + bright images (open scenes)
      - high-norm proprioception (joint stress / extreme poses)
    """
    brightness = obs.image.mean().item()
    stress = obs.proprioception.norm().item()
    return brightness - 0.3 * stress


class PreferenceDataset(Dataset):
    """
    Dataset of human (or synthetic) preference pairs for RLHF training.

    Usage
    -----
    >>> ds = PreferenceDataset.synthetic(n_pairs=500)
    >>> loader = DataLoader(ds, batch_size=32, collate_fn=ds.collate_fn)
    """

    def __init__(self, pairs: List[PreferencePair] | None = None) -> None:
        self.pairs: List[PreferencePair] = pairs or []

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[
        torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
        torch.Tensor,
    ]:
        return self.pairs[idx].to_tensors()

    # ------------------------------------------------------------------
    # Building / augmenting the dataset
    # ------------------------------------------------------------------

    def add(self, obs_a: RobotObservation, obs_b: RobotObservation, preference: float) -> None:
        """Add a single (obs_a, obs_b, preference) pair."""
        self.pairs.append(PreferencePair(obs_a, obs_b, preference))

    def add_pair(self, pair: PreferencePair) -> None:
        self.pairs.append(pair)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def synthetic(
        cls,
        n_pairs: int = 200,
        noise: float = 0.1,
        device: str | torch.device = "cpu",
        seed: int = 42,
    ) -> "PreferenceDataset":
        """
        Generate *n_pairs* synthetic preference pairs.

        Each pair consists of two random observations; the preference label
        is determined by the proxy reward with optional label-flip noise.

        Parameters
        ----------
        n_pairs : number of preference pairs
        noise   : probability of flipping the ground-truth label (human error sim)
        device  : torch device
        seed    : random seed for reproducibility
        """
        rng = random.Random(seed)
        torch.manual_seed(seed)

        ds = cls()
        for _ in range(n_pairs):
            obs_a = RobotObservation.random(device=device)
            obs_b = RobotObservation.random(device=device)

            r_a = _proxy_reward(obs_a)
            r_b = _proxy_reward(obs_b)
            pref = 1.0 if r_a >= r_b else 0.0

            # Simulate human label noise
            if rng.random() < noise:
                pref = 1.0 - pref

            ds.add(obs_a, obs_b, pref)

        return ds

    # ------------------------------------------------------------------
    # Collate
    # ------------------------------------------------------------------

    @staticmethod
    def collate_fn(
        batch: List[Tuple[
            torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor,
            torch.Tensor,
        ]]
    ) -> Tuple[
        torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
        torch.Tensor,
    ]:
        """
        Collate a list of (img_a, prop_a, img_b, prop_b, pref) tuples
        into batched tensors.
        """
        img_a, prop_a, img_b, prop_b, pref = zip(*batch)
        return (
            torch.stack(img_a),
            torch.stack(prop_a),
            torch.stack(img_b),
            torch.stack(prop_b),
            torch.stack(pref),
        )
