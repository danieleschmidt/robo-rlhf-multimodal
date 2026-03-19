"""
RobotObservation: multimodal observation container for robotics RLHF.

Each observation pairs an RGB image (3×64×64) with a proprioception vector
(joint angles + velocities). This is the atomic unit flowing through the
entire preference-learning pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import torch


# Canonical shapes — change here and everything downstream adapts.
IMAGE_SHAPE = (3, 64, 64)   # C × H × W
PROPRIO_DIM = 14            # 7 joints × (angle + velocity)


@dataclass
class RobotObservation:
    """
    A single multimodal robot observation.

    Attributes:
        image: RGB image tensor, shape (3, 64, 64), values in [0, 1].
        proprioception: Joint-state vector, shape (PROPRIO_DIM,).
            Layout: [q0…q6, dq0…dq6] (angles then velocities).
    """

    image: torch.Tensor          # (3, H, W)
    proprioception: torch.Tensor  # (PROPRIO_DIM,)

    def __post_init__(self) -> None:
        if self.image.shape != torch.Size(IMAGE_SHAPE):
            raise ValueError(
                f"image must have shape {IMAGE_SHAPE}, got {tuple(self.image.shape)}"
            )
        if self.proprioception.shape != torch.Size([PROPRIO_DIM]):
            raise ValueError(
                f"proprioception must have shape ({PROPRIO_DIM},), "
                f"got {tuple(self.proprioception.shape)}"
            )

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def random(cls, device: str | torch.device = "cpu") -> "RobotObservation":
        """Create a random observation (useful for testing/demos)."""
        return cls(
            image=torch.rand(IMAGE_SHAPE, device=device),
            proprioception=torch.randn(PROPRIO_DIM, device=device),
        )

    @classmethod
    def zeros(cls, device: str | torch.device = "cpu") -> "RobotObservation":
        """Create a zero observation."""
        return cls(
            image=torch.zeros(IMAGE_SHAPE, device=device),
            proprioception=torch.zeros(PROPRIO_DIM, device=device),
        )

    # ------------------------------------------------------------------
    # Batching helpers
    # ------------------------------------------------------------------

    @staticmethod
    def batch_images(obs_list: list["RobotObservation"]) -> torch.Tensor:
        """Stack images from a list of observations → (N, C, H, W)."""
        return torch.stack([o.image for o in obs_list])

    @staticmethod
    def batch_proprios(obs_list: list["RobotObservation"]) -> torch.Tensor:
        """Stack proprioception vectors → (N, PROPRIO_DIM)."""
        return torch.stack([o.proprioception for o in obs_list])

    def to(self, device: str | torch.device) -> "RobotObservation":
        """Move tensors to *device* (returns new instance)."""
        return RobotObservation(
            image=self.image.to(device),
            proprioception=self.proprioception.to(device),
        )

    def __repr__(self) -> str:
        return (
            f"RobotObservation(image={tuple(self.image.shape)}, "
            f"proprio={tuple(self.proprioception.shape)}, "
            f"device={self.image.device})"
        )
