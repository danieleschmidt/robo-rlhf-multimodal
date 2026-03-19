"""
MultimodalEncoder: fuses image + proprioception into a fixed-size embedding.

Architecture
------------
  image (3×64×64)  ──► CNN  ──►  image_feat (128)  ──┐
                                                        ├──► fusion MLP ──► z (256)
  proprio (14,)    ──► MLP  ──►  prop_feat  (128)  ──┘

The fused representation z is the input to the RewardModel and optionally
to a policy head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from robo_rlhf.observation import RobotObservation, IMAGE_SHAPE, PROPRIO_DIM


IMAGE_FEAT_DIM = 128
PROP_FEAT_DIM = 128
FUSED_DIM = 256


class ImageEncoder(nn.Module):
    """Small CNN for 3×64×64 images → 128-d feature vector."""

    def __init__(self, out_dim: int = IMAGE_FEAT_DIM) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            # 3×64×64 → 32×30×30
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 32×30×30 → 64×14×14
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64×14×14 → 64×6×6
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64×6×6 → 64×1×1 (global avg pool)
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),          # → (64,)
        )
        self.proj = nn.Sequential(
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 64, 64) → (B, out_dim)"""
        return self.proj(self.cnn(x))


class ProprioEncoder(nn.Module):
    """3-layer MLP for proprioception vector → 128-d feature vector."""

    def __init__(self, in_dim: int = PROPRIO_DIM, out_dim: int = PROP_FEAT_DIM) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim) → (B, out_dim)"""
        return self.mlp(x)


class MultimodalEncoder(nn.Module):
    """
    Encode a RobotObservation (or batched tensors) into a fused embedding.

    Parameters
    ----------
    image_feat_dim  : output dim of the image branch (default 128)
    prop_feat_dim   : output dim of the proprioception branch (default 128)
    fused_dim       : dim of the fused representation (default 256)
    """

    def __init__(
        self,
        image_feat_dim: int = IMAGE_FEAT_DIM,
        prop_feat_dim: int = PROP_FEAT_DIM,
        fused_dim: int = FUSED_DIM,
    ) -> None:
        super().__init__()
        self.image_enc = ImageEncoder(out_dim=image_feat_dim)
        self.prop_enc = ProprioEncoder(out_dim=prop_feat_dim)
        self.fusion = nn.Sequential(
            nn.Linear(image_feat_dim + prop_feat_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim, fused_dim),
        )
        self.out_dim = fused_dim

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def encode_tensors(
        self, images: torch.Tensor, proprios: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode raw tensors.

        Parameters
        ----------
        images   : (B, 3, 64, 64)
        proprios : (B, PROPRIO_DIM)

        Returns
        -------
        z : (B, fused_dim)
        """
        img_feat = self.image_enc(images)
        prp_feat = self.prop_enc(proprios)
        return self.fusion(torch.cat([img_feat, prp_feat], dim=1))

    def encode_obs(self, obs: RobotObservation) -> torch.Tensor:
        """Encode a single (unbatched) RobotObservation → (1, fused_dim)."""
        images = obs.image.unsqueeze(0).to(next(self.parameters()).device)
        proprios = obs.proprioception.unsqueeze(0).to(next(self.parameters()).device)
        return self.encode_tensors(images, proprios)

    def encode_obs_list(self, obs_list: list[RobotObservation]) -> torch.Tensor:
        """Batch-encode a list of RobotObservations → (N, fused_dim)."""
        device = next(self.parameters()).device
        images = RobotObservation.batch_images(obs_list).to(device)
        proprios = RobotObservation.batch_proprios(obs_list).to(device)
        return self.encode_tensors(images, proprios)

    def forward(
        self,
        images: torch.Tensor,
        proprios: torch.Tensor,
    ) -> torch.Tensor:
        """Alias for encode_tensors — called by PyTorch training loops."""
        return self.encode_tensors(images, proprios)
