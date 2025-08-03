"""
Policy network architectures for multimodal control.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np

from robo_rlhf.models.encoders import VisionEncoder, ProprioceptionEncoder


class VisionLanguageActor(nn.Module):
    """
    Vision-language policy network for robotic control.
    
    Combines visual observations with proprioceptive states
    using transformer-based fusion.
    """
    
    def __init__(
        self,
        vision_encoder: str = "resnet18",
        proprioception_dim: int = 7,
        action_dim: int = 7,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_language: bool = False
    ):
        """
        Initialize vision-language actor.
        
        Args:
            vision_encoder: Type of vision encoder
            proprioception_dim: Dimension of proprioceptive input
            action_dim: Dimension of action output
            hidden_dim: Hidden layer dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
            use_language: Whether to use language conditioning
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_language = use_language
        
        # Vision encoder
        self.vision_encoder = VisionEncoder(
            encoder_type=vision_encoder,
            output_dim=hidden_dim
        )
        
        # Proprioception encoder
        self.proprio_encoder = ProprioceptionEncoder(
            input_dim=proprioception_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim
        )
        
        # Language encoder (if used)
        if use_language:
            from transformers import AutoModel
            self.language_encoder = AutoModel.from_pretrained("bert-base-uncased")
            self.language_proj = nn.Linear(768, hidden_dim)
        
        # Multimodal fusion with transformer
        self.fusion_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Action distribution parameters
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(
        self,
        rgb: torch.Tensor,
        proprioception: torch.Tensor,
        language: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the actor.
        
        Args:
            rgb: RGB images [B, C, H, W]
            proprioception: Proprioceptive states [B, D]
            language: Language tokens [B, L] (optional)
            deterministic: Whether to return deterministic actions
            
        Returns:
            actions: Predicted actions [B, action_dim]
            log_probs: Log probabilities of actions
        """
        batch_size = rgb.shape[0]
        
        # Encode modalities
        vision_features = self.vision_encoder(rgb)  # [B, hidden_dim]
        proprio_features = self.proprio_encoder(proprioception)  # [B, hidden_dim]
        
        # Prepare sequence for transformer
        features = [
            vision_features.unsqueeze(1),  # [B, 1, hidden_dim]
            proprio_features.unsqueeze(1)   # [B, 1, hidden_dim]
        ]
        
        # Add language features if available
        if self.use_language and language is not None:
            lang_features = self.language_encoder(language).last_hidden_state
            lang_features = self.language_proj(lang_features)  # [B, L, hidden_dim]
            features.append(lang_features)
        
        # Concatenate features
        multimodal_features = torch.cat(features, dim=1)  # [B, N, hidden_dim]
        
        # Fuse with transformer
        fused_features = self.fusion_layers(multimodal_features)  # [B, N, hidden_dim]
        
        # Global pooling
        pooled_features = fused_features.mean(dim=1)  # [B, hidden_dim]
        
        # Predict actions
        action_mean = self.action_head(pooled_features)  # [B, action_dim]
        
        if deterministic:
            return action_mean, torch.zeros_like(action_mean)
        
        # Sample from Gaussian distribution
        action_std = self.log_std.exp()
        action_dist = torch.distributions.Normal(action_mean, action_std)
        actions = action_dist.sample()
        log_probs = action_dist.log_prob(actions).sum(dim=-1)
        
        # Tanh squashing for bounded actions
        actions = torch.tanh(actions)
        
        return actions, log_probs
    
    def get_action(
        self,
        rgb: np.ndarray,
        proprioception: np.ndarray,
        language: Optional[str] = None
    ) -> np.ndarray:
        """
        Get action for deployment (numpy interface).
        
        Args:
            rgb: RGB image
            proprioception: Proprioceptive state
            language: Language instruction (optional)
            
        Returns:
            Action array
        """
        # Convert to tensors
        rgb_tensor = torch.from_numpy(rgb).float().unsqueeze(0)
        proprio_tensor = torch.from_numpy(proprioception).float().unsqueeze(0)
        
        # Move to device
        device = next(self.parameters()).device
        rgb_tensor = rgb_tensor.to(device)
        proprio_tensor = proprio_tensor.to(device)
        
        # Get action
        with torch.no_grad():
            action, _ = self.forward(
                rgb_tensor,
                proprio_tensor,
                deterministic=True
            )
        
        return action.cpu().numpy()[0]


class MultimodalActor(nn.Module):
    """
    General multimodal actor supporting various input modalities.
    """
    
    def __init__(
        self,
        modalities: List[str],
        modality_dims: Dict[str, int],
        action_dim: int,
        hidden_dim: int = 256,
        fusion_type: str = "concat"
    ):
        """
        Initialize multimodal actor.
        
        Args:
            modalities: List of modality names
            modality_dims: Input dimensions for each modality
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
            fusion_type: How to fuse modalities ('concat', 'attention', 'gated')
        """
        super().__init__()
        
        self.modalities = modalities
        self.action_dim = action_dim
        self.fusion_type = fusion_type
        
        # Create encoders for each modality
        self.encoders = nn.ModuleDict()
        for modality in modalities:
            if modality == "rgb" or modality == "depth":
                self.encoders[modality] = VisionEncoder(
                    encoder_type="resnet18",
                    output_dim=hidden_dim
                )
            else:
                self.encoders[modality] = nn.Sequential(
                    nn.Linear(modality_dims[modality], hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
        
        # Fusion module
        if fusion_type == "concat":
            fusion_dim = hidden_dim * len(modalities)
            self.fusion = nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim),
                nn.ReLU()
            )
        elif fusion_type == "attention":
            self.fusion = nn.MultiheadAttention(
                hidden_dim,
                num_heads=4,
                batch_first=True
            )
        elif fusion_type == "gated":
            self.fusion = GatedFusion(
                num_modalities=len(modalities),
                hidden_dim=hidden_dim
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Action prediction
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass through multimodal actor.
        
        Args:
            inputs: Dictionary of modality inputs
            
        Returns:
            Predicted actions
        """
        # Encode each modality
        encoded = []
        for modality in self.modalities:
            if modality in inputs:
                features = self.encoders[modality](inputs[modality])
                encoded.append(features)
        
        # Fuse modalities
        if self.fusion_type == "concat":
            fused = torch.cat(encoded, dim=-1)
            fused = self.fusion(fused)
        elif self.fusion_type == "attention":
            stacked = torch.stack(encoded, dim=1)  # [B, M, D]
            fused, _ = self.fusion(stacked, stacked, stacked)
            fused = fused.mean(dim=1)  # Pool over modalities
        elif self.fusion_type == "gated":
            fused = self.fusion(encoded)
        
        # Predict actions
        actions = self.action_head(fused)
        return torch.tanh(actions)  # Bounded actions


class GatedFusion(nn.Module):
    """Gated fusion module for combining modalities."""
    
    def __init__(self, num_modalities: int, hidden_dim: int):
        """Initialize gated fusion."""
        super().__init__()
        
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
            for _ in range(num_modalities)
        ])
        
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Apply gated fusion."""
        gated = []
        for i, feat in enumerate(features):
            gate = self.gates[i](feat)
            gated.append(gate * feat)
        
        concatenated = torch.cat(gated, dim=-1)
        return self.transform(concatenated)