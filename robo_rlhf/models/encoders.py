"""
Encoder architectures for different modalities.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class VisionEncoder(nn.Module):
    """
    Vision encoder for RGB/depth images.
    
    Supports various backbone architectures.
    """
    
    def __init__(
        self,
        encoder_type: str = "resnet18",
        output_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Initialize vision encoder.
        
        Args:
            encoder_type: Type of encoder ('resnet18', 'resnet50', 'efficientnet')
            output_dim: Output feature dimension
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()
        
        self.encoder_type = encoder_type
        self.output_dim = output_dim
        
        # Load backbone
        if encoder_type == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            backbone_dim = 512
        elif encoder_type == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        elif encoder_type == "efficientnet":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            backbone_dim = 1280
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Remove final classification layer
        if "resnet" in encoder_type:
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif "efficientnet" in encoder_type:
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        # Spatial pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to features.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Features [B, output_dim]
        """
        # Extract features
        features = self.backbone(x)
        
        # Pool if needed
        if len(features.shape) == 4:
            features = self.pool(features)
            features = features.flatten(1)
        
        # Project to output dimension
        output = self.projection(features)
        
        return output
    
    def extract_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial feature maps without pooling.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Spatial features [B, C', H', W']
        """
        # Get features before pooling
        if "resnet" in self.encoder_type:
            # Stop before avgpool
            features = x
            for layer in list(self.backbone.children())[:-1]:
                features = layer(features)
        else:
            features = self.backbone(x)
        
        return features


class ProprioceptionEncoder(nn.Module):
    """
    Encoder for proprioceptive sensor data.
    
    Handles joint positions, velocities, forces, etc.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        """
        Initialize proprioception encoder.
        
        Args:
            input_dim: Dimension of proprioceptive input
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of MLP layers
            dropout: Dropout probability
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build MLP layers
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(nn.ReLU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            current_dim = hidden_dim
        
        # Final layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # Optional: Learn importance weights for different proprioceptive channels
        self.channel_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode proprioceptive data.
        
        Args:
            x: Proprioceptive input [B, input_dim]
            
        Returns:
            Encoded features [B, output_dim]
        """
        # Apply channel attention
        attention = self.channel_attention(x)
        x = x * attention
        
        # Encode
        features = self.encoder(x)
        
        return features


class TemporalEncoder(nn.Module):
    """
    Encoder for temporal sequences using LSTM/GRU.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        bidirectional: bool = False
    ):
        """
        Initialize temporal encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: RNN hidden dimension
            output_dim: Output dimension
            num_layers: Number of RNN layers
            rnn_type: Type of RNN ('lstm' or 'gru')
            bidirectional: Whether to use bidirectional RNN
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        
        # RNN module
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # Output projection
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.projection = nn.Linear(rnn_output_dim, output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode temporal sequence.
        
        Args:
            x: Input sequence [B, T, D]
            lengths: Sequence lengths [B]
            
        Returns:
            Encoded features [B, output_dim]
        """
        # Pack if lengths provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
        
        # RNN forward pass
        if self.rnn_type == "lstm":
            output, (hidden, cell) = self.rnn(x)
        else:
            output, hidden = self.rnn(x)
        
        # Unpack if needed
        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True
            )
        
        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward
            hidden = torch.cat([
                hidden[-2, :, :],  # Forward
                hidden[-1, :, :]   # Backward
            ], dim=1)
        else:
            hidden = hidden[-1, :, :]
        
        # Project to output dimension
        output = self.projection(hidden)
        
        return output


class CrossModalEncoder(nn.Module):
    """
    Cross-modal attention encoder for relating different modalities.
    """
    
    def __init__(
        self,
        dim_a: int,
        dim_b: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize cross-modal encoder.
        
        Args:
            dim_a: Dimension of modality A
            dim_b: Dimension of modality B
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        # Project to common dimension
        self.proj_a = nn.Linear(dim_a, hidden_dim)
        self.proj_b = nn.Linear(dim_b, hidden_dim)
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Self-attention for refinement
        self.self_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode with cross-modal attention.
        
        Args:
            feat_a: Features from modality A [B, D_a]
            feat_b: Features from modality B [B, D_b]
            
        Returns:
            Cross-modal features [B, hidden_dim]
        """
        # Project to common space
        a = self.proj_a(feat_a).unsqueeze(1)  # [B, 1, D]
        b = self.proj_b(feat_b).unsqueeze(1)  # [B, 1, D]
        
        # Cross-attention: A attends to B
        a_to_b, _ = self.cross_attention(a, b, b)
        
        # Cross-attention: B attends to A
        b_to_a, _ = self.cross_attention(b, a, a)
        
        # Concatenate
        combined = torch.cat([a_to_b, b_to_a], dim=2)  # [B, 1, 2D]
        
        # Self-attention refinement
        refined, _ = self.self_attention(combined, combined, combined)
        
        # Output projection
        output = self.output(refined).squeeze(1)  # [B, D]
        
        return output