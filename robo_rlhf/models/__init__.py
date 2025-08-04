"""
Neural network models for multimodal RLHF.
"""

# Optional PyTorch-based models
try:
    from robo_rlhf.models.actors import VisionLanguageActor, MultimodalActor
    from robo_rlhf.models.encoders import VisionEncoder, ProprioceptionEncoder
    HAS_TORCH = True
    # Try to import rewards module if it exists
    try:
        from robo_rlhf.models.rewards import RewardModel, EnsembleRewardModel
    except ImportError:
        RewardModel = None
        EnsembleRewardModel = None
except ImportError as e:
    HAS_TORCH = False
    VisionLanguageActor = None
    MultimodalActor = None
    VisionEncoder = None
    ProprioceptionEncoder = None
    RewardModel = None
    EnsembleRewardModel = None
    print(f"Warning: PyTorch models not available ({e})")

__all__ = [
    "VisionLanguageActor",
    "MultimodalActor",
    "VisionEncoder",
    "ProprioceptionEncoder",
    "RewardModel",
    "EnsembleRewardModel",
]