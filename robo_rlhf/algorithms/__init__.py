"""
RLHF algorithms for multimodal policy learning.
"""

# Optional PyTorch-based algorithms
try:
    from robo_rlhf.algorithms.rlhf import MultimodalRLHF
    from robo_rlhf.algorithms.reward_learning import RewardLearner, BradleyTerryModel
    from robo_rlhf.algorithms.ppo import PPOTrainer
    HAS_TORCH = True
except ImportError as e:
    HAS_TORCH = False
    MultimodalRLHF = None
    RewardLearner = None
    PPOTrainer = None
    BradleyTerryModel = None
    print(f"Warning: PyTorch algorithms not available ({e})")

__all__ = [
    "MultimodalRLHF",
    "RewardLearner",
    "BradleyTerryModel",
    "PPOTrainer",
]