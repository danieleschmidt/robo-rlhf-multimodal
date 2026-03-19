"""
robo_rlhf — Multimodal RLHF pipeline for robotics.

Core components
---------------
RobotObservation   : multimodal observation (image + proprioception)
MultimodalEncoder  : CNN + MLP encoder → fused embedding
RewardModel        : Bradley-Terry preference model
PreferenceDataset  : dataset of (obs_a, obs_b, preference) pairs
RLHFTrainer        : reward learning + PPO policy fine-tuning
"""

__version__ = "0.2.0"
__author__ = "Daniel Schmidt"

from robo_rlhf.observation import RobotObservation, IMAGE_SHAPE, PROPRIO_DIM
from robo_rlhf.encoder import MultimodalEncoder, ImageEncoder, ProprioEncoder
from robo_rlhf.reward_model import RewardModel
from robo_rlhf.preference_dataset import PreferenceDataset, PreferencePair
from robo_rlhf.rlhf_trainer import RLHFTrainer, TrainerConfig, RobotPolicy

__all__ = [
    "RobotObservation",
    "IMAGE_SHAPE",
    "PROPRIO_DIM",
    "MultimodalEncoder",
    "ImageEncoder",
    "ProprioEncoder",
    "RewardModel",
    "PreferenceDataset",
    "PreferencePair",
    "RLHFTrainer",
    "TrainerConfig",
    "RobotPolicy",
]
