"""
Robo-RLHF-Multimodal: Multimodal Reinforcement Learning from Human Feedback for Robotics.

End-to-end pipeline for collecting teleoperation data, gathering human preferences,
and fine-tuning policies using state-of-the-art multimodal RLHF techniques.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"

from robo_rlhf.collectors import TeleOpCollector
from robo_rlhf.preference import PreferencePairGenerator, PreferenceServer
from robo_rlhf.algorithms import MultimodalRLHF
from robo_rlhf.models import VisionLanguageActor

__all__ = [
    "TeleOpCollector",
    "PreferencePairGenerator", 
    "PreferenceServer",
    "MultimodalRLHF",
    "VisionLanguageActor",
]