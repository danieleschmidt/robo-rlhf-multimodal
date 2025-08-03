"""
Teleoperation data collection interfaces for robot demonstrations.
"""

from robo_rlhf.collectors.base import TeleOpCollector, DemonstrationData
from robo_rlhf.collectors.devices import (
    SpaceMouseController,
    KeyboardController,
    VRController
)
from robo_rlhf.collectors.recorder import DemonstrationRecorder

__all__ = [
    "TeleOpCollector",
    "DemonstrationData",
    "SpaceMouseController",
    "KeyboardController",
    "VRController",
    "DemonstrationRecorder",
]