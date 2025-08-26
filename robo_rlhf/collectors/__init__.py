"""
Teleoperation data collection interfaces for robot demonstrations.
"""

# Core imports
from robo_rlhf.collectors.base import TeleOpCollector, DemonstrationData

# Optional imports with graceful fallback
try:
    from robo_rlhf.collectors.devices import (
        SpaceMouseController,
        KeyboardController,
        VRController
    )
except ImportError:
    SpaceMouseController, KeyboardController, VRController = None, None, None

try:
    from robo_rlhf.collectors.recorder import DemonstrationRecorder
except ImportError:
    DemonstrationRecorder = None

__all__ = [
    "TeleOpCollector",
    "DemonstrationData",
    "SpaceMouseController",
    "KeyboardController",
    "VRController",
    "DemonstrationRecorder",
]