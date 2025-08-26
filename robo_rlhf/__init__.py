"""
Robo-RLHF-Multimodal: Multimodal Reinforcement Learning from Human Feedback for Robotics.

End-to-end pipeline for collecting teleoperation data, gathering human preferences,
and fine-tuning policies using state-of-the-art multimodal RLHF techniques.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"

# Core imports (always available)
from robo_rlhf.collectors import TeleOpCollector

# Optional imports with graceful fallback
try:
    from robo_rlhf.preference import PreferencePairGenerator, PreferenceServer
except ImportError:
    PreferencePairGenerator, PreferenceServer = None, None

try:
    from robo_rlhf.algorithms import MultimodalRLHF
except ImportError:
    MultimodalRLHF = None

try:
    from robo_rlhf.models import VisionLanguageActor
except ImportError:
    VisionLanguageActor = None

# Quantum-inspired autonomous capabilities (with graceful fallback)
try:
    from robo_rlhf.quantum import (
        QuantumTaskPlanner,
        QuantumDecisionEngine,
        QuantumOptimizer,
        MultiObjectiveOptimizer,
        AutonomousSDLCExecutor,
        PredictiveAnalytics,
        ResourcePredictor
    )
except ImportError as e:
    print(f"Warning: Quantum modules not available: {e}")
    QuantumTaskPlanner = None
    QuantumDecisionEngine = None
    QuantumOptimizer = None
    MultiObjectiveOptimizer = None
    AutonomousSDLCExecutor = None
    PredictiveAnalytics = None
    ResourcePredictor = None

__all__ = [
    "TeleOpCollector",
    "PreferencePairGenerator", 
    "PreferenceServer",
    "MultimodalRLHF",
    "VisionLanguageActor",
    # Quantum capabilities
    "QuantumTaskPlanner",
    "QuantumDecisionEngine",
    "QuantumOptimizer", 
    "MultiObjectiveOptimizer",
    "AutonomousSDLCExecutor",
    "PredictiveAnalytics",
    "ResourcePredictor",
]