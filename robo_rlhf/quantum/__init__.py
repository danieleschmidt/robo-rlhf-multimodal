"""
Quantum-inspired autonomous task planning and SDLC execution.

This module implements quantum computing principles for autonomous decision-making,
self-optimization, and intelligent task planning in robotics RLHF systems.
"""

from robo_rlhf.quantum.planner import QuantumTaskPlanner, QuantumDecisionEngine
from robo_rlhf.quantum.optimizer import QuantumOptimizer, MultiObjectiveOptimizer
from robo_rlhf.quantum.autonomous import AutonomousSDLCExecutor
from robo_rlhf.quantum.analytics import PredictiveAnalytics, ResourcePredictor

__all__ = [
    "QuantumTaskPlanner",
    "QuantumDecisionEngine", 
    "QuantumOptimizer",
    "MultiObjectiveOptimizer",
    "AutonomousSDLCExecutor",
    "PredictiveAnalytics",
    "ResourcePredictor",
]