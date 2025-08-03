"""
Simulator environments for robotic tasks.
"""

from robo_rlhf.envs.base import RobotEnv
from robo_rlhf.envs.mujoco_envs import MujocoManipulation, MujocoLocomotion

__all__ = [
    "RobotEnv",
    "MujocoManipulation",
    "MujocoLocomotion",
]