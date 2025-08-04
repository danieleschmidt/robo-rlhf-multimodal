"""
Environment creation and registration.
"""

import gymnasium as gym
from typing import Any, Dict, Optional
import numpy as np

from robo_rlhf.envs.base import RobotEnv
from robo_rlhf.envs.mujoco_envs import MujocoManipulation, MujocoLocomotion


def make_env(env_name: str, **kwargs) -> gym.Env:
    """
    Create environment by name.
    
    Args:
        env_name: Name of the environment
        **kwargs: Additional environment parameters
        
    Returns:
        Gym environment instance
    """
    env_name = env_name.lower()
    
    # MuJoCo environments
    if env_name in ["mujocomanipulation", "mujoco_manipulation"]:
        return MujocoManipulation(**kwargs)
    elif env_name == "mujoco_reach":
        return MujocoManipulation(task="reach", **kwargs)
    elif env_name == "mujoco_pick":
        return MujocoManipulation(task="pick_and_place", **kwargs)
    elif env_name == "mujoco_stack":
        return MujocoManipulation(task="stacking", **kwargs)
    elif env_name == "mujoco_locomotion":
        return MujocoLocomotion(**kwargs)
    
    # Isaac Sim environments (placeholder)
    elif env_name.startswith("isaac"):
        try:
            from robo_rlhf.envs.isaac_envs import IsaacSimEnv
            return IsaacSimEnv(task=env_name.replace("isaac_", ""), **kwargs)
        except ImportError:
            raise ImportError("Isaac Sim environments require omniverse and isaac-sim installation")
    
    # Standard Gym environments (with vision wrapper)
    elif env_name in ["cartpole", "pendulum", "mountaincar", "lunarlander"]:
        base_env = gym.make(f"{env_name.title()}-v1" if env_name != "lunarlander" else "LunarLander-v2")
        return VisionWrapper(base_env, **kwargs)
    
    # Try to create as standard gym environment
    else:
        try:
            return gym.make(env_name, **kwargs)
        except gym.error.UnregisteredEnv:
            raise ValueError(f"Unknown environment: {env_name}. Available: mujoco_manipulation, mujoco_reach, mujoco_pick, mujoco_stack, isaac_*, cartpole, pendulum, mountaincar, lunarlander")


class VisionWrapper(gym.Wrapper):
    """
    Wrapper to add vision observations to standard gym environments.
    
    Renders the environment and includes the image in observations.
    """
    
    def __init__(
        self,
        env: gym.Env,
        image_size: int = 64,
        include_state: bool = True,
        grayscale: bool = False
    ):
        """
        Initialize vision wrapper.
        
        Args:
            env: Base environment
            image_size: Size of rendered images
            include_state: Whether to include original state
            grayscale: Whether to convert to grayscale
        """
        super().__init__(env)
        
        self.image_size = image_size
        self.include_state = include_state
        self.grayscale = grayscale
        
        # Setup observation space
        if grayscale:
            image_shape = (1, image_size, image_size)
        else:
            image_shape = (3, image_size, image_size)
        
        if include_state:
            self.observation_space = gym.spaces.Dict({
                "pixels": gym.spaces.Box(
                    low=0, high=255, shape=image_shape, dtype=np.uint8
                ),
                "state": env.observation_space
            })
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=image_shape, dtype=np.uint8
            )
    
    def reset(self, **kwargs):
        """Reset environment and return observation with image."""
        obs, info = self.env.reset(**kwargs)
        return self._get_obs(obs), info
    
    def step(self, action):
        """Step environment and return observation with image."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._get_obs(obs), reward, terminated, truncated, info
    
    def _get_obs(self, state_obs):
        """Get observation including rendered image."""
        # Render environment
        try:
            image = self.env.render()
        except:
            image = None
        
        if image is None:
            # Create blank image if rendering fails
            if self.grayscale:
                image = np.zeros((1, self.image_size, self.image_size), dtype=np.uint8)
            else:
                image = np.zeros((3, self.image_size, self.image_size), dtype=np.uint8)
        else:
            # Resize image
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB image - would use cv2 in production
                import numpy as np
                # Simple nearest neighbor resize (would use proper interpolation in production)
                h, w = image.shape[:2]
                image = image[::h//self.image_size, ::w//self.image_size]
                if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
                    # Pad if needed
                    pad_h = max(0, self.image_size - image.shape[0])
                    pad_w = max(0, self.image_size - image.shape[1])
                    image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
                    image = image[:self.image_size, :self.image_size]
                
                if self.grayscale:
                    # Convert to grayscale
                    image = np.mean(image, axis=2, keepdims=True).astype(np.uint8)
                    image = image.transpose(2, 0, 1)  # HWC -> CHW
                else:
                    image = image.transpose(2, 0, 1)  # HWC -> CHW
        
        if self.include_state:
            return {
                "pixels": image,
                "state": state_obs
            }
        else:
            return image


# Environment registry for easy access
ENV_REGISTRY = {
    "mujoco_manipulation": {
        "class": MujocoManipulation,
        "description": "MuJoCo manipulation tasks",
        "modalities": ["rgb", "depth", "proprioception"]
    },
    "mujoco_reach": {
        "class": lambda **kwargs: MujocoManipulation(task="reach", **kwargs),
        "description": "MuJoCo reaching task",
        "modalities": ["rgb", "proprioception"]
    },
    "mujoco_pick": {
        "class": lambda **kwargs: MujocoManipulation(task="pick_and_place", **kwargs),
        "description": "MuJoCo pick and place task",
        "modalities": ["rgb", "depth", "proprioception", "force"]
    },
    "mujoco_locomotion": {
        "class": MujocoLocomotion,
        "description": "MuJoCo locomotion tasks",
        "modalities": ["rgb", "proprioception", "imu"]
    }
}


def list_environments() -> Dict[str, Dict[str, Any]]:
    """List all available environments."""
    return ENV_REGISTRY.copy()


def get_env_info(env_name: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific environment."""
    return ENV_REGISTRY.get(env_name)


__all__ = [
    "RobotEnv",
    "MujocoManipulation", 
    "MujocoLocomotion",
    "make_env",
    "VisionWrapper",
    "list_environments",
    "get_env_info"
]