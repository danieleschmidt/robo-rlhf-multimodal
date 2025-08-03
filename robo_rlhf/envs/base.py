"""
Base environment class for robotic tasks.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Any, Optional
from abc import ABC, abstractmethod


class RobotEnv(gym.Env, ABC):
    """
    Base class for robot environments.
    
    Provides common interface for multimodal observations.
    """
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        observation_modalities: list = ["rgb", "proprioception"],
        action_dim: int = 7,
        max_episode_steps: int = 1000
    ):
        """
        Initialize robot environment.
        
        Args:
            render_mode: Rendering mode ('human', 'rgb_array')
            observation_modalities: List of observation types
            action_dim: Dimension of action space
            max_episode_steps: Maximum steps per episode
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.observation_modalities = observation_modalities
        self.max_episode_steps = max_episode_steps
        
        # Define action space (continuous control)
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
        
        # Define observation space
        self.observation_space = self._build_observation_space()
        
        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        
    @abstractmethod
    def _build_observation_space(self) -> gym.spaces.Space:
        """Build the observation space based on modalities."""
        pass
    
    @abstractmethod
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        pass
    
    @abstractmethod
    def _compute_reward(self) -> float:
        """Compute reward for current state."""
        pass
    
    @abstractmethod
    def _check_success(self) -> bool:
        """Check if task is successfully completed."""
        pass
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Reset robot state
        self._reset_robot()
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            "episode_step": self.current_step,
            "episode_reward": self.episode_reward
        }
        
        return observation, info
    
    @abstractmethod
    def _reset_robot(self) -> None:
        """Reset robot to initial configuration."""
        pass
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute action in environment.
        
        Args:
            action: Action to execute
            
        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode ended (success/failure)
            truncated: Whether episode was truncated (timeout)
            info: Additional information
        """
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Execute action
        self._execute_action(action)
        
        # Update step counter
        self.current_step += 1
        
        # Get new observation
        observation = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward()
        self.episode_reward += reward
        
        # Check termination
        success = self._check_success()
        terminated = success or self._check_failure()
        truncated = self.current_step >= self.max_episode_steps
        
        # Prepare info
        info = {
            "episode_step": self.current_step,
            "episode_reward": self.episode_reward,
            "success": success,
            "is_success": success  # For compatibility
        }
        
        return observation, reward, terminated, truncated, info
    
    @abstractmethod
    def _execute_action(self, action: np.ndarray) -> None:
        """Execute action on robot."""
        pass
    
    @abstractmethod
    def _check_failure(self) -> bool:
        """Check if task has failed."""
        pass
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render environment.
        
        Returns:
            Rendered frame if render_mode is 'rgb_array'
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_human()
        return None
    
    @abstractmethod
    def _render_frame(self) -> np.ndarray:
        """Render current frame as RGB array."""
        pass
    
    def _render_human(self) -> None:
        """Render for human viewing (optional)."""
        pass
    
    def close(self) -> None:
        """Clean up environment resources."""
        pass


class MultimodalObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper to provide multimodal observations.
    
    Converts standard gym observations to multimodal format.
    """
    
    def __init__(
        self,
        env: gym.Env,
        modalities: list = ["rgb", "proprioception"]
    ):
        """
        Initialize wrapper.
        
        Args:
            env: Base environment
            modalities: List of observation modalities
        """
        super().__init__(env)
        self.modalities = modalities
        
        # Update observation space
        spaces = {}
        
        if "rgb" in modalities:
            spaces["rgb"] = gym.spaces.Box(
                low=0, high=255,
                shape=(224, 224, 3),
                dtype=np.uint8
            )
        
        if "depth" in modalities:
            spaces["depth"] = gym.spaces.Box(
                low=0, high=10.0,
                shape=(224, 224, 1),
                dtype=np.float32
            )
        
        if "proprioception" in modalities:
            # Assume some standard proprioceptive dimensions
            spaces["proprioception"] = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(7,),  # Joint positions
                dtype=np.float32
            )
        
        self.observation_space = gym.spaces.Dict(spaces)
    
    def observation(self, obs: Any) -> Dict[str, np.ndarray]:
        """
        Convert observation to multimodal format.
        
        Args:
            obs: Original observation
            
        Returns:
            Multimodal observation dictionary
        """
        multimodal_obs = {}
        
        if "rgb" in self.modalities:
            # Get RGB image
            if hasattr(self.env, 'render'):
                rgb = self.env.render()
                if rgb is not None:
                    multimodal_obs["rgb"] = rgb
            
        if "proprioception" in self.modalities:
            # Extract proprioceptive data
            if isinstance(obs, np.ndarray):
                multimodal_obs["proprioception"] = obs[:7]  # First 7 dims
            elif isinstance(obs, dict) and "robot_state" in obs:
                multimodal_obs["proprioception"] = obs["robot_state"]
        
        return multimodal_obs