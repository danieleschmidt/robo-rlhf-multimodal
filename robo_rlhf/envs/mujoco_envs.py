"""
MuJoCo-based robotic environments.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Optional, Any
from robo_rlhf.envs.base import RobotEnv


class MujocoManipulation(RobotEnv):
    """
    MuJoCo manipulation environment for pick-and-place tasks.
    """
    
    def __init__(
        self,
        task: str = "pick_and_place",
        render_mode: Optional[str] = None,
        observation_modalities: list = ["rgb", "proprioception"],
        randomize: bool = True
    ):
        """
        Initialize MuJoCo manipulation environment.
        
        Args:
            task: Task type ('pick_and_place', 'stacking', 'insertion')
            render_mode: Rendering mode
            observation_modalities: Observation types
            randomize: Whether to randomize initial conditions
        """
        self.task = task
        self.randomize = randomize
        
        # Task-specific parameters
        self.target_position = np.array([0.5, 0.0, 0.1])
        self.object_position = np.array([0.3, 0.1, 0.05])
        self.gripper_position = np.array([0.0, 0.0, 0.3])
        
        # Success thresholds
        self.position_threshold = 0.05
        self.gripper_threshold = 0.02
        
        super().__init__(
            render_mode=render_mode,
            observation_modalities=observation_modalities,
            action_dim=7,  # 3D position + 3D orientation + gripper
            max_episode_steps=200
        )
    
    def _build_observation_space(self) -> gym.spaces.Space:
        """Build observation space."""
        spaces = {}
        
        if "rgb" in self.observation_modalities:
            spaces["rgb"] = gym.spaces.Box(
                low=0, high=255,
                shape=(224, 224, 3),
                dtype=np.uint8
            )
        
        if "proprioception" in self.observation_modalities:
            # Joint positions (7) + gripper state (1) + object position (3)
            spaces["proprioception"] = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(11,),
                dtype=np.float32
            )
        
        if "depth" in self.observation_modalities:
            spaces["depth"] = gym.spaces.Box(
                low=0, high=10.0,
                shape=(224, 224, 1),
                dtype=np.float32
            )
        
        return gym.spaces.Dict(spaces)
    
    def _reset_robot(self) -> None:
        """Reset robot to initial configuration."""
        # Reset gripper position
        self.gripper_position = np.array([0.0, 0.0, 0.3])
        self.gripper_open = True
        
        # Randomize object and target positions if needed
        if self.randomize:
            # Random object position
            self.object_position = np.array([
                np.random.uniform(0.2, 0.4),
                np.random.uniform(-0.2, 0.2),
                0.05
            ])
            
            # Random target position
            self.target_position = np.array([
                np.random.uniform(0.4, 0.6),
                np.random.uniform(-0.2, 0.2),
                0.1
            ])
        
        self.object_grasped = False
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        obs = {}
        
        if "rgb" in self.observation_modalities:
            # Simulate RGB image
            obs["rgb"] = self._render_frame()
        
        if "proprioception" in self.observation_modalities:
            # Combine gripper state and object position
            proprio = np.concatenate([
                self.gripper_position,
                np.array([0.0, 0.0, 0.0, 1.0]),  # Quaternion (mock)
                [1.0 if self.gripper_open else 0.0],
                self.object_position
            ])
            obs["proprioception"] = proprio.astype(np.float32)
        
        if "depth" in self.observation_modalities:
            # Simulate depth image
            obs["depth"] = np.random.randn(224, 224, 1).astype(np.float32)
        
        return obs
    
    def _execute_action(self, action: np.ndarray) -> None:
        """Execute action on robot."""
        # Update gripper position (first 3 dims)
        self.gripper_position += action[:3] * 0.05  # Scale down
        
        # Clip to workspace
        self.gripper_position = np.clip(
            self.gripper_position,
            [-0.5, -0.5, 0.0],
            [0.7, 0.5, 0.5]
        )
        
        # Gripper control (last dim)
        if action[-1] > 0:
            self.gripper_open = False
        else:
            self.gripper_open = True
        
        # Check grasping
        if not self.gripper_open and not self.object_grasped:
            dist_to_object = np.linalg.norm(
                self.gripper_position - self.object_position
            )
            if dist_to_object < self.gripper_threshold:
                self.object_grasped = True
        
        # Move object with gripper if grasped
        if self.object_grasped and not self.gripper_open:
            self.object_position = self.gripper_position.copy()
            self.object_position[2] -= 0.05  # Offset for gripper
        elif self.gripper_open:
            self.object_grasped = False
    
    def _compute_reward(self) -> float:
        """Compute reward for current state."""
        reward = 0.0
        
        if self.task == "pick_and_place":
            # Distance reward
            dist_to_object = np.linalg.norm(
                self.gripper_position - self.object_position
            )
            dist_to_target = np.linalg.norm(
                self.object_position - self.target_position
            )
            
            # Reaching reward
            reward -= dist_to_object * 0.1
            
            # Grasping reward
            if self.object_grasped:
                reward += 1.0
            
            # Placement reward
            reward -= dist_to_target * 0.2
            
            # Success reward
            if dist_to_target < self.position_threshold:
                reward += 10.0
        
        return reward
    
    def _check_success(self) -> bool:
        """Check if task is successfully completed."""
        if self.task == "pick_and_place":
            dist_to_target = np.linalg.norm(
                self.object_position - self.target_position
            )
            return dist_to_target < self.position_threshold
        return False
    
    def _check_failure(self) -> bool:
        """Check if task has failed."""
        # Check if object fell off table
        if self.object_position[2] < 0:
            return True
        
        # Check workspace violations
        if np.any(np.abs(self.gripper_position[:2]) > 1.0):
            return True
        
        return False
    
    def _render_frame(self) -> np.ndarray:
        """Render current frame as RGB array."""
        # Create simple visualization
        frame = np.ones((224, 224, 3), dtype=np.uint8) * 200
        
        # Draw gripper (blue)
        gripper_pixel = self._world_to_pixel(self.gripper_position)
        if gripper_pixel is not None:
            x, y = gripper_pixel
            frame[max(0, y-5):min(224, y+5), max(0, x-5):min(224, x+5)] = [0, 0, 255]
        
        # Draw object (green)
        object_pixel = self._world_to_pixel(self.object_position)
        if object_pixel is not None:
            x, y = object_pixel
            frame[max(0, y-4):min(224, y+4), max(0, x-4):min(224, x+4)] = [0, 255, 0]
        
        # Draw target (red)
        target_pixel = self._world_to_pixel(self.target_position)
        if target_pixel is not None:
            x, y = target_pixel
            frame[max(0, y-6):min(224, y+6), max(0, x-6):min(224, x+6)] = [255, 0, 0]
        
        return frame
    
    def _world_to_pixel(self, pos: np.ndarray) -> Optional[tuple]:
        """Convert world position to pixel coordinates."""
        # Simple orthographic projection
        x = int((pos[0] + 0.5) * 224)
        y = int((0.5 - pos[1]) * 224)
        
        if 0 <= x < 224 and 0 <= y < 224:
            return (x, y)
        return None


class MujocoLocomotion(RobotEnv):
    """
    MuJoCo locomotion environment for quadruped/humanoid control.
    """
    
    def __init__(
        self,
        robot_type: str = "quadruped",
        render_mode: Optional[str] = None,
        observation_modalities: list = ["proprioception"],
        terrain: str = "flat"
    ):
        """
        Initialize locomotion environment.
        
        Args:
            robot_type: Type of robot ('quadruped', 'humanoid')
            render_mode: Rendering mode
            observation_modalities: Observation types
            terrain: Terrain type ('flat', 'rough', 'stairs')
        """
        self.robot_type = robot_type
        self.terrain = terrain
        
        # Robot state
        self.robot_position = np.zeros(3)
        self.robot_velocity = np.zeros(3)
        self.robot_orientation = np.array([0, 0, 0, 1])  # Quaternion
        
        # Joint configuration
        if robot_type == "quadruped":
            self.num_joints = 12  # 3 per leg
        else:  # humanoid
            self.num_joints = 21
        
        self.joint_positions = np.zeros(self.num_joints)
        self.joint_velocities = np.zeros(self.num_joints)
        
        super().__init__(
            render_mode=render_mode,
            observation_modalities=observation_modalities,
            action_dim=self.num_joints,
            max_episode_steps=1000
        )
    
    def _build_observation_space(self) -> gym.spaces.Space:
        """Build observation space."""
        spaces = {}
        
        if "proprioception" in self.observation_modalities:
            # Position (3) + velocity (3) + orientation (4) + 
            # joint positions + joint velocities
            dim = 10 + 2 * self.num_joints
            spaces["proprioception"] = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(dim,),
                dtype=np.float32
            )
        
        if "rgb" in self.observation_modalities:
            spaces["rgb"] = gym.spaces.Box(
                low=0, high=255,
                shape=(224, 224, 3),
                dtype=np.uint8
            )
        
        return gym.spaces.Dict(spaces)
    
    def _reset_robot(self) -> None:
        """Reset robot to initial configuration."""
        self.robot_position = np.array([0, 0, 0.5])
        self.robot_velocity = np.zeros(3)
        self.robot_orientation = np.array([0, 0, 0, 1])
        
        # Random initial joint configuration
        self.joint_positions = np.random.randn(self.num_joints) * 0.1
        self.joint_velocities = np.zeros(self.num_joints)
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        obs = {}
        
        if "proprioception" in self.observation_modalities:
            proprio = np.concatenate([
                self.robot_position,
                self.robot_velocity,
                self.robot_orientation,
                self.joint_positions,
                self.joint_velocities
            ])
            obs["proprioception"] = proprio.astype(np.float32)
        
        if "rgb" in self.observation_modalities:
            obs["rgb"] = self._render_frame()
        
        return obs
    
    def _execute_action(self, action: np.ndarray) -> None:
        """Execute action on robot."""
        # Apply joint torques
        self.joint_velocities += action * 0.01  # Simple integration
        self.joint_velocities *= 0.95  # Damping
        self.joint_positions += self.joint_velocities * 0.01
        
        # Update robot position based on gait
        forward_speed = np.mean(np.abs(self.joint_velocities)) * 0.1
        self.robot_velocity[0] = forward_speed
        self.robot_position += self.robot_velocity * 0.01
    
    def _compute_reward(self) -> float:
        """Compute reward for current state."""
        # Forward progress reward
        reward = self.robot_velocity[0]
        
        # Upright bonus
        if self.robot_position[2] > 0.3:
            reward += 0.1
        
        # Energy penalty
        reward -= 0.001 * np.sum(self.joint_velocities ** 2)
        
        return reward
    
    def _check_success(self) -> bool:
        """Check if task is successfully completed."""
        # Reach target distance
        return self.robot_position[0] > 10.0
    
    def _check_failure(self) -> bool:
        """Check if robot has fallen."""
        return self.robot_position[2] < 0.1
    
    def _render_frame(self) -> np.ndarray:
        """Render current frame."""
        # Simple visualization
        frame = np.ones((224, 224, 3), dtype=np.uint8) * 150
        
        # Draw robot as circle
        robot_x = int(112 + self.robot_position[0] * 10)
        robot_y = int(112 - self.robot_position[2] * 50)
        
        if 0 <= robot_x < 224 and 0 <= robot_y < 224:
            frame[max(0, robot_y-10):min(224, robot_y+10),
                  max(0, robot_x-10):min(224, robot_x+10)] = [0, 100, 200]
        
        return frame