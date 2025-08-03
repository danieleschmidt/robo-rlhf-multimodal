"""
Base teleoperation collector for recording robot demonstrations.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
import cv2
from datetime import datetime
import threading
import queue

@dataclass
class DemonstrationData:
    """Container for a single demonstration."""
    episode_id: str
    timestamp: str
    observations: Dict[str, np.ndarray]
    actions: np.ndarray
    rewards: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    success: bool = False
    duration: float = 0.0
    
    def save(self, path: Path) -> None:
        """Save demonstration to disk."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        meta = {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "success": self.success,
            "duration": self.duration,
            "metadata": self.metadata or {}
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        # Save observations
        for key, obs in self.observations.items():
            if key in ["rgb", "depth"]:
                # Save images/videos
                video_path = path / f"{key}.mp4"
                self._save_video(obs, video_path)
            else:
                # Save numerical data
                np.save(path / f"{key}.npy", obs)
        
        # Save actions and rewards
        np.save(path / "actions.npy", self.actions)
        if self.rewards is not None:
            np.save(path / "rewards.npy", self.rewards)
    
    def _save_video(self, frames: np.ndarray, path: Path) -> None:
        """Save image sequence as video."""
        if len(frames.shape) == 3:
            frames = frames[np.newaxis, ...]
        
        height, width = frames.shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(path), fourcc, 30.0, (width, height))
        
        for frame in frames:
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = frame[:, :, :3]
            out.write(frame.astype(np.uint8))
        out.release()


class TeleOpCollector:
    """
    Main interface for collecting teleoperation demonstrations.
    
    Supports multiple input devices and environments.
    """
    
    def __init__(
        self,
        env,
        modalities: List[str] = ["rgb", "proprioception"],
        device: str = "keyboard",
        recording_fps: int = 30,
        buffer_size: int = 10000
    ):
        """
        Initialize teleoperation collector.
        
        Args:
            env: Gymnasium-compatible environment
            modalities: List of observation modalities to record
            device: Input device type ('keyboard', 'spacemouse', 'vr_controller')
            recording_fps: Target recording framerate
            buffer_size: Maximum buffer size for observations
        """
        self.env = env
        self.modalities = modalities
        self.device_type = device
        self.fps = recording_fps
        self.buffer_size = buffer_size
        
        # Initialize buffers
        self.obs_buffer = {mod: queue.Queue(maxsize=buffer_size) for mod in modalities}
        self.action_buffer = queue.Queue(maxsize=buffer_size)
        self.reward_buffer = queue.Queue(maxsize=buffer_size)
        
        # Initialize controller
        self._init_controller(device)
        
        # Recording state
        self.recording = False
        self.current_episode = None
        self.episodes = []
        
    def _init_controller(self, device: str) -> None:
        """Initialize the input controller."""
        if device == "keyboard":
            from robo_rlhf.collectors.devices import KeyboardController
            self.controller = KeyboardController(self.env.action_space)
        elif device == "spacemouse":
            from robo_rlhf.collectors.devices import SpaceMouseController
            self.controller = SpaceMouseController(self.env.action_space)
        elif device == "vr_controller":
            from robo_rlhf.collectors.devices import VRController
            self.controller = VRController(self.env.action_space)
        else:
            raise ValueError(f"Unknown device type: {device}")
    
    def collect(
        self,
        num_episodes: int,
        save_dir: Optional[str] = None,
        max_steps_per_episode: int = 1000,
        render: bool = True
    ) -> List[DemonstrationData]:
        """
        Collect demonstrations through teleoperation.
        
        Args:
            num_episodes: Number of episodes to collect
            save_dir: Directory to save demonstrations
            max_steps_per_episode: Maximum steps per episode
            render: Whether to render the environment
            
        Returns:
            List of collected demonstrations
        """
        demonstrations = []
        save_path = Path(save_dir) if save_dir else None
        
        print(f"Starting teleoperation collection for {num_episodes} episodes")
        print(f"Using {self.device_type} controller")
        print("Press SPACE to start/stop recording, ESC to finish episode")
        
        for episode_idx in range(num_episodes):
            print(f"\n--- Episode {episode_idx + 1}/{num_episodes} ---")
            demo = self._collect_episode(
                episode_idx,
                max_steps_per_episode,
                render
            )
            
            if demo is not None:
                demonstrations.append(demo)
                
                # Save demonstration
                if save_path:
                    episode_path = save_path / f"episode_{episode_idx:04d}"
                    demo.save(episode_path)
                    print(f"Saved demonstration to {episode_path}")
                
                # Show statistics
                print(f"Episode completed - Success: {demo.success}, "
                      f"Duration: {demo.duration:.2f}s, "
                      f"Steps: {len(demo.actions)}")
        
        print(f"\nCollection complete! Collected {len(demonstrations)} demonstrations")
        return demonstrations
    
    def _collect_episode(
        self,
        episode_idx: int,
        max_steps: int,
        render: bool
    ) -> Optional[DemonstrationData]:
        """Collect a single episode."""
        # Reset environment
        obs, info = self.env.reset()
        
        # Initialize episode data
        episode_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{episode_idx:04d}"
        observations = {mod: [] for mod in self.modalities}
        actions = []
        rewards = []
        
        # Wait for user to start
        print("Press SPACE to start recording...")
        self.controller.wait_for_start()
        
        start_time = time.time()
        step = 0
        done = False
        
        while not done and step < max_steps:
            # Get action from controller
            action = self.controller.get_action()
            
            if action is None:  # User requested to stop
                break
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Record data
            for mod in self.modalities:
                if mod in obs:
                    observations[mod].append(obs[mod])
                elif mod == "rgb" and "pixels" in obs:
                    observations[mod].append(obs["pixels"])
                elif mod == "proprioception" and "robot_state" in obs:
                    observations[mod].append(obs["robot_state"])
            
            actions.append(action)
            rewards.append(reward)
            
            # Render if requested
            if render:
                self.env.render()
            
            # Update for next iteration
            obs = next_obs
            step += 1
            
            # Control recording rate
            time.sleep(1.0 / self.fps)
        
        duration = time.time() - start_time
        
        # Convert lists to arrays
        for mod in observations:
            if observations[mod]:
                observations[mod] = np.array(observations[mod])
        
        if not actions:
            print("No actions recorded, skipping episode")
            return None
        
        return DemonstrationData(
            episode_id=episode_id,
            timestamp=datetime.now().isoformat(),
            observations=observations,
            actions=np.array(actions),
            rewards=np.array(rewards),
            success=info.get("success", done and not truncated),
            duration=duration,
            metadata={
                "env_name": self.env.spec.id if hasattr(self.env, "spec") else "unknown",
                "device": self.device_type,
                "fps": self.fps,
                "steps": step
            }
        )
    
    def replay(self, demonstration: DemonstrationData, render: bool = True) -> None:
        """Replay a collected demonstration."""
        print(f"Replaying demonstration {demonstration.episode_id}")
        
        obs, _ = self.env.reset()
        
        for step, action in enumerate(demonstration.actions):
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if render:
                self.env.render()
            
            if terminated or truncated:
                break
            
            time.sleep(1.0 / self.fps)
        
        print(f"Replay complete - {step + 1} steps")