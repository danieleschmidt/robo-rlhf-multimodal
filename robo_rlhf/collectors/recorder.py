"""
Demonstration recorder for multimodal robotic data collection.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datetime import datetime
import threading
import queue
import logging

from robo_rlhf.collectors.base import DemonstrationData


class DemonstrationRecorder:
    """
    Advanced demonstration recorder with real-time processing.
    
    Supports multi-threaded recording, compression, and metadata enrichment.
    """
    
    def __init__(
        self,
        output_dir: str,
        compression: bool = True,
        buffer_size: int = 1000,
        metadata_enrichment: bool = True
    ):
        """
        Initialize demonstration recorder.
        
        Args:
            output_dir: Directory to save demonstrations
            compression: Whether to compress video data
            buffer_size: Buffer size for async recording
            metadata_enrichment: Whether to add rich metadata
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.compression = compression
        self.buffer_size = buffer_size
        self.metadata_enrichment = metadata_enrichment
        
        # Recording state
        self.recording = False
        self.current_episode = None
        self.episode_counter = 0
        
        # Async recording
        self.record_queue = queue.Queue(maxsize=buffer_size)
        self.record_thread = None
        self.stop_recording_thread = False
        
        # Statistics
        self.stats = {
            "total_episodes": 0,
            "total_duration": 0.0,
            "successful_episodes": 0,
            "failed_episodes": 0,
            "average_episode_length": 0.0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def start_episode(
        self,
        episode_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start recording a new episode.
        
        Args:
            episode_id: Custom episode ID (auto-generated if None)
            metadata: Additional metadata for the episode
            
        Returns:
            Episode ID
        """
        if self.recording:
            raise RuntimeError("Already recording an episode")
        
        # Generate episode ID if not provided
        if episode_id is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            episode_id = f"episode_{timestamp}_{self.episode_counter:04d}"
        
        self.current_episode = {
            "id": episode_id,
            "start_time": time.time(),
            "observations": [],
            "actions": [],
            "rewards": [],
            "metadata": metadata or {},
            "frame_count": 0
        }
        
        self.recording = True
        self.episode_counter += 1
        
        # Start async recording thread
        if self.record_thread is None or not self.record_thread.is_alive():
            self.stop_recording_thread = False
            self.record_thread = threading.Thread(
                target=self._async_recording_loop,
                daemon=True
            )
            self.record_thread.start()
        
        self.logger.info(f"Started recording episode: {episode_id}")
        return episode_id
    
    def record_step(
        self,
        observation: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float = 0.0,
        info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a single step of the episode.
        
        Args:
            observation: Environment observation
            action: Action taken
            reward: Reward received
            info: Additional step information
        """
        if not self.recording or self.current_episode is None:
            raise RuntimeError("No episode currently being recorded")
        
        step_data = {
            "timestamp": time.time(),
            "observation": observation,
            "action": action.copy(),
            "reward": reward,
            "info": info or {},
            "frame_idx": self.current_episode["frame_count"]
        }
        
        # Add to async queue
        try:
            self.record_queue.put_nowait(step_data)
            self.current_episode["frame_count"] += 1
        except queue.Full:
            self.logger.warning("Recording buffer full, dropping frame")
    
    def stop_episode(
        self,
        success: bool = True,
        final_metadata: Optional[Dict[str, Any]] = None
    ) -> DemonstrationData:
        """
        Stop recording and save the episode.
        
        Args:
            success: Whether the episode was successful
            final_metadata: Additional metadata to add
            
        Returns:
            Saved demonstration data
        """
        if not self.recording or self.current_episode is None:
            raise RuntimeError("No episode currently being recorded")
        
        # Stop recording
        self.recording = False
        
        # Wait for queue to empty
        self.record_queue.join()
        
        # Finalize episode
        end_time = time.time()
        duration = end_time - self.current_episode["start_time"]
        
        # Combine metadata
        metadata = self.current_episode["metadata"].copy()
        if final_metadata:
            metadata.update(final_metadata)
        
        if self.metadata_enrichment:
            metadata.update({
                "recording_settings": {
                    "compression": self.compression,
                    "buffer_size": self.buffer_size
                },
                "performance_stats": {
                    "total_frames": len(self.current_episode["observations"]),
                    "average_fps": len(self.current_episode["observations"]) / duration if duration > 0 else 0,
                    "dropped_frames": self.current_episode["frame_count"] - len(self.current_episode["observations"])
                }
            })
        
        # Create demonstration data
        demo = DemonstrationData(
            episode_id=self.current_episode["id"],
            timestamp=datetime.now().isoformat(),
            observations=self._combine_observations(self.current_episode["observations"]),
            actions=np.array(self.current_episode["actions"]),
            rewards=np.array(self.current_episode["rewards"]) if self.current_episode["rewards"] else None,
            success=success,
            duration=duration,
            metadata=metadata
        )
        
        # Save to disk
        episode_path = self.output_dir / self.current_episode["id"]
        demo.save(episode_path)
        
        # Update statistics
        self._update_stats(success, duration, len(self.current_episode["actions"]))
        
        # Clean up
        self.current_episode = None
        
        self.logger.info(f"Saved episode: {demo.episode_id} (success: {success}, duration: {duration:.2f}s)")
        return demo
    
    def _async_recording_loop(self) -> None:
        """Async recording loop to process queued data."""
        while not self.stop_recording_thread:
            try:
                # Get step data with timeout
                step_data = self.record_queue.get(timeout=0.1)
                
                if self.current_episode is not None:
                    # Add to episode buffers
                    self.current_episode["observations"].append(step_data["observation"])
                    self.current_episode["actions"].append(step_data["action"])
                    self.current_episode["rewards"].append(step_data["reward"])
                
                self.record_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in recording loop: {e}")
    
    def _combine_observations(self, obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Combine list of observations into arrays."""
        if not obs_list:
            return {}
        
        combined = {}
        for key in obs_list[0].keys():
            combined[key] = np.array([obs[key] for obs in obs_list])
        
        return combined
    
    def _update_stats(self, success: bool, duration: float, steps: int) -> None:
        """Update recording statistics."""
        self.stats["total_episodes"] += 1
        self.stats["total_duration"] += duration
        
        if success:
            self.stats["successful_episodes"] += 1
        else:
            self.stats["failed_episodes"] += 1
        
        # Update average episode length
        total_episodes = self.stats["total_episodes"]
        current_avg = self.stats["average_episode_length"]
        self.stats["average_episode_length"] = (
            (current_avg * (total_episodes - 1) + steps) / total_episodes
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get recording statistics."""
        stats = self.stats.copy()
        
        if stats["total_episodes"] > 0:
            stats["success_rate"] = stats["successful_episodes"] / stats["total_episodes"]
            stats["average_duration"] = stats["total_duration"] / stats["total_episodes"]
        else:
            stats["success_rate"] = 0.0
            stats["average_duration"] = 0.0
        
        return stats
    
    def load_demonstrations(
        self,
        episode_ids: Optional[List[str]] = None
    ) -> List[DemonstrationData]:
        """
        Load demonstrations from disk.
        
        Args:
            episode_ids: Specific episodes to load (all if None)
            
        Returns:
            List of loaded demonstrations
        """
        demonstrations = []
        
        if episode_ids is None:
            # Load all episodes
            episode_dirs = [d for d in self.output_dir.iterdir() if d.is_dir()]
        else:
            episode_dirs = [self.output_dir / eid for eid in episode_ids]
        
        for episode_dir in episode_dirs:
            if not episode_dir.exists():
                self.logger.warning(f"Episode directory not found: {episode_dir}")
                continue
            
            try:
                demo = self._load_demonstration(episode_dir)
                demonstrations.append(demo)
            except Exception as e:
                self.logger.error(f"Failed to load demonstration from {episode_dir}: {e}")
        
        return demonstrations
    
    def _load_demonstration(self, episode_dir: Path) -> DemonstrationData:
        """Load a single demonstration from directory."""
        # Load metadata
        metadata_path = episode_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load observations
        observations = {}
        for obs_file in episode_dir.glob("*.npy"):
            if obs_file.stem not in ["actions", "rewards"]:
                observations[obs_file.stem] = np.load(obs_file)
        
        # Load actions
        actions = np.load(episode_dir / "actions.npy")
        
        # Load rewards if available
        rewards_path = episode_dir / "rewards.npy"
        rewards = np.load(rewards_path) if rewards_path.exists() else None
        
        return DemonstrationData(
            episode_id=metadata["episode_id"],
            timestamp=metadata["timestamp"],
            observations=observations,
            actions=actions,
            rewards=rewards,
            success=metadata.get("success", False),
            duration=metadata.get("duration", 0.0),
            metadata=metadata.get("metadata", {})
        )
    
    def cleanup(self) -> None:
        """Clean up resources and stop recording."""
        self.recording = False
        self.stop_recording_thread = True
        
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=5.0)
        
        self.logger.info("Demonstration recorder cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()