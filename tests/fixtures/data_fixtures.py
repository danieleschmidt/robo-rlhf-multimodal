"""Data fixtures for testing."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def create_sample_demonstration_data(data_dir: Path, num_episodes: int = 5) -> Path:
    """Create sample demonstration data for testing."""
    demo_dir = data_dir / "demonstrations"
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    for episode_id in range(num_episodes):
        episode_data = {
            "episode_id": episode_id,
            "observations": [],
            "actions": [],
            "rewards": [],
            "metadata": {
                "task": "test_task",
                "success": episode_id % 2 == 0,  # Half successful
                "total_reward": np.random.rand() * 100,
                "episode_length": 50 + np.random.randint(-10, 10),
            }
        }
        
        # Generate trajectory data
        for step in range(episode_data["metadata"]["episode_length"]):
            obs = {
                "rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "depth": np.random.rand(224, 224).astype(np.float32),
                "proprioception": np.random.rand(7).astype(np.float32),
                "timestamp": step * 0.1,
            }
            action = np.random.rand(7).astype(np.float32)
            reward = np.random.rand()
            
            episode_data["observations"].append(obs)
            episode_data["actions"].append(action)
            episode_data["rewards"].append(reward)
        
        # Save as pickle file
        episode_file = demo_dir / f"episode_{episode_id:03d}.pkl"
        with open(episode_file, "wb") as f:
            pickle.dump(episode_data, f)
    
    return demo_dir


def create_sample_preference_data(data_dir: Path, num_pairs: int = 100) -> Path:
    """Create sample preference data for testing."""
    preference_dir = data_dir / "preferences"
    preference_dir.mkdir(parents=True, exist_ok=True)
    
    preferences = []
    for pair_id in range(num_pairs):
        preference = {
            "pair_id": pair_id,
            "trajectory_1": {
                "episode_id": np.random.randint(0, 5),
                "start_idx": np.random.randint(0, 20),
                "end_idx": np.random.randint(30, 50),
            },
            "trajectory_2": {
                "episode_id": np.random.randint(0, 5), 
                "start_idx": np.random.randint(0, 20),
                "end_idx": np.random.randint(30, 50),
            },
            "preference": np.random.choice([0, 1, -1]),  # 0: traj1, 1: traj2, -1: equal
            "confidence": np.random.rand(),
            "annotator_id": f"annotator_{np.random.randint(0, 3)}",
            "annotation_time": np.random.rand() * 60,  # seconds
            "metadata": {
                "task": "test_task",
                "difficulty": np.random.choice(["easy", "medium", "hard"]),
            }
        }
        preferences.append(preference)
    
    # Save preferences
    preference_file = preference_dir / "preferences.json"
    with open(preference_file, "w") as f:
        json.dump(preferences, f, indent=2)
    
    return preference_dir


def create_sample_model_checkpoint(data_dir: Path) -> Path:
    """Create sample model checkpoint for testing."""
    checkpoint_dir = data_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Mock model state dict
    model_state = {
        "vision_encoder.weight": np.random.randn(512, 1024).astype(np.float32),
        "vision_encoder.bias": np.random.randn(512).astype(np.float32),
        "policy_head.weight": np.random.randn(7, 512).astype(np.float32),
        "policy_head.bias": np.random.randn(7).astype(np.float32),
    }
    
    checkpoint = {
        "model_state_dict": model_state,
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "epoch": 50,
        "train_loss": 0.123,
        "val_loss": 0.145,
        "config": {
            "model": {"hidden_dim": 512, "action_dim": 7},
            "training": {"batch_size": 32, "learning_rate": 3e-4},
        },
        "metadata": {
            "training_time": 3600,  # 1 hour
            "num_parameters": sum(p.size for p in model_state.values()),
        }
    }
    
    checkpoint_file = checkpoint_dir / "best_model.pkl"
    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint, f)
    
    return checkpoint_file


def create_sample_config_file(data_dir: Path) -> Path:
    """Create sample configuration file for testing."""
    config_dir = data_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        "experiment": {
            "name": "test_experiment",
            "description": "Test configuration for unit tests",
            "seed": 42,
        },
        "model": {
            "type": "multimodal_policy",
            "vision_encoder": "clip_vit_b32",
            "proprioception_dim": 7,
            "action_dim": 7,
            "hidden_dim": 512,
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 3e-4,
            "num_epochs": 100,
            "optimizer": "adamw",
            "scheduler": "cosine",
        },
        "data": {
            "demonstration_dir": "demonstrations/",
            "preference_dir": "preferences/", 
            "num_preference_pairs": 1000,
            "segment_length": 50,
        },
        "evaluation": {
            "num_episodes": 10,
            "max_episode_length": 1000,
            "save_videos": False,
        },
        "environment": {
            "type": "mujoco_manipulation",
            "task": "pick_and_place",
            "render": False,
        }
    }
    
    config_file = config_dir / "test_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    
    return config_file


class MockDataLoader:
    """Mock data loader for testing."""
    
    def __init__(self, data: List[Dict[str, Any]], batch_size: int):
        self.data = data
        self.batch_size = batch_size
        self.current_idx = 0
    
    def __iter__(self):
        self.current_idx = 0
        return self
    
    def __next__(self):
        if self.current_idx >= len(self.data):
            raise StopIteration
        
        batch_end = min(self.current_idx + self.batch_size, len(self.data))
        batch = self.data[self.current_idx:batch_end]
        self.current_idx = batch_end
        
        return batch
    
    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size


def create_mock_dataset(num_samples: int = 1000) -> List[Dict[str, Any]]:
    """Create mock dataset for testing."""
    dataset = []
    for i in range(num_samples):
        sample = {
            "id": i,
            "observation": {
                "rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "proprioception": np.random.rand(7).astype(np.float32),
            },
            "action": np.random.rand(7).astype(np.float32),
            "reward": np.random.rand(),
            "next_observation": {
                "rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "proprioception": np.random.rand(7).astype(np.float32),
            },
            "done": np.random.rand() < 0.05,  # 5% chance of episode end
        }
        dataset.append(sample)
    
    return dataset