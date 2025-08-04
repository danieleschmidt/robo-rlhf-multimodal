"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# Optional PyTorch import
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Create mock torch for tests that don't need it
    torch = MagicMock()
    torch.cuda.is_available.return_value = False

# Test configuration
os.environ["TESTING"] = "1"
os.environ["LOG_LEVEL"] = "DEBUG"


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(scope="session") 
def device() -> torch.device:
    """Test device (CPU for CI, GPU if available locally)."""
    if torch.cuda.is_available() and not os.getenv("CI"):
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture
def random_seed() -> int:
    """Random seed for reproducible tests."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed


@pytest.fixture
def sample_observation() -> Dict[str, Any]:
    """Sample multimodal observation for testing."""
    return {
        "rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "depth": np.random.rand(224, 224).astype(np.float32),
        "proprioception": np.random.rand(7).astype(np.float32),
        "force": np.random.rand(6).astype(np.float32),
    }


@pytest.fixture
def sample_action() -> np.ndarray:
    """Sample robot action for testing."""
    return np.random.rand(7).astype(np.float32)


@pytest.fixture
def sample_trajectory(sample_observation: Dict[str, Any], sample_action: np.ndarray) -> Dict[str, Any]:
    """Sample trajectory for testing."""
    length = 50
    trajectory = {
        "observations": [],
        "actions": [],
        "rewards": np.random.rand(length).astype(np.float32),
        "dones": np.zeros(length, dtype=bool),
        "info": {},
    }
    
    for _ in range(length):
        # Vary observations slightly
        obs = {}
        for key, value in sample_observation.items():
            noise = np.random.normal(0, 0.1, value.shape).astype(value.dtype)
            obs[key] = np.clip(value + noise, 0, 255 if key == "rgb" else None)
        trajectory["observations"].append(obs)
        
        # Vary actions slightly
        action_noise = np.random.normal(0, 0.05, sample_action.shape)
        trajectory["actions"].append(sample_action + action_noise)
    
    trajectory["dones"][-1] = True  # Mark trajectory as complete
    return trajectory


@pytest.fixture
def sample_preference_pair(sample_trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Sample preference pair for testing."""
    # Create two slightly different trajectories
    traj1 = sample_trajectory.copy()
    traj2 = sample_trajectory.copy()
    
    # Make second trajectory slightly different
    traj2["rewards"] = traj2["rewards"] + np.random.normal(0, 0.1, len(traj2["rewards"]))
    
    return {
        "trajectory_1": traj1,
        "trajectory_2": traj2,
        "preference": 0,  # Prefer trajectory 1
        "confidence": 0.8,
        "annotator_id": "test_annotator",
        "metadata": {"task": "test_task", "segment_length": 50},
    }


@pytest.fixture
def mock_environment():
    """Mock environment for testing."""
    env = MagicMock()
    env.observation_space.spaces = {
        "rgb": MagicMock(shape=(224, 224, 3)),
        "depth": MagicMock(shape=(224, 224)),
        "proprioception": MagicMock(shape=(7,)),
    }
    env.action_space.shape = (7,)
    env.reset.return_value = {
        "rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "depth": np.random.rand(224, 224).astype(np.float32),
        "proprioception": np.random.rand(7).astype(np.float32),
    }
    env.step.return_value = (
        env.reset.return_value,  # observation
        0.0,  # reward
        False,  # done
        {},  # info
    )
    return env


@pytest.fixture
def mock_model(device: torch.device):
    """Mock neural network model for testing."""
    model = MagicMock()
    model.device = device
    model.forward.return_value = torch.randn(1, 7, device=device)  # Sample action
    model.parameters.return_value = [torch.randn(100, device=device)]
    model.state_dict.return_value = {"test_param": torch.randn(100, device=device)}
    return model


@pytest.fixture
def config_dict() -> Dict[str, Any]:
    """Sample configuration dictionary."""
    return {
        "model": {
            "vision_encoder": "clip_vit_b32",
            "hidden_dim": 512,
            "action_dim": 7,
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 3e-4,
            "num_epochs": 10,
        },
        "data": {
            "min_trajectory_length": 10,
            "max_trajectory_length": 1000,
        },
        "evaluation": {
            "num_episodes": 10,
            "max_episode_length": 1000,
        },
    }


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment variables and configuration."""
    # Ensure test mode
    os.environ["TESTING"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if torch.cuda.is_available() else ""
    
    # Suppress warnings for cleaner test output
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    yield
    
    # Cleanup after all tests
    if "TESTING" in os.environ:
        del os.environ["TESTING"]


@pytest.fixture
def mock_mujoco_env():
    """Mock MuJoCo environment for testing."""
    with patch("robo_rlhf.envs.mujoco.MujocoManipulation") as mock:
        env = MagicMock()
        env.reset.return_value = {
            "rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "proprioception": np.random.rand(7).astype(np.float32),
        }
        env.step.return_value = (env.reset.return_value, 0.0, False, {})
        mock.return_value = env
        yield env


@pytest.fixture  
def mock_isaac_env():
    """Mock Isaac Sim environment for testing."""
    with patch("robo_rlhf.envs.isaac.IsaacSimulation") as mock:
        env = MagicMock()
        env.reset.return_value = {
            "rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "depth": np.random.rand(224, 224).astype(np.float32),
            "proprioception": np.random.rand(7).astype(np.float32),
        }
        env.step.return_value = (env.reset.return_value, 0.0, False, {})
        mock.return_value = env
        yield env


@pytest.fixture
def performance_benchmark():
    """Fixture for performance benchmarking tests."""
    import time
    
    class Benchmark:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            self.end_time = time.perf_counter()
            return self.end_time - self.start_time
        
        @property 
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Benchmark()


# Pytest hooks for custom test behavior
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "mujoco: mark test as requiring MuJoCo"
    )
    config.addinivalue_line(
        "markers", "isaac: mark test as requiring Isaac Sim"
    )
    config.addinivalue_line(
        "markers", "ros: mark test as requiring ROS2"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test items based on markers and environment."""
    # Skip GPU tests if no GPU available
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
    
    # Skip simulation tests in CI unless explicitly enabled
    if os.getenv("CI") and not os.getenv("RUN_SIM_TESTS"):
        skip_sim = pytest.mark.skip(reason="Simulation tests disabled in CI")
        for item in items:
            if "mujoco" in item.keywords or "isaac" in item.keywords:
                item.add_marker(skip_sim)
    
    # Skip ROS tests if ROS not available
    try:
        import rclpy
    except ImportError:
        skip_ros = pytest.mark.skip(reason="ROS2 not available")
        for item in items:
            if "ros" in item.keywords:
                item.add_marker(skip_ros)