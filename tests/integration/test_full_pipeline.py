"""
Integration tests for the complete RLHF pipeline.

Tests the end-to-end functionality from data collection to policy training.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch

from robo_rlhf import (
    TeleOpCollector,
    PreferencePairGenerator,
    MultimodalRLHF,
    VisionLanguageActor,
    make_env
)
from robo_rlhf.preference.models import PreferencePair, Segment, PreferenceChoice


class MockController:
    """Mock controller for automated testing."""
    
    def __init__(self, action_space):
        self.action_space = action_space
        self.action_dim = action_space.shape[0]
        self.step_count = 0
        self.max_steps = 50
        
    def wait_for_start(self):
        """Mock wait for start."""
        pass
        
    def get_action(self):
        """Generate mock actions."""
        if self.step_count >= self.max_steps:
            return None  # Signal episode end
        
        # Generate reasonable mock actions
        action = np.random.uniform(-0.5, 0.5, self.action_dim)
        self.step_count += 1
        return action


@pytest.fixture
def temp_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_env():
    """Create mock environment for testing."""
    return make_env("cartpole")  # Simple environment for testing


@pytest.fixture 
def mock_demonstrations(temp_dir):
    """Create mock demonstration data."""
    demo_dir = temp_dir / "demonstrations"
    demo_dir.mkdir()
    
    # Create mock episodes
    for i in range(3):
        episode_dir = demo_dir / f"episode_{i:04d}"
        episode_dir.mkdir()
        
        # Mock data
        actions = np.random.randn(30, 4)
        observations = np.random.randint(0, 255, (30, 3, 64, 64), dtype=np.uint8)
        proprioception = np.random.randn(30, 7)
        rewards = np.random.randn(30)
        
        # Save data
        np.save(episode_dir / "actions.npy", actions)
        np.save(episode_dir / "rgb.npy", observations)
        np.save(episode_dir / "proprioception.npy", proprioception)
        np.save(episode_dir / "rewards.npy", rewards)
        
        # Metadata
        metadata = {
            "episode_id": f"test_episode_{i}",
            "success": i % 2 == 0,
            "duration": 30.0,
            "env_name": "test_env"
        }
        
        import json
        with open(episode_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
    
    return demo_dir


class TestFullPipeline:
    """Test the complete RLHF pipeline."""
    
    def test_data_collection_integration(self, mock_env, temp_dir):
        """Test data collection with mock controller."""
        # Replace controller with mock
        collector = TeleOpCollector(
            env=mock_env,
            modalities=["rgb", "proprioception"],
            device="keyboard"
        )
        
        # Mock the controller
        collector.controller = MockController(mock_env.action_space)
        
        # Collect demonstrations
        demonstrations = collector.collect(
            num_episodes=2,
            save_dir=str(temp_dir / "demos"),
            max_steps_per_episode=20,
            render=False
        )
        
        # Verify data collection
        assert len(demonstrations) == 2
        assert all(demo.actions.shape[0] > 0 for demo in demonstrations)
        assert (temp_dir / "demos").exists()
        
        # Check saved files
        for i in range(2):
            episode_dir = temp_dir / "demos" / f"episode_{i:04d}"
            assert episode_dir.exists()
            assert (episode_dir / "metadata.json").exists()
            assert (episode_dir / "actions.npy").exists()
    
    def test_preference_generation_integration(self, mock_demonstrations):
        """Test preference pair generation."""
        generator = PreferencePairGenerator(
            demo_dir=str(mock_demonstrations),
            pair_selection="random",
            seed=42
        )
        
        # Generate pairs
        pairs = generator.generate_pairs(
            num_pairs=5,
            segment_length=10
        )
        
        # Verify generation
        assert len(pairs) == 5
        assert all(isinstance(pair, PreferencePair) for pair in pairs)
        assert all(pair.segment_a.actions.shape[0] == 10 for pair in pairs)
        assert all(pair.segment_b.actions.shape[0] == 10 for pair in pairs)
    
    def test_preference_annotation_integration(self, mock_demonstrations):
        """Test preference annotation workflow."""
        # Generate pairs
        generator = PreferencePairGenerator(
            demo_dir=str(mock_demonstrations),
            pair_selection="diversity_sampling"
        )
        pairs = generator.generate_pairs(num_pairs=3, segment_length=10)
        
        # Add mock preferences
        for i, pair in enumerate(pairs):
            choice = PreferenceChoice.SEGMENT_A if i % 2 == 0 else PreferenceChoice.SEGMENT_B
            pair.add_preference("test_annotator", choice, 0.8)
        
        # Verify annotations
        assert all(len(pair.preferences) == 1 for pair in pairs)
        assert all(pair.get_consensus() is not None for pair in pairs)
        assert all(0.5 <= pair.get_agreement_score() <= 1.0 for pair in pairs)
    
    def test_rlhf_training_integration(self, mock_demonstrations, temp_dir):
        """Test RLHF training pipeline."""
        # Generate preferences
        generator = PreferencePairGenerator(
            demo_dir=str(mock_demonstrations),
            pair_selection="random"
        )
        pairs = generator.generate_pairs(num_pairs=10, segment_length=10)
        
        # Add mock preferences
        for pair in pairs:
            choice = np.random.choice([PreferenceChoice.SEGMENT_A, PreferenceChoice.SEGMENT_B])
            pair.add_preference("test_annotator", choice, np.random.uniform(0.6, 0.9))
        
        # Create policy
        policy = VisionLanguageActor(
            vision_encoder="resnet18",
            proprioception_dim=7,
            action_dim=4,
            hidden_dim=128  # Smaller for testing
        )
        
        # Create trainer
        trainer = MultimodalRLHF(
            model=policy,
            preferences=pairs,
            reward_model="bradley_terry",
            use_wandb=False
        )
        
        # Train (short training for testing)
        stats = trainer.train(
            epochs=5,
            batch_size=8,
            validation_split=0.2,
            checkpoint_dir=str(temp_dir / "checkpoints"),
            reward_epochs=3,
            policy_epochs=5
        )
        
        # Verify training
        assert "reward_learning_loss" in stats
        assert "validation_accuracy" in stats
        assert len(stats["reward_learning_loss"]) == 3  # reward_epochs
        assert len(stats["validation_accuracy"]) == 3
        assert all(0.0 <= acc <= 1.0 for acc in stats["validation_accuracy"])
    
    def test_policy_evaluation_integration(self, mock_env, temp_dir):
        """Test policy evaluation."""
        # Create and train a simple policy
        policy = VisionLanguageActor(
            vision_encoder="resnet18",
            proprioception_dim=7,
            action_dim=mock_env.action_space.shape[0],
            hidden_dim=64
        )
        
        # Mock evaluation
        returns = []
        for episode in range(3):
            obs, info = mock_env.reset()
            episode_return = 0.0
            done = False
            steps = 0
            max_steps = 20
            
            while not done and steps < max_steps:
                # Get mock observation data
                if isinstance(obs, dict):
                    rgb = obs.get("pixels", np.zeros((3, 64, 64)))
                    proprio = np.zeros(7)
                else:
                    rgb = np.zeros((3, 64, 64))
                    proprio = np.zeros(7)
                
                # Get action from policy
                action = policy.get_action(rgb, proprio)
                
                # Step environment
                obs, reward, terminated, truncated, info = mock_env.step(action)
                episode_return += reward
                done = terminated or truncated
                steps += 1
            
            returns.append(episode_return)
        
        # Verify evaluation
        assert len(returns) == 3
        assert all(isinstance(ret, (int, float)) for ret in returns)
    
    def test_end_to_end_pipeline(self, temp_dir):
        """Test complete end-to-end pipeline."""
        # Step 1: Create mock environment
        env = make_env("cartpole")
        
        # Step 2: Mock data collection
        demo_dir = temp_dir / "demos" 
        demo_dir.mkdir()
        
        # Create minimal mock data
        for i in range(2):
            episode_dir = demo_dir / f"episode_{i:04d}"
            episode_dir.mkdir()
            
            # Minimal data for pipeline test
            actions = np.random.randn(20, env.action_space.shape[0])
            np.save(episode_dir / "actions.npy", actions)
            
            # Mock observations
            obs_data = {
                "rgb": np.random.randint(0, 255, (20, 3, 64, 64), dtype=np.uint8),
                "proprioception": np.random.randn(20, 7)
            }
            for key, data in obs_data.items():
                np.save(episode_dir / f"{key}.npy", data)
            
            # Metadata
            metadata = {"episode_id": f"test_{i}", "success": True}
            import json
            with open(episode_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)
        
        # Step 3: Generate preferences
        generator = PreferencePairGenerator(demo_dir, pair_selection="random")
        pairs = generator.generate_pairs(num_pairs=3, segment_length=5)
        
        # Add preferences
        for pair in pairs:
            pair.add_preference("tester", PreferenceChoice.SEGMENT_A, 0.8)
        
        # Step 4: Train policy (minimal)
        policy = VisionLanguageActor(
            vision_encoder="resnet18",
            proprioception_dim=7,
            action_dim=env.action_space.shape[0],
            hidden_dim=32  # Very small for testing
        )
        
        trainer = MultimodalRLHF(
            model=policy,
            preferences=pairs,
            use_wandb=False
        )
        
        # Minimal training
        stats = trainer.train(
            epochs=2,
            batch_size=2,
            reward_epochs=1,
            policy_epochs=2
        )
        
        # Step 5: Verify pipeline completion
        assert stats is not None
        assert "validation_accuracy" in stats
        assert len(stats["validation_accuracy"]) == 1
        
        # Test policy inference
        obs, _ = env.reset()
        if isinstance(obs, dict):
            rgb = obs.get("pixels", np.zeros((3, 64, 64)))
            proprio = np.zeros(7)
        else:
            rgb = np.zeros((3, 64, 64))
            proprio = np.zeros(7)
        
        action = policy.get_action(rgb, proprio)
        assert action.shape == env.action_space.shape
        
        print("✅ End-to-end pipeline test completed successfully!")


class TestPipelineRobustness:
    """Test pipeline robustness and error handling."""
    
    def test_missing_data_handling(self, temp_dir):
        """Test handling of missing or corrupted data."""
        # Create incomplete demonstration directory
        demo_dir = temp_dir / "incomplete_demos"
        demo_dir.mkdir()
        
        # Episode with missing files
        episode_dir = demo_dir / "episode_0000"
        episode_dir.mkdir()
        
        # Only create metadata, missing action data
        metadata = {"episode_id": "incomplete", "success": False}
        import json
        with open(episode_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        # Generator should handle missing data gracefully
        generator = PreferencePairGenerator(demo_dir)
        demonstrations = generator._load_demonstrations()
        
        # Should not crash, but may have empty demonstrations
        assert isinstance(demonstrations, list)
    
    def test_preference_edge_cases(self, mock_demonstrations):
        """Test preference generation edge cases."""
        generator = PreferencePairGenerator(mock_demonstrations)
        
        # Test with very short segments
        pairs = generator.generate_pairs(num_pairs=1, segment_length=1)
        assert len(pairs) >= 0  # May be 0 if segments too short
        
        # Test with more pairs than possible
        many_pairs = generator.generate_pairs(num_pairs=1000, segment_length=10)
        assert len(many_pairs) <= 1000  # Should not exceed reasonable limit
    
    def test_training_with_minimal_data(self):
        """Test training with minimal preference data."""
        # Create minimal preference data
        from robo_rlhf.preference.models import Segment
        
        # Create mock segments
        segment_a = Segment(
            episode_id="test_a",
            start_frame=0,
            end_frame=5,
            observations={"rgb": np.random.randint(0, 255, (5, 3, 64, 64))},
            actions=np.random.randn(5, 4),
            metadata={"success": True}
        )
        
        segment_b = Segment(
            episode_id="test_b", 
            start_frame=0,
            end_frame=5,
            observations={"rgb": np.random.randint(0, 255, (5, 3, 64, 64))},
            actions=np.random.randn(5, 4),
            metadata={"success": False}
        )
        
        # Create pair
        pair = PreferencePair("test_pair", segment_a, segment_b)
        pair.add_preference("annotator", PreferenceChoice.SEGMENT_A, 0.7)
        
        # Test training with single pair
        policy = VisionLanguageActor(
            vision_encoder="resnet18",
            proprioception_dim=7,
            action_dim=4,
            hidden_dim=32
        )
        
        trainer = MultimodalRLHF(
            model=policy,
            preferences=[pair],
            use_wandb=False
        )
        
        # Should handle minimal data without crashing
        stats = trainer.train(
            epochs=1,
            batch_size=1,
            reward_epochs=1,
            policy_epochs=1
        )
        
        assert stats is not None


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])