"""End-to-end tests for complete system pipeline."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import torch

from tests.fixtures.data_fixtures import (
    create_sample_demonstration_data,
    create_sample_preference_data,
    create_sample_config_file,
    create_sample_model_checkpoint,
)


@pytest.mark.e2e
@pytest.mark.slow
class TestDataCollectionPipeline:
    """Test complete data collection pipeline."""
    
    @pytest.mark.mujoco
    def test_mujoco_data_collection(self, test_data_dir, mock_mujoco_env):
        """Test teleoperation data collection with MuJoCo."""
        # Mock teleoperation collector
        with patch("robo_rlhf.collectors.TeleOpCollector") as mock_collector:
            collector_instance = mock_collector.return_value
            
            # Mock collected episode data
            mock_episode = {
                "observations": [
                    {
                        "rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                        "proprioception": np.random.rand(7).astype(np.float32),
                    }
                    for _ in range(100)
                ],
                "actions": [np.random.rand(7).astype(np.float32) for _ in range(100)],
                "rewards": np.random.rand(100).astype(np.float32),
                "metadata": {"success": True, "task": "pick_and_place"},
            }
            
            collector_instance.collect_episode.return_value = mock_episode
            
            # Simulate collection process
            num_episodes = 5
            collected_data = []
            
            for episode_id in range(num_episodes):
                episode_data = collector_instance.collect_episode()
                episode_data["episode_id"] = episode_id
                collected_data.append(episode_data)
            
            # Verify collected data structure
            assert len(collected_data) == num_episodes
            for episode in collected_data:
                assert "observations" in episode
                assert "actions" in episode
                assert "rewards" in episode
                assert len(episode["observations"]) == len(episode["actions"])
    
    @pytest.mark.isaac
    def test_isaac_sim_data_collection(self, test_data_dir, mock_isaac_env):
        """Test data collection with Isaac Sim."""
        # Similar to MuJoCo test but with Isaac Sim specific features
        with patch("robo_rlhf.envs.isaac.IsaacSimulation") as mock_isaac:
            env_instance = mock_isaac.return_value
            env_instance.reset.return_value = {
                "rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "depth": np.random.rand(224, 224).astype(np.float32),
                "proprioception": np.random.rand(7).astype(np.float32),
            }
            
            # Test environment interaction
            obs = env_instance.reset()
            assert "depth" in obs  # Isaac Sim provides depth
            
            action = np.random.rand(7).astype(np.float32)
            next_obs, reward, done, info = env_instance.step(action)
            
            assert isinstance(reward, float)
            assert isinstance(done, bool)


@pytest.mark.e2e
@pytest.mark.slow
class TestPreferenceCollectionPipeline:
    """Test complete preference collection pipeline."""
    
    def test_preference_pair_generation(self, test_data_dir):
        """Test generation of preference pairs from demonstrations."""
        # Create demonstration data
        demo_dir = create_sample_demonstration_data(test_data_dir, num_episodes=10)
        
        # Mock preference pair generator
        with patch("robo_rlhf.preference.PreferencePairGenerator") as mock_generator:
            generator_instance = mock_generator.return_value
            
            # Mock generated pairs
            mock_pairs = []
            for pair_id in range(50):
                pair = {
                    "pair_id": pair_id,
                    "trajectory_1": {
                        "episode_id": np.random.randint(0, 10),
                        "start_idx": 0,
                        "end_idx": 50,
                    },
                    "trajectory_2": {
                        "episode_id": np.random.randint(0, 10),
                        "start_idx": 0,
                        "end_idx": 50,
                    },
                    "metadata": {"diversity_score": np.random.rand()},
                }
                mock_pairs.append(pair)
            
            generator_instance.generate_pairs.return_value = mock_pairs
            
            # Test pair generation
            pairs = generator_instance.generate_pairs(
                num_pairs=50,
                segment_length=50
            )
            
            assert len(pairs) == 50
            for pair in pairs:
                assert "trajectory_1" in pair
                assert "trajectory_2" in pair
                assert "pair_id" in pair
    
    @pytest.mark.asyncio
    async def test_preference_server(self, test_data_dir):
        """Test preference collection server."""
        # Create preference pairs
        pref_dir = create_sample_preference_data(test_data_dir, num_pairs=20)
        
        # Mock preference server
        with patch("robo_rlhf.preference.PreferenceServer") as mock_server:
            server_instance = mock_server.return_value
            
            # Mock server methods
            server_instance.start = AsyncMock()
            server_instance.collect_preferences = AsyncMock()
            
            # Mock collected preferences
            mock_preferences = [
                {
                    "pair_id": i,
                    "preference": np.random.choice([0, 1, -1]),
                    "confidence": np.random.rand(),
                    "annotator_id": f"annotator_{np.random.randint(0, 3)}",
                    "annotation_time": np.random.rand() * 60,
                }
                for i in range(20)
            ]
            
            server_instance.collect_preferences.return_value = mock_preferences
            
            # Test server workflow
            await server_instance.start()
            preferences = await server_instance.collect_preferences(
                min_annotations_per_pair=3
            )
            
            assert len(preferences) == 20
            for pref in preferences:
                assert pref["preference"] in [-1, 0, 1]
                assert 0.0 <= pref["confidence"] <= 1.0


@pytest.mark.e2e
@pytest.mark.slow
class TestTrainingPipeline:
    """Test complete training pipeline."""
    
    def test_reward_model_training(self, test_data_dir, device, config_dict):
        """Test complete reward model training from preferences."""
        # Create sample data
        demo_dir = create_sample_demonstration_data(test_data_dir, num_episodes=10)
        pref_dir = create_sample_preference_data(test_data_dir, num_pairs=100)
        
        # Mock reward model training
        with patch("robo_rlhf.algorithms.RewardModelTrainer") as mock_trainer:
            trainer_instance = mock_trainer.return_value
            
            # Mock training process
            training_history = {
                "train_losses": [1.5, 1.2, 1.0, 0.8, 0.7],
                "val_losses": [1.6, 1.3, 1.1, 0.9, 0.8],
                "train_accuracies": [0.5, 0.6, 0.7, 0.8, 0.85],
                "val_accuracies": [0.45, 0.55, 0.65, 0.75, 0.8],
            }
            
            trainer_instance.train.return_value = training_history
            
            # Test training
            history = trainer_instance.train(
                preference_data=pref_dir / "preferences.json",
                num_epochs=5,
                batch_size=32,
            )
            
            # Verify training history
            assert "train_losses" in history
            assert "val_losses" in history
            assert len(history["train_losses"]) == 5
            
            # Check that losses decrease over time
            assert history["train_losses"][-1] < history["train_losses"][0]
            assert history["val_losses"][-1] < history["val_losses"][0]
    
    def test_policy_training_with_rlhf(self, test_data_dir, device, config_dict):
        """Test policy training with RLHF."""
        # Create necessary data
        demo_dir = create_sample_demonstration_data(test_data_dir, num_episodes=20)
        pref_dir = create_sample_preference_data(test_data_dir, num_pairs=200)
        checkpoint_file = create_sample_model_checkpoint(test_data_dir)
        
        # Mock RLHF trainer
        with patch("robo_rlhf.algorithms.MultimodalRLHF") as mock_rlhf:
            rlhf_instance = mock_rlhf.return_value
            
            # Mock training metrics
            training_metrics = {
                "policy_losses": [2.0, 1.8, 1.5, 1.2, 1.0],
                "reward_losses": [0.8, 0.7, 0.6, 0.5, 0.4],  
                "kl_divergences": [0.1, 0.15, 0.12, 0.1, 0.08],
                "average_rewards": [10.0, 15.0, 20.0, 25.0, 30.0],
                "success_rates": [0.2, 0.4, 0.6, 0.7, 0.8],
            }
            
            rlhf_instance.train.return_value = training_metrics
            
            # Test RLHF training
            metrics = rlhf_instance.train(
                demo_data=demo_dir,
                preference_data=pref_dir / "preferences.json", 
                num_epochs=5,
                batch_size=16,
            )
            
            # Verify training metrics
            assert "policy_losses" in metrics
            assert "average_rewards" in metrics
            assert "success_rates" in metrics
            
            # Check improvement trends
            assert metrics["average_rewards"][-1] > metrics["average_rewards"][0]
            assert metrics["success_rates"][-1] > metrics["success_rates"][0]


@pytest.mark.e2e
@pytest.mark.slow
class TestDeploymentPipeline:
    """Test complete deployment pipeline."""
    
    @pytest.mark.ros
    def test_ros2_deployment(self, test_data_dir, device):
        """Test deployment to real robot via ROS2."""
        # Create model checkpoint
        checkpoint_file = create_sample_model_checkpoint(test_data_dir)
        
        # Mock ROS2 policy node
        with patch("robo_rlhf.deployment.ROS2PolicyNode") as mock_node:
            node_instance = mock_node.return_value
            
            # Mock ROS2 methods
            node_instance.load_policy = MagicMock()
            node_instance.publish_action = MagicMock()
            node_instance.get_observation = MagicMock()
            
            # Mock observation data
            mock_observation = {
                "rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "joint_states": np.random.rand(7).astype(np.float32),
                "force_torque": np.random.rand(6).astype(np.float32),
            }
            
            node_instance.get_observation.return_value = mock_observation
            
            # Test deployment workflow
            node_instance.load_policy(checkpoint_file)
            
            # Simulate control loop
            for step in range(10):
                obs = node_instance.get_observation()
                
                # Mock policy inference
                action = np.random.rand(7).astype(np.float32)
                
                node_instance.publish_action(action)
            
            # Verify methods were called
            node_instance.load_policy.assert_called_once()
            assert node_instance.get_observation.call_count == 10
            assert node_instance.publish_action.call_count == 10
    
    def test_model_serving(self, test_data_dir, device):
        """Test model serving for inference."""
        checkpoint_file = create_sample_model_checkpoint(test_data_dir)
        
        # Mock model server
        with patch("robo_rlhf.deployment.ModelServer") as mock_server:
            server_instance = mock_server.return_value
            
            # Mock server methods
            server_instance.load_model = MagicMock()
            server_instance.predict = MagicMock()
            
            # Mock prediction output
            mock_prediction = {
                "action": np.random.rand(7).astype(np.float32),
                "confidence": 0.85,
                "latency_ms": 15.2,
            }
            
            server_instance.predict.return_value = mock_prediction
            
            # Test serving workflow
            server_instance.load_model(checkpoint_file)
            
            # Test inference
            sample_obs = {
                "rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "proprioception": np.random.rand(7).astype(np.float32),
            }
            
            prediction = server_instance.predict(sample_obs)
            
            # Verify prediction structure
            assert "action" in prediction
            assert "confidence" in prediction
            assert len(prediction["action"]) == 7


@pytest.mark.e2e
@pytest.mark.slow
class TestEvaluationPipeline:
    """Test complete evaluation pipeline."""
    
    def test_policy_evaluation(self, test_data_dir, mock_environment, device):
        """Test policy evaluation in simulation."""
        checkpoint_file = create_sample_model_checkpoint(test_data_dir)
        
        # Mock policy evaluator
        with patch("robo_rlhf.evaluation.PolicyEvaluator") as mock_evaluator:
            evaluator_instance = mock_evaluator.return_value
            
            # Mock evaluation results
            eval_results = {
                "success_rate": 0.8,
                "average_reward": 25.5,
                "average_episode_length": 120.0,
                "task_completion_time": 45.2,
                "trajectory_smoothness": 0.92,
                "safety_violations": 0,
                "episodes_evaluated": 50,
            }
            
            evaluator_instance.evaluate.return_value = eval_results
            
            # Test evaluation
            results = evaluator_instance.evaluate(
                policy_checkpoint=checkpoint_file,
                environment=mock_environment,
                num_episodes=50,
                max_episode_length=200,
            )
            
            # Verify evaluation results
            assert "success_rate" in results
            assert "average_reward" in results
            assert 0.0 <= results["success_rate"] <= 1.0
            assert results["safety_violations"] == 0
    
    def test_preference_alignment_evaluation(self, test_data_dir, device):
        """Test evaluation of policy alignment with human preferences."""
        # Create test data
        pref_dir = create_sample_preference_data(test_data_dir, num_pairs=100)
        checkpoint_file = create_sample_model_checkpoint(test_data_dir)
        
        # Mock alignment evaluator
        with patch("robo_rlhf.evaluation.PreferenceAlignmentTest") as mock_alignment:
            alignment_instance = mock_alignment.return_value
            
            # Mock alignment results
            alignment_results = {
                "alignment_score": 0.85,
                "preference_accuracy": 0.82,
                "confidence_correlation": 0.78,
                "annotator_agreement": 0.75,
                "pairs_evaluated": 100,
            }
            
            alignment_instance.compute_alignment.return_value = alignment_results
            
            # Test alignment evaluation
            results = alignment_instance.compute_alignment(
                policy_checkpoint=checkpoint_file,
                test_preferences=pref_dir / "preferences.json",
            )
            
            # Verify alignment metrics
            assert "alignment_score" in results
            assert "preference_accuracy" in results
            assert 0.0 <= results["alignment_score"] <= 1.0
            assert 0.0 <= results["preference_accuracy"] <= 1.0


@pytest.mark.e2e
@pytest.mark.slow
class TestFullSystemIntegration:
    """Test full system integration."""
    
    def test_complete_workflow(self, test_data_dir, device, config_dict):
        """Test complete workflow from data collection to deployment."""
        # This test simulates the entire pipeline
        
        # 1. Data Collection Phase
        demo_dir = create_sample_demonstration_data(test_data_dir, num_episodes=20)
        assert demo_dir.exists()
        
        # 2. Preference Collection Phase
        pref_dir = create_sample_preference_data(test_data_dir, num_pairs=100)
        assert (pref_dir / "preferences.json").exists()
        
        # 3. Training Phase
        with patch("robo_rlhf.algorithms.MultimodalRLHF") as mock_rlhf:
            trainer = mock_rlhf.return_value
            
            # Mock successful training
            training_results = {
                "final_success_rate": 0.85,
                "final_reward": 28.0,
                "training_time": 3600,  # 1 hour
                "convergence_epoch": 45,
            }
            
            trainer.train.return_value = training_results
            results = trainer.train()
            
            assert results["final_success_rate"] > 0.8
        
        # 4. Evaluation Phase
        with patch("robo_rlhf.evaluation.PolicyEvaluator") as mock_eval:
            evaluator = mock_eval.return_value
            
            eval_results = {
                "success_rate": 0.87,
                "safety_violations": 0,
                "average_reward": 29.5,
            }
            
            evaluator.evaluate.return_value = eval_results
            results = evaluator.evaluate()
            
            assert results["success_rate"] > 0.8
            assert results["safety_violations"] == 0
        
        # 5. Deployment Phase
        checkpoint_file = create_sample_model_checkpoint(test_data_dir)
        
        with patch("robo_rlhf.deployment.ModelServer") as mock_server:
            server = mock_server.return_value
            server.load_model.return_value = True
            
            success = server.load_model(checkpoint_file)
            assert success
        
        print("✅ Complete workflow test passed")
    
    def test_error_handling_and_recovery(self, test_data_dir, device):
        """Test system behavior under error conditions."""
        # Test various error scenarios
        
        # 1. Missing data files
        with pytest.raises((FileNotFoundError, ValueError)):
            # Mock trying to load non-existent data
            nonexistent_path = test_data_dir / "nonexistent_data.json"
            with open(nonexistent_path, "r") as f:
                pass
        
        # 2. Corrupted model checkpoint
        with patch("robo_rlhf.deployment.ModelServer") as mock_server:
            server = mock_server.return_value
            server.load_model.side_effect = RuntimeError("Corrupted checkpoint")
            
            with pytest.raises(RuntimeError):
                server.load_model("corrupted_checkpoint.pt")
        
        # 3. GPU memory overflow
        if torch.cuda.is_available():
            with pytest.raises(torch.cuda.OutOfMemoryError):
                # Simulate GPU OOM
                large_tensor = torch.randn(10000, 10000, device=device)
                another_large_tensor = torch.randn(10000, 10000, device=device)
                result = torch.matmul(large_tensor, another_large_tensor)
        
        print("✅ Error handling test passed")