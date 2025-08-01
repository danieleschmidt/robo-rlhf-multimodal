"""Integration tests for the training pipeline."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from tests.fixtures.data_fixtures import (
    create_sample_demonstration_data,
    create_sample_preference_data,
    create_sample_config_file,
)


@pytest.mark.integration
class TestDataPipeline:
    """Test data loading and preprocessing pipeline."""
    
    def test_demonstration_data_loading(self, test_data_dir):
        """Test loading demonstration data."""
        # Create sample data
        demo_dir = create_sample_demonstration_data(test_data_dir, num_episodes=10)
        
        # Verify data structure
        assert demo_dir.exists()
        demo_files = list(demo_dir.glob("*.pkl"))
        assert len(demo_files) == 10
        
        # Test that each file has expected structure
        import pickle
        for demo_file in demo_files:
            with open(demo_file, "rb") as f:
                episode_data = pickle.load(f)
            
            required_keys = ["episode_id", "observations", "actions", "rewards", "metadata"]
            for key in required_keys:
                assert key in episode_data
            
            # Check data consistency
            num_steps = len(episode_data["observations"])
            assert len(episode_data["actions"]) == num_steps
            assert len(episode_data["rewards"]) == num_steps
    
    def test_preference_data_loading(self, test_data_dir):
        """Test loading preference data."""
        # Create sample preference data
        pref_dir = create_sample_preference_data(test_data_dir, num_pairs=50)
        
        # Verify data structure
        assert pref_dir.exists()
        pref_file = pref_dir / "preferences.json"
        assert pref_file.exists()
        
        # Load and validate preferences
        import json
        with open(pref_file, "r") as f:
            preferences = json.load(f)
        
        assert len(preferences) == 50
        
        for pref in preferences:
            required_keys = ["pair_id", "trajectory_1", "trajectory_2", "preference", "annotator_id"]
            for key in required_keys:
                assert key in pref
            
            # Validate preference values
            assert pref["preference"] in [-1, 0, 1]
            assert 0.0 <= pref["confidence"] <= 1.0
    
    def test_data_preprocessing(self, sample_trajectory, device):
        """Test data preprocessing for training."""
        # Test observation preprocessing
        observations = sample_trajectory["observations"]
        
        # Mock preprocessing steps
        processed_obs = []
        for obs in observations:
            # Normalize RGB
            rgb_normalized = obs["rgb"].astype(np.float32) / 255.0
            assert 0.0 <= rgb_normalized.min() <= rgb_normalized.max() <= 1.0
            
            # Check proprioception range
            proprio = obs["proprioception"]
            assert len(proprio) == 7
            
            processed_obs.append({
                "rgb": rgb_normalized,
                "proprioception": proprio,
            })
        
        assert len(processed_obs) == len(observations)


@pytest.mark.integration
class TestModelIntegration:
    """Test integration between different model components."""
    
    def test_multimodal_model_pipeline(self, sample_observation, device):
        """Test full multimodal model forward pass."""
        batch_size = 4
        
        # Mock vision encoder
        vision_encoder = MagicMock()
        vision_encoder.return_value = torch.randn(batch_size, 512, device=device)
        
        # Mock proprioception encoder  
        proprio_encoder = MagicMock()
        proprio_encoder.return_value = torch.randn(batch_size, 256, device=device)
        
        # Mock fusion layer
        fusion_layer = torch.nn.Linear(512 + 256, 512).to(device)
        
        # Mock policy head
        policy_head = torch.nn.Linear(512, 7).to(device)
        
        # Simulate forward pass
        batch_rgb = torch.randn(batch_size, 3, 224, 224, device=device)
        batch_proprio = torch.randn(batch_size, 7, device=device)
        
        vision_features = vision_encoder(batch_rgb)
        proprio_features = proprio_encoder(batch_proprio)
        
        # Feature fusion
        combined_features = torch.cat([vision_features, proprio_features], dim=1)
        fused_features = fusion_layer(combined_features)
        
        # Policy output
        actions = policy_head(fused_features)
        
        assert actions.shape == (batch_size, 7)
        assert torch.isfinite(actions).all()
    
    def test_reward_model_integration(self, sample_preference_pair, device):
        """Test reward model integration with preference data."""
        # Mock reward model
        reward_model = torch.nn.Sequential(
            torch.nn.Linear(512 + 7, 256),  # features + action
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        ).to(device)
        
        batch_size = 8
        
        # Mock trajectory features
        traj1_features = torch.randn(batch_size, 512, device=device)
        traj1_actions = torch.randn(batch_size, 7, device=device)
        
        traj2_features = torch.randn(batch_size, 512, device=device)
        traj2_actions = torch.randn(batch_size, 7, device=device)
        
        # Compute rewards
        traj1_input = torch.cat([traj1_features, traj1_actions], dim=1)
        traj2_input = torch.cat([traj2_features, traj2_actions], dim=1)
        
        rewards1 = reward_model(traj1_input)
        rewards2 = reward_model(traj2_input)
        
        # Compute preference probabilities
        reward_diff = rewards1 - rewards2
        pref_probs = torch.sigmoid(reward_diff)
        
        assert pref_probs.shape == (batch_size, 1)
        assert torch.all((pref_probs >= 0.0) & (pref_probs <= 1.0))


@pytest.mark.integration 
@pytest.mark.slow
class TestTrainingLoop:
    """Test training loop integration."""
    
    def test_basic_training_step(self, device, config_dict):
        """Test basic training step execution."""
        # Mock model components
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 7),
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config_dict["training"]["learning_rate"])
        criterion = torch.nn.MSELoss()
        
        # Mock training data
        batch_inputs = torch.randn(config_dict["training"]["batch_size"], 10, device=device)
        batch_targets = torch.randn(config_dict["training"]["batch_size"], 7, device=device)
        
        # Training step
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        
        # Check gradients
        total_grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        assert total_grad_norm > 0.0
        assert torch.isfinite(loss)
        
        # Optimization step
        optimizer.step()
    
    def test_preference_training_step(self, sample_preference_pair, device):
        """Test preference-based training step."""
        # Mock reward model
        reward_model = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(), 
            torch.nn.Linear(256, 1),
        ).to(device)
        
        optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-3)
        
        batch_size = 8
        
        # Mock trajectory features (would come from encoded observations)
        traj1_features = torch.randn(batch_size, 512, device=device)
        traj2_features = torch.randn(batch_size, 512, device=device)
        
        # Mock preference labels (0: prefer traj1, 1: prefer traj2)
        preference_labels = torch.randint(0, 2, (batch_size, 1), device=device).float()
        
        # Forward pass
        rewards1 = reward_model(traj1_features)
        rewards2 = reward_model(traj2_features)
        
        # Preference probability (probability of preferring traj2 over traj1)
        reward_diff = rewards2 - rewards1
        pref_logits = reward_diff  # Raw logits
        
        # Loss computation (cross-entropy)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pref_logits, preference_labels
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0
    
    def test_validation_step(self, device):
        """Test validation step execution."""
        model = torch.nn.Linear(10, 1).to(device)
        model.eval()  # Set to evaluation mode
        
        val_inputs = torch.randn(20, 10, device=device)
        val_targets = torch.randn(20, 1, device=device)
        
        with torch.no_grad():
            val_outputs = model(val_inputs)
            val_loss = torch.nn.functional.mse_loss(val_outputs, val_targets)
        
        assert torch.isfinite(val_loss)
        assert val_loss.item() >= 0.0
        
        # Ensure no gradients computed during validation
        for param in model.parameters():
            assert param.grad is None


@pytest.mark.integration
class TestConfigurationManagement:
    """Test configuration management and model instantiation."""
    
    def test_config_loading(self, test_data_dir):
        """Test loading configuration from file."""
        config_file = create_sample_config_file(test_data_dir)
        
        import json
        with open(config_file, "r") as f:
            config = json.load(f)
        
        # Validate config structure
        required_sections = ["experiment", "model", "training", "data", "evaluation"]
        for section in required_sections:
            assert section in config
        
        # Validate model config
        model_config = config["model"]
        required_model_keys = ["type", "action_dim", "hidden_dim"]
        for key in required_model_keys:
            assert key in model_config
    
    def test_model_instantiation_from_config(self, config_dict, device):
        """Test model instantiation from configuration."""
        model_config = config_dict["model"]
        
        # Mock model factory
        def create_model(config):
            return torch.nn.Sequential(
                torch.nn.Linear(config["hidden_dim"], config["hidden_dim"]),
                torch.nn.ReLU(),
                torch.nn.Linear(config["hidden_dim"], config["action_dim"]),
            )
        
        model = create_model(model_config).to(device)
        
        # Test model with sample input
        sample_input = torch.randn(1, model_config["hidden_dim"], device=device)
        output = model(sample_input)
        
        assert output.shape == (1, model_config["action_dim"])
        assert torch.isfinite(output).all()


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndPipeline:
    """Test end-to-end training pipeline."""
    
    def test_mini_training_run(self, test_data_dir, device, config_dict):
        """Test a minimal end-to-end training run."""
        # Create sample data
        demo_dir = create_sample_demonstration_data(test_data_dir, num_episodes=5)
        pref_dir = create_sample_preference_data(test_data_dir, num_pairs=20)
        
        # Mock minimal model
        model = torch.nn.Sequential(
            torch.nn.Linear(512 + 7, 256),  # features + action dim
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),  # reward output
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Mock training loop
        num_epochs = 3
        batch_size = 4
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Mock batch generation
            for batch_idx in range(5):  # 5 batches per epoch
                # Mock batch data
                batch_features = torch.randn(batch_size, 512, device=device)
                batch_actions = torch.randn(batch_size, 7, device=device)
                batch_labels = torch.randint(0, 2, (batch_size, 1), device=device).float()
                
                # Forward pass
                batch_input = torch.cat([batch_features, batch_actions], dim=1)
                outputs = model(batch_input)
                
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    outputs, batch_labels
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            assert avg_loss >= 0.0
            assert np.isfinite(avg_loss)
        
        # Verify model has been updated
        assert any(param.grad is not None for param in model.parameters() if param.requires_grad)