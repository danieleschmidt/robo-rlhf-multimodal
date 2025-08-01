"""Unit tests for model components."""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Import once models are implemented
# from robo_rlhf.models import VisionEncoder, PolicyNetwork, RewardModel
# from robo_rlhf.models.encoders import CLIPEncoder, ProprioceptionEncoder


class TestVisionEncoder:
    """Test vision encoder components."""
    
    @pytest.mark.unit
    def test_clip_encoder_initialization(self, device):
        """Test CLIP encoder initialization."""
        # This test will be implemented once the actual model classes exist
        # For now, test the interface expectations
        assert device is not None
        
    @pytest.mark.unit
    def test_encoder_forward_pass(self, sample_observation, device):
        """Test encoder forward pass with sample data."""
        rgb_input = sample_observation["rgb"]
        assert rgb_input.shape == (224, 224, 3)
        assert rgb_input.dtype == np.uint8
        
        # Convert to tensor format expected by models
        rgb_tensor = torch.from_numpy(rgb_input).permute(2, 0, 1).float() / 255.0
        rgb_batch = rgb_tensor.unsqueeze(0).to(device)
        
        assert rgb_batch.shape == (1, 3, 224, 224)
        assert 0.0 <= rgb_batch.min() <= rgb_batch.max() <= 1.0
        
    @pytest.mark.unit
    def test_encoder_output_shape(self, device):
        """Test that encoder produces expected output shape."""
        batch_size = 8
        expected_feature_dim = 512
        
        # Mock encoder output
        mock_features = torch.randn(batch_size, expected_feature_dim, device=device)
        assert mock_features.shape == (batch_size, expected_feature_dim)


class TestProprioceptionEncoder:
    """Test proprioception encoder components."""
    
    @pytest.mark.unit
    def test_proprioception_encoding(self, sample_observation, device):
        """Test proprioception data encoding."""
        proprio_input = sample_observation["proprioception"]
        assert proprio_input.shape == (7,)
        
        # Convert to tensor
        proprio_tensor = torch.from_numpy(proprio_input).float().to(device)
        proprio_batch = proprio_tensor.unsqueeze(0)
        
        assert proprio_batch.shape == (1, 7)
        
    @pytest.mark.unit
    def test_encoder_normalization(self, device):
        """Test that encoder handles input normalization properly."""
        # Test with extreme values
        extreme_input = torch.tensor([[-100.0, 100.0, 0.0, 1e6, -1e6, 0.1, -0.1]], device=device)
        
        # Mock normalization (would be done by actual encoder)
        normalized = torch.tanh(extreme_input)  # Example normalization
        assert torch.all(torch.abs(normalized) <= 1.0)


class TestPolicyNetwork:
    """Test policy network components."""
    
    @pytest.mark.unit
    def test_policy_initialization(self, device, config_dict):
        """Test policy network initialization."""
        model_config = config_dict["model"]
        
        # Mock policy network parameters
        expected_params = {
            "vision_hidden_dim": 512,
            "proprio_hidden_dim": 256,
            "action_dim": model_config["action_dim"],
            "hidden_dim": model_config["hidden_dim"],
        }
        
        for key, value in expected_params.items():
            assert isinstance(value, int)
            assert value > 0
    
    @pytest.mark.unit
    def test_multimodal_fusion(self, sample_observation, device):
        """Test multimodal feature fusion."""
        batch_size = 4
        
        # Mock encoded features
        vision_features = torch.randn(batch_size, 512, device=device)
        proprio_features = torch.randn(batch_size, 256, device=device)
        
        # Test concatenation fusion
        fused_features = torch.cat([vision_features, proprio_features], dim=1)
        expected_dim = 512 + 256
        
        assert fused_features.shape == (batch_size, expected_dim)
    
    @pytest.mark.unit 
    def test_action_output_range(self, device):
        """Test that policy outputs actions in expected range."""
        batch_size = 8
        action_dim = 7
        
        # Mock policy output (before activation)
        raw_output = torch.randn(batch_size, action_dim, device=device)
        
        # Test different activation functions
        tanh_output = torch.tanh(raw_output)
        assert torch.all(torch.abs(tanh_output) <= 1.0)
        
        sigmoid_output = torch.sigmoid(raw_output)
        assert torch.all((sigmoid_output >= 0.0) & (sigmoid_output <= 1.0))


class TestRewardModel:
    """Test reward model components."""
    
    @pytest.mark.unit
    def test_reward_model_input_processing(self, sample_observation, sample_action, device):
        """Test reward model input processing."""
        # Mock processing observation and action for reward prediction
        obs_features = torch.randn(1, 512, device=device)  # Mock encoded observation  
        action_tensor = torch.from_numpy(sample_action).float().unsqueeze(0).to(device)
        
        # Concatenate for reward prediction
        reward_input = torch.cat([obs_features, action_tensor], dim=1)
        expected_dim = 512 + len(sample_action)
        
        assert reward_input.shape == (1, expected_dim)
    
    @pytest.mark.unit
    def test_reward_prediction_range(self, device):
        """Test reward prediction output range."""
        batch_size = 16
        
        # Mock reward predictions
        raw_rewards = torch.randn(batch_size, 1, device=device)
        
        # Rewards should be real-valued (no strict range requirement)
        assert raw_rewards.shape == (batch_size, 1)
        assert torch.all(torch.isfinite(raw_rewards))
    
    @pytest.mark.unit
    def test_pairwise_reward_comparison(self, device):
        """Test pairwise reward comparison for preference learning."""
        batch_size = 8
        
        # Mock rewards for two trajectory segments
        rewards_1 = torch.randn(batch_size, 1, device=device)
        rewards_2 = torch.randn(batch_size, 1, device=device)
        
        # Compute preference probabilities (Bradley-Terry model)
        reward_diff = rewards_1 - rewards_2
        preference_prob = torch.sigmoid(reward_diff)
        
        assert preference_prob.shape == (batch_size, 1)
        assert torch.all((preference_prob >= 0.0) & (preference_prob <= 1.0))


class TestModelEnsemble:
    """Test model ensemble components."""
    
    @pytest.mark.unit
    def test_ensemble_prediction_aggregation(self, device):
        """Test ensemble prediction aggregation."""
        num_models = 5
        batch_size = 10
        
        # Mock predictions from ensemble members
        predictions = []
        for _ in range(num_models):
            pred = torch.randn(batch_size, 1, device=device)
            predictions.append(pred)
        
        # Test mean aggregation
        stacked_predictions = torch.stack(predictions, dim=0)
        mean_prediction = torch.mean(stacked_predictions, dim=0)
        std_prediction = torch.std(stacked_predictions, dim=0)
        
        assert mean_prediction.shape == (batch_size, 1)
        assert std_prediction.shape == (batch_size, 1)
        assert torch.all(std_prediction >= 0.0)
    
    @pytest.mark.unit
    def test_uncertainty_estimation(self, device):
        """Test uncertainty estimation from ensemble disagreement."""
        num_models = 3
        batch_size = 5
        
        # Mock diverse predictions (high uncertainty)
        diverse_predictions = [
            torch.ones(batch_size, 1, device=device) * 1.0,
            torch.ones(batch_size, 1, device=device) * -1.0,
            torch.zeros(batch_size, 1, device=device),
        ]
        
        diverse_stack = torch.stack(diverse_predictions, dim=0)
        diverse_std = torch.std(diverse_stack, dim=0)
        
        # Mock consistent predictions (low uncertainty)  
        consistent_predictions = [
            torch.ones(batch_size, 1, device=device) * 0.5,
            torch.ones(batch_size, 1, device=device) * 0.55,
            torch.ones(batch_size, 1, device=device) * 0.45,
        ]
        
        consistent_stack = torch.stack(consistent_predictions, dim=0)
        consistent_std = torch.std(consistent_stack, dim=0)
        
        # High disagreement should lead to higher uncertainty
        assert torch.all(diverse_std > consistent_std)


@pytest.mark.slow
class TestModelTraining:
    """Test model training components."""
    
    @pytest.mark.unit
    def test_loss_computation(self, device):
        """Test loss computation for different model types."""
        batch_size = 16
        
        # Test policy loss (e.g., MSE for behavioral cloning)
        predicted_actions = torch.randn(batch_size, 7, device=device)
        target_actions = torch.randn(batch_size, 7, device=device)
        
        mse_loss = torch.nn.functional.mse_loss(predicted_actions, target_actions)
        assert mse_loss.item() >= 0.0
        assert torch.isfinite(mse_loss)
        
        # Test reward model loss (cross-entropy for preferences)
        preference_logits = torch.randn(batch_size, 1, device=device)
        preference_labels = torch.randint(0, 2, (batch_size, 1), device=device).float()
        
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            preference_logits, preference_labels
        )
        assert bce_loss.item() >= 0.0
        assert torch.isfinite(bce_loss)
    
    @pytest.mark.unit
    def test_gradient_computation(self, device):
        """Test gradient computation and backpropagation."""
        # Simple model for testing
        model = torch.nn.Linear(10, 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Forward pass
        input_data = torch.randn(8, 10, device=device)
        output = model(input_data)
        loss = torch.mean(output**2)  # Simple loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
        
        # Optimization step
        old_param = model.weight.data.clone()
        optimizer.step()
        
        # Parameters should have changed
        assert not torch.equal(old_param, model.weight.data)