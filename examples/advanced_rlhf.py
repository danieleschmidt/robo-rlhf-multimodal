#!/usr/bin/env python3
"""
Advanced RLHF example with custom reward models and distributed training.

This example shows more sophisticated usage including:
- Custom reward model architectures
- Ensemble preference learning
- Distributed training setup
- Advanced evaluation metrics
- Real-time policy deployment
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import wandb

from robo_rlhf import (
    MultimodalRLHF,
    VisionLanguageActor,
    make_env
)
from robo_rlhf.algorithms.reward_learning import RewardModel
from robo_rlhf.models.encoders import VisionEncoder, ProprioceptionEncoder


class EnsembleRewardModel(nn.Module):
    """
    Ensemble reward model for more robust preference learning.
    
    Uses multiple reward models and combines their predictions.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_models: int = 3,
        dropout: float = 0.1
    ):
        """Initialize ensemble reward model."""
        super().__init__()
        
        self.num_models = num_models
        
        # Create ensemble of reward models
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(num_models)
        ])
        
        # Learned combination weights
        self.combination_weights = nn.Parameter(torch.ones(num_models) / num_models)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble."""
        # Get predictions from all models
        predictions = torch.stack([model(x) for model in self.models], dim=-1)
        
        # Weighted combination
        weights = torch.softmax(self.combination_weights, dim=0)
        ensemble_pred = torch.sum(predictions * weights, dim=-1, keepdim=True)
        
        return ensemble_pred
    
    def get_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction uncertainty from ensemble disagreement."""
        with torch.no_grad():
            predictions = torch.stack([model(x) for model in self.models], dim=-1)
            uncertainty = torch.std(predictions, dim=-1, keepdim=True)
        return uncertainty


class HierarchicalRewardModel(nn.Module):
    """
    Hierarchical reward model that reasons about sub-goals.
    
    Predicts rewards at multiple temporal scales.
    """
    
    def __init__(
        self,
        vision_dim: int = 512,
        proprio_dim: int = 7,
        action_dim: int = 7,
        hidden_dim: int = 256,
        num_levels: int = 3
    ):
        """Initialize hierarchical reward model."""
        super().__init__()
        
        self.num_levels = num_levels
        
        # Feature encoders
        self.vision_encoder = VisionEncoder(output_dim=vision_dim)
        self.proprio_encoder = ProprioceptionEncoder(
            input_dim=proprio_dim,
            output_dim=hidden_dim
        )
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Multi-scale reward predictors
        self.reward_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(vision_dim + hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            for _ in range(num_levels)
        ])
        
        # Temporal attention for combining scales
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
    def forward(
        self,
        rgb: torch.Tensor,
        proprioception: torch.Tensor,
        actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through hierarchical model."""
        batch_size = rgb.shape[0]
        
        # Encode modalities
        vision_features = self.vision_encoder(rgb)
        proprio_features = self.proprio_encoder(proprioception)
        action_features = self.action_encoder(actions)
        
        # Combine features
        combined_features = torch.cat([
            vision_features,
            proprio_features,
            action_features
        ], dim=-1)
        
        # Predict rewards at different scales
        scale_rewards = []
        for i, reward_head in enumerate(self.reward_heads):
            scale_reward = reward_head(combined_features)
            scale_rewards.append(scale_reward)
        
        # Combine multi-scale predictions
        scale_features = torch.stack(scale_rewards, dim=1)  # [B, num_levels, 1]
        
        # Use attention to weight different scales
        attended_features, attention_weights = self.temporal_attention(
            scale_features, scale_features, scale_features
        )
        
        # Final reward prediction
        final_reward = attended_features.mean(dim=1)  # [B, 1]
        
        return {
            "reward": final_reward,
            "scale_rewards": scale_rewards,
            "attention_weights": attention_weights
        }


def advanced_training_loop(
    policy: nn.Module,
    preferences: List,
    config: Dict,
    use_ensemble: bool = True,
    use_hierarchical: bool = False
) -> Dict:
    """
    Advanced training loop with custom components.
    
    Args:
        policy: Policy model to train
        preferences: Preference data
        config: Training configuration
        use_ensemble: Whether to use ensemble reward model
        use_hierarchical: Whether to use hierarchical reward model
        
    Returns:
        Training statistics
    """
    print("🚀 Starting Advanced RLHF Training")
    
    # Initialize Weights & Biases
    if config.get("use_wandb", False):
        wandb.init(
            project=config.get("project_name", "robo-rlhf-advanced"),
            config=config,
            name=f"advanced_rlhf_{config.get('experiment_name', 'default')}"
        )
    
    # Create custom reward model
    if use_ensemble:
        print("Using Ensemble Reward Model")
        reward_model = EnsembleRewardModel(
            input_dim=config.get("feature_dim", 512),
            hidden_dim=config.get("reward_hidden_dim", 256),
            num_models=config.get("num_ensemble_models", 3)
        )
    elif use_hierarchical:
        print("Using Hierarchical Reward Model")
        reward_model = HierarchicalRewardModel(
            vision_dim=config.get("vision_dim", 512),
            proprio_dim=config.get("proprio_dim", 7),
            action_dim=config.get("action_dim", 7)
        )
    else:
        print("Using Standard Reward Model")
        reward_model = RewardModel(
            input_dim=config.get("feature_dim", 512),
            hidden_dim=config.get("reward_hidden_dim", 256)
        )
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    reward_model = reward_model.to(device)
    
    print(f"Training on device: {device}")
    
    # Create trainer with custom reward model
    trainer = MultimodalRLHF(
        model=policy,
        preferences=preferences,
        reward_model="custom",  # Use custom reward model
        optimizer=config.get("optimizer", "adamw"),
        lr=config.get("lr", 3e-4),
        device=device,
        use_wandb=config.get("use_wandb", False)
    )
    
    # Replace with custom reward model
    trainer.reward_learner.model = reward_model
    
    # Advanced training configuration
    training_config = {
        "epochs": config.get("epochs", 100),
        "batch_size": config.get("batch_size", 32),
        "validation_split": config.get("validation_split", 0.1),
        "checkpoint_dir": config.get("checkpoint_dir", "checkpoints/"),
        "reward_epochs": config.get("reward_epochs", 50),
        "policy_epochs": config.get("policy_epochs", 100)
    }
    
    # Train with advanced features
    stats = trainer.train(**training_config)
    
    # Advanced evaluation
    print("\n📊 Running Advanced Evaluation")
    
    # Uncertainty analysis
    if use_ensemble:
        print("Analyzing prediction uncertainty...")
        uncertainty_stats = analyze_uncertainty(reward_model, preferences)
        stats["uncertainty_analysis"] = uncertainty_stats
    
    # Multi-scale analysis
    if use_hierarchical:
        print("Analyzing hierarchical predictions...")
        hierarchical_stats = analyze_hierarchical_predictions(reward_model, preferences)
        stats["hierarchical_analysis"] = hierarchical_stats
    
    # Log final metrics
    if config.get("use_wandb", False):
        wandb.log({
            "final_reward_accuracy": stats["validation_accuracy"][-1],
            "final_policy_reward": stats["policy_reward"][-1],
            "training_epochs": len(stats["policy_loss"])
        })
        wandb.finish()
    
    return stats


def analyze_uncertainty(
    ensemble_model: EnsembleRewardModel,
    preferences: List
) -> Dict:
    """Analyze prediction uncertainty from ensemble model."""
    print("  Computing ensemble uncertainty...")
    
    uncertainties = []
    accuracies = []
    
    ensemble_model.eval()
    with torch.no_grad():
        for pair in preferences[:100]:  # Sample for analysis
            # Extract features (simplified)
            features_a = torch.randn(1, ensemble_model.models[0][0].in_features)
            features_b = torch.randn(1, ensemble_model.models[0][0].in_features)
            
            # Get predictions and uncertainty
            pred_a = ensemble_model(features_a)
            pred_b = ensemble_model(features_b)
            
            uncertainty_a = ensemble_model.get_uncertainty(features_a)
            uncertainty_b = ensemble_model.get_uncertainty(features_b)
            
            # Compute metrics
            uncertainties.extend([uncertainty_a.item(), uncertainty_b.item()])
            
            # Prediction accuracy (simplified)
            consensus = pair.get_consensus()
            if consensus is not None:
                predicted_better = pred_a > pred_b
                actual_better = consensus.value == "SEGMENT_A"
                accuracies.append(predicted_better.item() == actual_better)
    
    return {
        "mean_uncertainty": np.mean(uncertainties),
        "std_uncertainty": np.std(uncertainties),
        "accuracy": np.mean(accuracies) if accuracies else 0.0,
        "num_samples": len(uncertainties)
    }


def analyze_hierarchical_predictions(
    hierarchical_model: HierarchicalRewardModel,
    preferences: List
) -> Dict:
    """Analyze hierarchical predictions at different scales."""
    print("  Analyzing multi-scale predictions...")
    
    scale_correlations = []
    attention_patterns = []
    
    hierarchical_model.eval()
    with torch.no_grad():
        for pair in preferences[:50]:  # Sample for analysis
            # Create dummy inputs (would use real data in practice)
            rgb = torch.randn(1, 3, 64, 64)
            proprio = torch.randn(1, 7)
            actions = torch.randn(1, 7)
            
            # Get hierarchical predictions
            outputs = hierarchical_model(rgb, proprio, actions)
            
            # Analyze scale correlations
            scale_rewards = [r.item() for r in outputs["scale_rewards"]]
            scale_correlations.append(scale_rewards)
            
            # Analyze attention patterns
            attention_weights = outputs["attention_weights"].squeeze().cpu().numpy()
            attention_patterns.append(attention_weights)
    
    # Compute statistics
    scale_correlations = np.array(scale_correlations)
    attention_patterns = np.array(attention_patterns)
    
    return {
        "scale_correlations": np.corrcoef(scale_correlations.T).tolist(),
        "mean_attention": np.mean(attention_patterns, axis=0).tolist(),
        "attention_entropy": -np.sum(
            np.mean(attention_patterns, axis=0) * 
            np.log(np.mean(attention_patterns, axis=0) + 1e-8)
        ),
        "num_scales": hierarchical_model.num_levels
    }


def deploy_policy_realtime(
    policy: nn.Module,
    env_name: str,
    duration: float = 60.0
) -> Dict:
    """
    Deploy policy for real-time control.
    
    Args:
        policy: Trained policy model
        env_name: Environment name
        duration: Deployment duration in seconds
        
    Returns:
        Deployment statistics
    """
    print(f"🤖 Deploying policy for real-time control ({duration}s)")
    
    # Create environment
    env = make_env(env_name)
    
    # Initialize tracking
    stats = {
        "total_steps": 0,
        "total_reward": 0.0,
        "control_frequency": [],
        "action_magnitudes": [],
        "episodes": 0
    }
    
    import time
    start_time = time.time()
    
    policy.eval()
    obs, info = env.reset()
    episode_reward = 0.0
    
    while time.time() - start_time < duration:
        step_start = time.time()
        
        # Get action from policy
        if isinstance(obs, dict):
            rgb = obs.get("pixels", obs.get("rgb", np.zeros((3, 64, 64))))
            proprio = obs.get("robot_state", obs.get("proprioception", np.zeros(7)))
        else:
            rgb = np.zeros((3, 64, 64))
            proprio = obs[:7] if len(obs) >= 7 else np.zeros(7)
        
        # Ensure proper format
        rgb = rgb.astype(np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        
        action = policy.get_action(rgb, proprio)
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Update statistics
        stats["total_steps"] += 1
        stats["total_reward"] += reward
        episode_reward += reward
        stats["action_magnitudes"].append(np.linalg.norm(action))
        
        # Calculate control frequency
        step_time = time.time() - step_start
        stats["control_frequency"].append(1.0 / step_time if step_time > 0 else 0.0)
        
        # Handle episode end
        if terminated or truncated:
            print(f"Episode {stats['episodes'] + 1} completed: reward={episode_reward:.2f}")
            stats["episodes"] += 1
            obs, info = env.reset()
            episode_reward = 0.0
        
        # Render for visualization
        env.render()
    
    # Compute final statistics
    stats["average_frequency"] = np.mean(stats["control_frequency"])
    stats["average_action_magnitude"] = np.mean(stats["action_magnitudes"])
    stats["average_reward_per_step"] = stats["total_reward"] / max(stats["total_steps"], 1)
    
    print(f"✅ Deployment complete!")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Episodes: {stats['episodes']}")
    print(f"  Average frequency: {stats['average_frequency']:.1f} Hz")
    print(f"  Average reward: {stats['average_reward_per_step']:.3f}")
    
    return stats


def main():
    """Run advanced RLHF example."""
    print("🎯 Advanced RLHF Example")
    print("=" * 50)
    
    # Configuration
    config = {
        "experiment_name": "advanced_manipulation",
        "epochs": 50,
        "batch_size": 64,
        "lr": 1e-4,
        "optimizer": "adamw",
        "use_wandb": False,  # Set to True to log to W&B
        "project_name": "robo-rlhf-advanced",
        "feature_dim": 512,
        "vision_dim": 512,
        "proprio_dim": 7,
        "action_dim": 7,
        "num_ensemble_models": 5,
        "checkpoint_dir": "checkpoints/advanced/"
    }
    
    # Create policy
    policy = VisionLanguageActor(
        vision_encoder="resnet50",  # Larger model
        proprioception_dim=config["proprio_dim"],
        action_dim=config["action_dim"],
        hidden_dim=512,
        num_heads=8,
        num_layers=6,  # Deeper model
        dropout=0.1
    )
    
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Load or create preference data
    # (In practice, you would load real preference data)
    print("Creating mock preference data...")
    preferences = []  # Would load real preferences here
    
    # Run advanced training
    stats = advanced_training_loop(
        policy=policy,
        preferences=preferences,
        config=config,
        use_ensemble=True,  # Try ensemble reward model
        use_hierarchical=False
    )
    
    print("Training completed!")
    
    # Deploy for real-time control
    deployment_stats = deploy_policy_realtime(
        policy=policy,
        env_name="mujoco_manipulation",
        duration=30.0  # 30 seconds
    )
    
    print("\n🎉 Advanced example completed!")
    print(f"Final validation accuracy: {stats.get('validation_accuracy', [0])[-1]:.1%}")
    print(f"Deployment frequency: {deployment_stats['average_frequency']:.1f} Hz")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()