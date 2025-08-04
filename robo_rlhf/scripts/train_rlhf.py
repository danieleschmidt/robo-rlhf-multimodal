"""
Script for training RLHF policies.
"""

import argparse
import json
import pickle
from pathlib import Path
import yaml
import torch

from robo_rlhf.algorithms.rlhf import MultimodalRLHF
from robo_rlhf.models.actors import VisionLanguageActor, MultimodalActor


def load_config(config_path: str) -> dict:
    """Load training configuration."""
    config_path = Path(config_path)
    
    if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    elif config_path.suffix == ".json":
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")


def create_model(config: dict) -> torch.nn.Module:
    """Create policy model from configuration."""
    model_config = config.get("model", {})
    model_type = model_config.get("type", "vision_language")
    
    if model_type == "vision_language":
        return VisionLanguageActor(
            vision_encoder=model_config.get("vision_encoder", "resnet18"),
            proprioception_dim=model_config.get("proprioception_dim", 7),
            action_dim=model_config.get("action_dim", 7),
            hidden_dim=model_config.get("hidden_dim", 512),
            num_heads=model_config.get("num_heads", 8),
            num_layers=model_config.get("num_layers", 4),
            dropout=model_config.get("dropout", 0.1),
            use_language=model_config.get("use_language", False)
        )
    elif model_type == "multimodal":
        return MultimodalActor(
            modalities=model_config.get("modalities", ["rgb", "proprioception"]),
            modality_dims=model_config.get("modality_dims", {"proprioception": 7}),
            action_dim=model_config.get("action_dim", 7),
            hidden_dim=model_config.get("hidden_dim", 256),
            fusion_type=model_config.get("fusion_type", "concat")
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main(args: argparse.Namespace) -> None:
    """Main training function."""
    print("Starting RLHF training")
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
        print(f"Loaded config from: {args.config}")
    
    # Override config with command line arguments
    training_config = config.get("training", {})
    epochs = args.epochs or training_config.get("epochs", 100)
    batch_size = args.batch_size or training_config.get("batch_size", 32)
    lr = args.lr or training_config.get("lr", 3e-4)
    
    # Load preferences
    preferences = None
    if args.preferences:
        with open(args.preferences, 'rb') as f:
            preferences = pickle.load(f)
        print(f"Loaded {len(preferences)} preference pairs from: {args.preferences}")
    
    # Create model
    model = create_model(config)
    print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    trainer = MultimodalRLHF(
        model=model,
        preferences=preferences,
        reward_model=training_config.get("reward_model", "bradley_terry"),
        optimizer=training_config.get("optimizer", "adamw"),
        lr=lr,
        use_wandb=args.wandb
    )
    
    # Train
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    stats = trainer.train(
        epochs=epochs,
        batch_size=batch_size,
        validation_split=training_config.get("validation_split", 0.1),
        checkpoint_dir=args.output,
        reward_epochs=training_config.get("reward_epochs", 50),
        policy_epochs=training_config.get("policy_epochs", epochs)
    )
    
    # Save final results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training statistics
    with open(output_dir / "training_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    # Save model
    torch.save(model.state_dict(), output_dir / "final_policy.pt")
    
    print(f"\n=== Training Complete ===")
    print(f"Final reward model accuracy: {stats['validation_accuracy'][-1]:.3f}")
    print(f"Average policy reward: {sum(stats['policy_reward']) / len(stats['policy_reward']):.3f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RLHF policy")
    
    parser.add_argument("--config", help="Training configuration file")
    parser.add_argument("--preferences", help="Preference pairs file")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size") 
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--output", default="checkpoints/", help="Output directory")
    parser.add_argument("--wandb", action="store_true", help="Use W&B logging")
    
    args = parser.parse_args()
    main(args)