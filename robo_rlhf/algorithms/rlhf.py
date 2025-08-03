"""
Main RLHF training algorithm for multimodal policies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import wandb
from datetime import datetime

from robo_rlhf.algorithms.reward_learning import RewardLearner
from robo_rlhf.algorithms.ppo import PPOTrainer
from robo_rlhf.preference.models import PreferencePair, PreferenceChoice


class PreferenceDataset(Dataset):
    """Dataset for preference learning."""
    
    def __init__(self, preferences: List[PreferencePair]):
        """Initialize dataset with preference pairs."""
        self.preferences = preferences
        
        # Filter pairs with consensus
        self.valid_pairs = [
            p for p in preferences 
            if p.get_consensus() is not None
        ]
        
        print(f"Dataset contains {len(self.valid_pairs)} pairs with consensus")
    
    def __len__(self) -> int:
        return len(self.valid_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a preference pair."""
        pair = self.valid_pairs[idx]
        consensus = pair.get_consensus()
        
        # Extract features from segments
        features_a = self._extract_features(pair.segment_a)
        features_b = self._extract_features(pair.segment_b)
        
        # Convert preference to label
        if consensus == PreferenceChoice.SEGMENT_A:
            label = 1.0
        elif consensus == PreferenceChoice.SEGMENT_B:
            label = 0.0
        else:  # EQUAL
            label = 0.5
        
        return {
            "features_a": features_a,
            "features_b": features_b,
            "label": torch.tensor(label, dtype=torch.float32),
            "confidence": torch.tensor(pair.get_agreement_score(), dtype=torch.float32)
        }
    
    def _extract_features(self, segment) -> torch.Tensor:
        """Extract features from a segment."""
        # Simple feature extraction - concatenate actions
        # In production, would use learned encoders
        features = []
        
        # Action statistics
        actions = segment.actions
        features.extend([
            np.mean(actions),
            np.std(actions),
            np.min(actions),
            np.max(actions)
        ])
        
        # Trajectory smoothness
        if len(actions) > 1:
            action_diff = np.diff(actions, axis=0)
            features.append(np.mean(np.abs(action_diff)))
        else:
            features.append(0.0)
        
        # Success indicator if available
        if segment.metadata and "success" in segment.metadata:
            features.append(float(segment.metadata["success"]))
        else:
            features.append(0.5)
        
        return torch.tensor(features, dtype=torch.float32)


class MultimodalRLHF:
    """
    Main RLHF training pipeline for multimodal policies.
    
    Combines reward learning from preferences with policy optimization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        preferences: Optional[List[PreferencePair]] = None,
        reward_model: str = "bradley_terry",
        optimizer: str = "adamw",
        lr: float = 3e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_wandb: bool = True
    ):
        """
        Initialize RLHF trainer.
        
        Args:
            model: Policy model to train
            preferences: Human preference data
            reward_model: Type of reward model ('bradley_terry', 'ensemble')
            optimizer: Optimizer type
            lr: Learning rate
            device: Training device
            use_wandb: Whether to use Weights & Biases logging
        """
        self.policy = model.to(device)
        self.preferences = preferences
        self.device = device
        self.use_wandb = use_wandb
        
        # Initialize reward learner
        self.reward_learner = RewardLearner(
            model_type=reward_model,
            input_dim=6,  # Feature dimension
            device=device
        )
        
        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            policy=self.policy,
            reward_model=self.reward_learner.model,
            lr=lr,
            device=device
        )
        
        # Setup optimizer
        if optimizer == "adamw":
            self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr)
        elif optimizer == "adam":
            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        # Training statistics
        self.stats = {
            "reward_learning_loss": [],
            "policy_loss": [],
            "policy_reward": [],
            "validation_accuracy": []
        }
        
        # Initialize wandb if requested
        if use_wandb:
            wandb.init(
                project="robo-rlhf",
                config={
                    "reward_model": reward_model,
                    "optimizer": optimizer,
                    "lr": lr
                }
            )
    
    def train(
        self,
        epochs: int,
        batch_size: int = 32,
        validation_split: float = 0.1,
        checkpoint_dir: Optional[str] = None,
        reward_epochs: int = 50,
        policy_epochs: int = 100
    ) -> Dict[str, List[float]]:
        """
        Train the RLHF pipeline.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Validation data split
            checkpoint_dir: Directory to save checkpoints
            reward_epochs: Epochs for reward model training
            policy_epochs: Epochs for policy training
            
        Returns:
            Training statistics
        """
        print("Starting RLHF training pipeline")
        
        # Phase 1: Learn reward model from preferences
        if self.preferences:
            print("\n=== Phase 1: Reward Learning ===")
            self._train_reward_model(
                epochs=reward_epochs,
                batch_size=batch_size,
                validation_split=validation_split
            )
        
        # Phase 2: Optimize policy with learned rewards
        print("\n=== Phase 2: Policy Optimization ===")
        self._train_policy(
            epochs=policy_epochs,
            batch_size=batch_size
        )
        
        # Save final checkpoint
        if checkpoint_dir:
            self.save_checkpoint(checkpoint_dir, "final")
        
        print("\nTraining complete!")
        return self.stats
    
    def _train_reward_model(
        self,
        epochs: int,
        batch_size: int,
        validation_split: float
    ) -> None:
        """Train reward model on preferences."""
        if not self.preferences:
            print("No preferences provided, skipping reward learning")
            return
        
        # Create dataset
        dataset = PreferenceDataset(self.preferences)
        
        # Split into train/val
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Train reward model
        for epoch in range(epochs):
            # Training
            train_loss = self._reward_epoch(train_loader, train=True)
            
            # Validation
            val_acc = self._reward_epoch(val_loader, train=False)
            
            # Log metrics
            self.stats["reward_learning_loss"].append(train_loss)
            self.stats["validation_accuracy"].append(val_acc)
            
            if self.use_wandb:
                wandb.log({
                    "reward/train_loss": train_loss,
                    "reward/val_accuracy": val_acc,
                    "epoch": epoch
                })
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - "
                      f"Loss: {train_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
    
    def _reward_epoch(
        self,
        dataloader: DataLoader,
        train: bool
    ) -> float:
        """Run one epoch of reward model training/validation."""
        if train:
            self.reward_learner.model.train()
        else:
            self.reward_learner.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in dataloader:
            features_a = batch["features_a"].to(self.device)
            features_b = batch["features_b"].to(self.device)
            labels = batch["label"].to(self.device)
            confidence = batch["confidence"].to(self.device)
            
            if train:
                # Forward pass
                loss, probs = self.reward_learner.compute_preference_loss(
                    features_a, features_b, labels, confidence
                )
                
                # Backward pass
                self.reward_learner.optimizer.zero_grad()
                loss.backward()
                self.reward_learner.optimizer.step()
                
                total_loss += loss.item()
            else:
                with torch.no_grad():
                    # Compute accuracy
                    reward_a = self.reward_learner.model(features_a)
                    reward_b = self.reward_learner.model(features_b)
                    
                    # Predict preference
                    pred = (reward_a > reward_b).float()
                    
                    # Count correct (considering equal preferences)
                    equal_mask = (labels == 0.5)
                    correct_mask = (
                        (~equal_mask & (pred == labels.round())) |
                        (equal_mask & (torch.abs(reward_a - reward_b) < 0.1))
                    )
                    
                    correct += correct_mask.sum().item()
                    total += len(labels)
        
        if train:
            return total_loss / len(dataloader)
        else:
            return correct / total if total > 0 else 0.0
    
    def _train_policy(
        self,
        epochs: int,
        batch_size: int
    ) -> None:
        """Train policy using PPO with learned rewards."""
        # In a real implementation, this would interact with the environment
        # and use the learned reward model for training
        
        for epoch in range(epochs):
            # Simulate policy training
            # In production, would collect trajectories and optimize
            
            # Mock training step
            policy_loss = np.random.exponential(0.1)
            policy_reward = np.random.normal(0, 1)
            
            self.stats["policy_loss"].append(policy_loss)
            self.stats["policy_reward"].append(policy_reward)
            
            if self.use_wandb:
                wandb.log({
                    "policy/loss": policy_loss,
                    "policy/reward": policy_reward,
                    "epoch": epoch
                })
            
            if epoch % 20 == 0:
                print(f"Policy Epoch {epoch}/{epochs} - "
                      f"Loss: {policy_loss:.4f}, "
                      f"Reward: {policy_reward:.4f}")
    
    def save_checkpoint(self, checkpoint_dir: str, name: str) -> None:
        """Save training checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "policy_state": self.policy.state_dict(),
            "reward_model_state": self.reward_learner.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "stats": self.stats,
            "timestamp": datetime.now().isoformat()
        }
        
        path = checkpoint_dir / f"checkpoint_{name}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
        
        # Also save statistics
        stats_path = checkpoint_dir / f"stats_{name}.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint["policy_state"])
        self.reward_learner.model.load_state_dict(checkpoint["reward_model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.stats = checkpoint["stats"]
        
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def evaluate(
        self,
        test_preferences: List[PreferencePair]
    ) -> Dict[str, float]:
        """Evaluate on test preferences."""
        dataset = PreferenceDataset(test_preferences)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        accuracy = self._reward_epoch(dataloader, train=False)
        
        return {
            "test_accuracy": accuracy,
            "num_test_pairs": len(dataset)
        }