"""
Robotics RLHF Demo
==================

Demonstrates the full pipeline:
  1. Generate synthetic human preference data
  2. Train a multimodal reward model on those preferences
  3. Evaluate preference prediction accuracy
  4. Fine-tune a policy via PPO using the learned reward

Run with:
  ~/anaconda3/bin/python3 demo.py
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

import torch

from robo_rlhf import (
    RobotObservation,
    MultimodalEncoder,
    RewardModel,
    PreferenceDataset,
    RLHFTrainer,
    TrainerConfig,
)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  Robotics RLHF Demo   (device: {device})")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Synthetic preference data
    # ------------------------------------------------------------------
    print("▶ Generating 500 synthetic preference pairs …")
    train_ds = PreferenceDataset.synthetic(n_pairs=500, noise=0.1, device=device, seed=42)
    eval_ds  = PreferenceDataset.synthetic(n_pairs=100, noise=0.0, device=device, seed=99)
    print(f"  Train pairs: {len(train_ds)}  |  Eval pairs: {len(eval_ds)}")

    # Show an example pair
    img_a, prop_a, img_b, prop_b, pref = train_ds[0]
    print(f"\n  Example pair:")
    print(f"    obs_a image shape : {tuple(img_a.shape)}")
    print(f"    obs_a proprio shape: {tuple(prop_a.shape)}")
    print(f"    preference         : {'A' if pref.item() == 1.0 else 'B'} preferred")

    # ------------------------------------------------------------------
    # 2. Build encoder + trainer
    # ------------------------------------------------------------------
    print("\n▶ Building MultimodalEncoder …")
    encoder = MultimodalEncoder()
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Encoder parameters: {n_params:,}")

    cfg = TrainerConfig(
        reward_epochs=8,
        reward_batch_size=32,
        ppo_updates=6,
        ppo_rollout_steps=64,
        device=device,
    )
    trainer = RLHFTrainer(config=cfg, encoder=encoder)

    total_params = sum(
        p.numel() for p in trainer.reward_model.parameters()
    )
    print(f"  Reward model parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # 3. Train reward model
    # ------------------------------------------------------------------
    print("\n▶ Training reward model on preferences …")
    reward_history = trainer.train_reward_model(train_ds)
    print(f"\n  Loss trajectory (per epoch): {[f'{v:.4f}' for v in reward_history]}")

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    print("\n▶ Evaluating preference prediction accuracy …")
    # Before vs after comparison (train a fresh model on 0 data for baseline)
    baseline_trainer = RLHFTrainer(config=TrainerConfig(device=device))
    baseline_acc = baseline_trainer.evaluate_reward_model(eval_ds)
    trained_acc  = trainer.evaluate_reward_model(eval_ds)

    print(f"  Baseline (untrained) accuracy : {baseline_acc:.1%}")
    print(f"  Trained reward model accuracy : {trained_acc:.1%}")

    # ------------------------------------------------------------------
    # 5. PPO fine-tuning
    # ------------------------------------------------------------------
    print("\n▶ Fine-tuning policy via PPO with learned reward …")
    ppo_metrics = trainer.train_policy()

    first_reward = ppo_metrics[0]["mean_reward"]
    last_reward  = ppo_metrics[-1]["mean_reward"]
    print(f"\n  Mean reward — first update: {first_reward:.4f}  →  last update: {last_reward:.4f}")

    # ------------------------------------------------------------------
    # 6. Single-observation inference
    # ------------------------------------------------------------------
    print("\n▶ Single observation reward inference …")
    obs = RobotObservation.random(device=device)
    trainer.reward_model.eval()
    with torch.no_grad():
        r = trainer.reward_model.reward_from_obs(obs)
    print(f"  Random obs reward: {r.item():.4f}")

    print(f"\n{'='*60}")
    print("  Demo complete ✓")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
