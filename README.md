# robo-rlhf-multimodal

**Multimodal Reinforcement Learning from Human Feedback for Robotics.**

Combines image observations (RGB camera) with proprioceptive state (joint angles, velocities) under a Bradley-Terry preference model. A small PPO loop fine-tunes a continuous-action policy using the learned reward.

---

## Architecture

```
image (3×64×64)  ──► CNN  ──► img_feat (128)  ──┐
                                                   ├──► fusion MLP ──► z (256)
proprio (14,)    ──► MLP  ──► prp_feat (128)  ──┘
                                                   └──► reward head ──► r ∈ ℝ
```

Human preference pairs `(obs_a, obs_b, preference)` train the reward model via the **Bradley-Terry** model:

```
P(A ≻ B) = σ(r(A) − r(B))
```

A lightweight **PPO** loop then fine-tunes a Gaussian policy using the frozen reward model as the reward signal.

---

## Core Components

| Class | File | Description |
|---|---|---|
| `RobotObservation` | `observation.py` | Multimodal obs container: image `(3,64,64)` + proprio `(14,)` |
| `MultimodalEncoder` | `encoder.py` | CNN + MLP → fused 256-d embedding |
| `RewardModel` | `reward_model.py` | Bradley-Terry scalar reward from preference pairs |
| `PreferenceDataset` | `preference_dataset.py` | Dataset of `(obs_a, obs_b, preference)` with synthetic generation |
| `RLHFTrainer` | `rlhf_trainer.py` | Stage 1: reward learning; Stage 2: PPO fine-tuning |

---

## Quick Start

```bash
# Clone
git clone https://github.com/danieleschmidt/robo-rlhf-multimodal
cd robo-rlhf-multimodal

# Install (conda recommended — needs PyTorch)
pip install -e .

# Run the demo
python demo.py
```

Example output:
```
▶ Training reward model on preferences …
  Reward epoch 1/8  loss=0.6925
  Reward epoch 8/8  loss=0.0010

▶ Evaluating preference prediction accuracy …
  Baseline (untrained) accuracy : 49.0%
  Trained reward model accuracy : 65.0%

▶ Fine-tuning policy via PPO with learned reward …
  PPO update 1/6  mean_reward=0.44
  PPO update 6/6  mean_reward=0.72
```

---

## Usage

```python
from robo_rlhf import (
    RobotObservation,
    MultimodalEncoder,
    RewardModel,
    PreferenceDataset,
    RLHFTrainer,
    TrainerConfig,
)

# Build an observation
obs = RobotObservation.random()          # image (3,64,64) + proprio (14,)

# Encode it
enc = MultimodalEncoder()
z = enc.encode_obs(obs)                  # (1, 256)

# Train from preferences
dataset = PreferenceDataset.synthetic(n_pairs=500)
trainer = RLHFTrainer(config=TrainerConfig(reward_epochs=20))
trainer.train_reward_model(dataset)

# Evaluate
acc = trainer.evaluate_reward_model(dataset)
print(f"Preference accuracy: {acc:.1%}")

# PPO fine-tuning
trainer.train_policy()
```

---

## Proprioception Layout

```
Index  0– 6   joint angles   q0…q6   (rad)
Index  7–13   joint velocities dq0…dq6 (rad/s)
```

Total: 14 dimensions (7-DOF arm, e.g. Franka Panda).  
Change `PROPRIO_DIM` in `observation.py` to adapt.

---

## Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/test_pipeline.py -v
# 26 passed
```

---

## Design Choices

- **No VLM dependency** — visual encoding uses a small custom CNN (207 K params total encoder), not CLIP or ResNet-50. Runs fast on CPU or any CUDA GPU.
- **Bradley-Terry** — the standard choice for pairwise preferences; avoids absolute reward labelling.
- **Self-contained PPO** — no RL library dependency. The implementation is ~80 lines and correct for continuous action spaces.
- **Synthetic env** — the PPO demo uses random observations so no MuJoCo license is required. Plug in any real `gym`-compatible env by replacing `_collect_rollout`.

---

## License

MIT
