# Robo-RLHF-Multimodal

End-to-end pipeline for multimodal reinforcement learning from human feedback (RLHF) in robotics. Collect teleoperation data with images and proprioception, gather human preferences, and fine-tune policies using state-of-the-art multimodal RLHF techniques inspired by OpenAI's February 2025 robotics update.

## Overview

Robo-RLHF-Multimodal bridges the gap between human preferences and robotic behavior by providing a complete framework for collecting multimodal demonstrations, soliciting human feedback, and training robust policies that align with human intent. The system supports both MuJoCo and Isaac Sim environments.

## Key Features

- **Multimodal Data Collection**: Synchronized capture of RGB/depth images, proprioceptive states, and action sequences
- **Human Preference Interface**: Web-based annotation tool for collecting comparative feedback
- **RLHF Pipeline**: State-of-the-art preference learning with multimodal transformers
- **Simulator Support**: Native integration with MuJoCo and NVIDIA Isaac Sim
- **Real Robot Interface**: ROS2 bridge for deploying learned policies
- **Distributed Training**: Multi-GPU support for large-scale preference learning

## Installation

```bash
# Core installation
pip install robo-rlhf-multimodal

# With MuJoCo support
pip install robo-rlhf-multimodal[mujoco]

# With Isaac Sim support
pip install robo-rlhf-multimodal[isaac]

# Full installation (all simulators + real robot support)
pip install robo-rlhf-multimodal[full]
```

## Quick Start

### 1. Collect Teleoperation Data

```python
from robo_rlhf import TeleOpCollector
from robo_rlhf.envs import MujocoManipulation

# Initialize environment and collector
env = MujocoManipulation(task="pick_and_place")
collector = TeleOpCollector(
    env=env,
    modalities=["rgb", "depth", "proprioception"],
    device="spacemouse"  # or "keyboard", "vr_controller"
)

# Collect demonstrations
demos = collector.collect(
    num_episodes=100,
    save_dir="data/demonstrations"
)
```

### 2. Generate Preference Pairs

```python
from robo_rlhf import PreferencePairGenerator

# Create preference pairs from demonstrations
generator = PreferencePairGenerator(
    demo_dir="data/demonstrations",
    pair_selection="diversity_sampling"
)

pairs = generator.generate_pairs(
    num_pairs=1000,
    segment_length=50  # frames per segment
)
```

### 3. Collect Human Preferences

```python
from robo_rlhf import PreferenceServer

# Launch web interface for preference collection
server = PreferenceServer(
    pairs=pairs,
    port=8080,
    annotators=["expert1", "expert2", "expert3"]
)

server.launch()  # Opens web UI at localhost:8080

# Collect preferences
preferences = server.collect_preferences(
    min_annotations_per_pair=3,
    agreement_threshold=0.7
)
```

### 4. Train Multimodal RLHF Policy

```python
from robo_rlhf import MultimodalRLHF
from robo_rlhf.models import VisionLanguageActor

# Initialize model
model = VisionLanguageActor(
    vision_encoder="clip_vit_b32",
    proprioception_dim=7,
    action_dim=7,
    hidden_dim=512
)

# Train with RLHF
trainer = MultimodalRLHF(
    model=model,
    preferences=preferences,
    reward_model="bradley_terry",
    optimizer="adamw",
    lr=3e-4
)

trainer.train(
    epochs=100,
    batch_size=32,
    validation_split=0.1
)
```

## Architecture

```
robo-rlhf-multimodal/
├── robo_rlhf/
│   ├── collectors/        # Teleoperation interfaces
│   ├── envs/             # Simulator environments
│   │   ├── mujoco/       # MuJoCo tasks
│   │   └── isaac/        # Isaac Sim tasks
│   ├── models/           # Neural architectures
│   │   ├── encoders/     # Vision/proprioception encoders
│   │   ├── actors/       # Policy networks
│   │   └── rewards/      # Learned reward models
│   ├── algorithms/       # RLHF algorithms
│   ├── preference/       # Human feedback collection
│   └── deployment/       # Real robot deployment
├── examples/             # Complete examples
├── configs/             # Task configurations
└── web_ui/              # Preference annotation interface
```

## Supported Tasks

### MuJoCo Tasks
- **Manipulation**: Pick-and-place, stacking, insertion
- **Locomotion**: Quadruped walking, humanoid control
- **Dexterous**: In-hand manipulation, tool use

### Isaac Sim Tasks
- **Factory**: Assembly, sorting, packaging
- **Kitchen**: Cooking tasks, dishwashing
- **Warehouse**: Palletizing, inventory management

## Advanced Usage

### Custom Reward Models

```python
from robo_rlhf.models import RewardModel

class CustomRewardModel(RewardModel):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.encoder = MultimodalEncoder(input_dim)
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs, action):
        features = self.encoder(obs)
        return self.reward_head(torch.cat([features, action], dim=-1))

# Use custom reward model
trainer = MultimodalRLHF(
    model=policy,
    reward_model=CustomRewardModel(input_dim=512)
)
```

### Distributed Training

```python
from robo_rlhf.distributed import DistributedRLHF

# Multi-GPU training
trainer = DistributedRLHF(
    model=model,
    preferences=preferences,
    num_gpus=8,
    gradient_accumulation_steps=4
)

# Launch distributed training
trainer.train_distributed(
    epochs=200,
    checkpoint_dir="checkpoints/",
    use_wandb=True
)
```

### Real Robot Deployment

```python
from robo_rlhf.deployment import ROS2PolicyNode

# Deploy to real robot
node = ROS2PolicyNode(
    policy_checkpoint="checkpoints/best_policy.pt",
    action_topic="/robot/command",
    observation_topics={
        "rgb": "/camera/rgb/image_raw",
        "depth": "/camera/depth/image_raw",
        "joint_states": "/robot/joint_states"
    }
)

# Run policy at 30Hz
node.run(control_frequency=30)
```

## Evaluation

### Policy Evaluation

```python
from robo_rlhf.evaluation import PolicyEvaluator

evaluator = PolicyEvaluator(
    env=env,
    metrics=["success_rate", "completion_time", "smoothness"]
)

results = evaluator.evaluate(
    policy=trained_policy,
    num_episodes=100,
    render=True
)
```

### Human Preference Alignment

```python
from robo_rlhf.evaluation import PreferenceAlignmentTest

# Test if learned behavior matches human preferences
alignment_test = PreferenceAlignmentTest(
    policy=trained_policy,
    test_preferences=held_out_preferences
)

alignment_score = alignment_test.compute_alignment()
print(f"Policy alignment with human preferences: {alignment_score:.3f}")
```

## Web UI for Preference Collection

The built-in web interface makes it easy to collect high-quality preference data:

```bash
# Launch preference collection server
python -m robo_rlhf.preference_server \
    --pairs data/preference_pairs.pkl \
    --output data/preferences.json \
    --port 8080
```

Features:
- Side-by-side video comparison
- Keyboard shortcuts for quick annotation
- Progress tracking and statistics
- Multi-annotator support with inter-rater reliability

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/danieleschmidt/robo-rlhf-multimodal
cd robo-rlhf-multimodal
pip install -e ".[dev]"
pytest tests/
```

## Citation

```bibtex
@software{robo_rlhf_multimodal,
  title={Robo-RLHF-Multimodal: Multimodal Reinforcement Learning from Human Feedback for Robotics},
  author={Daniel Schmidt},
  year={2025},
  url={https://github.com/danieleschmidt/robo-rlhf-multimodal}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- OpenAI Robotics for pioneering work in robotic RLHF
- The MuJoCo and Isaac Sim teams for excellent simulators
- All contributors to the open-source robotics community
