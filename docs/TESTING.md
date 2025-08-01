# Testing Guide

This document outlines the testing strategy and best practices for robo-rlhf-multimodal.

## Testing Philosophy

We follow a pyramid testing approach:
- **Unit Tests**: Fast, isolated tests for individual components
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Full pipeline validation
- **Performance Tests**: Benchmark critical paths

## Test Structure

```
tests/
├── unit/
│   ├── test_collectors.py
│   ├── test_models.py
│   ├── test_algorithms.py
│   └── test_preference.py
├── integration/
│   ├── test_data_pipeline.py
│   ├── test_training_pipeline.py
│   └── test_deployment.py
├── e2e/
│   ├── test_full_pipeline.py
│   └── test_real_robot.py
├── performance/
│   ├── test_benchmarks.py
│   └── test_memory_usage.py
└── fixtures/
    ├── sample_data/
    └── mock_envs/
```

## Running Tests

### All Tests
```bash
pytest
```

### By Category
```bash
# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Skip slow tests
pytest -m "not slow"

# Run with coverage
pytest --cov=robo_rlhf --cov-report=html
```

### Parallel Execution
```bash
# Run tests in parallel
pytest -n auto
```

## Test Configuration

### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow tests (>30s)
    gpu: Tests requiring GPU
    real_robot: Tests requiring real robot
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
```

## Test Fixtures

### Common Fixtures
```python
import pytest
from robo_rlhf.envs import MujocoManipulation

@pytest.fixture
def mock_env():
    """Lightweight mock environment for testing."""
    return MockEnvironment()

@pytest.fixture
def sample_demonstration():
    """Sample demonstration data."""
    return load_sample_data("demo_pick_place.pkl")

@pytest.fixture(scope="session")
def mujoco_env():
    """Real MuJoCo environment (slow, session-scoped)."""
    return MujocoManipulation(task="pick_and_place")
```

## Unit Testing Best Practices

### Test Organization
```python
class TestTeleOpCollector:
    """Test suite for TeleOpCollector."""
    
    def test_initialization(self):
        """Test collector initialization."""
        collector = TeleOpCollector(env=mock_env)
        assert collector.env is not None
    
    def test_collect_episode(self, mock_env):
        """Test episode collection."""
        collector = TeleOpCollector(env=mock_env)
        episode = collector.collect_episode()
        assert len(episode.observations) > 0
        assert len(episode.actions) > 0
```

### Mocking External Dependencies
```python
@patch('robo_rlhf.collectors.spacemouse.SpaceMouse')
def test_spacemouse_integration(self, mock_spacemouse):
    """Test SpaceMouse integration."""
    mock_device = Mock()
    mock_spacemouse.return_value = mock_device
    
    collector = TeleOpCollector(device="spacemouse")
    action = collector.get_action()
    
    mock_device.read.assert_called_once()
```

## Integration Testing

### Database Integration
```python
def test_preference_storage_integration(tmp_path):
    """Test preference data storage and retrieval."""
    db_path = tmp_path / "test.db"
    store = PreferenceStore(db_path)
    
    # Store preferences
    preferences = generate_test_preferences()
    store.save_preferences(preferences)
    
    # Retrieve and verify
    retrieved = store.load_preferences()
    assert len(retrieved) == len(preferences)
```

### Model Training Integration
```python
@pytest.mark.slow
def test_reward_model_training(sample_preferences):
    """Test reward model training pipeline."""
    model = RewardModel(input_dim=128)
    trainer = RewardTrainer(model)
    
    trainer.train(preferences=sample_preferences, epochs=5)
    
    # Verify model learned something
    loss_before = trainer.initial_loss
    loss_after = trainer.final_loss
    assert loss_after < loss_before
```

## End-to-End Testing

### Full Pipeline Test
```python
@pytest.mark.e2e
@pytest.mark.slow
def test_full_rlhf_pipeline(tmp_path):
    """Test complete RLHF pipeline."""
    # 1. Collect demonstrations
    collector = TeleOpCollector(env=mock_env)
    demos = collector.collect(num_episodes=10)
    
    # 2. Generate preference pairs
    generator = PreferencePairGenerator(demos)
    pairs = generator.generate_pairs(num_pairs=50)
    
    # 3. Collect preferences (simulate)
    preferences = simulate_human_preferences(pairs)
    
    # 4. Train reward model
    reward_model = RewardModel(input_dim=128)
    reward_trainer = RewardTrainer(reward_model)
    reward_trainer.train(preferences)
    
    # 5. Train policy
    policy = VisionLanguageActor()
    rlhf_trainer = MultimodalRLHF(policy, reward_model)
    rlhf_trainer.train(epochs=10)
    
    # 6. Evaluate policy
    evaluator = PolicyEvaluator(env=mock_env)
    results = evaluator.evaluate(policy, num_episodes=5)
    
    assert results['success_rate'] > 0.5
```

## Performance Testing

### Benchmark Tests
```python
def test_data_collection_benchmark(benchmark, mock_env):
    """Benchmark data collection performance."""
    collector = TeleOpCollector(env=mock_env)
    
    result = benchmark(collector.collect_episode)
    
    # Verify performance requirements
    assert result.duration < 1.0  # < 1 second per episode
```

### Memory Usage Tests
```python
@pytest.mark.gpu
def test_model_memory_usage():
    """Test model memory usage doesn't exceed limits."""
    import torch
    
    initial_memory = torch.cuda.memory_allocated()
    
    model = VisionLanguageActor()
    model.cuda()
    
    # Simulate training step
    batch = generate_sample_batch(batch_size=32)
    model.forward(batch)
    
    peak_memory = torch.cuda.max_memory_allocated()
    memory_used = (peak_memory - initial_memory) / 1024**3  # GB
    
    assert memory_used < 8.0  # Max 8GB GPU memory
```

## Test Data Management

### Sample Data Generation
```python
def generate_test_demonstrations(num_episodes=10):
    """Generate synthetic test demonstrations."""
    demonstrations = []
    for _ in range(num_episodes):
        demo = {
            'observations': np.random.randn(100, 128),
            'actions': np.random.randn(100, 7),
            'rewards': np.random.randn(100),
            'metadata': {'task': 'test', 'success': True}
        }
        demonstrations.append(demo)
    return demonstrations
```

### Data Fixtures
Store reusable test data in `tests/fixtures/`:
- `sample_demonstrations.pkl`
- `test_preferences.json`
- `mock_environments.yml`

## Continuous Integration

### GitHub Actions Configuration
- Run tests on multiple Python versions (3.8-3.11)
- Test with and without optional dependencies
- GPU tests on self-hosted runners
- Performance regression detection

### Coverage Requirements
- Minimum 80% line coverage
- 90% coverage for core modules
- Branch coverage for critical paths

## Test Documentation

### Docstring Format
```python
def test_preference_aggregation():
    """Test human preference aggregation.
    
    This test verifies that multiple human annotations
    are correctly aggregated using Bradley-Terry model.
    
    Tests:
        - Multiple annotators with agreement
        - Handling disagreement between annotators
        - Edge cases with missing annotations
    """
    pass
```

### Test Planning
Document test scenarios in advance:
- Happy path scenarios
- Edge cases and error conditions
- Performance requirements
- Integration points
- Real-world usage patterns