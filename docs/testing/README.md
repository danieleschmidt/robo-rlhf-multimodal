# Testing Infrastructure

This document describes the comprehensive testing infrastructure for the Robo-RLHF-Multimodal project.

## Overview

Our testing strategy covers multiple levels and types of testing to ensure code quality, performance, and reliability:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Test system performance and scalability  
- **Contract Tests**: Test API contracts and interfaces
- **Mutation Tests**: Test the quality of our test suite
- **Security Tests**: Test for security vulnerabilities

## Test Structure

```
tests/
├── conftest.py              # Main test configuration and fixtures
├── unit/                    # Unit tests
│   ├── test_core_components.py
│   ├── test_models.py
│   └── test_quantum_sdlc.py
├── integration/             # Integration tests
│   ├── test_full_pipeline.py
│   └── test_training_pipeline.py
├── e2e/                     # End-to-end tests
│   └── test_full_pipeline.py
├── performance/             # Performance tests
│   ├── conftest.py          # Performance test fixtures
│   └── test_benchmarks.py
├── contract/                # Contract tests
│   ├── conftest.py          # Contract test fixtures
│   └── contracts/           # API contract definitions
├── mutation/                # Mutation tests
│   ├── conftest.py          # Mutation test fixtures
│   └── test_mutation.py
└── fixtures/                # Test data and fixtures
    └── data_fixtures.py
```

## Running Tests

### Basic Test Commands

```bash
# Run all tests
pytest

# Run specific test types
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests only
pytest tests/e2e/                     # End-to-end tests only

# Run tests with coverage
pytest --cov=robo_rlhf --cov-report=html

# Run tests in parallel
pytest -n auto

# Run tests with specific markers
pytest -m "not slow"                 # Skip slow tests
pytest -m "unit"                     # Run only unit tests
pytest -m "performance"              # Run only performance tests
```

### Advanced Test Commands

```bash
# Performance testing
pytest tests/performance/ -v --benchmark-only

# Contract testing
pytest tests/contract/ -v

# Mutation testing (requires additional setup)
pytest tests/mutation/ -v

# Security testing
bandit -r robo_rlhf/
safety check

# Generate comprehensive test report
pytest --cov=robo_rlhf --cov-report=html --html=report.html --self-contained-html
```

## Test Markers

We use pytest markers to categorize tests:

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.e2e`: End-to-end tests
- `@pytest.mark.performance`: Performance tests
- `@pytest.mark.slow`: Slow-running tests
- `@pytest.mark.gpu`: Tests requiring GPU
- `@pytest.mark.mujoco`: Tests requiring MuJoCo
- `@pytest.mark.isaac`: Tests requiring Isaac Sim
- `@pytest.mark.ros`: Tests requiring ROS2

## Test Configuration

### pytest.ini

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --cov=robo_rlhf
    --cov-report=html
    --cov-report=term-missing
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    performance: Performance tests
    slow: Slow running tests
    gpu: Tests requiring GPU
    mujoco: Tests requiring MuJoCo
    isaac: Tests requiring Isaac Sim
    ros: Tests requiring ROS2
```

### tox.ini

```ini
[tox]
envlist = py38,py39,py310,py311,lint,docs,security

[testenv]
deps = 
    pytest>=7.0.0
    pytest-cov>=4.0.0
    pytest-xdist>=2.5.0
    pytest-benchmark>=4.0.0
commands = pytest {posargs}

[testenv:lint]
deps =
    flake8>=5.0.0
    mypy>=0.991
    black>=22.0.0
    isort>=5.10.0
    bandit>=1.7.0
commands =
    flake8 robo_rlhf tests
    mypy robo_rlhf
    black --check robo_rlhf tests
    isort --check-only robo_rlhf tests
    bandit -r robo_rlhf

[testenv:security]
deps =
    bandit>=1.7.0
    safety>=2.0.0
commands =
    bandit -r robo_rlhf
    safety check
```

## Performance Testing

### Configuration

Performance tests use specialized fixtures and monitoring:

```python
@pytest.fixture
def performance_monitor():
    """Monitor system performance during tests."""
    monitor = PerformanceMonitor()
    yield monitor
    monitor.stop_monitoring()

@pytest.fixture
def benchmark_config():
    """Performance thresholds."""
    return {
        "cpu_threshold": 80.0,
        "memory_threshold": 80.0,
        "time_threshold": 5.0,
        "throughput_threshold": 100
    }
```

### Running Performance Tests

```bash
# Run performance tests with monitoring
pytest tests/performance/ --benchmark-only

# Run load tests
pytest tests/performance/ -m load

# Run stress tests
pytest tests/performance/ -m stress
```

## Contract Testing

### API Contract Definition

```json
{
  "endpoint": "/api/v1/preferences",
  "method": "POST",
  "request_schema": {
    "type": "object",
    "properties": {
      "pair_id": {"type": "string"},
      "preference": {"type": "integer"},
      "annotator_id": {"type": "string"}
    },
    "required": ["pair_id", "preference", "annotator_id"]
  },
  "response_schema": {
    "type": "object",
    "properties": {
      "id": {"type": "string"},
      "status": {"type": "string"}
    },
    "required": ["id", "status"]
  },
  "status_codes": [201, 400, 422]
}
```

### Contract Test Example

```python
@pytest.mark.contract
async def test_preference_api_contract(api_client, contract_validator):
    """Test preference API against contract."""
    response = await api_client.post("/api/v1/preferences", {
        "pair_id": "test_pair",
        "preference": 1,
        "annotator_id": "test_user"
    })
    
    assert contract_validator.validate_response("/api/v1/preferences", response)
```

## Mutation Testing

Mutation testing helps evaluate the quality of our test suite by introducing small code changes and checking if tests catch them.

### Running Mutation Tests

```bash
# Run mutation tests
pytest tests/mutation/ -v

# Generate mutation report
pytest tests/mutation/ --mutation-report=html
```

### Mutation Score Targets

- **Target**: 80% mutation score
- **Minimum**: 70% mutation score
- **Critical modules**: 90% mutation score

## Test Data and Fixtures

### Test Data Management

- **fixtures/data_fixtures.py**: Common test data
- **conftest.py**: Shared fixtures and configuration
- **Temporary data**: Use `tmp_path` fixture for temporary files

### Example Fixtures

```python
@pytest.fixture
def sample_preference_data():
    """Sample preference data for testing."""
    return {
        "pair_id": "test_pair_001",
        "video_a": "path/to/video_a.mp4",
        "video_b": "path/to/video_b.mp4",
        "annotations": [
            {"annotator": "user1", "preference": 1},
            {"annotator": "user2", "preference": 0}
        ]
    }
```

## Continuous Integration

### GitHub Actions

Tests run automatically on:
- Pull requests
- Pushes to main branch
- Nightly builds

### Test Matrix

- **Python versions**: 3.8, 3.9, 3.10, 3.11
- **Operating systems**: Ubuntu, macOS, Windows
- **Dependencies**: Minimal, full, with MuJoCo, with Isaac Sim

## Code Coverage

### Coverage Targets

- **Overall**: 90% code coverage
- **Critical modules**: 95% code coverage
- **New code**: 100% code coverage

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=robo_rlhf --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Best Practices

### Test Writing Guidelines

1. **Use descriptive test names** that explain what is being tested
2. **Follow AAA pattern**: Arrange, Act, Assert
3. **Test one thing at a time** - each test should verify one behavior
4. **Use appropriate fixtures** for setup and teardown
5. **Mock external dependencies** to ensure test isolation
6. **Include both positive and negative test cases**
7. **Test edge cases and error conditions**

### Performance Test Guidelines

1. **Set realistic performance targets** based on production requirements
2. **Use consistent test environments** for reliable results
3. **Monitor resource usage** during tests
4. **Test with realistic data sizes**
5. **Include load and stress testing**

### Example Test Structure

```python
class TestRLHFTrainer:
    """Test RLHF trainer functionality."""
    
    @pytest.fixture
    def trainer_config(self):
        """Training configuration fixture."""
        return {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10
        }
    
    @pytest.fixture
    def mock_model(self):
        """Mock model fixture."""
        model = Mock()
        model.train = Mock()
        model.eval = Mock()
        return model
    
    def test_trainer_initialization(self, trainer_config, mock_model):
        """Test trainer initializes correctly."""
        # Arrange
        trainer = RLHFTrainer(mock_model, trainer_config)
        
        # Act & Assert
        assert trainer.model == mock_model
        assert trainer.learning_rate == trainer_config["learning_rate"]
    
    @pytest.mark.asyncio
    async def test_training_step(self, trainer_config, mock_model):
        """Test single training step."""
        # Arrange
        trainer = RLHFTrainer(mock_model, trainer_config)
        batch_data = {"states": torch.randn(32, 10), "actions": torch.randn(32, 5)}
        
        # Act
        loss = await trainer.training_step(batch_data)
        
        # Assert
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        mock_model.train.assert_called_once()
```

## Troubleshooting

### Common Issues

1. **Import errors**: Check PYTHONPATH and virtual environment
2. **GPU tests failing**: Ensure CUDA is available and properly configured
3. **Slow tests**: Use `-n auto` for parallel execution
4. **Memory issues**: Use `--maxfail=1` to stop on first failure

### Debug Mode

```bash
# Run tests with detailed output
pytest -v -s

# Drop into debugger on failure
pytest --pdb

# Run only failed tests from last run
pytest --lf
```

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
- [tox documentation](https://tox.wiki/)