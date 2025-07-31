# Development Guide

This guide covers development setup, architecture, and best practices for contributing to Robo-RLHF-Multimodal.

## Quick Start

```bash
git clone https://github.com/danielschmidt/robo-rlhf-multimodal
cd robo-rlhf-multimodal
pip install -e ".[dev]"
pre-commit install
pytest
```

## Architecture Overview

```
robo_rlhf/
├── collectors/        # Data collection interfaces
├── envs/             # Simulation environments  
├── models/           # Neural network architectures
├── algorithms/       # RLHF training algorithms
├── preference/       # Human feedback collection
└── deployment/       # Real robot deployment
```

## Development Environment

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- MuJoCo license (for MuJoCo environments)

### Optional Dependencies
- Isaac Sim (for NVIDIA environments)
- ROS2 (for real robot deployment)

### IDE Setup

**VS Code Extensions**:
- Python
- Pylance  
- Black Formatter
- GitLens

**Settings**:
```json
{
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true
}
```

## Testing Strategy

### Unit Tests
- Test individual components in isolation
- Mock external dependencies
- Fast execution (<1s per test)

### Integration Tests  
- Test component interactions
- Use lightweight environments
- Moderate execution time

### End-to-End Tests
- Full pipeline testing
- Real environment integration
- Slower execution (marked as `slow`)

### Running Tests
```bash
# All tests
pytest

# Unit tests only
pytest -m unit

# Skip slow tests  
pytest -m "not slow"

# With coverage
pytest --cov=robo_rlhf --cov-report=html
```

## Code Quality

### Formatting
```bash
black robo_rlhf tests
isort robo_rlhf tests
```

### Linting
```bash
flake8 robo_rlhf tests
mypy robo_rlhf
```

### Pre-commit Hooks
Automatically run on commit:
- Code formatting (Black, isort)
- Linting (flake8, mypy)
- Security checks
- Large file detection

## Debugging

### Environment Setup
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed MuJoCo logging
import mujoco
mujoco.set_logging_level("debug")
```

### Common Issues
- **CUDA out of memory**: Reduce batch size
- **MuJoCo license**: Set `MUJOCO_KEY_PATH`
- **Isaac Sim**: Ensure proper installation path

## Performance Optimization

### Profiling
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# Your code here
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(10)
```

### Memory Management
- Use `torch.cuda.empty_cache()` after training
- Monitor GPU memory with `nvidia-smi`
- Profile memory usage with `memory_profiler`

## Contributing Guidelines

1. **Branch Naming**: `feature/description`, `fix/description`
2. **Commit Messages**: Follow conventional commits
3. **Pull Requests**: Include tests and documentation
4. **Code Review**: Address all feedback promptly

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release branch
4. Run full test suite
5. Create GitHub release
6. Publish to PyPI