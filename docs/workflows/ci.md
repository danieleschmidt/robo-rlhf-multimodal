# Continuous Integration Workflow

This document describes the CI/CD pipeline requirements for the robo-rlhf-multimodal project.

## Overview

The CI pipeline should validate code quality, run tests, and ensure deployability across multiple Python versions and environments.

## Required GitHub Actions Workflows

### 1. Test Workflow (`.github/workflows/test.yml`)

```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", 3.11]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        pytest --cov=robo_rlhf --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### 2. Quality Workflow (`.github/workflows/quality.yml`)

```yaml
name: Code Quality

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  quality:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run linting
      run: |
        flake8 robo_rlhf tests
        mypy robo_rlhf
    
    - name: Check formatting
      run: |
        black --check robo_rlhf tests
        isort --check-only robo_rlhf tests
    
    - name: Security scan
      run: |
        bandit -r robo_rlhf
        safety check
```

### 3. Build Workflow (`.github/workflows/build.yml`)

```yaml
name: Build

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"
    
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Check package
      run: twine check dist/*
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: dist
        path: dist/
```

## Security Scanning

### Dependencies
- **Safety**: Scans for known security vulnerabilities in dependencies
- **Bandit**: Static security analysis for Python code
- **GitGuardian**: Secrets detection in commits

### Configuration

Add to `pyproject.toml`:
```toml
[tool.bandit]
exclude_dirs = ["tests"]
skips = ["B101"]  # Skip assert_used test

[tool.safety]
ignore = []  # Add CVE IDs to ignore if needed
```

## Performance Testing

### Benchmark Tests
- Use `pytest-benchmark` for performance regression testing
- Include in CI for critical paths
- Set performance thresholds

### Example Configuration
```python
def test_data_collection_performance(benchmark):
    result = benchmark(collect_episode, env, device)
    assert result.duration < 1.0  # Max 1 second per episode
```

## Docker Integration

### Multi-stage Build
```dockerfile
FROM python:3.10-slim as builder
WORKDIR /app
COPY pyproject.toml ./
RUN pip install build && python -m build

FROM python:3.10-slim as runtime
COPY --from=builder /app/dist/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl
```

## Deployment Pipeline

### Staging
- Automatic deployment to staging on merge to `develop`
- Integration tests against staging environment
- Performance benchmarks

### Production
- Manual approval required for production deployment
- Blue-green deployment strategy
- Automated rollback on failure

## Monitoring Integration

### Health Checks
- Application startup verification
- Database connectivity
- External service dependencies

### Observability
- Structured logging
- Metrics collection (Prometheus)
- Distributed tracing (OpenTelemetry)

## Branch Protection

Recommended branch protection rules for `main`:
- Require pull request reviews (2 reviewers)
- Require status checks to pass
- Require branches to be up to date
- Restrict pushes to main branch
- Require signed commits