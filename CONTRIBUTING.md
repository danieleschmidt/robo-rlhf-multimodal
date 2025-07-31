# Contributing to Robo-RLHF-Multimodal

Thank you for considering contributing to Robo-RLHF-Multimodal! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/danielschmidt/robo-rlhf-multimodal
   cd robo-rlhf-multimodal
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Run tests**:
   ```bash
   pytest
   ```

4. **Run code quality checks**:
   ```bash
   black robo_rlhf tests
   isort robo_rlhf tests  
   flake8 robo_rlhf tests
   mypy robo_rlhf
   ```

5. **Submit a pull request**

## Code Style

- Follow PEP 8 style guidelines
- Use Black for code formatting (line length: 88)
- Use isort for import sorting
- Include type hints where appropriate
- Write comprehensive docstrings

## Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Aim for >90% test coverage
- Use descriptive test names

## Commit Messages

Follow conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes  
- `docs:` for documentation changes
- `test:` for test additions/modifications
- `refactor:` for code refactoring

## Pull Request Process

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure CI passes
4. Request review from maintainers
5. Address feedback promptly

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## Questions?

Open an issue or reach out to maintainers via GitHub discussions.