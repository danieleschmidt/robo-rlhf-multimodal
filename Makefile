.PHONY: help install install-dev test lint format clean build upload docs

help:
	@echo "Available commands:"
	@echo "  install     Install package"
	@echo "  install-dev Install development dependencies"
	@echo "  test        Run tests"
	@echo "  lint        Run linting"
	@echo "  format      Format code"
	@echo "  clean       Clean build artifacts"
	@echo "  build       Build package"
	@echo "  docs        Build documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest

test-cov:
	pytest --cov=robo_rlhf --cov-report=html --cov-report=term-missing

lint:
	flake8 robo_rlhf tests
	mypy robo_rlhf

format:
	black robo_rlhf tests examples
	isort robo_rlhf tests examples

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

docs:
	@echo "Documentation build not yet implemented"