# ============================================================================
# Comprehensive Makefile for Robo-RLHF-Multimodal
# ============================================================================

.PHONY: help install install-dev test test-all lint format clean build docs
.PHONY: docker-build docker-run docker-dev docker-gpu docker-clean
.PHONY: security-check release pre-commit ci-setup benchmark
.PHONY: up up-dev up-gpu up-monitoring down logs
.DEFAULT_GOAL := help

# ============================================================================
# Configuration
# ============================================================================
PROJECT_NAME := robo-rlhf-multimodal
PYTHON := python3
PIP := pip3
DOCKER_REGISTRY ?= 
IMAGE_TAG ?= latest
COMPOSE_PROFILES ?= 

# Colors for output
BOLD := \033[1m
RESET := \033[0m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
BLUE := \033[34m

# ============================================================================
# Help
# ============================================================================
help: ## Show this help message
	@echo "$(BOLD)Robo-RLHF-Multimodal Development Makefile$(RESET)"
	@echo ""
	@echo "$(BOLD)Installation:$(RESET)"
	@grep -E '^[a-zA-Z_-]+.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(install|install-dev)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Development:$(RESET)"
	@grep -E '^[a-zA-Z_-]+.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(test|lint|format|clean|pre-commit)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Docker:$(RESET)"
	@grep -E '^[a-zA-Z_-]+.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(docker-|up|down)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Build & Release:$(RESET)"
	@grep -E '^[a-zA-Z_-]+.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(build|release|security-check)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(RED)%-20s$(RESET) %s\n", $$1, $$2}'

# ============================================================================
# Installation
# ============================================================================
install: ## Install package for production use
	@echo "$(GREEN)Installing package...$(RESET)"
	$(PIP) install -e .

install-dev: ## Install package with development dependencies
	@echo "$(GREEN)Installing development dependencies...$(RESET)"
	$(PIP) install -e ".[dev]"
	pre-commit install
	@echo "$(GREEN)Development environment ready!$(RESET)"

# ============================================================================
# Testing
# ============================================================================
test: ## Run basic tests
	@echo "$(BLUE)Running tests...$(RESET)"
	pytest tests/ -v

test-fast: ## Run tests with fail-fast and last-failed-first
	@echo "$(BLUE)Running fast tests...$(RESET)"
	pytest tests/ -x --ff -q

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	pytest tests/ --cov=robo_rlhf --cov-report=html --cov-report=term-missing --cov-report=xml

test-unit: ## Run only unit tests
	@echo "$(BLUE)Running unit tests...$(RESET)"
	pytest tests/unit/ -v -m "unit"

test-integration: ## Run only integration tests
	@echo "$(BLUE)Running integration tests...$(RESET)"
	pytest tests/integration/ -v -m "integration"

test-e2e: ## Run only end-to-end tests
	@echo "$(BLUE)Running end-to-end tests...$(RESET)"
	pytest tests/e2e/ -v -m "e2e"

test-gpu: ## Run GPU tests (requires CUDA)
	@echo "$(BLUE)Running GPU tests...$(RESET)"
	pytest tests/ -v -m "gpu"

test-all: test-cov test-gpu ## Run all tests including coverage and GPU tests
	@echo "$(GREEN)All tests completed!$(RESET)"

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(RESET)"
	pytest tests/performance/ -v --benchmark-only

# ============================================================================
# Code Quality
# ============================================================================
lint: ## Run linting (flake8, mypy, bandit)
	@echo "$(BLUE)Running linting...$(RESET)"
	flake8 robo_rlhf/ tests/ --count --statistics
	mypy robo_rlhf/
	bandit -r robo_rlhf/ -x robo_rlhf/tests/

format: ## Format code (black, isort)
	@echo "$(BLUE)Formatting code...$(RESET)"
	black robo_rlhf/ tests/ examples/ || true
	isort robo_rlhf/ tests/ examples/ || true

format-check: ## Check code formatting without making changes
	@echo "$(BLUE)Checking code formatting...$(RESET)"
	black --check robo_rlhf/ tests/ examples/ || true
	isort --check-only robo_rlhf/ tests/ examples/ || true

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(RESET)"
	pre-commit run --all-files

security-check: ## Run security checks
	@echo "$(RED)Running security checks...$(RESET)"
	bandit -r robo_rlhf/ -f json -o security-report.json || true
	safety check --json --output safety-report.json || true
	@echo "$(GREEN)Security check completed. Reports saved.$(RESET)"

# ============================================================================
# Cleanup
# ============================================================================
clean: ## Clean build artifacts and cache
	@echo "$(YELLOW)Cleaning build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf .tox/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	find . -type f -name "*.swp" -delete
	@echo "$(GREEN)Cleanup completed!$(RESET)"

clean-docker: ## Clean Docker images and containers
	@echo "$(YELLOW)Cleaning Docker resources...$(RESET)"
	docker system prune -f
	docker volume prune -f
	@echo "$(GREEN)Docker cleanup completed!$(RESET)"

# ============================================================================
# Build & Package
# ============================================================================
build: clean ## Build Python package
	@echo "$(RED)Building package...$(RESET)"
	$(PYTHON) -m build

build-check: ## Check build without actually building
	@echo "$(BLUE)Checking build configuration...$(RESET)"
	$(PYTHON) -m build --wheel --no-isolation --skip-dependency-check

docs: ## Build documentation (placeholder)
	@echo "$(YELLOW)Documentation build not yet implemented$(RESET)"

# ============================================================================
# Docker Commands
# ============================================================================
docker-build: ## Build Docker images for all targets
	@echo "$(YELLOW)Building Docker images...$(RESET)"
	docker build --target=production -t $(PROJECT_NAME):$(IMAGE_TAG) .
	docker build --target=development -t $(PROJECT_NAME)-dev:$(IMAGE_TAG) .
	docker build --target=gpu-production -t $(PROJECT_NAME)-gpu:$(IMAGE_TAG) .
	@echo "$(GREEN)Docker images built successfully!$(RESET)"

docker-build-prod: ## Build production Docker image only
	@echo "$(YELLOW)Building production Docker image...$(RESET)"
	docker build --target=production -t $(PROJECT_NAME):$(IMAGE_TAG) .

docker-build-dev: ## Build development Docker image only
	@echo "$(YELLOW)Building development Docker image...$(RESET)"
	docker build --target=development -t $(PROJECT_NAME)-dev:$(IMAGE_TAG) .

docker-build-gpu: ## Build GPU Docker image only
	@echo "$(YELLOW)Building GPU Docker image...$(RESET)"
	docker build --target=gpu-production -t $(PROJECT_NAME)-gpu:$(IMAGE_TAG) .

docker-run: ## Run production container
	@echo "$(YELLOW)Running production container...$(RESET)"
	docker run -it --rm -p 8080:8080 $(PROJECT_NAME):$(IMAGE_TAG)

docker-dev: ## Run development container
	@echo "$(YELLOW)Running development container...$(RESET)"
	docker run -it --rm -v $(PWD):/workspace $(PROJECT_NAME)-dev:$(IMAGE_TAG)

docker-gpu: ## Run GPU container (requires nvidia-docker)
	@echo "$(YELLOW)Running GPU container...$(RESET)"
	docker run -it --rm --gpus all -p 8080:8080 $(PROJECT_NAME)-gpu:$(IMAGE_TAG)

# ============================================================================
# Docker Compose Commands
# ============================================================================
up: ## Start production services
	@echo "$(YELLOW)Starting production services...$(RESET)"
	docker compose up -d

up-dev: ## Start development services
	@echo "$(YELLOW)Starting development services...$(RESET)"
	docker compose --profile dev up -d

up-gpu: ## Start GPU training services
	@echo "$(YELLOW)Starting GPU services...$(RESET)"
	docker compose --profile gpu up -d

up-monitoring: ## Start with monitoring stack
	@echo "$(YELLOW)Starting services with monitoring...$(RESET)"
	docker compose --profile monitoring up -d

up-all: ## Start all services (dev + gpu + monitoring)
	@echo "$(YELLOW)Starting all services...$(RESET)"
	docker compose --profile dev --profile gpu --profile monitoring up -d

down: ## Stop all services
	@echo "$(YELLOW)Stopping all services...$(RESET)"
	docker compose down

down-volumes: ## Stop services and remove volumes
	@echo "$(RED)Stopping services and removing volumes...$(RESET)"
	docker compose down -v

logs: ## Show logs from all services
	docker compose logs -f

logs-app: ## Show logs from main application
	docker compose logs -f robo-rlhf

# ============================================================================
# Release & Deployment
# ============================================================================
release-check: test-all lint security-check ## Run full release checklist
	@echo "$(GREEN)Release checks completed!$(RESET)"

release-build: release-check build ## Build release package after checks
	@echo "$(GREEN)Release package built!$(RESET)"

ci-setup: ## Setup for CI environment
	@echo "$(BLUE)Setting up CI environment...$(RESET)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"

# ============================================================================
# Development Utilities
# ============================================================================
jupyter: ## Start Jupyter Lab server
	@echo "$(BLUE)Starting Jupyter Lab...$(RESET)"
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

tensorboard: ## Start TensorBoard server
	@echo "$(BLUE)Starting TensorBoard...$(RESET)"
	tensorboard --logdir=logs/ --port=6006 --host=0.0.0.0

serve-docs: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(RESET)"
	@echo "$(YELLOW)Documentation serving not yet implemented$(RESET)"

# ============================================================================
# Database & Services
# ============================================================================
db-reset: ## Reset database (removes all data)
	@echo "$(RED)Resetting database...$(RESET)"
	docker compose stop mongodb
	docker compose rm -f mongodb
	docker volume rm robo-rlhf-multimodal_mongodb_data || true
	docker compose up -d mongodb
	@echo "$(GREEN)Database reset completed!$(RESET)"

db-backup: ## Backup database
	@echo "$(BLUE)Creating database backup...$(RESET)"
	mkdir -p backups
	docker compose exec mongodb mongodump --out /data/backup
	docker compose cp mongodb:/data/backup ./backups/$(shell date +%Y%m%d_%H%M%S)
	@echo "$(GREEN)Database backup completed!$(RESET)"

# ============================================================================
# Monitoring
# ============================================================================
stats: ## Show project statistics
	@echo "$(BOLD)Project Statistics:$(RESET)"
	@echo "Python files: $$(find robo_rlhf/ -name '*.py' | wc -l)"
	@echo "Test files: $$(find tests/ -name '*.py' | wc -l)"
	@echo "Lines of code: $$(find robo_rlhf/ -name '*.py' -exec wc -l {} + | tail -1 | awk '{print $$1}')"
	@echo "Lines of tests: $$(find tests/ -name '*.py' -exec wc -l {} + | tail -1 | awk '{print $$1}')"

health-check: ## Check health of running services
	@echo "$(BLUE)Checking service health...$(RESET)"
	@docker compose ps
	@echo "\n$(BLUE)Testing endpoints...$(RESET)"
	@curl -f http://localhost:8080/health || echo "Main app not accessible"
	@curl -f http://localhost:9090/-/healthy || echo "Prometheus not accessible"
	@curl -f http://localhost:3000/api/health || echo "Grafana not accessible"