# Multi-stage Dockerfile for Robo-RLHF-Multimodal
# Build stages: base, development, production

# ============================================================================
# Base stage - common dependencies
# ============================================================================
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    libopengl0 \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libgtk-3-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 robo && \
    useradd --uid 1000 --gid robo --create-home --shell /bin/bash robo

# Set working directory
WORKDIR /workspace

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ============================================================================
# Development stage - includes dev tools and full package
# ============================================================================
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Install additional development tools
RUN pip install --no-cache-dir \
    jupyter \
    ipython \
    tensorboard \
    wandb \
    matplotlib \
    seaborn \
    plotly

# Copy source code
COPY --chown=robo:robo . .

# Install package in development mode
RUN pip install --no-cache-dir -e .

# Setup pre-commit hooks
RUN pre-commit install || echo "Pre-commit hooks setup failed"

# Switch to non-root user
USER robo

# Expose ports for development services
EXPOSE 8000 8080 6006 8888

# Default command for development
CMD ["bash"]

# ============================================================================
# Production stage - minimal runtime environment
# ============================================================================
FROM base as production

# Install only production dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy only necessary source code
COPY --chown=robo:robo robo_rlhf/ ./robo_rlhf/
COPY --chown=robo:robo README.md ./
COPY --chown=robo:robo LICENSE ./

# Install package in production mode
RUN pip install --no-cache-dir .

# Create necessary directories
RUN mkdir -p /workspace/data /workspace/logs /workspace/checkpoints && \
    chown -R robo:robo /workspace

# Switch to non-root user
USER robo

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose application port
EXPOSE 8080

# Production command
CMD ["python", "-m", "robo_rlhf.preference.server"]

# ============================================================================
# GPU stage - extends production with CUDA support
# ============================================================================
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as gpu-base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    cmake \
    curl \
    git \
    libopengl0 \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libgtk-3-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symbolic links for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Create non-root user
RUN groupadd --gid 1000 robo && \
    useradd --uid 1000 --gid robo --create-home --shell /bin/bash robo

# Set working directory
WORKDIR /workspace

# Install Python dependencies with CUDA support
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

FROM gpu-base as gpu-production

# Install package dependencies
RUN pip install --no-cache-dir -e .

# Copy source code
COPY --chown=robo:robo robo_rlhf/ ./robo_rlhf/
COPY --chown=robo:robo README.md LICENSE ./

# Install package
RUN pip install --no-cache-dir .

# Create directories
RUN mkdir -p /workspace/data /workspace/logs /workspace/checkpoints && \
    chown -R robo:robo /workspace

# Switch to non-root user
USER robo

# GPU health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print(torch.cuda.is_available())" && \
        curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080

# GPU production command
CMD ["python", "-m", "robo_rlhf.algorithms.train_rlhf"]