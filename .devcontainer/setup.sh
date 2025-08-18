#!/bin/bash
set -e

echo "🚀 Setting up Robo-RLHF-Multimodal development environment..."

# Update system packages
echo "📦 Updating system packages..."
apt-get update && apt-get upgrade -y

# Install development dependencies
echo "🔧 Installing development tools..."
apt-get install -y \
    git \
    curl \
    wget \
    vim \
    htop \
    build-essential \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install Python development dependencies
echo "🐍 Installing Python development tools..."
pip install --upgrade pip setuptools wheel

# Install project in development mode
echo "📚 Installing project dependencies..."
if [ -f "/workspace/pyproject.toml" ]; then
    cd /workspace
    pip install -e ".[dev,test,docs]"
else
    echo "⚠️  pyproject.toml not found, skipping pip install"
fi

# Install pre-commit hooks
echo "🔒 Setting up pre-commit hooks..."
if [ -f "/workspace/.pre-commit-config.yaml" ]; then
    cd /workspace
    pre-commit install
    pre-commit install --hook-type commit-msg
else
    echo "⚠️  .pre-commit-config.yaml not found, skipping pre-commit setup"
fi

# Setup Git configuration for container
echo "🔧 Setting up Git configuration..."
git config --global --add safe.directory /workspace
git config --global init.defaultBranch main
git config --global pull.rebase false

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p /workspace/{data,checkpoints,logs,backups,reports}

# Install additional development tools
echo "🛠️  Installing additional development tools..."
pip install \
    jupyterlab \
    tensorboard \
    wandb \
    pytest-xdist \
    pytest-cov \
    pytest-mock \
    ipdb \
    memory-profiler \
    line-profiler

# Setup Jupyter extensions
echo "🪐 Setting up Jupyter Lab..."
jupyter lab --generate-config
echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.port = 8888" >> ~/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.allow_root = True" >> ~/.jupyter/jupyter_lab_config.py

# Install MuJoCo if not present
echo "🤖 Checking for MuJoCo installation..."
if [ ! -d "/root/.mujoco" ]; then
    echo "Installing MuJoCo..."
    mkdir -p /root/.mujoco
    cd /root/.mujoco
    wget https://github.com/deepmind/mujoco/releases/download/2.3.7/mujoco-2.3.7-linux-x86_64.tar.gz
    tar -xf mujoco-2.3.7-linux-x86_64.tar.gz
    rm mujoco-2.3.7-linux-x86_64.tar.gz
    echo 'export MUJOCO_PATH="/root/.mujoco/mujoco-2.3.7"' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$MUJOCO_PATH/lib"' >> ~/.bashrc
fi

# Setup environment variables
echo "🌍 Setting up environment variables..."
echo 'export PYTHONPATH="/workspace:$PYTHONPATH"' >> ~/.bashrc
echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc

# Install ROS2 (if needed for robot integration)
echo "🤖 Checking for ROS2..."
if ! command -v ros2 &> /dev/null; then
    echo "Installing ROS2 Humble..."
    locale-gen en_US en_US.UTF-8
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
    export LANG=en_US.UTF-8
    
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
    
    apt-get update
    apt-get install -y ros-humble-desktop-full python3-rosdep
    
    rosdep init || true
    rosdep update || true
    
    echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
fi

# Set executable permissions for scripts
echo "🔑 Setting executable permissions..."
chmod +x /workspace/scripts/automation/*.py || true
chmod +x /workspace/scripts/*.py || true

# Run tests to verify setup
echo "🧪 Running quick verification tests..."
cd /workspace
if [ -f "pytest.ini" ]; then
    python -m pytest tests/test_basic.py -v || echo "⚠️  Some tests failed, but continuing..."
fi

# Final setup messages
echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Quick start commands:"
echo "  • Run tests: make test"
echo "  • Start Jupyter: jupyter lab --ip 0.0.0.0 --port 8888 --allow-root"
echo "  • Lint code: make lint"
echo "  • Format code: make format"
echo "  • Start preference UI: python -m robo_rlhf.preference_server"
echo ""
echo "🔗 Available ports:"
echo "  • 8080 - Preference Collection UI"
echo "  • 8888 - Jupyter Lab"
echo "  • 6006 - TensorBoard"
echo "  • 3000 - Grafana Dashboard"
echo "  • 9090 - Prometheus Metrics"
echo ""
echo "Happy coding! 🚀"