#!/bin/bash
# =============================================================================
# THEMIS Environment Setup Script for Ubuntu 20.04
# =============================================================================

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    echo -e "${2}${1}${NC}"
}

print_info() {
    print_msg "[INFO] $1" "$BLUE"
}

print_success() {
    print_msg "[SUCCESS] $1" "$GREEN"
}

print_warning() {
    print_msg "[WARNING] $1" "$YELLOW"
}

print_error() {
    print_msg "[ERROR] $1" "$RED"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root is not recommended. Please run as a regular user."
    exit 1
fi

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

print_info "Project directory: $PROJECT_DIR"

# =============================================================================
# Step 1: System Dependencies
# =============================================================================
print_info "Step 1: Installing system dependencies..."

sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    vim \
    htop \
    tmux \
    tree \
    zip \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

print_success "System dependencies installed."

# =============================================================================
# Step 2: Python Environment
# =============================================================================
print_info "Step 2: Setting up Python environment..."

# Install Python 3.10 (recommended for this project)
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3.10-distutils \
    python3-pip

# Set Python 3.10 as default
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Install pip for Python 3.10
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

print_success "Python environment setup complete."

# =============================================================================
# Step 3: CUDA and cuDNN (if NVIDIA GPU available)
# =============================================================================
print_info "Step 3: Checking CUDA installation..."

if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    
    # Check if CUDA is installed
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -c2-)
        print_success "CUDA version: $CUDA_VERSION"
    else
        print_warning "CUDA not found. Please install CUDA manually if needed."
        print_info "Recommended: CUDA 11.8 or 12.1"
        print_info "Visit: https://developer.nvidia.com/cuda-downloads"
    fi
else
    print_warning "No NVIDIA GPU detected. Skipping CUDA setup."
fi

# =============================================================================
# Step 4: Create Virtual Environment
# =============================================================================
print_info "Step 4: Creating Python virtual environment..."

VENV_DIR="${PROJECT_DIR}/.venv"

if [ -d "$VENV_DIR" ]; then
    print_warning "Virtual environment already exists at $VENV_DIR"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        python3.10 -m venv "$VENV_DIR"
        print_success "Virtual environment recreated."
    fi
else
    python3.10 -m venv "$VENV_DIR"
    print_success "Virtual environment created at $VENV_DIR"
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Upgrade pip
pip install --upgrade pip setuptools wheel

print_success "Virtual environment activated."

# =============================================================================
# Step 5: Install PyTorch
# =============================================================================
print_info "Step 5: Installing PyTorch..."

# Detect CUDA version for PyTorch installation
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    
    if [[ "$CUDA_VERSION" == "12."* ]]; then
        print_info "Installing PyTorch with CUDA 12.1 support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$CUDA_VERSION" == "11."* ]]; then
        print_info "Installing PyTorch with CUDA 11.8 support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        print_warning "Unknown CUDA version. Installing CPU version of PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
else
    print_info "Installing CPU version of PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

print_success "PyTorch installed."

# =============================================================================
# Step 6: Install Project Dependencies
# =============================================================================
print_info "Step 6: Installing project dependencies..."

# Install requirements
if [ -f "${PROJECT_DIR}/requirements-agentic-eval.txt" ]; then
    pip install -r "${PROJECT_DIR}/requirements-agentic-eval.txt"
    print_success "requirements-agentic-eval.txt installed."
fi

# Install additional dependencies for new models
pip install \
    transformers>=4.40.0 \
    accelerate>=0.28.0 \
    bitsandbytes>=0.43.0 \
    sentencepiece \
    protobuf \
    timm \
    einops \
    flash-attn --no-build-isolation \
    ultralytics \
    pyiqa \
    paddleocr \
    paddlepaddle-gpu \
    timm \
    kornia \
    albumentations \
    opencv-python \
    pillow \
    scipy \
    scikit-learn \
    pandas \
    matplotlib \
    seaborn \
    tqdm \
    pyyaml \
    python-dotenv \
    langgraph \
    langchain \
    requests \
    aiohttp

print_success "Project dependencies installed."

# =============================================================================
# Step 7: Install vLLM for Efficient Inference (Optional)
# =============================================================================
print_info "Step 7: Installing vLLM for efficient inference (optional)..."

read -p "Do you want to install vLLM for efficient model inference? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install vllm
    print_success "vLLM installed."
else
    print_info "Skipping vLLM installation."
fi

# =============================================================================
# Step 8: Create Environment Variables File
# =============================================================================
print_info "Step 8: Creating environment configuration..."

ENV_FILE="${PROJECT_DIR}/.env"

if [ -f "$ENV_FILE" ]; then
    print_warning ".env file already exists."
else
    cat > "$ENV_FILE" << 'EOF'
# =============================================================================
# THEMIS Environment Configuration
# =============================================================================

# Model Paths (adjust according to your setup)
MODEL_DIR=/path/to/THEMIS/models

# Qwen VL Models
PLANNER_MODEL_PATH=${MODEL_DIR}/Qwen2.5-VL-3B-Instruct
JUDGE_MODEL_PATH=${MODEL_DIR}/Qwen2.5-VL-3B-Instruct
REFLECTOR_MODEL_PATH=${MODEL_DIR}/Qwen2.5-VL-7B-Instruct

# Expert Model Paths
IMAGENET_MODEL_PATH=${MODEL_DIR}/efficientnetv2_s
CLIP_MODEL_PATH=${MODEL_DIR}/clip_vit_b32
PLACES365_MODEL_PATH=${MODEL_DIR}/places365
YOLO_MODEL_PATH=${MODEL_DIR}/yolo

# Device Configuration
DEVICE=cuda
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Model Profiles (fast, standard, strong)
PLANNER_PROFILE=fast
JUDGE_PROFILE=fast
REFLECTOR_PROFILE=standard

# Local expert routing
LOCAL_PRIMARY_MODEL=${MODEL_DIR}/Qwen2.5-VL-3B-Instruct
LOCAL_SEMANTIC_ENABLED=true
LOCAL_SEMANTIC_MODEL=${MODEL_DIR}/Qwen2.5-VL-3B-Instruct
SEMANTIC_LOCAL_FAST_MODEL=${MODEL_DIR}/Qwen2.5-VL-3B-Instruct
SEMANTIC_LOCAL_STRONGER_MODEL=${MODEL_DIR}/Qwen2.5-VL-7B-Instruct
LOCAL_SEMANTIC_DEVICE=cuda:3
LOCAL_ARTIFACT_ENABLED=true
LOCAL_ARTIFACT_METRICS=maniqa,musiq,niqe
LOCAL_ARTIFACT_DEVICE=cuda:4
REPORT_MODEL=${MODEL_DIR}/Qwen2.5-VL-7B-Instruct

# Evaluation Settings
MAX_PLAN_REVISIONS=2
MAX_REFLECTION_REVISIONS=1
SEMANTIC_ESCALATION_THRESHOLD=0.72

# Logging
LOG_LEVEL=INFO
LOG_DIR=./logs
EOF
    print_success ".env file created at $ENV_FILE"
    print_info "Please edit $ENV_FILE to configure your environment."
fi

# =============================================================================
# Step 9: Create Necessary Directories
# =============================================================================
print_info "Step 9: Creating project directories..."

mkdir -p "${PROJECT_DIR}/models"
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/outputs"
mkdir -p "${PROJECT_DIR}/cache"

print_success "Project directories created."

# =============================================================================
# Step 10: Verify Installation
# =============================================================================
print_info "Step 10: Verifying installation..."

python3 << 'PYTHON_SCRIPT'
import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError as e:
    print(f"PyTorch import error: {e}")

try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except ImportError:
    print("Transformers not installed")

try:
    import ultralytics
    print(f"Ultralytics version: {ultralytics.__version__}")
except ImportError:
    print("Ultralytics not installed")

try:
    import pyiqa
    print(f"pyiqa installed")
except ImportError:
    print("pyiqa not installed")

print("\nInstallation verification complete!")
PYTHON_SCRIPT

print_success "Installation verification complete."

# =============================================================================
# Final Instructions
# =============================================================================
echo ""
echo "=========================================="
echo "THEMIS Environment Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Download models: python scripts/download_models.py --all --model-dir ./models"
echo "3. Test models: python scripts/test_models.py"
echo "4. Run evaluation: python -m src.agentic_eval.run_single --help"
echo ""
echo "To activate the virtual environment in the future:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "For more information, see DEPLOYMENT.md"
echo "=========================================="
