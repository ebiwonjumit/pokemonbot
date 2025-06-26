#!/bin/bash

# Pokemon RL Bot Setup Script
# This script sets up the development environment for the Pokemon RL bot

set -e  # Exit on any error

echo "🎮 Pokemon RL Bot Setup Script"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on supported OS
check_os() {
    print_status "Checking operating system..."
    
    case "$(uname -s)" in
        Linux*)     OS="Linux";;
        Darwin*)    OS="macOS";;
        CYGWIN*)    OS="Windows";;
        MINGW*)     OS="Windows";;
        *)          OS="Unknown";;
    esac
    
    print_status "Detected OS: $OS"
    
    if [[ "$OS" != "Linux" && "$OS" != "macOS" ]]; then
        print_error "This script is designed for Linux and macOS. For Windows, please use WSL2."
        exit 1
    fi
}

# Check for required system dependencies
check_dependencies() {
    print_status "Checking system dependencies..."
    
    local missing_deps=()
    
    # Check for Python 3.8+
    if command -v python3 &> /dev/null; then
        python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        print_status "Found Python $python_version"
        
        if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_success "Python version is compatible"
        else
            print_error "Python 3.8+ is required, found $python_version"
            missing_deps+=("python3.8+")
        fi
    else
        print_error "Python 3 not found"
        missing_deps+=("python3")
    fi
    
    # Check for Git
    if ! command -v git &> /dev/null; then
        print_error "Git not found"
        missing_deps+=("git")
    else
        print_success "Git is available"
    fi
    
    # Check for curl/wget
    if ! command -v curl &> /dev/null && ! command -v wget &> /dev/null; then
        print_error "Neither curl nor wget found"
        missing_deps+=("curl or wget")
    else
        print_success "Download tool available"
    fi
    
    # OS-specific checks
    if [[ "$OS" == "Linux" ]]; then
        # Check for build essentials
        if ! command -v gcc &> /dev/null; then
            print_warning "GCC not found, may be needed for some packages"
        fi
        
        # Check for X11 development libraries (for VBA-M)
        if [[ ! -f /usr/include/X11/Xlib.h ]] && [[ ! -f /usr/local/include/X11/Xlib.h ]]; then
            print_warning "X11 development libraries may be needed for VBA-M emulator"
        fi
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_status "Please install missing dependencies and run this script again"
        
        if [[ "$OS" == "Linux" ]]; then
            print_status "On Ubuntu/Debian: sudo apt update && sudo apt install python3 python3-pip git curl build-essential libx11-dev"
            print_status "On CentOS/RHEL: sudo yum install python3 python3-pip git curl gcc gcc-c++ libX11-devel"
        elif [[ "$OS" == "macOS" ]]; then
            print_status "Install via Homebrew: brew install python3 git"
            print_status "Install Xcode Command Line Tools: xcode-select --install"
        fi
        
        exit 1
    fi
}

# Setup Python virtual environment
setup_venv() {
    print_status "Setting up Python virtual environment..."
    
    if [[ -d "venv" ]]; then
        print_warning "Virtual environment already exists"
        read -p "Remove existing venv and create new one? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf venv
        else
            print_status "Skipping virtual environment creation"
            return 0
        fi
    fi
    
    python3 -m venv venv
    
    if [[ ! -f "venv/bin/activate" ]]; then
        print_error "Failed to create virtual environment"
        exit 1
    fi
    
    print_success "Virtual environment created"
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    
    if [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_error "Virtual environment not found. Run setup first."
        exit 1
    fi
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Upgrade pip first
    python -m pip install --upgrade pip
    
    # Install dependencies from requirements.txt
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Setup VBA-M emulator
setup_vbam() {
    print_status "Setting up VBA-M emulator..."
    
    # Create emulator directory if it doesn't exist
    mkdir -p emulator
    
    if [[ "$OS" == "Linux" ]]; then
        if command -v vbam &> /dev/null; then
            print_success "VBA-M already installed system-wide"
            return 0
        fi
        
        print_status "Installing VBA-M for Linux..."
        
        # Try package manager first
        if command -v apt &> /dev/null; then
            sudo apt update
            sudo apt install -y vbam
        elif command -v yum &> /dev/null; then
            sudo yum install -y vbam
        elif command -v pacman &> /dev/null; then
            sudo pacman -S vbam-sdl
        else
            print_warning "Could not install VBA-M via package manager"
            print_status "Please install VBA-M manually or compile from source"
            print_status "GitHub: https://github.com/visualboyadvance-m/visualboyadvance-m"
        fi
        
    elif [[ "$OS" == "macOS" ]]; then
        if command -v vbam &> /dev/null; then
            print_success "VBA-M already installed"
            return 0
        fi
        
        print_status "Installing VBA-M for macOS..."
        
        if command -v brew &> /dev/null; then
            brew install vbam
        else
            print_warning "Homebrew not found"
            print_status "Please install VBA-M manually:"
            print_status "1. Install Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            print_status "2. Install VBA-M: brew install vbam"
        fi
    fi
    
    # Verify installation
    if command -v vbam &> /dev/null; then
        print_success "VBA-M emulator ready"
    else
        print_warning "VBA-M not found in PATH, you may need to install it manually"
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating project directories..."
    
    local dirs=(
        "roms"
        "saves"
        "models"
        "logs"
        "logs/tensorboard"
        "data"
        "data/screenshots"
        "data/recordings"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    done
    
    print_success "Project directories created"
}

# Setup configuration files
setup_config() {
    print_status "Setting up configuration files..."
    
    # Copy .env.example to .env if it doesn't exist
    if [[ ! -f ".env" ]] && [[ -f ".env.example" ]]; then
        cp .env.example .env
        print_success "Created .env file from template"
        print_warning "Please edit .env file to configure your settings"
    elif [[ -f ".env" ]]; then
        print_status ".env file already exists"
    else
        print_error ".env.example not found"
    fi
    
    # Create default config.json if it doesn't exist
    if [[ ! -f "config.json" ]]; then
        cat > config.json << EOF
{
    "emulator": {
        "vbam_path": "vbam",
        "rom_path": "roms/pokemon_leafgreen.gba",
        "save_path": "saves/pokemon_leafgreen.sav",
        "headless": true,
        "frame_skip": 4,
        "speed_multiplier": 1.0
    },
    "training": {
        "total_timesteps": 1000000,
        "learning_rate": 0.0003,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "clip_range": 0.2,
        "save_frequency": 10000,
        "log_frequency": 1000
    },
    "environment": {
        "action_space_size": 9,
        "observation_space": [84, 84, 4],
        "reward_scaling": 1.0,
        "max_episode_steps": 10000
    },
    "web": {
        "host": "0.0.0.0",
        "port": 5000,
        "debug": false,
        "secret_key": "pokemon-rl-production-key"
    },
    "logging": {
        "level": "INFO",
        "file": "logs/pokemon_rl.log",
        "max_size": "10MB",
        "backup_count": 5
    }
}
EOF
        print_success "Created default config.json"
    else
        print_status "config.json already exists"
    fi
}

# Download sample ROM (placeholder)
setup_rom() {
    print_status "Setting up ROM files..."
    
    if [[ ! -f "roms/pokemon_leafgreen.gba" ]]; then
        print_warning "Pokemon Leaf Green ROM not found"
        print_status "Please place your legally obtained Pokemon Leaf Green ROM file at:"
        print_status "  roms/pokemon_leafgreen.gba"
        print_status ""
        print_status "Note: You must own a legal copy of Pokemon Leaf Green to use this bot."
        
        # Create a placeholder file
        touch "roms/pokemon_leafgreen.gba"
        echo "# Place your Pokemon Leaf Green ROM here" > "roms/README.txt"
    else
        print_success "ROM file found"
    fi
}

# Setup GPU support (CUDA/MPS)
setup_gpu() {
    print_status "Checking for GPU support..."
    
    # Check for NVIDIA GPU (CUDA)
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        
        # Check if PyTorch with CUDA is installed
        if python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            print_success "PyTorch with CUDA support is ready"
        else
            print_warning "PyTorch CUDA support not detected"
            print_status "Consider installing PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        fi
    elif [[ "$OS" == "macOS" ]]; then
        # Check for Apple Silicon (MPS)
        if python -c "import torch; print('MPS available:', torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
            print_success "Apple Silicon GPU (MPS) support detected"
        else
            print_warning "MPS support not available or not properly configured"
        fi
    else
        print_status "No GPU acceleration detected, will use CPU"
    fi
}

# Run tests
run_tests() {
    print_status "Running basic tests..."
    
    # Test Python imports
    python -c "
import sys
print(f'Python version: {sys.version}')

# Test core imports
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
except ImportError as e:
    print(f'PyTorch import error: {e}')

try:
    import gym
    print(f'Gym version: {gym.__version__}')
except ImportError as e:
    print(f'Gym import error: {e}')

try:
    import stable_baselines3
    print(f'Stable-Baselines3 version: {stable_baselines3.__version__}')
except ImportError as e:
    print(f'Stable-Baselines3 import error: {e}')

try:
    import flask
    print(f'Flask version: {flask.__version__}')
except ImportError as e:
    print(f'Flask import error: {e}')

try:
    import cv2
    print(f'OpenCV version: {cv2.__version__}')
except ImportError as e:
    print(f'OpenCV import error: {e}')
" || print_warning "Some imports failed"
    
    print_success "Basic tests completed"
}

# Main setup function
main() {
    echo
    print_status "Starting Pokemon RL Bot setup..."
    echo
    
    # Change to script directory
    cd "$(dirname "$0")/.."
    
    check_os
    check_dependencies
    setup_venv
    activate_venv
    install_python_deps
    setup_vbam
    create_directories
    setup_config
    setup_rom
    setup_gpu
    run_tests
    
    echo
    print_success "Setup completed successfully!"
    echo
    print_status "Next steps:"
    print_status "1. Place your Pokemon Leaf Green ROM in roms/pokemon_leafgreen.gba"
    print_status "2. Edit .env and config.json files to configure your settings"
    print_status "3. Run training: python scripts/train.py"
    print_status "4. Start web dashboard: python scripts/monitor.py"
    echo
    print_status "To activate the virtual environment later:"
    print_status "  source venv/bin/activate"
    echo
    print_success "Happy training! 🎮"
}

# Handle interrupts gracefully
trap 'print_error "Setup interrupted by user"; exit 1' INT TERM

# Run main function
main "$@"
