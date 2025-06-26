# GitHub Copilot Instructions for Pokemon RL Bot

This project is a **Pokemon Leaf Green Reinforcement Learning Bot** built with Python, using VBA-M emulator, OpenAI Gym, Stable-Baselines3 PPO, and Flask for web monitoring.

## Project Overview

**Goal**: Create an AI agent that learns to play Pokemon Leaf Green using reinforcement learning.

**Key Components**:
- **Emulator Integration**: VBA-M Game Boy Advance emulator with Python bindings
- **RL Environment**: Custom OpenAI Gym environment for Pokemon game states
- **Agent**: PPO (Proximal Policy Optimization) agent with custom CNN architecture
- **Web Dashboard**: Real-time monitoring and control interface
- **Cloud Deployment**: Docker containerization with AWS/GCP support

## Architecture

```
src/
├── game/           # Game environment and emulator control
├── agent/          # RL agent and neural networks
├── web/            # Flask web dashboard
└── utils/          # Logging, metrics, cloud storage

scripts/            # Training, monitoring, deployment scripts
```

## Key Technologies

- **Python 3.9+** - Main programming language
- **PyTorch** - Deep learning framework
- **Stable-Baselines3** - RL algorithms (PPO)
- **OpenAI Gym** - RL environment interface
- **VBA-M** - Game Boy Advance emulator
- **Flask + Socket.IO** - Web dashboard
- **OpenCV** - Image processing
- **Docker** - Containerization
- **AWS/GCP** - Cloud deployment

## Development Guidelines

### Code Style
- Follow PEP 8 with 100 character line limit
- Use type hints for all functions
- Comprehensive docstrings for classes and methods
- Error handling with proper logging

### File Organization
- Keep related functionality together in modules
- Use clear, descriptive naming
- Separate concerns (game logic, RL agent, web interface)

### Testing
- Unit tests for core functionality
- Integration tests for emulator interface
- Mock external dependencies where appropriate

## Common Tasks

### Adding New Features
1. **Game State Processing**: Modify `src/game/state_parser.py`
2. **Reward Engineering**: Update `src/game/reward_calculator.py`
3. **Neural Network Changes**: Edit `src/agent/neural_network.py`
4. **Web Interface**: Add to `src/web/` components

### Training Improvements
- Experiment with different CNN architectures
- Adjust reward functions for better learning
- Implement curriculum learning strategies
- Add new observation features

### Debugging
- Use comprehensive logging throughout
- Monitor training metrics via TensorBoard
- Real-time debugging through web dashboard
- Check emulator state and game progression

## Best Practices

1. **Performance**: Optimize image processing and state extraction
2. **Stability**: Handle emulator crashes gracefully
3. **Monitoring**: Track training progress and system resources
4. **Documentation**: Keep README and code comments updated
5. **Version Control**: Use semantic versioning for releases

## Environment Setup

The project uses virtual environments and has specific dependencies:
- VBA-M emulator must be installed system-wide
- CUDA support for GPU training (optional)
- ROM files are required but not included (legal reasons)

## Deployment Options

- **Local**: Direct Python execution
- **Docker**: CPU and GPU variants available
- **Cloud**: AWS ECS, GCP Cloud Run, Azure Container Instances
- **Monitoring**: Prometheus + Grafana stack included

When helping with this project, consider the RL training pipeline, real-time web monitoring requirements, and production deployment constraints.
