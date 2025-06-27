# Pokemon Leaf Green Reinforcement Learning Bot

AI agent that learns to play Pokemon Leaf Green using PPO reinforcement learning with VBA-M emulator integration.

## Features

- **PPO Agent**: Trains on 84x84 game frames using custom CNN architecture
- **VBA-M Integration**: Controls Game Boy Advance emulator directly
- **Web Dashboard**: Real-time monitoring at http://localhost:7500
- **Docker Support**: CPU and GPU deployment options

## Architecture

```
src/
├── game/           # Emulator wrapper and environment  
├── agent/          # PPO agent and neural networks
├── web/            # Flask dashboard
└── utils/          # Logging and metrics
```

## Quick Start

### Prerequisites
- Python 3.8+
- VBA-M emulator: `brew install visualboyadvance-m` (macOS)
- Pokemon Leaf Green ROM (place in `roms/` directory)

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Copy ROM file
cp /path/to/pokemon_leaf_green.gba roms/

# Start web dashboard
python scripts/simple_dashboard.py
# Open http://localhost:7500
```

### Training
```bash
# Start training
python scripts/train.py --config config.json

# Monitor via dashboard
python scripts/monitor.py --config config.json
```

## Configuration

### Key Settings (config.json)
- **Learning Rate**: 3e-4
- **Batch Size**: 64  
- **Frame Stack**: 4 (84x84 grayscale frames)
- **Action Space**: 9 discrete actions (D-pad + A,B,Start,Select + No-op)

### Reward System
- **+100** per gym badge
- **+10** per new Pokemon caught
- **-0.01** per step (efficiency)
- **Exploration bonuses** for new areas

## Docker

```bash
# CPU version
docker-compose up

# GPU version  
docker-compose -f docker-compose.gpu.yml up
```

## Testing

```bash
# Run all tests
python scripts/run_tests.py

# Individual tests  
python scripts/tests/test_setup.py
python scripts/tests/test_bot_control.py
```

## 🚀 Cloud Deployment (GCP)

Your project is configured for your **pokebot-45** GCP project:

```bash
# Deploy to Google Cloud Platform
python scripts/deploy.py deploy

# Check status
python scripts/deploy.py status

# Test emulator in cloud
python scripts/deploy.py test-emulator

# Manage instance
python scripts/deploy.py start|stop
```

**GCP Configuration:**
- Project: `pokebot-45`
- Instance: `pokemon-rl-bot`
- Zone: `us-central1-a`
- Machine: `n1-standard-4` (CPU) or `deploy-gpu` for GPU support
