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
python simple_dashboard.py
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
# Test setup
python test_setup.py

# Test bot control
python test_bot_control.py
```
