# Pokemon RL Bot - Project Structure

This document outlines the complete organized project structure for the Pokemon Leaf Green Reinforcement Learning Bot.

## Directory Layout

```
pokemonBot/
├── README.md                   # Main project documentation
├── requirements.txt           # Python dependencies
├── config.json               # Bot configuration
├── .env.example              # Environment variables template
├── Dockerfile                # Docker container (CPU)
├── Dockerfile.gpu            # Docker container (GPU)
├── docker-compose.yml        # Multi-container setup
├── .gitignore               # Git ignore rules
│
├── src/                     # Source code
│   ├── __init__.py
│   ├── game/                # Game environment and emulator control
│   │   ├── __init__.py
│   │   ├── emulator.py      # VBA-M emulator interface
│   │   ├── environment.py   # OpenAI Gym environment
│   │   ├── state_parser.py  # Game state extraction
│   │   └── reward_calculator.py # Reward function logic
│   │
│   ├── agent/               # RL agent and neural networks
│   │   ├── __init__.py
│   │   ├── ppo_agent.py     # PPO implementation
│   │   ├── neural_network.py # CNN architecture
│   │   └── trainer.py       # Training loop
│   │
│   ├── utils/               # Utilities and helpers
│   │   ├── __init__.py
│   │   ├── logger.py        # Logging system
│   │   ├── metrics.py       # Performance metrics
│   │   └── cloud_storage.py # Cloud storage integration
│   │
│   └── web/                 # Web dashboard
│       ├── __init__.py
│       ├── app.py           # Flask application
│       ├── websocket_handler.py # Real-time updates
│       ├── templates/       # HTML templates
│       │   └── index.html
│       └── static/          # CSS/JS files
│           ├── style.css
│           └── script.js
│
├── scripts/                 # Deployment and utility scripts
│   ├── train.py            # Start training
│   ├── deploy-gcp.sh       # GCP deployment script
│   ├── deploy.py           # Cloud deployment manager
│   ├── deploy.sh           # General deployment script
│   ├── setup.sh            # Environment setup
│   ├── monitor.py          # System monitoring
│   ├── run_tests.py        # Test runner
│   ├── capture_emulator.py # Screen capture utility
│   ├── simple_dashboard.py # Basic dashboard
│   │
│   ├── dashboards/         # Dashboard variants
│   │   ├── local_dashboard.py    # macOS optimized dashboard
│   │   ├── cloud_dashboard.py    # Headless cloud dashboard
│   │   └── enhanced_dashboard.py # Full-featured dashboard
│   │
│   └── tests/              # Test scripts
│       ├── quick_emulator_test.py
│       ├── test_bot_control.py
│       ├── test_cloud_emulator.sh
│       └── test_setup.py
│
├── docs/                   # Documentation
│   ├── CLOUD_ARCHITECTURE.md
│   ├── DEPLOYMENT_STATUS.md
│   ├── FIRST_TEST_SUMMARY.md
│   ├── GCP_DEPLOYMENT_READY.md
│   ├── GCP_INSTANCE_CONFIG.md
│   └── PROJECT_STRUCTURE.md
│
├── data/                   # Training data and recordings
│   ├── recordings/         # Game recordings
│   └── screenshots/        # Debug screenshots
│
├── models/                 # Trained models
│   └── (saved model files)
│
├── logs/                   # Application logs
│   ├── training.log        # Training progress
│   ├── agent.log          # Agent actions
│   ├── emulator.log       # Emulator events
│   └── web.log            # Dashboard logs
│
├── roms/                   # Game ROM files
│   └── pokemon_leaf_green.gba
│
└── saves/                  # Game save states
    └── (save files)
```

## 🚀 Updated Commands

### Local Testing
```bash
# Run all tests
python scripts/run_tests.py

# Run individual tests
python scripts/tests/test_setup.py
python scripts/tests/test_bot_control.py

# Start local dashboard
python scripts/simple_dashboard.py
```

### Cloud Deployment
```bash
# Using Python wrapper (recommended)
python scripts/deploy.py deploy      # Full deployment
python scripts/deploy.py start       # Start instance
python scripts/deploy.py stop        # Stop instance
python scripts/deploy.py status      # Check status
python scripts/deploy.py test-emulator # Test VBA-M

# Using bash script directly
./scripts/deploy-gcp.sh deploy
./scripts/deploy-gcp.sh test-emulator
```

### Training
```bash
# Local training
python scripts/train.py --config config.json

# Monitor training
python scripts/monitor.py --config config.json
```

## ✅ Benefits of This Organization

1. **Clear Separation**: Scripts vs core code vs tests
2. **Easy Navigation**: Everything in logical directories
3. **Proper Paths**: All imports work correctly
4. **Maintainable**: Easy to add new scripts/tests
5. **Git Friendly**: Proper .gitignore patterns

## 🧪 Test Structure

- **test_setup.py**: Checks Python environment and dependencies
- **test_bot_control.py**: Tests game control and automation
- **quick_emulator_test.py**: Quick VBA-M functionality test
- **test_cloud_emulator.sh**: Cloud-specific emulator testing
- **run_tests.py**: Runs all tests with summary

## 📋 Next Steps

1. Test the reorganized structure: `python scripts/run_tests.py`
2. Deploy to cloud: `python scripts/deploy.py deploy`
3. Test cloud emulator: `python scripts/deploy.py test-emulator`
4. Start training in the cloud!

All paths and imports have been updated to work with the new structure.

## Key Components

### Dashboards
- **simple_dashboard.py**: Basic web interface for testing
- **local_dashboard.py**: macOS optimized with native screen capture
- **cloud_dashboard.py**: Headless Linux cloud environment
- **enhanced_dashboard.py**: Full-featured with live video feed

### Deployment Scripts
- **deploy-gcp.sh**: Automated GCP Compute Engine deployment
- **deploy.py**: Python deployment manager with cloud integration
- **setup.sh**: Local environment setup

### Test Scripts
- **test_setup.py**: Verify local environment
- **test_bot_control.py**: Test bot control systems
- **quick_emulator_test.py**: Quick emulator functionality test
- **test_cloud_emulator.sh**: Cloud environment emulator test

## Usage Commands

### Local Development
```bash
# Start local dashboard
python scripts/simple_dashboard.py

# Start macOS optimized dashboard
python scripts/dashboards/local_dashboard.py

# Run tests
python scripts/run_tests.py
python scripts/tests/test_setup.py
```

### Cloud Deployment
```bash
# Deploy to GCP
./scripts/deploy-gcp.sh deploy

# Start cloud dashboard
python scripts/dashboards/cloud_dashboard.py

# Monitor deployment
python scripts/monitor.py
```

### Training
```bash
# Start training locally
python scripts/train.py

# Start training with config
python scripts/train.py --config config.json
```

## File Organization Principles

1. **Separation of Concerns**: Core logic (`src/`), scripts (`scripts/`), docs (`docs/`)
2. **Environment-Specific**: Separate dashboards and scripts for local vs cloud
3. **Modular Structure**: Clear separation between game, agent, utils, and web components
4. **Easy Discovery**: Logical grouping in subdirectories with clear naming

This structure provides better maintainability, easier navigation, and clearer separation between development, testing, and deployment components.
