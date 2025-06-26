# Pokemon RL Bot - Reorganized Project Structure

## 📁 New Directory Organization

```
pokemon-rl-bot/
├── src/                           # Core application code
│   ├── game/                      # Game environment and emulator
│   ├── agent/                     # RL agent and neural networks  
│   ├── web/                       # Flask web dashboard
│   └── utils/                     # Utilities and helpers
├── scripts/                       # Executable scripts
│   ├── deploy.py                  # Python deployment manager
│   ├── deploy-gcp.sh             # GCP deployment script  
│   ├── run_tests.py              # Test runner
│   ├── simple_dashboard.py       # Simple web dashboard
│   ├── setup.sh                  # Environment setup
│   ├── train.py                  # Training script
│   ├── monitor.py                # Monitoring script
│   └── tests/                    # Test scripts
│       ├── test_setup.py         # Environment setup test
│       ├── test_bot_control.py   # Bot control test
│       ├── quick_emulator_test.py # Quick emulator test
│       └── test_cloud_emulator.sh # Cloud emulator test
├── roms/                         # Pokemon ROM files
├── models/                       # Trained models
├── logs/                         # Training logs
├── data/                         # Data files
└── saves/                        # Game save states
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
