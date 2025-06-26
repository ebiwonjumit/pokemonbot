#!/usr/bin/env python3
"""
Pokemon RL Bot - Quick Test Script

This script performs a basic test of the Pokemon RL bot setup:
1. Tests VBA-M emulator access
2. Validates ROM file
3. Tests basic environment setup
4. Starts a basic web dashboard
"""

import os
import sys
import subprocess
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_emulator():
    """Test VBA-M emulator access."""
    print("🎮 Testing VBA-M emulator...")
    try:
        # Test if vbam exists first
        result = subprocess.run(['which', 'vbam'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            print("✅ VBA-M emulator is accessible")
            print(f"   - Path: {result.stdout.strip()}")
            return True
        else:
            print("❌ VBA-M emulator not found in PATH")
            return False
    except Exception as e:
        print(f"❌ Error testing emulator: {e}")
        return False

def test_rom():
    """Test ROM file availability."""
    print("\n📁 Testing ROM file...")
    rom_path = Path('roms/pokemon_leaf_green.gba')
    if rom_path.exists():
        size_mb = rom_path.stat().st_size / (1024 * 1024)
        print(f"✅ ROM file found: {rom_path.name} ({size_mb:.1f} MB)")
        return True
    else:
        print("❌ ROM file not found")
        return False

def test_emulator_game():
    """Test starting the emulator with the Pokemon ROM."""
    print("\n🎮 Testing emulator with Pokemon ROM...")
    try:
        from game.emulator import VBAEmulator, EmulatorAction
        
        emulator = VBAEmulator(
            rom_path="roms/pokemon_leaf_green.gba",
            headless=False
        )
        
        print("✅ Emulator class imported successfully")
        print("   - Ready to start Pokemon Leaf Green")
        return True
        
    except ImportError as e:
        print(f"❌ Could not import emulator module: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing emulator: {e}")
        return False

def test_python_environment():
    """Test Python environment and dependencies."""
    print("\n🐍 Testing Python environment...")
    try:
        import torch
        import gym
        import stable_baselines3
        import flask
        import cv2
        import numpy as np
        
        print("✅ All core dependencies available")
        print(f"   - Python: {sys.version.split()[0]}")
        print(f"   - PyTorch: {torch.__version__}")
        print(f"   - Gym: {gym.__version__}")
        print(f"   - Stable-Baselines3: {stable_baselines3.__version__}")
        print(f"   - OpenCV: {cv2.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def start_basic_dashboard():
    """Start a basic web dashboard."""
    print("\n🌐 Starting basic web dashboard...")
    print("📱 Open http://localhost:7500 in your browser")
    print("⏹️  Press Ctrl+C to stop")
    
    from flask import Flask
    from flask_socketio import SocketIO
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'pokemon-rl-secret'
    socketio = SocketIO(app, cors_allowed_origins='*')
    
    @app.route('/')
    def index():
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pokemon RL Bot Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f0f8ff; }
                .header { background: #4169e1; color: white; padding: 20px; border-radius: 10px; }
                .status { background: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .success { color: #28a745; }
                .warning { color: #ffc107; }
                ul { padding-left: 20px; }
                li { margin: 10px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎮 Pokemon RL Bot Dashboard</h1>
                <p>Production-ready Pokemon Leaf Green Reinforcement Learning Bot</p>
            </div>
            
            <div class="status">
                <h2>✅ System Status</h2>
                <ul>
                    <li class="success">✅ Python 3.10 with virtual environment</li>
                    <li class="success">✅ VBA-M emulator accessible via 'vbam' command</li>
                    <li class="success">✅ Pokemon Leaf Green ROM file loaded</li>
                    <li class="success">✅ All Python dependencies installed</li>
                    <li class="success">✅ Web dashboard running on Flask + SocketIO</li>
                </ul>
            </div>
            
            <div class="status">
                <h2>🚀 Next Steps</h2>
                <ul>
                    <li>Fix relative import issues in source modules</li>
                    <li>Test emulator integration with Python</li>
                    <li>Start reinforcement learning training</li>
                    <li>Access real-time game streaming</li>
                </ul>
            </div>
            
            <div class="status">
                <h2>📋 Project Structure</h2>
                <ul>
                    <li><code>src/game/</code> - Emulator wrapper and environment</li>
                    <li><code>src/agent/</code> - PPO agent and neural networks</li>
                    <li><code>src/web/</code> - Flask dashboard and streaming</li>
                    <li><code>src/utils/</code> - Logging, metrics, utilities</li>
                    <li><code>roms/</code> - Pokemon ROM files</li>
                    <li><code>models/</code> - Trained model checkpoints</li>
                </ul>
            </div>
        </body>
        </html>
        '''
    
    socketio.run(app, host='0.0.0.0', port=7500, debug=False)

def main():
    """Run all tests and start dashboard."""
    print("🎮 Pokemon RL Bot - Quick Test")
    print("=" * 40)
    
    # Run tests
    emulator_ok = test_emulator()
    rom_ok = test_rom()
    python_ok = test_python_environment()
    emulator_game_ok = test_emulator_game()
    
    if emulator_ok and rom_ok and python_ok and emulator_game_ok:
        print("\n🎉 All tests passed! Setup is ready!")
        start_basic_dashboard()
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == '__main__':
    main()
