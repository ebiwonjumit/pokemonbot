#!/usr/bin/env python3
"""
Pokemon RL Bot Deployment Manager
Simple Python wrapper for deployment operations with your pokebot-45 GCP project.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"🚀 {description}")
    print(f"Running: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def deploy_to_gcp():
    """Deploy to Google Cloud Platform using your pokebot-45 project."""
    script_path = Path(__file__).parent / "deploy-gcp.sh"
    return run_command(["bash", str(script_path), "deploy"], "GCP Deployment (pokebot-45)")

def deploy_gpu():
    """Deploy to GCP with GPU support."""
    script_path = Path(__file__).parent / "deploy-gcp.sh"
    return run_command(["bash", str(script_path), "deploy-gpu"], "GCP GPU Deployment (pokebot-45)")

def start_instance():
    """Start the GCP instance."""
    script_path = Path(__file__).parent / "deploy-gcp.sh"
    return run_command(["bash", str(script_path), "start"], "Start GCP Instance (pokemon-rl-bot)")

def stop_instance():
    """Stop the GCP instance."""
    script_path = Path(__file__).parent / "deploy-gcp.sh"
    return run_command(["bash", str(script_path), "stop"], "Stop GCP Instance (pokemon-rl-bot)")

def test_emulator():
    """Test the emulator on GCP."""
    script_path = Path(__file__).parent / "deploy-gcp.sh"
    return run_command(["bash", str(script_path), "test-emulator"], "Test Emulator on GCP")

def show_status():
    """Show instance status."""
    script_path = Path(__file__).parent / "deploy-gcp.sh"
    return run_command(["bash", str(script_path), "status"], "Show Instance Status")

def main():
    parser = argparse.ArgumentParser(description="Pokemon RL Bot Deployment Manager")
    parser.add_argument("action", choices=[
        "deploy", "deploy-gpu", "start", "stop", "status", "test-emulator", "help"
    ], help="Action to perform")
    
    args = parser.parse_args()
    
    print("🎮 Pokemon RL Bot - Deployment Manager")
    print("=" * 40)
    print(f"GCP Project: pokebot-45")
    print(f"Instance: pokemon-rl-bot")
    print(f"Zone: us-central1-a")
    print("=" * 40)
    
    if args.action == "deploy":
        success = deploy_to_gcp()
    elif args.action == "deploy-gpu":
        success = deploy_gpu()
    elif args.action == "start":
        success = start_instance()
    elif args.action == "stop":
        success = stop_instance()
    elif args.action == "status":
        success = show_status()
    elif args.action == "test-emulator":
        success = test_emulator()
    elif args.action == "help":
        print("\n📋 Available Actions:")
        print("  deploy        - Deploy bot to GCP (CPU)")
        print("  deploy-gpu    - Deploy bot to GCP (GPU)")
        print("  start         - Start the GCP instance")
        print("  stop          - Stop the GCP instance")
        print("  status        - Show instance status")
        print("  test-emulator - Test VBA-M emulator on GCP")
        print("\n🚀 Quick Start:")
        print("  python scripts/deploy.py deploy")
        print("  python scripts/deploy.py start")
        print("  python scripts/deploy.py test-emulator")
        return True
    else:
        print(f"❌ Unknown action: {args.action}")
        return False
    
    if not success:
        print("\n❌ Deployment failed!")
        sys.exit(1)
    else:
        print("\n✅ Operation completed successfully!")

if __name__ == "__main__":
    main()
