#!/usr/bin/env python3
"""
Pokemon RL Bot Deployment Manager
Simple Python wrapper for deployment operations.
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
    """Deploy to Google Cloud Platform."""
    script_path = Path(__file__).parent / "deploy-gcp.sh"
    return run_command(["bash", str(script_path), "deploy"], "GCP Deployment")

def start_instance():
    """Start the GCP instance."""
    script_path = Path(__file__).parent / "deploy-gcp.sh"
    return run_command(["bash", str(script_path), "start"], "Start GCP Instance")

def stop_instance():
    """Stop the GCP instance."""
    script_path = Path(__file__).parent / "deploy-gcp.sh"
    return run_command(["bash", str(script_path), "stop"], "Stop GCP Instance")

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
        "deploy", "start", "stop", "status", "test-emulator", "help"
    ], help="Action to perform")
    
    args = parser.parse_args()
    
    print("🎮 Pokemon RL Bot - Deployment Manager")
    print("=" * 40)
    
    if args.action == "deploy":
        success = deploy_to_gcp()
    elif args.action == "start":
        success = start_instance()
    elif args.action == "stop":
        success = stop_instance()
    elif args.action == "status":
        success = show_status()
    elif args.action == "test-emulator":
        success = test_emulator()
    elif args.action == "help":
        parser.print_help()
        print("\nAvailable actions:")
        print("  deploy        - Deploy bot to GCP (full setup)")
        print("  start         - Start existing GCP instance")
        print("  stop          - Stop GCP instance")
        print("  status        - Show instance status")
        print("  test-emulator - Test VBA-M emulator on GCP")
        return 0
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
