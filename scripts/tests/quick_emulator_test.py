#!/usr/bin/env python3
"""
Quick emulator test to verify VBA-M can start Pokemon Leaf Green
"""

import subprocess
import time
import os
import signal
from pathlib import Path

def test_emulator_launch():
    """Test launching VBA-M with Pokemon ROM."""
    print("🎮 Testing VBA-M emulator launch...")
    
    rom_path = Path("roms/pokemon_leaf_green.gba")
    if not rom_path.exists():
        print("❌ ROM file not found!")
        return False
    
    try:
        # Start VBA-M with the ROM
        cmd = [
            "vbam",
            str(rom_path),
            # Remove invalid options for now - will configure later
        ]
        
        print(f"Starting: {' '.join(cmd)}")
        process = subprocess.Popen(cmd)
        
        # Let it run for a few seconds
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ VBA-M started successfully!")
            print("📱 You should see the Pokemon Leaf Green game window")
            print("⏹️  Stopping emulator in 5 seconds...")
            
            time.sleep(5)
            
            # Terminate the process
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                
            print("✅ Emulator stopped successfully")
            return True
        else:
            print(f"❌ VBA-M failed to start. Exit code: {process.returncode}")
            return False
            
    except FileNotFoundError:
        print("❌ VBA-M executable not found")
        return False
    except Exception as e:
        print(f"❌ Error starting emulator: {e}")
        return False

def test_emulator_with_controls():
    """Test VBA-M with basic controls."""
    print("\n🕹️ Testing VBA-M with controls...")
    
    rom_path = Path("roms/pokemon_leaf_green.gba")
    
    try:
        # Start VBA-M
        cmd = [
            "vbam",
            str(rom_path),
        ]
        
        print("Starting emulator for control test...")
        process = subprocess.Popen(cmd)
        
        print("⏳ Waiting for game to load...")
        time.sleep(5)
        
        if process.poll() is None:
            print("✅ Game is running!")
            print("🎮 You can now test manual controls in the game window")
            print("   - Arrow keys: D-pad")
            print("   - Z: A button")
            print("   - X: B button")
            print("   - Enter: Start")
            print("   - Backslash: Select")
            print("\n⏹️  Press any key here to stop the test...")
            
            input()  # Wait for user input
            
            # Stop the emulator
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                
            print("✅ Control test completed")
            return True
        else:
            print(f"❌ Game failed to start. Exit code: {process.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error in control test: {e}")
        return False

if __name__ == "__main__":
    print("🎮 Pokemon RL Bot - Emulator Test")
    print("=" * 40)
    
    # Test basic launch
    launch_ok = test_emulator_launch()
    
    if launch_ok:
        print("\n" + "=" * 40)
        response = input("Do you want to test manual controls? (y/n): ")
        if response.lower() == 'y':
            test_emulator_with_controls()
    
    print("\n🏁 Emulator test completed!")
