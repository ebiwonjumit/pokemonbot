#!/usr/bin/env python3
"""
Test bot game control - Verifies the bot can actually control the Pokemon game
"""

import sys
import os
import time
import subprocess
import cv2
import numpy as np
from pathlib import Path

# Add the src directory to Python path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_emulator_control():
    """Test basic emulator control without our complex environment."""
    print("🎮 Testing basic Pokemon game control...")
    
    rom_path = Path("roms/pokemon_leaf_green.gba")
    if not rom_path.exists():
        print("❌ ROM file not found!")
        return False
    
    try:
        # Start the emulator
        print("🚀 Starting Pokemon Leaf Green...")
        process = subprocess.Popen(["vbam", str(rom_path)])
        
        print("⏳ Waiting for game to load (10 seconds)...")
        time.sleep(10)
        
        if process.poll() is None:
            print("✅ Game is running!")
            print("\n🕹️ MANUAL TEST INSTRUCTIONS:")
            print("=" * 50)
            print("1. You should see the Pokemon Leaf Green game window")
            print("2. The game should be at the title screen or intro")
            print("3. Try these controls in the game window:")
            print("   - Arrow keys: Move/Navigate menus")
            print("   - Z key: A button (confirm/interact)")
            print("   - X key: B button (cancel/back)")
            print("   - Enter: Start button (pause menu)")
            print("   - Backslash (\\): Select button")
            print("\n🎯 TEST GOALS:")
            print("- Navigate past the title screen")
            print("- Start a new game or load existing save")
            print("- Move the character around")
            print("- Interact with objects/NPCs")
            
            print("\n⏰ Game will run for 60 seconds for testing...")
            print("⏹️  Press Ctrl+C in this terminal to stop early")
            
            try:
                time.sleep(60)
            except KeyboardInterrupt:
                print("\n⏹️ Test stopped by user")
            
            # Stop the game
            process.terminate()
            try:
                process.wait(timeout=5)
                print("✅ Game stopped successfully")
            except subprocess.TimeoutExpired:
                process.kill()
                print("🔥 Force killed the game process")
            
            # Ask for user feedback
            print("\n📋 TEST RESULTS:")
            print("=" * 30)
            response = input("Did the game respond to your controls? (y/n): ")
            
            if response.lower() == 'y':
                print("✅ SUCCESS: Basic game control is working!")
                return True
            else:
                print("❌ FAILED: Game controls are not responding")
                return False
        else:
            print(f"❌ Game failed to start. Exit code: {process.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        if 'process' in locals() and process.poll() is None:
            process.terminate()
        return False
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

def test_screen_capture():
    """Test if we can capture the game screen."""
    print("\n📷 Testing screen capture capabilities...")
    
    try:
        import pyautogui
        import PIL
        
        # Take a screenshot
        screenshot = pyautogui.screenshot()
        screenshot_array = np.array(screenshot)
        
        print("✅ Screen capture working")
        print(f"   - Screenshot size: {screenshot_array.shape}")
        print("   - We can capture what's on screen")
        
        # Save a test screenshot
        screenshot.save("test_screenshot.png")
        print("   - Saved test_screenshot.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Screen capture failed: {e}")
        return False

def test_automated_input():
    """Test automated input to the game."""
    print("\n🤖 Testing automated game input...")
    
    rom_path = Path("roms/pokemon_leaf_green.gba")
    
    try:
        import pyautogui
        
        # Start the game
        print("🚀 Starting game for automated input test...")
        process = subprocess.Popen(["vbam", str(rom_path)])
        
        print("⏳ Waiting for game to load...")
        time.sleep(8)
        
        if process.poll() is None:
            print("🤖 Sending automated inputs to game...")
            
            # Make sure the game window is focused
            # Note: This is a basic test - in real implementation we'd find the window properly
            
            # Send some basic inputs
            commands = [
                ("z", "A button (confirm)"),
                ("z", "A button again"),
                ("down", "Down arrow"),
                ("z", "A button (select)"),
            ]
            
            for key, description in commands:
                print(f"   Sending: {description}")
                pyautogui.press(key)
                time.sleep(1)
            
            print("✅ Automated inputs sent successfully")
            print("📺 Watch the game window to see if inputs took effect")
            
            time.sleep(5)
            
            # Stop the game
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            
            # Get user feedback
            response = input("Did you see the automated inputs affect the game? (y/n): ")
            if response.lower() == 'y':
                print("🎉 SUCCESS: Automated game control is working!")
                return True
            else:
                print("❌ Automated inputs didn't seem to work")
                return False
        else:
            print("❌ Game didn't start for automated test")
            return False
            
    except Exception as e:
        print(f"❌ Automated input test failed: {e}")
        return False

def main():
    """Run all bot control tests."""
    print("🤖 Pokemon Bot Control Test Suite")
    print("=" * 40)
    print("This will test if our bot can control Pokemon Leaf Green")
    print()
    
    # Test 1: Basic emulator control
    basic_control = test_basic_emulator_control()
    
    # Test 2: Screen capture
    screen_capture = test_screen_capture()
    
    # Test 3: Automated input (only if basic control works)
    automated_input = False
    if basic_control:
        response = input("\nDo you want to test automated input? (y/n): ")
        if response.lower() == 'y':
            automated_input = test_automated_input()
    
    # Results
    print("\n" + "=" * 40)
    print("🏁 TEST RESULTS SUMMARY")
    print("=" * 40)
    print(f"✅ Basic Game Control: {'PASS' if basic_control else 'FAIL'}")
    print(f"✅ Screen Capture:     {'PASS' if screen_capture else 'FAIL'}")
    print(f"✅ Automated Input:    {'PASS' if automated_input else 'SKIP' if not basic_control else 'FAIL'}")
    
    if basic_control and screen_capture:
        print("\n🎉 EXCELLENT! Your bot has the basic requirements to control Pokemon!")
        print("🚀 Next steps:")
        print("   - Set up the full RL environment")
        print("   - Train the PPO agent")
        print("   - Start the web dashboard")
        return True
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
        print("🔧 You may need to:")
        print("   - Adjust VBA-M settings")
        print("   - Check screen permissions on macOS")
        print("   - Verify game window focus")
        return False

if __name__ == "__main__":
    main()
