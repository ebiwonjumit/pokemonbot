#!/usr/bin/env python3
"""
Pokemon RL Bot Test Runner
Runs all tests to verify the setup is working correctly.
"""

import os
import sys
import subprocess
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent

def run_test(test_name, test_script):
    """Run a single test and return the result."""
    print(f"\n{'='*50}")
    print(f"🧪 Running {test_name}")
    print(f"{'='*50}")
    
    try:
        # Change to project root directory
        os.chdir(project_root)
        
        # Run the test script
        result = subprocess.run([
            sys.executable, test_script
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"✅ {test_name} PASSED")
            return True
        else:
            print(f"❌ {test_name} FAILED")
            return False
            
    except Exception as e:
        print(f"❌ {test_name} ERROR: {e}")
        return False

def main():
    """Run all tests."""
    print("🎮 Pokemon RL Bot - Test Suite")
    print("=" * 40)
    
    tests = [
        ("Environment Setup", "scripts/tests/test_setup.py"),
        ("Bot Control", "scripts/tests/test_bot_control.py"),
        ("Quick Emulator", "scripts/tests/quick_emulator_test.py"),
    ]
    
    results = []
    
    for test_name, test_script in tests:
        if Path(test_script).exists():
            results.append(run_test(test_name, test_script))
        else:
            print(f"⚠️ Test script not found: {test_script}")
            results.append(False)
    
    # Summary
    print(f"\n{'='*50}")
    print("🏁 TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{test_name:30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready!")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
