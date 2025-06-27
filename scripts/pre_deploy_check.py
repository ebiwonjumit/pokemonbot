#!/usr/bin/env python3
"""
Pre-deployment check for Pokemon RL Bot GCP deployment
Verifies all components are ready for cloud deployment
"""

import os
import sys
from pathlib import Path

def check_files():
    """Check if all required files exist"""
    print("📁 Checking required files...")
    
    required_files = [
        "scripts/deploy-gcp.sh",
        "scripts/deploy.py", 
        "scripts/dashboards/cloud_dashboard.py",
        "scripts/dashboards/local_dashboard.py",
        "src/web/templates/index.html",
        "src/web/static/script.js",
        "src/web/static/style.css",
        "requirements.txt",
        "config.json",
        "roms/pokemon_leaf_green.gba"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"  ✅ {file_path}")
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    
    print("  ✅ All required files present")
    return True

def check_python_imports():
    """Check if required Python modules can be imported"""
    print("\n🐍 Checking Python imports...")
    
    required_modules = [
        ("flask", "Flask web framework"),
        ("flask_socketio", "WebSocket support"),
        ("psutil", "Process management"),
        ("cv2", "OpenCV for image processing"),
        ("numpy", "Numerical computing"),
        ("pathlib", "Path utilities"),
        ("subprocess", "Process management"),
        ("threading", "Threading support"),
        ("base64", "Base64 encoding"),
    ]
    
    failed_imports = []
    for module, description in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module} - {description}")
        except ImportError as e:
            failed_imports.append((module, description, str(e)))
            print(f"  ❌ {module} - {description} (FAILED: {e})")
    
    if failed_imports:
        print(f"\n❌ Failed imports: {len(failed_imports)}")
        return False
    
    print("  ✅ All required modules importable")
    return True

def check_dashboards():
    """Check if both dashboards can be imported"""
    print("\n🖥️  Checking dashboards...")
    
    # Add current directory to path for testing
    import sys
    sys.path.insert(0, '.')
    
    try:
        from scripts.dashboards.local_dashboard import LocalDashboard
        print("  ✅ Local dashboard imports successfully")
    except Exception as e:
        print(f"  ❌ Local dashboard import failed: {e}")
        return False
    
    try:
        from scripts.dashboards.cloud_dashboard import CloudDashboard
        print("  ✅ Cloud dashboard imports successfully")
    except Exception as e:
        print(f"  ❌ Cloud dashboard import failed: {e}")
        return False
    
    return True

def check_gcp_config():
    """Check GCP configuration"""
    print("\n☁️  Checking GCP configuration...")
    
    # Check if gcloud is available
    import subprocess
    try:
        result = subprocess.run(['gcloud', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("  ✅ gcloud CLI available")
        else:
            print("  ⚠️  gcloud CLI not working properly")
            return False
    except FileNotFoundError:
        print("  ❌ gcloud CLI not found - install from https://cloud.google.com/sdk")
        return False
    except subprocess.TimeoutExpired:
        print("  ⚠️  gcloud CLI timeout")
        return False
    
    # Check if deploy script is executable
    deploy_script = Path("scripts/deploy-gcp.sh")
    if deploy_script.exists():
        if os.access(deploy_script, os.X_OK):
            print("  ✅ Deployment script is executable")
        else:
            print("  ⚠️  Deployment script needs execute permission")
            print("    Run: chmod +x scripts/deploy-gcp.sh")
            return False
    
    return True

def main():
    """Run all pre-deployment checks"""
    print("🎮 Pokemon RL Bot - Pre-Deployment Check")
    print("="*50)
    
    checks = [
        ("Files", check_files),
        ("Python Imports", check_python_imports), 
        ("Dashboards", check_dashboards),
        ("GCP Configuration", check_gcp_config)
    ]
    
    all_passed = True
    results = []
    
    for check_name, check_func in checks:
        try:
            passed = check_func()
            results.append((check_name, passed))
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"  ❌ {check_name} check failed with error: {e}")
            results.append((check_name, False))
            all_passed = False
    
    print("\n" + "="*50)
    print("📋 DEPLOYMENT READINESS SUMMARY")
    print("="*50)
    
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {check_name}")
    
    if all_passed:
        print("\n🚀 READY FOR DEPLOYMENT!")
        print("  Run: python scripts/deploy.py deploy")
        return True
    else:
        print("\n⚠️  NOT READY - Fix issues above before deploying")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
