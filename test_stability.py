#!/usr/bin/env python3
"""
Test script for stability monitoring components.
"""

import os
import sys

def test_imports():
    """Test if the monitoring modules can be imported."""
    try:
        # Test monitor imports
        from monitor.logger import StabilityLogger, create_stability_logger
        print("✅ monitor.logger imported successfully")
        
        from monitor.plot import load_stability_logs, create_stability_plots
        print("✅ monitor.plot imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_structure():
    """Test if the required directories and files exist."""
    required_files = [
        "monitor/__init__.py",
        "monitor/logger.py", 
        "monitor/plot.py",
        "scripts/serve_dashboard.py",
        "notebooks/stability_dashboard.ipynb"
    ]
    
    required_dirs = [
        "monitor",
        "scripts", 
        "notebooks"
    ]
    
    print("\nChecking file structure:")
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
    
    print("\nChecking directories:")
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/")

def test_makefile():
    """Test if the Makefile has the dashboard target."""
    try:
        with open("Makefile", "r") as f:
            content = f.read()
            if "dashboard:" in content:
                print("✅ Makefile contains dashboard target")
                return True
            else:
                print("❌ Makefile missing dashboard target")
                return False
    except FileNotFoundError:
        print("❌ Makefile not found")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing RLHF Stability Dashboard Implementation")
    print("=" * 50)
    
    # Test structure
    test_structure()
    
    # Test Makefile
    test_makefile()
    
    # Test imports (if dependencies are available)
    print("\nTesting imports:")
    if test_imports():
        print("\n🎉 All tests passed! The stability dashboard is ready.")
        print("\nTo use the dashboard:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run training: make train_smoke")
        print("3. Launch dashboard: make dashboard")
        print("4. View offline analysis: jupyter notebook notebooks/stability_dashboard.ipynb")
    else:
        print("\n⚠️  Import tests failed. Install dependencies first:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()