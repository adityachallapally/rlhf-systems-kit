#!/usr/bin/env python3
"""
TRL Integration Installation Script

This script helps install and configure the TRL integration for the RLHF Systems Kit.
It handles dependency installation, configuration, and verification.
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8+")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    dependencies = [
        "torch>=2.0.0",
        "transformers>=4.30.0", 
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "trl>=0.7.0",
        "peft>=0.5.0",
        "bitsandbytes>=0.41.0",
        "tensorboard>=2.13.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "psutil>=5.9.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "streamlit>=1.28.0",
        "plotly>=5.15.0",
        "pandas>=2.0.0",
        "pytest>=7.0.0"
    ]
    
    for dep in dependencies:
        if not run_command(f"pip install '{dep}'", f"Installing {dep}"):
            return False
    
    return True


def verify_installation():
    """Verify that all components are properly installed."""
    print("\n🔍 Verifying installation...")
    
    required_modules = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("trl", "TRL"),
        ("peft", "PEFT"),
        ("datasets", "Datasets"),
        ("accelerate", "Accelerate"),
        ("tensorboard", "TensorBoard"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas")
    ]
    
    all_verified = True
    
    for module_name, display_name in required_modules:
        try:
            importlib.import_module(module_name)
            print(f"✅ {display_name} is available")
        except ImportError:
            print(f"❌ {display_name} is not available")
            all_verified = False
    
    return all_verified


def test_trl_integration():
    """Test the TRL integration components."""
    print("\n🧪 Testing TRL integration...")
    
    try:
        # Test basic imports
        from rlhf_core.trl_integration import (
            TRLIntegrationConfig,
            PPOMonitoringCallback,
            CheckpointAnalyzer,
            RewardModelIntegrator
        )
        print("✅ TRL integration imports successful")
        
        # Test configuration
        config = TRLIntegrationConfig()
        print("✅ Configuration creation successful")
        
        # Test callback creation
        callback = PPOMonitoringCallback(log_dir="./test_logs")
        print("✅ PPO monitoring callback creation successful")
        
        # Test analyzer creation
        analyzer = CheckpointAnalyzer(log_dir="./test_logs")
        print("✅ Checkpoint analyzer creation successful")
        
        # Test integrator creation
        integrator = RewardModelIntegrator(log_dir="./test_logs")
        print("✅ Reward model integrator creation successful")
        
        # Clean up test logs
        import shutil
        if os.path.exists("./test_logs"):
            shutil.rmtree("./test_logs")
        
        return True
        
    except Exception as e:
        print(f"❌ TRL integration test failed: {e}")
        return False


def create_example_config():
    """Create an example configuration file."""
    print("\n📝 Creating example configuration...")
    
    config_content = '''# TRL Integration Configuration Example
# Copy this file and modify as needed for your use case

from rlhf_core.trl_integration import TRLIntegrationConfig

# Basic configuration
config = TRLIntegrationConfig(
    # Model configuration
    model_name="gpt2",
    learning_rate=1e-5,
    batch_size=4,
    mini_batch_size=1,
    
    # PPO configuration
    ppo_epochs=4,
    cliprange=0.2,
    vf_coef=0.1,
    kl_penalty="kl",
    init_kl_coef=0.2,
    
    # Monitoring configuration
    enable_profiling=True,
    enable_checkpoint_analysis=True,
    enable_reward_monitoring=True,
    anomaly_detection_threshold=3.0,
    
    # Logging configuration
    logging_dir="./logs",
    save_freq=100,
    eval_freq=50,
    project_name="my-rlhf-project"
)

# Advanced configuration for larger models
advanced_config = TRLIntegrationConfig(
    model_name="gpt2-medium",
    learning_rate=5e-6,
    batch_size=2,
    mini_batch_size=1,
    gradient_accumulation_steps=4,
    
    # Enable mixed precision for memory efficiency
    fp16=True,
    
    # More frequent monitoring for debugging
    save_freq=50,
    eval_freq=25,
    
    # Stricter anomaly detection
    anomaly_detection_threshold=2.0,
    
    logging_dir="./advanced_logs",
    project_name="advanced-rlhf-project"
)
'''
    
    config_path = "trl_integration_config_example.py"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Example configuration created: {config_path}")
    return True


def print_next_steps():
    """Print next steps for the user."""
    print("\n🎉 TRL Integration Installation Complete!")
    print("=" * 60)
    print("\n📚 Next Steps:")
    print("1. Review the example configuration: trl_integration_config_example.py")
    print("2. Run the integration example:")
    print("   python examples/trl_integration_example.py")
    print("3. Read the documentation:")
    print("   docs/TRL_INTEGRATION.md")
    print("4. Run the tests:")
    print("   python tests/test_trl_integration.py")
    print("\n🔗 Key Integration Points:")
    print("   🔥 Training Callbacks - Real-time monitoring during training")
    print("   🔥 PPO-Specific Monitoring - Specialized PPO debugging")
    print("   ⚡ Checkpoint Analysis - Model health monitoring")
    print("   ⚡ Reward Model Integration - Reward model reliability")
    print("\n💡 Quick Start:")
    print("   from rlhf_core.trl_integration import TRLIntegrationManager")
    print("   manager = TRLIntegrationManager(TRLIntegrationConfig())")
    print("   trainer = manager.setup_trl_trainer('gpt2', 'your_dataset')")
    print("   results = manager.train_with_monitoring(num_steps=100)")


def main():
    """Main installation function."""
    print("🚀 TRL Integration Installation Script")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Dependency installation failed. Please check the errors above.")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Installation verification failed. Some dependencies are missing.")
        sys.exit(1)
    
    # Test TRL integration
    if not test_trl_integration():
        print("\n❌ TRL integration test failed. Please check the installation.")
        sys.exit(1)
    
    # Create example configuration
    create_example_config()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()