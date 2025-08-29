# Installation Guide

This guide will help you install and set up the RLHF Systems Kit on your system.

## Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: At least 4GB RAM (8GB recommended)
- **Storage**: At least 2GB free space
- **GPU**: Optional, but recommended for faster training

## Installation Methods

### Method 1: Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/rlhf-systems-kit.git
cd rlhf-systems-kit

# Install dependencies
pip install -r requirements.txt

# Verify installation
make check
```

### Method 2: Development Install

```bash
# Clone the repository
git clone https://github.com/yourusername/rlhf-systems-kit.git
cd rlhf-systems-kit

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Verify installation
make check
```

### Method 3: Using Conda

```bash
# Create a new conda environment
conda create -n rlhf python=3.9
conda activate rlhf

# Clone the repository
git clone https://github.com/yourusername/rlhf-systems-kit.git
cd rlhf-systems-kit

# Install PyTorch (CPU version)
conda install pytorch cpuonly -c pytorch

# Install other dependencies
pip install -r requirements.txt

# Verify installation
make check
```

## GPU Support

### CUDA Installation

If you have an NVIDIA GPU and want to use CUDA acceleration:

1. **Install NVIDIA drivers** (if not already installed)
2. **Install CUDA Toolkit** (version 11.8 or 12.1 recommended)
3. **Install PyTorch with CUDA support**:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Verify GPU Support

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'torch'`

**Solution**: Install PyTorch first:
```bash
pip install torch torchvision torchaudio
```

#### 2. CUDA Issues

**Problem**: `CUDA out of memory`

**Solution**: Reduce batch size or use CPU:
```bash
python train.py --batch_size 2 --device cpu
```

#### 3. Permission Errors

**Problem**: `Permission denied` when installing packages

**Solution**: Use `--user` flag or virtual environment:
```bash
pip install --user -r requirements.txt
# OR
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 4. Version Conflicts

**Problem**: Package version conflicts

**Solution**: Create a fresh virtual environment:
```bash
python -m venv fresh_env
source fresh_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### System-Specific Issues

#### Linux

- **Missing libraries**: Install system dependencies:
  ```bash
  sudo apt-get update
  sudo apt-get install python3-dev python3-pip python3-venv
  ```

#### macOS

- **M1/M2 Macs**: Use the ARM64 version of PyTorch:
  ```bash
  pip install torch torchvision torchaudio
  ```

#### Windows

- **Visual Studio**: Install Visual Studio Build Tools
- **Path issues**: Ensure Python and pip are in your PATH

## Verification

After installation, verify everything works:

```bash
# Check basic functionality
python demo.py

# Run tests (if pytest is installed)
python test_implementation.py

# Check installation
make check
```

## Next Steps

Once installation is complete:

1. **Run the smoke test**:
   ```bash
   make train_smoke
   ```

2. **Explore the code**:
   - `rlhf_core/policy.py` - Policy model implementation
   - `rlhf_core/reward.py` - Reward model implementation
   - `rlhf_core/ppo.py` - PPO training loop
   - `train.py` - Main training script

3. **Customize training**:
   ```bash
   python train.py --epochs 5 --batch_size 8 --learning_rate 2e-5
   ```

## Getting Help

If you encounter issues:

1. Check the [troubleshooting section](#troubleshooting)
2. Search existing [GitHub issues](https://github.com/yourusername/rlhf-systems-kit/issues)
3. Create a new issue with:
   - Your operating system and Python version
   - Complete error message
   - Steps to reproduce the issue

## Contributing

Want to contribute? See our [contributing guide](CONTRIBUTING.md) for details on:
- Setting up a development environment
- Running tests
- Submitting pull requests
- Code style guidelines