# RLHF Systems Kit

A comprehensive, open-source toolkit for **understanding, profiling, and debugging RLHF (Reinforcement Learning from Human Feedback) pipelines** at small scale (GPT-2 / ResNet-18 level). This project aims to **expose internals**, **optimize efficiency**, and **provide production-ready dashboards** that ML engineers and researchers can use immediately.

## 🎯 What This Project Solves

RLHF training is notoriously difficult to debug and optimize. This toolkit addresses the core challenges:

- **🔍 Visibility**: Real-time monitoring of training stability, KL divergence, reward variance, and gradient norms
- **⚡ Performance**: Profiling tools to identify bottlenecks in rollout, reward scoring, PPO updates, and evaluation
- **💾 Memory Optimization**: Intelligent suggestions for batch sizes, gradient accumulation, and model placement
- **🔄 Integration**: Drop-in adapters for popular RLHF frameworks (TRL, OpenRLHF)
- **📊 Production Ready**: Live dashboards, automated reporting, and CI/CD integration

## 🚀 Key Features

- **Thin RLHF Runner**: Minimal, reproducible PPO implementation with GPT-2 policy and toy reward models
- **Advanced Profiler**: Comprehensive timing analysis, memory profiling, and flame graph generation
- **Training Stability Dashboard**: Real-time monitoring with automated warning systems
- **Memory Optimizer**: Per-model memory analysis and intelligent configuration suggestions
- **Framework Adapters**: Seamless integration with TRL and OpenRLHF
- **Automated CI/CD**: Reproducible builds, testing, and documentation generation

---

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Development Roadmap](#development-roadmap)
- [FAQ](#faq)

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ RAM (16GB+ recommended)
- 2GB+ GPU VRAM (for GPU training)

### Option 1: Install from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/YOURNAME/rlhf-systems-kit.git
cd rlhf-systems-kit

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install additional dependencies for profiling
pip install torch-tb-profiler nvidia-ml-py3
```

### Option 2: Docker Installation

```bash
# Build the Docker image
docker build -t rlhf-systems-kit .

# Run with GPU support
docker run --gpus all -it -v $(pwd):/workspace rlhf-systems-kit
```

### Option 3: Conda Installation

```bash
# Create conda environment
conda create -n rlhf-systems python=3.9
conda activate rlhf-systems

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install the toolkit
pip install -e .
```

---

## 🚀 Quick Start

### 1. Basic Training Run

```bash
# Run a small PPO smoke test (CPU, ~2 minutes)
make train_smoke

# Run with GPU acceleration
CUDA_VISIBLE_DEVICES=0 make train_smoke
```

### 2. Profiling and Analysis

```bash
# Generate comprehensive profiling report
make profile

# View memory usage analysis
make mem_report

# Generate optimization suggestions
make suggest_config
```

### 3. Live Dashboard

```bash
# Start the monitoring dashboard
make dashboard

# Open http://localhost:8000 in your browser
```

### 4. Complete Pipeline

```bash
# Run everything: training, profiling, analysis, and reporting
make all
```

---

## 🏗️ Project Structure

```
rlhf-systems-kit/
├── rlhf_core/           # Core RLHF implementation
│   ├── __init__.py
│   ├── policy.py        # GPT-2 policy network
│   ├── reward.py        # Toy reward models
│   ├── ppo.py          # PPO training loop
│   ├── sampler.py      # Experience collection
│   └── evaluator.py    # Training evaluation
├── profiler/            # Performance profiling tools
│   ├── __init__.py
│   ├── hooks.py        # PyTorch profiler hooks
│   ├── traces.py       # Trace collection and analysis
│   └── reports.py      # Profiling report generation
├── monitor/             # Training stability monitoring
│   ├── __init__.py
│   ├── callbacks.py    # Training callbacks
│   ├── dashboard.py    # FastAPI dashboard server
│   └── metrics.py      # Stability metrics calculation
├── memopt/              # Memory optimization tools
│   ├── __init__.py
│   ├── analyzer.py     # Memory usage analysis
│   └── optimizer.py    # Configuration optimization
├── adapters/            # Framework integrations
│   ├── __init__.py
│   ├── trl.py         # TRL integration
│   └── openrlhf.py    # OpenRLHF integration
├── tools/               # Command-line utilities
│   ├── run_profile.py  # Profiling runner
│   ├── mem_report.py   # Memory report generator
│   ├── suggest_config.py # Configuration optimizer
│   └── build_report.py # Report builder
├── examples/            # Integration examples
│   ├── trl_integration.py
│   ├── openrlhf_integration.py
│   └── custom_policy.py
├── notebooks/           # Jupyter notebooks
│   ├── stability_dashboard.ipynb
│   ├── profiling_analysis.ipynb
│   └── memory_optimization.ipynb
├── tests/               # Test suite
│   ├── test_policy.py
│   ├── test_ppo.py
│   └── test_profiler.py
├── docs/                # Documentation
│   ├── api.md
│   ├── examples.md
│   └── troubleshooting.md
├── configs/             # Configuration files
│   ├── default.yaml
│   ├── gpu_optimized.yaml
│   └── debug.yaml
├── scripts/             # Utility scripts
│   ├── setup_dev.sh
│   ├── run_benchmarks.sh
│   └── generate_docs.sh
├── requirements.txt      # Python dependencies
├── requirements-dev.txt  # Development dependencies
├── environment.lock      # Locked dependency versions
├── Dockerfile           # Container definition
├── docker-compose.yml   # Multi-service setup
├── Makefile             # Build automation
├── setup.py             # Package configuration
├── pyproject.toml       # Modern Python packaging
├── .github/             # GitHub configuration
│   ├── workflows/
│   │   ├── ci.yml      # Continuous integration
│   │   ├── release.yml # Release automation
│   │   └── docs.yml    # Documentation deployment
│   └── ISSUE_TEMPLATE/ # Issue templates
├── .pre-commit-config.yaml # Code quality hooks
├── README.md            # This file
└── CHANGELOG.md         # Version history
```

---

## 🔧 Core Components

### RLHF Core (`rlhf_core/`)

The foundation of the toolkit, providing a minimal but complete RLHF implementation:

- **Policy Network**: GPT-2 based policy with configurable size and architecture
- **Reward Models**: Sentiment analysis, preference learning, and custom reward functions
- **PPO Loop**: Complete PPO implementation with GAE, KL penalty, and value function
- **Sampler**: Efficient experience collection with configurable batch sizes
- **Evaluator**: Comprehensive training metrics and validation

### Profiler (`profiler/`)

Advanced performance analysis tools:

- **Timing Analysis**: Per-stage breakdown of training steps
- **Memory Profiling**: GPU and CPU memory usage tracking
- **Flame Graphs**: Visual performance bottleneck identification
- **Trace Collection**: Detailed operation-level profiling
- **Report Generation**: Automated profiling summaries

### Monitor (`monitor/`)

Real-time training stability monitoring:

- **Live Dashboard**: FastAPI-based web interface
- **Stability Metrics**: KL divergence, entropy, reward variance, gradient norms
- **Alert System**: Automated warnings for training issues
- **Historical Analysis**: Training trend visualization
- **Export Tools**: Metrics export for external analysis

### Memory Optimizer (`memopt/`)

Intelligent memory management:

- **Usage Analysis**: Per-model memory breakdown
- **Optimization Suggestions**: Batch size, gradient accumulation, model placement
- **Activation Checkpointing**: Memory vs. compute trade-off analysis
- **Multi-GPU Support**: Distributed training optimization

---

## 📖 Usage Examples

### Basic Training

```python
from rlhf_core import PPOTrainer, GPT2Policy, SentimentReward

# Initialize components
policy = GPT2Policy(model_size="tiny", max_length=128)
reward_model = SentimentReward()
trainer = PPOTrainer(
    policy=policy,
    reward_model=reward_model,
    batch_size=32,
    learning_rate=1e-5
)

# Train
trainer.train(
    num_epochs=10,
    steps_per_epoch=100,
    eval_every=50
)
```

### Profiling Integration

```python
from profiler import ProfilerHooks
from monitor import StabilityMonitor

# Add profiling hooks
profiler = ProfilerHooks()
trainer.add_hooks(profiler)

# Add monitoring
monitor = StabilityMonitor()
trainer.add_callbacks(monitor)

# Train with profiling
trainer.train(num_epochs=5)
```

### Framework Integration

```python
from adapters import TRLAdapter

# Wrap TRL trainer with our tools
trl_trainer = TRLAdapter(
    base_trainer=your_trl_trainer,
    enable_profiling=True,
    enable_monitoring=True
)

# Use normally - all our tools are automatically integrated
trl_trainer.train()
```

### Custom Reward Model

```python
from rlhf_core import BaseRewardModel

class CustomReward(BaseRewardModel):
    def __init__(self, target_length=50):
        super().__init__()
        self.target_length = target_length
    
    def compute_reward(self, responses, prompts=None):
        # Custom reward logic
        rewards = []
        for response in responses:
            length = len(response.split())
            # Reward responses close to target length
            reward = 1.0 / (1.0 + abs(length - self.target_length))
            rewards.append(reward)
        return torch.tensor(rewards)
```

---

## 🔌 API Reference

### Core Classes

#### `PPOTrainer`

Main training orchestrator for RLHF.

```python
class PPOTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        reward_model: BaseRewardModel,
        batch_size: int = 32,
        learning_rate: float = 1e-5,
        kl_coef: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01
    ):
        """
        Initialize PPO trainer.
        
        Args:
            policy: Policy network to train
            reward_model: Reward model for feedback
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            kl_coef: KL divergence penalty coefficient
            value_coef: Value function loss coefficient
            entropy_coef: Entropy bonus coefficient
        """
```

#### `GPT2Policy`

GPT-2 based policy network.

```python
class GPT2Policy(BasePolicy):
    def __init__(
        self,
        model_size: str = "tiny",
        max_length: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9
    ):
        """
        Initialize GPT-2 policy.
        
        Args:
            model_size: Model size ('tiny', 'small', 'medium', 'large')
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
```

### Profiling API

#### `ProfilerHooks`

```python
class ProfilerHooks:
    def __init__(self, output_dir: str = "profiles"):
        """
        Initialize profiler hooks.
        
        Args:
            output_dir: Directory to save profiling data
        """
    
    def on_step_begin(self, step: int):
        """Called at the beginning of each training step."""
    
    def on_step_end(self, step: int, metrics: Dict):
        """Called at the end of each training step."""
```

### Monitoring API

#### `StabilityMonitor`

```python
class StabilityMonitor:
    def __init__(
        self,
        warning_thresholds: Dict = None,
        export_metrics: bool = True
    ):
        """
        Initialize stability monitor.
        
        Args:
            warning_thresholds: Custom warning thresholds
            export_metrics: Whether to export metrics to files
        """
```

---

## 🚨 Troubleshooting

### Common Issues

#### CUDA Out of Memory

```bash
# Check GPU memory usage
nvidia-smi

# Reduce batch size
export BATCH_SIZE=16

# Enable gradient accumulation
export GRAD_ACCUM_STEPS=4

# Use memory optimization
make mem_report
make suggest_config
```

#### Training Instability

```bash
# Check stability metrics
make dashboard

# Analyze KL divergence
python -m monitor.analyze_kl --log_dir logs/

# Adjust hyperparameters
export KL_COEF=0.1
export LEARNING_RATE=5e-6
```

#### Profiling Issues

```bash
# Check profiler installation
python -c "import torch.profiler; print('OK')"

# Run with CPU profiling only
export PROFILE_GPU=false

# Check output directory permissions
ls -la profiles/
```

### Performance Tuning

#### Memory Optimization

```bash
# Generate memory report
make mem_report

# Get optimization suggestions
make suggest_config

# Apply suggested config
python -m memopt.apply_config --config profiles/suggested_config.yaml
```

#### Training Speed

```bash
# Profile training loop
make profile

# Analyze bottlenecks
python -m profiler.analyze_traces --trace_dir profiles/

# Optimize data loading
export NUM_WORKERS=4
export PIN_MEMORY=true
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
make train_smoke VERBOSE=1

# Check detailed logs
tail -f logs/training.log
```

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
# Clone and setup
git clone https://github.com/YOURNAME/rlhf-systems-kit.git
cd rlhf-systems-kit

# Install development dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Run tests
make test

# Check code quality
make lint
make format
```

### Contribution Guidelines

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** following our coding standards
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Commit your changes**: `git commit -m 'Add amazing feature'`
7. **Push to your branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive docstrings
- Include unit tests for new features
- Update relevant documentation

### Testing

```bash
# Run all tests
make test

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/profiler/

# Run with coverage
make test-cov

# Run performance benchmarks
make benchmark
```

---

## 🗺️ Development Roadmap

### Phase 1: Foundation (Current)
- [x] Basic PPO implementation
- [x] GPT-2 policy network
- [x] Toy reward models
- [x] Basic profiling hooks

### Phase 2: Core Features (Next 2 months)
- [ ] Advanced memory optimization
- [ ] Multi-GPU training support
- [ ] Enhanced stability monitoring
- [ ] Comprehensive test suite

### Phase 3: Production Ready (3-6 months)
- [ ] Distributed training support
- [ ] Advanced reward modeling
- [ ] Model serving capabilities
- [ ] Enterprise features

### Phase 4: Ecosystem (6+ months)
- [ ] Additional policy architectures
- [ ] More reward model types
- [ ] Integration with more frameworks
- [ ] Community models and datasets

---

## ❓ FAQ

### General Questions

**Q: Is this production-ready?**
A: The toolkit is designed for research and development. While it includes production features like monitoring and profiling, it's primarily intended for understanding and debugging RLHF systems.

**Q: What's the performance overhead?**
A: Profiling adds <5% overhead, monitoring adds <2% overhead. Memory optimization tools have negligible impact.

**Q: Can I use this with my existing RLHF code?**
A: Yes! The adapters allow you to integrate our tools with TRL, OpenRLHF, or custom implementations.

### Technical Questions

**Q: What's the minimum hardware requirement?**
A: 8GB RAM, 2GB GPU VRAM for basic usage. 16GB RAM, 8GB GPU VRAM recommended for full features.

**Q: How do I customize the reward model?**
A: Inherit from `BaseRewardModel` and implement the `compute_reward` method. See examples for details.

**Q: Can I profile only specific parts of training?**
A: Yes, you can selectively enable/disable profiling hooks for different training stages.

### Support Questions

**Q: Where can I get help?**
A: Check the troubleshooting section, open an issue on GitHub, or join our community discussions.

**Q: How do I report bugs?**
A: Use the GitHub issue template with detailed reproduction steps and system information.

**Q: Can I contribute even if I'm new to RLHF?**
A: Absolutely! We welcome contributions at all levels. Start with documentation or simple bug fixes.

---

## 📚 Additional Resources

### Documentation
- [API Reference](docs/api.md)
- [Examples Gallery](docs/examples.md)
- [Troubleshooting Guide](docs/troubleshooting.md)
- [Performance Tuning](docs/performance.md)

### Research Papers
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [RLHF Paper](https://arxiv.org/abs/2203.02155)
- [Training Stability](https://arxiv.org/abs/2303.08713)

### Community
- [GitHub Discussions](https://github.com/YOURNAME/rlhf-systems-kit/discussions)
- [Discord Server](https://discord.gg/rlhf-community)
- [Twitter](https://twitter.com/rlhf_systems)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built on top of PyTorch and the broader ML ecosystem
- Inspired by the RLHF research community
- Thanks to all contributors and users

---

## 📊 Project Status

![CI Status](https://github.com/YOURNAME/rlhf-systems-kit/workflows/CI/badge.svg)
![Code Coverage](https://codecov.io/gh/YOURNAME/rlhf-systems-kit/branch/main/graph/badge.svg)
![PyPI Version](https://badge.fury.io/py/rlhf-systems-kit.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**Last Updated**: December 2024  
**Version**: 0.1.0-alpha  
**Python Support**: 3.8+  
**PyTorch Support**: 2.0+