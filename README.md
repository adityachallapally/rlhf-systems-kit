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

### Quick Install

```bash
# Clone the repository
git clone https://github.com/adityachallapally/rlhf-systems-kit.git
cd rlhf-systems-kit

# Install dependencies
pip install -r requirements.txt

# Verify installation
make check
```

### Development Install

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### GPU Support

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 🚀 Quick Start

### Run Smoke Test

```bash
# Quick smoke test (<2 minutes)
make train_smoke

# Or run directly
python train.py --epochs 2 --steps_per_epoch 6 --batch_size 2
```

### Custom Training

```bash
# Quick training (~5 minutes)
make train_quick

# Full training (~15 minutes)
make train_full

# Custom parameters
python train.py \
    --epochs 5 \
    --steps_per_epoch 10 \
    --batch_size 4 \
    --max_new_tokens 15 \
    --learning_rate 1e-5 \
    --kl_coef 0.1 \
    --seed 42
```

---

## 📁 Project Structure

```
rlhf-systems-kit/
├── rlhf_core/           # Core RLHF implementation
│   ├── __init__.py
│   ├── policy.py        # GPT-2 policy wrapper
│   ├── reward.py        # Toy reward models
│   └── ppo.py          # PPO training loop
├── runs/                # Training outputs
│   ├── logs/           # JSONL logs
│   ├── tb/             # TensorBoard files
│   └── checkpoints/    # Model checkpoints
├── train.py            # Main training script
├── demo.py             # Demo without dependencies
├── test_implementation.py # Test script
├── Makefile            # Build targets
├── requirements.txt    # Dependencies
├── setup.py            # Package setup
├── INSTALL.md          # Installation guide
└── README.md           # This file
```

---

## 🔧 Core Components

### 1. Policy Model (`rlhf_core/policy.py`)

A thin wrapper around GPT-2 models that provides:
- Text generation with sampling
- Log probability computation
- KL divergence calculation
- Checkpoint saving/loading

```python
from rlhf_core.policy import PolicyModel

# Create policy model
policy = PolicyModel(model_name="sshleifer/tiny-gpt2", device="cpu")

# Sample text
sequences, logprobs = policy.sample(prompt_ids, max_new_tokens=20)

# Get log probabilities
logprobs = policy.get_logprobs(input_ids)
```

### 2. Reward Model (`rlhf_core/reward.py`)

Lightweight reward models including:
- **Sentiment classifier**: Keyword-based positive/negative scoring
- **Preference net**: Rewards specific phrase endings
- **Length penalty**: Encourages moderate sequence lengths

```python
from rlhf_core.reward import ToyRewardModel

# Create reward model
reward_model = ToyRewardModel(device="cpu")

# Compute rewards
rewards = reward_model.compute_reward(sequences, reward_type="sentiment")
```

### 3. PPO Trainer (`rlhf_core/ppo.py`)

Complete PPO implementation with:
- KL divergence penalty
- Advantage computation
- Gradient clipping
- Checkpointing

```python
from rlhf_core.ppo import PPOTrainer

# Create trainer
trainer = PPOTrainer(
    policy_model=policy,
    reference_model=reference,
    reward_model=reward_model,
    device="cpu"
)

# Training step
metrics = trainer.train_step(prompts, max_new_tokens=20)
```

---

## 📊 Logging & Monitoring

### Log Formats

- **JSONL logs**: `runs/logs/train.jsonl` - Machine-readable metrics
- **TensorBoard**: `runs/tb/` - Interactive visualizations
- **Console output**: Real-time training progress

### Key Metrics

- `total_loss`: Combined PPO + KL loss
- `policy_loss`: PPO policy gradient loss
- `kl_loss`: KL divergence penalty
- `reward_mean/std`: Reward statistics
- `kl_mean/std`: KL divergence statistics
- `clip_fraction`: PPO clipping frequency

### Example Log Entry

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "step": 15,
  "epoch": 2,
  "total_loss": 0.85,
  "policy_loss": 0.65,
  "kl_loss": 0.20,
  "reward_mean": 1.2,
  "kl_mean": 0.15,
  "clip_fraction": 0.1
}
```

---

## 🔍 Reproducibility

### Deterministic Training

```python
# Set random seeds
set_seed(42)

# Ensures identical results across runs
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
```

### Checkpointing

```python
# Save checkpoint
trainer.save_checkpoint("checkpoint.pt")

# Load checkpoint
trainer.load_checkpoint("checkpoint.pt")
```

---

## 🧪 Testing & Verification

### Demo Script
```bash
python demo.py  # Works without external dependencies
```

### Test Script
```bash
python test_implementation.py  # Requires dependencies
```

### Makefile Targets
```bash
make check      # Verify installation
make clean      # Clean generated files
make tensorboard # Launch TensorBoard
```

---

## 🚀 Performance Targets

### Smoke Test (<2 minutes)
- **Epochs**: 2
- **Steps per epoch**: 6
- **Batch size**: 2
- **Max new tokens**: 10
- **Target**: Complete training loop demonstration

### Quick Training (~5 minutes)
- **Epochs**: 5
- **Steps per epoch**: 10
- **Batch size**: 4
- **Max new tokens**: 15

### Full Training (~15 minutes)
- **Epochs**: 10
- **Steps per epoch**: 15
- **Batch size**: 8
- **Max new tokens**: 20

---

## 🔧 Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seed` | 42 | Random seed for reproducibility |
| `--device` | auto | Device (auto/cpu/cuda) |
| `--epochs` | 3 | Number of training epochs |
| `--steps_per_epoch` | 8 | Steps per epoch |
| `--batch_size` | 4 | Batch size |
| `--max_new_tokens` | 15 | Maximum new tokens to generate |
| `--learning_rate` | 1e-5 | Learning rate |
| `--kl_coef` | 0.1 | KL penalty coefficient |

### Model Configuration

- **Policy Model**: `sshleifer/tiny-gpt2` (very small GPT-2)
- **Reference Model**: Same as policy (frozen)
- **Reward Model**: Lightweight sentiment classifier

---

## 📈 Example Output

### Training Progress

```
Starting RLHF training run: runs/run_20240115_103000
Device: cpu, Seed: 42
Models initialized successfully
Created 10 sample prompts

Starting training loop...
Starting epoch 1/3
Epoch 1, Step 0: Loss=0.8500, Reward=1.2000, KL=0.1500
Epoch 1, Step 5: Loss=0.9000, Reward=1.7000, KL=0.1750

Epoch 1 complete:
  Avg Reward: 1.4500
  Avg KL: 0.1625
  Avg Loss: 0.8750
```

### Final Results

```
Training completed successfully!
Total time: 95.23 seconds
Total steps: 24
Output directory: runs/run_20240115_103000
Logs: runs/run_20240115_103000/logs/train.jsonl
TensorBoard: runs/run_20240115_103000/tb
Checkpoints: runs/run_20240115_103000/checkpoints
```

---

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or max_new_tokens
2. **Training too slow**: Reduce epochs or steps_per_epoch
3. **Import errors**: Install dependencies with `pip install -r requirements.txt`

### Debug Mode

```bash
# Run with verbose logging
python train.py --epochs 1 --steps_per_epoch 2 --batch_size 1
```

---

## 🔮 Development Roadmap

### **M1: Runner (✅ COMPLETED)**
- ✅ GPT-2 policy model (tiny)
- ✅ Toy reward model (sentiment classifier)
- ✅ PPO loop with deterministic seed
- ✅ Logs: JSONL + TensorBoard
- ✅ Target: <2 min smoke run on CPU or single GPU

### **M2: Profiler (Next)**
- 🔄 Instrument stages with timers + traces
- 🔄 Use `torch.profiler` to emit timeline + CSV op stats
- 🔄 Optional `nsys` wrapper if installed

### **M3: Stability Dashboard**
- 🔄 Real-time + offline monitoring of RLHF health
- 🔄 Metrics: KL value, KL target error, entropy, reward mean/variance
- 🔄 FastAPI server for live charts

### **M4: Memory Optimizer**
- 🔄 Per-module CUDA memory stats
- 🔄 Tools for memory analysis and config suggestions

### **M5: Adapters**
- 🔄 Enable profiling/monitoring in TRL and OpenRLHF
- 🔄 Drop-in integration with existing frameworks

### **M6: CI + Docs**
- 🔄 Dockerfile + CI workflow
- 🔄 Automated testing and documentation

---

## 🤝 Contributing

### Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests and ensure they pass
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Development Setup

```bash
# Clone and setup
git clone https://github.com/adityachallapally/rlhf-systems-kit.git
cd rlhf-systems-kit
pip install -e ".[dev]"

# Run tests
make all-checks
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for all functions
- Run `make format` before committing

---

## 📚 API Reference

### PolicyModel

```python
class PolicyModel(nn.Module):
    def __init__(self, model_name: str = "sshleifer/tiny-gpt2", device: str = "cpu")
    def forward(self, input_ids, attention_mask=None) -> Dict[str, torch.Tensor]
    def sample(self, prompt_ids, max_new_tokens=20, temperature=1.0) -> Tuple[torch.Tensor, torch.Tensor]
    def get_logprobs(self, input_ids, attention_mask=None) -> torch.Tensor
    def get_kl_divergence(self, input_ids, reference_model) -> torch.Tensor
```

### ToyRewardModel

```python
class ToyRewardModel(nn.Module):
    def __init__(self, model_name: str = "sshleifer/tiny-gpt2", device: str = "cpu")
    def compute_reward(self, input_ids, reward_type="sentiment") -> torch.Tensor
    def get_reward_stats(self, rewards) -> Dict[str, float]
```

### PPOTrainer

```python
class PPOTrainer:
    def __init__(self, policy_model, reference_model, reward_model, device="cpu", **kwargs)
    def train_step(self, prompts, max_new_tokens=20, batch_size=4) -> Dict[str, float]
    def train_epoch(self, prompts, steps_per_epoch=10, **kwargs) -> List[Dict[str, float]]
    def save_checkpoint(self, path: str)
    def load_checkpoint(self, path: str)
```

---

## ❓ FAQ

### Q: Why use such small models?
A: Small models (GPT-2 tiny) ensure fast iteration and debugging while maintaining the core RLHF dynamics.

### Q: How do I scale this to larger models?
A: The architecture is designed to be model-agnostic. Simply change the model_name parameter to use larger models.

### Q: Can I use my own reward function?
A: Yes! Extend the ToyRewardModel class or implement your own reward computation logic.

### Q: How do I monitor training progress?
A: Use the built-in logging (JSONL + TensorBoard) or extend the monitoring system.

### Q: Is this production-ready?
A: This is a research and debugging toolkit. For production, consider using established frameworks like TRL or OpenRLHF.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Acknowledgments

- Hugging Face for the Transformers library
- OpenAI for the PPO algorithm
- The RLHF research community
- Contributors and maintainers

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/adityachallapally/rlhf-systems-kit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/adityachallapally/rlhf-systems-kit/discussions)
- **Documentation**: [Wiki](https://github.com/adityachallapally/rlhf-systems-kit/wiki)

---

**Ready to get started? Run `make train_smoke` to see the RLHF training loop in action!** 🚀
