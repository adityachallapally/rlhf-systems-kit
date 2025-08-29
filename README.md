# RLHF Systems Kit

This project is an open-source toolkit for **understanding, profiling, and debugging RLHF pipelines** at small scale (GPT-2 / ResNet-18 level). The aim is to **expose internals**, **optimize efficiency**, and **provide dashboards** that engineers can use immediately.

It combines:

* A **thin RLHF runner** (toy PPO loop, GPT-2 policy, toy reward model).
* A **Profiler** (time/memory breakdown, traces, flame graphs).
* A **Training Stability Dashboard** (monitor KL, drift, gradient norms, reward variance).
* A **Memory Optimizer** (per-model memory usage, batch/placement suggestions).
* **Adapters** for TRL and OpenRLHF.
* **CI + Docs** for reproducibility and adoption.

The goal is to ship something that is **practical** for engineers to run today and **credible** enough to showcase deep RL systems knowledge.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd rlhf-systems-kit

# Install dependencies
pip install -r requirements.txt

# Verify installation
make check
```

### Run Smoke Test

```bash
# Quick smoke test (<2 minutes)
make train_smoke

# Or run directly
python train.py --epochs 2 --steps_per_epoch 6 --batch_size 2
```

---

## 📁 Repository Structure

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
└── README.md          # This file
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

## 🎯 Training Scripts

### Main Training Script (`train.py`)

```bash
# Basic training
python train.py

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

### Makefile Targets

```bash
# Quick smoke test (<2 minutes)
make train_smoke

# Quick training (~5 minutes)
make train_quick

# Full training (~15 minutes)
make train_full

# Check installation
make check

# Launch TensorBoard
make tensorboard
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

## 🧪 Testing

### Run Tests

```bash
# Test implementation
python test_implementation.py

# Run demo (no dependencies required)
python demo.py
```

### Test Coverage

- Policy model functionality
- Reward model computation
- PPO training loop
- Reproducibility
- Logging and checkpointing

---

## 🚀 Performance Targets

### Smoke Test (<2 minutes)
- **Epochs**: 2
- **Steps per epoch**: 6
- **Batch size**: 2
- **Max new tokens**: 10
- **Target**: Complete training loop

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

## 🔮 Future Enhancements

### Planned Features

- **Profiler integration**: Time/memory breakdown
- **Stability dashboard**: Real-time monitoring
- **Memory optimization**: Batch size suggestions
- **TRL/OpenRLHF adapters**: Drop-in integration
- **CI/CD pipeline**: Automated testing

### Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes
4. Add tests
5. Submit pull request

---

## 📚 References

- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [RLHF Paper](https://arxiv.org/abs/2203.02155)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Acknowledgments

- Hugging Face for the Transformers library
- OpenAI for the PPO algorithm
- The RLHF research community

---

**Ready to get started? Run `make train_smoke` to see the RLHF training loop in action!** 🚀