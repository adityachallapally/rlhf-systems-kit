# TRL Integration Guide

This document provides comprehensive guidance on using the seamless TRL (Transformers Reinforcement Learning) integration with the RLHF Systems Kit. The integration focuses on the most critical integration points for production-ready RLHF training.

## 🎯 Integration Overview

The TRL integration provides four critical capabilities:

1. **🔥 Training Callbacks** - Real-time monitoring during training (80% user value)
2. **🔥 PPO-Specific Monitoring** - Specialized PPO debugging and optimization (70% user value)  
3. **⚡ Checkpoint Analysis** - Model health monitoring (60% user value)
4. **⚡ Reward Model Integration** - Reward model reliability (50% user value)

## 🚀 Quick Start

### Installation

```bash
# Install TRL and dependencies
pip install trl>=0.7.0 peft>=0.5.0 bitsandbytes>=0.41.0

# Install the RLHF Systems Kit
pip install -e .
```

### Basic Usage

```python
from rlhf_core.trl_integration import TRLIntegrationConfig, TRLIntegrationManager

# Configure integration
config = TRLIntegrationConfig(
    model_name="gpt2",
    learning_rate=1e-5,
    batch_size=4,
    enable_profiling=True,
    enable_checkpoint_analysis=True,
    enable_reward_monitoring=True
)

# Initialize manager
integration_manager = TRLIntegrationManager(config)

# Setup TRL trainer
trainer = integration_manager.setup_trl_trainer(
    model_name="gpt2",
    dataset_name="imdb"
)

# Train with comprehensive monitoring
results = integration_manager.train_with_monitoring(
    num_steps=100,
    save_checkpoints=True
)
```

## 🔥 Training Callbacks (Critical)

Training callbacks provide real-time monitoring during training, enabling immediate issue detection and debugging.

### Features

- **Real-time Metrics**: Monitor KL divergence, policy loss, value loss, rewards
- **Anomaly Detection**: Automatic detection of training anomalies with configurable thresholds
- **Memory Monitoring**: Track memory usage and GPU utilization
- **Comprehensive Logging**: JSONL logs for all training events

### Usage

```python
from rlhf_core.trl_integration import PPOMonitoringCallback

# Create callback with custom settings
callback = PPOMonitoringCallback(
    anomaly_threshold=3.0,  # Z-score threshold for anomaly detection
    log_dir="./logs/callbacks",
    enable_detailed_logging=True
)

# The callback is automatically integrated with TRL trainer
# Monitor training in real-time
for step in range(num_steps):
    callback.on_step_begin(step, {})
    # ... training step ...
    logs = trainer.step()
    callback.on_step_end(step, logs)
```

### Anomaly Detection

The callback automatically detects anomalies in:

- **KL Divergence**: Sudden spikes indicating policy collapse
- **Policy Loss**: Unusual loss patterns
- **Value Loss**: Value function instability
- **Rewards**: Reward hacking or model degradation
- **Memory Usage**: Memory leaks or inefficient usage

### Log Format

```json
{
  "event": "step_end",
  "step": 100,
  "timestamp": "2024-01-15T10:30:00",
  "ppo_metrics": {
    "kl_div": 0.12,
    "policy_loss": 0.45,
    "value_loss": 0.23,
    "reward": 2.1,
    "clip_ratio": 0.18
  },
  "anomalies": [
    {
      "metric": "kl_div",
      "value": 0.8,
      "z_score": 3.2,
      "severity": "high"
    }
  ],
  "memory_usage": {
    "rss_mb": 1024.5,
    "gpu_memory_mb": 2048.0
  }
}
```

## 🔥 PPO-Specific Monitoring (Critical)

PPO-specific monitoring provides specialized debugging and optimization for the most common RLHF algorithm.

### Features

- **PPO Metrics Tracking**: Specialized tracking of PPO-specific metrics
- **Ratio Analysis**: Monitor policy/value loss ratios and KL/policy ratios
- **Trend Analysis**: Calculate trends for all metrics (increasing/decreasing/stable)
- **Epoch Statistics**: Comprehensive epoch-level analysis

### Key Metrics

```python
# PPO-specific metrics automatically tracked
ppo_metrics = {
    "kl_div": 0.12,           # KL divergence between policy and reference
    "policy_loss": 0.45,      # Policy gradient loss
    "value_loss": 0.23,       # Value function loss
    "reward": 2.1,            # Average reward
    "clip_ratio": 0.18,       # Clipping ratio
    "entropy": 0.8,           # Policy entropy
    "kl_policy_ratio": 0.27,  # KL/policy loss ratio
    "value_policy_ratio": 0.51 # Value/policy loss ratio
}
```

### Trend Analysis

The system automatically calculates trends for all metrics:

```python
trend_analysis = {
    "kl_div": "stable",      # increasing/decreasing/stable
    "policy_loss": "decreasing",
    "value_loss": "stable",
    "reward": "increasing"
}
```

### Epoch Statistics

Comprehensive statistics for each epoch:

```python
epoch_stats = {
    "kl_div": {
        "mean": 0.12,
        "std": 0.05,
        "min": 0.08,
        "max": 0.18,
        "trend": "stable"
    },
    # ... similar for all metrics
}
```

## ⚡ Checkpoint Analysis (High Priority)

Checkpoint analysis provides model health monitoring to prevent training failures.

### Features

- **Health Score**: Overall model health score (0-1)
- **Weight Analysis**: Analyze weight statistics and detect issues
- **Training State Analysis**: Monitor optimizer state and training metrics
- **Comparison**: Compare with reference checkpoints
- **Automated Recommendations**: Generate recommendations based on analysis

### Usage

```python
from rlhf_core.trl_integration import CheckpointAnalyzer

analyzer = CheckpointAnalyzer(log_dir="./logs/checkpoints")

# Analyze checkpoint
analysis = analyzer.analyze_checkpoint(
    model_path="./checkpoints/step_100.pt",
    step=100,
    reference_checkpoint="./checkpoints/step_50.pt"  # Optional
)

print(f"Health Score: {analysis['health_score']:.2f}")
print(f"Issues: {analysis['issues']}")
print(f"Recommendations: {analysis['recommendations']}")
```

### Health Score Calculation

The health score (0-1) is calculated based on:

- **Weight Statistics**: Mean, std, norm of model weights
- **Training State**: Learning rate, optimizer state
- **Issues Detected**: Number and severity of issues
- **Comparison**: Drift from reference checkpoint

### Issue Detection

Automatically detects:

- **Large Weight Values**: Weights with mean > 10 or std > 5
- **High Weight Norms**: Norms > 100
- **Learning Rate Issues**: Too high (>1e-2) or too low (<1e-8)
- **Model Drift**: Significant changes from reference

### Analysis Output

```json
{
  "step": 100,
  "health_score": 0.85,
  "file_size_mb": 245.6,
  "weight_stats": {
    "layer1.weight": {
      "mean": 0.12,
      "std": 0.45,
      "norm": 23.4
    }
  },
  "issues": [
    "High weight variance in layer2.weight: 5.2"
  ],
  "recommendations": [
    "Consider weight regularization",
    "Check learning rate schedule"
  ]
}
```

## ⚡ Reward Model Integration (High Priority)

Reward model integration provides unique RLDK capabilities for reward model reliability.

### Features

- **Reliability Metrics**: Stability, consistency, variance tracking
- **Anomaly Detection**: Detect anomalous reward patterns
- **Trend Analysis**: Monitor reward trends over time
- **Automated Recommendations**: Generate recommendations for reward model improvement

### Usage

```python
from rlhf_core.trl_integration import RewardModelIntegrator

integrator = RewardModelIntegrator(log_dir="./logs/rewards")

# Monitor reward model
analysis = integrator.monitor_reward_model(
    reward_scores=[2.1, 2.3, 1.9, 2.0, 2.2],
    step=100,
    context={"batch_id": "batch_001"}
)

print(f"Stability: {analysis['reliability_metrics']['stability']:.2f}")
print(f"Consistency: {analysis['reliability_metrics']['consistency']:.2f}")
```

### Reliability Metrics

- **Stability**: Inverse of coefficient of variation (higher is better)
- **Consistency**: Percentage of rewards within expected range
- **Variance**: Variance of recent rewards
- **Mean Reward**: Average reward over time

### Anomaly Detection

Detects anomalous rewards using:

- **Z-Score Analysis**: Rewards with |z-score| > threshold
- **Historical Comparison**: Compare with recent reward history
- **Severity Classification**: High/medium severity anomalies

### Recommendations

Automatically generates recommendations:

- **Low Stability**: "Consider reward model calibration"
- **Low Consistency**: "Review reward model training data"
- **High Anomaly Rate**: "Investigate reward model training"

## 🔧 Configuration

### TRLIntegrationConfig

```python
@dataclass
class TRLIntegrationConfig:
    # Training configuration
    model_name: str = "gpt2"
    learning_rate: float = 1e-5
    batch_size: int = 4
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    ppo_epochs: int = 4
    max_grad_norm: float = 1.0
    
    # PPO specific
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    gamma: float = 1.0
    lam: float = 0.95
    kl_penalty: str = "kl"
    target: float = 6.0
    horizon: float = 10000.0
    init_kl_coef: float = 0.2
    adap_kl_ctrl: bool = True
    
    # Monitoring and logging
    log_with: str = "tensorboard"
    logging_dir: str = "./logs"
    save_freq: int = 100
    eval_freq: int = 50
    project_name: str = "rlhf-trl-integration"
    
    # Advanced monitoring
    enable_profiling: bool = True
    enable_checkpoint_analysis: bool = True
    enable_reward_monitoring: bool = True
    anomaly_detection_threshold: float = 3.0
    
    # Device and optimization
    device: str = "auto"
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
```

## 📊 Monitoring Dashboard

The integration provides comprehensive monitoring through:

### Real-time Metrics

- **Training Progress**: Step count, epoch progress
- **PPO Metrics**: KL divergence, policy/value losses, rewards
- **System Metrics**: Memory usage, GPU utilization
- **Anomaly Alerts**: Real-time anomaly notifications

### Historical Analysis

- **Trend Analysis**: Long-term trend visualization
- **Health Scores**: Checkpoint health over time
- **Reward Reliability**: Reward model stability metrics
- **Performance Profiling**: Detailed timing analysis

### Automated Reporting

- **Training Reports**: Comprehensive training summaries
- **Anomaly Reports**: Detailed anomaly analysis
- **Recommendation Reports**: Actionable improvement suggestions

## 🚨 Troubleshooting

### Common Issues

1. **TRL Import Error**
   ```bash
   pip install trl>=0.7.0
   ```

2. **CUDA Memory Issues**
   ```python
   config = TRLIntegrationConfig(
       batch_size=1,  # Reduce batch size
       fp16=True,     # Enable mixed precision
       device="cpu"   # Use CPU if needed
   )
   ```

3. **Anomaly Detection Too Sensitive**
   ```python
   callback = PPOMonitoringCallback(
       anomaly_threshold=5.0  # Increase threshold
   )
   ```

### Performance Optimization

1. **Enable Profiling**
   ```python
   config = TRLIntegrationConfig(
       enable_profiling=True
   )
   ```

2. **Optimize Logging**
   ```python
   config = TRLIntegrationConfig(
       save_freq=200,  # Reduce checkpoint frequency
       eval_freq=100   # Reduce evaluation frequency
   )
   ```

## 📈 Best Practices

### Training Setup

1. **Start Small**: Begin with small models and datasets
2. **Monitor Closely**: Use real-time monitoring from the start
3. **Save Checkpoints**: Regular checkpointing with health analysis
4. **Track Trends**: Monitor long-term trends, not just individual metrics

### Anomaly Handling

1. **Set Appropriate Thresholds**: Adjust anomaly thresholds based on your use case
2. **Investigate Anomalies**: Don't ignore detected anomalies
3. **Use Recommendations**: Follow automated recommendations
4. **Document Issues**: Keep track of resolved issues

### Performance Monitoring

1. **Memory Monitoring**: Watch for memory leaks
2. **GPU Utilization**: Ensure efficient GPU usage
3. **Training Speed**: Monitor steps per second
4. **Convergence**: Track convergence metrics

## 🔗 Integration with Existing Workflows

### TensorBoard Integration

```python
config = TRLIntegrationConfig(
    log_with="tensorboard",
    logging_dir="./tensorboard_logs"
)
```

### Weights & Biases Integration

```python
config = TRLIntegrationConfig(
    log_with="wandb",
    project_name="my-rlhf-project"
)
```

### Custom Logging

```python
# Extend the base callback class
class CustomCallback(PPOMonitoringCallback):
    def on_step_end(self, step, logs):
        super().on_step_end(step, logs)
        # Add custom logging logic
        self.custom_logger.log(step, logs)
```

## 📚 Examples

See the `examples/` directory for comprehensive examples:

- `trl_integration_example.py`: Complete integration example
- `ppo_monitoring_demo.py`: PPO-specific monitoring demo
- `checkpoint_analysis_demo.py`: Checkpoint analysis demo
- `reward_model_demo.py`: Reward model integration demo

## 🤝 Contributing

Contributions are welcome! Please see the main project README for contribution guidelines.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.