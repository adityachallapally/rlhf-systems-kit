#!/usr/bin/env python3
"""
TRL Integration Example

This example demonstrates the seamless integration with TRL (Transformers Reinforcement Learning)
showing all the critical integration points:

1. Training Callbacks (🔥 Critical) - Real-time monitoring during training
2. PPO-Specific Monitoring (🔥 Critical) - Specialized PPO debugging and optimization  
3. Checkpoint Analysis (⚡ High) - Model health monitoring
4. Reward Model Integration (⚡ High) - Reward model reliability

Usage:
    python examples/trl_integration_example.py --model gpt2 --steps 50 --batch-size 2
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from datetime import datetime

# TRL Integration imports
from rlhf_core.trl_integration import (
    TRLIntegrationConfig,
    TRLIntegrationManager,
    PPOMonitoringCallback,
    CheckpointAnalyzer,
    RewardModelIntegrator
)

# TRL imports
try:
    from trl import PPOTrainer, PPOConfig
    from trl.core import LengthSampler
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
    from datasets import Dataset
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    print("Warning: TRL not available. Install with: pip install trl>=0.7.0")


def create_sample_dataset(num_samples: int = 100) -> Dataset:
    """Create a sample dataset for demonstration."""
    
    # Sample prompts for RLHF training
    prompts = [
        "The weather today is",
        "I think that artificial intelligence",
        "The best way to learn programming is",
        "Climate change is",
        "The future of technology",
        "I believe that education",
        "The most important skill for",
        "When I think about the environment",
        "The role of government should be",
        "Technology has changed"
    ]
    
    # Generate dataset
    data = []
    for i in range(num_samples):
        prompt = prompts[i % len(prompts)]
        data.append({
            "query": prompt,
            "input_ids": [1, 2, 3, 4, 5],  # Simplified token IDs
            "attention_mask": [1, 1, 1, 1, 1]
        })
    
    return Dataset.from_list(data)


def create_reward_model(model_name: str = "gpt2"):
    """Create a simple reward model for demonstration."""
    
    class SimpleRewardModel(torch.nn.Module):
        def __init__(self, base_model_name: str):
            super().__init__()
            self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
            self.reward_head = torch.nn.Linear(self.base_model.config.hidden_size, 1)
        
        def forward(self, input_ids, attention_mask=None):
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # Use the last hidden state of the last token
            last_hidden_states = outputs.hidden_states[-1]
            last_token_hidden = last_hidden_states[:, -1, :]
            reward = self.reward_head(last_token_hidden)
            return reward
    
    return SimpleRewardModel(model_name)


def demonstrate_training_callbacks():
    """Demonstrate training callbacks functionality."""
    print("\n🔥 DEMONSTRATING TRAINING CALLBACKS")
    print("=" * 50)
    
    # Create PPO monitoring callback
    callback = PPOMonitoringCallback(
        anomaly_threshold=2.0,
        log_dir="./demo_logs/callbacks",
        enable_detailed_logging=True
    )
    
    # Simulate training steps
    for step in range(10):
        # Simulate step beginning
        callback.on_step_begin(step, {"step": step})
        
        # Simulate training metrics (with some anomalies)
        logs = {
            "kl_div": np.random.normal(0.1, 0.05),
            "policy_loss": np.random.normal(0.5, 0.1),
            "value_loss": np.random.normal(0.3, 0.05),
            "reward": np.random.normal(2.0, 0.5),
            "clip_ratio": np.random.normal(0.2, 0.05),
            "entropy": np.random.normal(0.8, 0.1)
        }
        
        # Add some anomalies
        if step == 5:
            logs["kl_div"] = 1.0  # High KL divergence
        if step == 8:
            logs["policy_loss"] = 2.0  # High policy loss
        
        # Simulate step end
        callback.on_step_end(step, logs)
    
    # Simulate epoch
    callback.on_epoch_begin(0, {})
    callback.on_epoch_end(0, {})
    
    print("✅ Training callbacks demonstration completed")
    print("📊 Check ./demo_logs/callbacks/ for detailed logs")


def demonstrate_ppo_monitoring():
    """Demonstrate PPO-specific monitoring."""
    print("\n🔥 DEMONSTRATING PPO-SPECIFIC MONITORING")
    print("=" * 50)
    
    # Create PPO monitoring callback with specialized settings
    ppo_callback = PPOMonitoringCallback(
        anomaly_threshold=1.5,
        log_dir="./demo_logs/ppo_monitoring",
        enable_detailed_logging=True
    )
    
    # Simulate PPO training with various scenarios
    scenarios = [
        {"name": "Normal Training", "kl_div": 0.1, "policy_loss": 0.5, "value_loss": 0.3},
        {"name": "High KL Divergence", "kl_div": 0.8, "policy_loss": 0.6, "value_loss": 0.4},
        {"name": "Policy Collapse", "kl_div": 0.05, "policy_loss": 0.1, "value_loss": 0.2},
        {"name": "Value Function Issues", "kl_div": 0.15, "policy_loss": 0.5, "value_loss": 1.0},
        {"name": "Reward Hacking", "kl_div": 0.2, "policy_loss": 0.3, "value_loss": 0.3}
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"📈 Testing scenario: {scenario['name']}")
        
        logs = {
            "kl_div": scenario["kl_div"],
            "policy_loss": scenario["policy_loss"],
            "value_loss": scenario["value_loss"],
            "reward": np.random.normal(2.0, 0.3),
            "clip_ratio": np.random.normal(0.2, 0.02),
            "entropy": np.random.normal(0.8, 0.05)
        }
        
        ppo_callback.on_step_begin(i, logs)
        ppo_callback.on_step_end(i, logs)
    
    print("✅ PPO-specific monitoring demonstration completed")
    print("📊 Check ./demo_logs/ppo_monitoring/ for detailed analysis")


def demonstrate_checkpoint_analysis():
    """Demonstrate checkpoint analysis functionality."""
    print("\n⚡ DEMONSTRATING CHECKPOINT ANALYSIS")
    print("=" * 50)
    
    # Create checkpoint analyzer
    analyzer = CheckpointAnalyzer(log_dir="./demo_logs/checkpoint_analysis")
    
    # Create a dummy checkpoint for demonstration
    dummy_checkpoint = {
        "model": {
            "layer1.weight": torch.randn(100, 50),
            "layer1.bias": torch.randn(50),
            "layer2.weight": torch.randn(50, 10),
            "layer2.bias": torch.randn(10)
        },
        "trainer_state": {
            "global_step": 100,
            "learning_rate": 1e-5,
            "epoch": 5
        }
    }
    
    # Save dummy checkpoint
    checkpoint_path = "./demo_logs/dummy_checkpoint.pt"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(dummy_checkpoint, checkpoint_path)
    
    # Analyze checkpoint
    analysis = analyzer.analyze_checkpoint(checkpoint_path, step=100)
    
    print(f"📊 Checkpoint Health Score: {analysis['health_score']:.2f}")
    print(f"📁 File Size: {analysis['file_size_mb']:.2f} MB")
    
    if analysis['issues']:
        print("⚠️  Issues detected:")
        for issue in analysis['issues']:
            print(f"   - {issue}")
    else:
        print("✅ No issues detected")
    
    if analysis['recommendations']:
        print("💡 Recommendations:")
        for rec in analysis['recommendations']:
            print(f"   - {rec}")
    
    print("✅ Checkpoint analysis demonstration completed")
    print("📊 Check ./demo_logs/checkpoint_analysis/ for detailed logs")


def demonstrate_reward_model_integration():
    """Demonstrate reward model integration."""
    print("\n⚡ DEMONSTRATING REWARD MODEL INTEGRATION")
    print("=" * 50)
    
    # Create reward model integrator
    integrator = RewardModelIntegrator(log_dir="./demo_logs/reward_integration")
    
    # Simulate reward model monitoring over time
    scenarios = [
        {"name": "Normal Rewards", "rewards": [2.1, 2.3, 1.9, 2.0, 2.2]},
        {"name": "High Variance", "rewards": [1.0, 4.0, 0.5, 3.5, 1.5]},
        {"name": "Reward Drift", "rewards": [3.0, 3.2, 3.1, 3.3, 3.0]},
        {"name": "Anomalous Rewards", "rewards": [2.0, 2.1, 10.0, 2.0, 1.9]},  # One outlier
        {"name": "Consistent Rewards", "rewards": [2.0, 2.0, 2.0, 2.0, 2.0]}
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"🎯 Testing scenario: {scenario['name']}")
        
        analysis = integrator.monitor_reward_model(
            reward_scores=scenario["rewards"],
            step=i * 10,
            context={"scenario": scenario["name"]}
        )
        
        reliability = analysis.get("reliability_metrics", {})
        print(f"   Stability: {reliability.get('stability', 'N/A'):.2f}")
        print(f"   Consistency: {reliability.get('consistency', 'N/A'):.2f}")
        
        if analysis["anomalies"]:
            print(f"   ⚠️  {len(analysis['anomalies'])} anomalies detected")
        
        if analysis["recommendations"]:
            print(f"   💡 {len(analysis['recommendations'])} recommendations")
    
    print("✅ Reward model integration demonstration completed")
    print("📊 Check ./demo_logs/reward_integration/ for detailed logs")


def demonstrate_full_trl_integration():
    """Demonstrate full TRL integration with all components."""
    print("\n🚀 DEMONSTRATING FULL TRL INTEGRATION")
    print("=" * 50)
    
    if not TRL_AVAILABLE:
        print("❌ TRL not available. Skipping full integration demo.")
        return
    
    # Configuration
    config = TRLIntegrationConfig(
        model_name="gpt2",
        learning_rate=1e-5,
        batch_size=2,
        mini_batch_size=1,
        ppo_epochs=2,
        enable_profiling=True,
        enable_checkpoint_analysis=True,
        enable_reward_monitoring=True,
        logging_dir="./demo_logs/full_integration",
        save_freq=10,
        eval_freq=5
    )
    
    # Initialize integration manager
    integration_manager = TRLIntegrationManager(config)
    
    print("✅ TRL Integration Manager initialized")
    print(f"📁 Logging directory: {config.logging_dir}")
    
    # Create sample dataset
    dataset = create_sample_dataset(num_samples=20)
    print(f"📊 Created sample dataset with {len(dataset)} samples")
    
    try:
        # Setup TRL trainer (this would normally load a real model)
        print("🔧 Setting up TRL trainer...")
        # Note: In a real scenario, you would call:
        # trainer = integration_manager.setup_trl_trainer("gpt2", dataset)
        print("✅ TRL trainer setup completed (simulated)")
        
        # Simulate training with monitoring
        print("🏃 Starting training with comprehensive monitoring...")
        
        # Simulate training results
        training_results = {
            "total_steps": 20,
            "start_time": datetime.now().isoformat(),
            "checkpoints_saved": [
                {"step": 10, "path": "./demo_logs/full_integration/checkpoint_step_10", "health_score": 0.85},
                {"step": 20, "path": "./demo_logs/full_integration/checkpoint_step_20", "health_score": 0.92}
            ],
            "anomalies_detected": [
                {"step": 5, "issue": "High KL divergence", "details": {"kl_div": 0.8, "threshold": 0.5}},
                {"step": 15, "issue": "Reward anomaly", "details": {"reward": 5.2, "z_score": 3.5}}
            ],
            "final_metrics": {
                "kl_div": {"mean": 0.12, "std": 0.05, "trend": "stable"},
                "policy_loss": {"mean": 0.45, "std": 0.08, "trend": "decreasing"},
                "reward": {"mean": 2.1, "std": 0.3, "trend": "increasing"}
            },
            "end_time": datetime.now().isoformat()
        }
        
        # Generate training report
        report_path = integration_manager.generate_training_report(training_results)
        print(f"📋 Training report generated: {report_path}")
        
        print("✅ Full TRL integration demonstration completed")
        
    except Exception as e:
        print(f"❌ Error in full integration demo: {e}")
        print("This is expected if TRL is not properly installed or configured.")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="TRL Integration Demonstration")
    parser.add_argument("--model", default="gpt2", help="Model name to use")
    parser.add_argument("--steps", type=int, default=50, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--demo", choices=["all", "callbacks", "ppo", "checkpoint", "reward", "full"], 
                       default="all", help="Which demo to run")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🎯 TRL Integration Demonstration")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Steps: {args.steps}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Demo: {args.demo}")
    print("=" * 60)
    
    # Create demo logs directory
    os.makedirs("./demo_logs", exist_ok=True)
    
    # Run demonstrations based on selection
    if args.demo in ["all", "callbacks"]:
        demonstrate_training_callbacks()
    
    if args.demo in ["all", "ppo"]:
        demonstrate_ppo_monitoring()
    
    if args.demo in ["all", "checkpoint"]:
        demonstrate_checkpoint_analysis()
    
    if args.demo in ["all", "reward"]:
        demonstrate_reward_model_integration()
    
    if args.demo in ["all", "full"]:
        demonstrate_full_trl_integration()
    
    print("\n🎉 All demonstrations completed!")
    print("📁 Check the ./demo_logs/ directory for detailed logs and analysis")
    print("\n💡 Key Integration Points Demonstrated:")
    print("   🔥 Training Callbacks - Real-time monitoring during training")
    print("   🔥 PPO-Specific Monitoring - Specialized PPO debugging")
    print("   ⚡ Checkpoint Analysis - Model health monitoring")
    print("   ⚡ Reward Model Integration - Reward model reliability")


if __name__ == "__main__":
    main()