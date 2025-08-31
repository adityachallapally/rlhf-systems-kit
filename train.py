#!/usr/bin/env python3
"""
RLHF Training Script

A minimal, reproducible RLHF training loop that can run on CPU or single GPU in under 2 minutes.
"""

import os
import json
import time
import argparse
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging
from pathlib import Path

from rlhf_core.policy import PolicyModel
from rlhf_core.reward import ToyRewardModel
from rlhf_core.ppo import PPOTrainer
from rlhf_core.profiler import ProfilerManager, stage_timer
from rlhf_core.logging import JSONLLogger, write_sysinfo, create_run_dir, update_latest_symlink


def set_all_seeds(seed: int):
    """Set all random seeds for complete determinism."""
    # Set Python random seed
    random.seed(seed)
    print(f"✅ Set random.seed({seed})")
    
    # Set numpy random seed
    np.random.seed(seed)
    print(f"✅ Set np.random.seed({seed})")
    
    # Set PyTorch random seeds
    torch.manual_seed(seed)
    print(f"✅ Set torch.manual_seed({seed})")
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        print(f"✅ Set torch.cuda.manual_seed_all({seed})")
    
    # Set PyTorch determinism flags
    torch.use_deterministic_algorithms(True, warn_only=True)
    print("✅ Set torch.use_deterministic_algorithms(True)")
    
    torch.backends.cudnn.benchmark = False
    print("✅ Set torch.backends.cudnn.benchmark = False")
    
    torch.backends.cudnn.deterministic = True
    print("✅ Set torch.backends.cudnn.deterministic = True")
    
    # Disable TF32 for deterministic matmul kernels
    torch.backends.cuda.matmul.allow_tf32 = False
    print("✅ Set torch.backends.cuda.matmul.allow_tf32 = False")
    
    torch.backends.cudnn.allow_tf32 = False
    print("✅ Set torch.backends.cudnn.allow_tf32 = False")
    
    # Set CUDA workspace config for deterministic CUBLAS
    if torch.cuda.is_available():
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        print("✅ Set CUBLAS_WORKSPACE_CONFIG = :4096:8")
    
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ Set PYTHONHASHSEED = {seed}")
    
    # Set additional environment variables for determinism
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print("✅ Set CUDA_LAUNCH_BLOCKING = 1")
    
    # Set OMP threads for deterministic CPU operations
    os.environ['OMP_NUM_THREADS'] = '1'
    print("✅ Set OMP_NUM_THREADS = 1")
    
    print(f"\n🎯 All determinism flags set for seed {seed}")


def setup_logging(log_dir: str):
    """Setup logging to both file and console."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup file logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_sample_prompts() -> list:
    """Create sample prompts for training."""
    prompts = [
        "The weather today is",
        "I really enjoyed",
        "This movie was",
        "The food tasted",
        "I feel very",
        "The book was",
        "My day was",
        "The music sounds",
        "I think that",
        "The game was"
    ]
    return prompts


def main():
    parser = argparse.ArgumentParser(description='RLHF Training Script')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=8, help='Steps per epoch')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--max_new_tokens', type=int, default=15, help='Maximum new tokens to generate')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--kl_coef', type=float, default=0.1, help='KL penalty coefficient')
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='Checkpoint every N steps')
    parser.add_argument('--output_dir', type=str, default='runs', help='Output directory')
    parser.add_argument('--run_dir', type=str, help='Specific run directory (auto-generated if not specified)')
    parser.add_argument('--profiler', type=str, choices=['on', 'off'], default='off', 
                       help='Enable PyTorch profiler (on/off)')
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_all_seeds(args.seed)
    print(f"Set random seed: {args.seed}")
    
    # Also set from environment if present (for testing)
    env_seed = os.getenv("SEED")
    if env_seed:
        env_seed = int(env_seed)
        if env_seed != args.seed:
            print(f"Warning: Environment SEED={env_seed} differs from CLI --seed={args.seed}")
            print(f"Using CLI seed: {args.seed}")
    
    # Create run directory
    if args.run_dir:
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = create_run_dir(args.output_dir)
    
    # Create subdirectories
    log_dir = os.path.join(run_dir, "logs")
    tb_dir = os.path.join(run_dir, "tb")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(log_dir)
    logger.info(f"Starting RLHF training run: {run_dir}")
    logger.info(f"Device: {device}, Seed: {args.seed}")
    
    # Write system information
    sysinfo = write_sysinfo(run_dir, args.seed)
    logger.info(f"System info written to {run_dir}/sysinfo.json")
    
    # Setup TensorBoard writer
    writer = SummaryWriter(tb_dir)
    
    # Setup JSONL logging for metrics
    metrics_logger = JSONLLogger(os.path.join(run_dir, 'metrics.jsonl'))
    
    # Setup profiler if enabled
    profiler_enabled = args.profiler == 'on'
    profiler = ProfilerManager(run_dir, enabled=profiler_enabled)
    
    # Create models
    logger.info("Initializing models...")
    with stage_timer("init_models", run_dir):
        # Policy model (will be trained)
        policy_model = PolicyModel(device=device)
        
        # Reference model (frozen, for KL penalty)
        reference_model = PolicyModel(device=device)
        reference_model.set_train_mode(False)
        
        # Reward model
        reward_model = ToyRewardModel(device=device)
        
        logger.info("Models initialized successfully")
    
    # Create PPO trainer
    trainer = PPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        reward_model=reward_model,
        device=device,
        learning_rate=args.learning_rate,
        kl_coef=args.kl_coef,
        seed=args.seed
    )
    
    # Create sample prompts
    prompts = create_sample_prompts()
    logger.info(f"Created {len(prompts)} sample prompts")
    
    # Training loop
    logger.info("Starting training loop...")
    start_time = time.time()
    
    total_steps = 0
    
    try:
        with profiler:
            for epoch in range(args.epochs):
                logger.info(f"Starting epoch {epoch + 1}/{args.epochs}")
                
                # Train for one epoch
                with stage_timer("ppo_update", run_dir):
                    epoch_metrics = trainer.train_epoch(
                        prompts=prompts,
                        steps_per_epoch=args.steps_per_epoch,
                        max_new_tokens=args.max_new_tokens,
                        batch_size=args.batch_size
                    )
                
                # Log epoch metrics
                for step_metrics in epoch_metrics:
                    total_steps += 1
                    
                    # Standardize metrics format
                    step_time = (time.time() - start_time) * 1000  # Convert to milliseconds relative to start
                    
                    # Create standardized metrics entry
                    metrics_entry = {
                        'step': total_steps,
                        'phase': 'ppo_update',
                        'loss': step_metrics.get('total_loss', float('nan')),
                        'reward_mean': step_metrics.get('reward_mean', float('nan')),
                        'reward_var': step_metrics.get('reward_var', float('nan')),
                        'kl': step_metrics.get('kl_mean', float('nan')),
                        'entropy': step_metrics.get('entropy', float('nan')),
                        'clip_frac': step_metrics.get('clip_fraction', float('nan')),
                        'grad_norm': step_metrics.get('grad_norm', float('nan')),
                        'lr': args.learning_rate,
                        'time_ms': step_time,
                        'seed': args.seed,
                        'run_id': os.path.basename(run_dir)
                    }
                    
                    # Log to JSONL
                    metrics_logger.log(metrics_entry)
                    
                    # Log to TensorBoard
                    for key, value in metrics_entry.items():
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            writer.add_scalar(key, value, total_steps)
                    
                    # Save checkpoint periodically
                    if total_steps % args.checkpoint_interval == 0:
                        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{total_steps}.pt")
                        trainer.save_checkpoint(checkpoint_path)
                        logger.info(f"Saved checkpoint: {checkpoint_path}")
                
                # Log epoch summary
                epoch_rewards = [m.get('reward_mean', 0) for m in epoch_metrics if not np.isnan(m.get('reward_mean', 0))]
                epoch_kl = [m.get('kl_mean', 0) for m in epoch_metrics if not np.isnan(m.get('kl_mean', 0))]
                epoch_loss = [m.get('total_loss', 0) for m in epoch_metrics if not np.isnan(m.get('total_loss', 0))]
                
                if epoch_rewards:
                    logger.info(f"Epoch {epoch + 1} complete:")
                    logger.info(f"  Avg Reward: {np.mean(epoch_rewards):.4f}")
                    logger.info(f"  Avg KL: {np.mean(epoch_kl):.4f}")
                    logger.info(f"  Avg Loss: {np.mean(epoch_loss):.4f}")
                
                # Check if we're approaching time limit (2 minutes)
                elapsed_time = time.time() - start_time
                if elapsed_time > 110:  # Stop at 110 seconds to ensure <2 minutes
                    logger.info(f"Approaching time limit ({elapsed_time:.1f}s), stopping early")
                    break
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
    finally:
        # Ensure profiler and metrics logger are properly closed
        if profiler_enabled:
            profiler.__exit__(None, None, None)
        metrics_logger.close()
    
    # Final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_checkpoint.pt")
    trainer.save_checkpoint(final_checkpoint_path)
    logger.info(f"Saved final checkpoint: {final_checkpoint_path}")
    
    # Training summary
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Output directory: {run_dir}")
    
    # Close TensorBoard writer
    writer.close()
    
    # Create a symlink to the latest run
    update_latest_symlink(run_dir, args.output_dir)
    
    print(f"\nTraining completed successfully!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Output directory: {run_dir}")
    print(f"Metrics: {run_dir}/metrics.jsonl")
    print(f"TensorBoard: {tb_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    
    if profiler_enabled:
        print(f"Profiler artifacts:")
        print(f"  - Trace: {run_dir}/trace.json")
        print(f"  - Op stats: {run_dir}/op_stats.csv")
        print(f"  - Stage times: {run_dir}/stage_times.json")
    
    return run_dir


if __name__ == "__main__":
    main()