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

from rlhf_core.policy import PolicyModel
from rlhf_core.reward import ToyRewardModel
from rlhf_core.ppo import PPOTrainer
from monitor.logger import create_stability_logger


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


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


def log_metrics(writer: SummaryWriter, metrics: dict, step: int, log_file: str):
    """Log metrics to both TensorBoard and JSONL file."""
    # Log to TensorBoard
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(key, value, step)
    
    # Log to JSONL file
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'step': step,
        **metrics
    }
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


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
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"Set random seed: {args.seed}")
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
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
    
    # Setup TensorBoard writer
    writer = SummaryWriter(tb_dir)
    
    # Setup JSONL logging
    jsonl_log_file = os.path.join(log_dir, 'train.jsonl')
    
    # Setup stability monitoring
    stability_logger = create_stability_logger(log_dir, tb_dir)
    
    # Create models
    logger.info("Initializing models...")
    
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
        kl_coef=args.kl_coef
    )
    
    # Create sample prompts
    prompts = create_sample_prompts()
    logger.info(f"Created {len(prompts)} sample prompts")
    
    # Training loop
    logger.info("Starting training loop...")
    start_time = time.time()
    
    total_steps = 0
    
    try:
        for epoch in range(args.epochs):
            logger.info(f"Starting epoch {epoch + 1}/{args.epochs}")
            
            # Train for one epoch
            epoch_metrics = trainer.train_epoch(
                prompts=prompts,
                steps_per_epoch=args.steps_per_epoch,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size
            )
            
            # Log epoch metrics
            for step_metrics in epoch_metrics:
                total_steps += 1
                log_metrics(writer, step_metrics, total_steps, jsonl_log_file)
                
                # Log stability metrics
                if 'rewards' in step_metrics and 'advantages' in step_metrics and 'kl_penalty' in step_metrics:
                    stability_metrics = stability_logger.compute_stability_metrics(
                        policy_model=policy_model,
                        reference_model=reference_model,
                        rewards=step_metrics['rewards'],
                        advantages=step_metrics['advantages'],
                        kl_penalty=step_metrics['kl_penalty'],
                        policy_loss=step_metrics['policy_loss'],
                        clip_fraction=step_metrics.get('clip_fraction', 0.0),
                        learning_rate=step_metrics.get('learning_rate', args.learning_rate)
                    )
                    
                    # Get tokens processed for throughput calculation
                    tokens_processed = step_metrics.get('sequence_length_mean', 0) * step_metrics.get('batch_size', 1)
                    
                    stability_logger.log_metrics(
                        step=total_steps,
                        metrics=stability_metrics,
                        tokens_processed=int(tokens_processed),
                        batch_size=step_metrics.get('batch_size', 1)
                    )
                
                # Save checkpoint periodically
                if total_steps % args.checkpoint_interval == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{total_steps}.pt")
                    trainer.save_checkpoint(checkpoint_path)
                    logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Log epoch summary
            epoch_rewards = [m['reward_mean'] for m in epoch_metrics]
            epoch_kl = [m['kl_mean'] for m in epoch_metrics]
            epoch_loss = [m['total_loss'] for m in epoch_metrics]
            
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
    
    # Close stability logger
    stability_logger.close()
    
    # Create a symlink to the latest run
    latest_link = os.path.join(args.output_dir, "latest")
    if os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(run_dir, latest_link)
    
    print(f"\nTraining completed successfully!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Output directory: {run_dir}")
    print(f"Logs: {jsonl_log_file}")
    print(f"TensorBoard: {tb_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    
    return run_dir


if __name__ == "__main__":
    main()