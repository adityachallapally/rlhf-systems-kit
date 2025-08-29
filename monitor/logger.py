"""
Stability monitoring logger for RLHF training.

This module provides functions to log key stability metrics during training,
including KL divergence, policy entropy, reward statistics, and gradient norms.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class StabilityLogger:
    """Logger for RLHF training stability metrics."""
    
    def __init__(self, log_dir: str, tb_dir: str):
        """Initialize the stability logger.
        
        Args:
            log_dir: Directory for JSONL logs
            tb_dir: Directory for TensorBoard logs
        """
        self.log_dir = log_dir
        self.tb_dir = tb_dir
        
        # Ensure directories exist
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(tb_dir, exist_ok=True)
        
        # Setup JSONL logging
        self.stability_log_file = os.path.join(log_dir, 'stability.jsonl')
        
        # Setup TensorBoard writer
        self.writer = SummaryWriter(tb_dir)
        
        # Track throughput
        self.last_step_time = time.time()
        self.last_tokens_processed = 0
        
    def log_metrics(self, 
                   step: int, 
                   metrics: Dict[str, Any],
                   tokens_processed: int = 0,
                   batch_size: int = 1) -> None:
        """Log stability metrics to both JSONL and TensorBoard.
        
        Args:
            step: Training step number
            metrics: Dictionary of metrics to log
            tokens_processed: Number of tokens processed in this step
            batch_size: Batch size for this step
        """
        # Calculate throughput
        current_time = time.time()
        time_delta = current_time - self.last_step_time
        if time_delta > 0:
            tokens_per_second = tokens_processed / time_delta
        else:
            tokens_per_second = 0.0
        
        # Prepare log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'tokens_per_second': tokens_per_second,
            'batch_size': batch_size,
            **metrics
        }
        
        # Log to JSONL file
        with open(self.stability_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Log to TensorBoard
        for key, value in log_entry.items():
            if key in ['timestamp', 'step', 'batch_size']:
                continue
            if isinstance(value, (int, float)) and not np.isnan(value):
                self.writer.add_scalar(f'stability/{key}', value, step)
        
        # Update timing for next step
        self.last_step_time = current_time
        self.last_tokens_processed = tokens_processed
    
    def compute_stability_metrics(self, 
                                policy_model,
                                reference_model,
                                rewards: torch.Tensor,
                                advantages: torch.Tensor,
                                kl_penalty: torch.Tensor,
                                policy_loss: torch.Tensor,
                                clip_fraction: float,
                                learning_rate: float) -> Dict[str, float]:
        """Compute comprehensive stability metrics.
        
        Args:
            policy_model: Current policy model
            reference_model: Reference model for KL calculation
            rewards: Reward tensor from current batch
            advantages: Advantage tensor from current batch
            kl_penalty: KL divergence penalty tensor
            policy_loss: Policy loss value
            clip_fraction: PPO clip fraction
            learning_rate: Current learning rate
            
        Returns:
            Dictionary of stability metrics
        """
        # KL divergence metrics
        kl_mean = kl_penalty.mean().item()
        kl_std = kl_penalty.std().item()
        
        # KL target error (assuming target KL is 0.1, can be made configurable)
        kl_target = 0.1
        kl_target_error = kl_mean - kl_target
        
        # Policy entropy (approximate using logprobs)
        with torch.no_grad():
            # Get a sample of logprobs to estimate entropy
            sample_input = torch.randint(0, 1000, (1, 10)).to(policy_model.device)
            logprobs = policy_model.get_logprobs(sample_input)
            probs = torch.softmax(logprobs, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        
        # Reward statistics
        reward_mean = rewards.mean().item()
        reward_std = rewards.std().item()
        
        # Advantage statistics
        advantage_mean = advantages.mean().item()
        advantage_std = advantages.std().item()
        
        # Gradient norm (global L2 norm of policy gradients)
        grad_norm = 0.0
        if policy_model.model.training:
            total_norm = 0.0
            param_count = 0
            for param in policy_model.get_trainable_params():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            if param_count > 0:
                grad_norm = total_norm ** 0.5
        
        # PPO clip fraction
        ppo_clip_fraction = clip_fraction
        
        return {
            'kl': kl_mean,
            'kl_std': kl_std,
            'kl_target_err': kl_target_error,
            'entropy': entropy,
            'reward_mean': reward_mean,
            'reward_std': reward_std,
            'advantage_mean': advantage_mean,
            'advantage_std': advantage_std,
            'grad_norm': grad_norm,
            'ppo_clip_fraction': ppo_clip_fraction,
            'learning_rate': learning_rate,
            'policy_loss': policy_loss.item() if hasattr(policy_loss, 'item') else policy_loss
        }
    
    def close(self):
        """Close the logger and TensorBoard writer."""
        self.writer.close()


def create_stability_logger(log_dir: str, tb_dir: str) -> StabilityLogger:
    """Create a stability logger instance.
    
    Args:
        log_dir: Directory for JSONL logs
        tb_dir: Directory for TensorBoard logs
        
    Returns:
        StabilityLogger instance
    """
    return StabilityLogger(log_dir, tb_dir)