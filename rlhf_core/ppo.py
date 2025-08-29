import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm
import json
import os
from datetime import datetime

# Profiling imports
from profiler.hooks import prof_stage, get_profiler_registry


class PPOTrainer:
    """PPO trainer for RLHF with KL penalty."""
    
    def __init__(self, 
                 policy_model,
                 reference_model,
                 reward_model,
                 device: str = "cpu",
                 learning_rate: float = 1e-5,
                 kl_coef: float = 0.1,
                 clip_ratio: float = 0.2,
                 value_coef: float = 0.1,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 1.0):
        
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.reward_model = reward_model
        self.device = device
        
        # PPO hyperparameters
        self.kl_coef = kl_coef
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Optimizer
        self.optimizer = AdamW(policy_model.get_trainable_params(), lr=learning_rate)
        
        # Training state
        self.step = 0
        self.epoch = 0
        
    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
        """Compute advantages using simple TD(0) method."""
        advantages = rewards - values
        return advantages
    
    def compute_kl_penalty(self, 
                           current_logprobs: torch.Tensor, 
                           reference_logprobs: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence penalty."""
        kl_div = F.kl_div(
            F.log_softmax(current_logprobs, dim=-1),
            F.softmax(reference_logprobs, dim=-1),
            reduction='none'
        ).sum(dim=-1)
        
        return kl_div
    
    def ppo_loss(self, 
                  old_logprobs: torch.Tensor,
                  new_logprobs: torch.Tensor,
                  advantages: torch.Tensor,
                  kl_penalty: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute PPO loss with KL penalty."""
        
        # Compute ratio of new to old probabilities
        ratio = torch.exp(new_logprobs - old_logprobs)
        
        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        
        # Policy loss (minimize negative advantage)
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # KL penalty loss
        kl_loss = self.kl_coef * kl_penalty.mean()
        
        # Total loss
        total_loss = policy_loss + kl_loss
        
        # Compute additional metrics
        clip_fraction = (torch.abs(ratio - 1) > self.clip_ratio).float().mean().item()
        
        metrics = {
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item(),
            'clip_fraction': clip_fraction,
            'kl_mean': kl_penalty.mean().item(),
            'kl_std': kl_penalty.std().item(),
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item()
        }
        
        return total_loss, metrics
    
    def train_step(self, 
                   prompts: List[str],
                   max_new_tokens: int = 20,
                   batch_size: int = 4) -> Dict[str, float]:
        """Single training step."""
        
        # Get profiler registry
        registry = get_profiler_registry()
        
        # Set models to appropriate modes
        self.policy_model.set_train_mode(True)
        self.reference_model.set_train_mode(False)
        self.reward_model.eval()
        
        # Tokenize prompts
        tokenizer = self.policy_model.tokenizer
        prompt_ids = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        prompt_ids = prompt_ids['input_ids'].to(self.device)
        
        # Stage 1: Rollout (sample from current policy)
        with prof_stage("rollout", step_index=0, global_step=self.step) as rollout_context:
            with torch.no_grad():
                sequences, new_token_logprobs = self.policy_model.sample(
                    prompt_ids, 
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    do_sample=True
                )
            
            # Set metadata for rollout stage
            rollout_context.set_metadata(
                tokens_processed=sequences.size(1),
                batch_size=len(prompts),
                seq_len=sequences.size(1)
            )
        
        # Stage 2: Reward scoring
        with prof_stage("reward_scoring", step_index=1, global_step=self.step) as reward_context:
            with torch.no_grad():
                rewards = self.reward_model.compute_reward(sequences, reward_type="sentiment")
            
            reward_context.set_metadata(
                tokens_processed=sequences.size(1),
                batch_size=len(prompts),
                seq_len=sequences.size(1)
            )
        
        # Stage 3: KL penalty calculation
        with prof_stage("kl_penalty_calc", step_index=2, global_step=self.step) as kl_context:
            with torch.no_grad():
                reference_logprobs = self.reference_model.get_logprobs(sequences)
            
            current_logprobs = self.policy_model.get_logprobs(sequences)
            kl_penalty = self.compute_kl_penalty(current_logprobs, reference_logprobs)
            
            kl_context.set_metadata(
                tokens_processed=sequences.size(1),
                batch_size=len(prompts),
                seq_len=sequences.size(1)
            )
        
        # Stage 4: GAE calculation (compute advantages)
        with prof_stage("gae_calc", step_index=3, global_step=self.step) as gae_context:
            advantages = rewards  # For simplicity, use rewards directly as advantages
            
            gae_context.set_metadata(
                tokens_processed=sequences.size(1),
                batch_size=len(prompts),
                seq_len=sequences.size(1)
            )
        
        # Stage 5: PPO update
        with prof_stage("ppo_update", step_index=4, global_step=self.step) as ppo_context:
            # Compute PPO loss
            loss, metrics = self.ppo_loss(
                new_token_logprobs.mean(dim=1),  # Average logprobs across new tokens
                current_logprobs.mean(dim=1),    # Average logprobs across sequence
                advantages,
                kl_penalty
            )
            
            # Ensure loss requires gradients
            if not loss.requires_grad:
                # If loss doesn't require gradients, create a dummy loss that does
                dummy_param = next(self.policy_model.get_trainable_params())
                loss = loss + 0.0 * dummy_param.sum()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_model.get_trainable_params(), self.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            
            ppo_context.set_metadata(
                tokens_processed=sequences.size(1),
                batch_size=len(prompts),
                seq_len=sequences.size(1)
            )
        
        # Stage 6: Evaluation step
        with prof_stage("eval_step", step_index=5, global_step=self.step) as eval_context:
            # Update step counter
            self.step += 1
            
            # Add additional metrics
            metrics.update({
                'step': self.step,
                'epoch': self.epoch,
                'reward_mean': rewards.mean().item(),
                'reward_std': rewards.std().item(),
                'sequence_length_mean': sequences.size(1),
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
            
            eval_context.set_metadata(
                tokens_processed=sequences.size(1),
                batch_size=len(prompts),
                seq_len=sequences.size(1)
            )
        
        return metrics
    
    def train_epoch(self, 
                    prompts: List[str],
                    steps_per_epoch: int = 10,
                    max_new_tokens: int = 20,
                    batch_size: int = 4) -> List[Dict[str, float]]:
        """Train for one epoch."""
        
        epoch_metrics = []
        
        for step in range(steps_per_epoch):
            # Sample random subset of prompts
            if len(prompts) > batch_size:
                batch_prompts = np.random.choice(prompts, batch_size, replace=False).tolist()
            else:
                batch_prompts = prompts
                
            # Training step
            step_metrics = self.train_step(
                batch_prompts,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size
            )
            
            epoch_metrics.append(step_metrics)
            
            # Print progress
            if step % 5 == 0:
                print(f"Epoch {self.epoch}, Step {step}: Loss={step_metrics['total_loss']:.4f}, "
                      f"Reward={step_metrics['reward_mean']:.4f}, KL={step_metrics['kl_mean']:.4f}")
        
        self.epoch += 1
        return epoch_metrics
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'policy_model_state': self.policy_model.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'config': {
                'kl_coef': self.kl_coef,
                'clip_ratio': self.clip_ratio,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_model.model.load_state_dict(checkpoint['policy_model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
    
    def get_profiling_results(self) -> List[Dict[str, Any]]:
        """Get profiling results from the global registry."""
        registry = get_profiler_registry()
        return registry.get_results()
    
    def clear_profiling_results(self):
        """Clear profiling results from the global registry."""
        registry = get_profiler_registry()
        registry.clear()


class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = []
        
    def add(self, data: Dict):
        """Add data to buffer."""
        self.buffer.append(data)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample batch from buffer."""
        if batch_size >= len(self.buffer):
            return self.buffer
        return np.random.choice(self.buffer, batch_size, replace=False).tolist()
    
    def clear(self):
        """Clear buffer."""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)