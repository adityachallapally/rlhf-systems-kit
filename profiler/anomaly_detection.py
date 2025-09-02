"""
Anomaly detection hooks for RLHF training.

Provides hooks for detecting training anomalies including learning rate changes,
gradient norms, and other training stability issues.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from collections import deque
import warnings


class AnomalyDetectionHook:
    """Hook for detecting training anomalies."""
    
    def __init__(self, 
                 lr_change_threshold: float = 0.1,
                 lr_history_size: int = 10,
                 grad_norm_threshold: float = 10.0,
                 epsilon: float = 1e-8):
        """
        Initialize anomaly detection hook.
        
        Args:
            lr_change_threshold: Threshold for learning rate change detection (0.1 = 10%)
            lr_history_size: Number of recent learning rates to track
            grad_norm_threshold: Threshold for gradient norm anomaly detection
            epsilon: Small value to prevent division by zero
        """
        self.lr_change_threshold = lr_change_threshold
        self.lr_history_size = lr_history_size
        self.grad_norm_threshold = grad_norm_threshold
        self.epsilon = epsilon
        
        # Learning rate history for each parameter group
        self.lr_history: Dict[int, deque] = {}
        
        # Anomaly callbacks
        self.anomaly_callbacks: List[Callable] = []
        
    def register_anomaly_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Register a callback to be called when an anomaly is detected.
        
        Args:
            callback: Function that takes (anomaly_type, anomaly_data) as arguments
        """
        self.anomaly_callbacks.append(callback)
    
    def after_step(self, step: int, step_duration: float, **kwargs):
        """
        Called after each training step to analyze for anomalies.
        
        Args:
            step: Current training step number
            step_duration: Duration of the step in seconds
            **kwargs: Additional context including model, optimizer, loss, etc.
        """
        # Only analyze if we have the required training context
        model = kwargs.get('model')
        optimizer = kwargs.get('optimizer')
        loss = kwargs.get('loss')
        
        if model is None or optimizer is None or loss is None:
            # If we don't have the required context, we can't perform analysis
            # This is the issue mentioned in the bug report - the hook needs
            # the training context to be passed from the profiler
            return
        
        # Analyze training step for anomalies
        self.analyze_training_step(step, step_duration, model, optimizer, loss, **kwargs)
    
    def analyze_training_step(self, 
                            step: int, 
                            step_duration: float, 
                            model: torch.nn.Module,
                            optimizer: torch.optim.Optimizer,
                            loss: torch.Tensor,
                            **kwargs):
        """
        Analyze the training step for anomalies.
        
        Args:
            step: Current training step number
            step_duration: Duration of the step in seconds
            model: The model being trained
            optimizer: The optimizer
            loss: The loss tensor
            **kwargs: Additional context
        """
        anomalies_detected = []
        
        # Check learning rate changes
        lr_anomalies = self._check_learning_rate_anomalies(optimizer, step)
        anomalies_detected.extend(lr_anomalies)
        
        # Check gradient norms
        grad_anomalies = self._check_gradient_anomalies(model, step)
        anomalies_detected.extend(grad_anomalies)
        
        # Check loss anomalies
        loss_anomalies = self._check_loss_anomalies(loss, step)
        anomalies_detected.extend(loss_anomalies)
        
        # Notify callbacks of any anomalies
        for anomaly_type, anomaly_data in anomalies_detected:
            for callback in self.anomaly_callbacks:
                try:
                    callback(anomaly_type, anomaly_data)
                except Exception as e:
                    warnings.warn(f"Anomaly callback failed: {e}")
    
    def _check_learning_rate_anomalies(self, optimizer: torch.optim.Optimizer, step: int) -> List[tuple]:
        """Check for learning rate anomalies."""
        anomalies = []
        
        for group_idx, param_group in enumerate(optimizer.param_groups):
            current_lr = param_group['lr']
            
            # Initialize history for this group if needed
            if group_idx not in self.lr_history:
                self.lr_history[group_idx] = deque(maxlen=self.lr_history_size)
            
            # Add current LR to history
            self.lr_history[group_idx].append(current_lr)
            recent_lrs = list(self.lr_history[group_idx])
            
            # Check for significant LR changes (only if we have at least 2 values)
            if len(recent_lrs) >= 2:
                for i in range(1, len(recent_lrs)):
                    prev_lr = recent_lrs[i-1]
                    curr_lr = recent_lrs[i]
                    
                    # FIXED: Guard against division by zero when LR reaches 0
                    if abs(prev_lr) < self.epsilon:
                        # Skip zero values to avoid division by zero
                        continue
                    
                    # Calculate change ratio
                    change_ratio = abs(curr_lr - prev_lr) / prev_lr
                    
                    if change_ratio > self.lr_change_threshold:
                        anomaly_data = {
                            'step': step,
                            'group_idx': group_idx,
                            'prev_lr': prev_lr,
                            'curr_lr': curr_lr,
                            'change_ratio': change_ratio,
                            'threshold': self.lr_change_threshold
                        }
                        anomalies.append(('lr_change', anomaly_data))
        
        return anomalies
    
    def _check_gradient_anomalies(self, model: torch.nn.Module, step: int) -> List[tuple]:
        """Check for gradient norm anomalies."""
        anomalies = []
        
        try:
            # Calculate total gradient norm
            total_norm = 0.0
            param_count = 0
            
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                
                if total_norm > self.grad_norm_threshold:
                    anomaly_data = {
                        'step': step,
                        'grad_norm': total_norm,
                        'threshold': self.grad_norm_threshold,
                        'param_count': param_count
                    }
                    anomalies.append(('grad_norm', anomaly_data))
        
        except Exception as e:
            warnings.warn(f"Failed to check gradient norms: {e}")
        
        return anomalies
    
    def _check_loss_anomalies(self, loss: torch.Tensor, step: int) -> List[tuple]:
        """Check for loss anomalies."""
        anomalies = []
        
        try:
            loss_value = loss.item()
            
            # Check for NaN or infinite loss
            if not torch.isfinite(loss):
                anomaly_data = {
                    'step': step,
                    'loss_value': loss_value,
                    'anomaly_type': 'nan' if torch.isnan(loss) else 'inf'
                }
                anomalies.append(('loss_nan_inf', anomaly_data))
            
            # Check for extremely large loss values
            elif loss_value > 1000.0:  # Arbitrary threshold
                anomaly_data = {
                    'step': step,
                    'loss_value': loss_value,
                    'threshold': 1000.0
                }
                anomalies.append(('loss_large', anomaly_data))
        
        except Exception as e:
            warnings.warn(f"Failed to check loss anomalies: {e}")
        
        return anomalies


class StepProfiler:
    """Profiler that tracks training steps and calls registered hooks."""
    
    def __init__(self):
        self.hooks: List[AnomalyDetectionHook] = []
        self.step_count = 0
    
    def register_hook(self, hook: AnomalyDetectionHook):
        """Register an anomaly detection hook."""
        self.hooks.append(hook)
    
    def end_step(self, 
                 step: int, 
                 step_duration: float,
                 model: Optional[torch.nn.Module] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 loss: Optional[torch.Tensor] = None,
                 **kwargs):
        """
        Called at the end of each training step.
        
        Args:
            step: Current training step number
            step_duration: Duration of the step in seconds
            model: The model being trained (required for anomaly detection)
            optimizer: The optimizer (required for anomaly detection)
            loss: The loss tensor (required for anomaly detection)
            **kwargs: Additional context to pass to hooks
        """
        self.step_count = step
        
        # Call all registered hooks with the training context
        for hook in self.hooks:
            try:
                hook.after_step(
                    step=step,
                    step_duration=step_duration,
                    model=model,
                    optimizer=optimizer,
                    loss=loss,
                    **kwargs
                )
            except Exception as e:
                warnings.warn(f"Hook {hook.__class__.__name__} failed: {e}")


# Global step profiler instance
_global_step_profiler = StepProfiler()


def get_step_profiler() -> StepProfiler:
    """Get the global step profiler instance."""
    return _global_step_profiler


def register_anomaly_hook(hook: AnomalyDetectionHook):
    """Register an anomaly detection hook with the global profiler."""
    _global_step_profiler.register_hook(hook)


def end_step(step: int, 
             step_duration: float,
             model: Optional[torch.nn.Module] = None,
             optimizer: Optional[torch.optim.Optimizer] = None,
             loss: Optional[torch.Tensor] = None,
             **kwargs):
    """
    Convenience function to call the global step profiler's end_step method.
    
    This is the function that should be called from training loops to ensure
    that anomaly detection hooks receive the proper training context.
    """
    _global_step_profiler.end_step(
        step=step,
        step_duration=step_duration,
        model=model,
        optimizer=optimizer,
        loss=loss,
        **kwargs
    )