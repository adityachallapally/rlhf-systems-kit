"""
Simplified anomaly detection hooks for testing without torch dependencies.

This is a simplified version that demonstrates the fixes without requiring torch.
"""

from typing import Dict, List, Optional, Any, Callable
from collections import deque
import warnings


class AnomalyDetectionHook:
    """Hook for detecting training anomalies (simplified version)."""
    
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
        # Filter out the training context parameters from kwargs to avoid conflicts
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['model', 'optimizer', 'loss']}
        self.analyze_training_step(step, step_duration, model, optimizer, loss, **filtered_kwargs)
    
    def analyze_training_step(self, 
                            step: int, 
                            step_duration: float, 
                            model: Any,
                            optimizer: Any,
                            loss: Any,
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
        
        # Check gradient norms (simplified)
        grad_anomalies = self._check_gradient_anomalies_simple(model, step)
        anomalies_detected.extend(grad_anomalies)
        
        # Check loss anomalies (simplified)
        loss_anomalies = self._check_loss_anomalies_simple(loss, step)
        anomalies_detected.extend(loss_anomalies)
        
        # Notify callbacks of any anomalies
        for anomaly_type, anomaly_data in anomalies_detected:
            for callback in self.anomaly_callbacks:
                try:
                    callback(anomaly_type, anomaly_data)
                except Exception as e:
                    warnings.warn(f"Anomaly callback failed: {e}")
    
    def _check_learning_rate_anomalies(self, optimizer: Any, step: int) -> List[tuple]:
        """Check for learning rate anomalies."""
        anomalies = []
        
        # Simulate optimizer parameter groups
        if hasattr(optimizer, 'param_groups'):
            param_groups = optimizer.param_groups
        else:
            # For testing without real optimizer
            param_groups = [{'lr': 0.01}]
        
        for group_idx, param_group in enumerate(param_groups):
            current_lr = param_group.get('lr', 0.01)
            
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
    
    def _check_gradient_anomalies_simple(self, model: Any, step: int) -> List[tuple]:
        """Check for gradient norm anomalies (simplified version)."""
        anomalies = []
        
        # Simplified gradient check - just return empty for testing
        # In real implementation, this would check actual gradient norms
        return anomalies
    
    def _check_loss_anomalies_simple(self, loss: Any, step: int) -> List[tuple]:
        """Check for loss anomalies (simplified version)."""
        anomalies = []
        
        # Simplified loss check - just return empty for testing
        # In real implementation, this would check for NaN, inf, etc.
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
                 model: Optional[Any] = None,
                 optimizer: Optional[Any] = None,
                 loss: Optional[Any] = None,
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
             model: Optional[Any] = None,
             optimizer: Optional[Any] = None,
             loss: Optional[Any] = None,
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