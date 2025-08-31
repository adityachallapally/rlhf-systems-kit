"""
Profiling hooks for RLHF training stages.

Provides context managers and decorators to instrument training stages with
timing and memory measurements.
"""

import time
import psutil
import torch
import threading
from contextlib import contextmanager
from typing import Dict, Optional, Any, Callable
from functools import wraps
import os


class ProfilerContext:
    """Context manager for profiling a training stage."""
    
    def __init__(self, stage_name: str, step_index: int = 0, global_step: int = 0):
        self.stage_name = stage_name
        self.step_index = step_index
        self.global_step = global_step
        self.start_time = None
        self.start_cpu_mem = None
        self.start_cuda_mem = None
        self.start_cuda_peak = None
        self.end_cuda_peak = None
        
        # Get process for memory tracking
        self.process = psutil.Process()
        
        # Thread-local storage for CUDA memory tracking
        self._thread_local = threading.local()
    
    def __enter__(self):
        # Record start time
        self.start_time = time.time()
        
        # Record start CPU memory (RSS)
        try:
            self.start_cpu_mem = self.process.memory_info().rss / 1024 / 1024  # MB
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self.start_cpu_mem = 0.0
        
        # Record start CUDA memory
        if torch.cuda.is_available():
            try:
                self.start_cuda_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                self.start_cuda_peak = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            except:
                self.start_cuda_mem = 0.0
                self.start_cuda_peak = 0.0
        else:
            self.start_cuda_mem = 0.0
            self.start_cuda_peak = 0.0
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Record end time
        end_time = time.time()
        wall_time = end_time - self.start_time  # Keep in seconds
        
        # Record end CPU memory
        try:
            end_cpu_mem = self.process.memory_info().rss / 1024 / 1024  # MB
            cpu_mem_delta = end_cpu_mem - self.start_cpu_mem
        except (psutil.NoProcess, psutil.AccessDenied):
            cpu_mem_delta = 0.0
        
        # Record end CUDA memory
        if torch.cuda.is_available():
            try:
                end_cuda_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                self.end_cuda_peak = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                cuda_mem_current = end_cuda_mem
                cuda_mem_peak = self.end_cuda_peak
            except:
                cuda_mem_current = 0.0
                cuda_mem_peak = 0.0
        else:
            cuda_mem_current = 0.0
            cuda_mem_peak = 0.0
        
        # Create result dictionary
        result = {
            'stage': self.stage_name,
            'wall_time': wall_time,  # Changed from wall_time_ms to wall_time (seconds)
            'cpu_mem_mb': cpu_mem_delta,
            'cuda_mem_mb_current': cuda_mem_current,
            'cuda_mem_mb_peak': cuda_mem_peak,
            'tokens_processed': None,  # Will be set by caller if known
            'batch_size': None,        # Will be set by caller if known
            'seq_len': None,           # Will be set by caller if known
            'step_index': self.step_index,
            'global_step': self.global_step
        }
        
        # Store results in thread-local storage for retrieval
        if not hasattr(self._thread_local, 'results'):
            self._thread_local.results = {}
        
        self._thread_local.results[self.stage_name] = result
        
        # Also store in global registry
        registry = get_profiler_registry()
        registry.add_result(result)
    
    def get_results(self) -> Dict[str, Any]:
        """Get the profiling results for this stage."""
        if hasattr(self._thread_local, 'results') and self.stage_name in self._thread_local.results:
            return self._thread_local.results[self.stage_name].copy()
        return {}
    
    def set_metadata(self, tokens_processed: Optional[int] = None, 
                    batch_size: Optional[int] = None, 
                    seq_len: Optional[int] = None):
        """Set additional metadata for the stage."""
        if hasattr(self._thread_local, 'results') and self.stage_name in self._thread_local.results:
            if tokens_processed is not None:
                self._thread_local.results[self.stage_name]['tokens_processed'] = tokens_processed
            if batch_size is not None:
                self._thread_local.results[self.stage_name]['batch_size'] = batch_size
            if seq_len is not None:
                self._thread_local.results[self.stage_name]['seq_len'] = seq_len


@contextmanager
def prof_stage(stage_name: str, step_index: int = 0, global_step: int = 0):
    """Context manager for profiling a training stage.
    
    Args:
        stage_name: Name of the stage being profiled
        step_index: Index within the current step
        global_step: Global training step number
    
    Yields:
        ProfilerContext: Context object that can be used to set metadata
    """
    context = ProfilerContext(stage_name, step_index, global_step)
    with context:
        yield context


def prof_stage_decorator(stage_name: str):
    """Decorator for profiling a function as a training stage.
    
    Args:
        stage_name: Name of the stage being profiled
    
    Returns:
        Decorated function that profiles execution
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with prof_stage(stage_name) as context:
                result = func(*args, **kwargs)
                return result
        return wrapper
    return decorator


# Global registry for collecting profiling results across threads
class ProfilerRegistry:
    """Global registry for collecting profiling results."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._results = []
    
    def add_result(self, result: Dict[str, Any]):
        """Add a profiling result to the registry."""
        with self._lock:
            self._results.append(result)
    
    def get_results(self) -> list:
        """Get all collected results."""
        with self._lock:
            return self._results.copy()
    
    def clear(self):
        """Clear all results."""
        with self._lock:
            self._results.clear()
    
    def get_stage_results(self, stage_name: str) -> list:
        """Get results for a specific stage."""
        with self._lock:
            return [r for r in self._results if r.get('stage') == stage_name]


# Global instance
registry = ProfilerRegistry()


def get_profiler_registry() -> ProfilerRegistry:
    """Get the global profiler registry."""
    return registry