"""
Profiler package for RLHF training instrumentation.

This package provides hooks for timing and memory profiling of RLHF training stages,
plus utilities for generating reports and torch profiler traces.
"""

from .hooks import prof_stage, ProfilerContext
from .report import ProfilerReport
from .trace import TorchProfiler
from .anomaly_detection import (
    AnomalyDetectionHook, 
    StepProfiler, 
    get_step_profiler, 
    register_anomaly_hook, 
    end_step
)

__all__ = [
    'prof_stage', 
    'ProfilerContext', 
    'ProfilerReport', 
    'TorchProfiler',
    'AnomalyDetectionHook',
    'StepProfiler', 
    'get_step_profiler', 
    'register_anomaly_hook', 
    'end_step'
]