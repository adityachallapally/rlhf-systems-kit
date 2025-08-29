"""
Profiler package for RLHF training instrumentation.

This package provides hooks for timing and memory profiling of RLHF training stages,
plus utilities for generating reports and torch profiler traces.
"""

from .hooks import prof_stage, ProfilerContext
from .report import ProfilerReport
from .trace import TorchProfiler

__all__ = ['prof_stage', 'ProfilerContext', 'ProfilerReport', 'TorchProfiler']