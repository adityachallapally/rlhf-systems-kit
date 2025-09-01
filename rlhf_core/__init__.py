"""
RLHF Core Components

Core components for RLHF training including policy models, reward models, PPO training,
profiling utilities, logging, and divergence analysis.
"""

from .policy import PolicyModel
from .reward import ToyRewardModel
from .ppo import PPOTrainer
from .profiler import ProfilerManager, stage_timer
from .logging import JSONLLogger, write_sysinfo, create_run_dir, update_latest_symlink
from .divergence import DivergenceReport, first_divergence, generate_drift_card, analyze_multiple_runs

__version__ = "0.1.0"
__all__ = [
    'PolicyModel',
    'ToyRewardModel', 
    'PPOTrainer',
    'ProfilerManager',
    'stage_timer',
    'JSONLLogger',
    'write_sysinfo',
    'create_run_dir',
    'update_latest_symlink',
    'DivergenceReport',
    'first_divergence',
    'generate_drift_card',
    'analyze_multiple_runs'
]