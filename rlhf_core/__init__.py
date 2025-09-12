"""
RLHF Core Components

Core components for RLHF training including policy models, reward models, PPO training,
profiling utilities, logging, and TRL integration.
"""

# Core components with optional imports
try:
    from .policy import PolicyModel
    POLICY_AVAILABLE = True
except ImportError:
    POLICY_AVAILABLE = False

try:
    from .reward import ToyRewardModel
    REWARD_AVAILABLE = True
except ImportError:
    REWARD_AVAILABLE = False

try:
    from .ppo import PPOTrainer
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False

try:
    from .profiler import ProfilerManager, stage_timer
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False

try:
    from .logging import JSONLLogger, write_sysinfo, create_run_dir, update_latest_symlink
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False

# TRL Integration imports
try:
    from .trl_integration import (
        TRLIntegrationConfig,
        TRLIntegrationManager,
        PPOMonitoringCallback,
        CheckpointAnalyzer,
        RewardModelIntegrator,
        TrainingCallback
    )
    TRL_INTEGRATION_AVAILABLE = True
except ImportError:
    TRL_INTEGRATION_AVAILABLE = False

__version__ = "0.1.0"

# Build __all__ dynamically based on available modules
__all__ = []

if POLICY_AVAILABLE:
    __all__.append('PolicyModel')

if REWARD_AVAILABLE:
    __all__.append('ToyRewardModel')

if PPO_AVAILABLE:
    __all__.append('PPOTrainer')

if PROFILER_AVAILABLE:
    __all__.extend(['ProfilerManager', 'stage_timer'])

if LOGGING_AVAILABLE:
    __all__.extend(['JSONLLogger', 'write_sysinfo', 'create_run_dir', 'update_latest_symlink'])

# Add TRL integration to __all__ if available
if TRL_INTEGRATION_AVAILABLE:
    __all__.extend([
        'TRLIntegrationConfig',
        'TRLIntegrationManager', 
        'PPOMonitoringCallback',
        'CheckpointAnalyzer',
        'RewardModelIntegrator',
        'TrainingCallback'
    ])