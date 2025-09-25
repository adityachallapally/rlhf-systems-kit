"""
RLHF Core Components

Core components for RLHF training including policy models, reward models, PPO training,
profiling utilities, logging, and TRL integration.
"""

from .policy import PolicyModel
from .reward import ToyRewardModel
from .ppo import PPOTrainer
from .profiler import ProfilerManager, stage_timer
from .logging import JSONLLogger, write_sysinfo, create_run_dir, update_latest_symlink

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

# OpenRLHF Integration imports
try:
    from .openrlhf_integration import (
        OpenRLHFIntegrationConfig,
        OpenRLHFIntegrationManager,
        OpenRLHFPPOMonitoringCallback,
        OpenRLHFCheckpointAnalyzer,
        OpenRLHFRewardModelIntegrator,
        OpenRLHFTrainingCallback,
        MockOpenRLHFTrainer
    )
    OPENRLHF_INTEGRATION_AVAILABLE = True
except ImportError:
    OPENRLHF_INTEGRATION_AVAILABLE = False

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
    'update_latest_symlink'
]

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

# Add OpenRLHF integration to __all__ if available
if OPENRLHF_INTEGRATION_AVAILABLE:
    __all__.extend([
        'OpenRLHFIntegrationConfig',
        'OpenRLHFIntegrationManager',
        'OpenRLHFPPOMonitoringCallback',
        'OpenRLHFCheckpointAnalyzer',
        'OpenRLHFRewardModelIntegrator',
        'OpenRLHFTrainingCallback',
        'MockOpenRLHFTrainer'
    ])