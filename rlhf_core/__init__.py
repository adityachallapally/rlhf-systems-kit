"""
RLHF Core Package

A minimal, reproducible RLHF training loop implementation.
"""

from .policy import PolicyModel
from .reward import ToyRewardModel, SimpleClassifierReward
from .ppo import PPOTrainer, RolloutBuffer

__version__ = "0.1.0"
__all__ = [
    "PolicyModel",
    "ToyRewardModel", 
    "SimpleClassifierReward",
    "PPOTrainer",
    "RolloutBuffer"
]