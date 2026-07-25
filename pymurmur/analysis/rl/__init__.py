"""Reinforcement-learning bridge — Gymnasium environment + reward composite.

P12.2 MurmurationEnv (gym_env.py) and S3.9 reward composite
(rewards.py), grouped here since they're consumed together by the
MARL training pipeline.
"""

from .gym_env import MurmurationEnv  # noqa: F401
from .rewards import (  # noqa: F401
    RewardConfig,
    compute_reward,
    reward_linearity_check,
)

__all__ = [
    "MurmurationEnv",
    "RewardConfig",
    "compute_reward",
    "reward_linearity_check",
]
