"""Execution-grounded GRPO rewards (Stage 2) + the multi-turn trajectory reward (Stage 3)."""

from __future__ import annotations

from text2sql_rlvr.rewards.multiturn import (
    CreditScheme,
    MultiTurnRewardConfig,
    TrajectoryBreakdown,
    score_trajectory,
)
from text2sql_rlvr.rewards.reward import (
    RewardArm,
    RewardBreakdown,
    RewardConfig,
    make_reward_fn,
    score_completion,
)

__all__ = [
    "CreditScheme",
    "MultiTurnRewardConfig",
    "RewardArm",
    "RewardBreakdown",
    "RewardConfig",
    "TrajectoryBreakdown",
    "make_reward_fn",
    "score_completion",
    "score_trajectory",
]
