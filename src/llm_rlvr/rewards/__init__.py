"""Multi-turn trajectory reward for tool-loop / self-correction analysis.

The single-turn execution-grounded reward lives with the text-to-SQL task (`llm_rlvr.tasks.sql`)
and the task-agnostic shaping core lives in `llm_rlvr.harness`. This package holds the
higher-order trajectory reward that credits a multi-turn episode on top of that single-turn score.
"""

from __future__ import annotations

from llm_rlvr.rewards.multiturn import (
    CreditScheme,
    MultiTurnRewardConfig,
    TrajectoryBreakdown,
    score_trajectory,
)

__all__ = [
    "CreditScheme",
    "MultiTurnRewardConfig",
    "TrajectoryBreakdown",
    "score_trajectory",
]
