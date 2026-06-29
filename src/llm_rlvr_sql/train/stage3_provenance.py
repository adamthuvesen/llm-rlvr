"""Shared provenance contract for Stage 3-style GRPO runs."""

from __future__ import annotations

from typing import Literal

from llm_rlvr_sql._base import Frozen
from llm_rlvr_sql.env.text2sql_env import FeedbackConfig
from llm_rlvr_sql.rewards.multiturn import MultiTurnRewardConfig
from llm_rlvr_sql.train.grpo import GrpoProvenance

Stage3Rollout = Literal["multiturn", "forced_revision"]


class Stage3GrpoProvenance(Frozen):
    """Common provenance for GRPO runs that add an environment and trajectory reward."""

    grpo: GrpoProvenance
    environment: FeedbackConfig
    multiturn_reward: MultiTurnRewardConfig
    rollout: Stage3Rollout
    prompt_style: str = "cot"
