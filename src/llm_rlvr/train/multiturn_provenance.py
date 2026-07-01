"""Multi-turn GRPO provenance.

Records the base model, warm start, GRPO hyperparameters, library versions, git state, environment
config, and multi-turn reward config needed to reconstruct a checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from llm_rlvr.env.text2sql_env import FeedbackConfig
from llm_rlvr.rewards.multiturn import MultiTurnRewardConfig
from llm_rlvr.train.grpo import GrpoHyperparams, GrpoProvenance, build_grpo_provenance
from llm_rlvr.train.stage3_provenance import Stage3GrpoProvenance


class MultiTurnProvenance(Stage3GrpoProvenance):
    """Everything needed to reproduce a multi-turn GRPO checkpoint."""

    grpo: GrpoProvenance
    rollout: Literal["multiturn"] = "multiturn"
    # Multi-turn prompts depart from the frozen single-turn `cot` prompt: the instruction
    # must offer the run_sql protocol or the policy never calls it.
    prompt_style: str = "cot"


def build_multiturn_provenance(
    *,
    base_model: str,
    warm_start_adapter: str | Path,
    hyperparams: GrpoHyperparams,
    environment: FeedbackConfig,
    multiturn_reward: MultiTurnRewardConfig,
    dataset: str,
    dataset_revision: str,
    n_train: int,
    n_dev: int,
    prompt_style: str = "cot",
    base_model_revision: str | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> MultiTurnProvenance:
    """Assemble multi-turn checkpoint provenance on top of the GRPO builder."""
    return MultiTurnProvenance(
        prompt_style=prompt_style,
        grpo=build_grpo_provenance(
            base_model=base_model,
            warm_start_adapter=warm_start_adapter,
            hyperparams=hyperparams,
            reward=multiturn_reward.base,
            dataset=dataset,
            dataset_revision=dataset_revision,
            n_train=n_train,
            n_dev=n_dev,
            base_model_revision=base_model_revision,
            commit=commit,
            dirty=dirty,
        ),
        environment=environment,
        multiturn_reward=multiturn_reward,
    )
