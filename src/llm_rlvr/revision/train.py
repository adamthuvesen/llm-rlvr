"""Forced-revision GRPO training driver and provenance."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from llm_rlvr._base import Frozen
from llm_rlvr.data.models import DatabaseSchema, Example
from llm_rlvr.env.text2sql_env import FeedbackConfig
from llm_rlvr.eval.completions import write_json
from llm_rlvr.revision.feedback import REVISE_INSTRUCTION
from llm_rlvr.revision.rollout import (
    build_db_id_lookup,
    make_forced_revision_reward_fn,
    make_forced_revision_rollout,
)
from llm_rlvr.rewards.multiturn import MultiTurnRewardConfig, TrajectoryBreakdown
from llm_rlvr.train.diagnostics import CollapseDiagnostics, compute_diagnostics
from llm_rlvr.train.grpo import (
    GrpoHyperparams,
    build_grpo_config_from_kwargs,
    build_grpo_provenance,
    build_grpo_rows,
    grpo_config_kwargs,
    load_trainable_policy,
    split_train_dev,
)
from llm_rlvr.train.stage3_provenance import Stage3GrpoProvenance

DIAGNOSTICS_FILENAME = "d8-diagnostics.json"
PROVENANCE_FILENAME = "provenance.json"
DIAGNOSTICS_WINDOW = 128


class ForcedRevisionProvenance(Stage3GrpoProvenance):
    rollout: Literal["forced_revision"] = "forced_revision"
    prompt_style: str = "cot"
    revise_instruction: str = REVISE_INSTRUCTION


class DiagnosticsSeries(Frozen):
    window_episodes: int
    windows: tuple[CollapseDiagnostics, ...]
    overall: CollapseDiagnostics | None


class _EpisodeLog:
    def __init__(self) -> None:
        self.episodes: list[tuple[list[str | None], TrajectoryBreakdown]] = []
        self.windows: list[CollapseDiagnostics] = []

    def __call__(self, sqls: list[str | None], breakdown: TrajectoryBreakdown) -> None:
        self.episodes.append((sqls, breakdown))
        if len(self.episodes) % DIAGNOSTICS_WINDOW == 0:
            window = compute_diagnostics(self.episodes[-DIAGNOSTICS_WINDOW:])
            self.windows.append(window)
            print(f"[fr-d8] episodes={len(self.episodes)} {window.model_dump_json()}")

    def series(self) -> DiagnosticsSeries:
        return DiagnosticsSeries(
            window_episodes=DIAGNOSTICS_WINDOW,
            windows=tuple(self.windows),
            overall=compute_diagnostics(self.episodes) if self.episodes else None,
        )


def build_trainer_pieces(
    examples: Sequence[Example],
    schemas: Mapping[str, DatabaseSchema],
    db_root: str | Path,
    *,
    hyperparams: GrpoHyperparams,
    environment: FeedbackConfig,
    reward_config: MultiTurnRewardConfig,
    on_episode: Callable[[list[str | None], TrajectoryBreakdown], None],
):
    from datasets import Dataset

    train_examples, dev_examples = split_train_dev(
        examples, dev_size=hyperparams.eval_slice_size, seed=hyperparams.seed
    )
    rows = build_grpo_rows(train_examples, schemas, style="cot")
    rollout = make_forced_revision_rollout(db_root, build_db_id_lookup(rows), feedback=environment)
    reward_fn = make_forced_revision_reward_fn(db_root, config=reward_config, on_episode=on_episode)
    return train_examples, dev_examples, Dataset.from_list(rows), rollout, reward_fn


def build_forced_revision_provenance(
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
    base_model_revision: str | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> ForcedRevisionProvenance:
    return ForcedRevisionProvenance(
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


def run_forced_revision_train(
    examples: Sequence[Example],
    schemas: Mapping[str, DatabaseSchema],
    db_root: str | Path,
    output_dir: str | Path,
    *,
    base_model: str,
    warm_start_adapter: str | Path,
    dataset: str,
    dataset_revision: str,
    base_model_revision: str | None = None,
    hyperparams: GrpoHyperparams | None = None,
    environment: FeedbackConfig | None = None,
    multiturn_reward: MultiTurnRewardConfig | None = None,
    max_steps: int | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> Path:
    import torch
    from trl import GRPOTrainer

    hp = hyperparams or GrpoHyperparams()
    env_cfg = environment or FeedbackConfig()
    reward_cfg = multiturn_reward or MultiTurnRewardConfig()

    log = _EpisodeLog()
    train_examples, dev_examples, train_dataset, rollout, reward_fn = build_trainer_pieces(
        examples,
        schemas,
        db_root,
        hyperparams=hp,
        environment=env_cfg,
        reward_config=reward_cfg,
        on_episode=log,
    )

    desired = grpo_config_kwargs(hp, output_dir)
    if max_steps is not None:
        desired["max_steps"] = max_steps

    policy = load_trainable_policy(base_model, base_model_revision, warm_start_adapter)
    torch.cuda.reset_peak_memory_stats()
    trainer = GRPOTrainer(
        model=policy,
        reward_funcs=[reward_fn],
        args=build_grpo_config_from_kwargs(desired),
        train_dataset=train_dataset,
        rollout_func=rollout,
    )
    trainer.train()
    trainer.save_model(str(output_dir))

    write_json(log.series(), Path(output_dir) / DIAGNOSTICS_FILENAME)
    provenance = build_forced_revision_provenance(
        base_model=base_model,
        warm_start_adapter=warm_start_adapter,
        hyperparams=hp,
        environment=env_cfg,
        multiturn_reward=reward_cfg,
        dataset=dataset,
        dataset_revision=dataset_revision,
        n_train=len(train_examples),
        n_dev=len(dev_examples),
        base_model_revision=base_model_revision,
        commit=commit,
        dirty=dirty,
    )
    write_json(provenance, Path(output_dir) / PROVENANCE_FILENAME)
    return Path(output_dir)
