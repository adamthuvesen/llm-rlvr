"""Multi-turn GRPO wiring and smoke driver. See docs/status.md for current scope."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from llm_rlvr_sql._base import Frozen
from llm_rlvr_sql.data.models import DatabaseSchema, Example
from llm_rlvr_sql.data.paths import resolve_database
from llm_rlvr_sql.env.text2sql_env import FeedbackConfig, make_environment_factory
from llm_rlvr_sql.eval.completions import write_json
from llm_rlvr_sql.eval.parsing import extract_sql
from llm_rlvr_sql.harness import completion_text
from llm_rlvr_sql.rewards.multiturn import (
    MultiTurnRewardConfig,
    TrajectoryBreakdown,
    score_trajectory,
)
from llm_rlvr_sql.train.diagnostics import CollapseDiagnostics, compute_diagnostics
from llm_rlvr_sql.train.grpo import (
    GrpoHyperparams,
    build_grpo_config_from_kwargs,
    build_grpo_rows,
    grpo_config_kwargs,
    load_trainable_policy,
    split_train_dev,
)
from llm_rlvr_sql.train.grpo_smoke import (
    IsRatioCheck,
    MaskCheck,
    capturing_grpo_trainer,
    check_is_ratio,
    check_tool_masking,
    logged_mean,
)
from llm_rlvr_sql.train.multiturn_provenance import (
    MultiTurnProvenance,
    build_multiturn_provenance,
)

SMOKE_REPORT_FILENAME = "smoke-report.json"


class MultiTurnHyperparams(GrpoHyperparams):
    """Stage-2 GRPO budget + the tool-loop bound (= the env's own turn budget, defense in depth)."""

    max_tool_calling_iterations: int = 3


def multiturn_grpo_config_kwargs(
    hyperparams: MultiTurnHyperparams, output_dir: str | Path
) -> dict[str, object]:
    """The Stage-2 GRPOConfig kwargs + the tool-loop bound."""
    return {
        **grpo_config_kwargs(hyperparams, output_dir),
        "max_tool_calling_iterations": hyperparams.max_tool_calling_iterations,
    }


def make_multiturn_reward_fn(
    db_root: str | Path,
    *,
    config: MultiTurnRewardConfig | None = None,
    resolver: Callable[[str, str | Path], Path] = resolve_database,
    on_episode: Callable[[list[str | None], TrajectoryBreakdown], None] | None = None,
) -> Callable[..., list[float]]:
    """Wrap `score_trajectory` in the TRL reward convention for environment rollouts.

    The trainer passes `environments=` (one env per rollout slot, aligned with `completions`);
    each env's recorded `turns` are the episode's tool queries, the completion is the final
    answer. `on_episode` (optional) sees every episode's SQLs + breakdown — the smoke and the
    diagnostics logging hang off it without touching reward values.
    """
    cfg = config or MultiTurnRewardConfig()

    def reward(completions, gold_sql, db_id, environments=None, **_kwargs) -> list[float]:
        if environments is None:
            raise ValueError(
                "multi-turn reward needs the trainer's environments= kwarg; "
                "was GRPOTrainer built with environment_factory?"
            )
        rewards: list[float] = []
        for completion, gold, dbid, env in zip(
            completions, gold_sql, db_id, environments, strict=True
        ):
            text = completion_text(completion)
            queries = [turn.query for turn in env.turns]
            breakdown = score_trajectory(queries, text, gold, resolver(dbid, db_root), config=cfg)
            if on_episode is not None:
                on_episode([*queries, extract_sql(text)], breakdown)
            rewards.append(breakdown.total)
        return rewards

    reward.__name__ = f"multiturn_reward_{cfg.credit}"
    return reward


class MultiTurnSmokeReport(Frozen):
    """The smoke's full evidence: assertions, tool-use census, envelope, and provenance."""

    passed: bool
    failures: tuple[str, ...]
    n_train: int
    n_steps: int
    wall_time_s: float
    mean_step_time_s: float | None
    n_episodes: int
    episodes_with_tool_calls: int
    episodes_with_2plus_tool_turns: int
    max_tool_turns: int
    tool_call_frequency: float | None  # trainer-logged tools/call_frequency mean
    tool_failure_frequency: float | None
    mask: MaskCheck
    is_ratio: IsRatioCheck
    diagnostics: CollapseDiagnostics | None  # None when no episodes were scored
    peak_torch_memory_gib: float | None
    device_memory_used_gib: float | None  # torch + vLLM colocate, from the device
    completions_mean_length: float | None
    provenance: MultiTurnProvenance


def smoke_failures(
    *,
    episodes_with_tool_calls: int,
    episodes_with_2plus_tool_turns: int,
    mask: MaskCheck,
    is_ratio: IsRatioCheck,
) -> tuple[str, ...]:
    """The D5 verdict, pure: which smoke assertions failed (empty = paid runs may proceed)."""
    failures: list[str] = []
    if episodes_with_tool_calls == 0:
        failures.append("no episode called run_sql — the tool loop never fired")
    if episodes_with_2plus_tool_turns == 0:
        failures.append("no episode used >=2 tool turns — the spec needs a real 2-turn rollout")
    if mask.masked_tokens_total == 0:
        failures.append("no completion token was tool-masked")
    if not mask.feedback_in_masked_text:
        failures.append("env feedback never appeared in the masked (loss-excluded) spans")
    if mask.feedback_in_unmasked_text:
        failures.append("env feedback leaked into gradient-carrying tokens")
    if not is_ratio.in_band:
        failures.append(
            f"importance-sampling ratio outside the Stage-2 band {is_ratio.band}: "
            f"means avg {is_ratio.mean_of_means}, n_logged {is_ratio.n_logged}"
        )
    return tuple(failures)


def run_multiturn_smoke(
    examples: Sequence[Example],
    schemas: Mapping[str, DatabaseSchema],
    db_root: str | Path,
    output_dir: str | Path,
    *,
    base_model: str,
    warm_start_adapter: str | Path | None,
    dataset: str,
    dataset_revision: str,
    base_model_revision: str | None = None,
    hyperparams: MultiTurnHyperparams | None = None,
    environment: FeedbackConfig | None = None,
    multiturn_reward: MultiTurnRewardConfig | None = None,
    prompt_style: str = "mt_demo",
    max_steps: int | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> Path:
    """Run the de-risk smoke (tasks 2.1/2.2) on the GPU; persist `MultiTurnSmokeReport` JSON.

    Trains a few real multi-turn GRPO steps from the Stage-2 adapter, capturing every rollout
    batch's `tool_mask`, then asserts the D5 conditions. Raises `RuntimeError` on a failed
    assertion — AFTER writing the report, so the evidence survives the non-zero exit.
    """
    import torch
    from datasets import Dataset  # lazy: heavy training stack, train extra only
    from trl import GRPOTrainer

    hp = hyperparams or MultiTurnHyperparams()
    env_cfg = environment or FeedbackConfig()
    reward_cfg = multiturn_reward or MultiTurnRewardConfig()
    # Defense in depth (design D2) only works if both budgets agree; provenance records the
    # env's max_turns, so a silent divergence here would be invisible later. Refuse it.
    if hp.max_tool_calling_iterations != env_cfg.max_turns:
        raise ValueError(
            f"max_tool_calling_iterations ({hp.max_tool_calling_iterations}) must equal the "
            f"environment turn budget ({env_cfg.max_turns})"
        )

    train_examples, dev_examples = split_train_dev(
        examples, dev_size=hp.eval_slice_size, seed=hp.seed
    )
    train_dataset = Dataset.from_list(build_grpo_rows(train_examples, schemas, style=prompt_style))

    episodes: list[tuple[list[str | None], TrajectoryBreakdown]] = []
    reward_fn = make_multiturn_reward_fn(
        db_root,
        config=reward_cfg,
        on_episode=lambda sqls, breakdown: episodes.append((sqls, breakdown)),
    )

    captured: list[dict[str, list[list[int]]]] = []

    desired = multiturn_grpo_config_kwargs(hp, output_dir)
    if max_steps is not None:
        desired["max_steps"] = max_steps

    # With a warm-start and beta != 0, TRL snapshots the adapter as the frozen ref policy.
    # Without one, this is the fresh identity-LoRA entrenchment-probe arm.
    policy = load_trainable_policy(
        base_model, base_model_revision, warm_start_adapter, hyperparams=hp
    )

    torch.cuda.reset_peak_memory_stats()
    capturing_trainer_cls = capturing_grpo_trainer(GRPOTrainer)
    trainer = capturing_trainer_cls(
        model=policy,
        reward_funcs=[reward_fn],
        args=build_grpo_config_from_kwargs(desired),
        train_dataset=train_dataset,
        environment_factory=make_environment_factory(db_root, config=env_cfg),
        captured=captured,
    )
    started = time.perf_counter()
    trainer.train()
    wall_time_s = time.perf_counter() - started

    tokenizer = trainer.processing_class

    def decode(ids: Sequence[int]) -> str:
        return tokenizer.decode(list(ids), skip_special_tokens=True)

    mask = check_tool_masking(
        [row for batch in captured for row in batch["completion_ids"]],
        [row for batch in captured for row in batch["completion_mask"]],
        [row for batch in captured for row in batch["tool_mask"]],
        decode,
    )
    log_history = trainer.state.log_history
    is_ratio = check_is_ratio(log_history)

    tool_turn_counts = [len(sqls) - 1 for sqls, _ in episodes]  # last entry is the final answer
    with_tools = sum(1 for n in tool_turn_counts if n >= 1)
    with_two_plus = sum(1 for n in tool_turn_counts if n >= 2)
    failures = smoke_failures(
        episodes_with_tool_calls=with_tools,
        episodes_with_2plus_tool_turns=with_two_plus,
        mask=mask,
        is_ratio=is_ratio,
    )

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    report = MultiTurnSmokeReport(
        passed=not failures,
        failures=failures,
        n_train=len(train_examples),
        n_steps=int(trainer.state.global_step),
        wall_time_s=wall_time_s,
        mean_step_time_s=logged_mean(log_history, "step_time"),
        n_episodes=len(episodes),
        episodes_with_tool_calls=with_tools,
        episodes_with_2plus_tool_turns=with_two_plus,
        max_tool_turns=max(tool_turn_counts, default=0),
        tool_call_frequency=logged_mean(log_history, "tools/call_frequency"),
        tool_failure_frequency=logged_mean(log_history, "tools/failure_frequency"),
        mask=mask,
        is_ratio=is_ratio,
        diagnostics=compute_diagnostics(episodes) if episodes else None,
        peak_torch_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
        device_memory_used_gib=(total_bytes - free_bytes) / 2**30,
        completions_mean_length=logged_mean(log_history, "completions/mean_length"),
        provenance=build_multiturn_provenance(
            base_model=base_model,
            warm_start_adapter=(
                warm_start_adapter
                if warm_start_adapter is not None
                else "none (fresh identity LoRA from base — entrenchment probe)"
            ),
            # Down-cast to the declared provenance type; the tool-loop bound it would lose is
            # pinned to environment.max_turns by the equality check above.
            hyperparams=GrpoHyperparams.model_validate(
                hp.model_dump(exclude={"max_tool_calling_iterations"})
            ),
            environment=env_cfg,
            multiturn_reward=reward_cfg,
            dataset=dataset,
            dataset_revision=dataset_revision,
            n_train=len(train_examples),
            n_dev=len(dev_examples),
            prompt_style=prompt_style,
            base_model_revision=base_model_revision,
            commit=commit,
            dirty=dirty,
        ),
    )
    out = write_json(report, Path(output_dir) / SMOKE_REPORT_FILENAME)
    if failures:
        raise RuntimeError(
            "multi-turn smoke FAILED — paid runs are blocked (design-D5 fallback ladder): "
            + "; ".join(failures)
            + f" (report: {out})"
        )
    return out
