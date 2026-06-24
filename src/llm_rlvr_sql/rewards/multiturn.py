"""Multi-turn trajectory reward (Stage 3, design D3/D4).

An episode under TRL's tool loop is: zero-or-more `run_sql` tool calls (bare SQL, recorded by
the environment), then the model's final answer (a full `<think>`+SQL completion). Credit:

  - **hybrid** (default): the final answer's Stage-2 reward dominates, blended with the mean
    per-turn *execution progress* (each tool query scored 1.0 result-match / 0.1 executable /
    0.0 — MT-GRPO: outcome-only credit underperforms on tool use) plus an **improvement-delta
    bonus**: only *positive* between-turn progress deltas earn it, and an unchanged-SQL
    resubmission earns exactly zero (SCoRe guard: reward fixing, not re-submitting).
  - **trajectory**: outcome-only (the final answer's reward, nothing else) — the credit ablation.

Per-turn progress and the delta share one scale (the execution tiers), so the bonus is never a
units mismatch against the shaped final total. The first-turn KL anchor (the other SCoRe guard)
is a training-loss term over turn-1 token spans — it lives in the trainer wiring; the switch
here is recorded in provenance. Scoring is verification-grade (full row caps, empty-result
guard): the agent-facing preview caps in `env/` never decide reward.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from llm_rlvr_sql._base import Frozen
from llm_rlvr_sql.eval.parsing import extract_sql
from llm_rlvr_sql.eval.verifier import GoldExecutionError, VerifierConfig, execution_match
from llm_rlvr_sql.rewards.reward import RewardBreakdown, RewardConfig, score_completion

CreditScheme = Literal["hybrid", "trajectory"]


class MultiTurnRewardConfig(Frozen):
    """Credit assignment across turns + the SCoRe-guard switches."""

    base: RewardConfig = RewardConfig()
    credit: CreditScheme = "hybrid"
    # Hybrid blend: the final answer dominates; per-turn progress and improvement shape the path.
    w_final: float = 1.0
    w_per_turn: float = 0.3
    w_delta: float = 0.3
    # SCoRe guard switches (ON by default). The KL anchor is enforced in the trainer loss; the
    # delta bonus is enforced here. Both are recorded in provenance either way.
    first_turn_kl_anchor: bool = True
    delta_bonus: bool = True


class TrajectoryBreakdown(Frozen):
    """The components behind one episode's reward — for transparency, tests, and diagnostics."""

    total: float
    final: RewardBreakdown
    turn_progress: tuple[float, ...]  # execution tier per turn (tool calls + final answer)
    delta_bonus: float
    turns_used: int  # tool calls + the final answer


def _execution_tier(
    sql: str | None, gold_sql: str, db_path: str | Path, cfg: RewardConfig
) -> float:
    """The near-binary execution tier: 1.0 result-match / 0.1 executable / 0.0 — one scale for
    per-turn progress regardless of the final-answer arm. Empty-result guard stays on."""
    if sql is None:
        return 0.0
    verifier = VerifierConfig(guard_trivial_empty=cfg.guard_trivial_empty)
    try:
        result = execution_match(sql, gold_sql, db_path, config=verifier)
    except GoldExecutionError:
        return 0.0
    if result.matched:
        return 1.0
    return 0.0 if result.reason == "pred_error" else 0.1


def _improvement_delta(sqls: Sequence[str | None], tiers: Sequence[float]) -> float:
    """Sum of positive between-turn progress deltas; an unchanged-SQL resubmission earns zero."""
    bonus = 0.0
    for i in range(1, len(tiers)):
        previous, current = sqls[i - 1], sqls[i]
        unchanged = (
            previous is not None and current is not None and previous.strip() == current.strip()
        )
        if unchanged:
            continue
        bonus += max(0.0, tiers[i] - tiers[i - 1])
    return bonus


def score_trajectory(
    tool_queries: Sequence[str],
    final_completion: str,
    gold_sql: str,
    db_path: str | Path,
    *,
    config: MultiTurnRewardConfig | None = None,
) -> TrajectoryBreakdown:
    """Score one episode (its `run_sql` queries + the final answer). Pure and CPU-testable."""
    cfg = config or MultiTurnRewardConfig()

    final = score_completion(final_completion, gold_sql, db_path, config=cfg.base)
    sqls: list[str | None] = [*tool_queries, extract_sql(final_completion)]
    tiers = [_execution_tier(sql, gold_sql, db_path, cfg.base) for sql in sqls]

    if cfg.credit == "trajectory":
        total = final.total
        delta = 0.0
    else:
        delta = _improvement_delta(sqls, tiers) if cfg.delta_bonus else 0.0
        total = (
            cfg.w_final * final.total
            + cfg.w_per_turn * (sum(tiers) / len(tiers))
            + cfg.w_delta * delta
        )

    return TrajectoryBreakdown(
        total=total,
        final=final,
        turn_progress=tuple(tiers),
        delta_bonus=delta,
        turns_used=len(sqls),
    )
