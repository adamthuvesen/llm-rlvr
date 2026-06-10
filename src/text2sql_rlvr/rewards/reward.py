"""Execution-grounded GRPO reward (Stage 2, design D2).

One reward over `(completion, gold_sql, database)`: parse the SQL, execute it in the read-only
sandbox, and score it against gold with the execution-match verifier — a verifiable signal, not a
model judgement. Two arms, selected by config:

  - **shaped** (SQL-R1): a weighted sum of `format` + `executable` + `result_match` + `length`.
  - **near_binary** (Arctic): `1.0` exact / `0.1` executable / `0` else — and nothing else.

The training reward runs the verifier with the **empty-result guard** on (`guard_trivial_empty`),
so an always-empty query cannot farm credit where gold is empty. The `format` component gives full
credit only when `<think>` *leads* the completion (no prose preamble) — the Stage-1 cold-start
emits a preamble before `<think>` on ~99% of completions, ~54% of the generated tokens, so rewarding
a leading `<think>` turns that wasted rollout budget into an RL signal toward the cheaper format.

The core `score_completion` is pure and CPU-testable; `make_reward_fn` wraps it in the TRL
`GRPOTrainer` reward-function calling convention (pulling `gold_sql` + `db_id` from the batch
kwargs and resolving the database per example).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from text2sql_rlvr._base import Frozen
from text2sql_rlvr.data.spider import resolve_database
from text2sql_rlvr.eval.parsing import extract_sql, well_formed
from text2sql_rlvr.eval.verifier import (
    GoldExecutionError,
    VerifierConfig,
    execution_match,
)

RewardArm = Literal["shaped", "near_binary"]


class RewardConfig(Frozen):
    """Selects the arm and the shaped-arm component weights — the shaped-vs-binary ablation knob."""

    arm: RewardArm = "shaped"
    # Shaped-arm component weights. result_match dominates; the rest shape the path to it.
    w_format: float = 0.2
    w_executable: float = 0.2
    w_result_match: float = 1.0
    w_length: float = 0.1
    # format component: credit for a well-formed completion whose <think> does NOT lead (has a
    # prose preamble). Full credit (1.0) requires a leading <think>; this caps the preamble case.
    preamble_credit: float = 0.3
    # length component (a conciseness proxy in characters — the reward layer has no tokenizer):
    # full credit up to `length_target_chars`, linear decay to 0 at `length_max_chars`.
    length_target_chars: int = 400
    length_max_chars: int = 1600
    # Empty-result guard on for training (blocks the always-empty reward hack).
    guard_trivial_empty: bool = True


class RewardBreakdown(Frozen):
    """The components behind a reward — surfaced for transparency and unit tests."""

    total: float
    format: float
    executable: float
    result_match: float
    length: float
    reason: str


def _completion_text(completion: object) -> str:
    """Extract assistant text from a TRL completion (a string, a message, or a message list)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, Sequence):
        for message in reversed(completion):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content", ""))
        return _completion_text(completion[-1]) if completion else ""
    return str(completion)


def _format_score(text: str, cfg: RewardConfig) -> float:
    """1.0 for a well-formed completion led by <think>; `preamble_credit` if a preamble precedes
    it; 0.0 if malformed (wrong/missing think tags, zero-or-many fences, a non-sql fence)."""
    if not well_formed(text):
        return 0.0
    leads = text.lstrip().lower().startswith("<think>")
    return 1.0 if leads else cfg.preamble_credit


def _length_score(text: str, cfg: RewardConfig) -> float:
    """Conciseness term: full credit up to the target, linear decay to 0 at the max."""
    n = len(text.strip())
    if n <= cfg.length_target_chars:
        return 1.0
    if n >= cfg.length_max_chars:
        return 0.0
    span = cfg.length_max_chars - cfg.length_target_chars
    return (cfg.length_max_chars - n) / span


def score_completion(
    text: str,
    gold_sql: str,
    db_path: str | Path,
    *,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Score one completion against gold on one database. Pure and side-effect-free.

    Executes the prediction once through the verifier (read-only sandbox) and derives both the
    `executable` and `result_match` components from the single verdict, so no query runs twice.
    """
    cfg = config or RewardConfig()
    fmt = _format_score(text, cfg)
    length = _length_score(text, cfg)

    sql = extract_sql(text)
    if sql is None:
        return _assemble(cfg, fmt=fmt, executable=0.0, matched=0.0, length=length, reason="no_sql")

    verifier = VerifierConfig(guard_trivial_empty=cfg.guard_trivial_empty)
    try:
        result = execution_match(sql, gold_sql, db_path, config=verifier)
    except GoldExecutionError:
        # The example/DB pairing is broken, not the prediction — don't reward, don't crash.
        return _assemble(
            cfg, fmt=fmt, executable=0.0, matched=0.0, length=length, reason="gold_error"
        )

    # pred_error is the only reason the prediction did not execute; every other verdict ran.
    executable = 0.0 if result.reason == "pred_error" else 1.0
    matched = 1.0 if result.matched else 0.0
    return _assemble(
        cfg, fmt=fmt, executable=executable, matched=matched, length=length, reason=result.reason
    )


def _assemble(
    cfg: RewardConfig,
    *,
    fmt: float,
    executable: float,
    matched: float,
    length: float,
    reason: str,
) -> RewardBreakdown:
    if cfg.arm == "near_binary":
        total = 1.0 if matched else (0.1 if executable else 0.0)
    else:
        total = (
            cfg.w_format * fmt
            + cfg.w_executable * executable
            + cfg.w_result_match * matched
            + cfg.w_length * length
        )
    return RewardBreakdown(
        total=total,
        format=fmt,
        executable=executable,
        result_match=matched,
        length=length,
        reason=reason,
    )


def make_reward_fn(
    db_root: str | Path,
    *,
    config: RewardConfig | None = None,
    resolver: Callable[[str, str | Path], Path] = resolve_database,
) -> Callable[..., list[float]]:
    """Wrap `score_completion` in the TRL `GRPOTrainer` reward-function convention.

    TRL calls the returned function with `completions` and the remaining dataset columns as
    aligned keyword lists; this reward needs `gold_sql` and `db_id`. The database is resolved per
    example from `db_id` under `db_root` (Spider-train by default).
    """
    cfg = config or RewardConfig()

    def reward(completions, gold_sql, db_id, **_kwargs) -> list[float]:
        rewards: list[float] = []
        for completion, gold, dbid in zip(completions, gold_sql, db_id, strict=True):
            db_path = resolver(dbid, db_root)
            breakdown = score_completion(_completion_text(completion), gold, db_path, config=cfg)
            rewards.append(breakdown.total)
        return rewards

    reward.__name__ = f"execution_reward_{cfg.arm}"
    return reward
