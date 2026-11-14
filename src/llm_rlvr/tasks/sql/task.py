"""Text-to-SQL GRPO rows and the execution-grounded training reward.

This binds the SQL parser and execution-match verifier to the task-agnostic harness shaping
(`harness.score_answer`), and builds the GRPO rows the reward reads.

For the verifier the reference is the expected SQL and the context is the database path. The
training reward (`score_completion` / `make_reward_fn`) runs the verifier with the empty-result
guard on (`guard_trivial_empty`, blocking the always-empty reward hack); evaluation uses the
official unguarded semantics. The shaped arm is a weighted sum of
`format + executable + result_match + length`; the near-binary arm is `1.0` exact /
`0.1` executable / `0` else.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from llm_rlvr.data.models import DatabaseSchema, Example
from llm_rlvr.data.paths import resolve_database
from llm_rlvr.data.splits import assert_train_split
from llm_rlvr.harness import RewardBreakdown, Verdict, completion_text, score_answer
from llm_rlvr.harness import RewardConfig as _ShapingConfig
from llm_rlvr.tasks.sql.parsing import extract_sql, well_formed
from llm_rlvr.tasks.sql.prompt import build_prompts
from llm_rlvr.tasks.sql.verifier import GoldExecutionError, VerifierConfig, execution_match

# The reason tag for a completion that contains no SQL.
NO_SQL_REASON = "no_sql"


class RewardConfig(_ShapingConfig):
    """The harness shaping config plus the SQL-only empty-result guard.

    Adds `guard_trivial_empty` to the generic shaping (arm + component weights + format/length).
    The guard is on for training: an always-empty query cannot earn credit just because the expected
    result is empty.
    """

    guard_trivial_empty: bool = True


def sql_verdict(answer: str, gold_sql: str, db_path: str | Path, config: VerifierConfig) -> Verdict:
    """Execution-match `answer` against `gold_sql` on one database, as a harness `Verdict`.

    Raises `GoldExecutionError` when the reference query itself fails, so the generic reward records
    a `gold_error` without special-casing SQL.
    """
    result = execution_match(answer, gold_sql, db_path, config=config)
    return Verdict(
        matched=result.matched,
        valid=result.reason not in {"pred_error", "pred_truncated"},
        reason=result.reason,
    )


def score_completion(
    text: str,
    gold_sql: str,
    db_path: str | Path,
    *,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Score one completion against the expected SQL on one database.

    Executes the prediction once through the SQL verifier (read-only sandbox) and derives the
    `executable` and `result_match` components from the single verdict.
    """
    cfg = config or RewardConfig()
    verifier_config = VerifierConfig(guard_trivial_empty=cfg.guard_trivial_empty)
    return score_answer(
        text,
        extract=extract_sql,
        well_formed=well_formed,
        verify=lambda sql: sql_verdict(sql, gold_sql, db_path, verifier_config),
        config=cfg,
        no_answer_reason=NO_SQL_REASON,
    )


def make_reward_fn(
    db_root: str | Path,
    *,
    config: RewardConfig | None = None,
    resolver: Callable[[str, str | Path], Path] = resolve_database,
) -> Callable[..., list[float]]:
    """Wrap `score_completion` in the TRL `GRPOTrainer` reward-function convention.

    TRL calls the returned function with `completions` and the remaining dataset columns
    (`gold_sql`, `db_id`) as aligned keyword lists; the database is resolved per example from
    `db_id` under `db_root`.
    """
    cfg = config or RewardConfig()
    if not cfg.guard_trivial_empty:
        raise ValueError("SQL training reward requires guard_trivial_empty=True")

    def reward(
        completions: Sequence[object],
        gold_sql: Sequence[str],
        db_id: Sequence[str],
        **_kwargs: object,
    ) -> list[float]:
        rewards: list[float] = []
        for completion, gold, dbid in zip(completions, gold_sql, db_id, strict=True):
            db_path = resolver(dbid, db_root)
            breakdown = score_completion(completion_text(completion), gold, db_path, config=cfg)
            if breakdown.reason == "gold_error":
                raise GoldExecutionError(
                    f"training reference became unverifiable for db_id={dbid!r}"
                )
            rewards.append(breakdown.total)
        return rewards

    reward.__name__ = f"execution_reward_{cfg.arm}"
    return reward


def build_grpo_rows(
    examples: Sequence[Example], schemas: Mapping[str, DatabaseSchema], *, style: str = "cot"
) -> list[dict[str, object]]:
    """One row per example: the chat `prompt` plus the `gold_sql` + `db_id` the reward needs."""
    assert_train_split(examples, context="SQL GRPO rows")
    prompts = build_prompts(examples, schemas, style=style)
    return [
        {"prompt": prompt, "gold_sql": example.gold_sql, "db_id": example.db_id}
        for example, prompt in zip(examples, prompts, strict=True)
    ]
