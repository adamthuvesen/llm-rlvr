"""Text-to-SQL: the reference `Task` for the RLVR harness.

This wires the SQL pieces — the defensive answer parser (`eval.parsing`), the execution-match
verifier (`eval.verifier`), and the schema-aware prompt/row builders (`eval.generate`) — into the
task-agnostic `harness` protocols, and supplies the SQL execution-grounded GRPO reward (Stage 2,
design D2) on top of the harness shaping core.

For the verifier the reference is the gold SQL and the context is the database path. The training
reward (`score_completion` / `make_reward_fn`) runs the verifier with the empty-result guard on
(`guard_trivial_empty`, blocking the always-empty reward hack); evaluation uses the official
unguarded semantics. The shaped arm is a weighted sum of `format + executable + result_match +
length`; the near-binary arm is `1.0` exact / `0.1` executable / `0` else.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from llm_rlvr.data.models import DatabaseSchema, Example
from llm_rlvr.data.paths import resolve_database
from llm_rlvr.eval.generate import Messages, build_prompts
from llm_rlvr.eval.parsing import extract_sql, well_formed
from llm_rlvr.eval.verifier import VerifierConfig, execution_match
from llm_rlvr.harness import RewardBreakdown, Verdict, Verifier, completion_text, score_answer
from llm_rlvr.harness import RewardConfig as _ShapingConfig

# The reason tag for a completion that contains no SQL.
NO_SQL_REASON = "no_sql"


class RewardConfig(_ShapingConfig):
    """The harness shaping config plus the SQL-only empty-result guard.

    Adds `guard_trivial_empty` to the generic shaping (arm + component weights + format/length).
    The guard is on for training — an always-empty query cannot farm credit where gold is empty —
    and is passed through to the verifier.
    """

    guard_trivial_empty: bool = True


class SqlAnswerParser:
    """Extract the SQL from a completion and judge the trained `<think>` + ```sql``` format."""

    def extract(self, text: str) -> str | None:
        return extract_sql(text)

    def well_formed(self, text: str) -> bool:
        return well_formed(text)


SQL_PARSER = SqlAnswerParser()


class SqlVerifier:
    """Execution-match verifier as a harness `Verifier`: reference = gold SQL, context = db path.

    Raises `GoldExecutionError` (a `ReferenceUnverifiable`) when the gold query itself fails, so the
    generic reward records a `gold_error` without special-casing SQL.
    """

    def __init__(self, config: VerifierConfig | None = None) -> None:
        self._config = config or VerifierConfig()

    def verify(self, answer: str, reference: object, context: object) -> Verdict:
        result = execution_match(answer, str(reference), context, config=self._config)  # type: ignore[arg-type]
        return Verdict(
            matched=result.matched,
            valid=result.reason != "pred_error",
            reason=result.reason,
        )


def score_completion(
    text: str,
    gold_sql: str,
    db_path: str | Path,
    *,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Score one completion against gold on one database. Pure and side-effect-free.

    Executes the prediction once through the SQL verifier (read-only sandbox) and derives the
    `executable` and `result_match` components from the single verdict.
    """
    cfg = config or RewardConfig()
    verifier = SqlVerifier(VerifierConfig(guard_trivial_empty=cfg.guard_trivial_empty))
    return score_answer(
        text,
        gold_sql,
        db_path,
        parser=SQL_PARSER,
        verifier=verifier,
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

    def reward(completions, gold_sql, db_id, **_kwargs) -> list[float]:
        rewards: list[float] = []
        for completion, gold, dbid in zip(completions, gold_sql, db_id, strict=True):
            db_path = resolver(dbid, db_root)
            breakdown = score_completion(completion_text(completion), gold, db_path, config=cfg)
            rewards.append(breakdown.total)
        return rewards

    reward.__name__ = f"execution_reward_{cfg.arm}"
    return reward


class SqlTask:
    """The text-to-SQL `Task` (see `llm_rlvr.harness.Task`)."""

    name = "sql"
    parser: SqlAnswerParser = SQL_PARSER

    def make_verifier(self) -> Verifier:
        return SqlVerifier(VerifierConfig())

    def make_reward_fn(
        self,
        context_root: str | Path,
        *,
        config: RewardConfig | None = None,
        resolver: Callable[[str, str | Path], Path] = resolve_database,
    ) -> Callable[..., list[float]]:
        return make_reward_fn(context_root, config=config, resolver=resolver)

    def build_prompts(
        self,
        examples: Sequence[Example],
        resources: Mapping[str, DatabaseSchema],
        *,
        style: str = "cot",
        **_kwargs: object,
    ) -> list[Messages]:
        return build_prompts(examples, resources, style=style)

    def build_grpo_rows(
        self,
        examples: Sequence[Example],
        resources: Mapping[str, DatabaseSchema],
        *,
        style: str = "cot",
        **_kwargs: object,
    ) -> list[dict[str, object]]:
        """One row per example: the chat `prompt` plus the `gold_sql` + `db_id` the reward needs."""
        prompts = build_prompts(examples, resources, style=style)
        return [
            {"prompt": prompt, "gold_sql": example.gold_sql, "db_id": example.db_id}
            for example, prompt in zip(examples, prompts, strict=True)
        ]


def sql_task() -> SqlTask:
    """The text-to-SQL task instance."""
    return SqlTask()
