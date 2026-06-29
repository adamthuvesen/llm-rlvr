"""Pre-registered revision probe: untrained Stage-2 policy fix/harm rates (Stage 3b)."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path

from llm_rlvr_sql._base import Frozen
from llm_rlvr_sql.data.models import DatabaseSchema
from llm_rlvr_sql.data.paths import resolve_database, sha256_of
from llm_rlvr_sql.data.schema import serialize_schema
from llm_rlvr_sql.env.sandbox import SqliteSandbox
from llm_rlvr_sql.env.text2sql_env import (
    AGENT_ROW_CAP,
    AGENT_TIMEOUT_S,
    FeedbackConfig,
    format_feedback,
)
from llm_rlvr_sql.eval.completions import CompletionSet
from llm_rlvr_sql.eval.parsing import extract_sql
from llm_rlvr_sql.eval.verifier import VerifierConfig, execution_match
from llm_rlvr_sql.revision.feedback import is_error_feedback

REVISE_INSTRUCTION = (
    "You are an expert data analyst. Given a SQLite database schema, a question, your previous "
    "query, and the feedback from executing that query, reconsider your answer. If the query "
    "errored or its result does not answer the question, revise it; if it is already correct, "
    "you may keep it. Reason step by step inside <think></think>, then give the final query in "
    "a ```sql ... ``` code block."
)

# Pre-registered gates (grpo-forced-revision tasks.md, task 1.1 — committed before the run).
G1_FIX_RATE = 0.10
G2_HARM_RATE = 0.10
RIGHT_SAMPLE_SIZE = 300


class RevisionPromptRow(Frozen):
    """One probe row: the item's prior state and the revision prompt the model will see."""

    item_id: int
    db_id: str
    question: str
    # Scoring-only metadata (the same pattern as the GRPO rows' gold_sql) — never in `prompt`.
    gold_sql: str
    was_correct: bool
    previous_sql: str
    feedback: str
    prompt: tuple[dict[str, object], ...]


class RevisionProbeReport(Frozen):
    """The probe's evidence, scored against the pre-registered gates."""

    source_completions: str
    source_sha256: str
    n_source: int
    n_skipped: int  # unparseable SQL or gold execution errors — excluded, counted
    n_wrong: int
    n_right: int
    n_right_sampled: int
    n_fixed: int
    n_fixed_from_error_feedback: int
    n_fixed_from_result_feedback: int
    n_wrong_with_error_feedback: int
    n_harmed: int
    fix_rate: float
    harm_rate: float
    # Net effect of one forced revision pass over the full source set, with the harm rate
    # extrapolated from the right-answer sample to all right answers. Context, not gated.
    net_ex_delta_estimate: float
    g1_fix_rate_threshold: float
    g2_harm_rate_threshold: float
    g1_passed: bool
    g2_passed: bool
    passed: bool
    git_commit: str = "unknown"


def _revision_prompt(
    schema_text: str, question: str, previous_sql: str, feedback: str
) -> tuple[dict[str, object], ...]:
    user = (
        f"Database schema:\n{schema_text}\n\nQuestion: {question}\n\n"
        f"Your previous query:\n```sql\n{previous_sql}\n```\n\n{feedback}"
    )
    return (
        {"role": "system", "content": REVISE_INSTRUCTION},
        {"role": "user", "content": user},
    )


def build_revision_prompts(
    completion_set: CompletionSet,
    schemas: Mapping[str, DatabaseSchema],
    db_root: str | Path,
    *,
    right_sample_size: int = RIGHT_SAMPLE_SIZE,
    seed: int = 0,
    feedback_config: FeedbackConfig | None = None,
    verifier: VerifierConfig | None = None,
) -> tuple[tuple[RevisionPromptRow, ...], int]:
    """All wrong items + a seeded sample of right items, each with real execution feedback.

    Returns (rows, n_skipped). Verdicts use the committed verifier defaults so the wrong/right
    split re-derives the published Stage-2 number; feedback uses the env's exact caps and
    formatter, so the probe shows the model what the training loop would.
    """
    fb_config = feedback_config or FeedbackConfig()
    verifier_config = verifier or VerifierConfig()
    wrong: list[RevisionPromptRow] = []
    right: list[RevisionPromptRow] = []
    n_skipped = 0
    for item_id, item in enumerate(completion_set.items):
        example = item.example
        sql = item.samples[0].sql
        if sql is None:
            n_skipped += 1
            continue
        db_path = resolve_database(example.db_id, db_root)
        try:
            matched = execution_match(
                sql, example.gold_sql, db_path, config=verifier_config
            ).matched
        except Exception:  # GoldExecutionError and kin: nothing to probe on this item
            n_skipped += 1
            continue
        sandbox = SqliteSandbox(db_path, timeout_s=AGENT_TIMEOUT_S, row_cap=AGENT_ROW_CAP)
        feedback = format_feedback(sandbox.execute(sql), fb_config)
        row = RevisionPromptRow(
            item_id=item_id,
            db_id=example.db_id,
            question=example.question,
            gold_sql=example.gold_sql,
            was_correct=matched,
            previous_sql=sql,
            feedback=feedback,
            prompt=_revision_prompt(
                serialize_schema(schemas[example.db_id]), example.question, sql, feedback
            ),
        )
        (right if matched else wrong).append(row)

    rng = random.Random(seed)
    sampled_right = (
        rng.sample(right, right_sample_size) if len(right) > right_sample_size else right
    )
    return (*wrong, *sampled_right), n_skipped


def write_prompt_rows(rows: Sequence[RevisionPromptRow], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(row.model_dump_json() for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    return out


def read_prompt_rows(path: str | Path) -> tuple[RevisionPromptRow, ...]:
    return tuple(
        RevisionPromptRow.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
    )


def score_revision_probe(
    rows: Sequence[RevisionPromptRow],
    completions: Sequence[str],
    db_root: str | Path,
    *,
    source_completions: str | Path,
    n_source: int,
    n_right_total: int,
    n_skipped: int,
    verifier: VerifierConfig | None = None,
    commit: str | None = None,
) -> RevisionProbeReport:
    """Score one revision per row against gold and judge the pre-registered gates."""
    verifier_config = verifier or VerifierConfig()
    if len(rows) != len(completions):
        raise ValueError(f"{len(rows)} rows but {len(completions)} completions")

    n_wrong = sum(1 for r in rows if not r.was_correct)
    n_sampled_right = sum(1 for r in rows if r.was_correct)
    n_fixed = n_fixed_err = n_fixed_res = n_harmed = n_wrong_err = 0
    for row, text in zip(rows, completions, strict=True):
        new_sql = extract_sql(text)
        has_error_feedback = is_error_feedback(row.feedback)
        if not row.was_correct and has_error_feedback:
            n_wrong_err += 1
        try:
            matched = (
                execution_match(
                    new_sql,
                    row.gold_sql,
                    resolve_database(row.db_id, db_root),
                    config=verifier_config,
                ).matched
                if new_sql is not None
                else False
            )
        except Exception:
            matched = False
        if not row.was_correct and matched:
            n_fixed += 1
            if has_error_feedback:
                n_fixed_err += 1
            else:
                n_fixed_res += 1
        if row.was_correct and not matched:
            n_harmed += 1

    fix_rate = n_fixed / n_wrong if n_wrong else 0.0
    harm_rate = n_harmed / n_sampled_right if n_sampled_right else 0.0
    net = (n_fixed - harm_rate * n_right_total) / n_source if n_source else 0.0
    g1 = fix_rate >= G1_FIX_RATE
    g2 = harm_rate <= G2_HARM_RATE
    from llm_rlvr_sql.eval.provenance import git_commit

    return RevisionProbeReport(
        source_completions=str(source_completions),
        source_sha256=sha256_of(source_completions),
        n_source=n_source,
        n_skipped=n_skipped,
        n_wrong=n_wrong,
        n_right=n_right_total,
        n_right_sampled=n_sampled_right,
        n_fixed=n_fixed,
        n_fixed_from_error_feedback=n_fixed_err,
        n_fixed_from_result_feedback=n_fixed_res,
        n_wrong_with_error_feedback=n_wrong_err,
        n_harmed=n_harmed,
        fix_rate=fix_rate,
        harm_rate=harm_rate,
        net_ex_delta_estimate=net,
        g1_fix_rate_threshold=G1_FIX_RATE,
        g2_harm_rate_threshold=G2_HARM_RATE,
        g1_passed=g1,
        g2_passed=g2,
        passed=g1 and g2,
        git_commit=commit if commit is not None else git_commit(),
    )
