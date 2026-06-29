"""Forced-revision agent metrics — fix/harm decomposition of one revision pass (CPU).

Re-derivable from the committed eval artifacts: the CompletionSet supplies gold (and the final
verdicts `make eval` already reproduces); the transcripts supply the attempt and feedback. Each
episode is classified by (attempt verdict, final verdict):

    fixed   = wrong  → right     (the self-correction the change exists to train)
    harmed  = right  → wrong     (the SCoRe harm channel)
    kept    = verdict unchanged

Rates use the probe's denominators (fix rate over attempt-wrong, harm rate over attempt-right)
so the trained numbers compare directly against the untrained probe (fix 14.9%, harm 6.3%).
"""

from __future__ import annotations

from pathlib import Path

from llm_rlvr._base import Frozen
from llm_rlvr.data.paths import resolve_database
from llm_rlvr.eval.completions import CompletionSet
from llm_rlvr.eval.forced_revision_generate import (
    ForcedRevisionPolicy,
    ForcedRevisionTranscriptSet,
)
from llm_rlvr.eval.parsing import extract_sql
from llm_rlvr.eval.verifier import VerifierConfig, execution_match
from llm_rlvr.revision.feedback import is_error_feedback


class FrMetricsReport(Frozen):
    """The revision pass decomposed: where the final EX came from."""

    n: int
    n_skipped_gold_error: int
    attempt_ex: float  # EX if the attempt had been the final answer (turn-1 quality)
    final_ex: float  # EX of the revision (matches the CompletionSet score)
    n_attempt_wrong: int
    n_fixed: int
    n_fixed_from_error_feedback: int
    n_fixed_from_result_feedback: int
    fix_rate: float  # n_fixed / n_attempt_wrong
    n_attempt_right: int
    n_harmed: int
    harm_rate: float  # n_harmed / n_attempt_right
    n_revised: int  # final SQL differs from attempt SQL
    net_delta: float  # final_ex - attempt_ex


class ForcedRevisionEvalReport(Frozen):
    """Run-level accounting for a fresh forced-revision inference artifact."""

    policy: ForcedRevisionPolicy
    n: int
    n_skipped_gold_error: int
    attempt_ex: float
    final_ex: float
    n_attempt_wrong: int
    n_fixed: int
    n_fixed_from_error_feedback: int
    n_fixed_from_result_feedback: int
    fix_rate: float
    n_attempt_right: int
    n_harmed: int
    harm_rate: float
    n_sql_changed: int
    net_delta: float
    n_error_feedback: int
    n_second_generation: int
    second_generation_rate: float
    n_kept_attempt: int
    runtime_notes: tuple[str, ...] = ()
    cost_notes: tuple[str, ...] = ()


def _eval_report_from_metrics(
    metrics: FrMetricsReport,
    *,
    policy: ForcedRevisionPolicy,
    n_error_feedback: int,
    n_second_generation: int,
    n_items: int,
) -> ForcedRevisionEvalReport:
    return ForcedRevisionEvalReport(
        policy=policy,
        n=metrics.n,
        n_skipped_gold_error=metrics.n_skipped_gold_error,
        attempt_ex=metrics.attempt_ex,
        final_ex=metrics.final_ex,
        n_attempt_wrong=metrics.n_attempt_wrong,
        n_fixed=metrics.n_fixed,
        n_fixed_from_error_feedback=metrics.n_fixed_from_error_feedback,
        n_fixed_from_result_feedback=metrics.n_fixed_from_result_feedback,
        fix_rate=metrics.fix_rate,
        n_attempt_right=metrics.n_attempt_right,
        n_harmed=metrics.n_harmed,
        harm_rate=metrics.harm_rate,
        n_sql_changed=metrics.n_revised,
        net_delta=metrics.net_delta,
        n_error_feedback=n_error_feedback,
        n_second_generation=n_second_generation,
        second_generation_rate=n_second_generation / n_items if n_items else 0.0,
        n_kept_attempt=n_items - n_second_generation,
    )


def compute_fr_metrics(
    completion_set: CompletionSet,
    transcripts: ForcedRevisionTranscriptSet,
    db_root: str | Path,
    *,
    verifier: VerifierConfig | None = None,
) -> FrMetricsReport:
    """Score attempt and final per episode; aggregate the fix/harm decomposition."""
    if len(completion_set.items) != len(transcripts.items):
        raise ValueError(
            f"{len(completion_set.items)} completions but {len(transcripts.items)} transcripts"
        )
    config = verifier or VerifierConfig()

    n_skipped = 0
    n_wrong = n_fixed = n_fixed_err = n_fixed_res = 0
    n_right = n_harmed = n_revised = 0
    attempt_correct = final_correct = 0
    n = 0
    for item, episode in zip(completion_set.items, transcripts.items, strict=True):
        example = item.example
        if example.db_id != episode.db_id or example.question != episode.question:
            raise ValueError("completion set and transcripts are not aligned")
        db_path = resolve_database(example.db_id, db_root)
        attempt_sql = extract_sql(episode.attempt_text)
        final_sql = item.samples[0].sql

        def verdict(sql: str | None, gold: str = example.gold_sql, db: Path = db_path) -> bool:
            if sql is None:
                return False
            return execution_match(sql, gold, db, config=config).matched

        try:
            attempt_ok = verdict(attempt_sql)
            final_ok = verdict(final_sql)
        except Exception:  # GoldExecutionError and kin — nothing to decompose on this item
            n_skipped += 1
            continue

        n += 1
        attempt_correct += attempt_ok
        final_correct += final_ok
        has_error_feedback = is_error_feedback(episode.feedback)
        if attempt_sql is not None and final_sql is not None:
            n_revised += attempt_sql.strip() != final_sql.strip()
        elif attempt_sql != final_sql:
            n_revised += 1
        if not attempt_ok:
            n_wrong += 1
            if final_ok:
                n_fixed += 1
                if has_error_feedback:
                    n_fixed_err += 1
                else:
                    n_fixed_res += 1
        else:
            n_right += 1
            if not final_ok:
                n_harmed += 1

    return FrMetricsReport(
        n=n,
        n_skipped_gold_error=n_skipped,
        attempt_ex=attempt_correct / n if n else 0.0,
        final_ex=final_correct / n if n else 0.0,
        n_attempt_wrong=n_wrong,
        n_fixed=n_fixed,
        n_fixed_from_error_feedback=n_fixed_err,
        n_fixed_from_result_feedback=n_fixed_res,
        fix_rate=n_fixed / n_wrong if n_wrong else 0.0,
        n_attempt_right=n_right,
        n_harmed=n_harmed,
        harm_rate=n_harmed / n_right if n_right else 0.0,
        n_revised=n_revised,
        net_delta=(final_correct - attempt_correct) / n if n else 0.0,
    )


def compute_forced_revision_eval_report(
    completion_set: CompletionSet,
    transcripts: ForcedRevisionTranscriptSet,
    db_root: str | Path,
    *,
    policy: ForcedRevisionPolicy,
    verifier: VerifierConfig | None = None,
) -> ForcedRevisionEvalReport:
    """Combine EX/fix/harm metrics with second-generation accounting."""
    metrics = compute_fr_metrics(completion_set, transcripts, db_root, verifier=verifier)
    n_items = len(transcripts.items)
    n_second_generation = sum(episode.revision_requested for episode in transcripts.items)
    n_error_feedback = sum(is_error_feedback(episode.feedback) for episode in transcripts.items)
    if policy == "error_only" and n_second_generation != n_error_feedback:
        raise ValueError(
            "error_only report accounting mismatch: "
            f"{n_second_generation} second generations for {n_error_feedback} error feedbacks"
        )
    return _eval_report_from_metrics(
        metrics,
        policy=policy,
        n_error_feedback=n_error_feedback,
        n_second_generation=n_second_generation,
        n_items=n_items,
    )
