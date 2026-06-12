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

from text2sql_rlvr._base import Frozen
from text2sql_rlvr.data.spider import resolve_database
from text2sql_rlvr.eval.completions import CompletionSet
from text2sql_rlvr.eval.forced_revision_generate import ForcedRevisionTranscriptSet
from text2sql_rlvr.eval.parsing import extract_sql
from text2sql_rlvr.eval.verifier import VerifierConfig, execution_match


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
        is_error_feedback = episode.feedback.startswith("Execution feedback: error")
        if attempt_sql is not None and final_sql is not None:
            n_revised += attempt_sql.strip() != final_sql.strip()
        elif attempt_sql != final_sql:
            n_revised += 1
        if not attempt_ok:
            n_wrong += 1
            if final_ok:
                n_fixed += 1
                if is_error_feedback:
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
