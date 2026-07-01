"""Forced-revision agent metrics — fix/harm decomposition of one revision pass (CPU).

Re-derivable from saved eval artifacts: the CompletionSet supplies gold (and the final
verdicts `make eval` already reproduces); the transcripts supply the attempt and feedback. Each
episode is classified by (attempt verdict, final verdict):

    fixed   = wrong  → right     (the self-correction the change exists to train)
    harmed  = right  → wrong     (the SCoRe harm channel)
    kept    = verdict unchanged

Rates use the probe's denominators (fix rate over attempt-wrong, harm rate over attempt-right)
so the trained numbers compare directly against the untrained probe (fix 14.9%, harm 6.3%).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm_rlvr._base import Frozen
from llm_rlvr.data.paths import resolve_database
from llm_rlvr.eval.completions import CompletionSet, ProblemCompletions
from llm_rlvr.eval.forced_revision_generate import (
    ForcedRevisionEpisode,
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


@dataclass(frozen=True)
class _EpisodeScore:
    attempt_ok: bool
    final_ok: bool
    revised: bool
    error_feedback: bool


@dataclass
class _FrCounters:
    n_skipped: int = 0
    n_wrong: int = 0
    n_fixed: int = 0
    n_fixed_err: int = 0
    n_fixed_res: int = 0
    n_right: int = 0
    n_harmed: int = 0
    n_revised: int = 0
    attempt_correct: int = 0
    final_correct: int = 0

    @property
    def n(self) -> int:
        return self.n_wrong + self.n_right

    def add(self, score: _EpisodeScore) -> None:
        self.attempt_correct += score.attempt_ok
        self.final_correct += score.final_ok
        self.n_revised += score.revised

        if score.attempt_ok:
            self.n_right += 1
            self.n_harmed += not score.final_ok
            return

        self.n_wrong += 1
        if score.final_ok:
            self.n_fixed += 1
            if score.error_feedback:
                self.n_fixed_err += 1
            else:
                self.n_fixed_res += 1

    def report(self) -> FrMetricsReport:
        n = self.n
        return FrMetricsReport(
            n=n,
            n_skipped_gold_error=self.n_skipped,
            attempt_ex=self.attempt_correct / n if n else 0.0,
            final_ex=self.final_correct / n if n else 0.0,
            n_attempt_wrong=self.n_wrong,
            n_fixed=self.n_fixed,
            n_fixed_from_error_feedback=self.n_fixed_err,
            n_fixed_from_result_feedback=self.n_fixed_res,
            fix_rate=self.n_fixed / self.n_wrong if self.n_wrong else 0.0,
            n_attempt_right=self.n_right,
            n_harmed=self.n_harmed,
            harm_rate=self.n_harmed / self.n_right if self.n_right else 0.0,
            n_revised=self.n_revised,
            net_delta=(self.final_correct - self.attempt_correct) / n if n else 0.0,
        )


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


def _assert_aligned(episode: ForcedRevisionEpisode, *, db_id: str, question: str) -> None:
    if db_id != episode.db_id or question != episode.question:
        raise ValueError("completion set and transcripts are not aligned")


def _sql_changed(attempt_sql: str | None, final_sql: str | None) -> bool:
    if attempt_sql is not None and final_sql is not None:
        return attempt_sql.strip() != final_sql.strip()
    return attempt_sql != final_sql


def _execution_ok(
    sql: str | None,
    *,
    gold_sql: str,
    db_path: Path,
    config: VerifierConfig,
) -> bool:
    if sql is None:
        return False
    return execution_match(sql, gold_sql, db_path, config=config).matched


def _score_episode(
    item: ProblemCompletions,
    episode: ForcedRevisionEpisode,
    db_root: str | Path,
    config: VerifierConfig,
) -> _EpisodeScore | None:
    example = item.example
    _assert_aligned(episode, db_id=example.db_id, question=example.question)
    db_path = resolve_database(example.db_id, db_root)
    attempt_sql = extract_sql(episode.attempt_text)
    final_sql = item.samples[0].sql

    try:
        attempt_ok = _execution_ok(
            attempt_sql, gold_sql=example.gold_sql, db_path=db_path, config=config
        )
        final_ok = _execution_ok(
            final_sql, gold_sql=example.gold_sql, db_path=db_path, config=config
        )
    except Exception:  # GoldExecutionError and kin — nothing to decompose on this item
        return None

    return _EpisodeScore(
        attempt_ok=attempt_ok,
        final_ok=final_ok,
        revised=_sql_changed(attempt_sql, final_sql),
        error_feedback=is_error_feedback(episode.feedback),
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

    counters = _FrCounters()
    for item, episode in zip(completion_set.items, transcripts.items, strict=True):
        score = _score_episode(item, episode, db_root, config)
        if score is None:
            counters.n_skipped += 1
            continue
        counters.add(score)

    return counters.report()


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
