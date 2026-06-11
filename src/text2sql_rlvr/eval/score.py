"""Score a `CompletionSet` into execution accuracy — deterministic and backend-free.

Takes the first sample per example (pass@1 baseline), runs the execution-match verifier against
the gold database, and aggregates. No model is loaded, so this runs on a CPU box and reproduces
the same number every time.

`execution_accuracy` is `n_correct / n` over all items: completions with no SQL and items where
the gold query itself errors both count as failures in the denominator. `n_gold_errors` is
reported separately and should be 0 on a healthy run (a non-zero count signals a broken
example/database pairing, not a model failure). `well_formed_rate` is the fraction of raw
completions in the trained `<think>` + ```sql``` shape — the cold-start's format metric,
independent of execution accuracy.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from text2sql_rlvr._base import Frozen
from text2sql_rlvr.data.spider import resolve_database
from text2sql_rlvr.eval.completions import CompletionSet, ProblemCompletions
from text2sql_rlvr.eval.parsing import well_formed
from text2sql_rlvr.eval.provenance import GenerationProvenance
from text2sql_rlvr.eval.verifier import GoldExecutionError, VerifierConfig, execution_match


class ScoreReport(Frozen):
    execution_accuracy: float
    well_formed_rate: float
    n: int
    n_correct: int
    n_gold_errors: int
    reasons: dict[str, int]
    verifier: VerifierConfig
    provenance: GenerationProvenance


def _score_item(
    item: ProblemCompletions, db_root: str | Path, cfg: VerifierConfig
) -> tuple[bool, str, bool]:
    """One item's verdict: (well_formed, reason, matched). Runs on any thread — the sandbox
    opens a fresh read-only connection per execution, so items are independent."""
    formed = bool(item.samples) and well_formed(item.samples[0].text)
    sql = item.samples[0].sql if item.samples else None
    if sql is None:
        return formed, "no_sql", False
    db_path = resolve_database(item.example.db_id, db_root)
    try:
        result = execution_match(sql, item.example.gold_sql, db_path, config=cfg)
    except GoldExecutionError:
        return formed, "gold_error", False
    return formed, result.reason, result.matched


def score_completion_set(
    completion_set: CompletionSet,
    db_root: str | Path,
    *,
    config: VerifierConfig | None = None,
    workers: int = 1,
) -> ScoreReport:
    """Aggregate pass@1 single-DB execution accuracy over a completion set.

    `workers` > 1 scores items on a thread pool (SQLite runs outside the GIL; BIRD's large
    databases dominate wall-clock). Verdicts are aggregated in item order, so the report is
    byte-identical at any worker count.
    """
    cfg = config or VerifierConfig()
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    if workers == 1:
        verdicts = [_score_item(item, db_root, cfg) for item in completion_set.items]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            verdicts = list(
                pool.map(lambda item: _score_item(item, db_root, cfg), completion_set.items)
            )

    reasons: Counter[str] = Counter()
    correct = 0
    gold_errors = 0
    n_well_formed = 0
    for formed, reason, matched in verdicts:
        if formed:
            n_well_formed += 1
        reasons[reason] += 1
        if reason == "gold_error":
            gold_errors += 1
        if matched:
            correct += 1

    n = len(completion_set.items)
    return ScoreReport(
        execution_accuracy=correct / n if n else 0.0,
        well_formed_rate=n_well_formed / n if n else 0.0,
        n=n,
        n_correct=correct,
        n_gold_errors=gold_errors,
        reasons=dict(reasons),
        verifier=cfg,
        provenance=completion_set.provenance,
    )
