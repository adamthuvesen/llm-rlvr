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
from pathlib import Path

from text2sql_rlvr._base import Frozen
from text2sql_rlvr.data.spider import resolve_database
from text2sql_rlvr.eval.completions import CompletionSet
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


def score_completion_set(
    completion_set: CompletionSet,
    db_root: str | Path,
    *,
    config: VerifierConfig | None = None,
) -> ScoreReport:
    """Aggregate pass@1 single-DB execution accuracy over a completion set."""
    cfg = config or VerifierConfig()
    reasons: Counter[str] = Counter()
    correct = 0
    gold_errors = 0
    n_well_formed = 0

    for item in completion_set.items:
        if item.samples and well_formed(item.samples[0].text):
            n_well_formed += 1
        sql = item.samples[0].sql if item.samples else None
        if sql is None:
            reasons["no_sql"] += 1
            continue
        db_path = resolve_database(item.example.db_id, db_root)
        try:
            result = execution_match(sql, item.example.gold_sql, db_path, config=cfg)
        except GoldExecutionError:
            gold_errors += 1
            reasons["gold_error"] += 1
            continue
        reasons[result.reason] += 1
        if result.matched:
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
