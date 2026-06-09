"""baseline-evaluation: deterministic, model-free scoring of a CompletionSet."""

from __future__ import annotations

from pathlib import Path

from text2sql_rlvr.data.models import Example
from text2sql_rlvr.eval.completions import (
    CompletionSet,
    GenerationSample,
    ProblemCompletions,
)
from text2sql_rlvr.eval.provenance import GenerationProvenance
from text2sql_rlvr.eval.score import score_completion_set

EXAMPLE = Example(db_id="concert", question="q", gold_sql="SELECT a FROM t", split="dev")


def _completion_set(provenance: GenerationProvenance, *sqls: str | None) -> CompletionSet:
    items = tuple(
        ProblemCompletions(
            example=EXAMPLE,
            samples=(GenerationSample(text=sql or "", sql=sql),),
        )
        for sql in sqls
    )
    return CompletionSet(provenance=provenance, items=items)


def test_accuracy_and_reasons(provenance: GenerationProvenance, scoring_root: Path) -> None:
    cset = _completion_set(
        provenance,
        "SELECT a FROM t",  # correct
        "SELECT a FROM t WHERE a = 99",  # wrong (empty)
    )
    report = score_completion_set(cset, scoring_root)
    assert (report.n, report.n_correct) == (2, 1)
    assert report.execution_accuracy == 0.5
    assert report.reasons["match"] == 1
    assert report.reasons["denotation_mismatch"] == 1


def test_scoring_is_deterministic(provenance: GenerationProvenance, scoring_root: Path) -> None:
    cset = _completion_set(provenance, "SELECT a FROM t", "SELECT a FROM t WHERE a = 99")
    first = score_completion_set(cset, scoring_root).execution_accuracy
    second = score_completion_set(cset, scoring_root).execution_accuracy
    assert first == second


def test_missing_sql_counts_as_no_sql(provenance: GenerationProvenance, scoring_root: Path) -> None:
    report = score_completion_set(_completion_set(provenance, None), scoring_root)
    assert report.n_correct == 0
    assert report.reasons["no_sql"] == 1
