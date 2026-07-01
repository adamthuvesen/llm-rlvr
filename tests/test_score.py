"""baseline-evaluation: deterministic, model-free scoring of a CompletionSet."""

from __future__ import annotations

from pathlib import Path

from llm_rlvr.data.models import Example
from llm_rlvr.eval.completions import (
    CompletionSet,
    GenerationSample,
    ProblemCompletions,
)
from llm_rlvr.eval.provenance import GenerationProvenance
from llm_rlvr.eval.score import score_completion_set

EXAMPLE = Example(db_id="concert", question="q", gold_sql="SELECT a FROM t", split="dev")


def _completion_text(sql: str) -> str:
    return f"<think>x</think>\n```sql\n{sql}\n```"


def _completion_set(provenance: GenerationProvenance, *sqls: str | None) -> CompletionSet:
    items = tuple(
        ProblemCompletions(
            example=EXAMPLE,
            samples=(GenerationSample(text=_completion_text(sql) if sql is not None else ""),),
        )
        for sql in sqls
    )
    return CompletionSet(
        provenance=provenance.model_copy(update={"n_examples": len(items)}), items=items
    )


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


def test_well_formed_rate_counts_trained_format_only(
    provenance: GenerationProvenance, scoring_root: Path
) -> None:
    well = "<think>count</think>\n```sql\nSELECT a FROM t\n```"
    bare = "SELECT a FROM t"  # correct SQL, but not the trained shape
    items = tuple(
        ProblemCompletions(
            example=EXAMPLE,
            samples=(GenerationSample(text=text),),
        )
        for text in (well, bare)
    )
    report = score_completion_set(CompletionSet(provenance=provenance, items=items), scoring_root)
    assert report.well_formed_rate == 0.5
    assert report.execution_accuracy == 0.5


def test_raw_sql_completions_are_unparseable(
    provenance: GenerationProvenance, scoring_root: Path
) -> None:
    item = ProblemCompletions(
        example=EXAMPLE,
        samples=(GenerationSample(text="SELECT a FROM t"),),
    )
    report = score_completion_set(
        CompletionSet(provenance=provenance.model_copy(update={"n_examples": 1}), items=(item,)),
        scoring_root,
    )
    assert report.execution_accuracy == 0.0
    assert report.well_formed_rate == 0.0
    assert report.reasons["no_sql"] == 1


def test_parallel_scoring_is_byte_identical(
    provenance: GenerationProvenance, scoring_root: Path
) -> None:
    # 8.1: threaded scoring must aggregate in item order — same report at any worker count.
    cs = _completion_set(
        provenance,
        "SELECT a FROM t",
        "SELECT b FROM t",
        "SELECT nope FROM missing",
        None,
        "SELECT a FROM t WHERE a=99",
    )
    serial = score_completion_set(cs, scoring_root, workers=1)
    threaded = score_completion_set(cs, scoring_root, workers=4)
    assert serial == threaded


def test_workers_must_be_positive(provenance: GenerationProvenance, scoring_root: Path) -> None:
    cs = _completion_set(provenance, "SELECT a FROM t")
    try:
        score_completion_set(cs, scoring_root, workers=0)
    except ValueError as exc:
        assert "workers" in str(exc)
    else:
        raise AssertionError("workers=0 must be rejected")
