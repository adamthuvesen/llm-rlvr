"""baseline-evaluation: deterministic, model-free scoring of a CompletionSet."""

from __future__ import annotations

from pathlib import Path

import pytest

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


@pytest.mark.parametrize(
    "updates",
    [
        {"execution_accuracy": 1.0},
        {"n_correct": 3},
        {"n_gold_errors": 1},
        {"reasons": {"match": 1}},
        {"reasons": {"match": -1, "no_sql": 3}},
        {"well_formed_rate": 0.2},
        {"n": 3},
        {"execution_accuracy": float("nan")},
    ],
)
def test_report_rejects_inconsistent_counts(
    provenance: GenerationProvenance, scoring_root: Path, updates: dict[str, object]
) -> None:
    from llm_rlvr.eval.score import ScoreReport

    report = score_completion_set(
        _completion_set(provenance, "SELECT a FROM t", None), scoring_root
    )
    with pytest.raises(ValueError):
        ScoreReport.model_validate({**report.model_dump(), **updates})


def test_score_provenance_tracks_inputs_and_actual_database(
    provenance: GenerationProvenance, scoring_root: Path
) -> None:
    import hashlib

    first = score_completion_set(_completion_set(provenance, "SELECT a FROM t"), scoring_root)
    changed = score_completion_set(_completion_set(provenance, "SELECT b FROM t"), scoring_root)
    assert first.scoring is not None and changed.scoring is not None
    assert first.scoring.completion_set_sha256 != changed.scoring.completion_set_sha256
    assert first.scoring.scorer_policy
    assert len(first.scoring.scorer_source_sha256) == 64
    assert first.scoring.databases[0].db_id == "concert"
    assert (
        first.scoring.databases[0].sha256
        == hashlib.sha256((scoring_root / "concert/concert.sqlite").read_bytes()).hexdigest()
    )
    with pytest.raises(TypeError):
        first.reasons["match"] = 0  # type: ignore[index]


def test_old_score_report_without_scoring_provenance_still_reads(
    provenance: GenerationProvenance, scoring_root: Path
) -> None:
    from llm_rlvr.eval.score import ScoreReport

    report = score_completion_set(_completion_set(provenance, "SELECT a FROM t"), scoring_root)
    legacy = report.model_dump()
    legacy.pop("scoring")
    legacy.pop("outcomes")
    restored = ScoreReport.model_validate(legacy)
    assert restored.scoring is None
    assert restored.outcomes is None
    assert restored.execution_accuracy == 1


def test_no_sql_does_not_claim_database_identity(
    provenance: GenerationProvenance, tmp_path: Path
) -> None:
    report = score_completion_set(_completion_set(provenance, None), tmp_path / "missing")
    assert report.scoring is not None
    assert report.scoring.databases == ()


def test_gold_errors_stay_in_denominator_and_only_first_sample_is_scored(
    provenance: GenerationProvenance, scoring_root: Path
) -> None:
    broken = EXAMPLE.model_copy(update={"gold_sql": "SELECT missing FROM t"})
    items = (
        ProblemCompletions(
            example=EXAMPLE,
            samples=(
                GenerationSample(text=_completion_text("SELECT b FROM t")),
                GenerationSample(text=_completion_text("SELECT a FROM t")),
            ),
        ),
        ProblemCompletions(
            example=broken, samples=(GenerationSample(text=_completion_text("SELECT a FROM t")),)
        ),
    )
    report = score_completion_set(CompletionSet(provenance=provenance, items=items), scoring_root)
    assert report.n == 2
    assert report.execution_accuracy == 0
    assert report.n_gold_errors == 1
    assert report.reasons["gold_error"] == 1
    assert report.outcomes is not None
    assert report.outcomes[1].index == 1
    assert report.outcomes[1].db_id == "concert"
    assert report.outcomes[1].reason == "gold_error"
    assert "missing" in (report.outcomes[1].detail or "")
    assert "concert.sqlite" in (report.outcomes[1].detail or "")


def test_outcomes_and_aggregates_cannot_disagree(
    provenance: GenerationProvenance, scoring_root: Path
) -> None:
    from llm_rlvr.eval.score import ScoreReport

    report = score_completion_set(_completion_set(provenance, "SELECT a FROM t"), scoring_root)
    data = report.model_dump()
    data["outcomes"][0]["reason"] = "no_sql"
    data["outcomes"][0]["matched"] = False
    with pytest.raises(ValueError, match="reasons must match outcomes"):
        ScoreReport.model_validate(data)
