"""baseline-evaluation: CompletionSet assembly and file round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_rlvr.data.models import Example
from llm_rlvr.eval.completions import (
    CompletionSet,
    GenerationSample,
    ProblemCompletions,
    assemble_completion_set,
    read_completion_set,
    write_completion_set,
)
from llm_rlvr.eval.provenance import GenerationProvenance

EXAMPLE = Example(db_id="concert", question="how many?", gold_sql="SELECT a FROM t", split="dev")


def _provenance(provenance: GenerationProvenance, n_examples: int = 1) -> GenerationProvenance:
    return provenance.model_copy(update={"n_examples": n_examples})


def test_assemble_extracts_sql(provenance: GenerationProvenance) -> None:
    cset = assemble_completion_set(
        [EXAMPLE], ["```sql\nSELECT a FROM t\n```"], _provenance(provenance)
    )
    assert cset.items[0].samples[0].sql == "SELECT a FROM t"


def test_assemble_length_mismatch_raises(provenance: GenerationProvenance) -> None:
    with pytest.raises(ValueError, match="examples but"):
        assemble_completion_set([EXAMPLE], [], provenance)


def test_completion_set_file_round_trip(provenance: GenerationProvenance, tmp_path: Path) -> None:
    cset = assemble_completion_set(
        [EXAMPLE], ["```sql\nSELECT a FROM t\n```"], _provenance(provenance)
    )
    path = write_completion_set(cset, tmp_path / "runs" / "cs.json")
    assert read_completion_set(path) == cset


def test_generation_sample_derives_sql_from_text() -> None:
    sample = GenerationSample(text="<think>x</think>\n```sql\nSELECT a FROM t\n```")
    assert sample.sql == "SELECT a FROM t"


def test_generation_sample_rejects_mismatched_sql_cache() -> None:
    with pytest.raises(ValueError, match="does not match"):
        GenerationSample(text="```sql\nSELECT a FROM t\n```", sql="SELECT b FROM t")


def test_generation_sample_accepts_legacy_raw_sql_text() -> None:
    sample = GenerationSample(text="SELECT a FROM t", sql="SELECT a FROM t")
    assert sample.sql == "SELECT a FROM t"


def test_problem_completions_requires_at_least_one_sample() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        ProblemCompletions(example=EXAMPLE, samples=())


def test_completion_set_count_must_match_provenance(provenance: GenerationProvenance) -> None:
    item = ProblemCompletions(
        example=EXAMPLE,
        samples=(GenerationSample(text="```sql\nSELECT a FROM t\n```"),),
    )
    with pytest.raises(ValueError, match=r"provenance\.n_examples"):
        CompletionSet(provenance=_provenance(provenance, n_examples=2), items=(item,))
