"""baseline-evaluation: CompletionSet assembly and file round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_rlvr.artifacts.provenance import GenerationProvenance
from llm_rlvr.data.models import Example
from llm_rlvr.eval.completions import (
    CompletionSet,
    GenerationSample,
    ProblemCompletions,
    assemble_completion_set,
    read_completion_set,
    write_completion_set,
)

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


def test_generation_sample_rejects_sql_cache_without_extracted_sql() -> None:
    with pytest.raises(ValueError, match="no extracted SQL"):
        GenerationSample(text="SELECT a FROM t", sql="SELECT a FROM t")


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


@pytest.mark.parametrize("field,value", [("dataset", "bird"), ("split", "train")])
def test_completion_metadata_must_match_examples(
    provenance: GenerationProvenance, field: str, value: str
) -> None:
    item = ProblemCompletions(
        example=EXAMPLE.model_copy(update={field: value}),
        samples=(GenerationSample(text=""),),
    )
    with pytest.raises(ValueError, match=field):
        CompletionSet(provenance=_provenance(provenance), items=(item,))


def test_existing_completion_output_requires_explicit_overwrite(
    provenance: GenerationProvenance, tmp_path: Path
) -> None:
    original = assemble_completion_set([EXAMPLE], [""], _provenance(provenance))
    changed = assemble_completion_set([EXAMPLE], ["different"], _provenance(provenance))
    path = write_completion_set(original, tmp_path / "completions.json")
    with pytest.raises(FileExistsError, match="already exists"):
        write_completion_set(changed, path)
    assert read_completion_set(path) == original
    write_completion_set(changed, path, overwrite=True)
    assert read_completion_set(path) == changed


def test_managed_completion_input_requires_publication_and_intact_hashes(
    provenance: GenerationProvenance, tmp_path: Path
) -> None:
    from llm_rlvr.artifacts.runs import begin_run, publish_complete

    root = tmp_path / "run"
    begin_run(root, kind="generation", request=provenance)
    completion_set = assemble_completion_set([EXAMPLE], [""], _provenance(provenance))
    path = write_completion_set(completion_set, root / "completions.json")
    with pytest.raises(ValueError, match="incomplete"):
        read_completion_set(path)
    publish_complete(root, artifacts=["completions.json"])
    assert read_completion_set(path) == completion_set
    path.write_text("tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        read_completion_set(path)


def test_v1_identity_matches_fixed_legacy_fixture_and_round_trip() -> None:
    fixture = Path(__file__).with_name("fixtures") / "completion_identity_v1.json"
    restored = CompletionSet.model_validate_json(fixture.read_text())
    expected = "ccb5f51045cbc7952f060db64a56c71a1d405388204cdf75c0bbb97d0e01524e"
    assert restored.schema_version == 1
    assert restored.canonical_sha256 == expected
    assert (
        CompletionSet.model_validate_json(restored.model_dump_json()).canonical_sha256 == expected
    )


def test_future_optional_defaults_do_not_change_v1_identity() -> None:
    class FutureProvenance(GenerationProvenance):
        future_note: str = "new default"

    class FutureCompletionSet(CompletionSet):
        provenance: FutureProvenance  # pyright: ignore[reportIncompatibleVariableOverride]  # narrowing the field on purpose
        future_note: str = "new default"

    fixture = Path(__file__).with_name("fixtures") / "completion_identity_v1.json"
    original = CompletionSet.model_validate_json(fixture.read_text())
    future = FutureCompletionSet.model_validate_json(fixture.read_text())
    assert future.canonical_sha256 == original.canonical_sha256
    with pytest.raises(ValueError, match="schema_version"):
        CompletionSet.model_validate({**original.model_dump(), "schema_version": 2})
