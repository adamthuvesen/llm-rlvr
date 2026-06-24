"""baseline-evaluation: CompletionSet assembly and file round-trip."""

from __future__ import annotations

from pathlib import Path

from llm_rlvr_sql.data.models import Example
from llm_rlvr_sql.eval.completions import (
    assemble_completion_set,
    read_completion_set,
    write_completion_set,
)
from llm_rlvr_sql.eval.provenance import GenerationProvenance

EXAMPLE = Example(db_id="concert", question="how many?", gold_sql="SELECT a FROM t", split="dev")


def test_assemble_extracts_sql(provenance: GenerationProvenance) -> None:
    cset = assemble_completion_set([EXAMPLE], ["```sql\nSELECT a FROM t\n```"], provenance)
    assert cset.items[0].samples[0].sql == "SELECT a FROM t"


def test_assemble_length_mismatch_raises(provenance: GenerationProvenance) -> None:
    import pytest

    with pytest.raises(ValueError, match="examples but"):
        assemble_completion_set([EXAMPLE], [], provenance)


def test_completion_set_file_round_trip(provenance: GenerationProvenance, tmp_path: Path) -> None:
    cset = assemble_completion_set([EXAMPLE], ["```sql\nSELECT a FROM t\n```"], provenance)
    path = write_completion_set(cset, tmp_path / "runs" / "cs.json")
    assert read_completion_set(path) == cset
