"""spider-dataset: live load at the pinned revision (network — deselected by default)."""

from __future__ import annotations

import pytest

from llm_rlvr_sql.data.spider import load_examples


@pytest.mark.integration
def test_dev_split_loads_typed_examples_at_pinned_revision() -> None:
    examples = load_examples("dev")
    # Spider dev has 1034 examples; assert the pin yields the published size and typed fields.
    assert len(examples) == 1034
    first = examples[0]
    assert first.db_id and first.question and first.gold_sql
    assert first.split == "dev"
