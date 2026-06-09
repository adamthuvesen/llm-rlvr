"""bird-dataset: the pinned BIRD-dev bundle fetches, verifies, and resolves (network; large)."""

from __future__ import annotations

import pytest

from text2sql_rlvr.data import ensure_bird_dev, load_bird_examples, resolve_database
from text2sql_rlvr.data.spider import load_schemas


@pytest.mark.integration
def test_bird_dev_fetch_and_resolve() -> None:
    bird = ensure_bird_dev()
    examples = load_bird_examples(bird.dev_json)
    schemas = load_schemas(bird.tables_json)
    assert len(examples) == 1534  # BIRD-dev (2024-06-27 release)
    assert len(schemas) == 11
    assert resolve_database(examples[0].db_id, bird.database_root).exists()
