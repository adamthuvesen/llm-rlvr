"""spider-dataset: the pinned DB archive fetches, verifies, and resolves (network; large)."""

from __future__ import annotations

import pytest

from llm_rlvr_sql.data import ensure_spider_databases, load_schemas, resolve_database


@pytest.mark.integration
def test_pinned_databases_fetch_and_resolve() -> None:
    dbs = ensure_spider_databases()
    schemas = load_schemas(dbs.tables_json)
    assert len(schemas) == 166  # Spider train+dev databases
    db_path = resolve_database("concert_singer", dbs.database_root)
    assert db_path.exists()
