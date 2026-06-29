"""Shared session fixtures for integration tests."""

from __future__ import annotations

import pytest

from llm_rlvr_sql.data.bird import ensure_bird_dev
from llm_rlvr_sql.data.spider import ensure_spider_databases


@pytest.fixture(scope="session")
def spider_db_root():
    return ensure_spider_databases().database_root


@pytest.fixture(scope="session")
def bird_db_root():
    return ensure_bird_dev().database_root
