"""Shared fixtures: tiny in-tree SQLite databases (no network)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def _build(path: Path, rows: list[tuple[int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE t (a INTEGER, b TEXT);")
    conn.executemany("INSERT INTO t (a, b) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()


@pytest.fixture
def tiny_db(tmp_path: Path) -> Path:
    """A one-table DB in its own directory: rows (1,'x'), (2,'y'), (2,'y')."""
    db = tmp_path / "db" / "t.sqlite"
    _build(db, [(1, "x"), (2, "y"), (2, "y")])
    return db


@pytest.fixture
def test_suite_dir(tmp_path: Path) -> Path:
    """A directory of two DB instances with different contents (a distilled test suite)."""
    d = tmp_path / "suite"
    _build(d / "inst0.sqlite", [(1, "x"), (2, "y")])
    _build(d / "inst1.sqlite", [(5, "p"), (6, "q"), (7, "r")])
    return d
