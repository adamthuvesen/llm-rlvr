"""sql-execution-sandbox: read-only, timeout, row cap, structured errors, multi-statement."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from llm_rlvr.env import ExecutionError, ExecutionResult, SqliteSandbox


def test_select_returns_rows(tiny_db: Path) -> None:
    out = SqliteSandbox(tiny_db).execute("SELECT a, b FROM t ORDER BY a")
    assert isinstance(out, ExecutionResult)
    assert out.rows == ((1, "x"), (2, "y"), (2, "y"))


def test_write_cannot_modify(tiny_db: Path) -> None:
    out = SqliteSandbox(tiny_db).execute("INSERT INTO t (a, b) VALUES (9, 'z')")
    assert isinstance(out, ExecutionError)
    assert out.error_type == "read_only"
    # The database on disk is unchanged.
    conn = sqlite3.connect(tiny_db)
    assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 3
    conn.close()


def test_row_cap_truncates_and_flags(tiny_db: Path) -> None:
    out = SqliteSandbox(tiny_db, row_cap=2).execute("SELECT a, b FROM t")
    assert isinstance(out, ExecutionResult)
    assert out.row_count == 2
    assert out.truncated is True


def test_timeout_interrupts_long_query(tiny_db: Path) -> None:
    # Counting to a billion forces a full scan before the single output row; the watchdog
    # interrupts it well before completion.
    slow = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 1000000000) "
        "SELECT COUNT(*) FROM c"
    )
    started = time.monotonic()
    out = SqliteSandbox(tiny_db, timeout_s=0.05).execute(slow)
    elapsed = time.monotonic() - started
    assert isinstance(out, ExecutionError)
    assert out.error_type == "timeout"
    assert elapsed < 1.0  # bounded, not a hang


def test_invalid_sql_is_structured_error(tiny_db: Path) -> None:
    out = SqliteSandbox(tiny_db).execute("SELECT nope FROM t")
    assert isinstance(out, ExecutionError)
    assert out.error_type == "syntax"


def test_multi_statement_rejected(tiny_db: Path) -> None:
    out = SqliteSandbox(tiny_db).execute("SELECT 1; SELECT 2")
    assert isinstance(out, ExecutionError)
    assert out.error_type == "multi_statement"


def test_database_path_uri_metacharacters_are_escaped(tmp_path: Path) -> None:
    db = tmp_path / "db#frag.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE t (a int)")
        conn.execute("INSERT INTO t VALUES (1)")

    out = SqliteSandbox(db).execute("SELECT a FROM t")

    assert isinstance(out, ExecutionResult)
    assert out.rows == ((1,),)
