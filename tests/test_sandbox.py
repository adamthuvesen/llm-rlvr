"""sql-execution-sandbox: read-only, timeout, row cap, structured errors, multi-statement."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

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
    assert out.status == "truncated"


def test_row_cap_exact_result_is_complete(tiny_db: Path) -> None:
    out = SqliteSandbox(tiny_db, row_cap=3).execute("SELECT a FROM t")
    assert isinstance(out, ExecutionResult)
    assert out.truncated is False
    assert out.status == "complete"


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
    db = tmp_path / "db #frag?query%25.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE t (a int)")
        conn.execute("INSERT INTO t VALUES (1)")

    out = SqliteSandbox(db).execute("SELECT a FROM t")

    assert isinstance(out, ExecutionResult)
    assert out.rows == ((1,),)


@pytest.mark.parametrize(
    "statement",
    [
        "ATTACH DATABASE '{target}' AS other",
        "VACUUM INTO '{target}'",
        "PRAGMA query_only = OFF",
        "PRAGMA writable_schema = ON",
        "PRAGMA journal_mode = DELETE",
        "CREATE TEMP TABLE temporary (a int)",
        "BEGIN TRANSACTION",
        "SELECT load_extension('{target}')",
    ],
)
def test_authorizer_denies_unsafe_actions(tiny_db: Path, tmp_path: Path, statement: str) -> None:
    target = tmp_path / "created.sqlite"
    before = tiny_db.read_bytes()

    out = SqliteSandbox(tiny_db).execute(statement.format(target=target))

    assert isinstance(out, ExecutionError)
    assert out.status == "error"
    assert out.error_type == "read_only"
    assert not target.exists()
    assert tiny_db.read_bytes() == before


def test_schema_pragma_remains_readable(tiny_db: Path) -> None:
    out = SqliteSandbox(tiny_db).execute("PRAGMA table_info(t)")
    assert isinstance(out, ExecutionResult)
    assert [row[1] for row in out.rows] == ["a", "b"]


def test_missing_database_is_structured_error_and_not_created(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sqlite"
    out = SqliteSandbox(missing).execute("SELECT 1")
    assert isinstance(out, ExecutionError)
    assert out.error_type == "operational"
    assert not missing.exists()


def test_corrupt_database_is_structured_error(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.sqlite"
    corrupt.write_text("not a sqlite database")
    out = SqliteSandbox(corrupt).execute("SELECT a FROM t")
    assert isinstance(out, ExecutionError)
    assert "not a database" in out.message


@pytest.mark.parametrize("timeout_s", [0, -1, float("nan"), float("inf"), True])
def test_invalid_timeout_rejected(tiny_db: Path, timeout_s: float) -> None:
    with pytest.raises(ValueError, match="timeout_s must be positive and finite"):
        SqliteSandbox(tiny_db, timeout_s=timeout_s)


@pytest.mark.parametrize("row_cap", [0, -1, 1.5, True])
def test_invalid_row_cap_rejected(tiny_db: Path, row_cap: int) -> None:
    with pytest.raises(ValueError, match="row_cap must be a positive integer"):
        SqliteSandbox(tiny_db, row_cap=row_cap)


@pytest.mark.parametrize("sql", ["SELECT a FROM t", "SELECT nope FROM t"])
def test_connection_closed_after_execution(
    tiny_db: Path, sql: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    connect = sqlite3.connect
    connections: list[sqlite3.Connection] = []

    def tracked_connect(*args, **kwargs) -> sqlite3.Connection:
        conn = connect(*args, **kwargs)
        connections.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    SqliteSandbox(tiny_db).execute(sql)

    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connections[0].execute("SELECT 1")
