"""Read-only SQLite sandbox — the safety boundary for every SQL execution.

Opens databases `mode=ro` (writes fail), enforces a wall-clock statement timeout via a watchdog
that calls `Connection.interrupt()`, caps materialized rows, rejects multi-statement input, and
returns structured results/errors instead of raising. No filesystem or network reach from SQL.
"""

from __future__ import annotations

import sqlite3
import threading
from math import isfinite
from pathlib import Path
from typing import Any, Literal

import sqlparse

from llm_rlvr._base import Frozen

ErrorType = Literal["syntax", "operational", "read_only", "timeout", "multi_statement"]


class ExecutionResult(Frozen):
    """A successful read-only execution: the rows (capped) and whether the cap was hit."""

    rows: tuple[tuple[Any, ...], ...]
    truncated: bool = False

    @property
    def status(self) -> Literal["complete", "truncated"]:
        return "truncated" if self.truncated else "complete"

    @property
    def row_count(self) -> int:
        return len(self.rows)


class ExecutionError(Frozen):
    """A failed execution, classified — never a raw traceback to the caller."""

    error_type: ErrorType
    message: str

    @property
    def status(self) -> Literal["error"]:
        return "error"


ExecutionOutcome = ExecutionResult | ExecutionError

_READ_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE}
_SCHEMA_PRAGMAS = {
    "table_info",
    "table_xinfo",
    "table_list",
    "index_info",
    "index_xinfo",
    "index_list",
    "foreign_key_list",
}
_READ_PRAGMAS = {"database_list", "compile_options", "query_only"}


def _authorize_read(
    action: int,
    arg1: str | None,
    arg2: str | None,
    database: str | None,
    source: str | None,
) -> int:
    if action in _READ_ACTIONS:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() not in {
        "load_extension",
        "readfile",
        "writefile",
    }:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA:
        name = (arg1 or "").lower()
        if name in _SCHEMA_PRAGMAS or (name in _READ_PRAGMAS and arg2 is None):
            return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _count_statements(sql: str) -> int:
    """Number of non-empty SQL statements (a bare comment counts as zero)."""
    return sum(1 for s in sqlparse.split(sql) if s.strip())


class SqliteSandbox:
    """Execute single read-only queries against one SQLite database.

    A fresh read-only connection is opened per `execute` call, so the sandbox is stateless and
    safe to reuse across examples. `row_cap` should be large for verification (full denotations)
    and small for agent-facing rollouts (the model only needs to see a truncated preview).
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        timeout_s: float = 30.0,
        row_cap: int = 5000,
    ) -> None:
        if isinstance(timeout_s, bool) or not isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        if isinstance(row_cap, bool) or not isinstance(row_cap, int) or row_cap <= 0:
            raise ValueError("row_cap must be a positive integer")
        self.db_path = Path(db_path)
        self.timeout_s = timeout_s
        self.row_cap = row_cap

    def execute(self, sql: str) -> ExecutionOutcome:
        if _count_statements(sql) > 1:
            return ExecutionError(
                error_type="multi_statement",
                message="only a single SQL statement may be executed",
            )

        uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
        conn: sqlite3.Connection | None = None
        watchdog: threading.Timer | None = None
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=self.timeout_s)
            # Match the vendored evaluator's decoding so denotations are byte-comparable.
            conn.text_factory = lambda b: b.decode(errors="ignore")
            conn.execute("PRAGMA query_only = ON")
            conn.set_authorizer(_authorize_read)
            watchdog = threading.Timer(self.timeout_s, conn.interrupt)
            watchdog.start()
            cursor = conn.execute(sql)
            rows = cursor.fetchmany(self.row_cap + 1)
            truncated = len(rows) > self.row_cap
            materialized = tuple(tuple(row) for row in rows[: self.row_cap])
            return ExecutionResult(rows=materialized, truncated=truncated)
        except sqlite3.OperationalError as exc:
            return _classify_operational(exc)
        except (sqlite3.ProgrammingError, sqlite3.Warning) as exc:
            # e.g. "You can only execute one statement at a time" (defensive; pre-rejected above).
            return ExecutionError(error_type="multi_statement", message=str(exc))
        except sqlite3.DatabaseError as exc:
            if getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_AUTH:
                return ExecutionError(error_type="read_only", message=str(exc))
            return ExecutionError(error_type="syntax", message=str(exc))
        finally:
            if watchdog is not None:
                watchdog.cancel()
                watchdog.join()
            if conn is not None:
                conn.close()


def _classify_operational(exc: sqlite3.OperationalError) -> ExecutionError:
    msg = str(exc)
    lowered = msg.lower()
    if "interrupted" in lowered:
        return ExecutionError(error_type="timeout", message="statement timed out")
    if (
        "readonly" in lowered
        or "read-only" in lowered
        or "attempt to write" in lowered
        or "not authorized" in lowered
    ):
        return ExecutionError(error_type="read_only", message=msg)
    if "syntax" in lowered or "no such" in lowered:
        return ExecutionError(error_type="syntax", message=msg)
    return ExecutionError(error_type="operational", message=msg)
