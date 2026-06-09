"""Execution environment: the read-only SQLite sandbox (safety boundary)."""

from text2sql_rlvr.env.sandbox import (
    ExecutionError,
    ExecutionResult,
    SqliteSandbox,
)

__all__ = ["ExecutionError", "ExecutionResult", "SqliteSandbox"]
