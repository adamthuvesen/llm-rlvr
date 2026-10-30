"""Execution-match verifier.

Reuses the official Test-Suite SQL Eval comparison core (`result_eq`: multiset,
column-order-insensitive) and its preprocessing, but executes through the read-only sandbox so
verification inherits the safety boundary. Order-sensitivity is gated on whether the *gold* query
has `ORDER BY` (the official rule). Pointing `db_dir` at the distilled multi-database test suite
upgrades single-DB execution-match to test-suite accuracy with no code change.

`tests/test_verifier_parity.py` asserts our verdicts agree with upstream `eval_exec_match`.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field

from llm_rlvr._base import Frozen
from llm_rlvr.env.sandbox import ExecutionError, ExecutionOutcome, ExecutionResult, SqliteSandbox
from llm_rlvr.harness.verifier import ReferenceUnverifiable
from llm_rlvr.third_party.test_suite_sql_eval.exec_eval import (
    postprocess,
    replace_cur_year,
    result_eq,
)
from llm_rlvr.third_party.test_suite_sql_eval.parse import remove_distinct

logger = logging.getLogger(__name__)

MatchReason = Literal[
    "match", "pred_error", "pred_truncated", "denotation_mismatch", "trivial_empty_guard"
]

# Verification needs full denotations, so the row cap is high (unlike agent-facing rollouts).
DEFAULT_VERIFY_ROW_CAP = 100_000
SCORER_POLICY_VERSION = "sqlite-execution-match-v2"


class VerifierConfig(Frozen):
    keep_distinct: bool = True
    timeout_s: float = Field(default=30.0, gt=0, allow_inf_nan=False, strict=True)
    row_cap: int = Field(default=DEFAULT_VERIFY_ROW_CAP, gt=0, strict=True)
    # Above this column count, skip the (potentially explosive) column-permutation search and
    # compare under the given column order only — caps the worst case. Logged when it triggers.
    max_permutation_cols: int = Field(default=6, gt=0, strict=True)
    # Training-reward guard: when on, empty==empty is NOT a match (blocks a degenerate
    # always-empty query from farming reward where gold is empty). Off = official semantics.
    guard_trivial_empty: bool = False


class MatchResult(Frozen):
    matched: bool
    reason: MatchReason


class GoldExecutionError(ReferenceUnverifiable):
    """The gold query failed to execute — the example/DB pairing is broken, not the prediction.

    A SQL-specific `ReferenceUnverifiable` (reason ``"gold_error"``), so the generic harness reward
    handles it without knowing about SQL.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, reason="gold_error")


def validate_reference(
    gold_sql: str, db_path: str | Path, *, config: VerifierConfig | None = None
) -> ExecutionResult:
    """Require a complete reference denotation before scoring or loading a training model."""
    cfg = config or VerifierConfig()
    query = postprocess(gold_sql)
    if not cfg.keep_distinct:
        query = remove_distinct(query)
    gold = _execute(Path(db_path), query, cfg)
    if isinstance(gold, ExecutionError):
        raise GoldExecutionError(
            f"gold query failed to execute on {db_path}: {gold.error_type}: {gold.message}"
        )
    if gold.truncated:
        raise GoldExecutionError(
            f"gold query result truncated on {db_path} at row_cap={cfg.row_cap}. "
            "Increase VerifierConfig.row_cap to compare complete results"
        )
    return gold


def execution_match(
    predicted_sql: str,
    gold_sql: str,
    db_path: str | Path,
    *,
    db_dir: str | Path | None = None,
    config: VerifierConfig | None = None,
) -> MatchResult:
    """Decide whether `predicted_sql`'s denotation matches `gold_sql`'s.

    Single-DB when `db_dir` is None; test-suite accuracy when `db_dir` points at the distilled
    databases (the prediction must match gold on every instance). Raises `GoldExecutionError` if
    the gold query itself fails to execute.
    """
    cfg = config or VerifierConfig()
    p_str, g_str = postprocess(predicted_sql), postprocess(gold_sql)
    if not cfg.keep_distinct:
        p_str, g_str = remove_distinct(p_str), remove_distinct(g_str)
    order_matters = "order by" in g_str.lower()

    for path in _resolve_db_paths(db_path, db_dir):
        gold = validate_reference(g_str, path, config=cfg)
        pred = _execute(path, p_str, cfg)
        if isinstance(pred, ExecutionError):
            return MatchResult(matched=False, reason="pred_error")
        if pred.truncated:
            return MatchResult(matched=False, reason="pred_truncated")
        gold_rows, pred_rows = list(gold.rows), list(pred.rows)
        if cfg.guard_trivial_empty and not gold_rows and not pred_rows:
            return MatchResult(matched=False, reason="trivial_empty_guard")
        if not _denotation_eq(gold_rows, pred_rows, order_matters, cfg):
            return MatchResult(matched=False, reason="denotation_mismatch")

    return MatchResult(matched=True, reason="match")


def _resolve_db_paths(db_path: str | Path, db_dir: str | Path | None) -> list[Path]:
    if db_dir is None:
        return [Path(db_path)]
    instances = sorted(Path(db_dir).glob("*.sqlite"))
    if not instances:
        raise ValueError(f"test-suite directory {db_dir} contains no *.sqlite databases")
    return instances


def _execute(db_path: Path, query: str, cfg: VerifierConfig) -> ExecutionOutcome:
    """Run a query with upstream preprocessing, preserving incomplete results and errors."""
    sandbox = SqliteSandbox(db_path, timeout_s=cfg.timeout_s, row_cap=cfg.row_cap)
    return sandbox.execute(replace_cur_year(query))


def _denotation_eq(
    gold_rows: list[tuple],
    pred_rows: list[tuple],
    order_matters: bool,
    cfg: VerifierConfig,
) -> bool:
    num_cols = _num_cols(gold_rows, pred_rows)
    if num_cols > cfg.max_permutation_cols:
        logger.warning(
            "result has %d columns (> cap %d); comparing under given column order only",
            num_cols,
            cfg.max_permutation_cols,
        )
        return _identity_order_eq(gold_rows, pred_rows, order_matters)
    return result_eq(gold_rows, pred_rows, order_matters=order_matters)


def _num_cols(gold_rows: list[tuple], pred_rows: list[tuple]) -> int:
    if gold_rows:
        return len(gold_rows[0])
    if pred_rows:
        return len(pred_rows[0])
    return 0


def _identity_order_eq(gold_rows: list[tuple], pred_rows: list[tuple], order_matters: bool) -> bool:
    """Bounded fallback for very wide results: compare without the column-permutation search."""
    if len(gold_rows) != len(pred_rows):
        return False
    if order_matters:
        return gold_rows == pred_rows
    return Counter(gold_rows) == Counter(pred_rows)
