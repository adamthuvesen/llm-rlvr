"""execution-match-verifier: semantics, empty guard, permutation cap, test-suite path."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from llm_rlvr.eval.verifier import (
    GoldExecutionError,
    VerifierConfig,
    execution_match,
)


def _matched(pred: str, gold: str, db: Path, **kw) -> bool:
    return execution_match(pred, gold, db, **kw).matched


def test_order_insensitive_without_gold_order_by(tiny_db: Path) -> None:
    # gold has no ORDER BY, so a differently-ordered prediction still matches.
    assert _matched("SELECT a FROM t ORDER BY a DESC", "SELECT a FROM t", tiny_db) is True


def test_order_sensitive_with_gold_order_by(tiny_db: Path) -> None:
    assert (
        _matched("SELECT a FROM t ORDER BY a DESC", "SELECT a FROM t ORDER BY a", tiny_db) is False
    )


def test_multiset_not_set(tiny_db: Path) -> None:
    # Same set {1,2} and same length (3), different duplicate counts -> non-match.
    pred = (
        "SELECT a FROM t WHERE a=1 UNION ALL SELECT a FROM t WHERE a=1 "
        "UNION ALL SELECT a FROM t WHERE a=2"
    )
    assert _matched(pred, "SELECT a FROM t", tiny_db) is False


def test_column_order_insensitive(tiny_db: Path) -> None:
    assert _matched("SELECT b, a FROM t", "SELECT a, b FROM t", tiny_db) is True


def test_empty_pred_against_nonempty_gold(tiny_db: Path) -> None:
    assert _matched("SELECT a FROM t WHERE a=99", "SELECT a FROM t", tiny_db) is False


def test_empty_pred_against_genuinely_empty_gold(tiny_db: Path) -> None:
    # Official semantics: both genuinely empty -> match.
    assert _matched("SELECT a FROM t WHERE a=88", "SELECT a FROM t WHERE a=99", tiny_db) is True


def test_trivial_empty_guard_blocks_reward_hacking(tiny_db: Path) -> None:
    guarded = VerifierConfig(guard_trivial_empty=True)
    result = execution_match(
        "SELECT a FROM t WHERE a=88", "SELECT a FROM t WHERE a=99", tiny_db, config=guarded
    )
    assert result.matched is False
    assert result.reason == "trivial_empty_guard"


def test_permutation_cap_bounds_wide_results(tiny_db: Path) -> None:
    # With the cap forced to 1, a 2-column result falls back to identity-order compare:
    # identical order matches, reordered does not — and it returns promptly (no hang).
    capped = VerifierConfig(max_permutation_cols=1)
    started = time.monotonic()
    same = execution_match("SELECT a, b FROM t", "SELECT a, b FROM t", tiny_db, config=capped)
    reordered = execution_match("SELECT b, a FROM t", "SELECT a, b FROM t", tiny_db, config=capped)
    assert time.monotonic() - started < 5.0
    assert same.matched is True
    assert reordered.matched is False  # documented fallback tradeoff above the cap


def test_test_suite_catches_spurious_single_db_match(tiny_db: Path, test_suite_dir: Path) -> None:
    inst0 = test_suite_dir / "inst0.sqlite"
    pred, gold = "SELECT a FROM t WHERE a < 3", "SELECT a FROM t"
    # Spurious on inst0 (rows happen to coincide) but wrong on inst1 -> test suite rejects it.
    assert _matched(pred, gold, inst0) is True
    assert _matched(pred, gold, inst0, db_dir=test_suite_dir) is False


def test_gold_failure_raises(tiny_db: Path) -> None:
    with pytest.raises(GoldExecutionError, match="gold query failed"):
        execution_match("SELECT a FROM t", "SELECT bad FROM nope", tiny_db)
