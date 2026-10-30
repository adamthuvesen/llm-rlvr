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


def test_gold_failure_includes_database_and_error(tiny_db: Path) -> None:
    with pytest.raises(GoldExecutionError) as exc:
        execution_match("SELECT a FROM t", "SELECT bad FROM nope", tiny_db)
    assert str(tiny_db) in str(exc.value)
    assert "syntax: no such table: nope" in str(exc.value)


def test_truncated_prediction_cannot_match_complete_reference(tiny_db: Path) -> None:
    result = execution_match(
        "SELECT a FROM t", "SELECT a FROM t LIMIT 2", tiny_db, config=VerifierConfig(row_cap=2)
    )
    assert result.matched is False
    assert result.reason == "pred_truncated"


def test_truncated_reference_is_unverifiable(tiny_db: Path) -> None:
    with pytest.raises(GoldExecutionError, match="gold query result truncated") as exc:
        execution_match(
            "SELECT a FROM t LIMIT 2", "SELECT a FROM t", tiny_db, config=VerifierConfig(row_cap=2)
        )
    assert "row_cap=2" in str(exc.value)
    assert str(tiny_db) in str(exc.value)
    assert "Increase VerifierConfig.row_cap" in str(exc.value)


@pytest.mark.parametrize("directory", ["missing", "empty"])
def test_explicit_empty_test_suite_rejected(tiny_db: Path, tmp_path: Path, directory: str) -> None:
    db_dir = tmp_path / directory
    if directory == "empty":
        db_dir.mkdir()
    with pytest.raises(ValueError, match=r"contains no \*.sqlite databases"):
        execution_match("SELECT a FROM t", "SELECT a FROM t", tiny_db, db_dir=db_dir)


@pytest.mark.parametrize(
    "field,value",
    [
        ("timeout_s", 0),
        ("timeout_s", -1),
        ("timeout_s", float("inf")),
        ("timeout_s", float("nan")),
        ("timeout_s", True),
        ("row_cap", 0),
        ("row_cap", 1.5),
        ("max_permutation_cols", 0),
        ("max_permutation_cols", 1.5),
    ],
)
def test_invalid_verifier_limits_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        VerifierConfig.model_validate({field: value})


def test_truncated_prediction_is_invalid_for_reward(tiny_db: Path) -> None:
    from llm_rlvr.tasks.sql import SqlVerifier

    verdict = SqlVerifier(VerifierConfig(row_cap=2)).verify(
        "SELECT a FROM t", "SELECT a FROM t LIMIT 2", tiny_db
    )
    assert not verdict.matched
    assert not verdict.valid
    assert verdict.reason == "pred_truncated"
