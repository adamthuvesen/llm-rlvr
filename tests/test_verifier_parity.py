"""execution-match-verifier: parity with the upstream official evaluator.

Our verifier reuses upstream `result_eq` but executes through the read-only sandbox. This asserts
the end-to-end verdicts agree with upstream `eval_exec_match` on a battery of pred/gold pairs, so
the safety wrapper did not change the metric. Local SQLite only — no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_rlvr.eval.verifier import VerifierConfig, execution_match
from llm_rlvr.third_party.test_suite_sql_eval.exec_eval import eval_exec_match

# (prediction, gold) pairs spanning match / column-order / order-by / empty / mismatch.
PAIRS = [
    ("SELECT a, b FROM t", "SELECT a, b FROM t"),
    ("SELECT b, a FROM t", "SELECT a, b FROM t"),
    ("SELECT a FROM t ORDER BY a DESC", "SELECT a FROM t"),
    ("SELECT a FROM t ORDER BY a DESC", "SELECT a FROM t ORDER BY a"),
    ("SELECT a FROM t WHERE a = 99", "SELECT a FROM t"),
    ("SELECT DISTINCT a FROM t", "SELECT a FROM t"),
    ("SELECT a FROM t WHERE a = 88", "SELECT a FROM t WHERE a = 99"),
    ("SELECT a, NULL FROM t", "SELECT NULL, a FROM t"),
    ("SELECT count(*), sum(a) FROM t", "SELECT sum(a), count(*) FROM t"),
    ("WITH c AS (SELECT a FROM t) SELECT a FROM c", "SELECT a FROM t"),
    ("SELECT a FROM t WHERE a IN (SELECT a FROM t)", "SELECT a FROM t"),
]


@pytest.mark.parametrize("keep_distinct", [True, False])
@pytest.mark.parametrize("pred,gold", PAIRS)
def test_parity_with_upstream(tiny_db: Path, pred: str, gold: str, keep_distinct: bool) -> None:
    ours = execution_match(
        pred, gold, tiny_db, config=VerifierConfig(keep_distinct=keep_distinct)
    ).matched
    upstream = (
        eval_exec_match(
            db=str(tiny_db),
            p_str=pred,
            g_str=gold,
            plug_value=False,
            keep_distinct=keep_distinct,
            progress_bar_for_each_datapoint=False,
        )
        == 1
    )
    assert ours == upstream, f"disagreement on (pred={pred!r}, gold={gold!r})"
