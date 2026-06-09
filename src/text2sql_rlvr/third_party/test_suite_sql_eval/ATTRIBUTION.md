# Vendored: Test-Suite SQL Eval

Source: <https://github.com/taoyds/test-suite-sql-eval> (Zhong, Yu, Klein, "Semantic Evaluation
for Text-to-SQL with Distilled Test Suites", EMNLP 2020). License: Apache 2.0 (see `LICENSE`).

We vendor only the **execution-match** path:

- `exec_eval.py` — `result_eq` (the result-set comparison core: multiset, column-order-insensitive)
  and `eval_exec_match` (ORDER-BY gating via `'order by' in g_str.lower()`, multi-DB scan).
- `parse.py` — `remove_distinct`, `get_all_preds_for_execution` (imported by `exec_eval.py`).

We do **not** vendor `process_sql.py` / the component-match path (it needs `nltk`); we only use
execution accuracy.

**Modifications to upstream code (behavior-preserving):**
1. The cross-module import in `exec_eval.py` was made package-relative (`from parse import …` →
   `from text2sql_rlvr.third_party.test_suite_sql_eval.parse import …`) so it resolves inside this
   package.
2. Five regex string literals were made raw (`"…\d…"` → `r"…\d…"`) to silence Python
   `SyntaxWarning`s for invalid escape sequences. Identical match behavior; future-proof.

The comparison core (`result_eq` and its helpers) is byte-for-byte upstream — no regex, no logic
change.

Our own wrapper (`text2sql_rlvr.eval.verifier`) reuses upstream `result_eq` for the comparison but
executes through the read-only sandbox; `tests/test_verifier_parity.py` asserts our verdicts agree
with upstream `eval_exec_match` on a sample.
