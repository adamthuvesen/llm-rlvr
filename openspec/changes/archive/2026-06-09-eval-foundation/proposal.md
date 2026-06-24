## Why

Before any RL training spend, the project needs a trustworthy **verifier** and an honest **"before
RL" baseline** — no gain can be claimed without them. This change builds the Stage-0 evaluation
foundation: load the data, execute SQL safely, score predictions with the *official*
execution-match semantics, and record the base model's accuracy. Everything here is CPU-testable
(except baseline generation) and is the substrate every later stage's reward and metric reuse, so
getting it correct now prevents a silently-wrong reward from corrupting all downstream training.

## What Changes

- Scaffold the repo: `uv`/`pyproject.toml`, `ruff` (line 100), `pytest`, and the
  `src/llm_rlvr_sql/` package layout, with a CPU/GPU dependency split (eval/data layer installs
  and tests without the training stack).
- **Spider data layer**: load Spider NL→SQL pairs from a **pinned** `xlangai/spider` HF parquet
  revision, plus the **separately-downloaded, checksummed** Spider SQLite databases (which
  `load_dataset` does not ship). Serialize each example's schema (tables / columns / types / FKs)
  into compact prompt text. BIRD-dev loading included as a reporting target (stretch task).
- **Read-only SQLite sandbox**: execute SQL with `mode=ro`, a statement timeout, and a row cap; no
  filesystem/network access. The safety boundary for every execution.
- **Execution-match verifier**: vendor the official Test-Suite SQL Eval (`taoyds/test-suite-sql-eval`)
  semantics — multiset comparison, column-order-insensitive, order-sensitivity gated on gold
  `ORDER BY`, with the column-permutation search **capped**. Unit-test **parity** against the
  official evaluator on a Spider-dev sample. Include an empty-result guard so empty==empty does not
  grant credit unless gold is genuinely empty.
- **Baseline evaluation**: a model-completion path (transformers CPU/MPS · vLLM GPU) + scoring that
  produces zero-shot and few-shot **execution accuracy** for Qwen2.5-Coder-3B-Instruct on Spider-dev
  (and BIRD-dev), with decoding settings, dataset revisions/checksums, and model/commit hashes
  recorded for reproducibility.

Out of scope (later stages): reward shaping, SFT, GRPO/training, the Modal training runner, LoRA,
multi-turn rollouts. This change stops at "we can load, execute, score correctly, and report a
trustworthy baseline."

## Capabilities

### New Capabilities
- `spider-dataset`: loading Spider (and BIRD-dev) NL→SQL examples plus their SQLite databases from
  pinned, checksummed artifacts, and serializing a database schema into compact prompt text.
- `sql-execution-sandbox`: executing a SQL query against a database read-only, with a statement
  timeout and row cap, returning rows or a structured error.
- `execution-match-verifier`: deciding whether a predicted SQL's result set matches the gold's
  under test-suite execution-match semantics, parity-tested against the official evaluator.
- `baseline-evaluation`: measuring and recording the base model's execution accuracy on Spider-dev
  (and BIRD-dev) with full provenance, as the reproducible "before RL" number.

### Modified Capabilities
<!-- None — this is the first change; no existing specs. -->

## Impact

- **New code**: `src/llm_rlvr_sql/{data,env,eval}/`, vendored `taoyds/test-suite-sql-eval` under a
  clearly-attributed third-party path, `pyproject.toml`, `Makefile`, tests under `tests/`.
- **New dependencies**: `datasets`, `pydantic`, and a generation backend (`transformers`/`torch`,
  with `vllm` behind a GPU extra). License note: Spider is CC-BY-SA-4.0 (attribute + share-alike);
  the vendored evaluator carries its upstream license.
- **External artifacts to pin**: the `xlangai/spider` HF parquet revision, the Spider DB zip
  (checksummed), the distilled test-suite DBs (for eval-time test-suite accuracy), and optionally
  BIRD-dev.
- **No training, no GPU training cost** — baseline generation is the only model-loading step and can
  run on a small dev slice (CPU/MPS) or a short Modal GPU pass.
