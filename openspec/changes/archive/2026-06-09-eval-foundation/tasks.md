## 1. Repo scaffolding

- [x] 1.1 Create `pyproject.toml` (uv, project `llm-rlvr-sql`, Python 3.11) with ruff (line length 100, the rule set mirroring `grpo-gain-decomposition`) and pytest config (network/integration deselected by default).
- [x] 1.2 Create the `src/llm_rlvr_sql/` package layout (`data/`, `env/`, `eval/`, `py.typed`) and `tests/`.
- [x] 1.3 Define optional extras: a CPU base layer (`datasets`, `pydantic`) that installs and tests without the GPU stack; a `generate` extra (`transformers`, `torch`); `vllm` behind a GPU extra. Verify `uv sync` installs the CPU layer alone.
- [x] 1.4 Add a `Makefile` (`install`, `fmt`, `lint`, `test`, `check`) and pre-commit mirroring the reference repo; `make check` = ruff + unit tests.

## 2. Spider data layer (`spider-dataset`)

- [x] 2.1 Pin the `xlangai/spider` HF parquet revision; loader returns strictly-typed Pydantic examples (`db_id`, `question`, `gold_sql`, split) and records the revision used.
- [x] 2.2 Fetch the Spider databases from a pinned HF-mirror commit SHA (via `huggingface_hub`), gated on a recorded **SHA256** (trust anchor; Yale = documented provenance origin, not the runtime fetch); resolve each `db_id` to `<db_id>.sqlite` + its `tables.json` entry; fail loudly on checksum mismatch or a missing database.
- [x] 2.3 Implement schema serialization as a pure function (tables / columns(types) / PK / FK → CREATE-TABLE-style text); deterministic output.
- [x] 2.4 Tests: dev-split count matches published Spider-dev size (integration, network-gated); schema-serialization determinism; checksum-mismatch and missing-DB both raise loudly.

> BIRD-dev loading is deferred to its own follow-up change (design R3): it has a different DB
> layout and an `evidence` field, and the run that uses it needs the GPU runner. Out of scope here.

## 3. Read-only SQLite sandbox (`sql-execution-sandbox`)

- [x] 3.1 Open databases via `file:<path>?mode=ro&immutable=1`; reject multi-statement input; return typed result/error models (no raw tracebacks).
- [x] 3.2 Enforce a wall-clock statement timeout via an interrupt watchdog; return a structured timeout error within a bounded time, leaving the connection usable.
- [x] 3.3 Cap returned rows via `fetchmany`; mark the result truncated when the cap is hit.
- [x] 3.4 Tests: a write statement cannot modify the DB; a `SELECT` returns rows; a slow query is interrupted at the timeout; oversized results truncate + flag; invalid SQL yields a structured error; multi-statement input is rejected.

## 4. Execution-match verifier (`execution-match-verifier`)

- [x] 4.1 Vendor `taoyds/test-suite-sql-eval` (`exec_eval.py`) under an attributed `third_party/` path with its upstream license.
- [x] 4.2 Wrap it for single-DB execution-match (multiset, column-order-insensitive, order-sensitivity gated on gold `ORDER BY`); **cap** the column-permutation search and `log` any dropped comparison.
- [x] 4.3 Add the empty-result guard (empty==empty is not a match unless gold is genuinely empty).
- [x] 4.4 Add the test-suite-accuracy path across distilled databases, gated on their availability (single-DB ships now; distilled DBs are a **scheduled fast-follow before any Stage-2 number is reported** — see design R4; fold in opportunistically if the download from the vendored repo is trivial).
- [x] 4.5 Parity test: run the wrapper and the upstream evaluator over a Spider-dev sample; any per-example disagreement fails the test.
- [x] 4.6 Unit tests per semantic: order-insensitive without gold `ORDER BY`; order-sensitive with it; multiset (duplicate-count) mismatch; column-order-insensitive; empty-guard; permutation-cap (no hang on a wide result).

## 5. Baseline evaluation (`baseline-evaluation`)

- [x] 5.1 Implement `generate(model, examples, sampling, backend)` dispatching `transformers` (CPU/MPS) | `vllm` (GPU); write a typed `CompletionSet` so scoring is model-free.
- [x] 5.2 Prompt builder (serialized schema + question + format instruction) and a fixed few-shot policy: **k=3, seeded random exemplars drawn from the `train` split** (cross-schema, never dev/test — no leakage), recorded in provenance. No retriever.
- [x] 5.3 Scoring CLI: read a `CompletionSet` → verifier → aggregate execution accuracy; deterministic and backend-free.
- [x] 5.4 Provenance model (dataset revision + DB-archive checksum, model + revision, decoding config, git commit/dirty, library versions) attached to every result.
- [x] 5.5 Run zero-shot + few-shot baseline on a deterministic Spider-dev slice (CPU/MPS) end-to-end and record the numbers; document the full-dev + BIRD-dev GPU pass in the RUNBOOK.
- [x] 5.6 Tests: re-scoring a committed `CompletionSet` is deterministic; provenance round-trips; prompt-builder snapshot.

## 6. Reproducibility & docs

- [x] 6.1 Repro manifest pinning every external artifact (HF revision, Spider DB-zip checksum, distilled test-suite DBs, BIRD-dev) + a `make eval` that reproduces the baseline number from a committed `CompletionSet`.
- [x] 6.2 `RUNBOOK.md`: fetching the databases, running the baseline (CPU slice vs GPU full), and running the parity test.
- [x] 6.3 `README` stub + `LICENSE`/attribution for Spider (CC-BY-SA-4.0) and the vendored evaluator.
- [x] 6.4 `make check` green (ruff + unit tests; network/integration deselected by default).
