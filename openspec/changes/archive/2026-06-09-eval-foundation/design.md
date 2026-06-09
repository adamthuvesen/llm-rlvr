## Context

This is the first change in the `text2sql-rlvr` project (greenfield repo). It builds the Stage-0
evaluation foundation that every later RL stage depends on: the same verifier that scores the
baseline becomes the result-match component of the training reward. The project deliberately mirrors
`grpo-gain-decomposition` conventions (package layout, Pydantic provenance, CPU/GPU extra split,
generation-backend pattern). A primary-source validation pass (see `local planning docs`) surfaced
several correctness traps that this design exists to avoid — chiefly that a hand-rolled SQL
result-comparison silently diverges from the accepted metric, and that the Spider SQLite databases
are not delivered by `load_dataset`.

## Goals / Non-Goals

**Goals:**
- A correct, parity-tested execution-match verifier and a safe read-only SQL sandbox.
- A reproducible "before RL" baseline (zero-shot + few-shot) for Qwen2.5-Coder-3B-Instruct on
  Spider-dev (and BIRD-dev), with full provenance.
- A package skeleton + CPU/GPU dependency split that lets the data/env/eval layer install and test
  without the training stack.

**Non-Goals:**
- No reward shaping, SFT, GRPO, LoRA, Modal training runner, or multi-turn rollouts (later stages).
- No new verifier semantics — we adopt the official Test-Suite SQL Eval behavior, not an improved
  metric.
- No leaderboard-grade BIRD pipeline; BIRD-dev is a reporting target, loaded but not optimized.

## Decisions

**D1 — Vendor the official Test-Suite SQL Eval; do not hand-roll "set compare."**
We copy `taoyds/test-suite-sql-eval` (`exec_eval.py`) under an attributed third-party path and call
it, rather than reimplementing result comparison. Rationale: a naive set-compare diverges from the
accepted metric in four documented ways — it must be (a) **multiset** (duplicate-exact, not set),
(b) **column-order-insensitive** (enumerate column permutations), (c) order-sensitive only when the
**gold** query has `ORDER BY`, and (d) bounded — the upstream column-permutation search is unbounded
and hangs on wide `SELECT`s, so we **cap** it. *Alternative considered:* reimplement from the paper
— rejected; it reintroduces exactly the false-positive/false-negative traps the vendored code
already solves. We add a **parity test** that runs our wrapper and the upstream evaluator over a
Spider-dev sample and fails on any disagreement.

**D2 — Pin two Spider artifacts, not one.** `load_dataset("xlangai/spider", revision=…)` returns
only Q/SQL parquet; the SQLite databases (`database/<db_id>/<db_id>.sqlite` + `tables.json`) are a
separate download. We pin (1) the HF parquet revision and (2) a **checksummed** Spider DB archive,
recording both in a repro manifest. *Alternative:* trust `load_dataset` to provide DBs — rejected;
it silently doesn't, leaving the verifier nothing to execute against.

**D3 — Read-only sandbox via SQLite URI + interrupt timer + row cap.** Open every DB with
`sqlite3.connect("file:<path>?mode=ro&immutable=1", uri=True)`; enforce a wall-clock statement
timeout (a watchdog thread calling `Connection.interrupt()`, or `set_progress_handler`); cap rows
with `fetchmany(row_cap)`; reject multi-statement input. No filesystem/network access is reachable
from SQL. *Alternative:* subprocess/container isolation — rejected as overkill for read-only SQLite
with no extension loading; revisit only if we ever allow writes.

**D4 — Two metrics, one verifier core.** The verifier supports both (a) **single-DB
execution-match** (cheap; the future dense training reward) and (b) **distilled multi-DB test-suite
accuracy** (the eval-time headline that catches spurious single-DB matches). The baseline reports
test-suite accuracy where the distilled DBs are available, single-DB execution-match otherwise. An
**empty-result guard** prevents empty==empty from scoring as correct unless the gold result is
genuinely empty.

**D5 — Schema serialization is a pure, swappable function.** Serialize `tables.json` into a compact
`CREATE TABLE`-style block (table → columns(types) + PK/FK edges), the format SQL-R1/OmniSQL-class
models expect. Keep it a pure `(schema) -> str` function with a fixed contract so we can A/B
serialization styles later without touching callers. *Alternative:* M-schema/JSON serialization —
deferred; start with the proven CREATE-TABLE style.

**D6 — Generation backend split, reusing the prior repo's pattern.** A single `generate(model,
examples, sampling, backend)` that dispatches to `transformers` (CPU/MPS, for dev slices) or `vllm`
(GPU, for full dev). Writes a `CompletionSet` so scoring is a separate, model-free CPU step.
`vllm`/`torch` live behind a GPU extra; the data/env/verifier/scoring layer installs without them.

**D7 — Strict Pydantic models + provenance.** Typed models (`extra="forbid"`) for an example
(`db_id`, `question`, `gold_sql`, serialized `schema`), a completion set, and a provenance record
(dataset revision + checksum, model + revision, decoding config, git commit/dirty, library
versions). Mirrors `grpo-gain-decomposition`'s `RunProvenance`.

## Risks / Trade-offs

- **Verifier drift from the official metric** → the parity test is a hard gate; any mismatch on the
  Spider-dev sample fails CI, not a warning.
- **Unbounded column-permutation search (hang/DoS on wide SELECTs)** → cap the permutation count and
  `log` when a case is dropped; never silently truncate.
- **Spider DB archive sourcing/licensing** → pin by checksum, document the source, honor Spider's
  CC-BY-SA-4.0 (attribute + share-alike) and the evaluator's upstream license.
- **BIRD schemas are large** → may overflow the prompt budget; flag long schemas, keep BIRD-dev a
  reporting stretch, not a blocker.
- **CPU baseline generation is slow** → baseline runs on a deterministic dev slice (CPU/MPS) or a
  short Modal GPU pass; full-dev baseline is a GPU step. Decoding settings are recorded so the later
  RL comparison matches them (avoid the self-consistency-vs-greedy mismatch the validation flagged).
- **Spider has known annotation errors** → acknowledged; test-suite comparison mitigates a documented
  ~6.5% single-DB false-positive rate but does not fix gold-label errors — report, don't paper over.

## Resolved Decisions (were open questions)

**R1 — Spider DB source: HF mirror pinned by commit SHA, gated by a recorded SHA256.** Yale is the
canonical origin but serves the databases via a Google Drive link that is a reproducibility hazard
for scripted/CI downloads (gdown breakage, quota, virus-scan interstitials). Pin a specific HF
mirror commit that ships the `database/` files, fetch via `huggingface_hub` (CDN, resumable,
immutable SHA), and gate on a SHA256 we compute and commit — the checksum is the authenticity
anchor, the mirror is transport. Document Yale as provenance.

**R2 — Few-shot policy: fixed k=3, seeded random exemplars from the train split, cross-schema, no
retriever.** The baseline must be a clean, reproducible reference, not an optimized pipeline.
Similarity selection (DAIL-SQL-style) lifts accuracy but adds a retriever and muddies attribution
(RL gain vs better retriever). Exemplars come from `train` only (no dev/test leakage) with a fixed
seed for byte-reproducibility. Similarity selection is a deliberate non-goal here (possible later
"base-model ceiling" arm).

**R3 — BIRD-dev: in-scope-but-gated, Spider-first.** The loader is dataset-agnostic; BIRD-dev is the
stretch task (2.4) and merge is gated on Spider-green. BIRD's larger schemas and `evidence` field
must not tangle the verifier/sandbox core (which the parity test anchors on Spider). Lands in this
change if there is runway, else an immediate follow-up with zero rework.

**R4 — Test-suite DBs: single-DB now, distilled test-suite DBs as a scheduled fast-follow.** The
single-DB parity test already proves we match the official comparison logic (identical for 1 or N
DBs), and single-DB execution-match is all the Stage-2 training reward needs. The multi-DB path is
spec'd and gated, so adding it is "drop in the DBs, flip the gate." Firm rule: the *reported*
headline and any reward-hacking-robustness claim MUST use test-suite accuracy, so it is scheduled
before any Stage-2 number is reported — not deferred indefinitely. Fold in opportunistically if the
download (from the same vendored `taoyds/test-suite-sql-eval` repo) proves trivial during impl.
