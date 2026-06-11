# Project: text2sql-rlvr

## What this is
A self-correcting **text-to-SQL agent trained with RL**. The model answers natural-language
questions over a database by **writing SQL, executing it, reading the result/error, and
revising** — with the self-correction *learned* via RL, not prompted. It is the build-and-train
companion to a prior repo (`grpo-gain-decomposition`) that focused on rigorous *measurement* of
RL gains; the throughline is verifying that RL gains are real.

The full validated plan lives at `a local planning doc` (v2,
stress-tested against primary sources June 2026).

## Approach (staged, each stage a checkpoint with a fallback)
- **Stage 0 — evaluation foundation** *(this is the first change)*: data loaders, a read-only
  SQLite sandbox, a vendored execution-match verifier, schema serialization, and a baseline
  "before RL" execution-accuracy number. All CPU-testable except baseline generation.
- **Stage 1 — SFT cold-start** (LoRA): teach the `<think>…</think>` + ```sql``` output format.
- **Stage 2 — single-turn GRPO** (RLVR, the de-risked core): GRPO with a shaped execution reward;
  multi-seed CIs; a Llama-3.2-3B cross-family control; a shaped-vs-near-binary reward ablation.
- **Stage 3 — multi-turn self-correction**: TRL `environment_factory` multi-turn rollout; the
  "is the gain real?" matched-compute feedback-vs-best-of-N control.

## Key decisions (locked)
- **Base model:** Qwen2.5-Coder-3B-**Instruct** (anchor; Qwen3-4B-Instruct-2507 noted as a stronger
  modern alternative). Cross-family control: Llama-3.2-3B-Instruct.
- **Data:** train/verify on **Spider** (clean, deterministic execution-match); **report BIRD-dev**
  too as the legible headline (vanilla Spider gains at 3B are small).
- **Verifier:** vendor the official Test-Suite SQL Eval (`taoyds/test-suite-sql-eval`,
  `exec_eval.py`) — do NOT hand-roll set-compare. It is multiset (duplicate-exact),
  column-order-insensitive, gates order-sensitivity on whether the *gold* has `ORDER BY`, and the
  column-permutation search must be capped. Unit-test parity against the official evaluator.
- **Spider artifacts:** pin TWO — the `xlangai/spider` HF parquet revision (Q/SQL pairs only) AND a
  separately-downloaded, **checksummed** Spider DB zip (`database/<db_id>/<db_id>.sqlite` +
  `tables.json`); `load_dataset` does NOT ship the SQLite DBs.
- **Training:** GRPO via TRL (pin **≥1.3**, not 1.0.0), vLLM colocate, single-GPU. **bf16 LoRA**
  (rank 32–64, `target_modules='all-linear'`, LoRA LR ≈10× full-FT). **No QLoRA.**
- **Reward hacking guard:** single-DB execution-match as the dense *training* reward (with an
  empty-result guard); the distilled multi-DB **test-suite accuracy** at *eval* time.

## Tech stack & conventions
- Python 3.11+ · **uv** · **ruff** (line length 100, the rule set from `grpo-gain-decomposition`)
  · **pytest** · strict Pydantic (`extra="forbid"`, typed models over `dict[str, Any]`).
- Package layout `src/text2sql_rlvr/` with subpackages: `data/`, `env/`, `rewards/`, `train/`,
  `eval/`, `report/`. Mirrors `grpo-gain-decomposition`.
- CPU/GPU dependency split via optional extras: the eval/verifier/data layer installs and tests on
  a CPU box; the training stack (TRL/vLLM/torch) is a separate `train` extra.
- Modal A100-80GB for GPU runs (reuse the `grpo-gain-decomposition` Modal patterns: two-layer image
  keyed on `pyproject.toml`+`uv.lock`, `--detach --spawn` for long runs, git-commit-in-provenance).
- **Provenance/repro:** deterministic seeds, pinned dataset revisions + checksums, recorded
  model/commit/config + library-version hashes; a `make eval` that reproduces headline numbers from
  committed artifacts.

## Conventions for AI agents
- Reuse patterns from `grpo-gain-decomposition` (loaders, provenance, eval-audit stats,
  Modal runner) rather than inventing new ones.
- Validate untrusted input loudly (no silent coercion); treat DB paths as read-only and confined.
- Never string-interpolate into SQL; the sandbox opens databases `mode=ro` with a statement timeout
  and a row cap, no filesystem/network access.
- Smallest useful change; match neighboring style; comments explain *why*.
