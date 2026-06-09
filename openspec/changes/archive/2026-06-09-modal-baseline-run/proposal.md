## Why

Stage 0 produced a validated baseline *harness* and a real number on a tiny local slice (Coder-3B
zero-shot 0.76 on 50 Spider-dev examples). The actual "before RL" figure the whole project is
measured against needs the **full** Spider-dev (1034) and the **BIRD-dev headline** — and a full
MPS run is hours, so it belongs on a GPU. This change builds the Modal GPU runner (the same
infrastructure every later training stage needs) and uses it to produce the real base-model
baseline, committed as the reference artifact future RL gains are compared against.

## What Changes

- **Modal GPU runner** (`modal_app.py`): a Modal app mirroring `grpo-gain-decomposition`'s pattern —
  a two-layer image keyed on `pyproject.toml` + `uv.lock` (heavy deps cached separately from
  source), a single **A100-80GB** function that runs the existing `generate()` on GPU via the
  **vLLM** backend, `--detach --spawn` for long runs, and the local git commit/dirty passed into
  provenance (the image strips `.git`). Results persist to a Modal Volume and are pulled locally.
- **BIRD-dev loader** (`bird-dataset`, the deferred eval-foundation task lands here): load BIRD-dev
  examples and databases through the **same typed `Example` / `DatabaseSchema` interface** as
  Spider. BIRD ships its own SQLite databases and an `evidence` (external-knowledge) field; pin +
  checksum its database archive self-hosted on HF like Spider, and serialize its schema with the
  existing serializer. Flag schemas that exceed the prompt-token budget.
- **Proper baseline run** (`baseline-evaluation`, modified): run Qwen2.5-Coder-3B-Instruct on **full
  Spider-dev (1034) and BIRD-dev**, **zero-shot + few-shot**, on the Modal GPU; commit the
  `CompletionSet`s + `ScoreReport`s as the headline reference artifacts (re-scorable on CPU via
  `make eval`). BIRD-dev is the legible headline (Spider is near-saturated for this base).
- **Docs**: extend `RUNBOOK.md` (the Modal run procedure) and `ARTIFACTS.md` (BIRD pins, Modal Volume).

Out of scope (a later change): SFT, GRPO, LoRA training configs, and any training run. This change
is GPU **inference** only — the proper baseline + the runner that later training reuses.

## Capabilities

### New Capabilities
- `modal-runner`: executing the generation pipeline on a single Modal A100-80GB GPU via vLLM, with
  provenance and detach/spawn for long runs; results persisted to a Modal Volume.
- `bird-dataset`: loading BIRD-dev NL→SQL examples and their SQLite databases from pinned,
  checksummed artifacts through the shared typed interface, including the `evidence` field.

### Modified Capabilities
- `baseline-evaluation`: the base-model baseline additionally reports **BIRD-dev** and is produced
  over **full dev** on GPU (not just a CPU/MPS slice), with the headline `CompletionSet`s committed.

## Impact

- **New code**: `modal_app.py` (Modal CLI app, run via `uv tool install modal`, not a project dep);
  `src/text2sql_rlvr/data/bird.py` + a `BIRD_DB_ARCHIVE` pin in `data/artifacts.py`.
- **Reuse (no change)**: `generate()` / `_run_vllm`, `score`, the verifier, prompt builder, and
  provenance from Stage 0 — only the Modal wrapper, the GPU vLLM path, and the BIRD loader are new.
- **New external artifacts**: the `gpu` extra (`vllm`) exercised for real; a self-hosted, checksummed
  BIRD-dev database archive on HF; a Modal account + W&B-style secret only if needed (generation
  needs neither training nor W&B).
- **Cost**: GPU **inference** only — a full-dev + BIRD baseline pass is a few A100-hours (~$5–15 on
  Modal A100-80GB), the first real spend. No training spend.
- **Pins to finalize**: the BIRD-dev HF dataset revision (Q/SQL) + the checksummed BIRD DB archive;
  a model commit for Qwen2.5-Coder-3B-Instruct.
