## Context

Stage 0 (`eval-foundation`, archived) built the generation→score→provenance harness and validated it
on a 50-example MPS slice (Coder-3B zero-shot 0.76). This change adds the GPU runner to produce the
real full-dev baseline and folds in the deferred BIRD-dev loader. It deliberately reuses the Stage-0
harness unchanged — only the Modal wrapper, the vLLM GPU path, and BIRD are new — and mirrors
`grpo-gain-decomposition`'s `modal_app.py` conventions, which we studied at project start. This is
the first GPU spend, but inference-only.

## Goals / Non-Goals

**Goals:**
- A Modal A100-80GB runner that executes the existing `generate()` on GPU via vLLM and persists
  results to a Volume, traceable to the exact code (git commit in provenance).
- A BIRD-dev loader through the shared typed interface, pinned + checksummed like Spider.
- The committed headline base-model baseline: full Spider-dev (1034) + BIRD-dev, zero-shot + few-shot.

**Non-Goals:**
- No SFT/GRPO/LoRA or any training (the next change). This is inference only.
- No new metric or verifier behavior — scoring reuses the Stage-0 execution-match verifier.
- No multi-GPU / distributed; a single A100-80GB is enough for 3B inference.

## Decisions

**M1 — Modal app mirrors the reference repo.** `modal_app.py` is run via the `modal` CLI
(`uv tool install modal`), not a project dependency. Two-layer image: layer 1 installs the pinned
GPU deps (keyed on `pyproject.toml` + `uv.lock`, so editing source doesn't re-download the ~GBs of
CUDA/torch/vLLM); layer 2 relinks the source editable. A single `@app.function(gpu="A100-80GB")`
runs generation; the `@app.local_entrypoint()` computes the git commit/dirty locally (the image
strips `.git`) and passes them in. *Alternative:* bake everything in one layer — rejected (re-downloads
the GPU stack on every source edit).

**M2 — Reuse `generate()` with the vLLM backend; do not re-implement.** The GPU function calls the
existing `generate(..., backend="vllm")`. The `gpu` extra (`vllm`) is exercised for real for the
first time, so CUDA/torch/vLLM versions must agree — re-verify at image build (the reference repo's
"re-verify at build"). *Alternative:* a Modal-specific generation path — rejected; the whole point of
Stage 0's backend split was that the GPU path is just a backend flag.

**M3 — Results to a Modal Volume, pulled locally; provenance unchanged.** The GPU function writes
`CompletionSet`s to a Volume; `modal volume get` pulls them for local CPU scoring (`make eval`-style),
keeping the model-free scoring guarantee. Provenance already captures model/decoding/dataset
revision/DB checksum/git commit — no change.

**M4 — BIRD through the shared typed interface, self-hosted + checksummed.** `bird.py` parses BIRD's
`dev.json` + `dev_tables.json` into the same `Example` / `DatabaseSchema` models and resolves
`<db_id>/<db_id>.sqlite`, reusing `serialize_schema`, the sandbox, and the verifier verbatim. BIRD's
SQLite databases are self-hosted on HF and pinned by commit SHA + SHA256, exactly like Spider
(`BIRD_DB_ARCHIVE` in `artifacts.py`). *Alternative:* a parallel BIRD-specific verifier — rejected;
execution-match is dataset-agnostic.

**M5 — BIRD `evidence` is part of the prompt.** BIRD questions frequently need the `evidence`
(external-knowledge hint) to be answerable, and the standard BIRD protocol includes it. Add an
optional `evidence: str | None = None` to `Example` (Spider examples leave it None) and have the
prompt builder append it when present. This is the one touch to a Stage-0 model, and it keeps the
BIRD baseline comparable to published BIRD numbers. *Alternative:* drop evidence — rejected; it makes
the BIRD baseline artificially low and non-standard.

**M6 — Full-dev, deterministic, committed.** Run the full Spider-dev (1034) and BIRD-dev, zero-shot
+ few-shot, greedy. Commit the `CompletionSet`s + `ScoreReport`s under `results/baseline/` as the
headline reference artifacts; `make eval` re-derives the numbers on CPU. BIRD-dev is the headline.

## Risks / Trade-offs

- **vLLM/CUDA/torch version drift at image build** → pin the `gpu` extra and re-verify on the first
  Modal build; a smoke run (`--limit 8`) before the full pass catches a broken image cheaply.
- **BIRD licensing** → BIRD is CC-BY-NC-4.0 (non-commercial); the self-host must carry that license +
  attribution, and the HF repo is for research use. Document it (distinct from Spider's
  CC-BY-SA).
- **`evidence` touches the shared `Example` model** → kept optional and defaulted None, so every
  Stage-0 test and the Spider path are unaffected; add a test that Spider examples still round-trip.
- **First GPU spend** → inference only (~$5–15); run a tiny smoke before the full pass, and use
  `--detach --spawn` so a client disconnect doesn't lose the run.
- **Modal client-disconnect losing a long run** → `--detach --spawn` (the reference repo's hard-won
  lesson: a synchronous `.remote()` in a detached app can be cancelled on disconnect).

## Open Questions

- Which HF source for BIRD-dev Q/SQL (and which BIRD release — the dataset has had revisions)?
- Pin a specific `Qwen/Qwen2.5-Coder-3B-Instruct` commit for the headline run (currently unpinned)?
- Few-shot exemplars for BIRD — drawn from BIRD train (consistent with R2's train-only rule) or
  reuse Spider exemplars? Likely BIRD-train for a fair BIRD baseline.
- Do we also need the distilled **test-suite** DBs now (R4) to report test-suite accuracy for the
  headline, or is single-DB execution-match acceptable for the baseline (test-suite as a fast-follow)?
