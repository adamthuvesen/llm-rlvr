# Architecture

Core design: **the execution-match verifier that scores the baseline is the same code used as the
RL training reward.** A bug in eval is a bug in training, and vice versa. Every result file carries
enough metadata to re-derive committed numbers from disk with no model in the loop.

The shipped result is single-turn RLVR (SFT cold-start → GRPO with execution-verified rewards). The
self-correction follow-ups (tool loop, forced revision) are documented negatives; their support code
stays in the tree, tested, but is not on the main path. See *Stages* at the bottom.

## Module map (dependency direction: top imports bottom, never the reverse)

```
report/        progression.py — base→SFT→RL table; seed-level CIs; single-DB caveat
train/         traces.py (STaR sampling + execution filter) · sft.py (LoRA SFT) · grpo.py (GRPO) ·
               multiturn.py (Stage-3 tool-loop smoke) · forced_revision.py (Stage-3b rollout_func) ·
               diagnostics.py (D8 collapse alarms) · tooluse.py (tool-use SFT dataset) ·
               multiturn_provenance.py (Stage-3 provenance contract)
rewards/       reward.py — shaped + near-binary arms over the verifier (TRL reward fn) ·
               multiturn.py — trajectory credit + improvement-delta bonus
eval/          completions.py (the CompletionSet artifact) · generate · score · verifier · parsing ·
               prompt · aggregate · provenance · cli · multiturn_generate ·
               forced_revision_generate · bestofn · fr_metrics · revision_probe
env/           sandbox.py — read-only SQLite execution (the safety boundary) ·
               text2sql_env.py — the run_sql tool environment + gold-free feedback
data/          models.py (typed Example/DatabaseSchema records) · spider.py · bird.py ·
               schema.py (schema → prompt text) · artifacts.py (pinned revisions + checksums)
_base.py       Frozen — pydantic base: frozen=True, extra="forbid"
third_party/   vendored Test-Suite SQL Eval (execution-match path only; Apache 2.0)
```

`modal_app.py` (repo root) is the only GPU surface; `src/` never imports Modal, and the GPU
functions import `text2sql_rlvr` lazily so the package installs and tests without the training stack.

## Data and result flow

```
HF mirrors (pinned)         GPU (Modal A100-80GB)              CPU (local, model-free)
HAL-9001/spider-databases → generate/sample/train on Volume → pull JSON → score → aggregate → report
HAL-9001/bird-dev           text2sql-runs:/runs/{data,sft,grpo}  results/ (committed) → make eval
```

- **CompletionSet** (`eval/completions.py`) is the boundary object: `generate` (the *only*
  model-loading step) writes one JSON per (model, dataset, decoding) cell; `score` reads it anywhere,
  deterministically, with no generation backend.
- **Committed result files re-derive**: everything under `results/` must be reproducible via
  `make eval` (`results/baseline/reproduce.py`) from the committed CompletionSets — no number in a
  doc without its source file.
- **Metadata chain**: every artifact embeds model+revision, dataset+revision, DB archive sha256,
  decoding/hyperparams, library versions, and git commit/dirty. The Modal image strips `.git`, so
  local entrypoints compute git state and pass it in (`_local_git_commit()` in `modal_app.py`).

## Key types and patterns

- **`Frozen`** (`_base.py`): every typed record and persisted artifact is immutable and rejects
  unknown keys — malformed artifacts fail loudly at parse time, not silently downstream.
- **Pure builder + thin GPU driver**: config assembly is pure and CPU-testable
  (`grpo_config_kwargs`, `build_grpo_config`, `split_train_dev`, `filter_traces`); the `run_*`
  functions are thin GPU drivers Modal calls. Unit tests cover the builders; GPU paths are
  integration-marked.
- **Lazy imports**: CLI subcommands and GPU backends (`vllm`, `trl`, `peft`, `torch`) import inside
  the functions that use them, so the CPU eval layer installs and runs without the training stack.
- **TRL version drift degrades to a warning**: `build_grpo_config_from_kwargs` keeps only the kwargs
  the installed `GRPOConfig` actually accepts and logs every dropped key loudly — a TRL field rename
  warns instead of crashing.

## The reward (Stage 2)

`rewards/reward.py` composes one per-completion score from a single sandbox execution:

- **shaped arm** (SQL-R1-style): `format + executable + result_match + length`, config-driven
  weights (0.2 / 0.2 / 1.0 / 0.1). `format` gives full credit only when `<think>` *leads* the
  completion; a prose preamble caps at `preamble_credit=0.3`. This is deliberate — the SFT model
  emits a preamble on ~99% of completions (~54% of generated tokens), and the reward pressure trains
  it toward the cheaper leading-`<think>` format.
- **near-binary arm** (Arctic-style): `1.0` exact / `0.1` executable / `0` else — the
  shaping-backfires ablation, nothing more.
- **Empty-result guard** (`guard_trivial_empty=True` everywhere, training and trace-filtering): an
  empty result only matches when gold is genuinely empty — blocks the always-empty reward hack.

## Evaluation discipline

- **Seed-level aggregation only** (`eval/aggregate.py`): mean + 95% Student-t CI over per-seed EX;
  rollouts are never pooled across seeds. The headline success check is the RL-vs-SFT gain CI
  excluding zero. Single-seed controls are point estimates with no CI claim
  (`report/progression.py` enforces both: ≥2 seeds renders a CI, n=1 renders "point").
- **EX is single-DB execution-match** with a disclosed ~6.5% false-positive caveat; the verifier
  already supports test-suite multi-DB via `db_dir` if the distilled DBs are self-hosted later.
- Baselines report both `direct` (literature-comparable) and `<think>`-matched (before-RL anchor)
  prompts — the untrained base scores *lower* with `<think>`, and hiding that would inflate the gain.

## Modal (GPU runner)

- App `text2sql-rlvr-baseline`; Volumes `text2sql-runs` (`/runs`) + `text2sql-hf-cache`.
- Two-layer image: the heavy CUDA/torch/vLLM layer is keyed on `pyproject.toml`+`uv.lock` only, so a
  source edit re-runs just the fast editable relink, not the multi-GB install.
- Image ignore patterns are **root-anchored** (`data/**`, not `**/data`) — a deep glob would strip
  the `src/text2sql_rlvr/data/` package out of the image (this bit once).
- Entrypoints are thin `@app.local_entrypoint`s, each mirroring a GPU `@app.function` and taking
  `--spawn`. Stages 0–2: `main` (baseline/adapter generate), `traces` (STaR sampling), `sft`,
  `grpo` — all base-model-parametric (Qwen default, Llama control). The self-correction surfaces add
  `mt_smoke`/`mt_eval` (Stage 3 tool loop), `fr_smoke`/`fr_train`/`fr_eval` (Stage 3b forced
  revision), and `generate_raw` (the probe's prompts-in/completions-out pass).

## Stages (what exists, where it's specified)

Every stage is archived under `openspec/changes/archive/`; each archived `design.md` and `tasks.md`
journal records the decisions and verdicts. Stages 0–2 shipped, with capabilities synced to
`openspec/specs/`:

- **Stage 0** — eval foundation (data loaders, sandbox, vendored verifier, baseline).
- **Stage 1** — SFT cold-start (LoRA; instills the `<think>` + ```sql``` format).
- **Stage 2** — single-turn GRPO (the shipped result: 3-seed CI excludes zero on both benchmarks,
  near-binary ablation, Llama cross-family control).

The two self-correction sequels both closed honestly **without a full training run**:

- **Stage 3** (`grpo-multi-turn`, tool choice) — closed without training: the 3B model never adopts
  the `run_sql` protocol (0 tool calls in 672 sampled episodes across base / SFT / GRPO / two
  tool-use-SFT checkpoints and every prompting strength); a two-arm probe ruled out RL formatting as
  the cause.
- **Stage 3b** (`grpo-forced-revision`) — removed tool choice via a hand-written TRL `rollout_func`
  (`train/forced_revision.py`: unconditional execution, loss-masked feedback splice, revision). Its
  smoke validated the masking + IS-ratio assertions on real spliced rollouts (the validation Stage 3
  never reached), and the sequence **aborted at the pre-registered pilot gate** (identity-resubmission
  drift; see `results/grpo-fr/` and the README progression table). Only a cheap inference-time
  error-only retry recovered most of the would-be gain.

The self-correction support code (env tool loop, forced-revision rollout + eval runner, multi-turn
rewards + D8 diagnostics, `select-bestofn`, `fr-metrics`) stays tested in the tree for a future
attempt on a larger base.
