# Architecture

The one-sentence design: **the execution-match verifier that scores the baseline is the same code
that becomes the RL training reward** — a bug in the eval layer is a bug in training. Every
artifact carries provenance, so committed numbers re-derive from disk.

## Module map (dependency direction: top imports bottom, never the reverse)

```
report/        progression.py — base→SFT→RL table; seed-level CIs; single-DB caveat
train/         traces.py (STaR sampling+filter) · sft.py (LoRA SFT) · grpo.py (GRPO)
rewards/       reward.py — shaped + near-binary arms over the verifier (TRL reward fn)
eval/          generate · score · verifier · parsing · prompt · aggregate · provenance · cli
env/           sandbox.py — read-only SQLite execution (the safety boundary)
data/          spider.py · bird.py · schema.py · artifacts.py (pinned checksums)
_base.py       Frozen — pydantic base: frozen=True, extra="forbid"
third_party/   vendored Test-Suite SQL Eval (execution-match path only; Apache 2.0)
```

`modal_app.py` (repo root) is the only GPU surface; `src/` never imports Modal.

## The data/artifact flow

```
HF mirrors (pinned)         GPU (Modal A100-80GB)              CPU (local, model-free)
HAL-9001/spider-databases → generate/sample/train on Volume → pull JSON → score → aggregate → report
HAL-9001/bird-dev           text2sql-runs:/runs/{data,sft,grpo}  results/ (committed) → make eval
```

- **CompletionSet** is the boundary object: `generate` (the *only* model-loading step) writes one
  JSON per (model, dataset, decoding) cell; `score` reads it anywhere, deterministically.
- **Committed artifacts re-derive**: everything under `results/` must be reproducible via
  `make eval` from the committed CompletionSets — no number in a doc without its artifact.
- **Provenance chain**: every artifact embeds model+revision, dataset+revision, DB archive sha256,
  decoding/hyperparams, library versions, and git commit/dirty. The Modal image strips `.git`, so
  local entrypoints compute git state and pass it in (`_local_git_commit()` in `modal_app.py`).

## Key types and patterns

- **`Frozen`** (`_base.py`): every typed record and persisted artifact is immutable and rejects
  unknown keys — malformed artifacts fail loudly at parse time, not silently downstream.
- **Pure builder + thin GPU driver**: config assembly is pure/CPU-testable
  (`grpo_config_kwargs`, `build_grpo_config`, `split_train_dev`, `filter_traces`); the `run_*`
  functions are thin GPU drivers. Unit tests cover the builders; GPU paths are integration-marked.
- **Lazy imports**: CLI subcommands and GPU backends (`vllm`, `trl`, `peft`) import inside
  functions so the CPU eval layer installs and runs without the training stack.
- **`build_grpo_config` filters kwargs** to the fields TRL's `GRPOConfig` actually accepts and
  warns loudly on drops — TRL version drift degrades to a warning, not a crash.

## The reward (Stage 2)

`rewards/reward.py` composes per-completion scores from the verifier:

- **shaped arm** (SQL-R1-style): `format + executable + result_match + length`, config-driven
  weights. `format` gives full credit only when `<think>` *leads* the completion (a prose preamble
  caps at `preamble_credit=0.3`) — this is deliberate: the SFT model wastes ~54% of tokens on
  preamble, and the reward pressure trains it away.
- **near-binary arm** (Arctic-style): 1.0 / 0.1 / 0 tiers — the shaping-backfires ablation.
- **Empty-result guard** (`guard_trivial_empty=True` everywhere, training and trace-filtering):
  an empty result only matches when gold is genuinely empty — blocks the always-empty reward hack.

## Evaluation discipline

- **Seed-level aggregation only** (`eval/aggregate.py`): mean + 95% Student-t CI over per-seed EX;
  never pool rollouts across seeds. Headline bar = the RL-vs-SFT gain CI excludes zero.
  Single-seed controls are point estimates — no CI claim (`report/progression.py` enforces both).
- **EX is single-DB execution-match** with a disclosed ~6.5% false-positive caveat; the verifier
  already supports test-suite multi-DB via `db_dir` if distilled DBs are self-hosted later.
- Baselines report both `direct` (literature-comparable) and `<think>`-matched (before-RL anchor)
  prompts — the untrained base scores *lower* with `<think>`, and hiding that would inflate gains.

## Modal (GPU runner)

- App `text2sql-rlvr-baseline`; Volumes `text2sql-runs` (`/runs`) + `text2sql-hf-cache`.
- Two-layer image: heavy CUDA/torch/vLLM layer keyed on `pyproject.toml`+`uv.lock` only; source
  edits re-run just the editable relink.
- Image ignore patterns are **root-anchored** (`data/**`, not `**/data`) — a deep glob would strip
  the `src/text2sql_rlvr/data/` package out of the image (this bit once).
- Entrypoints: `main` (baseline generate), `traces` (STaR sampling), `sft`, `grpo` — each mirrors
  a GPU function and takes `--spawn`; all are base-model-parametric (Qwen default, Llama control).

## Stages (what exists, where it's specified)

Stages 0–2 (eval foundation, SFT cold-start, single-turn GRPO) are archived in
`openspec/changes/archive/` with capabilities synced to `openspec/specs/`; the archived
`design.md`s record the decisions behind most of the above. Stage 3 (multi-turn self-correction)
is the active change at `openspec/changes/grpo-multi-turn/` — it adds `env/text2sql_env.py`
(the `run_sql` tool environment), `rewards/multiturn.py` (trajectory credit),
`train/diagnostics.py` (collapse alarms), and `train/multiturn.py` (TRL wiring + the GPU smoke).
