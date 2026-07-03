# Architecture

Core design: **the verifier that scores the baseline is the same code used as the RL training
reward.** A bug in eval is a bug in training, and vice versa. Generated result files carry enough
metadata to be scored without loading a model; the committed public bundle carries the compact
headline summary.

That verifier-as-reward idea is factored into a **task-agnostic harness** (`harness/`): a task plugs
in by implementing two small protocols — `AnswerParser` (pull the answer out of a completion) and
`Verifier` (check it, return a `Verdict`) — and bundling them as a `Task`. The shaped / near-binary
reward (`harness.score_answer`) is generic over those. **Text-to-SQL is the reference task**
(`tasks/sql.py`); to train your own model on your own verifiable problem, add a task — see
`docs/harness.md`.

The shipped result is single-turn RLVR on that SQL task: SFT cold-start followed by GRPO with
execution-verified rewards. Self-correction support code stays in the tree for analysis, but it is
not on the main training path.

## Module map (dependency direction: top imports bottom, never the reverse)

```
report/        progression.py — base→SFT→RL table; seed-level CIs; single-DB caveat
revision/      feedback.py · rollout.py · train.py · smoke.py · diagnostics.py · eval.py ·
               metrics.py · probe.py · episode.py — forced-revision analysis utilities
train/         traces.py (STaR sampling + execution filter) · sft.py (LoRA SFT) · grpo.py (GRPO) ·
               multiturn.py · diagnostics.py · tooluse.py · multiturn_provenance.py ·
               stage3_provenance.py · policy.py · lora.py · grpo_smoke.py
rewards/       multiturn.py — trajectory credit + improvement-delta bonus over the SQL single-turn
               reward (self-correction analysis only)
tasks/         sql.py — the text-to-SQL Task: SQL parser + verifier + execution reward (shaped /
               near-binary, empty-result guard) + prompt/row builders · __init__.py (task registry)
eval/          completions.py (the CompletionSet artifact) · generate · score · verifier · parsing ·
               prompt · prompts/ · messages · aggregate · provenance · runner ·
               multiturn_generate · bestofn · vllm_backend
env/           sandbox.py — read-only SQLite execution (the safety boundary) ·
               text2sql_env.py — the run_sql tool environment + reference-free feedback
data/          models.py (typed Example/DatabaseSchema records) · spider.py · bird.py ·
               schema.py (schema → prompt text) · artifacts.py (pinned revisions + checksums) ·
               paths.py · jsonl.py · splits.py
harness/       verifier.py (Verifier / Verdict / ReferenceUnverifiable) · answer.py (AnswerParser) ·
               reward.py (RewardConfig shaping + score_answer + completion_text) · task.py (Task
               protocol) — the task-agnostic RLVR core, imports only _base
_base.py       Frozen — pydantic base: frozen=True, extra="forbid"
cli.py         installed CPU CLI for generation, scoring, trace/tool-use builders, and reports
third_party/   vendored Test-Suite SQL Eval (execution-match path only; Apache 2.0)
```

Dependency direction: `harness/` sits at the bottom (it imports only `_base`); `tasks/sql.py` is the
adapter that binds the harness protocols to the SQL pieces in `eval/` + `data/`; `rewards/` and
`train/` build on the SQL task. Nothing under `harness/` imports SQL.

Training-only dependencies import lazily so the package installs and tests without the training
stack.

## Data and result flow

```
Pinned datasets      generation / training       CPU scoring and reporting
Spider + BIRD    ->  saved completions       ->   score -> aggregate -> report
                                                results/public/ -> make eval
```

- **CompletionSet** (`eval/completions.py`) is the boundary object: `generate` (the *only*
  model-loading step) writes one JSON per (model, dataset, decoding) cell; `score` reads it anywhere,
  deterministically, with no generation backend.
- **Evidence split**: `results/public/` is the tracked clean-checkout summary that `make eval`
  validates. Raw CompletionSet/ScoreReport files are not part of the public quickstart.
- **Metadata chain**: every artifact embeds model+revision, dataset+revision, DB archive sha256,
  decoding/hyperparams, library versions, and git commit/dirty.

## Key types and patterns

- **`Frozen`** (`_base.py`): every typed record and persisted artifact is immutable and rejects
  unknown keys — malformed artifacts are rejected at parse time, not silently downstream.
- **Pure builder + thin training driver**: config assembly is pure and CPU-testable
  (`grpo_config_kwargs`, `build_grpo_config`, `split_train_dev`, `filter_traces`); the `run_*`
  functions keep runtime work at the edge. Unit tests cover the builders; model-loading paths are
  integration-marked.
- **Lazy imports**: CLI subcommands and GPU backends (`vllm`, `trl`, `peft`, `torch`) import inside
  the functions that use them, so the CPU eval layer installs and runs without the training stack.
- **TRL version drift degrades to a warning**: `build_grpo_config_from_kwargs` keeps only the kwargs
  the local `GRPOConfig` actually accepts and logs every dropped key loudly — a TRL field rename
  warns instead of crashing.

## The reward (Stage 2)

`harness.score_answer` composes one per-completion score from an `AnswerParser` + a `Verifier`;
`tasks/sql.py` binds those to SQL (the `SqlAnswerParser` + execution-match `SqlVerifier`) and adds
the empty-result guard. The shaping is task-agnostic; only the parser/verifier are SQL. Per
completion, from a single sandbox execution:

- **shaped arm** (SQL-R1-style): `format + executable + result_match + length`, config-driven
  weights (0.2 / 0.2 / 1.0 / 0.1). `format` gives full credit only when `<think>` *leads* the
  completion; a prose preamble caps at `preamble_credit=0.3`. This is deliberate — the SFT model
  emits a preamble on ~99% of completions (~54% of generated tokens), and the reward pressure trains
  it toward the cheaper leading-`<think>` format.
- **near-binary arm** (Arctic-style): `1.0` exact / `0.1` executable / `0` else — the
  shaping-backfires ablation, nothing more.
- **Empty-result guard** (`guard_trivial_empty=True` everywhere, training and trace-filtering): an
  empty result only matches when the reference result is genuinely empty — blocks the always-empty
  reward hack.

## Evaluation discipline

- **Seed-level aggregation only** (`eval/aggregate.py`): mean + 95% Student-t CI over per-seed EX;
  rollouts are never pooled across seeds. The headline success check is the RL-vs-SFT gain CI
  excluding zero. Single-seed controls are point estimates with no CI claim
  (`report/progression.py` enforces both: ≥2 seeds renders a CI, n=1 renders "point").
- **EX is single-DB execution-match** with a disclosed ~6.5% false-positive caveat. Test-suite
  multi-DB accuracy would be stricter, but it is not part of the public result.
- Baselines report both `direct` (literature-comparable) and `<think>`-matched (before-RL anchor)
  prompts — the untrained base scores *lower* with `<think>`, and hiding that would inflate the gain.

## Supported Path

- Eval foundation: data loaders, sandbox, vendored verifier, and baseline.
- SFT cold-start: LoRA adapter that teaches the `<think>` + ```sql``` format.
- Single-turn GRPO: the shipped result with 3-seed confidence intervals, a near-binary ablation,
  and a Llama cross-family control.

Self-correction support code remains tested in the tree for analysis: env tool loop,
forced-revision rollout + eval runner, multi-turn rewards + diagnostics, `select-bestofn`,
and `fr-metrics`.
