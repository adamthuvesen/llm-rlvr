# rlvr-sql-agent

This is a project for learning, exploring, and testing SFT, RLVR, GRPO, and related post-training
ideas on a concrete text-to-SQL task.

This repo is mainly about a successful SFT -> RLVR training run for text-to-SQL.

The core result worked as intended: start with a 3B SQL model, teach it the expected answer
format with supervised fine-tuning, then improve it with execution feedback on one answer at a
time. On Qwen2.5-Coder-3B, SFT improved Spider and BIRD, then single-turn RLVR improved both
again. The RL gain is backed by three seeds, a reward ablation, a Llama control, and committed
evaluation files.

After that, the project tried a follow-up: make the same 3B model check and fix
its own SQL. That part did not succeed. When the model was offered a `run_sql` tool, it never chose
to call it. When every answer was executed for it and the model was forced to revise, the early
probe looked promising, but training drifted toward copying the first answer instead of revising
it. The pilot stop rule, set before the run, stopped the full training job.

To summarize: **SFT + single-turn RLVR produced the shipped result; 3B self-correction was
the follow-up that did not land.** The committed result files prove both claims.

This is the build-and-train companion to
[`grpo-gain-decomposition`](https://github.com/adamthuvesen/grpo-gain-decomposition), which
measured whether RL gains were real.

## Shipped Result

The shipped result is the single-turn training run. The same Qwen2.5-Coder-3B model improves at
each step:

| Benchmark | Before SFT | After SFT | After RLVR | Stepwise gains |
| --- | ---: | ---: | ---: | --- |
| Spider-dev | 0.625 | 0.678 | **0.779** | SFT +5.3pt, then RLVR +10.2pt |
| BIRD-dev | 0.313 | 0.344 | **0.406** | SFT +3.1pt, then RLVR +6.1pt |

The RL-vs-SFT gain is measured across three seeds, and the 95% confidence interval excludes zero
on both benchmarks. The same recipe also improved a Llama-3.2-3B control, which makes the Qwen
result less likely to be a one-model accident.

All numbers in this README are backed by committed completion sets and score reports. Reproduce
the published tables with:

```bash
make data
make eval
```

See [`ARTIFACTS.md`](ARTIFACTS.md) for the exact files, pins, caveats, and confidence intervals.

## Follow-Up: Self-Correction

Once the single-turn result was in hand, the later experiments asked a harder question: can this
3B model learn to check and fix its own SQL?

| Attempt | What happened | Verdict | Record |
| --- | --- | --- | --- |
| Tool-calling loop | The model was prompted and trained to call `run_sql`, but it made **0 tool calls in 672 sampled episodes** across the base, SFT, RL, and two tool-use-SFT checkpoints. | Failed before training a multi-turn model. At this size, the model did not learn the tool-calling protocol. | [`ARTIFACTS.md`](ARTIFACTS.md#stage-3-multi-turn-grpo--closed-without-training-resultsgrpo-mt), [`OpenSpec archive`](openspec/changes/archive/2026-06-12-grpo-multi-turn/tasks.md) |
| Forced revision | The runner executed every first answer and showed feedback before a required second answer. The untrained probe passed: 14.9% fixes, 6.3% harms. During the 60-step pilot, revision fell from 0.26 to 0.05 and the model drifted toward resubmitting the same answer. | Stopped at the pilot gate before the full run. The n=200 evaluation had a small revision gain, but it was inside noise and sat on degraded single-turn behavior. | [`ARTIFACTS.md`](ARTIFACTS.md#stage-3b-forced-revision-grpo--aborted-at-the-pilot-gate-resultsgrpo-fr), [`OpenSpec archive`](openspec/changes/archive/2026-06-12-grpo-forced-revision/tasks.md) |

Those negative results are kept because they define the boundary of the shipped work. The
self-correction code and diagnostics remain in the tree for future attempts on a larger base
model, but the evidence here does not support claiming a working self-correcting 3B SQL agent.

## Evidence Map

| Question | Short answer | Where to check |
| --- | --- | --- |
| Did SFT help? | Yes: Spider 0.625 -> 0.678, BIRD 0.313 -> 0.344. | [`ARTIFACTS.md`](ARTIFACTS.md#stage-1-sft-cold-start-resultssft-adapter-sft-r32), [`archive/2026-06-10-sft-coldstart`](openspec/changes/archive/2026-06-10-sft-coldstart/) |
| Did single-turn RL help beyond SFT? | Yes: Spider 0.678 -> 0.779, BIRD 0.344 -> 0.406, three-seed CI excludes zero. | [`ARTIFACTS.md`](ARTIFACTS.md#stage-2-single-turn-grpo-resultsgrpo--resultsllama), [`archive/2026-06-11-grpo-single-turn`](openspec/changes/archive/2026-06-11-grpo-single-turn/) |
| Was the gain just reward-shaping? | No: the near-binary reward ablation still beat SFT, though it was weaker on BIRD. | [`ARTIFACTS.md`](ARTIFACTS.md#stage-2-single-turn-grpo-resultsgrpo--resultsllama) |
| Did another 3B model show the same pattern? | Yes: Llama-3.2-3B improved from base -> SFT -> GRPO on Spider and BIRD. | [`ARTIFACTS.md`](ARTIFACTS.md#stage-2-single-turn-grpo-resultsgrpo--resultsllama) |
| Did tool use work? | No: 0/672 sampled episodes called `run_sql`. | [`ARTIFACTS.md`](ARTIFACTS.md#stage-3-multi-turn-grpo--closed-without-training-resultsgrpo-mt) |
| Did forced revision work? | No full run: the probe passed, but the pilot failed its stop gate. | [`ARTIFACTS.md`](ARTIFACTS.md#stage-3b-forced-revision-grpo--aborted-at-the-pilot-gate-resultsgrpo-fr) |

For operating details, use [`RUNBOOK.md`](RUNBOOK.md). For the full decision trail, use
[`openspec/changes/archive/`](openspec/changes/archive/).

## Repository Layout

```text
src/text2sql_rlvr/
  data/         Spider/BIRD loaders, typed examples, schema serialization
  env/          read-only SQLite sandbox and the run_sql environment
  eval/         generation, scoring, provenance, revision/tool-use eval runners
  rewards/      single-turn rewards and multi-turn trajectory scoring
  train/        trace sampling, LoRA SFT, GRPO, forced-revision rollout, diagnostics
  report/       progression tables and caveats
  third_party/  vendored Test-Suite SQL Eval execution verifier
```

## Usage

```bash
make install          # CPU environment: data, verifier, eval
make check            # lint, format check, unit tests
make test-integration # network/HuggingFace/model tests
```

Model generation uses the `text2sql-rlvr` CLI, and GPU work runs through Modal entrypoints. See
[`RUNBOOK.md`](RUNBOOK.md) before running baselines, training, or paid jobs.

## Data And License

Spider is CC-BY-SA-4.0. BIRD-dev is CC-BY-NC-4.0. The vendored evaluator is Apache 2.0. Datasets
and databases are pinned and fetched, not committed.

This repository's own code is MIT licensed.
