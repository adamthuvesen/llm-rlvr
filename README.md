# rlvr-sql

A text-to-SQL post-training project with a simple test: can a 3B coding model get better when SQL execution is the reward?

The shipped result is **yes for single-turn RLVR**. Starting from Qwen2.5-Coder-3B, supervised fine-tuning teaches the answer format, then GRPO with execution-verified rewards improves both Spider-dev and BIRD-dev again. The gain is backed by three RL seeds, a reward ablation, a Llama-3.2-3B control, and committed evaluation artifacts.

The follow-up result is more cautious: **3B self-correction did not train successfully**, but a cheap inference-time rule did help. Running the first SQL and asking for a second answer only when execution errors recovered much of the forced-revision gain with far fewer second generations.

![Training results](docs/figures/training-results.svg)

## Main Result

| Benchmark | Base | SFT | RLVR | Gain from SFT | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| Spider-dev | 0.625 | 0.678 | **0.779** | **+10.2pt** | 3 RL seeds, 95% CI excludes zero |
| BIRD-dev | 0.313 | 0.344 | **0.406** | **+6.1pt** | 3 RL seeds, 95% CI excludes zero |

What makes the result worth trusting:

- **The numbers re-run from committed artifacts.** `make eval` re-scores the saved completion sets; the README tables are not hand-entered experiment notes.
- **The RL gain is not one lucky seed.** The main Qwen result reports the mean of three GRPO seeds, and the RL-vs-SFT confidence interval excludes zero on both benchmarks.
- **The reward ablation still helps.** A near-binary reward also beats SFT, though it is weaker on BIRD, so the result is not only reward-shaping ceremony.
- **The pattern transfers to another 3B model.** Llama-3.2-3B improves under the same SFT -> GRPO recipe in the cross-family control.

See [`ARTIFACTS.md`](ARTIFACTS.md) for exact files, dataset pins, score reports, confidence intervals, and caveats.

## What Was Trained

The main run is intentionally narrow:

1. Generate one SQL answer for each question.
2. Execute it in a read-only SQLite sandbox.
3. Reward execution correctness, with guards against empty-result hacks.
4. Train LoRA adapters with GRPO.
5. Re-score Spider-dev and BIRD-dev from saved completions.

The model is not allowed to see hidden labels or write to databases. Datasets are pinned and fetched, not committed.

## Follow-Up: Execution-Error Retry

After the single-turn result landed, the repo tested whether the same 3B model could check and fix its own SQL.

![Execution-error retry](docs/figures/execution-error-retry.svg)

| Follow-up | Result | What to take from it |
| --- | --- | --- |
| Tool-calling loop | **Did not work.** Across 672 sampled episodes, base/SFT/RL/tool-use checkpoints made 0 `run_sql` calls. | At this size, the model did not learn the tool-use protocol. |
| Forced-revision training | **Stopped at the pilot gate.** The model drifted toward resubmitting the first answer instead of revising. | The pre-registered stop rule did its job; no full training claim is made. |
| Execution-error retry | **Worked as a cheap rule.** Ask for a second answer only when first execution errors. Spider: 0.768 -> 0.782 with 40/1034 second generations. BIRD: 0.405 -> 0.435 over locally decomposable items with 297/1534 second generations. Both had 0 harms. | Useful inference-time behavior, not evidence that self-correction training succeeded. |

That distinction is the point of the repo: the shipped research result is single-turn RLVR; the self-correction work is recorded as a boundary and a future direction.

## Reproduce The Tables

```bash
make data
make eval
```

`make eval` is CPU-only and model-free: it loads committed completion sets, runs the verifier, and prints the progression tables plus the retry follow-up numbers.

Core local checks:

```bash
make install          # CPU environment: data, verifier, eval
make check            # lint, format check, unit tests
make test-integration # network/HuggingFace/model tests
```

GPU generation and training run through Modal entrypoints. Read [`RUNBOOK.md`](RUNBOOK.md) before launching paid jobs.

## Repository Layout

```text
src/text2sql_rlvr/
  data/         Spider/BIRD loaders, typed examples, schema serialization
  env/          read-only SQLite sandbox and run_sql environment
  eval/         generation, scoring, provenance, revision/tool-use runners
  rewards/      single-turn rewards and multi-turn trajectory scoring
  train/        trace sampling, LoRA SFT, GRPO, forced-revision rollout, diagnostics
  report/       progression tables and caveats
  third_party/  vendored Test-Suite SQL Eval execution verifier
```

## Related Repo

This is the build-and-train companion to [`grpo-gain-decomposition`](https://github.com/adamthuvesen/grpo-gain-decomposition), which asks how much of RL reasoning gains are real versus measurement artifacts.

## Data And License

Spider is CC-BY-SA-4.0. BIRD-dev is CC-BY-NC-4.0. The vendored evaluator is Apache 2.0. Datasets and databases are pinned and fetched, not committed.

This repository's own code is MIT licensed.
