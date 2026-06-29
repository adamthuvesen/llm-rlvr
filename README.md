# llm-rlvr-sql

Text-to-SQL post-training with verifiable rewards. The short version: single-turn execution-verified GRPO worked well. Training the same 3B model to self-correct with execution feedback did not.

## Main Result

Starting from `Qwen/Qwen2.5-Coder-3B-Instruct`, the project uses:

1. **LoRA SFT** to teach the answer format.
2. **GRPO** with a reward computed by executing the model's SQL in a read-only SQLite sandbox.
3. **CPU** replay of committed completions to re-score every reported number.

Training results

| Benchmark  | Base  | SFT   | GRPO      | Gain vs SFT | Evidence                      |
| ---------- | ----- | ----- | --------- | ----------- | ----------------------------- |
| Spider-dev | 0.625 | 0.678 | **0.779** | **+10.2pt** | 3 seeds, 95% CI excludes zero |
| BIRD-dev   | 0.313 | 0.344 | **0.406** | **+6.1pt**  | 3 seeds, 95% CI excludes zero |

Evaluation steps:

- `make eval` re-derives the table from committed `CompletionSet` artifacts.
- GRPO is aggregated per seed; rollouts are not pooled to fake a tighter interval.
- The near-binary reward ablation still beats SFT, though it is weaker on BIRD.
- The same SFT -> GRPO process improves a Llama-3.2-3B control model.
- Training reward and evaluation use the same execution-match verifier.

Exact artifact paths, dataset pins, score reports, confidence intervals, and caveats live in
`[docs/ARTIFACTS.md](docs/ARTIFACTS.md)`.

## How It Works

The training path is deliberately small.

1. Load pinned Spider/BIRD examples and schemas.
2. Ask the model for SQL.
3. Execute predicted and gold SQL in a read-only SQLite sandbox.
4. Score denotation match.
5. Keep `guard_trivial_empty=True` so the model cannot win by returning empty results.
6. Train LoRA adapters with SFT, then GRPO.
7. Re-score saved completions without loading a model.

The model never sees hidden labels during reward computation. The sandbox cannot write to the
databases. Datasets are pinned and fetched; database files are not committed.

For the module map and design notes, see `[docs/architecture.md](docs/architecture.md)`.

## What Failed

The self-correction idea did not survive contact with the model.

Execution-error retry

| Attempt                            | What happened                                                                                                                                                                                                                                                          |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tool-calling loop                  | Across 672 sampled episodes, base/SFT/GRPO/tool-use checkpoints made 0 `run_sql` calls.                                                                                                                                                                                |
| Forced-revision training           | Kept as analysis tooling; it did not produce the shipped training result.                                                                                                                                                                                              |
| Error-only retry at inference time | Useful as a rule, not as training evidence: retry only when the first SQL execution errors. Spider moved 0.768 -> 0.782 with 40/1034 second generations; BIRD moved 0.405 -> 0.435 over locally decomposable items with 297/1534 second generations. Both had 0 harms. |

The full trail is in `[docs/status.md](docs/status.md)` and `[docs/ARTIFACTS.md](docs/ARTIFACTS.md)`.

## Reproduce

Set up the CPU dev environment and run the normal gate:

```bash
make install
make check
```

Run a tiny no-GPU demo:

```bash
make demo
```

Expected output:

```text
execution_accuracy=0.5000 well_formed_rate=0.0000 (1/2; gold_errors=0)
reasons={'match': 1, 'denotation_mismatch': 1}
```

Re-score the committed artifacts:

```bash
make data
make eval
```

`make eval` is CPU-only and model-free, but it is not a quick smoke test. It executes the committed
SQL completions against Spider and BIRD databases and prints the result tables.

GPU generation and training run through Modal entrypoints in `[modal_app.py](modal_app.py)`. Read
`[docs/RUNBOOK.md](docs/RUNBOOK.md)` before launching paid jobs.

## Caveats

- Accuracy is single-DB execution match, not test-suite multi-DB accuracy. The known single-DB
  false-positive caveat is disclosed in the reports.
- All main evaluations are greedy. The max-token difference between base/SFT and RL evals is
  quantified in `[docs/ARTIFACTS.md](docs/ARTIFACTS.md)`.
- Self-correction code remains tested, but it is not the headline result.

## Docs

- `[docs/architecture.md](docs/architecture.md)`: module map, data flow, reward design.
- `[docs/RUNBOOK.md](docs/RUNBOOK.md)`: setup, data fetches, Modal commands, paid-run discipline.
- `[docs/ARTIFACTS.md](docs/ARTIFACTS.md)`: pins, artifact paths, scores, confidence intervals.
- `[docs/status.md](docs/status.md)`: what is shipped and what remains as analysis code.

## Related Work

This is the build-and-train companion to
`[grpo-gain-decomposition](https://github.com/adamthuvesen/grpo-gain-decomposition)`, which studies
how much measured RL reasoning gain is real improvement versus evaluation artifact.

## Data and License

Spider is CC-BY-SA-4.0. BIRD-dev is CC-BY-NC-4.0. The vendored evaluator is Apache 2.0.
Datasets and databases are pinned and fetched, not committed.

This repository's own code is MIT licensed.
