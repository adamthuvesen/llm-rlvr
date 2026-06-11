# rlvr-sql-agent

Self-correcting text-to-SQL agent: **SFT cold-start → GRPO with execution-verified rewards
(RLVR)** — gains verified across seeds, reward designs, and model families. The model answers a
natural-language question over a database by writing SQL, executing it, reading the result/error,
and revising — with the self-correction *learned* via GRPO against a **verifiable execution
reward** (run the SQL, check the rows), not prompted.

This is the build-and-train companion to
[`grpo-gain-decomposition`](https://github.com/adamthuvesen/grpo-gain-decomposition) (which measured
whether RL gains are real). The throughline: *I build RL agents — and I verify the gains are real.*

## Status

**Stage 0 — evaluation foundation** (this layer): Spider/BIRD loaders, a read-only SQLite sandbox,
an execution-match verifier (parity-tested against the official Test-Suite SQL Eval), and a
reproducible "before RL" baseline. The verifier that scores the baseline is the same one that
becomes the Stage-2 training reward, so its correctness is load-bearing.

Later stages (SFT cold-start, single-turn GRPO, multi-turn self-correction) are tracked in
`openspec/`.

## Layout

```
src/text2sql_rlvr/
  data/         Spider/BIRD loaders, typed examples, schema serialization
  env/          read-only SQLite sandbox (the safety boundary)
  eval/         execution-match verifier, generation, scoring, provenance
  third_party/  vendored Test-Suite SQL Eval (execution-match path only; Apache 2.0)
```

## Usage

```bash
make install   # CPU env: data, env, verifier, eval (no GPU stack)
make check     # ruff + unit tests (the Stage-0 gate)
make test-integration   # network/HuggingFace/model tests (deselected by default)
```

The baseline runs through the `text2sql-rlvr` CLI: `generate` writes a completion set (the only
model-loading step), then `score` reads it on a CPU box. See `RUNBOOK.md`.

## Data & licensing

Spider is CC-BY-SA-4.0 (attribute + share-alike); the vendored evaluator is Apache 2.0 (see
`src/text2sql_rlvr/third_party/test_suite_sql_eval/`). Datasets and databases are pinned + fetched,
never committed.

## License

MIT (this repository's own code).
