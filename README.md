# rlvr-sql-agent

Self-correcting text-to-SQL agent: **SFT cold-start → GRPO with execution-verified rewards
(RLVR)** — gains verified across seeds, reward designs, and model families. The model answers a
natural-language question over a database by writing SQL, executing it, reading the result/error,
and revising — with the self-correction *learned* via GRPO against a **verifiable execution
reward** (run the SQL, check the rows), not prompted.

This is the build-and-train companion to
[`grpo-gain-decomposition`](https://github.com/adamthuvesen/grpo-gain-decomposition), which measured
whether RL gains are real.

## Status

Stages 0–2 are done; Stage 3 is in progress. See `openspec/` for the full record.

- **Stage 0 — evaluation foundation**: Spider/BIRD loaders, a read-only SQLite sandbox, and an
  execution-match verifier (parity-tested against the official Test-Suite SQL Eval). The verifier
  that scores the baselines is the same code that later becomes the training reward, so it has to
  be right.
- **Stage 1 — SFT cold-start**: a LoRA adapter taught the `<think>` + ```sql``` format from
  self-generated, execution-filtered traces.
- **Stage 2 — single-turn GRPO**: RL on top of the SFT adapter with an execution reward. Spider-dev
  +10.2pt and BIRD-dev +6.1pt over SFT (3 seeds, the gain CI excludes zero), and the gain holds
  under a reward redesign and on a second model family (Llama-3.2-3B).
- **Stage 3 — multi-turn self-correction** (in progress): the model runs its SQL, reads the
  error or result, and revises before answering — with the revision behavior trained by RL.

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
