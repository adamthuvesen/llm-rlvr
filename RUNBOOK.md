# Runbook: Stage-0 evaluation foundation

The procedure for fetching the data, running the base-model baseline, and reproducing the headline
number. Everything here is CPU/MPS-runnable except the full-dev GPU pass (noted below). Exact
external pins live in [`ARTIFACTS.md`](ARTIFACTS.md).

## One-time setup

```bash
make install            # CPU env (data, sandbox, verifier, eval) — no GPU stack
make check              # ruff + unit tests (the Stage-0 gate)
make data               # fetch + verify (SHA256) + extract the pinned Spider databases
```

`make data` downloads `spider_data.zip` from the pinned HF dataset commit (see `ARTIFACTS.md`),
checks its SHA256, and extracts to `data/spider/spider_data/` (gitignored). To run against an
already-extracted tree instead (offline), point at it:

```bash
export TEXT2SQL_SPIDER_ROOT=/path/to/spider_data   # skips the download
```

For generation you also need the model stack:

```bash
uv sync --extra generate   # transformers + torch (CPU/MPS); add --extra gpu for vLLM on a GPU box
```

## Run the baseline (zero-shot + few-shot, dev slice — CPU/MPS)

`generate` is the only model-loading step; it writes a `CompletionSet`. `score` reads it with no
model. Decoding settings, dataset revision, and the DB-archive checksum are recorded in provenance.

```bash
# zero-shot
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run text2sql-rlvr generate \
  --model Qwen/Qwen2.5-Coder-3B-Instruct --split dev \
  --tables-json data/spider/spider_data/tables.json --backend transformers \
  --out runs/baseline_3b_dev50.json --limit 50 --slice-seed 0 --max-new-tokens 512

# few-shot (k=3 seeded train exemplars; never dev/test — enforced)
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run text2sql-rlvr generate \
  --model Qwen/Qwen2.5-Coder-3B-Instruct --split dev \
  --tables-json data/spider/spider_data/tables.json --backend transformers \
  --out runs/baseline_3b_dev50_fewshot.json --limit 50 --slice-seed 0 \
  --few-shot-k 3 --few-shot-seed 0 --max-new-tokens 512

# score either run against the real databases (model-free, deterministic)
uv run text2sql-rlvr score --completions runs/baseline_3b_dev50.json \
  --db-root data/spider/spider_data/database --out results/baseline_3b_dev50.json
```

Recorded result (Coder-3B-Instruct, 50-example dev slice, greedy):

| Setting | Spider-dev EX | reasons |
| --- | --- | --- |
| zero-shot | **0.76** (38/50) | match 38, denotation_mismatch 7, pred_error 5 |
| few-shot (k=3) | **0.74** (37/50) | match 37, denotation_mismatch 10, pred_error 3 |

The zero-shot number matches SQL-R1's published base (~77% Spider-dev). Few-shot is flat-to-slightly
lower, as expected: the exemplars are fixed random cross-schema demonstrations (a clean reference,
**not** a retriever), so they shouldn't lift a strong instruct model. n=50 has a wide CI — this is
the wiring-at-scale proof, not the final figure. The chosen exemplars are recorded in each run's
provenance (`few_shot_exemplars`), drawn from `train` only (leakage-guarded).

## Reproduce the committed number (CPU only, no model)

`make eval` re-scores the committed `CompletionSet` at `results/baseline/completions.json` against
the fetched databases — deterministic, so a clean checkout reproduces it exactly:

```bash
make data && make eval     # -> execution_accuracy=0.7600 (38/50)
```

## The proper run (full dev + BIRD — GPU)

A full Spider-dev (1034) pass on MPS is hours; run it on a GPU. Same commands, drop `--limit`, and
use `--backend vllm` on a GPU box (`uv sync --extra gpu`). Report **BIRD-dev** alongside Spider —
the base is already ~76% on Spider, so the legible RL headroom is on BIRD (design decision). This
GPU pass is the Stage-2-onward baseline; wire it through Modal when the training runner lands.

## Verify the verifier (parity)

The execution-match verifier is parity-tested against the upstream Test-Suite SQL Eval on every
run of the suite:

```bash
uv run pytest tests/test_verifier_parity.py -q
```
