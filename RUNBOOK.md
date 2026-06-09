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

The 50-example slice above is a **local smoke** (n=50 has a wide CI); the headline numbers come from
the full-dev Modal GPU pass below.

## The headline baseline (full dev + BIRD — Modal GPU)

`modal_app.py` runs the same `generate()` on a single A100 via vLLM and writes the `CompletionSet`
to a Modal Volume. Run with the `modal` CLI (`uv tool install modal`); public HF data/model, no token.

```bash
modal run modal_app.py --dataset spider --limit 8                # day-1 smoke first
modal run --detach modal_app.py --dataset spider --spawn         # Spider-dev zero-shot (<think>)
modal run --detach modal_app.py --dataset spider --few-shot-k 3 --spawn
modal run --detach modal_app.py --dataset spider --prompt-style direct --spawn
modal run --detach modal_app.py --dataset bird --spawn           # BIRD-dev (headline)
modal run --detach modal_app.py --dataset bird --prompt-style direct --spawn

# pull a result from the Volume and score on CPU
modal volume get text2sql-runs spider-dev-zs.json ./runs/spider-dev-zs.json
uv run text2sql-rlvr score --completions runs/spider-dev-zs.json --db-root data/spider/spider_data/database
```

**Recorded base-model baseline** (Qwen2.5-Coder-3B-Instruct, greedy, full dev):

| Benchmark | `<think>` zero-shot | direct zero-shot | few-shot k=3 (`<think>`) |
| --- | --- | --- | --- |
| Spider-dev (1034) | 0.625 | **0.669** | 0.677 |
| BIRD-dev (1534) | 0.313 | **0.351** | — |

**Methodology — two prompt styles, on purpose.** The un-trained base reasons *counterproductively*:
the `<think>` chain-of-thought prompt depresses it (long rambling answers, ~1200 chars), while the
**direct** "output only SQL" prompt and the few-shot demonstrations both suppress the reasoning
(~130-char answers) and score higher. So we report two base numbers:
- **direct** (0.669 / 0.351) — the fair, literature-comparable standalone base capability;
- **`<think>` matched** (0.625 / 0.313) — the before-RL anchor, since the RL model will use `<think>`.

Reporting both prevents an artificially-low baseline from inflating the later RL gain. (Both sit
below SQL-R1's ~77% base, which adds schema-linking + a tuned harness we deliberately keep minimal.)
BIRD few-shot is deferred (it needs BIRD-train exemplars — a small follow-up).

## Reproduce the committed numbers (CPU only, no model)

Every headline number is a deterministic re-score of a committed `CompletionSet`:

```bash
make data && make eval     # re-derives the full table above
```

## Verify the verifier (parity)

The execution-match verifier is parity-tested against the upstream Test-Suite SQL Eval on every
run of the suite:

```bash
uv run pytest tests/test_verifier_parity.py -q
```
