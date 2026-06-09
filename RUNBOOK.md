# Runbook: evaluation foundation (Stage 0) + SFT cold-start (Stage 1)

The procedure for fetching the data, running the base-model baseline, building the cold-start
trace dataset, training the LoRA SFT adapter, and reproducing the headline numbers. Everything
here is CPU/MPS-runnable except the GPU passes (noted below). Exact external pins live in
[`ARTIFACTS.md`](ARTIFACTS.md).

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
modal run modal_app.py::main --dataset spider --limit 8          # day-1 smoke first
modal run --detach modal_app.py::main --dataset spider --spawn   # Spider-dev zero-shot (<think>)
modal run --detach modal_app.py::main --dataset spider --few-shot-k 3 --spawn
modal run --detach modal_app.py::main --dataset spider --prompt-style direct --spawn
modal run --detach modal_app.py::main --dataset bird --spawn     # BIRD-dev (headline)
modal run --detach modal_app.py::main --dataset bird --prompt-style direct --spawn

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

## Stage 1: SFT cold-start (trace-gen → filter → SFT → adapter eval)

The cold-start instills the `<think>…</think>` + ```sql``` format by self-distillation: sample the
base on Spider-train, keep only traces whose SQL execution-matches gold (the Stage-0 verifier is
the data filter), LoRA-SFT on those, and evaluate the adapter with the identical pipeline.

**1. Sample candidates** (GPU; N=8 per train question at temperature 0.8):

```bash
modal run modal_app.py::traces --limit 8                  # day-1 smoke first
modal run --detach modal_app.py::traces --spawn           # full Spider-train (~7k questions x 8)
```

**2. Filter into the trace dataset** (CPU; needs `make data`). Pull the CandidateSet, keep
execution-correct + well-formed traces (shortest per question, exact-deduped; the trivial-empty
guard is ON), and write the JSONL + manifest:

```bash
modal volume get text2sql-runs sft/candidates-n8-seed0.json ./runs/sft/candidates-n8-seed0.json
uv run text2sql-rlvr build-traces --candidates runs/sft/candidates-n8-seed0.json \
  --db-root data/spider/spider_data/database \
  --out data/sft/traces.jsonl --manifest results/sft/traces.manifest.json
```

The manifest (kept/total counts + full sampling provenance) is committed under `results/sft/`;
the JSONL bulk stays local/Volume (gitignored).

**3. Train the LoRA adapter** (GPU; rank 32, all-linear, bf16, completion-only loss). Push the
traces to the Volume, smoke a tiny run first, then the real one with the model commit pinned:

```bash
modal volume put text2sql-runs data/sft/traces.jsonl sft/traces.jsonl
modal run modal_app.py::sft --epochs 1 --run-id sft-smoke     # day-1 smoke: loads + trains + saves
modal run --detach modal_app.py::sft --run-id sft-r32 --model-revision <pinned-commit> --spawn
```

The adapter + `provenance.json` (base model + revision, LoRA config, trace-dataset SHA256,
library versions, git commit) land under `sft/<run-id>/` on the Volume.

**4. Evaluate the adapter** — same pipeline as the baseline, `--adapter` is the only difference
(vLLM loads the LoRA over the base; the adapter is recorded in provenance):

```bash
modal run --detach modal_app.py::main --dataset spider --adapter sft/sft-r32 --spawn
modal run --detach modal_app.py::main --dataset spider --adapter sft/sft-r32 --prompt-style direct --spawn
modal run --detach modal_app.py::main --dataset bird --adapter sft/sft-r32 --spawn

modal volume get text2sql-runs spider-dev-zs-sft-r32.json ./runs/spider-dev-zs-sft-r32.json
uv run text2sql-rlvr score --completions runs/spider-dev-zs-sft-r32.json \
  --db-root data/spider/spider_data/database
```

`score` now reports `well_formed_rate` alongside execution accuracy. **Success bar (D6):**
format rate near-100% on dev AND no EX regression below the base `<think>` anchor (0.625
Spider-dev / 0.313 BIRD-dev; target ≥ the direct anchor). If the bar is missed, the fallback
ladder is documented in `openspec/changes/sft-coldstart/design.md` (more traces → SynSQL pull →
teacher distillation).

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
