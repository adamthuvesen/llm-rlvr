# Runbook: Text-to-SQL RLVR

This runbook covers setup, local scoring, public-summary reproduction, full raw-artifact
re-scoring, and the supported Modal entrypoints. Exact pins and result paths live in
[artifacts.md](artifacts.md).

GPU generation and training can spend money. Get explicit approval for any paid run, state the rough
cost, and start with a small `--limit` before launching a full job. Long Modal runs use
`modal run --detach ... --spawn`; a connected local run can die with the laptop session.

## One-Time Setup

```bash
make install            # CPU env: data, sandbox, verifier, eval
make check              # ruff + format check + unit tests
make data               # fetch, verify, and extract the pinned Spider databases
```

`make data` downloads the pinned Spider database archive, checks its SHA256, and extracts it under
`data/spider/spider_data/` (gitignored). To use an existing extracted tree:

```bash
export TEXT2SQL_SPIDER_ROOT=/path/to/spider_data
```

For local model generation, install the optional generation stack:

```bash
uv sync --extra generate   # transformers + torch; add --extra gpu on a GPU host
```

## Local Generation And Scoring

`generate` is the only model-loading step. It writes a `CompletionSet` with provenance. `score`
reads that artifact with no model.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run llm-rlvr generate \
  --model Qwen/Qwen2.5-Coder-3B-Instruct --split dev \
  --tables-json data/spider/spider_data/tables.json --backend transformers \
  --out runs/baseline_3b_dev50.json --limit 50 --slice-seed 0 --max-new-tokens 512

uv run llm-rlvr score --completions runs/baseline_3b_dev50.json \
  --db-root data/spider/spider_data/database --out results/baseline_3b_dev50.json
```

Few-shot generation uses seeded train exemplars and never samples from dev/test:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run llm-rlvr generate \
  --model Qwen/Qwen2.5-Coder-3B-Instruct --split dev \
  --tables-json data/spider/spider_data/tables.json --backend transformers \
  --out runs/baseline_3b_dev50_fewshot.json --limit 50 --slice-seed 0 \
  --few-shot-k 3 --few-shot-seed 0 --max-new-tokens 512
```

## Modal Entry Points

`modal_app.py` runs generation and training on Modal. The app writes artifacts to the `text2sql-runs`
Volume and uses the same scorer as local evaluation.

Common generation commands:

```bash
modal run modal_app.py::main --dataset spider --limit 8
modal run --detach modal_app.py::main --dataset spider --spawn
modal run --detach modal_app.py::main --dataset spider --few-shot-k 3 --spawn
modal run --detach modal_app.py::main --dataset spider --prompt-style direct --spawn
modal run --detach modal_app.py::main --dataset bird --spawn
modal run --detach modal_app.py::main --dataset bird --prompt-style direct --spawn
```

Pull and score a generated artifact:

```bash
modal volume get text2sql-runs spider-dev-zs.json ./runs/spider-dev-zs.json
uv run llm-rlvr score --completions runs/spider-dev-zs.json \
  --db-root data/spider/spider_data/database
```

## SFT Cold-Start

The SFT adapter teaches the `<think>...</think>` + SQL answer format by self-distillation: sample
Spider-train, keep execution-correct well-formed traces, then train a LoRA adapter on those traces.

```bash
modal run modal_app.py::traces --limit 8
modal run --detach modal_app.py::traces --spawn

modal volume get text2sql-runs sft/candidates-n8-seed0.json ./runs/sft/candidates-n8-seed0.json
uv run llm-rlvr build-traces --candidates runs/sft/candidates-n8-seed0.json \
  --db-root data/spider/spider_data/database \
  --out data/sft/traces.jsonl --manifest results/sft/traces.manifest.json

modal volume put text2sql-runs data/sft/traces.jsonl sft/traces.jsonl
modal run --detach modal_app.py::sft --run-id sft-r32 --model-revision <pinned-commit> --spawn
```

Evaluate the adapter by passing `--adapter` to the same generation command:

```bash
modal run --detach modal_app.py::main --dataset spider --adapter sft/sft-r32 --spawn
modal run --detach modal_app.py::main --dataset bird --adapter sft/sft-r32 --spawn
```

## GRPO Training

GRPO starts from the SFT adapter and scores each rollout by executing SQL in the read-only sandbox.
The main result uses the shaped reward arm, `guard_trivial_empty=True`, four generations per prompt,
and seed-level aggregation.

```bash
modal run modal_app.py::grpo --run-id grpo-small --limit 32 --dev-size 8

for SEED in 0 1 2; do
  modal run --detach modal_app.py::grpo --run-id grpo-s$SEED --learning-rate 5e-5 \
    --seed $SEED --spawn
  sleep 45
done
```

Controls use the same entrypoint:

```bash
modal run --detach modal_app.py::grpo --run-id grpo-nb-s0 --learning-rate 5e-5 \
  --reward-arm near_binary --spawn

LB=unsloth/Llama-3.2-3B-Instruct
modal run --detach modal_app.py::traces --base-model $LB --spawn
modal run --detach modal_app.py::sft --traces sft/traces-llama.jsonl --run-id sft-llama-r32 \
  --base-model $LB --spawn
modal run --detach modal_app.py::grpo --run-id grpo-llama-s0 --warm-start sft/sft-llama-r32 \
  --base-model $LB --learning-rate 5e-5 --spawn
```

Evaluate any adapter with `--adapter` and score the pulled artifact on CPU:

```bash
modal run --detach modal_app.py::main --dataset bird --adapter grpo/grpo-s0 --spawn
modal volume get text2sql-runs bird-dev-zs-grpo-s0.json results/grpo/
uv run llm-rlvr score --completions results/grpo/bird-dev-zs-grpo-s0.json \
  --db-root <bird-db-root> --workers 8
```

## Reproduce Reported Numbers

Run this before claiming the public headline or after changing reporting code:

```bash
make eval
```

`make eval` is CPU-only and model-free; it validates the tracked bundle under `results/public/` and
checks that the README headline table matches it.

For full raw re-scoring, restore the external result bundle under gitignored `results/`, then run:

```bash
make data && make eval-full
```

`make eval-full` is CPU-only and model-free after the matching `CompletionSet` artifacts have been
restored. Wall time is dominated by BIRD execution and depends on available cores.

## Verify The Verifier

The execution-match verifier is parity-tested against the upstream Test-Suite SQL Eval path:

```bash
uv run pytest tests/test_verifier_parity.py -q
```
