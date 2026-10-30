# Runbook: Text-to-SQL RLVR

Use the CPU gate for code changes. Use full reproduction after restoring the external evidence bundle.
Artifact pins and historical result paths live in [artifacts.md](artifacts.md).

Every `modal_app.py` entrypoint spends money on Modal GPUs and refuses to launch without `--approved`.
Pass `--approved` only after Adam approves the run and its rough cost. Start with a smoke before a full run.
Long Modal runs use `modal run --detach`. Long entrypoints spawn by default, so a local disconnect does not kill them.

## Setup

```bash
make install
make check
make help
```

`make install` installs the locked CPU development environment.
`make check` checks the lockfile, Ruff, types, import contracts, unit tests, and the installed wheel.
It runs in seconds, so run it after every change. To iterate on one owner, run its test files
with `uv run --locked pytest tests/<file>.py`.

Fetch the pinned Spider database archive:

```bash
make data
```

The loader verifies the archive, extracts into a temporary tree, validates the tree, then publishes the cache.
To use an existing extracted bundle:

```bash
export TEXT2SQL_SPIDER_ROOT=/path/to/spider_data
```

Overrides retain unknown archive identity unless an archive was actually verified.
`TEXT2SQL_BIRD_ROOT` selects an extracted BIRD bundle in the same way.

Optional environments:

```bash
uv sync --locked --extra generate  # Transformers and Torch
uv sync --locked --extra adapters  # Transformers, Torch, and PEFT without vLLM
uv sync --locked --extra train --extra dev  # Training host
```

GPU work runs only through the Modal entrypoints below.

## Local Generation and Scoring

`generate` loads the model. `score` reads saved completions without a model.
Outputs use fresh paths. Atomic publication prevents partial single-file outputs, and existing paths fail with a nonzero exit.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --locked llm-rlvr generate \
  --model Qwen/Qwen2.5-Coder-3B-Instruct --split dev \
  --tables-json data/spider/spider_data/tables.json --backend transformers \
  --out runs/baseline_3b_dev50.json --limit 50 --slice-seed 0 --max-new-tokens 512

uv run --locked llm-rlvr score --completions runs/baseline_3b_dev50.json \
  --db-root data/spider/spider_data/database --out results/baseline_3b_dev50.score.json
```

Few-shot generation uses seeded train exemplars:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --locked llm-rlvr generate \
  --model Qwen/Qwen2.5-Coder-3B-Instruct --split dev \
  --tables-json data/spider/spider_data/tables.json --backend transformers \
  --out runs/baseline_3b_dev50_fewshot.json --limit 50 --slice-seed 0 \
  --few-shot-k 3 --few-shot-seed 0 --max-new-tokens 512
```

Local `--limit` uses seeded selection. Remote `--limit` uses a prefix. Provenance records the selection policy.
Score reports retain reference-query errors in the denominator and expose per-item diagnostics.

## Modal Entry Points

The Modal app writes to the `text2sql-runs` Volume.
Generation runs use `generation/<run-id>/`. Candidate runs use `traces/<run-id>/`.
A missing run ID uses a model tag and request digest. Existing run directories fail before model loading.
Choose a fresh run ID when repeating a run.

```bash
modal run modal_app.py::main --dataset spider --limit 8 --run-id spider-dev-smoke --approved
modal run --detach modal_app.py::main --dataset spider --approved
modal run --detach modal_app.py::main --dataset spider --few-shot-k 3 --approved
modal run --detach modal_app.py::main --dataset spider --prompt-style direct --approved
modal run --detach modal_app.py::main --dataset bird --approved
```

Managed directories publish `request.json` before execution and `complete.json` after output validation.
Restore all files from a managed directory before consuming its output:

```bash
mkdir -p runs/spider-dev-smoke
modal volume get text2sql-runs generation/spider-dev-smoke/request.json runs/spider-dev-smoke/request.json
modal volume get text2sql-runs generation/spider-dev-smoke/completions.json runs/spider-dev-smoke/completions.json
modal volume get text2sql-runs generation/spider-dev-smoke/complete.json runs/spider-dev-smoke/complete.json
uv run --locked llm-rlvr score --completions runs/spider-dev-smoke/completions.json \
  --db-root data/spider/spider_data/database
```

Readers verify the completion marker and output hashes. An interrupted managed run remains incomplete.
Use a fresh run after resolving its failure. Do not create a completion marker manually.
Legacy unmanaged artifact files remain readable.

## SFT Cold-Start

Sample Spider-train, filter execution-correct well-formed traces, then train the LoRA adapter.

```bash
modal run modal_app.py::traces --limit 8 --run-id trace-smoke --approved
modal run --detach modal_app.py::traces --approved

mkdir -p runs/trace-smoke
modal volume get text2sql-runs traces/trace-smoke/request.json runs/trace-smoke/request.json
modal volume get text2sql-runs traces/trace-smoke/candidates.json runs/trace-smoke/candidates.json
modal volume get text2sql-runs traces/trace-smoke/complete.json runs/trace-smoke/complete.json
uv run --locked llm-rlvr build-traces --candidates runs/trace-smoke/candidates.json \
  --db-root data/spider/spider_data/database \
  --out data/sft/traces.jsonl --manifest results/sft/traces.manifest.json

modal volume put text2sql-runs data/sft/traces.jsonl sft/traces.jsonl
modal volume put text2sql-runs results/sft/traces.manifest.json sft/traces.manifest.json
modal run --detach modal_app.py::sft --run-id sft-r32 --model-revision <pinned-commit> \
  --trace-manifest sft/traces.manifest.json --approved
```

Trace filtering rejects source metadata mismatches. New traces retain typed source examples.
SFT requires the matching manifest, trace hash, source lineage, and zero reference-query errors before model loading.
Historical trace files can remain readable while being insufficient for a new training run.
Periodic `checkpoint-*` adapters receive their own completion markers and checkpoint-specific provenance after successful training.
Restore each checkpoint's request, provenance, weights, config, and completion marker before evaluation.
An interrupted checkpoint remains incomplete even when its parent directory has a completion marker.

Evaluate an adapter with the generation entrypoint:

```bash
modal run --detach modal_app.py::main --dataset spider --adapter sft/sft-r32 --approved
modal run --detach modal_app.py::main --dataset bird --adapter sft/sft-r32 --approved
```

## GRPO Training

GRPO starts from SFT and scores rollouts through the same SQL verifier as CPU evaluation.
The SQL training path requires `guard_trivial_empty=True`.
Training configuration rejects unsupported TRL fields before training and records requested and effective settings.
Warm-start LoRA settings come from the inspected adapter.

```bash
modal run modal_app.py::grpo --run-id grpo-small --limit 32 --dev-size 8 --approved

for SEED in 0 1 2; do
  modal run --detach modal_app.py::grpo --run-id grpo-s$SEED --learning-rate 5e-5 \
    --seed $SEED --approved
  sleep 45
done
```

GRPO records source identities for its train/dev split.
SFT exposure remains unknown for an old adapter without source lineage. Do not infer a clean SFT holdout from a GRPO split alone.

Controls use the same entrypoint:

```bash
modal run --detach modal_app.py::grpo --run-id grpo-nb-s0 --learning-rate 5e-5 \
  --reward-arm near_binary --approved

LB=unsloth/Llama-3.2-3B-Instruct
modal run --detach modal_app.py::traces --base-model $LB --approved
modal run --detach modal_app.py::sft --traces sft/traces-llama.jsonl \
  --trace-manifest sft/traces-llama.manifest.json --run-id sft-llama-r32 --base-model $LB --approved
modal run --detach modal_app.py::grpo --run-id grpo-llama-s0 --warm-start sft/sft-llama-r32 \
  --base-model $LB --learning-rate 5e-5 --approved
```

Create and upload each family's filtered trace dataset and manifest before its SFT run.
The model revision and adapter identities must match.

## Reproduce Reported Numbers

```bash
make eval
```

Public mode validates tracked counts, claim references, README headline rows, and seed-level tables.
It requires no model, network, database, or raw completion artifacts.

Restore the external completion/report pairs and matching databases, then run:

```bash
make eval-full
```

Full mode executes the tracked `report/full.py` inventory. Every completion requires its historical `.score.json` report.
It reports all missing files before scoring. It validates the whole inventory, provenance, optional hash pins, and database paths.
It preserves historical reports and prints counts, accuracy deltas, reference errors, and available per-item changes.
Legacy hash pins and outcomes remain unknown when the saved evidence lacks them.

Use explicit database roots when several extracted bundles exist:

```bash
uv run --locked llm-rlvr reproduce --mode full \
  --spider-db-root /path/to/database --bird-db-root /path/to/dev_databases --workers 8
```

Environment overrides refer to extracted bundle roots. CLI database-root arguments refer to database subdirectories.
Full mode never downloads missing data. Fetch or restore data first.
Score changes or reference-query failures return nonzero after the complete comparison.
Missing or malformed inputs return nonzero without partial result output.

## Additional Verification

```bash
make package-check
make runtime-check
make test-integration
```

The wheel check installs locked CPU dependencies outside the checkout.
It imports every package module while blocking optional model dependencies, then runs CLI help, public evidence, and the SQLite demo.
CI configures the CPU gate and wheel check on Python 3.11 and 3.14.

The runtime check constructs real LoRA and SFT configurations without downloading a model or executing GPU training.
Missing training dependencies fail with an install command. They never become a skipped success.
Dataset integration tests use the actual pinned loaders and databases.
See [verification.md](verification.md) for the verification rubric and evidence limits.
