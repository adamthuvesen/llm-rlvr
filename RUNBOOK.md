# Runbook: eval foundation (Stage 0) → SFT cold-start (Stage 1) → single-turn GRPO (Stage 2) → forced-revision (Stage 3b, aborted at gate)

The procedure for fetching the data, running the base-model baseline, building the cold-start
trace dataset, training the LoRA SFT adapter, running single-turn GRPO with its controls, and
reproducing the headline numbers. Everything here is CPU/MPS-runnable except the GPU passes
(noted below). Exact external pins live in [`ARTIFACTS.md`](ARTIFACTS.md).

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

## Stage 2: single-turn GRPO (smoke → LR pilot → headline → controls)

GRPO trains the SFT adapter further with an execution-grounded reward: 4 rollouts per question
(vLLM colocate on one A100-80GB), each scored by running its SQL and comparing rows to gold
(`rewards/reward.py` — shaped arm by default, empty-result guard ON, full format credit only
when `<think>` leads). The spend ladder is mandatory: **smoke → pilot → paid runs**, each step
validating the next one's assumptions.

**Modal discipline** (learned the hard way): long runs use `modal run --detach … --spawn` — a
plain `modal run` dies with your laptop's connection and takes the run with it. Stagger app
creations by ~45s (3 simultaneous creations hit a rate limit). The HF "set HF_TOKEN for higher
rate limits" warning is benign noise for our public/ungated repos.

**1. Smoke** (~$0.50, ~10 min): proves the rig end-to-end (colocate fits, TRL↔vLLM logprobs
agree, config keys accepted) and measures step time for cost extrapolation:

```bash
modal run modal_app.py::grpo --run-id grpo-smoke --limit 32 --dev-size 8
```

**2. LR pilot** (~$5): short runs at 3 learning rates on a train subset, evaled on a Spider-dev
slice; pick by measured dev EX. Ours: `{5e-6, 1e-5, 5e-5}` → monotonic, **locked LR = 5e-5**.

```bash
modal run --detach modal_app.py::grpo --run-id grpo-pilot-lr5e5 --learning-rate 5e-5 \
  --limit 2000 --spawn   # repeat per LR, staggered
```

**3. Headline** (~$22): 3 seeds at the locked LR on full Spider-train — the **seed policy**:
never fewer than 3 seeds for a CI claim, never pool rollouts across seeds, aggregate per-seed EX
with a Student-t CI (`eval/aggregate.py`), success bar = the RL-vs-SFT gain CI excludes zero.

```bash
for SEED in 0 1 2; do
  modal run --detach modal_app.py::grpo --run-id grpo-s$SEED --learning-rate 5e-5 \
    --seed $SEED --spawn
  sleep 45
done
```

**4. Controls** (~$32): the reward ablation re-runs one seed with `--reward-arm near_binary`
(the shaped arm reuses headline seed 0 — already paid). The cross-family control runs the FULL
matched pipeline on Llama-3.2-3B — traces → SFT → GRPO, same configs, only the family changes:

```bash
modal run --detach modal_app.py::grpo --run-id grpo-nb-s0 --learning-rate 5e-5 \
  --reward-arm near_binary --spawn

LB=unsloth/Llama-3.2-3B-Instruct    # ungated mirror, byte-identical to meta-llama (no token)
modal run --detach modal_app.py::traces --base-model $LB --spawn
modal volume get text2sql-runs sft/candidates-llama-3.2-3b-instruct-n8-seed0.json runs/sft/
uv run text2sql-rlvr build-traces --candidates runs/sft/candidates-llama-3.2-3b-instruct-n8-seed0.json \
  --db-root data/spider/spider_data/database \
  --out data/sft/traces-llama.jsonl --manifest results/sft/traces-llama.manifest.json
modal volume put text2sql-runs data/sft/traces-llama.jsonl sft/traces-llama.jsonl
modal run --detach modal_app.py::sft --traces sft/traces-llama.jsonl --run-id sft-llama-r32 \
  --base-model $LB --spawn
modal run --detach modal_app.py::grpo --run-id grpo-llama-s0 --warm-start sft/sft-llama-r32 \
  --base-model $LB --learning-rate 5e-5 --spawn
```

**5. Evaluate any adapter** — identical pipeline, `--adapter` (+ `--base-model` for Llama; eval
artifacts get a family tag so they can't collide). Score on CPU with `--workers 8`:

```bash
modal run --detach modal_app.py::main --dataset bird --adapter grpo/grpo-s0 --spawn
modal volume get text2sql-runs bird-dev-zs-grpo-s0.json results/grpo/
uv run text2sql-rlvr score --completions results/grpo/bird-dev-zs-grpo-s0.json \
  --db-root <bird-db-root> --workers 8
```

**Colocate OOM fallback ladder** (untouched — the smoke fit at `gpu_memory_utilization=0.3`):
lower `vllm_gpu_memory_utilization` → halve `per_device_train_batch_size` (raise grad-accum to
match) → cap `max_completion_length` at 1024 → last resort, `num_generations` 4→2 (costs
group-relative signal; re-pilot before trusting results).

**Training health, what to watch** (from the headline + ablation logs): `kl` drifting past ~0.2
→ rely on dev-EX checkpoint selection, don't panic-stop; `frac_reward_zero_std` rising → groups
are saturating (expected under near-binary; a shaped-arm rise means the dev slice is too easy);
`clipped_ratio` > 0 → completions hitting the cap, raise it.

## Stage 3b: forced-revision GRPO (smoke → pilot → ABORTED at the pre-registered gate)

Stage 3 (tool election) closed without training — the 3B policy never calls an offered tool
(0/672; see ARTIFACTS). Stage 3b removed election: every attempt is executed unconditionally,
the gold-free feedback is spliced in loss-masked, and the policy revises (hand-written TRL
`rollout_func`; see `train/forced_revision.py`). The ladder and its verdicts:

```bash
modal run modal_app.py::fr_smoke                  # ~$1: mask + IS-ratio + no-NaN assertions — PASSED
modal run --detach modal_app.py::fr_train --run-id grpo-fr-pilot \
  --limit 2000 --max-steps 60 --spawn              # ~$1.5 pilot — GATE FAILED (P1/P2), ladder aborted
modal run --detach modal_app.py::fr_eval --dataset spider \
  --adapter grpo-fr/grpo-fr-pilot --limit 200 --spawn   # forced-revision inference harness
# best-of-N control: two sampled single-turn passes + gold-free CPU selection
modal run --detach modal_app.py::main --dataset spider --adapter grpo-fr/grpo-fr-pilot \
  --limit 200 --max-new-tokens 1024 --temperature 1.0 --sampling-seed 0 --spawn   # and seed 1
uv run text2sql-rlvr select-bestofn --completions results/grpo-fr/spider-dev-zs-grpo-fr-pilot-t1.0-seed{0,1}-n200.json \
  --db-root data/spider/spider_data/database \
  --out results/grpo-fr/spider-dev-bo2-grpo-fr-pilot-n200.json \
  --report results/grpo-fr/spider-dev-bo2-grpo-fr-pilot-n200.selection.json
uv run text2sql-rlvr fr-metrics --completions results/grpo-fr/spider-dev-fr-grpo-fr-pilot-n200.json \
  --transcripts results/grpo-fr/spider-dev-fr-grpo-fr-pilot-n200.transcripts.json \
  --db-root data/spider/spider_data/database   # fix/harm decomposition of the revision pass
```

The pilot's continue/abort thresholds (P1–P6) were committed to the change's `tasks.md` BEFORE
the pilot ran — same discipline as the probes. P1 (turn-1 quality) and P2 (delta-bonus floor)
failed: GRPO drifts toward identity-resubmission instead of amplifying fixes, so the full run
was never bought. `make eval` re-derives the n=200 documentation table.

## Reproduce the committed numbers (CPU only, no model)

Every number in this repo — baseline, SFT, the GRPO headline with CIs, both controls — is a
deterministic re-score of a committed `CompletionSet`:

```bash
make data && make eval     # ~9 min: 20 sets, threaded scoring + the 4 progression tables
```

## Verify the verifier (parity)

The execution-match verifier is parity-tested against the upstream Test-Suite SQL Eval on every
run of the suite:

```bash
uv run pytest tests/test_verifier_parity.py -q
```
