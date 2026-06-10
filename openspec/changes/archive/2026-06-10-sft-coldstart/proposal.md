## Why

Stage 2 GRPO needs a warm-started policy that already emits the `<think>…</think>` + ```sql``` format
it will reward — SCoRe's lesson is that a weak init makes self-correction RL collapse. The base
Qwen2.5-Coder-3B-Instruct does not reliably produce that format (the Stage-0 finding: the `<think>`
prompt makes it ramble and *lose* accuracy). Stage 1 is the short SFT cold-start that fixes the
behavior — and we build its training data the project's own way: sample the base, keep only the
traces whose SQL **execution-matches gold** (the Stage-0 verifier), and SFT on those. The verifier
that measures the gain also bootstraps the model.

## What Changes

- **Self-distilled, execution-filtered SFT dataset.** Run the existing Modal vLLM `generate()` on
  Spider-train with the `<think>` prompt (small-N sampling at modest temperature), score every
  completion with the Stage-0 verifier, and keep the execution-correct `(prompt, <think>+SQL)` pairs
  as a typed, provenance-stamped SFT dataset. No external teacher, no gold-imitation — the policy's
  own correct reasoning, in its own distribution.
- **LoRA SFT on Modal.** A new TRL `SFTTrainer` + PEFT LoRA training function on a single A100-80GB
  (bf16, `target_modules='all-linear'`, rank 32–64), trained on the trace dataset, saving the
  **adapter** to the `text2sql-runs` Volume with full provenance (base model, LoRA config, data
  hash, git commit).
- **Adapter evaluation.** The eval/generate path gains the ability to load a LoRA adapter over the
  base, so the SFT checkpoint is scored with the *same* verifier and we report **SFT-vs-base** EX +
  a well-formed-format rate on Spider-dev and BIRD-dev.
- **A `train` dependency extra** (`trl>=1.3`, `peft`, `accelerate`) and a `src/text2sql_rlvr/train/`
  module. RUNBOOK + ARTIFACTS updated with the SFT procedure and the checkpoint as a pinned artifact.
- Documented fallbacks if the checkpoint is weak (low format rate or EX regression): a small SynSQL
  pull, or teacher-distilled traces. Out of scope here.

## Capabilities

### New Capabilities
- `coldstart-trace-dataset`: build a self-distilled, execution-filtered SFT dataset from Spider-train
  by sampling the base model and keeping only execution-correct `<think>`+SQL traces, serialized as
  typed records with provenance.
- `sft-training`: train a LoRA adapter on the trace dataset on a Modal A100 (TRL `SFTTrainer` + PEFT,
  bf16), persisting the adapter + training provenance to the Volume.

### Modified Capabilities
- `baseline-evaluation`: extend the evaluation path to score a fine-tuned **adapter checkpoint**
  (load a LoRA adapter over the base), and to report a well-formed-format rate alongside execution
  accuracy — so the cold-start is measured against the base with the same verifier.

## Impact

- **New:** `src/text2sql_rlvr/train/` (LoRA SFT config + trainer wrapper, the trace-dataset builder
  reusing `eval.generate` + `eval.score`, typed SFT records); a `train` optional-dependency extra.
- **Modified:** `modal_app.py` (add the SFT training function + entrypoint; add an adapter arg to the
  generation function); `eval/generate.py` (load a LoRA adapter over the base in both the vLLM and
  transformers backends); `eval/provenance.py` (record base+adapter, LoRA config, trace-data hash);
  `pyproject.toml` / `uv.lock`; `Makefile` (extend `eval` to cover the SFT checkpoint); `RUNBOOK.md`
  + `ARTIFACTS.md`.
- **Reused unchanged:** the verifier, the SQLite sandbox, the score/provenance harness, the Spider
  loader, the two-layer Modal image pattern.
- **Out of scope:** GRPO / the reward module, multi-turn, the cross-family + reward-shaping controls
  (Stage 2+); SynSQL and teacher-distillation (documented fallbacks only).
- **First training spend** (~$10–20): the project pivots from measuring to training here.
