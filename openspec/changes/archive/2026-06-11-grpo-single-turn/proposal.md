## Why

Stage 1 produced a warm-started policy (`sft-r32`) that reliably emits the `<think>…</think>` +
```sql``` shape but does **not** improve execution accuracy beyond the base's in-distribution
competence (that was never SFT's job). Stage 2 is the de-risked core of the project and a shippable
deliverable on its own: train that policy with **single-turn GRPO** against a *verifiable*
execution-grounded reward — run the SQL, compare the rows — so the gain comes from the execution
signal, not from a fragile judge. The headline is reported on BIRD-dev (where 3B RL gains are large
and legible), multi-seed with CIs, and — the project's signature — with two first-class controls
that ask *"is the gain real?"*: a **cross-family replication** on Llama-3.2-3B and a
**shaped-vs-near-binary reward ablation**.

## What Changes

- **A composable, execution-grounded reward module** (`src/llm_rlvr_sql/rewards/`). One TRL-shaped
  reward-function signature, composed of four parts — `format` + `executable` + `result_match` +
  `length` (the SQL-R1 shape) — and a deliberately **near-binary Arctic-style arm**
  (`1.0` exact / `0.1` executable / `0` else). The training reward is cheap single-DB
  execution-match with an **empty-result guard** (empty==empty earns full credit only when gold is
  genuinely empty), reusing the Stage-0 sandbox + verifier. The reward arm is selected by config —
  this is what makes the shaped-vs-binary question a first-class ablation, not an assumption.
- **GRPO LoRA training on Modal**, warm-started from the Stage-1 adapter. TRL `GRPOTrainer` + PEFT
  (bf16, `target_modules='all-linear'`, rank 32–64, LoRA LR ≈ 5e-5), vLLM **colocate** rollouts on a
  single A100-80GB, `num_generations` 4, `max_completion_length` 1024–1536, each rollout's SQL
  executed for its reward. **Early-stop off a dev-EX slice**, not a fixed step budget. The base model
  is **config-parametric** (Qwen2.5-Coder-3B-Instruct default; Llama-3.2-3B-Instruct for the
  cross-family control). Adapter + full RL provenance (reward config, GRPO hyperparams, seed,
  warm-start adapter id) persist to the Volume.
- **Multi-seed RL evaluation with controls** (`src/llm_rlvr_sql/eval/` + `report/`). Run the
  headline at **3 seeds**, aggregate at the **seed level** with CIs (reuse the eval-audit /
  seed-level-t pattern), and report the **base → SFT → single-turn-RL** progression on Spider-dev and
  BIRD-dev. Report the **cross-family** arm (Llama, 1 seed — replicates or doesn't, both honest) and
  the **reward-shaping ablation** (shaped vs near-binary, 1 seed each). The eval EX metric is the
  multi-DB **test-suite accuracy** where the training reward is single-DB — demonstrating the
  reward-hacking awareness (the documented ~6.5% single-DB false-positive rate is exactly what
  test-suite eval catches).
- **`make eval` re-derives the RL-vs-SFT-vs-base table** (EX + CIs) on CPU from committed
  `CompletionSet`s, alongside the existing base + SFT rows.
- A `train` extra already carries `trl>=1.3` / `peft` / `accelerate`; this change adds the GRPO
  Modal function + entrypoint and the rewards module. RUNBOOK + ARTIFACTS gain the GRPO procedure,
  the seed policy, and the RL adapters as pinned artifacts.

## Capabilities

### New Capabilities
- `reward-functions`: a composable, execution-grounded reward over `(prompt, completion, gold)` —
  a four-part shaped reward (format, executable, result-match, length) and a near-binary arm,
  selected by config; single-DB execution-match for training with an empty-result guard; reuses the
  Stage-0 sandbox + verifier; deterministic and unit-tested against fixtures.
- `grpo-training`: train a LoRA adapter with TRL `GRPOTrainer` on a Modal A100, warm-started from the
  Stage-1 SFT adapter, vLLM colocate rollouts scored by the reward module, base-model-parametric
  (Qwen default, Llama control), multi-seed, early-stop on a dev-EX slice; persists the adapter +
  full RL provenance to the Volume.
- `rl-evaluation`: multi-seed, seed-level aggregation of RL-policy execution accuracy with CIs on
  Spider-dev + BIRD-dev (the base → SFT → RL progression as the headline on BIRD), the cross-family
  replication report, and the shaped-vs-near-binary reward-ablation report — all re-derivable on CPU
  from committed `CompletionSet`s.

### Modified Capabilities
<!-- None at the requirement level. The base-line adapter-eval path already scores any LoRA
     adapter over the base with the same verifier, so GRPO checkpoints reuse it unchanged; the
     base-model-parametric generation needed for the Llama control is an implementation detail of
     grpo-training/rl-evaluation, not a new baseline-evaluation requirement. -->

## Impact

- **New:** `src/llm_rlvr_sql/rewards/` (the composable reward + the near-binary arm + the
  empty-result guard, reusing `env.sandbox` + `eval.verifier`); `src/llm_rlvr_sql/train/grpo.py`
  (the `GRPOConfig` + trainer wrapper); RL multi-seed aggregation in `eval/` + tables in `report/`.
- **Modified:** `modal_app.py` (add the GRPO training function + entrypoint; vLLM colocate config);
  `eval/generate.py` (base-model-parametric so the Llama control reuses the adapter-eval path);
  `eval/provenance.py` (record reward config, GRPO hyperparams, seed, warm-start adapter id);
  `Makefile` (extend `eval` to re-derive the RL-vs-SFT-vs-base table with CIs); `RUNBOOK.md` +
  `ARTIFACTS.md`.
- **Reused unchanged:** the verifier + test-suite evaluator, the read-only SQLite sandbox, the
  Spider/BIRD loaders, the `CompletionSet`/`ScoreReport`/provenance harness, the adapter-eval path,
  the two-layer Modal image, the Stage-1 `sft-r32` adapter (the warm start).
- **Out of scope:** multi-turn self-correction and the feedback-vs-best-of-N control (Stage 3);
  full-FT (LoRA is decided); a trained reward model or test-time SQL-candidate selection (SQL-R1's
  best-case config, not ours); SynSQL data.
- **Cost:** the headline 3-seed Qwen run (~$50–60), the Llama cross-family arm (~$18), the
  shaped-vs-binary ablation (~$35) — ~$100–115 of GPU, the project's main training spend, gated
  before any Stage-3 commitment.
