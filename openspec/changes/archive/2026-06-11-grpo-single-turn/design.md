## Context

Stages 0–1 produced a trustworthy execution-match verifier, an honest base baseline, and the
`sft-r32` warm-start adapter (Spider-dev `<think>` 0.678, BIRD-dev 0.344, well-formed-format
99.2% / 96.0%). SFT instilled the format and locked in in-distribution competence; it did **not**
lift EX much past the base — by design. Stage 2 is where the gain comes from: single-turn GRPO from
the SFT policy against a *verifiable* execution reward. It is the de-risked, shippable core (the
fallback if multi-turn proves hard), so it carries the project's two signature controls — a
cross-family replication and a reward-shaping ablation.

This change reuses the Stage-0/1 harness wholesale (verifier, sandbox, loaders,
`CompletionSet`/`ScoreReport`/provenance, adapter-eval path, two-layer Modal image) and adds three
units: a composable reward module, a GRPO training function, and multi-seed RL aggregation + the
controls reporting.

## Goals / Non-Goals

**Goals:**
- A GRPO-trained LoRA adapter over Qwen2.5-Coder-3B-Instruct (warm-started from `sft-r32`) that
  improves **BIRD-dev** (and Spider-dev) execution accuracy over the SFT anchor, **3 seeds**, with a
  seed-level CI that excludes zero.
- The reward is verifiable execution-match, composable, and arm-selectable (shaped vs near-binary) —
  so the shaped-vs-binary debate is adjudicated on our own setup, not assumed.
- The cross-family control (Llama-3.2-3B-Instruct, 1 seed) is reported — replicates or doesn't, both
  honest — using the same GRPO recipe and the base-parametric eval path.
- Reward-hacking awareness is demonstrated: train on cheap single-DB execution-match with an
  empty-result guard; report the stricter multi-DB **test-suite** accuracy at eval.
- The full base → SFT → RL progression (+ controls) re-derives on CPU from committed artifacts via
  `make eval`.

**Non-Goals:**
- Multi-turn self-correction, the feedback-vs-best-of-N control (Stage 3).
- A trained reward model or test-time SQL-candidate selection (SQL-R1's *best-case* config; we run
  bare GRPO with a programmatic reward).
- Full-FT (LoRA is decided); QLoRA (weakens rollouts, breaks vLLM tooling).
- Beating published SOTA — the deliverable is an honest, controlled *gain*, headlined on BIRD.

## Decisions

### D1 — GRPO LoRA, warm-started from `sft-r32`, vLLM colocate on one A100-80GB
TRL `GRPOTrainer` + PEFT LoRA (bf16, `target_modules='all-linear'`, rank 32 / alpha 64), warm-started
by loading the Stage-1 adapter as the initial policy, vLLM **colocate** rollouts (the supported
single-GPU path; the colocate+PEFT hang is multi-GPU only). Each rollout's SQL is executed by the
Stage-0 verifier for its reward.
- *Why:* RL is the regime where LoRA provably matches full-FT ("LoRA Without Regret", reproduced on
  3B) at ~½ the trainer memory; a 3B + colocate vLLM fits one A100-80GB with headroom. Warm-starting
  from `sft-r32` (not the base) is the SCoRe lesson — a weak init makes RL unstable — and keeps the
  GRPO KL shock small since the traces were already in-distribution.
- *Alternatives:* full-FT (rejected, plan); base warm-start (rejected — undoes Stage 1); vLLM server
  mode on a 2nd GPU (the fallback if colocate OOMs, not the default).

### D2 — Composable reward: shaped 4-part + a near-binary arm, single-DB train / test-suite eval
`src/llm_rlvr_sql/rewards/` exposes one TRL reward-function signature over `(prompts, completions,
gold)`. The **shaped** arm (SQL-R1) sums `format` + `executable` + `result_match` + `length`; the
**near-binary** arm (Arctic) returns `1.0` exact / `0.1` executable / `0` else. The arm and its
weights are config, not hard-coded. The **training** reward uses cheap **single-DB** execution-match
(reusing `env.sandbox` + `eval.verifier`) with an **empty-result guard**: empty==empty earns full
credit only when the gold result is genuinely empty (kills the degenerate "always return nothing"
hack). The **eval** metric is the stricter multi-DB **test-suite** accuracy (the vendored evaluator),
so the eval is harder than the reward — surfacing, not hiding, single-DB false positives.
- *Why:* the composable arm is precisely what turns shaped-vs-binary from an assumption into a
  first-class ablation. The single-DB train / multi-DB eval split is the documented reward-hacking
  defense (the ~6.5% single-DB false-positive rate is what test-suite eval catches) — the project's
  eval taste applied to its own reward.
- *Alternatives:* one fixed reward (forecloses the ablation); test-suite reward at train time
  (too slow per rollout — DB execution, not generation, is the eval cost).

### D3 — GRPO hyperparameters: G=4, short completions, early-stop on dev-EX
`num_generations` 4 (2-GRPO shows G=2 keeps 97.6% of G=16 at 21% wall-clock; G=4 is low-risk and
halves generation vs G=8), `max_completion_length` 1024–1536 (SQL is short; keep the `length`/format
reward so SQL doesn't truncate), KL/`beta` per TRL defaults to start. **Early-stop / checkpoint-select
off a fixed ~500-prompt dev-EX slice evaluated every 50–100 steps**, full dev only at the end — GRPO
on small data plateaus fast, so stop paying once dev accuracy flattens. Watch `frac_reward_zero_std`
(the all-correct/all-wrong-group diagnostic that signals a dead signal at small G).
- *Why:* these are the cuts that shrink rollout generation (the dominant GRPO cost) without starving
  the RL signal; dev-EX early-stop beats a fixed step budget for spend.
- *Alternatives:* G=8 / 2048 tokens (the conservative ceiling, ~2× cost); fixed step budget
  (overpays past the plateau).

### D4 — GRPO LoRA learning rate: pilot before the 3-seed headline
The plan's two anchors **disagree**: SQL-R1's full-FT GRPO LR is `3e-7` (→ ~`3e-6` at the
"LoRA ≈ 10× full-FT" heuristic), while the locked "LoRA Without Regret" default reads `~5e-5`. RL is
LR-sensitive and a bad LR wastes the entire 3-seed budget, so run a **short single-seed LR pilot**
(e.g. `{5e-6, 1e-5, 5e-5}`, ~150 steps each on the dev-EX slice) and lock the LR off that curve
before spending on the headline. Default starting point: `1e-5` (between the anchors).
- *Why:* cheap insurance against silently training three seeds at a divergent or dead LR; the pilot
  is ~$3–5 against a ~$50 headline.
- *Alternatives:* trust either anchor blind (risks wasting the headline); a full LR sweep with seeds
  (over-spends before the gate).

### D5 — Controls are config flips on one recipe, not forks
The **cross-family** control runs the *same* GRPO function with the base model swapped to
Llama-3.2-3B-Instruct (1 seed); this requires `eval/generate.py` to be **base-model-parametric** so
the Llama+adapter eval reuses the adapter-eval path unchanged. The **reward-shaping ablation** runs
the same function with the reward arm flipped (shaped vs near-binary, 1 seed each). Eval decoding is
**matched** across base / SFT / RL (the baseline-evaluation provenance already records decoding) so
the comparison is apples-to-apples.
- *Why:* one recipe + config keeps the controls honest and cheap, and avoids a second code path that
  could drift from the headline run.
- *Alternatives:* separate training scripts per arm (drift risk, more surface).

### D6 — Multi-seed aggregation at the seed level, headlined on BIRD
3 seeds on the Qwen headline; aggregate at the **seed level** with CIs (reuse the eval-audit /
seed-level-t pattern from `grpo-gain-decomposition`) — never pool rollouts across seeds. Report the
base → SFT → RL progression on both benchmarks; **BIRD-dev is the headline** (3B Spider RL gains are
~1–2pt and may not clear the CI-excludes-zero bar; BIRD gains are larger and legible). Controls report
at 1 seed (point estimate, no CI claim).
- *Why:* 2 seeds gives no usable variance (Colas et al.); 3 is the floor for the headline. Seed-level
  aggregation is the honest unit; pooling inflates significance.

## Risks / Trade-offs

- **3B colocate OOM** → confirm the memory envelope with a short ~50-step colocate smoke *before* the
  3-seed spend (the Stage-0 plan called for this GPU smoke; verify it actually ran). Fallback ladder:
  lower `vllm_gpu_memory_utilization` → `vllm_enable_sleep_mode=True` → cut G / `max_completion_length`
  → vLLM server mode on a 2nd GPU.
- **TRL↔vLLM pairing fragility** → re-verify the pinned TRL (1.5.1) ↔ vLLM pair at image build; the
  pairing has broken before. A failed import is a build gate, not a runtime surprise.
- **Reward hacking / silent false-positives** → empty-result guard in the train reward; report the
  multi-DB test-suite EX at eval (stricter than the reward). Make a degenerate-query check loud, not
  a quiet pass.
- **Weak / unstable Spider gain at 3B** → headline on BIRD; treat a flat Spider curve as expected,
  not a failure. The GATE after this stage decides whether the gain justifies Stage-3 spend.
- **Dead RL signal at small G** → monitor `frac_reward_zero_std`; if groups are all-correct or
  all-wrong too often, raise G or temperature before blaming the reward.

## Migration Plan

Additive only — no persisted data or shipped interface changes. The reward module and GRPO function
are new; `generate.py`/`provenance.py` gain backward-compatible parameters (base-model id defaults to
Qwen; no-reward-config path unchanged). Rollback = don't invoke the GRPO entrypoint; the base + SFT
eval paths are byte-identical.

## Open Questions

- **Multi-DB test-suite eval availability.** The distilled test-suite databases were deferred in
  Stage 0 (R4). If they are not yet wired, either pull the distilled Spider test-suite DBs as part of
  this change, or report single-DB EX at eval *with the false-positive caveat disclosed* and defer
  test-suite eval. Resolve before the headline eval so the reported number's strictness is honest.
- **GRPO LoRA LR** — settled by the D4 pilot, but the pilot result should be recorded as the locked
  value in ARTIFACTS.
- **Was the Stage-0 GPU colocate smoke ever run?** If not, it is the first task here, before any paid
  training.
- **KL `beta` / clip range** — start at TRL defaults; revisit only if the pilot shows entropy
  collapse or runaway KL.
