# Proposal: grpo-forced-revision (Stage 3b: self-correction without tool election)

## Why

The archived `grpo-multi-turn` change (2026-06-12) closed on a probe-backed negative: the 3B
model never *elects* to call an offered execution tool — 0/672 sampled episodes across every
checkpoint rung and prompting strength, including the raw base model. The blocker was tool
election (an agentic judgment absent from 3B function-calling training), not self-correction
itself. This change removes election from the design entirely: the **harness executes every
answer unconditionally** (SCoRe / ReEx-SQL shape) — model writes SQL → sandbox executes →
deterministic gold-free feedback appended → model revises → final answer scored. Revision given
visible feedback is conditional generation, not a judgment call, so the policy has sampling mass
for GRPO to amplify — the thing the tool-election design never had.

## What Changes

- A **forced-execution rollout** (hand-written `rollout_func` — the archived design's named D1
  fallback): every episode runs attempt → execute → feedback → revision; feedback tokens
  loss-masked by our own masking (validated by the existing smoke assertions).
- Reuses unchanged from the archived change's shipped scaffolding: `SqliteText2SqlEnv` execution
  + `format_feedback` (gold-free), the multi-turn hybrid reward + improvement-delta bonus,
  collapse diagnostics, smoke protocol, multi-turn eval + transcripts.
- **Inherited success bar (sacred, from archived D6):** learned self-correction must beat the
  Stage-2 single-turn policy AND best-of-N at matched compute, or the result ships as a negative.
- **Inherited D8:** collapse diagnostics logged every step (turn-1 quality, edit distance,
  delta-reward distribution).

## Gate

A pre-registered, numerically-thresholded ~$1 probe (tasks 1.x) measures the *untrained*
policy's prompted-revision behavior before any build or training spend. The build is a separate
user decision even on a passing probe.

## Impact

- New code (only if the build is approved): the rollout_func + masking, a `revise` prompt
  builder. Probe-phase code: prompt builder + scoring (CPU) + one generic generation entrypoint.
- Cost: probe ~$1 (approved). Build/pilot/full run: NOT approved; priced at the archived
  change's ladder (~$1 smoke / ~$3–5 pilot / ~$25–35 full + control) if green-lit.
