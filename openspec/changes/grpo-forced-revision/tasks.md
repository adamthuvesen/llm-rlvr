# Tasks — grpo-forced-revision (Stage 3b)

## 1. Pre-registered revision probe (~$1, approved 2026-06-12)

- [x] 1.1 **GATE THRESHOLDS (pre-registered BEFORE the probe runs; this commit is the record).**
  Instrument: Stage-2 seed-0 committed completions (`results/grpo/spider-dev-zs-grpo-s0.json`),
  per-item verdicts re-derived with the committed verifier; every item's SQL executed in the
  real sandbox; one greedy revision turn from the same checkpoint (`grpo/grpo-s0`), prompted
  with its own previous query + the deterministic gold-free execution feedback, same
  final-answer format. Verdict = BOTH must hold:
  - **G1 — fix rate ≥ 10%**: of Spider-dev items the Stage-2 policy got WRONG, at least 10%
    become execution-matches after one untrained revision turn. *Rationale:* with G=4 rollouts,
    a ≥10% greedy fix rate puts sampled revision successes in enough groups to give GRPO a
    consistent positive advantage signal (the regime where Stage-2's reward worked,
    frac_zero_std ≈ 0.04); far below that, most groups are zero-variance and amplification
    stalls.
  - **G2 — harm rate ≤ 10%**: of a seeded 300-item sample of items the policy got RIGHT,
    forced reconsideration (success feedback shown, same prompt) breaks at most 10%.
    *Rationale:* harm is partially trainable-away (delta bonus rewards improvement only;
    resubmitting unchanged SQL is free), but untrained wholesale destruction (≫10%) means
    turn-1 anchoring and the harm channel dominate the learning problem — compounding exactly
    the SCoRe collapse risk D4/D8 exist to contain.
  - Reported, NOT gated: net-EX delta of one forced revision pass; fix rate split by feedback
    class (execution-error vs executed-but-wrong-rows — the latter dominates the RL'd policy's
    failures and is the hard class); unparseable/skip counts.
  - Outcome rule: PASS → stop and report (the build is a separate user decision). FAIL →
    document the no-go honestly; this change closes as a second cheap dodge.
- [x] 1.2 Probe implementation (CPU builders + one generic GPU generation pass) + committed
  probe report under `results/grpo-fr/` with full provenance; `make check` green.
- [x] 1.3 **Probe verdict (2026-06-12, report committed at `results/grpo-fr/revision-probe.json`,
  generation `raw_generate` greedy from `grpo/grpo-s0`, ~$0.30): PASS on both gates.**
  - **G1 PASS — fix rate 14.9%** (34/228 wrong answers corrected; threshold ≥ 10%). By
    feedback class: error-feedback 15/28 (54% — the easy class is highly fixable), 
    executed-but-wrong-rows 19/200 (9.5% — the hard class, modest but real signal).
  - **G2 PASS — harm rate 6.3%** (19/300 right answers broken; threshold ≤ 10%).
  - Context (not gated): untrained net-EX estimate of one forced revision pass = **−1.65pt**
    (the 6.3% harm on the 806-item right mass outweighs the 34 fixes). This is precisely the
    shape RL is for — amplify the fix behavior, suppress the harm channel (improvement-delta
    bonus + KL anchor + free identity-resubmission are already designed for it) — and exactly
    why the matched-compute best-of-N control (D6) stays the success bar.
  - **The forced-revision premise is validated where tool election failed: the policy HAS
    sampling mass on useful revision.** Per the pre-registered outcome rule: STOP here —
    section 2 (the build) is a separate user decision.

## 2. Build (NOT approved — requires user decision on a passing probe)

- [ ] 2.1 Hand-written forced-execution rollout_func + feedback-token masking; smoke with the
  archived change's assertions (mask + IS-ratio band) before any paid run.
- [ ] 2.2 Pilot with D8 diagnostics + transcript audit; then the D6 matched-compute ladder.
