# Tasks — grpo-forced-revision (Stage 3b)

## 1. Pre-registered revision probe (~$1, approved 2026-06-12)

- [ ] 1.1 **GATE THRESHOLDS (pre-registered BEFORE the probe runs; this commit is the record).**
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
- [ ] 1.2 Probe implementation (CPU builders + one generic GPU generation pass) + committed
  probe report under `results/grpo-fr/` with full provenance; `make check` green.
- [ ] 1.3 Probe verdict recorded here against G1/G2, with numbers.

## 2. Build (NOT approved — requires user decision on a passing probe)

- [ ] 2.1 Hand-written forced-execution rollout_func + feedback-token masking; smoke with the
  archived change's assertions (mask + IS-ratio band) before any paid run.
- [ ] 2.2 Pilot with D8 diagnostics + transcript audit; then the D6 matched-compute ladder.
