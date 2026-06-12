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

## 2. Build (APPROVED 2026-06-12 — user green-light on the passing probe; spend cap $45 total)

- [x] 2.1 Hand-written forced-execution rollout_func + feedback-token masking; smoke with the
  archived change's assertions (mask + IS-ratio band) before any paid run.
  - **SMOKE PASSED (2026-06-12, `grpo-fr-smoke2`, 32 train / 8 dev, 3 steps, 96 episodes;
    report committed at `results/grpo-fr/smoke-report.json`; ≈ $0.85 incl. one crashed launch
    — transformers-v5 `apply_chat_template` returns a dict by default, fixed with
    `return_dict=False`).** The assertions the archived change never got to validate now pass
    on real spliced rollouts: every one of 96 episodes carries a masked feedback segment
    (10,291 masked tokens; feedback present in masked spans, NEVER in gradient-carrying
    spans); IS ratio mean 0.982 in [0.8, 1.25], logp-diff 0.0104 ≈ Stage-2's 0.0076 (TRL
    #4543 clear on spliced sequences); 0 NaN losses, 0 NaN sampling logprobs; 0 unparseable
    attempts. Untrained D8 baselines for the pilot gates: turn-1 progress **0.924**,
    frac_revising **0.219**, edit distance 0.247, delta bonus **0.0094**, mean reward 1.43.
    Envelope: 42.4 GiB torch peak / 45.5 GiB device, **15.0 s/step**, completions ≈ 198
    tokens — full-run projection ≈ 812 steps ≈ 3.4 h ≈ **$9–12**, inside the P6 ceiling.
    P5 anchor (CPU, committed set): grpo-s0 single-turn EX on the first 200 Spider-dev items
    = **0.7050** → pilot FR eval must score ≥ 0.6550.
  - **Build (2026-06-12, committed):** TRL v1.5.1 `rollout_func` path (the archived D1
    fallback), verified against the tagged source: the rollout returns
    `prompt_ids`/`completion_ids`/`logprobs` plus an `env_mask` extra field the trainer
    consumes as `tool_mask` — the loss mask becomes `completion_mask * tool_mask` and the
    IS-ratio metrics are computed under the same combined mask, so spliced feedback tokens
    carry no gradient and never pollute the #4543 band check. Episode: greedy-free sampled
    attempt under the frozen `cot` prompt (no tool schema — the layer that killed Stage 3a is
    gone) → SQL extracted + executed unconditionally in the sandbox (agent caps, byte-same
    feedback as the probe; fixed gold-free message for unparseable attempts) → feedback +
    revise instruction spliced as a user turn via chat-template differencing (TRL's own
    `_get_tool_suffix_ids` technique, same disclosed one-newline artifact) → revision sampled
    → reward = the shipped multi-turn hybrid (`score_trajectory`: attempt as the executed
    turn, revision as the final answer; improvement-delta bonus live, identity resubmission
    free). D8 diagnostics logged every 128 episodes. 19 CPU tests; `make check` green.
- [ ] 2.2 Pilot with D8 diagnostics + transcript audit; then the D6 matched-compute ladder.
  - **PILOT GATES (pre-registered BEFORE the pilot runs; this commit is the record).**
    Pilot shape: `fr_train`, 2,000-question Spider-train subset, `max_steps 60`, LR 5e-5
    (Stage-2 locked), seed 0, G=4 — ~$2–3. Instruments: the windowed D8 series (window=128
    episodes), trainer logs, a hand audit of ≥20 pilot transcripts, and a 200-item Spider-dev
    forced-revision eval of the pilot checkpoint (~$0.5). CONTINUE to the full run only if
    ALL hold; otherwise ABORT (stop, commit evidence, report — no re-tuning past the cap):
    - **P1 — no turn-1 sandbagging (SCoRe collapse #1):** mean_turn1_progress over the last
      D8 window ≥ 0.85 × the first window's value.
    - **P2 — the fix channel is alive:** mean delta_bonus > 0.01 averaged over the second
      half of D8 windows (the policy earns real improvement bonuses for GRPO to amplify).
    - **P3 — training health:** every logged IS-ratio step mean in [0.8, 1.25]; zero NaN
      loss/grad_norm; zero NaN sampling logprobs.
    - **P4 — no trivial-edit collapse (SCoRe collapse #2):** NOT (frac_revising > 0.8 AND
      mean_edit_distance < 0.05) in the last D8 window.
    - **P5 — eval-path sanity:** pilot-checkpoint forced-revision EX on the first 200
      Spider-dev items ≥ (Stage-2 grpo-s0 single-turn EX on the same 200 items, re-scored
      from the committed CompletionSet) − 5pt. Catastrophic regression = structural bug or
      distribution shock; diagnose, don't pay for the full run.
    - **P6 — cost projection:** measured pilot step time × full-run step count must project
      the full training run ≤ $26. If over: train on a 4,000-question subset (~500 steps)
      and re-project; if still over, escalate to the user before spending.
    - Transcript audit (qualitative, can override to ABORT): revisions must be genuine edits
      responding to feedback — not feedback-marker imitation inside completions (the leak
      assertion also fires on this), not systematic answer-destruction on success feedback.
  - **PILOT VERDICT (2026-06-12, `grpo-fr-pilot`: 2,000-question subset, 60 steps @ LR 5e-5,
    1,920 episodes, ≈ $1.50; D8 series committed at `results/grpo-fr/pilot-d8-diagnostics.json`):
    GATE FAILED — P1 and P2 do not hold. Per the pre-registered rule: ABORT the ladder; no
    full run; the change closes as an honest negative at the pilot rung.**
    - **P1 FAIL (marginal):** last-window turn-1 progress 0.737 < 0.754 (= 0.85 × first
      window 0.887). Not the classic sandbag signature (final reward fell with it, w14
      reward 1.180), but the gate is the gate.
    - **P2 FAIL (clear):** mean delta-bonus over the second half of windows 0.0076–0.0085
      (by either half-split reading) < 0.01 — *below even the untrained smoke baseline
      (0.0094)*. The fix channel shrank under training instead of amplifying.
    - P3 PASS: every logged IS-ratio step mean in band (1.018 / 1.002 / 1.002 …); 0 NaNs.
      P4 PASS: no trivial-edit signature (frac_revising 0.109 last window). P6 would have
      passed: 15.5 s/step → full-run projection ≈ $10.
    - **Mechanism (the documented negative):** `frac_revising` fell monotonically-ish from
      0.26 (w0) to ~0.05 (w4–w14) — GRPO converges toward *identity resubmission*, the
      degenerate harm-avoidance policy the probe's economics predict: with ~78% of attempts
      already right and an untrained net-EX of −1.65pt per revision pass, the cheapest
      reward gain is "stop touching answers", not "fix the wrong ones". The starvation is
      visible in `frac_reward_zero_std` ≈ 0.56–0.71 (vs Stage-2 shaped ≈ 0.04): most G=4
      groups score identically (right attempt, kept, same reward) and contribute zero
      advantage, so the residual gradient is dominated by the harm channel. The
      improvement-delta bonus (w_delta 0.3 on a ≤1.0-tier delta) is too small a carrot
      against w_final ≈ 1.6 of "keep it safe". This is SCoRe's collapse mode in a new
      costume — reached with the guards ON (delta bonus live, KL anchor live, identity
      resubmission earning zero delta).
    - Pilot-checkpoint documentation evals (n=200 Spider-dev slice, pilot rung budget):
      recorded below for the abort report; they do not gate anything.
  - **D6 CONTROLS (pre-registered):** the full-run verdict compares, on full Spider-dev +
    BIRD-dev: (a) the FR policy under the forced-revision harness (greedy, 2 generations per
    question); (b) the Stage-2 seed-0 single-turn greedy anchors (0.7795 / 0.4081, committed);
    (c) **best-of-2 from the same FR checkpoint** — 2 sampled single-turn passes (T=1.0,
    seeds 0/1, `cot` prompt), gold-free selection (executes-without-error + non-empty
    heuristic, disclosed selection report committed) — the archived D6's "same checkpoint, no
    feedback" arm at matched compute. SUCCESS = (a) > (b) AND (a) > (c) on both benchmarks;
    anything else ships as an honest negative. Checkpoint under evaluation: the final one,
    unless the full-run D8 series fails P1/P4 late — then the latest `save_steps` checkpoint
    whose window passes, disclosed. Single-seed point estimates; no CI claims.
