# Current stage / what's done

Update this when stages move; keep the README progression table (the public 30-second
narrative) in sync with it.

**The project is complete at Stage 2; there is no active change.** The main result is the
Stage-2 single-turn GRPO result (3-seed CI excludes zero on both benchmarks, near-binary
ablation, Llama cross-family control; archived with stages 0–1). Two self-correction sequels
both closed honestly without a full training run:

- **Stage 3 (`grpo-multi-turn`, tool choice) — CLOSED 2026-06-12 without training.** The 3B
  model never adopts the `run_sql` protocol: 0 tool calls in 672 episodes across base / SFT /
  GRPO / two tool-use-SFT checkpoints and every prompting strength; a two-arm probe ruled out
  RL formatting as the cause. TRL #4543 stayed clear throughout.
- **Stage 3b (`grpo-forced-revision`) — ABORTED 2026-06-12 at the pilot gate set before the run.**
  No tool choice (runner executes every answer; hand-written TRL `rollout_func` with
  loss-masked feedback splice — its smoke validates mask + IS-ratio on real spliced rollouts).
  The probe passed (fix 14.9%, harm 6.3%), but the
  60-step pilot failed gates P1/P2: GRPO drifts toward identity-resubmission (frac_revising
  0.26→0.05, delta bonus below its untrained baseline, `frac_reward_zero_std` ≈ 0.6 — weak
  learning signal). Pilot-ckpt documentation evals (n=200): FR runner 0.720 vs Stage-2 anchor
  0.705 vs best-of-2 0.715 — small but inside noise, single-turn −2.5pt; the harm channel
  trained to 0%. Stage-3b spend ≈ $4.70. Evidence: docs/ARTIFACTS.md Stage-3b section,
  `results/grpo-fr/`, the archived change's `tasks.md`.

The self-correction support code (env tool loop, forced-revision rollout + eval runner,
multi-turn rewards + D8 diagnostics, `select-bestofn`, `fr-metrics`) stays in the tree, tested,
for any future attempt on a larger base.
