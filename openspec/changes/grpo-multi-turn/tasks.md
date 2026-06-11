# Tasks — grpo-multi-turn (Stage 3: multi-turn self-correction)

## 1. CPU scaffolding ($0 — all of this proceeds before the Stage-2 gate)

- [x] 1.1 `env/SqliteText2SqlEnv` BUILT to the **verified v1.5.1 contract** (researched from the
  tagged TRL source, recorded in design D2): duck-typed `reset(**dataset_row)` + public `run_sql`
  tool method (schema from hints + docstring), zero-arg `make_environment_factory`, instances
  pool-safe (reset fully restores), env-level turn budget (default 3) on top of
  `max_tool_calling_iterations`. Trainer owns the loop + tool_mask; reward reads `env.turns` via
  the `environments=` kwarg.
- [x] 1.2 Deterministic feedback formatting (`format_feedback`): classified + truncated errors;
  row-count + capped preview on success (`include_rows` is the pilot's error-only ablation knob);
  gold-free asserted in tests (no verdict/gold tokens in observations).
- [x] 1.3 Multi-turn reward `rewards/multiturn.py`: episode = tool queries + final answer; final
  scored by the Stage-2 arm, per-turn progress on one execution-tier scale (1.0/0.1/0, guard on),
  hybrid blend (w_final/w_per_turn/w_delta) vs `credit="trajectory"` ablation switch;
  improvement-delta bonus = positive deltas only, zero on unchanged-SQL resubmission.
- [~] 1.4 First-turn KL anchor: config switch (`first_turn_kl_anchor`, ON) + provenance recording
  DONE; the loss-term mechanics over turn-1 spans deliberately deferred to the smoke (design Open
  Question — cheapest place to learn the TRL API truth).
- [x] 1.5 Collapse diagnostics `train/diagnostics.py` (`compute_diagnostics`, CPU-pure): mean
  turns, frac-revising, SQL edit distance, turn-1 progress vs final reward, delta-bonus mean.
  (Turn-1 EX on the held-out slice is the trainer-callback half — lands with 2.x wiring.)
- [x] 1.6 Multi-turn provenance `train/multiturn_provenance.py`: nests the Stage-2 GrpoProvenance
  + environment FeedbackConfig + MultiTurnRewardConfig; round-trip tested.
- [x] 1.7 Unit tests `tests/test_multiturn.py` (20, no GPU): env reset/budget/pool-reuse/
  write-rejection/determinism/gold-free, feedback variants, hybrid vs trajectory credit, zero
  delta on resubmission + degradation, per-turn empty-guard, diagnostics, provenance round-trip.
  `make check` green (140 passed, lint clean).

## 2. De-risk (GPU smoke, ~$1)

- [ ] 2.1 2-turn rollout smoke on A100 colocate: assert completion_mask excludes tool-output
  tokens and the IS ratio is in the Stage-2 healthy band (TRL #4543 check). On failure: invoke
  the fallback ladder (rollout_func; else ship Stage 2 alone) — no paid run on a failed assertion.
  - **Attempt 1 (2026-06-11, `grpo-mt-smoke`, 32 train / 8 dev, 3 steps, report committed at
    `results/grpo-mt/smoke-report-attempt1.json`): FAILED on tool use, NOT on TRL #4543.**
    IS ratio healthy (mean 0.979 in [0.8, 1.25], logp-diff 0.0068 ≈ Stage-2's 0.0076); config
    accepted; envelope comfortable (36.3 GiB torch peak / 40.8 GiB device, 11.2 s/step). But
    0/96 episodes called `run_sql` — the Stage-2 `cot` prompt gives no protocol affordance and
    the trained direct-answer format dominates the chat-template tool schema. Fix: new `mt`
    prompt style (same final-answer format + check-then-answer protocol, gold-free), recorded
    in `MultiTurnProvenance.prompt_style` — a disclosed delta to the "prompt layer frozen"
    non-goal, which froze schema-linking features, not the env affordance. Mask + 2-turn
    assertions remain unvalidated until a re-run with `mt` shows real tool calls.
  - **Attempt 2 (2026-06-11, `grpo-mt-smoke2`, same shape + `mt` prompt, report committed at
    `results/grpo-mt/smoke-report-attempt2.json`): still 0/96 tool calls.** Ruled out the
    mechanical explanations on CPU: the Coder tokenizer's chat template byte-matches TRL's
    known `qwen2_5` template (so `parse_response` was active), and the rollout log shows the
    tool schema + `mt` instruction rendered in the prompt — the policy answers
    `<think>`+```sql``` anyway at full reward. Conclusion: p(tool_call) ≈ 0 under the Stage-2
    policy, so GRPO has no exploration signal to amplify — the protocol must be seeded
    (in-context demonstration or a small tool-use SFT pass), the same reason SCoRe needs its
    own stage-1. IS ratio again in band (0.963). Spend so far: 2 smoke attempts ≈ $1 total.
- [ ] 2.2 Measure the multi-turn memory + step-time envelope (3-turn transcripts); retune
  colocate `gpu_memory_utilization` and per-turn completion caps; re-extrapolate the cost.
  (Attempt-1 numbers above are zero-tool-call transcripts — a lower bound, not the envelope.)

## 3. Pilot (paid, ~$3–5 — requires the Stage-2 GATE recorded)

- [ ] 3.1 Short pilot run with diagnostics live; audit transcripts by hand for collapse
  signatures and feedback-channel reward hacking; decide feedback verbosity (error-only vs
  +rows-preview) from the two prompt variants.
- [ ] 3.2 Fix best-of-N's N from pilot mean-turns-used; lock the multi-turn hyperparams (LR
  reuse 5e-5 unless the pilot says otherwise).

## 4. Full run + control (paid, ~$25–35)

- [ ] 4.1 One full multi-turn GRPO seed from the Stage-2 adapter; dev-EX checkpoint selection;
  adapter + provenance to the Volume.
- [ ] 4.2 Best-of-N control at matched compute (same checkpoint, no feedback, gold-free
  selection heuristic, disclosed).

## 5. Evaluation + report

- [ ] 5.1 Multi-turn greedy eval on full Spider-dev + BIRD-dev; full transcripts persisted in the
  committed CompletionSets under `results/grpo-mt/`.
- [ ] 5.2 Agent metrics in `eval/`: self-correction rate, harm rate, turns-to-solve distribution,
  corrected-error-class breakdown; all re-derivable from the committed transcripts on CPU.
- [ ] 5.3 Extend the progression table: base → SFT → RL → multi-turn RL (1-seed point estimate,
  no CI claim) + the best-of-N control row, both benchmarks; `make eval` re-derives.
- [ ] 5.4 Tests: agent metrics from fixture transcripts; table renders the multi-turn and
  control rows with single-seed labeling.

## 6. Docs + gate

- [ ] 6.1 RUNBOOK: the multi-turn procedure (smoke → pilot → full → control) + fallback ladder.
- [ ] 6.2 ARTIFACTS: the multi-turn adapter, env/reward configs, the committed comparison table.
- [ ] 6.3 Record the Stage-3 verdict: did learned self-correction beat single-turn AND beat
  best-of-N at matched compute? (Both "yes" and "no" ship — the control makes either honest.)
