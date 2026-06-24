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
  - ~~**NEXT STEP (pending user decision, 2026-06-11):** pick a rung — (1) few-shot tool
    demonstration in the prompt, ~$0.50 smoke re-run (recommended first); (2) small tool-use
    SFT cold-start before GRPO, ~$5–10 + scaffolding; (3) stop and ship Stage 2 alone (the
    design-sanctioned honest outcome). 2.1 stays unticked until a smoke passes its assertions.~~
    Rung 1 picked and executed 2026-06-12 → attempt 3.
  - **Attempt 3 (2026-06-12, `grpo-mt-smoke3`, same shape + `mt_demo` prompt = `mt` instruction
    + a fixed in-context tool demonstration, report committed at
    `results/grpo-mt/smoke-report-attempt3.json`): still 0/96 tool calls — rung 1 FAILED.**
    The demonstration is structured chat messages (two `run_sql` tool calls: wrong column →
    real error feedback → revised query → success → final answer), rendered by the chat
    template itself, with feedback strings byte-matched to real sandbox execution by a unit
    test. The rollout log proves it rendered in full — `<tool_call>` wire format, both
    `<tool_response>` turns, the worked self-correction — directly above the target question,
    and the policy still answered `<think>`+```sql``` at full reward (1.60 on the logged
    sample; mean completion 65 tokens). With direct answers earning full reward and advantage
    0.00, GRPO sees no exploration signal. Conclusion: in-context seeding cannot move
    p(tool_call) off ≈0 for this policy — two RL-trained stages on the direct-answer format
    dominate a one-shot demonstration; the protocol must be trained in (tool-use SFT
    cold-start), exactly SCoRe's stage-1 rationale. IS ratio again in band (0.999, logp-diff
    0.0079); envelope 34.5 GiB torch peak / 36.5 GiB device, 14.8 s/step (longer prompts).
    Side observation: the demo slightly *hurt* direct answering (mean turn-1 progress 0.81 vs
    0.91 in attempt 2) — a longer prompt with no tool-use payoff. Spend ≈ $0.50; smoke total
    ≈ $1.50.
  - ~~**NEXT STEP (pending user decision, 2026-06-12):** the in-context rung is exhausted; two
    rungs remain — (2) small tool-use SFT cold-start before GRPO, ~$5–10 + scaffolding (the
    evidence now directly supports this: the format must be trained, not prompted); (3) stop
    and ship Stage 2 alone (the design-sanctioned honest outcome). A cheap diagnostic variant
    of (2)'s premise the design already names: re-run the smoke warm-started from the SFT
    adapter instead of `grpo-s0` (~$0.50) to test whether the GRPO stage specifically
    entrenched the direct-answer prior. 2.1 stays unticked until a smoke passes its assertions.~~
    **RESOLVED (user decision, 2026-06-12): rung 2 — tool-use SFT cold start.** Approved spend:
    ~$5–10 SFT + ~$0.50 smoke re-run; pilot/full GRPO (3.x onward) remain unapproved.
  - **Rung-2 plan + guardrail gates (recorded 2026-06-12, before any training):**
    1. *Dataset* (`train/tooluse.py`, `build-tooluse` CLI; CPU, $0): per-assistant-turn TRL
       prompt-completion rows from three transcript shapes — `revise` (corrupted query → real
       sandbox error → gold query → real success → final answer), `check` (gold query → success
       → final), `direct` (real Stage-1 trace completions re-homed under the `mt` instruction,
       protecting Stage-2 behavior). Feedback strings byte-match the rollout env (same sandbox
       caps + `format_feedback`); tool calls are structured `tool_calls` messages rendered by
       the chat template. Train-split only. Mix 600/300/600 transcripts, seed 0. Disclosed
       delta: SFT prompts lack the rendered `# Tools` system block (TRL's SFT template path
       does not take a `tools=` kwarg); rollout prompts include it — a superset context.
    2. *Mini-SFT* (`sft` entrypoint + new `--warm-start grpo/grpo-s0`): continues the Stage-2
       adapter, LR 2e-5 (between the GRPO 1e-5 default and Stage-1's 1e-4 — a nudge, not a
       rewrite), 1 epoch, otherwise Stage-1 config. Adapter → Volume `sft/tooluse-sft-r32`.
    3. *Gate (a) — forgetting:* single-turn greedy eval of the post-SFT checkpoint on full
       Spider-dev + BIRD-dev vs the Stage-2 seed-0 anchor (0.7795 / 0.4081); deltas recorded
       here before proceeding. No fixed kill threshold pre-registered — the decision rule is
       "small single-digit drop acceptable for a smoke-unblocking cold start; large drop =
       stop and reconsider."
    4. *Gate (b) — attribution baseline:* the same post-SFT checkpoint evaluated multi-turn
       (greedy, no RL) on both benchmarks, so any later Stage-3 RL gain decomposes into
       "taught the format" (SFT multi-turn − SFT single-turn) vs "learned to use feedback"
       (RL multi-turn − SFT multi-turn). Requires the new multi-turn eval path; transcripts
       committed like every CompletionSet.
    5. *Smoke re-run* (~$0.50, `mt` prompt — the demonstration is now trained in, not
       prompted): tick 2.1 only if assertions pass with real tool calls; else document
       attempt 4 and stop.
  - **Rung-2 execution (2026-06-12): mini-SFT `sft/tooluse-sft-r32` trained** (3,000 rows,
    manifest `results/grpo-mt/tooluse.manifest.json`, 188 steps, LR 2e-5, 1 epoch, ~14 min
    A100 ≈ $0.60; warm-start `grpo/grpo-s0`; final loss ≈ 0.47, token-acc ≈ 0.89).
  - **GATE RESULTS (2026-06-12, all four CompletionSets committed under `results/grpo-mt/`,
    transcripts included; re-derive with `llm-rlvr-sql score`):**
    - *Gate (a) — forgetting (single-turn greedy, `cot`, vs the Stage-2 seed-0 anchor):*
      Spider-dev **0.6983 vs 0.7795 (−8.1pt)**; BIRD-dev **0.3748 vs 0.4081 (−3.3pt)**.
      Both remain above the Stage-1 SFT level (0.678 / 0.344), but a large share of the
      Stage-2 RL gain is gone — the cold start is NOT behavior-neutral.
    - *Gate (b) — attribution baseline (multi-turn greedy, no RL):* Spider-dev **0.1151**,
      BIRD-dev **0.0469** — and **0/2,568 episodes called `run_sql`** (every episode answered
      on generation 1). The transcripts show why both numbers are so low: under the `mt`
      prompt the post-SFT policy answers in the *Stage-1 prose-preamble style*, often
      truncating at the 1,024-token cap before the ```sql``` block. These numbers measure
      prompt-context regression + truncation, not tool use.
    - *Post-hoc diagnosis (recorded for the next attempt, if any):* the `direct` mix rows —
      600 verbatim Stage-1 traces meant to *protect* Stage-2 behavior — are long
      (preamble-style), while tool-call completions are short, so by **token mass** the
      direct rows dominated the CE loss and taught "mt prompt → preamble answer" instead of
      "check first". They also reintroduced the preamble style that Stage-2's GRPO had
      specifically trained away, which is the likely mechanism of the gate-(a) drop. A
      disclosed train/rollout delta compounds this: SFT prompts lacked the rendered `# Tools`
      system block (TRL's SFT path takes no `tools=` kwarg), rollout prompts include it.
    - Caveat: gate (b) is greedy — a small nonzero p(tool_call) would be invisible; the
      temperature-1.0 smoke (96 sampled episodes) is the decisive instrument, per protocol.
  - **Attempt 4 (2026-06-12, `grpo-mt-smoke4`, same shape, `mt` prompt, warm-start
    `sft/tooluse-sft-r32`, report committed at `results/grpo-mt/smoke-report-attempt4.json`):
    still 0/96 tool calls — rung 2 as executed FAILED.** Temperature-1.0 sampling from the
    tool-use-SFT'd checkpoint confirms the greedy gate: p(tool_call) stayed ≈ 0. Two new
    signatures beyond the gate evals: (1) a fifth assertion fired — "env feedback leaked into
    gradient-carrying tokens" — because the policy now *generates* "Execution feedback …"
    phrases inside its own completions: it learned to imitate the transcript wording (one
    revise `<think>` template literally began with the marker phrase) instead of calling the
    tool; (2) the mt-context quality collapse is visible in-reward (mean turn-1 progress
    0.10 vs 0.91 in attempt 2, mean reward 0.21 vs 1.19). IS ratio in band again (0.978);
    envelope 50.5 GiB peak, 23.5 s/step. Rung-2 spend ≈ $4 (SFT $0.60 + four gate evals
    ≈ $2.50 + smoke $0.50); Stage-3 total ≈ $5.50. **The Stage-2 artifacts are untouched** —
    `grpo/grpo-s0` was only read, never written; the damaged checkpoint is the separate
    `sft/tooluse-sft-r32`.
    - *Why it failed (mechanism, from the committed evidence):* the SFT objective was
      dominated by the wrong tokens. By token mass, the 600 verbatim Stage-1 `direct` traces
      (long, preamble-style) outweighed the 900 short tool-call completions ~5:1, so the
      first-turn decision the model actually learned was "answer directly, in the old
      preamble style" — simultaneously failing to instill the tool call AND undoing Stage-2's
      learned no-preamble format (the gate-(a) −8.1pt). Compounding deltas: SFT prompts
      lacked the `# Tools` system block the rollout renders, and the think templates quoted
      the feedback marker verbatim.
    - *What a second iteration would change (NOT approved, recorded for the decision):*
      restart from `grpo/grpo-s0` (never stack on the damaged adapter); rebalance by tokens,
      not transcripts (strip preambles from direct completions or drop the direct arm to
      ~10% of token mass); pre-render prompts WITH the tools block as text rows (full byte
      parity with rollout); reword think templates to never quote the feedback marker;
      optionally upweight the bare-prompt→tool-call rows. Cost of one iteration ≈ $1.50
      (rebuild $0 + SFT $0.60 + forgetting gate $1 + smoke $0.50, skipping the multi-turn
      attribution evals until a smoke passes).
  - ~~**DECISION POINT (2026-06-12, pending user):** (i) one diagnosed-fix iteration of rung 2,
    ≈ $1.50, with the concrete changes above; or (ii) ship Stage 2 alone (unchanged, fully
    intact). Pilot/full multi-turn GRPO (3.x onward) remain unapproved either way. 2.1 stays
    unticked: four smoke attempts, none with a single tool call.~~ User picked (i), agreed as
    the **last cheap try** → attempt 5.
  - **Attempt 5 (2026-06-12, `grpo-mt-smoke5`, `mt` prompt, warm-start `sft/tooluse-sft2-r32`,
    report committed at `results/grpo-mt/smoke-report-attempt5.json`): still 0/96 tool calls —
    the diagnosed-fix iteration FAILED.** All four diagnosed fixes were implemented and
    verified: dataset v2 (`tooluse-v2.jsonl`, manifest committed; 2,600 rows, direct arm
    600→200 and preamble-stripped to leading-`<think>`, first-turn token mass ~3:1 in favor of
    tool calls — was 1:5 against), think templates no longer quote the feedback marker (the
    attempt-4 imitation assertion did NOT fire this time — that fix verifiably worked), and
    rows were pre-rendered to text through the real chat template WITH the `run_sql` tools
    block (prefix-split verified byte-clean against the actual Qwen template; a datasets
    silent-cast bug found and fixed en route — strings cast back into `List(Json)` features —
    one crashed launch ≈ $0.05). Result: the policy STILL never calls the tool, and its
    mt-context generation is *differently* degraded than v1 — short completions (mean 80.6
    tokens), at least one degenerate single-character sample in the rollout log, mean turn-1
    progress 0.11 / reward 0.24. IS ratio in band (0.966). Iteration spend ≈ $1.40 (SFT
    ≈ $0.85 incl. the crashed launch + smoke $0.50); Stage-3 total ≈ $7.
    - *Honest unknowns (recorded, not resolved — resolving needs spend beyond the agreed cap):*
      why a 163-step LR-2e-5 LoRA continuation degrades mt-context generation this much is
      undetermined. Candidate mechanisms: `max_length=2048` right-truncation clipping
      completions on long-schema rows; distribution shock of text-rendered rows vs the
      adapter's RL-tuned state; or the tool-call format simply requiring more optimization
      pressure than 900 short targets provide against ~190k steps of accumulated
      direct-answer training (base instruct → SFT → GRPO).
  - ~~**FINAL DECISION POINT (2026-06-12, pending user; the cheap-iteration well is dry):**
    five smoke attempts, zero tool calls in 480 episodes, two SFT recipes failed two different
    ways. Recommendation: **ship Stage 2 alone** (untouched, fully intact). The remaining
    alternative is a properly-scoped SCoRe-style cold start (on-policy sampled tool traces,
    eval-gated SFT, ~$10–20 + real design work) — new scope, not an iteration. 2.1 stays
    unticked either way.~~ User chose: ship Stage 2 + run a pre-registered ~$1 entrenchment
    probe first, gating any sequel → probe below.
  - **ENTRENCHMENT PROBE (2026-06-12, pre-registered gate before any sequel; two arms, same
    instrument as every attempt — temp-1.0 census, 96 episodes, `mt_demo` prompt; reports
    committed at `results/grpo-mt/smoke-report-probe-{base,sft}.json`): the entrenchment
    hypothesis is FALSIFIED.** The *raw base model* (fresh zero-init LoRA = identity; never
    touched by our training; Qwen-Instruct ships with native function-calling training) calls
    `run_sql` in **0/96** episodes; the pre-GRPO Stage-1 SFT checkpoint: **0/96**. TRL's own
    `tools/call_frequency` reads 0.0 in both arms, and the logged completions are clean direct
    answers — every checkpoint *chooses* to answer (base turn-1 progress 0.52, SFT 0.56; both
    competent direct answers under the mt context). The full curve is flat zero:
    | Checkpoint | Prompt | Tool calls |
    |---|---|---|
    | base (identity LoRA) | `mt_demo` | 0/96 |
    | SFT `sft-r32` (pre-GRPO) | `mt_demo` | 0/96 |
    | GRPO `grpo-s0` | `cot` / `mt` / `mt_demo` | 0/288 |
    | + tool-SFT v1/v2 | `mt` | 0/192 |
    - *Corrected finding:* the blocker was never RL format entrenchment — it is **protocol
      non-adoption at this scale/setting**: Qwen2.5-Coder-3B-Instruct does not elect to call
      an offered tool in this environment under any checkpoint or any strength of prompting,
      and even 900 supervised tool-call targets (two recipes) failed to flip its rollout
      behavior. Harness artifacts were ruled out as far as cheap evidence allows: the demo
      verifiably rendered, the chat template byte-matches TRL's parser schema, and TRL's own
      counter agrees with our census.
    - *Gate verdict (as pre-registered):* the deeper problem means the ~$10–20 on-policy
      cold-start sequel was very likely a write-off — **dodged for ≈ $1**. Total Stage-3
      spend ≈ $8.
  - **CLOSED per user decision (2026-06-12): ship Stage 2 alone.** 2.1 ends unticked —
    honest record: its assertions were never validated on a real tool-calling rollout, and
    after 7 censuses / 672 episodes across every checkpoint rung, zero tool calls were ever
    observed.
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
