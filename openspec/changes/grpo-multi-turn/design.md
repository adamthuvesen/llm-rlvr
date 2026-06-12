# Design: grpo-multi-turn (Stage 3)

## Context

Stage 2 (`grpo-single-turn`) validated the full single-turn rig: execution-grounded reward,
LoRA GRPO at LR 5e-5 on an A100-80GB with vLLM colocate, seed-level CIs, and a warm-start
continuation pattern (one adapter carries base → SFT → RL). Stage 3 upgrades the rollout from
"one shot, graded after" to a tool loop the model drives: write SQL → execute in the read-only
sandbox → read the error/result → revise → submit, ≤3 turns. The literature warns this is the
unstable frontier: SCoRe shows naive self-correction RL collapses (the model sandbags turn 1 or
makes only trivial edits); MT-GRPO shows outcome-only credit underperforms on tool use; TRL's
multi-turn importance-sampling correctness issue (#4543) is open in both vLLM modes.

Constraints carried from Stage 2: single-GPU colocate envelope, LoRA r32 all-linear, locked
LR 5e-5 unless a pilot says otherwise, strict provenance, all paid runs behind a smoke + pilot,
artifacts re-derivable on CPU.

## Goals / Non-Goals

**Goals**
- A trained multi-turn policy whose *learned* self-correction beats the Stage-2 single-turn
  policy on Spider-dev and BIRD-dev at greedy decoding.
- The matched-compute control: multi-turn-with-feedback vs best-of-N-without-feedback, so the
  claim "the model uses execution feedback" is evidenced, not asserted.
- Agent metrics: self-correction rate (turn-1-wrong → final-right), turns-to-solve, which error
  classes (syntax / schema / semantic) the loop actually fixes.
- Collapse diagnostics visible during training (turn-1 quality, edit distance between turns,
  delta-reward distribution) so failure is caught early, not post-hoc.

**Non-Goals**
- Multi-seed headline for Stage 3 (1 seed + control fits the budget; seed expansion only if the
  gate on this stage demands it).
- verl migration, multi-GPU, full fine-tuning, QLoRA.
- New datasets or schema-linking features; the prompt and data layers are frozen.
- Solving TRL #4543 upstream — we validate against it and fall back if it blocks.

## Decisions

**D1 — TRL `environment_factory` is the primary path; hand-written `rollout_func` is the
fallback/stretch.** The trainer natively runs the loop, masks tool-output tokens from the loss,
and handles re-tokenization (post-1.3 fixes; we pin 1.5.1). A hand-rolled loop re-implements all
of that for no headline value. Alternative (verl) is a scale-up option, not a correctness option.

**D2 — Environment: `SqliteText2SqlEnv` in `env/`, composing the Stage-0 sandbox.** One tool,
`run_sql(query)`. Feedback observation is deterministic and compact: error message (truncated,
stable formatting) on failure; row-count + first K rows preview on success. Turn budget 3.
The env never exposes gold SQL or gold rows — feedback is what a real analyst would see.
*Verified v1.5.1 contract (from the tagged source):* `environment_factory` is a zero-arg-callable
parameter on `GRPOTrainer.__init__`; the env is duck-typed — `reset(**dataset_row) -> str | None`
plus public methods auto-exposed as tools (schema from type hints + Google-style docstring).
The trainer owns the loop: the model ends an episode by answering instead of calling the tool,
bounded by `GRPOConfig.max_tool_calling_iterations` (set =3) *and* the env's own unit-tested
budget. Tool-output loss masking is automatic (`tool_mask`); reward functions receive the env
instances via an `environments=` kwarg — that is how per-turn state reaches the reward. Env
instances are pooled and reused across episodes, so `reset` must fully restore state. Episode
shape: N bare-SQL tool calls + one final `<think>`+SQL answer; the final answer is what the
Stage-2 reward scores, the tool calls are scored as execution-progress tiers.

**D3 — Credit assignment: per-turn/hybrid default, trajectory-level as the ablation.** Per-turn
execution reward blended with the final-outcome advantage (MT-GRPO evidence: outcome-only credit
underperforms badly on tool use; MTSQL-R1 shipped trajectory-level for SQL, so it is a workable
ablation arm, not the default). At N=3 with dense per-turn signal the gap may be small — measure,
don't assume.

**D4 — Collapse guards (SCoRe-derived), all three, on by default:**
- *Warm-start from the Stage-2 adapter* (continuation, same as Stage 2 warm-started from SFT).
- *First-turn KL anchor*: the turn-1 policy is anchored to the Stage-2 policy so the model cannot
  learn to sandbag attempt 1 to farm easy corrections.
- *Improvement-delta bonus*: reward the attempt1→attempt2 delta, not just per-turn absolutes —
  re-submitting an identical query earns nothing.
Diagnostics (D8) make a collapse signature visible regardless.

**D5 — De-risk TRL #4543 before any paid run.** The pre-gate smoke runs a real 2-turn SQL
rollout and asserts (a) `completion_mask` excludes tool-output tokens, (b) the IS ratio over
multi-turn completions stays in the healthy band we measured in Stage 2 (mean ≈1.0). If it
fails: fallback ladder = hand-written `rollout_func` → ship Stage 2 alone (both honest outcomes).

**D6 — The matched-compute control: best-of-N without feedback.** Same checkpoint, same
question set, N samples (N = mean turns used by the agent), verifier picks by execution success
*without seeing gold* (executes-without-error + non-empty heuristic, disclosed), final answer
scored normally. If multi-turn ≤ best-of-N, the "feedback" story dies at matched compute — and
we report that.

**D7 — Reward arm inside turns inherits the Stage-2 ablation verdict.** Default shaped (the
validated headline arm) with the same empty-result guard and preamble term; if the Stage-2
near-binary results overturn the choice, the config flips one field. The reward module is
already composable — no new architecture.

**D8 — Training diagnostics are logged every step, not bolted on after.** Logged per step: turn-1 EX on the dev
slice, mean turns used, between-turn edit distance, delta-reward distribution, frac of
trajectories that revise at all. These are the collapse alarms (SCoRe's failure is invisible in
aggregate reward until late).

**D9 — Evaluation reuses the Stage-2 discipline.** Greedy, full Spider-dev + BIRD-dev, same
single-DB EX + disclosed caveat, CompletionSets committed under `results/grpo-mt/`, `make eval`
re-derives. Multi-turn transcripts (all turns + feedback) are persisted in the CompletionSet so
the self-correction analysis is itself re-derivable.

## Risks / Trade-offs

- **[TRL #4543 IS correctness]** → D5 validation gate before spend; rollout_func fallback;
  Stage 2 shippable alone.
- **[SCoRe-style collapse despite guards]** → D8 diagnostics catch the signature early
  (turn-1 EX dropping while final reward rises = sandbagging; near-zero edit distance = trivial
  edits); kill and re-tune cheaply.
- **[KV/memory blowup: 3-turn transcripts ≫ single-turn]** → completion budget per turn capped;
  smoke measures the real envelope before the pilot; colocate `gpu_memory_utilization` retuned.
- **[Reward hacking via the feedback channel]** (e.g. learning to provoke informative errors
  rather than answer) → bounded turns, delta-bonus only on *improvement*, empty-result guard
  unchanged, transcript audit on the pilot.
- **[Cost overrun: rollouts are ~3× longer]** → pilot re-measures step time before the full run;
  the Stage-2 pattern (smoke → pilot → one full run) is mandatory, and num_generations stays 4.

## Migration Plan

No deployed surface; additive code. Sequencing: CPU scaffolding (env + reward + tests, $0) →
2-turn GPU smoke incl. D5 assertions (~$1) → short pilot incl. diagnostics + transcript audit
(~$3–5) → Stage-2 GATE recorded → 1 full seed + best-of-N control (~$25–35) → evals + report.
Rollback = don't merge the change; Stage 2 stands alone.

## Open Questions

- ~~**First-turn KL anchor mechanics in TRL**~~ RESOLVED (v1.5.1 source, during smoke prep):
  when the policy is a PeftModel with a pretrained adapter and `beta != 0`, the trainer
  snapshots it as a frozen `"ref"` adapter and computes the KL against it — so the stock
  `beta=0.04` already anchors the *whole completion* to the Stage-2 warm-start policy (this was
  equally true in Stage 2 vs the SFT policy). A turn-1-span-only anchor needs a custom loss;
  start with the stock whole-completion anchor and build the span-restricted variant only if
  pilot diagnostics show turn-1 sandbagging despite it.
- **Tool-use affordance** (updated after smoke attempts 1–3): the Stage-2 policy never calls
  `run_sql` under the frozen `cot` prompt (attempt 1, 0/96), under the `mt` protocol
  instruction (attempt 2, 0/96), or under `mt_demo` — the instruction plus a full in-context
  demonstration of the tool-call wire format, verified rendered in the rollout log (attempt 3,
  0/96). In-context seeding is exhausted: p(tool_call) ≈ 0 is a property of the trained policy,
  not of the prompt. Remaining rungs: a small tool-use SFT cold-start before GRPO, or
  warm-start from the SFT adapter instead (weaker direct-answer prior) as a cheap diagnostic;
  else ship Stage 2 alone.
- ~~**Group structure**~~ RESOLVED (v1.5.1 source): `num_generations` semantics are unchanged
  under environment_factory — G independent multi-turn trajectories per prompt (RepeatSampler),
  one env instance per rollout slot, advantages per group of G.
- **Best-of-N's N**: fixed N=3 (= turn budget) vs mean-turns-used (likely ~1.5–2)? Decide from
  pilot transcripts; matched compute is the principle, the pilot supplies the number.
- **Feedback verbosity**: error-only vs error+rows-preview on success — rows preview helps
  semantic errors but costs tokens. Pilot ablates informally (two prompt variants, dev slice).
