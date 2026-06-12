# Proposal: grpo-multi-turn (Stage 3: multi-turn self-correction)

## Why

Stage 2 proved single-turn GRPO delivers real, seed-stable execution-accuracy gains (BIRD-dev
+6.1pt, CI excludes zero). But the model still gets one shot: it never sees an error message or a
result set. Stage 3 is the project's title claim — a **self-correcting agent** that writes SQL,
executes it, reads the error/result, and revises, with the revision behavior *learned via RL*
rather than prompted. This is the capability the project is named for, and the single-turn
foundation it builds on is now validated.

**Gate:** paid Stage-3 training starts only after the Stage-2 GATE is recorded (task 8.5 of
`grpo-single-turn`: headline met; controls in flight). All CPU scaffolding ahead of the gate is $0.

## What Changes

- A **multi-turn rollout environment**: a `SqliteText2SqlEnv` exposing a `run_sql` tool over the
  Stage-0 read-only sandbox, driven natively by TRL's `environment_factory` — the trainer runs
  write → execute → read feedback → revise (≤3 turns), masks execution-output tokens from the
  loss, and manages token concatenation.
- A **multi-turn reward** with SCoRe-style collapse guards: per-turn/hybrid credit assignment
  (per-turn execution reward blended with final-outcome advantage), a **first-turn KL anchor**
  (the model must not learn to sandbag attempt 1), and a **between-turn improvement-delta bonus**
  (reward fixing, not re-submitting).
- **Multi-turn GRPO training** warm-started from the Stage-2 adapter, single A100-80GB vLLM
  colocate, LoRA — the same envelope as Stage 2.
- A **feedback-vs-best-of-N control** at matched compute: is the multi-turn gain real use of
  execution feedback, or just more samples? (The control most multi-turn papers skip.)
- **Agent-specific evaluation**: self-correction rate, turns-to-solve, error-type breakdown, and
  the single-turn-vs-multi-turn comparison on Spider-dev + BIRD-dev.
- A **pre-gate validation** of TRL's open multi-turn importance-sampling issue (#4543) on a real
  2-turn SQL rollout before any paid scaling.

## Capabilities

### New Capabilities
- `multiturn-environment`: the SQLite text-to-SQL tool environment (reset/run_sql, feedback
  formatting, turn budget, token masking contract) that the TRL trainer drives.
- `multiturn-reward`: per-turn/hybrid credit assignment, first-turn KL anchor, improvement-delta
  bonus, and the anti-collapse diagnostics.
- `multiturn-training`: the multi-turn GRPO trainer wiring (environment_factory, warm-start from
  the Stage-2 adapter, provenance, the IS-ratio/completion-mask validation).
- `agent-evaluation`: multi-turn inference + scoring (self-correction rate, turns-to-solve,
  error-type analysis) and the matched-compute best-of-N control.

### Modified Capabilities
<!-- none: the single-turn reward, sandbox, verifier, and rl-evaluation specs are unchanged;
     multi-turn additions are new capabilities that compose them -->

## Impact

- New code: `env/` gains the tool-environment class; `train/` gains the multi-turn GRPO driver;
  `eval/` gains multi-turn generation + agent metrics; `rewards/` gains the multi-turn arm.
- Reuses unchanged: read-only sandbox, execution-match verifier, seed-level aggregation,
  provenance, Modal runner, report module.
- Dependencies: TRL ≥1.5.1 already pinned (environment_factory landed post-1.3); no new packages
  expected.
- Cost envelope: ~$30–45 paid GPU (1 seed + the best-of-N control), $0 until the gate.
- Risk: TRL issue #4543 (multi-turn IS correctness, open, both vLLM modes) — explicitly
  de-risked before paid training; if it blocks, the fallback is a hand-written `rollout_func`
  (stretch path), and Stage 2 remains shippable alone.

## Resolution (2026-06-12): CLOSED — ship Stage 2 alone; finding probe-backed

**Decision (user): ship the Stage-2 results standalone.** The design-sanctioned honest outcome:
the de-risk gate blocked paid multi-turn training on evidence, at ≈ $8 total instead of $30–45.

**The finding (committed evidence under `results/grpo-mt/`, full journal under task 2.1 in
`tasks.md`):** Qwen2.5-Coder-3B-Instruct never adopts the `run_sql` tool protocol in this
environment — 0 tool calls in 672 sampled episodes across every checkpoint rung (base, Stage-1
SFT, Stage-2 GRPO, two tool-use SFT cold starts) and every prompting strength (bare `cot`,
protocol instruction, full in-context demonstration verified rendered). A pre-registered ~$1
two-arm probe **falsified the working "RL format entrenchment" hypothesis**: the raw base model,
which carries Qwen's native function-calling training, also answers directly in 0/96 episodes.
The blocker is protocol non-adoption at this model scale/setting, not anything our training did.
Secondary findings: TRL #4543 was clear in our vLLM-colocate envelope on every census (IS ratio
0.96–1.04, always in band), and two cheap supervised cold-start recipes not only failed to
instill the protocol but measurably damaged the policy (v1: −8.1pt Spider via token-mass
imbalance re-teaching preambles; v2: degraded generation, mechanism recorded as an open
unknown).

Disposition: the CPU scaffolding (env tool loop, multi-turn reward, collapse diagnostics, smoke
driver, tool-use dataset builder, multi-turn inference eval) is built, tested, and stays in the
tree. Tasks 2.2–6.3 are **not pursued** (not done); task 2.1 ends unticked. Delta specs are
**not synced** — no multi-turn capability shipped. The project's headline remains Stage 2:
Spider-dev 0.625→0.779, BIRD-dev 0.313→0.406, with controls.
