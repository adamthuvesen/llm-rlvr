# multiturn-training

## ADDED Requirements

### Requirement: Multi-turn GRPO via TRL environment_factory, warm-started from the Stage-2 adapter
The system SHALL train the multi-turn policy with TRL `GRPOTrainer` driving `SqliteText2SqlEnv`
through `environment_factory`, warm-started by loading the Stage-2 GRPO adapter as the trainable
initial policy (the established continuation pattern), LoRA all-linear rank 32 bf16 on a single
Modal A100-80GB with vLLM colocate. Execution-feedback tokens SHALL be masked from the loss.

#### Scenario: One adapter carries base → SFT → RL → multi-turn RL
- **WHEN** multi-turn training starts from the Stage-2 adapter and completes
- **THEN** a single adapter under a run-scoped Volume path evaluates the full progression, with
  base model, SFT adapter, and Stage-2 adapter unmodified.

#### Scenario: Tool-output tokens carry no gradient
- **WHEN** a multi-turn trajectory is tokenized for the loss
- **THEN** the execution-feedback (tool output) spans are excluded by the completion mask.

### Requirement: Importance-sampling correctness is validated before paid training
Before any paid multi-turn run, a smoke SHALL execute a real ≥2-turn rollout and assert (a) the
completion mask excludes all tool-output tokens and (b) the importance-sampling ratio stays
within the healthy band established in Stage 2. A failure SHALL block paid runs and trigger the
fallback ladder (hand-written rollout_func; else ship Stage 2 alone).

#### Scenario: TRL #4543 is checked on real rollouts, not assumed away
- **WHEN** the pre-gate smoke runs a 2-turn SQL rollout
- **THEN** the IS-ratio and mask assertions pass, or the run is declared blocked with the
  fallback ladder invoked — paid training does not proceed on a failed assertion.

### Requirement: Paid training is gated and follows the smoke → pilot → full-run ladder
Paid multi-turn training SHALL NOT start before the Stage-2 GATE decision is recorded. The spend
ladder SHALL be: CPU scaffolding ($0) → GPU smoke (~$1, incl. the IS validation and a measured
memory envelope) → short pilot (~$3–5, incl. diagnostics + transcript audit) → one full seed +
the best-of-N control. Provenance SHALL record the environment config (turn budget, feedback
format), reward config (credit blend, guards), warm-start identity, and library versions.

#### Scenario: No paid run before the gate
- **WHEN** the Stage-2 GATE is not yet recorded
- **THEN** only $0 CPU work proceeds; any GPU launch is out of scope.

#### Scenario: Provenance reconstructs the run
- **WHEN** a multi-turn adapter is produced
- **THEN** its provenance suffices to reconstruct the exact environment, reward, warm-start, and
  versions that trained it.
