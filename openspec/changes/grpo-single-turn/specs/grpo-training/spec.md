## ADDED Requirements

### Requirement: Train a LoRA adapter with GRPO on a Modal GPU, warm-started from the SFT adapter
The system SHALL train a LoRA adapter using TRL `GRPOTrainer` + PEFT (bf16,
`target_modules='all-linear'`, rank 32–64) on a single Modal A100-80GB, warm-started by loading the
Stage-1 SFT adapter as the initial policy, with vLLM colocate rollouts each scored by the
reward-functions module. The resulting adapter SHALL persist to the persistent Volume under a
run-scoped path, leaving the base model and the SFT adapter unmodified.

#### Scenario: GRPO writes a new LoRA adapter to the Volume
- **WHEN** GRPO training is run from the SFT warm-start
- **THEN** a new LoRA adapter (weights + config) is written to the Volume under a run-scoped path,
  and the base model and the Stage-1 SFT adapter are left unchanged.

#### Scenario: Rollouts are scored by the configured reward arm
- **WHEN** a training step generates rollouts for a group of prompts
- **THEN** each rollout's SQL is executed and scored by the selected reward arm (shaped or
  near-binary), and the group-relative advantages drive the GRPO update.

### Requirement: Base model is configurable for the cross-family control
The GRPO training and the adapter-eval generation path SHALL be parametric in the base model, so the
identical recipe trains and evaluates on Qwen2.5-Coder-3B-Instruct (default) and
Llama-3.2-3B-Instruct (the cross-family control) without a separate code path. The base-model
identity SHALL be recorded in provenance.

#### Scenario: The same recipe trains a different base family
- **WHEN** GRPO is launched with the base model set to Llama-3.2-3B-Instruct
- **THEN** training and subsequent adapter evaluation run through the same code path as the Qwen
  run, with the base-model identity recorded in provenance.

### Requirement: Early-stop and checkpoint-select on a held-out dev-EX slice
The system SHALL evaluate a fixed held-out dev-EX slice at a fixed step interval during training and
SHALL select the reported checkpoint off that dev-EX curve rather than a fixed final step, so spend
stops once dev accuracy plateaus. The dev slice SHALL be disjoint from the training prompts.

#### Scenario: Checkpoint is selected off the dev-EX curve
- **WHEN** training periodically evaluates the held-out dev-EX slice
- **THEN** the checkpoint with the best dev-EX is the one persisted and reported, not necessarily the
  final-step checkpoint.

#### Scenario: Dev slice does not leak into training
- **WHEN** the dev-EX slice is constructed
- **THEN** its prompts are disjoint from the GRPO training prompts.

### Requirement: Record full RL-training provenance
The system SHALL record, with each GRPO adapter checkpoint, the base model + revision, the warm-start
SFT adapter identity, the LoRA configuration, the reward arm + component weights, the GRPO
hyperparameters (`num_generations`, `max_completion_length`, learning rate, KL/beta, seed), the
training dataset identity, the relevant library versions (TRL, vLLM, peft, transformers), and the git
commit + dirty flag — enough to reproduce the run.

#### Scenario: Checkpoint ties to its full training configuration
- **WHEN** a GRPO adapter checkpoint is produced
- **THEN** its provenance captures base + revision, warm-start adapter, LoRA config, reward
  arm/weights, GRPO hyperparameters + seed, dataset identity, library versions, and git
  commit/dirty.

#### Scenario: Seed is recorded and applied
- **WHEN** a GRPO run is launched with a given seed
- **THEN** the seed is applied to the run and recorded in provenance, so a seed re-run is
  reproducible.
