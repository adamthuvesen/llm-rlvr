# sft-training Specification

## Purpose
TBD - created by archiving change sft-coldstart. Update Purpose after archive.
## Requirements
### Requirement: Train a LoRA adapter via SFT on a Modal GPU
The system SHALL fine-tune Qwen2.5-Coder-3B-Instruct on the cold-start trace dataset using a LoRA
adapter (bf16, `target_modules='all-linear'`) on a single Modal A100-80GB, and SHALL persist the
resulting adapter to the persistent Volume. Training loss SHALL be computed only over the assistant
`<think>`+SQL completion, not over the prompt (schema + question).

#### Scenario: SFT writes a LoRA adapter to the Volume
- **WHEN** SFT is run on the trace dataset
- **THEN** a LoRA adapter (weights + config) is written to the Volume under a run-scoped path, with
  the base model left unmodified.

#### Scenario: Completion-only loss
- **WHEN** a training example is tokenized for the loss
- **THEN** only the assistant completion tokens contribute to the loss; the prompt (schema, question,
  instruction) is masked out.

### Requirement: Record adapter-checkpoint provenance
The system SHALL record, with each adapter checkpoint, the base model and its revision, the LoRA
configuration, the hash/identity of the trace dataset it was trained on, the relevant library
versions (TRL, peft, transformers), and the git commit (and dirty flag).

#### Scenario: Checkpoint ties to its inputs
- **WHEN** an adapter checkpoint is produced
- **THEN** its provenance captures base model + revision, LoRA config, trace-dataset identity,
  library versions, and git commit/dirty — enough to retrain it.

