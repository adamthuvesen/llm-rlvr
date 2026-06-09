## ADDED Requirements

### Requirement: Measure base-model execution accuracy
The system SHALL produce zero-shot and few-shot execution-accuracy numbers for
Qwen2.5-Coder-3B-Instruct on Spider-dev using the execution-match verifier.

#### Scenario: Zero-shot baseline on Spider-dev
- **WHEN** the baseline is run zero-shot on Spider-dev
- **THEN** it generates SQL for each example, scores it with the verifier, and reports an aggregate
  execution-accuracy number.

#### Scenario: Few-shot baseline reported separately
- **WHEN** the baseline is run few-shot
- **THEN** it reports a distinct execution-accuracy number, with the few-shot exemplar policy
  recorded.

### Requirement: Separate model-free scoring from generation
The system SHALL write model completions to a stored artifact and score them in a separate,
model-free step, so scoring runs on a CPU box without a generation backend.

#### Scenario: Re-scoring a stored completion set is deterministic
- **WHEN** a stored completion set is scored twice
- **THEN** the reported execution-accuracy number is identical both times.

### Requirement: Record full provenance for the baseline number
The system SHALL record, with every baseline result, the dataset revision and database-archive
checksum, the model and its revision, the decoding configuration, the git commit (and dirty flag),
and the relevant library versions.

#### Scenario: Baseline result is reproducible from its provenance
- **WHEN** a baseline result is produced
- **THEN** its provenance captures dataset revision + checksum, model + revision, decoding settings,
  git commit/dirty, and library versions sufficient to reproduce the run.

#### Scenario: Decoding settings are recorded for later comparison
- **WHEN** the baseline is generated with a given decoding configuration (e.g. greedy vs sampled)
- **THEN** that configuration is recorded so later RL-vs-baseline comparisons use matching decoding.
