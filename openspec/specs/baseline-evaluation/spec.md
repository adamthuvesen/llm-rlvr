# baseline-evaluation Specification

## Purpose
Measure base-model and LoRA-adapter execution accuracy on full Spider-dev and BIRD-dev with the
execution-match verifier, separating model-free scoring from generation so every number re-derives
on CPU from a committed, fully-provenanced `CompletionSet`.

## Requirements
### Requirement: Measure base-model execution accuracy
The system SHALL produce zero-shot and few-shot execution-accuracy numbers for
Qwen2.5-Coder-3B-Instruct on **full Spider-dev and BIRD-dev** using the execution-match verifier —
reporting both (Spider-dev as the cleaner training/verify benchmark, BIRD-dev as the harder
real-world one). The `CompletionSet`s SHALL be committed so the numbers are re-derivable on CPU.

#### Scenario: Zero-shot baseline on full Spider-dev
- **WHEN** the baseline is run zero-shot on the full Spider-dev split
- **THEN** it generates SQL for each example, scores it with the verifier, and reports an aggregate
  execution-accuracy number.

#### Scenario: Few-shot baseline reported separately
- **WHEN** the baseline is run few-shot
- **THEN** it reports a distinct execution-accuracy number, with the few-shot exemplar policy
  recorded.

#### Scenario: BIRD-dev reported alongside Spider-dev
- **WHEN** the baseline run includes BIRD-dev
- **THEN** a BIRD-dev execution-accuracy number is reported using the same verifier and committed as
  a reference artifact alongside Spider-dev.

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

### Requirement: Evaluate a fine-tuned adapter checkpoint
The system SHALL evaluate a fine-tuned checkpoint by loading a LoRA adapter over the base model and
scoring its completions with the **same** execution-match verifier and committed-`CompletionSet`
pipeline used for the base baseline, so the cold-start is measured apples-to-apples against the base
on Spider-dev and BIRD-dev. The adapter identity SHALL be recorded in the completion provenance.

#### Scenario: Adapter checkpoint scored with the base verifier
- **WHEN** evaluation is run with an adapter checkpoint over the base
- **THEN** the adapter is loaded over the base, one completion per example is generated, and the
  result is scored by the identical verifier, with the adapter recorded in provenance.

#### Scenario: SFT-vs-base comparison is re-derivable
- **WHEN** the SFT checkpoint's `CompletionSet`s are committed
- **THEN** the SFT-vs-base execution-accuracy comparison re-derives on CPU from the committed
  artifacts, alongside the existing base numbers.

### Requirement: Report a well-formed-format rate
The system SHALL report, for a `CompletionSet`, the fraction of completions that parse to exactly one
`<think>…</think>` block and one ```sql``` fence, so the cold-start's format instillation is
measurable separately from execution accuracy.

#### Scenario: Format rate reported alongside accuracy
- **WHEN** a `CompletionSet` is scored
- **THEN** a well-formed-format rate is reported in addition to execution accuracy.

