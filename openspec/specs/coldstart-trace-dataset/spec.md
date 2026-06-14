# coldstart-trace-dataset Specification

## Purpose
Build the cold-start SFT dataset by self-distillation: sample the base model on Spider-train, keep
only execution-correct, well-formed `<think>`+SQL traces (one per question), and stamp the build
with reproducible provenance.

## Requirements
### Requirement: Build a self-distilled, execution-filtered SFT dataset
The system SHALL build the cold-start SFT dataset by sampling the base model on Spider-train with the
`<think>` prompt and keeping only completions whose SQL **execution-matches the gold** under the
Stage-0 verifier and that parse to a single well-formed `<think>…</think>` + ```sql``` trace. Each
kept record SHALL pair the prompt with its selected completion.

#### Scenario: Only execution-correct traces are kept
- **WHEN** candidate completions are sampled for a Spider-train question and scored against gold with
  the verifier
- **THEN** only candidates whose SQL execution-matches the gold are eligible, and a question with no
  execution-correct candidate contributes no record (it is dropped, not defaulted).

#### Scenario: Only well-formed traces are kept
- **WHEN** an execution-correct candidate does not parse to exactly one `<think>` block and one
  ```sql``` fence
- **THEN** it is excluded from the dataset.

#### Scenario: One trace per question
- **WHEN** a question has more than one execution-correct, well-formed candidate
- **THEN** exactly one is kept according to the configured selection policy (default: the shortest),
  with exact duplicates removed.

### Requirement: Build the dataset only from the train split
The system SHALL construct the trace dataset exclusively from the train split; it MUST NOT sample
from dev or test, so the cold-start data cannot leak the evaluation benchmarks.

#### Scenario: Dev/test are never sampled
- **WHEN** the trace dataset is built
- **THEN** every source question comes from the train split, and pointing the builder at a dev/test
  pool raises loudly rather than producing records.

### Requirement: Stamp the trace dataset with provenance
The system SHALL record, with the dataset, the base model (and revision), the sampling configuration
(N, temperature, top_p, seed), the dataset revision and database-archive checksum used for filtering,
and the kept-vs-total counts, so the dataset is reproducible and auditable.

#### Scenario: Dataset carries reproducible provenance
- **WHEN** a trace dataset is produced
- **THEN** its manifest captures base model + revision, sampling config, dataset revision + DB
  checksum, and kept/total counts sufficient to reproduce and audit the build.

