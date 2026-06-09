# bird-dataset Specification

## Purpose
TBD - created by archiving change modal-baseline-run. Update Purpose after archive.
## Requirements
### Requirement: Load BIRD-dev examples through the shared typed interface
The system SHALL load BIRD-dev NL→SQL examples as the same typed `Example` records used for Spider,
preserving BIRD's `evidence` (external-knowledge) field, from a pinned dataset revision.

#### Scenario: BIRD-dev loads as typed examples
- **WHEN** the loader is asked for BIRD-dev
- **THEN** it returns typed `Example`s with `db_id`, `question`, `gold_sql`, and `evidence`, in the
  same shape as Spider (Spider examples leave `evidence` unset).

### Requirement: Provide BIRD databases via a checksummed archive
The system SHALL obtain the BIRD SQLite databases from a separately-pinned, checksummed archive and
resolve each `db_id` to its `.sqlite` file, failing loudly on checksum mismatch or a missing database.

#### Scenario: Checksum mismatch fails loudly
- **WHEN** the BIRD database archive's computed checksum differs from the pinned value
- **THEN** loading aborts with a clear error and no examples are returned.

#### Scenario: Each example resolves to a database
- **WHEN** BIRD-dev examples are loaded with the database archive
- **THEN** each `db_id` resolves to an existing `.sqlite` file, and a missing one raises loudly.

### Requirement: Serialize BIRD schemas with the shared serializer
The system SHALL serialize BIRD database schemas with the existing schema serializer, flagging
schemas that exceed the configured prompt-token budget rather than silently truncating.

#### Scenario: BIRD schema serializes deterministically
- **WHEN** a BIRD schema is serialized twice
- **THEN** the output is byte-identical, and a schema exceeding the token budget is flagged.

### Requirement: Include evidence in the prompt when present
The system SHALL include an example's `evidence` in the constructed prompt when present, and SHALL
leave Spider prompts (no evidence) unchanged.

#### Scenario: BIRD evidence reaches the prompt
- **WHEN** a prompt is built for a BIRD example that carries `evidence`
- **THEN** the evidence text appears in the prompt.

#### Scenario: Spider prompts are unchanged
- **WHEN** a prompt is built for a Spider example with no evidence
- **THEN** the prompt is identical to the pre-BIRD behavior.

