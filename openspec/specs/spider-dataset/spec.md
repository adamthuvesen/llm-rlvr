# spider-dataset Specification

## Purpose
TBD - created by archiving change eval-foundation. Update Purpose after archive.
## Requirements
### Requirement: Load Spider examples from a pinned revision
The system SHALL load Spider NL→SQL examples from a pinned `xlangai/spider` HuggingFace parquet
revision, returning strictly-typed records (at minimum `db_id`, `question`, `gold_sql`) and the
split name. The revision MUST be recorded, never floating.

#### Scenario: Load the dev split
- **WHEN** the loader is asked for the Spider `dev` split at the pinned revision
- **THEN** it returns a sequence of typed examples, each with a non-empty `db_id`, `question`, and
  `gold_sql`, and the count matches the published Spider-dev size.

#### Scenario: Revision is pinned and recorded
- **WHEN** examples are loaded
- **THEN** the exact dataset revision used is captured in the returned metadata, so a later run can
  reproduce the same examples.

### Requirement: Provide the SQLite databases via a checksummed archive
The system SHALL obtain the Spider SQLite databases from a separately-pinned, **checksummed**
archive (the parquet dataset does not ship them), and SHALL fail loudly if the checksum does not
match or a required database is missing.

#### Scenario: Checksum mismatch fails loudly
- **WHEN** the Spider database archive's computed checksum differs from the pinned checksum
- **THEN** loading aborts with a clear error and no examples are returned.

#### Scenario: Every example resolves to a database file
- **WHEN** examples are loaded together with the database archive
- **THEN** each example's `db_id` resolves to an existing `<db_id>.sqlite` file and a `tables.json`
  schema entry, and a missing database raises a clear error rather than being skipped silently.

### Requirement: Serialize a database schema into compact prompt text
The system SHALL serialize a database's schema (tables, columns with types, primary keys, and
foreign-key edges) into compact, deterministic prompt text, exposed as a pure function of the schema.

#### Scenario: Deterministic CREATE-TABLE-style serialization
- **WHEN** the same database schema is serialized twice
- **THEN** the output is byte-identical and includes every table, its columns with types, and the
  primary-key / foreign-key relationships.

