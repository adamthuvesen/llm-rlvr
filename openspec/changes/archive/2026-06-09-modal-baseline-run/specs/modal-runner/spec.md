## ADDED Requirements

### Requirement: Run generation on a Modal A100 GPU
The system SHALL execute the existing generation pipeline on a single Modal A100-80GB GPU via the
vLLM backend, writing a `CompletionSet` to a persistent Modal Volume.

#### Scenario: GPU generation writes a completion set
- **WHEN** a baseline generation is launched on the Modal GPU function for a model, split, and decoding config
- **THEN** it generates one completion per example on the GPU and writes a `CompletionSet` to the Volume.

#### Scenario: Model-free scoring is preserved
- **WHEN** the `CompletionSet` is pulled from the Volume and scored on a CPU box
- **THEN** the execution-accuracy number is produced with no generation backend, identical on repeat.

### Requirement: Trace a GPU run to its source code
The system SHALL record, in each GPU run's provenance, the local git commit and dirty flag, computed
locally and passed into the function (the image excludes `.git`).

#### Scenario: Provenance captures the local commit
- **WHEN** a GPU run is launched from a working tree at a given commit
- **THEN** the resulting `CompletionSet`'s provenance records that commit and dirty flag.

### Requirement: Long runs survive client disconnect
The system SHALL support launching a long GPU run detached so it completes server-side independent of
the local client.

#### Scenario: Detached run completes after disconnect
- **WHEN** a full-dev run is launched detached/spawned and the local client disconnects
- **THEN** the run continues server-side and its results remain on the Volume.
