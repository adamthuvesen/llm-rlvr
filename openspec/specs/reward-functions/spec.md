# reward-functions Specification

## Purpose
Provide the execution-grounded GRPO reward over `(completion, gold, database)` — shaped and
near-binary arms selected by config, with the empty-result guard — as a deterministic,
side-effect-free function in the TRL reward-function convention.

## Requirements
### Requirement: Execution-grounded reward over a completion
The system SHALL provide a reward function over `(prompt, completion, gold)` that parses the SQL from
the completion, executes it in the read-only Stage-0 sandbox against the example's database, and
scores it against the gold result-set with the execution-match verifier — so the reward is a
verifiable execution signal, not a model judgement. The reward function SHALL expose the single TRL
reward-function calling convention so it drops into `GRPOTrainer`.

#### Scenario: Result-matching SQL earns the result-match credit
- **WHEN** a completion's parsed SQL executes successfully and its result-set matches the gold
  result-set under the verifier
- **THEN** the reward function returns the full result-match credit for that completion.

#### Scenario: Unparseable or non-executing SQL earns no result-match credit
- **WHEN** a completion has no extractable SQL, or its SQL raises an execution error
- **THEN** the reward function returns zero result-match credit (and zero executable credit for the
  error case), without raising.

### Requirement: Composable shaped reward and a near-binary arm, selected by config
The system SHALL provide two reward arms selected by configuration: a **shaped** arm that sums
`format` + `executable` + `result_match` + `length` components, and a **near-binary** arm returning
`1.0` for an exact result-match, `0.1` for executable-but-wrong, and `0` otherwise. The component
weights and the arm selection SHALL be configuration, not hard-coded, so shaped-vs-near-binary is a
config flip.

#### Scenario: Shaped arm sums its components
- **WHEN** the shaped arm is selected
- **THEN** the returned reward is the configured weighted sum of the format, executable,
  result-match, and length components for that completion.

#### Scenario: Near-binary arm returns the Arctic-style tiers
- **WHEN** the near-binary arm is selected
- **THEN** the returned reward is `1.0` for an exact result-match, `0.1` for executable-but-not-
  matching, and `0` for non-executable or unparseable SQL.

### Requirement: Empty-result guard against degenerate matches
The training reward SHALL apply an empty-result guard: a completion whose SQL returns an empty
result-set SHALL earn full result-match credit only when the gold result-set is genuinely empty;
an empty result against a non-empty gold SHALL NOT match.

#### Scenario: Empty matches empty only when gold is empty
- **WHEN** a completion's SQL returns an empty result-set
- **THEN** it earns full result-match credit if and only if the gold result-set is also empty;
  otherwise it earns no result-match credit.

### Requirement: Deterministic, side-effect-free scoring
The reward function SHALL be deterministic for a fixed `(completion, gold, database)` and SHALL run
through the read-only sandbox only — no writes, no network, statement timeout and row cap enforced by
the Stage-0 sandbox.

#### Scenario: Re-scoring a completion is stable
- **WHEN** the same completion is scored twice against the same gold and database
- **THEN** the reward is identical both times.

#### Scenario: Reward never mutates the database
- **WHEN** any completion (including malicious or malformed SQL) is scored
- **THEN** execution goes through the read-only sandbox and the database is left unchanged.

