# multiturn-environment

## ADDED Requirements

### Requirement: A SQLite text-to-SQL tool environment drivable by the TRL trainer
The system SHALL provide a `SqliteText2SqlEnv` (in `env/`) exposing one tool, `run_sql(query)`,
over the Stage-0 read-only sandbox, conforming to TRL's `environment_factory` contract so
`GRPOTrainer` natively runs the write → execute → observe → revise loop. The environment SHALL
enforce a hard turn budget (default 3) and SHALL terminate the episode on submission or budget
exhaustion.

#### Scenario: The trainer drives a multi-turn episode
- **WHEN** a rollout is generated under `environment_factory` for one question
- **THEN** the model's SQL is executed in the read-only sandbox, the formatted feedback is
  appended as the next observation, and the episode ends at submission or after the turn budget.

#### Scenario: The turn budget is enforced
- **WHEN** a rollout reaches the configured turn budget without submitting
- **THEN** the episode terminates and the last query is scored as the final answer.

### Requirement: Feedback observations are deterministic, compact, and gold-free
The environment SHALL format execution feedback deterministically: on error, a truncated, stably
formatted error message; on success, the row count plus a capped preview of leading rows. The
feedback SHALL NEVER include the gold SQL, the gold result, or any verifier verdict.

#### Scenario: Error feedback is informative but bounded
- **WHEN** an executed query raises a SQLite error
- **THEN** the observation contains the error message truncated to the configured cap, formatted
  identically across runs for the same error.

#### Scenario: Feedback leaks no gold signal
- **WHEN** any feedback observation is produced
- **THEN** it contains no gold SQL, gold rows, or correct/incorrect verdict — only what executing
  the model's own query returns.

### Requirement: Execution safety and limits are inherited from the Stage-0 sandbox
All tool executions SHALL run through the read-only sandbox (read-only mode, statement timeout,
row cap) with no new execution surface introduced by the environment.

#### Scenario: A destructive query cannot mutate the database
- **WHEN** a rollout emits `DROP TABLE` or any write statement via `run_sql`
- **THEN** the sandbox rejects it, the error becomes ordinary feedback, and the database file is
  unchanged.
