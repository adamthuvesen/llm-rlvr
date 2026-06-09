## ADDED Requirements

### Requirement: Execute SQL read-only
The system SHALL execute a query against a database opened read-only, such that no statement can
modify the database, the filesystem, or the network.

#### Scenario: SELECT returns rows
- **WHEN** a valid `SELECT` is executed against a database
- **THEN** the sandbox returns the result rows.

#### Scenario: A mutating statement cannot modify data
- **WHEN** a statement that would write (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ATTACH`, `PRAGMA`
  write) is executed
- **THEN** the sandbox rejects it or it fails under read-only mode, and the database on disk is
  unchanged.

### Requirement: Enforce a statement timeout
The system SHALL bound every execution by a wall-clock statement timeout and return a timeout error
rather than hanging.

#### Scenario: Long-running query is interrupted
- **WHEN** a query exceeds the configured statement timeout
- **THEN** execution is interrupted and a structured timeout error is returned within a bounded
  time, leaving the connection usable.

### Requirement: Cap returned rows
The system SHALL cap the number of rows materialized from a result set and signal when the cap was
hit.

#### Scenario: Oversized result is truncated and flagged
- **WHEN** a query would return more rows than the configured row cap
- **THEN** the sandbox returns at most the cap and marks the result as truncated.

### Requirement: Return structured errors and reject multi-statement input
The system SHALL return structured errors (error type and message) for invalid SQL and SHALL reject
input containing more than one statement, never surfacing a raw traceback or crashing the process.

#### Scenario: Invalid SQL yields a structured error
- **WHEN** syntactically or semantically invalid SQL is executed
- **THEN** the sandbox returns a structured error describing the failure, not an exception that
  propagates to the caller.

#### Scenario: Multiple statements are rejected
- **WHEN** input containing more than one SQL statement is submitted
- **THEN** the sandbox rejects it with a clear error.
