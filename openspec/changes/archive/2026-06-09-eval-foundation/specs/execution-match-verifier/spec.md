## ADDED Requirements

### Requirement: Compare result sets with test-suite execution-match semantics
The system SHALL decide whether a predicted SQL's result set matches the gold's using the official
Test-Suite SQL Eval semantics: multiset (duplicate-exact) comparison, column-order-insensitive, and
order-sensitive only when the gold query contains `ORDER BY`.

#### Scenario: Order-insensitive without gold ORDER BY
- **WHEN** the gold query has no `ORDER BY` and the prediction returns the same rows in a different
  order
- **THEN** the verifier reports a match.

#### Scenario: Order-sensitive with gold ORDER BY
- **WHEN** the gold query has `ORDER BY` and the prediction returns the correct rows in the wrong
  order
- **THEN** the verifier reports a non-match.

#### Scenario: Multiset, not set
- **WHEN** the prediction returns the same distinct rows as gold but with different duplicate counts
- **THEN** the verifier reports a non-match.

#### Scenario: Column-order-insensitive
- **WHEN** the prediction returns the same columns as gold in a different column order
- **THEN** the verifier reports a match.

### Requirement: Guard against trivial empty-result matches
The system SHALL NOT count a prediction as correct merely because both predicted and gold result
sets are empty, unless the gold result is genuinely empty for that database.

#### Scenario: Empty prediction against non-empty gold
- **WHEN** the prediction returns an empty result and the gold returns a non-empty result
- **THEN** the verifier reports a non-match.

#### Scenario: Empty prediction against genuinely empty gold
- **WHEN** the gold query genuinely returns an empty result on the database
- **THEN** an empty prediction is judged on the same basis as any other result, without special
  credit.

### Requirement: Bound the column-permutation search
The system SHALL cap the column-permutation search so wide result sets cannot cause a hang, and
SHALL log when a comparison is dropped due to the cap rather than silently treating it as a match.

#### Scenario: Wide result set does not hang
- **WHEN** comparing result sets with many columns
- **THEN** the permutation search is bounded by the configured cap and returns within a bounded time,
  logging any case it could not fully evaluate.

### Requirement: Provide single-DB and test-suite metrics with parity to the official evaluator
The system SHALL expose both single-database execution-match and (where distilled test-suite
databases are available) multi-database test-suite accuracy, and its verdicts MUST agree with the
upstream evaluator on a Spider-dev sample.

#### Scenario: Parity with the official evaluator
- **WHEN** the verifier and the upstream `taoyds/test-suite-sql-eval` are both run over a Spider-dev
  sample
- **THEN** their per-example verdicts agree, and any disagreement fails the parity test.

#### Scenario: Test-suite accuracy across distilled databases
- **WHEN** distilled test-suite databases are available for a schema
- **THEN** a prediction counts as correct only if it matches gold on every test-suite database
  instance, catching matches that are spurious on the single gold database.
