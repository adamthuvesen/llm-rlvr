## ADDED Requirements

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
