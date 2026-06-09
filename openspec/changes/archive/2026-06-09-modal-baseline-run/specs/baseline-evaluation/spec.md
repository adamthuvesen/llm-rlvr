## MODIFIED Requirements

### Requirement: Measure base-model execution accuracy
The system SHALL produce zero-shot and few-shot execution-accuracy numbers for
Qwen2.5-Coder-3B-Instruct on **full Spider-dev and BIRD-dev** using the execution-match verifier,
with BIRD-dev reported as the headline benchmark. The headline `CompletionSet`s SHALL be committed so
the numbers are re-derivable on CPU.

#### Scenario: Zero-shot baseline on full Spider-dev
- **WHEN** the baseline is run zero-shot on the full Spider-dev split
- **THEN** it generates SQL for each example, scores it with the verifier, and reports an aggregate
  execution-accuracy number.

#### Scenario: Few-shot baseline reported separately
- **WHEN** the baseline is run few-shot
- **THEN** it reports a distinct execution-accuracy number, with the few-shot exemplar policy
  recorded.

#### Scenario: BIRD-dev reported as the headline
- **WHEN** the baseline run includes BIRD-dev
- **THEN** a BIRD-dev execution-accuracy number is reported using the same verifier and committed as
  the headline reference artifact.
