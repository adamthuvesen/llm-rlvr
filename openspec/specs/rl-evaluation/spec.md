# rl-evaluation Specification

## Purpose
TBD - created by archiving change grpo-single-turn. Update Purpose after archive.
## Requirements
### Requirement: Report the RL gain multi-seed with seed-level confidence intervals
The system SHALL report the single-turn GRPO execution-accuracy gain across **at least 3 seeds**,
aggregated at the **seed level** (each seed contributes one execution-accuracy number; rollouts are
not pooled across seeds), with a confidence interval computed seed-level. The reported comparison
SHALL be the base → SFT → single-turn-RL progression on Spider-dev and BIRD-dev, with **BIRD-dev as
the headline**.

#### Scenario: Headline gain is aggregated at the seed level with a CI
- **WHEN** the 3-seed Qwen headline is reported
- **THEN** each seed contributes one EX number, and a seed-level confidence interval on the RL-vs-SFT
  gain is reported (rollouts are not pooled across seeds).

#### Scenario: Base → SFT → RL progression on both benchmarks
- **WHEN** the headline is reported
- **THEN** the base, SFT, and RL execution-accuracy numbers are shown for both Spider-dev and
  BIRD-dev, with BIRD-dev identified as the headline benchmark.

### Requirement: Evaluate with the multi-DB test-suite metric, matched decoding
The RL policy SHALL be evaluated with the stricter multi-DB **test-suite** execution-match metric
where available (stricter than the single-DB training reward), using decoding settings **matched** to
the base and SFT evaluations so the comparison is apples-to-apples. If the distilled test-suite
databases are not available, the system SHALL report single-DB execution accuracy and SHALL disclose
the single-DB false-positive caveat in the reported artifact.

#### Scenario: Decoding is matched across base, SFT, and RL
- **WHEN** the RL policy is evaluated for the comparison
- **THEN** its decoding configuration matches the base and SFT evaluations, and the match is recorded
  in provenance.

#### Scenario: Eval strictness is disclosed
- **WHEN** the test-suite databases are unavailable and single-DB EX is reported instead
- **THEN** the reported artifact discloses that the eval is single-DB and carries the single-DB
  false-positive caveat.

### Requirement: Report the cross-family replication control
The system SHALL report the cross-family control — single-turn GRPO on Llama-3.2-3B-Instruct (1 seed)
under the same recipe — as a point estimate of whether the gain replicates off the Qwen family,
presenting either outcome (replicates or does not) as an honest result without a CI claim at 1 seed.

#### Scenario: Llama control reported as a point estimate
- **WHEN** the cross-family control is reported
- **THEN** the Llama-3.2-3B RL-vs-SFT gain is shown as a single-seed point estimate, with no
  confidence-interval claim, labelled as the "is it a Qwen artifact?" check.

### Requirement: Report the shaped-vs-near-binary reward ablation
The system SHALL report the reward-shaping ablation — single-turn GRPO with the shaped arm vs the
near-binary arm (1 seed each) on the same base — so the shaped-vs-near-binary debate is adjudicated
on this setup.

#### Scenario: Both reward arms reported side by side
- **WHEN** the reward ablation is reported
- **THEN** the RL gain under the shaped arm and under the near-binary arm are shown side by side at
  matched configuration.

### Requirement: The RL-vs-SFT-vs-base table re-derives on CPU
The committed RL `CompletionSet`s SHALL be sufficient to re-derive the full base → SFT → RL table
(EX + CIs, including the controls and ablation) on a CPU box with no generation backend, via
`make eval`, alongside the existing base and SFT rows.

#### Scenario: make eval reproduces the RL table from committed artifacts
- **WHEN** `make eval` is run on a CPU box from the committed `CompletionSet`s
- **THEN** it re-derives the base → SFT → RL execution-accuracy table with CIs (and the control and
  ablation rows) without invoking a generation backend.

