# multiturn-reward

## ADDED Requirements

### Requirement: Per-turn/hybrid credit assignment with a trajectory-level ablation arm
The system SHALL score multi-turn trajectories with a hybrid scheme: a per-turn execution reward
(the configured Stage-2 arm applied to each turn's query) blended with the final-outcome
advantage. A trajectory-level (outcome-only) arm SHALL be selectable by config as the credit
ablation. Per-turn scoring SHALL reuse the Stage-2 reward module unchanged, including the
empty-result guard.

#### Scenario: Hybrid credit rewards intermediate progress
- **WHEN** a trajectory's turn 1 fails to execute but turn 2 returns the gold-matching rows
- **THEN** the trajectory reward reflects both the final success and the per-turn progression,
  under the configured blend.

#### Scenario: The credit ablation is one config switch
- **WHEN** the trajectory-level arm is selected
- **THEN** only the final answer's execution result contributes reward, with all other settings
  matched.

### Requirement: SCoRe collapse guards — first-turn anchor and improvement-delta bonus
The reward/loss SHALL include (a) a first-turn KL anchor penalizing turn-1 divergence from the
warm-start (Stage-2) policy, and (b) an improvement bonus on the between-turn reward delta, with
zero bonus when the revised query is unchanged or not improved. Both guards SHALL be
config-toggleable for diagnosis but ON by default.

#### Scenario: Sandbagging turn 1 is penalized
- **WHEN** the turn-1 policy drifts toward systematically weaker first attempts than the
  warm-start policy produces
- **THEN** the first-turn KL anchor term grows, opposing the drift.

#### Scenario: Identical resubmission earns no delta bonus
- **WHEN** a trajectory resubmits an unchanged query on a later turn
- **THEN** the improvement-delta bonus for that turn is zero.

### Requirement: Anti-collapse training diagnostics are logged
Training SHALL log per logging-step: turn-1 EX on the held-out slice, mean turns used,
between-turn edit distance, the delta-reward distribution, and the fraction of trajectories that
revise at all — so a collapse signature (turn-1 quality falling while final reward rises, or
edit distance collapsing to zero) is visible during the run.

#### Scenario: A collapse signature is observable mid-run
- **WHEN** the policy starts gaming corrections instead of improving turn 1
- **THEN** the logged turn-1 EX and edit-distance series expose the signature without waiting for
  a post-hoc eval.
