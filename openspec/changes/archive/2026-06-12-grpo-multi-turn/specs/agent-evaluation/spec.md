# agent-evaluation

## ADDED Requirements

### Requirement: Multi-turn inference produces re-derivable transcripts
The evaluation path SHALL run the trained policy in the same environment contract as training
(greedy decoding, same turn budget and feedback formatting) over full Spider-dev and BIRD-dev,
and SHALL persist the complete per-question transcript (every turn's query, feedback, and the
final answer) in the committed CompletionSet so all agent metrics re-derive on CPU.

#### Scenario: The transcript is the artifact
- **WHEN** a multi-turn eval completes
- **THEN** the committed CompletionSet contains every turn and feedback for every question, and
  `make eval` re-derives all reported agent metrics from it without a GPU.

### Requirement: Agent metrics — self-correction rate, turns-to-solve, error-type breakdown
Scoring SHALL report, alongside final execution accuracy: the self-correction rate (turn-1-wrong
and final-right fraction), the harm rate (turn-1-right and final-wrong), the turns-to-solve
distribution, and a breakdown of corrected failures by error class (non-executing vs executing-
but-wrong-rows). Final EX SHALL use the unchanged Stage-2 metric and caveat disclosure.

#### Scenario: Self-correction is measured, not asserted
- **WHEN** a multi-turn CompletionSet is scored
- **THEN** the report states how often revision turned a wrong first attempt into a right final
  answer, how often it broke a right one, and which error classes the fixes came from.

### Requirement: The matched-compute best-of-N control
The evaluation SHALL include a best-of-N control: the same checkpoint sampling N candidates per
question without execution feedback, with N matched to the agent's measured compute (mean turns
used, fixed from pilot transcripts), selected by a gold-free executability heuristic and scored
identically. The multi-turn-vs-best-of-N comparison SHALL be reported on both benchmarks, and a
null or negative result SHALL be reported as-is.

#### Scenario: Feedback value is isolated at matched compute
- **WHEN** the agent and the best-of-N control are scored on the same benchmark
- **THEN** the report shows both numbers side by side at matched sample budget, supporting (or
  refuting) that execution feedback adds value beyond extra samples.

### Requirement: The single-turn-vs-multi-turn progression row
The report SHALL extend the Stage-2 progression table with the multi-turn row (1 seed, point
estimate, no CI claim) on both benchmarks: base → SFT → RL (single-turn) → RL (multi-turn),
alongside the best-of-N control row.

#### Scenario: The headline table tells the whole story
- **WHEN** the Stage-3 report renders
- **THEN** the progression table shows the multi-turn point estimate against the Stage-2 anchors
  with the single-seed labeling rules applied.
