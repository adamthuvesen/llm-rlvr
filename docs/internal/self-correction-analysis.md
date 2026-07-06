# Self-Correction Analysis Notes

These notes preserve operational lessons from the self-correction work without making them part of
the public reader path. The shipped result remains single-turn GRPO.

## Tool-Loop Analysis

Multi-turn tool-use helpers remain in the tree for analysis: environment tool loop, multi-turn
reward and diagnostics, tool-use dataset builder, and multi-turn inference evaluation. The retained
artifacts live under `results/grpo-mt/`.

This 3B setup did not adopt the `run_sql` protocol under the tested prompting and checkpoint
variants. Future work should treat tool-use adoption as a separate capability target, preferably
with a larger base model or a stronger supervised tool-use stage.

Retained evidence includes:

- `results/grpo-mt/*tooluse-sft-r32*.json`
- `results/grpo-mt/tooluse*.manifest.json`

## Forced-Revision Analysis

Forced-revision helpers remain in the tree for analysis: revision rollout, feedback-token masking,
forced-revision evaluation, `select-bestofn`, and `fr-metrics`. The retained artifacts live under
`results/grpo-fr/`.

Error-triggered second generations can be useful as an inference policy. The retained training
evidence is analysis material, not another training claim.

Retained evidence includes:

- `results/grpo-fr/*grpo-fr-pilot*.json`
- `results/grpo-fr/*error-only*.json`
- Forced-revision transcript JSONs needed by the external full-artifact re-score
- Volume path `grpo-fr/grpo-fr-pilot/`
