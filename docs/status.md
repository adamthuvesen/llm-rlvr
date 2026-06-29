# Current Status

The repo's main result is the single-turn text-to-SQL GRPO training result:
three seeds, confidence intervals excluding zero on Spider and BIRD, a
near-binary reward ablation, and a Llama cross-family control.

Supported path:

- Eval foundation: data loaders, execution sandbox, verifier, and baseline runs.
- SFT cold-start: LoRA adapter that teaches the expected reasoning and SQL format.
- Single-turn GRPO: execution-grounded RL with the shipped 3-seed result.

Additional self-correction utilities remain in the tree for analysis:
multi-turn tool-loop diagnostics, forced-revision rollout/eval helpers,
`select-bestofn`, and `fr-metrics`. They are tested support code, not the
headline training result.
