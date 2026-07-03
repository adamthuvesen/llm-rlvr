# Current Status

The shipped result is **single-turn text-to-SQL GRPO**:

- Qwen2.5-Coder-3B-Instruct base model.
- LoRA SFT cold-start for the expected `<think>` + SQL answer format.
- GRPO with execution-grounded rewards in a read-only SQLite sandbox.
- Three-seed confidence intervals on Spider-dev and BIRD-dev.
- Near-binary reward ablation and Llama-3.2-3B cross-family control.

Supported path:

- Evaluation foundation: pinned data loaders, execution sandbox, verifier, and baseline runs.
- Training path: SFT cold-start, then single-turn GRPO.
- Reporting path: tracked public summary checked by `make eval`; external raw `CompletionSet`
  artifacts re-scored by `make eval-full` after restoring the local bundle.

Self-correction utilities remain in the tree for analysis, but they are not the public training path
or headline result.
