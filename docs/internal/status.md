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

## Development and evidence checks

`make check` runs the locked CPU gate, including import boundaries, public evidence, and the SQLite demo.
`make package-check` verifies an installed wheel outside the checkout.
`make runtime-check` explicitly validates real training API configuration and fails when dependencies are absent.

Full raw re-scoring ships as `llm_rlvr.report.full`. It validates the complete restored inventory before querying databases.
New score reports preserve per-item diagnostics and separate generation identity from scoring identity.
Historical artifacts remain readable, with unknown hashes or lineage disclosed.

These checks improve local reproducibility and failure detection. They do not add a training result or replace the Stage 2 evidence.
