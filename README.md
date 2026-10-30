# llm-rlvr

![License](https://img.shields.io/github/license/adamthuvesen/llm-rlvr) ![Python](https://img.shields.io/badge/python-3.14%2B-blue)

An RLVR harness for text-to-SQL. It trains `Qwen/Qwen2.5-Coder-3B-Instruct`
to write SQL by running the query and rewarding it when the result matches the reference answer,
so the reward comes from running the query rather than matching strings.

Training is run by [TRL](https://github.com/huggingface/trl). Tasks are made verifiable here with a
read-only SQLite sandbox, an execution-match verifier, and reward shaping. Text-to-SQL is included
as the reference task. New tasks are added by implementing an `AnswerParser` and a `Verifier`, then
reusing the same reward and training path.

## Result

Single-turn GRPO improves over the SFT checkpoint on both benchmarks. The tracked public summary
validates this table from clean-checkout artifacts.

| Benchmark | Base | SFT | GRPO | Gain vs SFT | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| Spider-dev | 0.625 | 0.678 | **0.779** | **+10.2pt** | 3 seeds, 95% CI excludes zero |
| BIRD-dev | 0.313 | 0.344 | **0.406** | **+6.1pt** | 3 seeds, 95% CI excludes zero |

The main run uses a shaped execution reward with `guard_trivial_empty=True`. Controls include a
near-binary reward ablation and a Llama-3.2-3B cross-family run.

## How It Works

1. Load pinned Spider/BIRD examples and schemas.
2. Ask the model for SQL.
3. Execute predicted and reference SQL in a read-only SQLite sandbox.
4. Score denotation match with the same verifier used as the training reward.
5. Hand that reward to TRL to train LoRA adapters, SFT first, then GRPO.
6. Re-score saved completions on CPU without loading a model.

The model never sees hidden labels during reward computation. Datasets are pinned and fetched;
database files are not committed. The module map and design notes are in
[docs/architecture.md](docs/architecture.md).

## Reproduce

Set up the CPU environment and run the normal gate:

```bash
make install
make check
```

Every other command is listed in [AGENTS.md](AGENTS.md#commands) and `make help`.

Run a tiny no-GPU demo:

```bash
make demo
```

Expected output:

```text
execution_accuracy=0.5000 well_formed_rate=0.0000 (1/2; gold_errors=0)
reasons={'match': 1, 'denotation_mismatch': 1}
```

Validate the tracked public evidence bundle:

```bash
make eval
```

`make eval` is CPU-only and model-free; it validates `results/public/claims.json`, checks that
README headline rows are derived from `results/public/headline_summary.json`, and reads the
reproduction policy in `results/public/provenance.json`.

The public reproduction path covers the CPU checks and tracked evidence bundle. Full raw re-scoring
is run with `make eval-full` after the external artifacts are restored. Training reruns require the
GPU extras (`trl`, `peft`, `accelerate`, `vllm`) and are outside the public quickstart.

Full reproduction ships in `llm_rlvr.report.full`. It checks every required artifact before scoring, then reports score and reference-error deltas.
Historical reports stay unchanged. Legacy artifacts without stored hashes or per-item outcomes retain those limits.
See [the runbook](docs/internal/runbook.md#reproduce-reported-numbers) for explicit database roots.

## Caveats

- Accuracy is single-DB execution match, not test-suite multi-DB accuracy.
- All main evaluations are greedy.

## Docs

- [docs/harness.md](docs/harness.md): the task-agnostic RLVR harness and how to add a task.
- [docs/architecture.md](docs/architecture.md): module owners, dependency boundaries, and artifact states.
- [docs/internal/runbook.md](docs/internal/runbook.md): commands, input requirements, and failure recovery.
- [docs/internal/artifacts.md](docs/internal/artifacts.md): public claims and external artifact inventory.
- [docs/internal/verification.md](docs/internal/verification.md): verification scope and change-quality rubric.

## Related Repositories

These repositories are separate codebases connected by model artifacts and measurement questions:

- [`esme-pretrain`](https://github.com/adamthuvesen/esme-pretrain): trains
  `Esme-214M-Base` from scratch.
- [`esme-posttrain`](https://github.com/adamthuvesen/esme-posttrain): adapts
  the base checkpoint with SFT, DPO, and verifier-backed RLVR.
- [`llm-infer`](https://github.com/adamthuvesen/llm-infer): loads, serves, and
  benchmarks exported Esme checkpoints.
- [`llm-rlvr`](https://github.com/adamthuvesen/llm-rlvr): this repo, an
  RLVR harness with text-to-SQL as the reference task.
- [`grpo-decomp`](https://github.com/adamthuvesen/grpo-decomp): measures where
  GRPO gains come from, separating reliability from new capability.

The [full-lifecycle retrospective](https://github.com/adamthuvesen/esme-pretrain/blob/main/docs/retrospective.md)
walks the whole Esme chain, with every claim linked to a committed artifact.

## Related Work

This is the build-and-train companion to
[grpo-decomp](https://github.com/adamthuvesen/grpo-decomp), which studies how much measured RL
reasoning gain is real improvement versus evaluation artifact.

## Data and License

Spider is CC-BY-SA-4.0. BIRD-dev is CC-BY-NC-4.0. The vendored evaluator is Apache 2.0. Datasets and
databases are pinned and fetched, not committed.

This repository's own code is MIT licensed.

## References

- Shao et al., [_DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models_](https://arxiv.org/abs/2402.03300), 2024.
- Wen et al., [_Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs_](https://arxiv.org/abs/2506.14245), 2025.
- Hu et al., [_LoRA: Low-Rank Adaptation of Large Language Models_](https://arxiv.org/abs/2106.09685), 2021.
- Hui et al., [_Qwen2.5-Coder Technical Report_](https://arxiv.org/abs/2409.12186), 2024.
- Yu et al., [_Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task_](https://arxiv.org/abs/1809.08887), 2018.
- Li et al., [_Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs_](https://arxiv.org/abs/2305.03111), 2023.
- Zhong et al., [_Semantic Evaluation for Text-to-SQL with Distilled Test Suites_](https://aclanthology.org/2020.emnlp-main.29/), 2020.
