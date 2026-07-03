# llm-rlvr

A compact **RLVR harness** with a text-to-SQL reference task.

The reusable part is task-agnostic: implement an `AnswerParser` and a `Verifier`, then reuse the
same reward, SFT/GRPO training path, and seed-level evaluation discipline. The reference task proves
that path on Spider and BIRD by training `Qwen/Qwen2.5-Coder-3B-Instruct` with a LoRA SFT cold-start
followed by GRPO against an execution-verified SQL reward.

## Result

Single-turn GRPO improves over the SFT checkpoint on both benchmarks. The raw completion and score
artifacts are generated outside git; `results/public/` keeps the compact tracked summary that
validates this table.

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
5. Train LoRA adapters with SFT, then GRPO.
6. Re-score saved completions on CPU without loading a model.

The model never sees hidden labels during reward computation. Datasets are pinned and fetched; database
files are not committed. The module map and design notes are in
[docs/architecture.md](docs/architecture.md).

## Reproduce

Set up the CPU environment and run the normal gate:

```bash
make install
make check
```

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

For a full raw re-score, restore the external result bundle under gitignored `results/`, fetch the
pinned databases, then run:

```bash
make data
make eval-full
```

`make eval-full` is CPU-only and model-free once local completion artifacts are present. It is the
full-artifact check, not a clean-clone public check.

GPU generation and training run through [modal_app.py](modal_app.py). Those jobs are intentionally
outside the CPU quickstart.

## Caveats

- Accuracy is single-DB execution match, not test-suite multi-DB accuracy.
- All main evaluations are greedy.
- Self-correction utilities remain tested analysis code, but the shipped training result is
  single-turn GRPO.

## Docs

- [docs/harness.md](docs/harness.md): the task-agnostic RLVR harness and how to add a task.
- [docs/architecture.md](docs/architecture.md): module map, data flow, reward design.

## Related Repositories

These repositories are separate codebases connected by model artifacts and
measurement questions:

- [`esme-pretrain`](https://github.com/adamthuvesen/esme-pretrain): trains
  `Esme-214M-Base` from scratch.
- [`esme-posttrain`](https://github.com/adamthuvesen/esme-posttrain): adapts
  the base checkpoint with SFT, DPO, and verifier-backed RLVR.
- [`llm-infer`](https://github.com/adamthuvesen/llm-infer): loads, serves, and
  benchmarks exported Esme checkpoints.
- [`llm-rlvr`](https://github.com/adamthuvesen/llm-rlvr): provides a reusable
  RLVR harness with text-to-SQL as the reference task.
- [`grpo-decomp`](https://github.com/adamthuvesen/grpo-decomp): measures where
  GRPO gains come from, separating reliability from new capability.

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
