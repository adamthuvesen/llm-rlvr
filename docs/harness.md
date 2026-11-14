# The RLVR harness: adding your own task

This repo is two things: **task-agnostic RLVR reward shaping and training executors**, and one
**task** (text-to-SQL) trained on them. The reusable parts are `harness.score_answer`, the prepared
SFT and GRPO executors, and the seed-level CI discipline. The GRPO driver (`run_grpo`), the
saved-artifact scoring path, and the task glue in `tasks/sql/` are SQL-only.

> A reward is **verifiable** when a deterministic program decides correctness: execute the answer,
> compare denotations, run a test suite, or check a number. The same program is the training reward
> *and* the offline scorer, so a bug in one is a bug in the other.

## The reward is already written

Do **not** write reward shaping. `harness.score_answer` turns three callables into a shaped or
near-binary RLVR reward:

```python
from llm_rlvr.harness import RewardConfig, Verdict, score_answer

breakdown = score_answer(
    completion_text,
    extract=my_extract,  # str -> the answer, or None if there is none
    well_formed=my_well_formed,  # str -> exactly the trained output shape?
    verify=lambda answer: my_verify(answer, reference),  # str -> Verdict
    config=RewardConfig(arm="shaped"),  # or arm="near_binary"
    no_answer_reason="no_answer",
)
breakdown.total  # the scalar reward; also .format/.executable/.result_match/.length/.reason
```

- Be defensive in `extract`: the pre-SFT base model is messy. Keep `well_formed` strict: one
  `<think>` block, then the answer. The reward handles the *leading-`<think>` vs. has-a-preamble*
  distinction.
- `verify` returns `Verdict(matched=..., valid=..., reason=...)`. A wrong-but-runnable answer is
  `valid=True, matched=False`; an answer that errors is `valid=False`. A matched `Verdict` must also
  be valid. If the **reference itself** is broken, raise
  `ReferenceUnverifiable("...", reason="...")`: the reward records it without giving credit.
- **shaped**: `w_format·format + w_executable·valid + w_result_match·matched + w_length·length`.
- **near_binary**: `1.0` correct / `0.1` valid-but-wrong / `0` else.

(`executable`/`result_match` on the breakdown are the generic `valid`/`matched` signals; the names
are kept because saved artifacts serialize them.) Any verifier guards, such as the SQL
empty-result guard that blocks the always-empty reward hack, live on the **verifier** you pass in,
not in `RewardConfig`.

`tests/test_harness.py` holds a toy integer task that exercises every reward path (match / wrong /
unparseable / no-answer / broken-reference, both arms) with no SQL involved.
`tasks/sql/task.py` is the SQL binding: `score_completion`, `make_reward_fn`, and `build_grpo_rows`.

## Training a different task

- **GRPO**: `train.prepared.execute_prepared_grpo` takes GRPO rows and a bound TRL reward function.
  `train.grpo.run_grpo` composes the SQL rows and reward with that executor.
- **SFT**: use `train.prepared.execute_prepared_sft` with validated conversations and your task's
  eligibility proof (example below). `train.sft.run_sft` composes SQL trace validation with that
  executor.
- **Seed-level CIs**: `eval/aggregate.py` works on per-seed numbers, so it is reusable as-is.
- **Offline scoring**: the `CompletionSet` artifact + `score_completion_set` are SQL-shaped (they
  embed `data.models.Example` with `gold_sql`/`db_id` and call the SQL verifier), and saved result
  JSON serializes that shape. A new task scores with `score_answer` directly, or grows its own
  artifact.

## Plugging in your own model

The model is already a parameter. Generation, SFT, and GRPO all take a
`--model` / `--base-model`, and the project ships a Llama-3.2-3B cross-family control alongside the
Qwen default, so a third base model is a config change. Your model needs to be a Hugging Face
`AutoModelForCausalLM` that the local generation and training backends can load.

## Prepare, train, and publish a different SFT task

This function prepares an integer-doubling dataset, verifies every answer, trains, and publishes a complete adapter.
Call it from an approved Modal GPU entrypoint. Pass a pinned model revision and a fresh output directory.
The task owns its source identities and verification proof. The generic executor validates conversations and records that evidence.

```python
import hashlib
import json
from pathlib import Path

from peft import LoraConfig
from llm_rlvr.artifacts.io import atomic_write_text, sha256_of
from llm_rlvr.train.prepared import (
    PreparedSftRow,
    TrainingEligibility,
    execute_prepared_sft,
    publish_training_run,
)


def train_integer_sft(base_model: str, revision: str, output: Path) -> Path:
    corpus = ((2, 4), (3, 6))
    if any(answer != 2 * number for number, answer in corpus):
        raise ValueError("integer-doubling reference verification failed")
    rows = [
        PreparedSftRow.model_validate(
            {
                "prompt": [{"role": "user", "content": str(number)}],
                "completion": [{"role": "assistant", "content": str(answer)}],
            }
        )
        for number, answer in corpus
    ]
    source_ids = tuple(
        hashlib.sha256(
            json.dumps(["integer-doubling-example-v1", number, answer]).encode()
        ).hexdigest()
        for number, answer in corpus
    )
    source = output.with_suffix(".traces.jsonl")
    atomic_write_text(source, "\n".join(row.model_dump_json() for row in rows) + "\n")
    proof = output.with_suffix(".proof.json")
    atomic_write_text(
        proof,
        json.dumps(
            {
                "policy": "integer-doubling-v1",
                "reference_errors": 0,
                "source_sha256": sha256_of(source),
                "source_ids": source_ids,
            }
        ),
    )
    lora_kwargs = {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_modules": "all-linear",
        "task_type": "CAUSAL_LM",
    }
    execution = execute_prepared_sft(
        rows,
        output,
        base_model=base_model,
        base_model_revision=revision,
        requested_config={
            "output_dir": str(output),
            "completion_only_loss": True,
            "packing": False,
            "max_length": 512,
            "report_to": [],
        },
        lora_config=LoraConfig(**lora_kwargs),
        requested_lora=lora_kwargs,
        eligibility=TrainingEligibility(
            verification_policy="integer-doubling-v1",
            source_artifact_sha256=sha256_of(source),
            proof_sha256=sha256_of(proof),
            source_ids=source_ids,
        ),
    )
    return publish_training_run(
        output,
        execution,
        base_model=base_model,
        base_model_revision=execution.base_model_revision,
        training_population_ids=source_ids,
    )
```

Both prepared executors return typed execution metadata after saving the adapter.
`publish_training_run` validates provenance, publishes periodic checkpoints, and writes `complete.json` last.
Use the returned execution metadata directly, or provide task provenance with matching `base_model` and explicit `base_model_revision`.
Custom GRPO follows the same publication path after `execute_prepared_grpo` returns.

## Changing a boundary

Run `make imports` after changing dependencies. Keep task-specific imports outside `harness/` and `train/prepared.py`.
The prepared GRPO executor accepts rows and a bound reward function. Prepared SFT accepts conversations and eligibility evidence.
SQL composition stays in the SQL-facing training entrypoints.
Add new SQL behavior in `tasks/sql/`.

Run `make check` for the CPU gate. Use `make runtime-check` on a host with the training dependencies installed.
The runtime check fails when a required dependency is absent. CPU checks do not prove GPU execution or provider compatibility.
