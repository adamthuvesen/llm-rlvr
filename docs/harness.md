# The RLVR harness: adding your own task

This repo is two things: a **task-agnostic RLVR harness** and one **reference task** (text-to-SQL)
trained on it. The reusable part is the verifier/reward core, the GRPO + SFT trainers, and the
seed-level CI discipline. To train your own model on your own problem, implement a small task and
reuse the machinery. One boundary matters: the saved-artifact *scoring* path is still SQL-shaped.
See "What stays SQL-only".

> A reward is **verifiable** when a deterministic program decides correctness: execute the answer,
> compare denotations, run a test suite, or check a number. The same program is the training reward
> *and* the offline scorer, so a bug in one is a bug in the other.

## The three protocols (`harness/`)

```python
from llm_rlvr.harness import AnswerParser, Verifier, Verdict, Task, score_answer
```

1. **`AnswerParser`**: pull the answer out of a raw completion and judge its format.
   ```python
   class AnswerParser(Protocol):
       def extract(self, text: str) -> str | None: ...   # the answer, or None if there is none
       def well_formed(self, text: str) -> bool: ...      # exactly the trained output shape?
   ```
   Be defensive in `extract`: the pre-SFT base model is messy. `well_formed` feeds the format
   reward and the SFT format-rate metric. Keep it strict: one `<think>` block, then the answer.
   The **reward** handles the *leading-`<think>` vs. has-a-preamble* distinction.

2. **`Verifier`**: check one answer against one reference and return a `Verdict`.
   ```python
   class Verifier(Protocol):
       def verify(self, answer: str, reference: object, context: object) -> Verdict: ...

   class Verdict:           # matched: correct?   valid: ran/parsed without error?   reason: tag
       matched: bool; valid: bool; reason: str
   ```
   `reference` and `context` are whatever your task needs (for SQL: reference query + database path;
   for math: expected number + nothing). A wrong-but-runnable answer is
   `valid=True, matched=False`; an answer that errors is `valid=False`. If the **reference itself**
   is broken (the reference query will not run),
   raise `ReferenceUnverifiable("...", reason="...")`. That's the example's fault, not the model's,
   and the reward records it without giving credit.

3. **`Task`**: bundle the verification APIs with the data→prompt glue. `make_reward_fn` and
   `build_grpo_rows` are called by the shared GRPO driver. `build_prompts` is exposed for
   task-specific prompt construction. See `harness/task.py` for the exact signatures and
   `tasks/sql/task.py` for the reference implementation.
   `Task[ExampleT, ResourceT, RewardT]` types the example, resource, and reward configuration inputs.
   A matched `Verdict` must also be valid. Malformed completion messages raise a clear error.

## The reward is already written

Do **not** write reward shaping. `harness.score_answer` turns your parser + verifier into a
shaped or near-binary RLVR reward:

```python
from llm_rlvr.harness import RewardConfig, score_answer

breakdown = score_answer(
    completion_text, reference, context,
    parser=MY_PARSER, verifier=MY_VERIFIER,
    config=RewardConfig(arm="shaped"),   # or arm="near_binary"
    no_answer_reason="no_answer",
)
breakdown.total  # the scalar reward; also .format/.executable/.result_match/.length/.reason
```

- **shaped**: `w_format·format + w_executable·valid + w_result_match·matched + w_length·length`.
- **near_binary**: `1.0` correct / `0.1` valid-but-wrong / `0` else.

(`executable`/`result_match` on the breakdown are the generic `valid`/`matched` signals; the names
are kept because saved artifacts serialize them.) Any verifier guards, such as the SQL
empty-result guard that blocks the always-empty reward hack, live on the **verifier** you pass in,
not in `RewardConfig`.

## A complete minimal task

The minimal example in `tests/test_harness.py` is a full non-SQL task in ~25 lines: a parser that
reads the fenced answer, and a verifier that checks an integer against the expected value. It
exercises every reward path (match / wrong / unparseable / no-answer / broken-reference, both arms)
with no SQL involved. Copy it as your starting point.

## Wiring it in

1. Implement `AnswerParser`, `Verifier`, and a `Task` for your problem (new module under `tasks/`).
2. Register it in `tasks/__init__.py`:
   ```python
   TASKS = {"sql": sql_task, "mytask": my_task}
   ```
3. Bring your data as typed records and a prompt builder (the SQL task uses `data/models.py`
   `Example` + `data/schema.py`; yours can differ. `build_grpo_rows` needs to emit a `prompt`
   plus whatever columns your reward reads).
4. Reuse the shared machinery:
   - **GRPO**: `run_grpo(..., task=your_task)` trains through your task: it builds rows via
     `task.build_grpo_rows` and scores rollouts via `task.make_reward_fn`. The config builders and
     the disjoint dev split are task-agnostic.
   - **SFT**: use `train.prepared.execute_prepared_sft` with validated conversations and your task's eligibility proof.
     `train.sft.run_sft` composes SQL trace validation with that executor. It requires SQL source lineage and a matching manifest.
   - **Seed-level CIs**: `eval/aggregate.py` works on per-seed numbers, so it is reusable as-is.
   - **Offline scoring**: the `CompletionSet` artifact + `score_completion_set` are SQL-shaped
     (they embed `data.models.Example` with `gold_sql`/`db_id` and call the SQL verifier). A new
     task scores with `score_answer` directly, or grows its own artifact.

## Plugging in your own model

The model is already a parameter. Generation, SFT, and GRPO all take a
`--model` / `--base-model`, and the project ships a Llama-3.2-3B cross-family control alongside the
Qwen default, so a third base model is a config change. Your model needs to be a Hugging Face
`AutoModelForCausalLM` that the local generation and training backends can load.

## What stays SQL-only

SQL-specific: `tasks/sql/verifier.py` + `env/sandbox.py` + `third_party/` (execution-match against
SQLite), the Spider/BIRD loaders in `data/`, `tasks/sql/task.py`, **and the `CompletionSet` artifact +
`score_completion_set`** (`eval/completions.py`, `eval/score.py`). They embed `Example` and call
the SQL verifier, and saved result JSON serializes that shape, so they can't generalize without
breaking re-derivation.

Task-agnostic: everything in `harness/`, the SFT/GRPO config builders + `run_grpo` (parametrized by
`task`), the prepared training executors, and the seed-level CI aggregation (`eval/aggregate.py`).

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
    PreparedSftRow, TrainingEligibility, execute_prepared_sft, publish_training_run,
)

def train_integer_sft(base_model: str, revision: str, output: Path) -> Path:
    corpus = ((2, 4), (3, 6))
    if any(answer != 2 * number for number, answer in corpus):
        raise ValueError("integer-doubling reference verification failed")
    rows = [PreparedSftRow.model_validate({
        "prompt": [{"role": "user", "content": str(number)}],
        "completion": [{"role": "assistant", "content": str(answer)}],
    }) for number, answer in corpus]
    source_ids = tuple(hashlib.sha256(json.dumps(
        ["integer-doubling-example-v1", number, answer]
    ).encode()).hexdigest() for number, answer in corpus)
    source = output.with_suffix(".traces.jsonl")
    atomic_write_text(source, "\n".join(row.model_dump_json() for row in rows) + "\n")
    proof = output.with_suffix(".proof.json")
    atomic_write_text(proof, json.dumps({
        "policy": "integer-doubling-v1", "reference_errors": 0,
        "source_sha256": sha256_of(source), "source_ids": source_ids,
    }))
    lora_kwargs = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.0,
                   "target_modules": "all-linear", "task_type": "CAUSAL_LM"}
    execution = execute_prepared_sft(
        rows, output, base_model=base_model, base_model_revision=revision,
        requested_config={"output_dir": str(output), "completion_only_loss": True,
                          "packing": False, "max_length": 512, "report_to": []},
        lora_config=LoraConfig(**lora_kwargs), requested_lora=lora_kwargs,
        eligibility=TrainingEligibility(
            verification_policy="integer-doubling-v1",
            source_artifact_sha256=sha256_of(source), proof_sha256=sha256_of(proof),
            source_ids=source_ids,
        ),
    )
    return publish_training_run(
        output, execution, base_model=base_model,
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
Old `eval` SQL imports are compatibility exports. Add new SQL behavior in `tasks/sql/`.

Run `make check` for the CPU gate. Use `make runtime-check` on a host with the training dependencies installed.
The runtime check fails when a required dependency is absent. CPU checks do not prove GPU execution or provider compatibility.
