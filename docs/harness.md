# The RLVR harness — adding your own task

This repo is two things: a **task-agnostic RLVR harness** and one **reference task** (text-to-SQL)
trained on it. The harness is the reusable part — SFT cold-start, GRPO with a *verifiable* reward,
and the disciplined eval/provenance layer — and it is independent of SQL. To train your own model on
your own problem, you implement a small task and reuse everything else.

> A reward is **verifiable** when a deterministic program decides correctness — execute the answer,
> compare denotations, run a test suite, check a number — rather than a learned judge. The same
> program is the training reward *and* the offline scorer, so a bug in one is a bug in the other.

## The three protocols (`harness/`)

```python
from llm_rlvr.harness import AnswerParser, Verifier, Verdict, Task, score_answer
```

1. **`AnswerParser`** — pull the answer out of a raw completion, and judge its format.
   ```python
   class AnswerParser(Protocol):
       def extract(self, text: str) -> str | None: ...   # the answer, or None if there is none
       def well_formed(self, text: str) -> bool: ...      # exactly the trained output shape?
   ```
   Be defensive in `extract`: the pre-SFT base model is messy. `well_formed` feeds the format
   reward and the SFT format-rate metric — keep it strict (one `<think>` block, then the answer).
   The *leading-`<think>` vs. has-a-preamble* distinction is the **reward's** job, not the parser's.

2. **`Verifier`** — check one answer against one reference, return a `Verdict`.
   ```python
   class Verifier(Protocol):
       def verify(self, answer: str, reference: object, context: object) -> Verdict: ...

   class Verdict:           # matched: correct?   valid: ran/parsed without error?   reason: tag
       matched: bool; valid: bool; reason: str
   ```
   `reference` and `context` are whatever your task needs (for SQL: gold query + database path; for
   math: gold number + nothing). A wrong-but-runnable answer is `valid=True, matched=False`; an
   answer that errors is `valid=False`. If the **reference itself** is broken (gold won't run),
   raise `ReferenceUnverifiable("...", reason="...")` — that's the example's fault, not the model's,
   and the reward records it without giving credit.

3. **`Task`** — bundle the above plus the data→prompt glue the shared SFT/GRPO/eval drivers call:
   `name`, `parser`, `make_verifier()`, `make_reward_fn(context_root, config=…)`,
   `build_prompts(...)`, `build_grpo_rows(...)`. See `harness/task.py` for the exact signatures and
   `tasks/sql.py` for the reference implementation.

## The reward is already written

You do **not** write reward shaping. `harness.score_answer` turns your parser + verifier into a
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
are kept because committed artifacts serialize them.) Any verifier guards — e.g. the SQL
empty-result guard that blocks the always-empty reward hack — live on the **verifier** you pass in,
not in `RewardConfig`.

## A complete minimal task

The toy in `tests/test_harness.py` is a full non-SQL task in ~25 lines: a parser that reads the
fenced answer, and a verifier that checks an integer against the gold. It exercises every reward
path (match / wrong / unparseable / no-answer / broken-reference, both arms) with no SQL involved —
copy it as your starting point.

## Wiring it in

1. Implement `AnswerParser`, `Verifier`, and a `Task` for your problem (new module under `tasks/`).
2. Register it in `tasks/__init__.py`:
   ```python
   TASKS = {"sql": sql_task, "mytask": my_task}
   ```
3. Bring your data as typed records and a prompt builder (the SQL task uses `data/models.py`
   `Example` + `data/schema.py`; yours can differ — `build_grpo_rows` just needs to emit a `prompt`
   plus whatever columns your reward reads).
4. Reuse the rest unchanged: LoRA SFT (`train/sft.py`), GRPO (`train/grpo.py` — the config builders
   and dev-split are task-agnostic), the `CompletionSet` artifact + `score`/`aggregate` eval, and
   the seed-level CI discipline.

## Plugging in your own model

The model is already a parameter — nothing to refactor. Generation, SFT, and GRPO all take a
`--model` / `--base-model`, and the project ships a Llama-3.2-3B cross-family control alongside the
Qwen default, so a third base model is a config change. Your model needs to be a Hugging Face
`AutoModelForCausalLM` that LoRA + vLLM-colocate can load. See `docs/RUNBOOK.md` for the commands.

## What stays SQL-only

`eval/verifier.py` + `env/sandbox.py` + `third_party/` (execution-match against SQLite), the Spider/
BIRD loaders in `data/`, and `tasks/sql.py` are the SQL specifics. Everything in `harness/`, the
SFT/GRPO config builders, and the `CompletionSet`/score/aggregate/provenance layer are not — that is
the line the refactor draws.
