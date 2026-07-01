"""Tool-use SFT dataset builder for the `run_sql` protocol.

Each record is a TRL conversational prompt-completion row, one row per assistant turn of a
demonstration transcript, in three shapes:

  - `revise`  (3 rows): corrupted query → error feedback → reference query → success feedback →
    final `<think>`+```sql``` answer.
  - `check`   (2 rows): reference query → success feedback → final answer.
  - `direct`  (1 row): no tool call — a trace completion under the `mt` system instruction, so
    direct-answer quality stays represented.

Feedback strings are produced by executing the query in the real sandbox with the environment's
own caps and `format_feedback` — the training targets byte-match what the rollout env will say.
Tool-call turns are structured `tool_calls` messages, so the chat template (not us) renders the
wire format the trainer's parser expects. Everything is train-split only (R2 no-leakage).
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from llm_rlvr._base import Frozen
from llm_rlvr.data.jsonl import write_jsonl
from llm_rlvr.data.models import DatabaseSchema, Example
from llm_rlvr.data.paths import resolve_database, sha256_of
from llm_rlvr.data.splits import assert_train_split
from llm_rlvr.env.sandbox import ExecutionError, SqliteSandbox
from llm_rlvr.env.text2sql_env import (
    AGENT_ROW_CAP,
    AGENT_TIMEOUT_S,
    FeedbackConfig,
    format_feedback,
)
from llm_rlvr.eval.messages import tool_call_message, user_content
from llm_rlvr.eval.prompt import MULTITURN_INSTRUCTION
from llm_rlvr.eval.provenance import git_commit, git_is_dirty
from llm_rlvr.train.traces import SftTrace

Message = dict[str, object]


class ToolUseMix(Frozen):
    """How many transcripts of each shape, and the seed that picks questions + templates.

    The defaults balance the first-turn decision by token mass, not transcript count. Direct rows
    are preamble-stripped so they do not dominate the shorter tool-call rows.
    """

    n_revise: int = 600
    n_check: int = 300
    n_direct: int = 200
    seed: int = 0


class ToolUseRow(Frozen):
    """One TRL prompt-completion row (one assistant turn of a demonstration transcript)."""

    db_id: str
    question: str
    shape: str  # revise | check | direct
    prompt: tuple[Message, ...]
    completion: tuple[Message, ...]


@dataclass(frozen=True)
class _ToolExampleRows:
    rows: tuple[ToolUseRow, ...] = ()
    shape: str | None = None
    gold_error: bool = False
    corruption_fallback: bool = False


class ToolUseCounts(Frozen):
    """Build counts: questions consumed, reference-query errors, corruption fallbacks."""

    n_questions: int
    n_gold_errors: int
    n_corruption_fallbacks: int  # wanted `revise` but no corruption errored → built `check`
    n_revise: int
    n_check: int
    n_direct: int
    n_direct_skipped: int = 0  # traces with no `<think>` span (nothing left after the strip)
    n_rows: int


class ToolUseManifest(Frozen):
    """Provenance for a tool-use dataset: enough to reproduce and audit the build."""

    dataset: str
    dataset_revision: str
    mix: ToolUseMix
    feedback: FeedbackConfig
    trace_dataset_sha256: str  # trace dataset the `direct` rows are drawn from
    counts: ToolUseCounts
    git_commit: str = "unknown"
    git_dirty: bool = False


# Seeded final-answer templates — varied so SFT does not teach one frozen sentence. The `<think>`
# text references observed feedback or the check outcome, never the reference query, and never
# quotes the env's "Execution feedback" marker.
_REVISE_THINKS = (
    "The first query failed — the error says {reason}. I fixed the query, re-checked it, "
    "and it executed and returned {rows}, which answers the question.",
    "My check surfaced an error ({reason}), so I revised the query. The second check executed "
    "and returned {rows}, matching what the question asks for.",
    "The check flagged {reason} in my first attempt. The corrected query checks out "
    "({rows}), so I commit to it.",
)
_CHECK_THINKS = (
    "I checked the query with run_sql: it executed and returned {rows}, consistent with the "
    "question, so I commit to it.",
    "The check executed cleanly and returned {rows} answering the question, so this query "
    "is my final answer.",
)


def corruption_candidates(sql: str, schema: DatabaseSchema) -> Iterator[str]:
    """Yield plausibly-wrong variants of `sql` (analyst-style column mistakes), most natural
    first. The caller keeps the first one the sandbox actually rejects — candidates here are
    only guesses."""
    referenced = [
        column.name
        for table in schema.tables
        for column in table.columns
        if re.search(rf"\b{re.escape(column.name)}\b", sql, re.IGNORECASE)
    ]
    unreferenced = [
        column.name
        for table in schema.tables
        if not re.search(rf"\b{re.escape(table.name)}\b", sql, re.IGNORECASE)
        for column in table.columns
    ]
    for name in referenced:
        # A column that exists in the schema but not in the queried tables — the classic join
        # mix-up, and the same mistake the mt_demo demonstration shows.
        for wrong in unreferenced:
            if wrong.lower() != name.lower():
                yield re.sub(rf"\b{re.escape(name)}\b", wrong, sql, count=1, flags=re.IGNORECASE)
                break
        # Misspellings: dropped last character, or a stray `_id` suffix.
        if len(name) > 3:
            yield re.sub(rf"\b{re.escape(name)}\b", name[:-1], sql, count=1, flags=re.IGNORECASE)
        yield re.sub(rf"\b{re.escape(name)}\b", f"{name}_id", sql, count=1, flags=re.IGNORECASE)
    # Queries that reference no column by name (e.g. COUNT(*)) still reference tables —
    # misspell one for the "no such table" error class.
    for table in schema.tables:
        if len(table.name) > 3 and re.search(rf"\b{re.escape(table.name)}\b", sql, re.IGNORECASE):
            yield re.sub(
                rf"\b{re.escape(table.name)}\b", table.name[:-1], sql, count=1, flags=re.IGNORECASE
            )


def _rows_phrase(feedback: str) -> str:
    match = re.search(r"returned (\d+ rows?(?: \(truncated\))?)", feedback)
    return match.group(1) if match else "a result"


def _reason_phrase(feedback: str) -> str:
    match = re.search(r"error \(\w+\): (.{1,80})", feedback)
    return match.group(1).strip() if match else "an execution error"


def _final_answer(think: str, sql: str) -> Message:
    return {"role": "assistant", "content": f"<think>{think}</think>\n```sql\n{sql}\n```"}


def _transcript_rows(
    example: Example,
    schema_text: str,
    turns: Sequence[tuple[Message, Message]],
    final: Message,
    shape: str,
) -> list[ToolUseRow]:
    """Split one demonstration transcript into per-assistant-turn prompt-completion rows."""
    base: list[Message] = [
        {"role": "system", "content": MULTITURN_INSTRUCTION},
        {"role": "user", "content": user_content(schema_text, example.question)},
    ]
    rows: list[ToolUseRow] = []
    prefix = list(base)
    for tool_call, tool_feedback in turns:
        rows.append(
            ToolUseRow(
                db_id=example.db_id,
                question=example.question,
                shape=shape,
                prompt=tuple(prefix),
                completion=(tool_call,),
            )
        )
        prefix = [*prefix, tool_call, tool_feedback]
    rows.append(
        ToolUseRow(
            db_id=example.db_id,
            question=example.question,
            shape=shape,
            prompt=tuple(prefix),
            completion=(final,),
        )
    )
    return rows


def _first_erroring_corruption(
    example: Example,
    schema: DatabaseSchema,
    sandbox: SqliteSandbox,
    feedback: FeedbackConfig,
) -> tuple[str, str] | None:
    for candidate in corruption_candidates(example.gold_sql, schema):
        outcome = sandbox.execute(candidate)
        if isinstance(outcome, ExecutionError):
            return candidate, format_feedback(outcome, feedback)
    return None


def _tool_example_rows(
    example: Example,
    schema: DatabaseSchema,
    db_root: str | Path,
    rng: random.Random,
    feedback: FeedbackConfig,
    *,
    want_revise: bool,
    can_check: bool,
) -> _ToolExampleRows:
    from llm_rlvr.data.schema import serialize_schema

    sandbox = SqliteSandbox(
        resolve_database(example.db_id, db_root),
        timeout_s=AGENT_TIMEOUT_S,
        row_cap=AGENT_ROW_CAP,
    )
    gold_outcome = sandbox.execute(example.gold_sql)
    if isinstance(gold_outcome, ExecutionError):
        return _ToolExampleRows(gold_error=True)

    schema_text = serialize_schema(schema)
    gold_feedback = format_feedback(gold_outcome, feedback)
    success_turn = (
        tool_call_message(example.gold_sql),
        {"role": "tool", "content": gold_feedback},
    )

    if want_revise:
        corrupted = _first_erroring_corruption(example, schema, sandbox, feedback)
        if corrupted is not None:
            bad_sql, bad_feedback = corrupted
            think = rng.choice(_REVISE_THINKS).format(
                reason=_reason_phrase(bad_feedback), rows=_rows_phrase(gold_feedback)
            )
            return _ToolExampleRows(
                rows=tuple(
                    _transcript_rows(
                        example,
                        schema_text,
                        [
                            (
                                tool_call_message(bad_sql),
                                {"role": "tool", "content": bad_feedback},
                            ),
                            success_turn,
                        ],
                        _final_answer(think, example.gold_sql),
                        "revise",
                    )
                ),
                shape="revise",
            )
        if not can_check:
            return _ToolExampleRows(corruption_fallback=True)

    think = rng.choice(_CHECK_THINKS).format(rows=_rows_phrase(gold_feedback))
    return _ToolExampleRows(
        rows=tuple(
            _transcript_rows(
                example,
                schema_text,
                [success_turn],
                _final_answer(think, example.gold_sql),
                "check",
            )
        ),
        shape="check",
        corruption_fallback=want_revise,
    )


def _direct_rows(
    traces: Sequence[SftTrace],
    rng: random.Random,
    n_target: int,
) -> tuple[list[ToolUseRow], int, int]:
    rows: list[ToolUseRow] = []
    direct_pool = list(traces)
    rng.shuffle(direct_pool)
    n_skipped = 0

    for trace in direct_pool:
        if len(rows) >= n_target:
            break
        think_match = re.search(r"<think>", trace.completion, re.IGNORECASE)
        if think_match is None:
            n_skipped += 1
            continue
        think_start = think_match.start()
        rows.append(
            ToolUseRow(
                db_id=trace.db_id,
                question=trace.question,
                shape="direct",
                prompt=(
                    {"role": "system", "content": MULTITURN_INSTRUCTION},
                    *(m for m in trace.prompt if m.get("role") != "system"),
                ),
                completion=({"role": "assistant", "content": trace.completion[think_start:]},),
            )
        )

    return rows, len(rows), n_skipped


def build_tooluse_rows(
    examples: Sequence[Example],
    schemas: Mapping[str, DatabaseSchema],
    db_root: str | Path,
    traces: Sequence[SftTrace],
    *,
    mix: ToolUseMix | None = None,
    feedback: FeedbackConfig | None = None,
) -> tuple[tuple[ToolUseRow, ...], ToolUseCounts]:
    """Build the mixed tool-use dataset from train examples and traces (CPU, real sandbox).

    Raises loudly on any non-train example (R2 no-leakage, same invariant as the trace builder).
    """
    mixed = mix or ToolUseMix()
    fb_config = feedback or FeedbackConfig()
    assert_train_split(examples, context="tool-use dataset")

    rng = random.Random(mixed.seed)
    order = list(examples)
    rng.shuffle(order)

    rows: list[ToolUseRow] = []
    n_revise = n_check = n_gold_errors = n_fallbacks = n_questions = 0
    for example in order:
        if n_revise >= mixed.n_revise and n_check >= mixed.n_check:
            break
        n_questions += 1
        result = _tool_example_rows(
            example,
            schemas[example.db_id],
            db_root,
            rng,
            fb_config,
            want_revise=n_revise < mixed.n_revise,
            can_check=n_check < mixed.n_check,
        )
        if result.gold_error:
            n_gold_errors += 1
            continue

        if result.corruption_fallback:
            n_fallbacks += 1
        rows.extend(result.rows)
        if result.shape == "revise":
            n_revise += 1
        elif result.shape == "check":
            n_check += 1

    # Direct rows keep the model's own reasoning style when it chooses not to call the tool.
    # Strip preambles so direct rows do not re-teach text before the leading `<think>` format.
    # A trace with no `<think>` span is skipped and counted.
    direct_rows, n_direct, n_direct_skipped = _direct_rows(traces, rng, mixed.n_direct)
    rows.extend(direct_rows)

    counts = ToolUseCounts(
        n_questions=n_questions,
        n_gold_errors=n_gold_errors,
        n_corruption_fallbacks=n_fallbacks,
        n_revise=n_revise,
        n_check=n_check,
        n_direct=n_direct,
        n_direct_skipped=n_direct_skipped,
        n_rows=len(rows),
    )
    return tuple(rows), counts


def render_text_row(
    prompt: Sequence[Message],
    completion: Sequence[Message],
    apply_chat_template: Callable[..., str],
    tools: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    """Render one conversational row through the real chat template, including the tools block.

    The split relies on the template being prefix-stable: rendering the prompt with the
    generation prompt appended must be a strict prefix of rendering prompt+completion. The
    completion's trailing newline (after `<|im_end|>`) is stripped so the row ends exactly on
    the EOS token and the trainer's own EOS append stays idempotent.
    """
    prefix = apply_chat_template(
        list(prompt), tools=list(tools), add_generation_prompt=True, tokenize=False
    )
    full = apply_chat_template(
        [*prompt, *completion], tools=list(tools), add_generation_prompt=False, tokenize=False
    )
    if not full.startswith(prefix):
        raise ValueError(
            "chat template render is not prefix-stable; cannot split prompt/completion"
        )
    return {"prompt": prefix, "completion": full[len(prefix) :].rstrip("\n")}


def write_tooluse_dataset(rows: Sequence[ToolUseRow], path: str | Path) -> Path:
    """Write TRL conversational prompt-completion JSONL (one row per line), shuffled-as-given."""
    records = [
        {
            "prompt": list(row.prompt),
            "completion": list(row.completion),
            "db_id": row.db_id,
            "question": row.question,
            "shape": row.shape,
        }
        for row in rows
    ]
    return write_jsonl(records, path)


def build_tooluse_manifest(
    *,
    dataset: str,
    dataset_revision: str,
    mix: ToolUseMix,
    feedback: FeedbackConfig,
    trace_dataset_path: str | Path,
    counts: ToolUseCounts,
    commit: str | None = None,
    dirty: bool | None = None,
) -> ToolUseManifest:
    """Assemble the manifest: mix + feedback config + source-trace hash + funnel counts."""
    return ToolUseManifest(
        dataset=dataset,
        dataset_revision=dataset_revision,
        mix=mix,
        feedback=feedback,
        trace_dataset_sha256=sha256_of(trace_dataset_path),
        counts=counts,
        git_commit=commit if commit is not None else git_commit(),
        git_dirty=dirty if dirty is not None else git_is_dirty(),
    )
