"""Tool-use SFT dataset (Stage-3 rung 2): teach the `run_sql` protocol the smokes proved the
Stage-2 policy will not pick up in-context (0/288 episodes across `cot`/`mt`/`mt_demo`).

Each record is a TRL conversational prompt-completion row, one row per assistant turn of a
demonstration transcript, in three shapes:

  - `revise`  (3 rows): tool call with a corrupted query → real error feedback → tool call with
    the gold query → real success feedback → final `<think>`+```sql``` answer.
  - `check`   (2 rows): tool call with the gold query → real success feedback → final answer.
  - `direct`  (1 row): no tool call — a real Stage-1 trace completion under the `mt` system
    instruction, so the policy keeps its trained direct-answer quality (the forgetting guard).

Feedback strings are produced by executing the query in the real sandbox with the environment's
own caps and `format_feedback` — the training targets byte-match what the rollout env will say.
Tool-call turns are structured `tool_calls` messages, so the chat template (not us) renders the
wire format the trainer's parser expects. Everything is train-split only (R2 no-leakage).
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

from llm_rlvr_sql._base import Frozen
from llm_rlvr_sql.data.jsonl import write_jsonl
from llm_rlvr_sql.data.models import DatabaseSchema, Example
from llm_rlvr_sql.data.paths import resolve_database, sha256_of
from llm_rlvr_sql.data.splits import assert_train_split
from llm_rlvr_sql.env.sandbox import ExecutionError, SqliteSandbox
from llm_rlvr_sql.env.text2sql_env import (
    AGENT_ROW_CAP,
    AGENT_TIMEOUT_S,
    FeedbackConfig,
    format_feedback,
)
from llm_rlvr_sql.eval.messages import tool_call_message, user_content
from llm_rlvr_sql.eval.prompt import MULTITURN_INSTRUCTION
from llm_rlvr_sql.eval.provenance import git_commit, git_is_dirty
from llm_rlvr_sql.train.traces import SftTrace

Message = dict[str, object]


class ToolUseMix(Frozen):
    """How many transcripts of each shape, and the seed that picks questions + templates.

    The defaults balance the first-turn decision by TOKEN mass, not transcript count —
    attempt 4's lesson: 600 long direct traces out-massed 900 short tool calls ~5:1 and
    taught direct answering. 200 preamble-stripped direct completions ≈ the ~900 tool-call
    rows' token mass.
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


class ToolUseCounts(Frozen):
    """Build-funnel counts: questions consumed, golds that failed, corruption fallbacks."""

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
    trace_dataset_sha256: str  # the Stage-1 traces the `direct` rows are drawn from
    counts: ToolUseCounts
    git_commit: str = "unknown"
    git_dirty: bool = False


# Seeded final-answer templates — varied so the SFT does not teach one frozen sentence. The
# `<think>` text references the *observed* feedback (revise) or the check outcome, never gold,
# and never quotes the env's "Execution feedback" marker (attempt 4 showed the policy will
# learn to *generate* the marker if a template contains it — the mask check then fires).
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


def build_tooluse_rows(
    examples: Sequence[Example],
    schemas: Mapping[str, DatabaseSchema],
    db_root: str | Path,
    traces: Sequence[SftTrace],
    *,
    mix: ToolUseMix | None = None,
    feedback: FeedbackConfig | None = None,
) -> tuple[tuple[ToolUseRow, ...], ToolUseCounts]:
    """Build the mixed tool-use dataset from train examples + Stage-1 traces (CPU, real sandbox).

    Raises loudly on any non-train example (R2 no-leakage, same invariant as the trace builder).
    """
    from llm_rlvr_sql.data.schema import serialize_schema

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
        sandbox = SqliteSandbox(
            resolve_database(example.db_id, db_root),
            timeout_s=AGENT_TIMEOUT_S,
            row_cap=AGENT_ROW_CAP,
        )
        gold_outcome = sandbox.execute(example.gold_sql)
        if isinstance(gold_outcome, ExecutionError):
            n_gold_errors += 1
            continue
        schema_text = serialize_schema(schemas[example.db_id])
        gold_feedback = format_feedback(gold_outcome, fb_config)
        success_turn = (
            tool_call_message(example.gold_sql),
            {"role": "tool", "content": gold_feedback},
        )

        want_revise = n_revise < mixed.n_revise
        corrupted: tuple[str, str] | None = None
        if want_revise:
            for candidate in corruption_candidates(example.gold_sql, schemas[example.db_id]):
                outcome = sandbox.execute(candidate)
                if isinstance(outcome, ExecutionError):
                    corrupted = (candidate, format_feedback(outcome, fb_config))
                    break

        if want_revise and corrupted is not None:
            bad_sql, bad_feedback = corrupted
            think = rng.choice(_REVISE_THINKS).format(
                reason=_reason_phrase(bad_feedback), rows=_rows_phrase(gold_feedback)
            )
            rows.extend(
                _transcript_rows(
                    example,
                    schema_text,
                    [
                        (tool_call_message(bad_sql), {"role": "tool", "content": bad_feedback}),
                        success_turn,
                    ],
                    _final_answer(think, example.gold_sql),
                    "revise",
                )
            )
            n_revise += 1
            continue

        if want_revise:
            n_fallbacks += 1  # no candidate errored; the question still yields a check transcript
        if n_check >= mixed.n_check:
            continue
        think = rng.choice(_CHECK_THINKS).format(rows=_rows_phrase(gold_feedback))
        rows.extend(
            _transcript_rows(
                example,
                schema_text,
                [success_turn],
                _final_answer(think, example.gold_sql),
                "check",
            )
        )
        n_check += 1

    # `direct` rows: real Stage-1 trace completions, re-homed under the mt instruction so the
    # policy keeps producing its own reasoning style when it chooses not to call the tool.
    # Preambles are STRIPPED (everything before `<think>`): Stage-2's GRPO trained the leading-
    # `<think>` format, and attempt 4 showed verbatim preamble traces both re-teach the preamble
    # and out-mass the tool-call rows. A trace with no `<think>` span is skipped and counted.
    direct_pool = list(traces)
    rng.shuffle(direct_pool)
    n_direct = n_direct_skipped = 0
    for trace in direct_pool:
        if n_direct >= mixed.n_direct:
            break
        think_start = trace.completion.find("<think>")
        if think_start == -1:
            n_direct_skipped += 1
            continue
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
        n_direct += 1

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
    """Render one conversational row to TEXT prompt/completion through the real chat template,
    WITH the tools block — byte parity with what the rollout renders (attempt 4's disclosed
    train/rollout delta, closed).

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
