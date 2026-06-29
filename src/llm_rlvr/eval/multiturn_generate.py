"""Multi-turn greedy evaluation: drive the `run_sql` tool loop at inference time (no RL).

The Stage-3 attribution baseline (and later the trained-policy eval, task 5.1): the model sees
the `mt` instruction + the tool schema, may call `run_sql` against the real read-only sandbox,
reads the feedback, and ends the episode with a final `<think>`+```sql``` answer. The final
answer goes into a normal `CompletionSet` (so the existing scorer and `make eval` work
unchanged, `backend="vllm-multiturn"` disclosed in provenance); the full transcripts are
persisted alongside as a `MultiTurnTranscriptSet` so the self-correction analysis is
re-derivable (design D9).

Parsing is the Qwen wire format the chat template teaches: a `<tool_call>{...json...}</tool_call>`
block. A completion with no parseable tool call is the episode's final answer. The env's own
turn budget bounds the loop; one extra generation is allowed after budget exhaustion so the
model can still answer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from llm_rlvr._base import Frozen
from llm_rlvr.data.models import DatabaseSchema, Example
from llm_rlvr.data.schema import serialize_schema
from llm_rlvr.env.text2sql_env import FeedbackConfig, SqliteText2SqlEnv, TurnRecord
from llm_rlvr.eval.completions import CompletionSet
from llm_rlvr.eval.messages import tool_call_message, user_content
from llm_rlvr.eval.prompt import MULTITURN_INSTRUCTION
from llm_rlvr.eval.provenance import DecodingConfig, GenerationProvenance
from llm_rlvr.eval.runner import finish_eval_run

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_call(text: str) -> str | None:
    """Extract the `run_sql` query from a completion, or None if it is a final answer.

    Anything malformed (bad JSON, wrong tool name, missing/non-string query) counts as a final
    answer — at training time the trainer's parser would reject it the same way.
    """
    match = _TOOL_CALL_RE.search(text)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if payload.get("name") != "run_sql":
        return None
    query = (payload.get("arguments") or {}).get("query")
    return query if isinstance(query, str) and query.strip() else None


class MultiTurnEpisode(Frozen):
    """One evaluated episode: the executed turns and the final answer text."""

    db_id: str
    question: str
    turns: tuple[TurnRecord, ...]
    n_generations: int
    final_text: str
    ended_by: str  # answer | budget | generation_cap


class MultiTurnTranscriptSet(Frozen):
    """All episodes of one multi-turn eval, with the same provenance as its CompletionSet."""

    provenance: GenerationProvenance
    items: tuple[MultiTurnEpisode, ...]


class _EpisodeState:
    def __init__(self, example: Example, schema_text: str, env: SqliteText2SqlEnv) -> None:
        self.example = example
        self.env = env
        self.messages: list[dict[str, object]] = [
            {"role": "system", "content": MULTITURN_INSTRUCTION},
            {
                "role": "user",
                "content": user_content(schema_text, example.question, example.evidence),
            },
        ]
        self.n_generations = 0
        self.final_text: str | None = None
        self.ended_by = "generation_cap"


def generate_multiturn(
    examples: Sequence[Example],
    schemas: Mapping[str, DatabaseSchema],
    db_root: str | Path,
    *,
    model: str,
    decoding: DecodingConfig,
    dataset: str,
    dataset_revision: str,
    split: str,
    adapter_path: str | Path | None = None,
    model_revision: str | None = None,
    db_archive_sha256: str | None = None,
    environment: FeedbackConfig | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> tuple[CompletionSet, MultiTurnTranscriptSet]:
    """Run the batched multi-turn greedy loop on the GPU; return (CompletionSet, transcripts)."""
    from transformers import AutoTokenizer
    from transformers.utils import get_json_schema

    from llm_rlvr.eval.vllm_backend import make_vllm_engine, vllm_sampling_params

    env_cfg = environment or FeedbackConfig()
    tokenizer = AutoTokenizer.from_pretrained(model)
    tools = [get_json_schema(SqliteText2SqlEnv(db_root, config=env_cfg).run_sql)]
    llm, chat_kwargs = make_vllm_engine(model, adapter_path=adapter_path)
    params = vllm_sampling_params(decoding)

    states: list[_EpisodeState] = []
    for example in examples:
        env = SqliteText2SqlEnv(db_root, config=env_cfg)
        env.reset(db_id=example.db_id)
        states.append(_EpisodeState(example, serialize_schema(schemas[example.db_id]), env))

    # ≤ max_turns tool executions per episode + one post-budget answer; the cap is defense in
    # depth against a policy that never stops calling the tool.
    generation_cap = env_cfg.max_turns + 2
    for _ in range(generation_cap):
        active = [s for s in states if s.final_text is None]
        if not active:
            break
        prompts = [
            tokenizer.apply_chat_template(
                s.messages, tools=tools, add_generation_prompt=True, tokenize=False
            )
            for s in active
        ]
        outputs = llm.generate(prompts, params, **chat_kwargs)
        for state, output in zip(active, outputs, strict=True):
            text = output.outputs[0].text
            state.n_generations += 1
            query = parse_tool_call(text)
            if query is None:
                state.final_text = text
                state.ended_by = "answer"
                continue
            already_exhausted = len(state.env.turns) >= env_cfg.max_turns
            feedback = state.env.run_sql(query)
            state.messages.append(tool_call_message(query))
            state.messages.append({"role": "tool", "content": feedback})
            if already_exhausted:
                # The model tool-called even after the budget message; one more generation
                # remains, after which the cap closes the episode.
                state.ended_by = "budget"

    final_texts = [s.final_text if s.final_text is not None else "" for s in states]
    completion_set = finish_eval_run(
        examples,
        final_texts,
        model=model,
        backend="vllm-multiturn",
        dataset=dataset,
        dataset_revision=dataset_revision,
        split=split,
        decoding=decoding,
        model_revision=model_revision,
        db_archive_sha256=db_archive_sha256,
        adapter_path=adapter_path,
        prompt_style="mt",
        commit=commit,
        dirty=dirty,
    )
    transcripts = MultiTurnTranscriptSet(
        provenance=completion_set.provenance,
        items=tuple(
            MultiTurnEpisode(
                db_id=s.example.db_id,
                question=s.example.question,
                turns=s.env.turns,
                n_generations=s.n_generations,
                final_text=s.final_text if s.final_text is not None else "",
                ended_by=s.ended_by if s.final_text is not None else "generation_cap",
            )
            for s in states
        ),
    )
    return completion_set, transcripts
