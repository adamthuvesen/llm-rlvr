"""Multi-turn eval (CPU half): tool-call parsing and transcript round-trip."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from llm_rlvr.data.models import DatabaseSchema, Example
from llm_rlvr.env.text2sql_env import TurnRecord
from llm_rlvr.eval.multiturn_generate import (
    MultiTurnEpisode,
    MultiTurnTranscriptSet,
    generate_multiturn,
    parse_tool_call,
)
from llm_rlvr.eval.provenance import DecodingConfig, GenerationProvenance


def test_parse_tool_call_qwen_wire_format() -> None:
    text = '<tool_call>\n{"name": "run_sql", "arguments": {"query": "SELECT 1"}}\n</tool_call>'
    assert parse_tool_call(text) == "SELECT 1"
    # Leading whitespace/text around the block is tolerated (vLLM completions often have it).
    assert parse_tool_call(f"\n{text}\n") == "SELECT 1"


def test_parse_tool_call_rejects_final_answers_and_malformed() -> None:
    assert parse_tool_call("<think>done</think>\n```sql\nSELECT 1\n```") is None
    assert parse_tool_call("<tool_call>\nnot json\n</tool_call>") is None
    assert (
        parse_tool_call(
            '<tool_call>{"name": "other_tool", "arguments": {"query": "x"}}</tool_call>'
        )
        is None
    )
    assert parse_tool_call('<tool_call>{"name": "run_sql", "arguments": {}}</tool_call>') is None
    assert (
        parse_tool_call('<tool_call>{"name": "run_sql", "arguments": {"query": "  "}}</tool_call>')
        is None
    )


def test_transcript_set_round_trips() -> None:
    provenance = GenerationProvenance(
        model="m",
        backend="vllm-multiturn",
        dataset="spider",
        dataset_revision="r",
        split="dev",
        n_examples=1,
        decoding=DecodingConfig(),
        prompt_style="mt",
    )
    transcripts = MultiTurnTranscriptSet(
        provenance=provenance,
        items=(
            MultiTurnEpisode(
                db_id="d",
                question="q",
                turns=(TurnRecord(query="SELECT 1", feedback="Execution feedback: ok"),),
                n_generations=2,
                final_text="<think>x</think>\n```sql\nSELECT 1\n```",
                ended_by="answer",
            ),
        ),
    )
    restored = MultiTurnTranscriptSet.model_validate_json(transcripts.model_dump_json())
    assert restored == transcripts
    assert restored.provenance.backend == "vllm-multiturn"


def test_multiturn_generation_passes_model_revision_to_tokenizer_and_vllm(
    monkeypatch: pytest.MonkeyPatch, scoring_root
) -> None:
    seen: dict[str, object] = {}

    class FakeTokenizer:
        def apply_chat_template(self, messages, tools, add_generation_prompt, tokenize):
            return "rendered prompt"

    def from_pretrained(model, **kwargs):
        seen["tokenizer_model"] = model
        seen["tokenizer_kwargs"] = kwargs
        return FakeTokenizer()

    class FakeLLM:
        def __init__(self, **kwargs) -> None:
            seen["llm_kwargs"] = kwargs

        def generate(self, prompts, params, **_kwargs):
            return [
                SimpleNamespace(
                    outputs=[SimpleNamespace(text="<think>x</think>\n```sql\nSELECT 1\n```")]
                )
                for _prompt in prompts
            ]

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=from_pretrained)
    transformers_utils = ModuleType("transformers.utils")
    transformers_utils.get_json_schema = lambda _fn: {"name": "run_sql"}
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", transformers_utils)
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **kwargs: kwargs),
    )
    example = Example(db_id="concert", question="q", gold_sql="SELECT a FROM t", split="dev")
    schema = DatabaseSchema(db_id="concert", tables=())

    completion_set, transcripts = generate_multiturn(
        [example],
        {"concert": schema},
        scoring_root,
        model="fake",
        model_revision="abc123",
        decoding=DecodingConfig(max_new_tokens=16),
        dataset="spider",
        dataset_revision="rev",
        split="dev",
    )

    assert seen["tokenizer_kwargs"] == {"revision": "abc123"}
    assert seen["llm_kwargs"]["revision"] == "abc123"
    assert seen["llm_kwargs"]["tokenizer_revision"] == "abc123"
    assert completion_set.provenance.model_revision == "abc123"
    assert transcripts.items[0].final_text.startswith("<think>")
