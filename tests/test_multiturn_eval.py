"""Multi-turn eval (CPU half): tool-call parsing and transcript round-trip."""

from __future__ import annotations

from text2sql_rlvr.env.text2sql_env import TurnRecord
from text2sql_rlvr.eval.multiturn_generate import (
    MultiTurnEpisode,
    MultiTurnTranscriptSet,
    parse_tool_call,
)
from text2sql_rlvr.eval.provenance import DecodingConfig, GenerationProvenance


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
