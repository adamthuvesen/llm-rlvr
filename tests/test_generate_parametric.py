"""baseline-evaluation: generation is base-model-parametric (the cross-family control path).

Stubs the model backend so the routing + provenance recording is testable on CPU — the real
model load is exercised only by the integration baseline run.
"""

from __future__ import annotations

import pytest

from llm_rlvr.artifacts.provenance import DecodingConfig
from llm_rlvr.data.models import Column, DatabaseSchema, Example, Table
from llm_rlvr.eval import generate as gen

QWEN = "Qwen/Qwen2.5-Coder-3B-Instruct"
LLAMA = "meta-llama/Llama-3.2-3B-Instruct"

_T = Table(name="t", columns=(Column(name="a", type="int"),))
SCHEMAS = {"db0": DatabaseSchema(db_id="db0", tables=(_T,))}
EXAMPLES = [Example(db_id="db0", question="q", gold_sql="SELECT a FROM t", split="dev")]


@pytest.fixture
def _stub_backend(monkeypatch: pytest.MonkeyPatch):
    """Replace the model backend with a recorder so no weights load."""
    seen: dict[str, object] = {}

    def fake_backend(prompts, *, model, backend, decoding, model_revision=None, adapter_path=None):
        seen["model"] = model
        seen["model_revision"] = model_revision
        seen["adapter_path"] = adapter_path
        return ["<think>x</think>\n```sql\nSELECT a FROM t\n```" for _ in prompts]

    monkeypatch.setattr(gen, "_run_backend", fake_backend)
    return seen


def _run(model: str, adapter_path=None, model_revision=None):
    return gen.generate(
        EXAMPLES,
        SCHEMAS,
        model=model,
        backend="vllm",
        decoding=DecodingConfig(),
        dataset="spider",
        dataset_revision="rev",
        split="dev",
        model_revision=model_revision,
        adapter_path=adapter_path,
    )


def test_qwen_base_records_qwen_in_provenance(_stub_backend) -> None:
    cs = _run(QWEN)
    assert _stub_backend["model"] == QWEN
    assert cs.provenance.model == QWEN
    assert cs.provenance.adapter is None  # base path: no adapter


def test_llama_base_routes_and_is_recorded(_stub_backend) -> None:
    cs = _run(LLAMA, adapter_path="grpo/grpo-llama-s0")
    assert _stub_backend["model"] == LLAMA  # the Llama base actually routed to the backend
    assert _stub_backend["adapter_path"] == "grpo/grpo-llama-s0"
    assert cs.provenance.model == LLAMA
    assert cs.provenance.adapter == "grpo/grpo-llama-s0"


def test_model_revision_routes_and_is_recorded(_stub_backend) -> None:
    cs = _run(QWEN, model_revision="abc123")
    assert _stub_backend["model_revision"] == "abc123"
    assert cs.provenance.model_revision == "abc123"
