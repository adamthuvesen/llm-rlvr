"""Shared vLLM engine bootstrap for generation backends."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from llm_rlvr.artifacts.provenance import DecodingConfig

Messages = list[dict[str, str]]


class _VllmOutput(Protocol):
    text: str


class _VllmRequestOutput(Protocol):
    outputs: Sequence[_VllmOutput]


class VllmEngine(Protocol):
    def chat(self, *args: Any, **kwargs: Any) -> Sequence[_VllmRequestOutput]: ...


def make_vllm_engine(
    model: str,
    *,
    model_revision: str | None = None,
    adapter_path: str | Path | None = None,
    max_lora_rank: int = 64,
) -> tuple[VllmEngine, dict[str, object]]:
    """Build a vLLM ``LLM`` and optional ``lora_request`` chat kwargs."""
    from vllm import LLM  # lazy import: GPU backend only

    chat_kwargs: dict[str, object] = {}
    llm_kwargs: dict[str, object] = {"model": model}
    if model_revision is not None:
        llm_kwargs["revision"] = model_revision
        llm_kwargs["tokenizer_revision"] = model_revision
    if adapter_path is not None:
        from vllm.lora.request import LoRARequest

        llm = LLM(**llm_kwargs, enable_lora=True, max_lora_rank=max_lora_rank)
        chat_kwargs["lora_request"] = LoRARequest("adapter", 1, str(adapter_path))
    else:
        llm = LLM(**llm_kwargs)
    return llm, chat_kwargs


def vllm_sampling_params(decoding: DecodingConfig, *, n: int = 1):
    from vllm import SamplingParams  # lazy import: GPU backend only

    return SamplingParams(
        n=n,
        temperature=decoding.temperature,
        top_p=decoding.top_p,
        max_tokens=decoding.max_new_tokens,
        seed=decoding.seed,
    )


def sample_vllm_completions(
    llm: VllmEngine,
    prompts: Sequence[Messages],
    decoding: DecodingConfig,
    *,
    chat_kwargs: dict[str, object] | None = None,
) -> list[str]:
    """Greedy or sampled chat completions, one string per prompt."""
    params = vllm_sampling_params(decoding)
    extra = chat_kwargs or {}
    results = llm.chat([list(p) for p in prompts], params, **extra)
    return [result.outputs[0].text for result in results]
