"""Shared vLLM engine bootstrap for generation backends."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from llm_rlvr.eval.provenance import DecodingConfig

Messages = list[dict[str, object]]


def make_vllm_engine(
    model: str,
    *,
    adapter_path: str | Path | None = None,
    max_lora_rank: int = 64,
) -> tuple[object, dict[str, object]]:
    """Build a vLLM ``LLM`` and optional ``lora_request`` chat kwargs."""
    from vllm import LLM  # lazy import: GPU backend only

    chat_kwargs: dict[str, object] = {}
    if adapter_path is not None:
        from vllm.lora.request import LoRARequest

        llm = LLM(model=model, enable_lora=True, max_lora_rank=max_lora_rank)
        chat_kwargs["lora_request"] = LoRARequest("adapter", 1, str(adapter_path))
    else:
        llm = LLM(model=model)
    return llm, chat_kwargs


def vllm_sampling_params(decoding: DecodingConfig):
    from vllm import SamplingParams  # lazy import: GPU backend only

    return SamplingParams(
        temperature=decoding.temperature,
        top_p=decoding.top_p,
        max_tokens=decoding.max_new_tokens,
        seed=decoding.seed,
    )


def sample_vllm_completions(
    llm: object,
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
