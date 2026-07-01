"""Shared trainable policy loading for SFT and GRPO warm starts."""

from __future__ import annotations

from pathlib import Path


def _revision_kwargs(base_model_revision: str | None) -> dict[str, str]:
    return {"revision": base_model_revision} if base_model_revision else {}


def load_base_causal_lm(base_model: str, base_model_revision: str | None):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype="bfloat16",
        **_revision_kwargs(base_model_revision),
    )


def load_trainable_lora_adapter(
    base_model: str,
    base_model_revision: str | None,
    adapter_path: str | Path,
):
    from peft import PeftModel

    base = load_base_causal_lm(base_model, base_model_revision)
    return PeftModel.from_pretrained(base, str(adapter_path), is_trainable=True)
