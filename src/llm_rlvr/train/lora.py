"""Shared PEFT LoRA kwargs for SFT and GRPO."""

from __future__ import annotations


def lora_kwargs(*, rank: int, alpha: int, dropout: float) -> dict[str, object]:
    return {
        "task_type": "CAUSAL_LM",
        "target_modules": "all-linear",
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "bias": "none",
    }
