"""Explicit training API verification, with no model download or GPU execution."""

from __future__ import annotations

import importlib

import pytest

from llm_rlvr.train.grpo import GrpoHyperparams, build_grpo_config
from llm_rlvr.train.sft import SftHyperparams, build_lora_config, build_sft_config


@pytest.mark.runtime
def test_installed_training_apis_support_required_configuration() -> None:
    try:
        peft = importlib.import_module("peft")
        trl = importlib.import_module("trl")
    except ImportError as exc:
        pytest.fail(
            "runtime-check requires the training dependencies. "
            "Run uv sync --locked --extra dev --extra train on the training host. "
            f"Missing import: {exc}",
            pytrace=False,
        )
    lora = build_lora_config(SftHyperparams())
    assert isinstance(lora, peft.LoraConfig)
    assert lora.target_modules == "all-linear"
    config = build_sft_config(SftHyperparams(), "out")
    assert isinstance(config, trl.SFTConfig)
    assert config.completion_only_loss is True
    grpo = build_grpo_config(GrpoHyperparams(), "out")
    assert isinstance(grpo, trl.GRPOConfig)
    assert grpo.num_generations == 4
    assert grpo.use_vllm is True
    assert grpo.vllm_mode == "colocate"
