"""sft-training: hyperparameter mapping, completion-only loss contract, provenance (no GPU)."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_rlvr.train.sft import (
    SftHyperparams,
    build_adapter_provenance,
    build_sft_config_from_kwargs,
    lora_kwargs,
    sft_config_kwargs,
)


def test_lora_targets_all_linear_with_decided_rank() -> None:
    kwargs = lora_kwargs(SftHyperparams())
    assert kwargs["target_modules"] == "all-linear"
    assert kwargs["r"] == 32
    assert kwargs["lora_alpha"] == 64
    assert kwargs["lora_dropout"] == 0.05
    assert kwargs["task_type"] == "CAUSAL_LM"


def test_sft_config_enforces_completion_only_loss() -> None:
    kwargs = sft_config_kwargs(SftHyperparams(), "out")
    # The D3 contract: loss only over the assistant completion, prompt masked, no packing.
    assert kwargs["completion_only_loss"] is True
    assert kwargs["packing"] is False
    assert kwargs["bf16"] is True
    assert kwargs["max_length"] == 2048
    assert kwargs["learning_rate"] == 1e-4
    assert kwargs["lr_scheduler_type"] == "cosine"
    assert kwargs["warmup_ratio"] == 0.03
    assert kwargs["num_train_epochs"] == 3


def test_real_configs_build_when_train_extra_installed() -> None:
    # Auto-skips on the CPU box; runs wherever the train extra is synced (Modal image, GPU box).
    peft = pytest.importorskip("peft")
    trl = pytest.importorskip("trl")

    from llm_rlvr.train.sft import build_lora_config, build_sft_config

    lora = build_lora_config(SftHyperparams())
    assert isinstance(lora, peft.LoraConfig)
    assert lora.target_modules == "all-linear"

    config = build_sft_config(SftHyperparams(), "out")
    assert isinstance(config, trl.SFTConfig)
    assert config.completion_only_loss is True


def test_sft_config_builder_drops_unsupported_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    @dataclass
    class FakeSFTConfig:
        output_dir: str
        bf16: bool

    monkeypatch.setitem(sys.modules, "trl", SimpleNamespace(SFTConfig=FakeSFTConfig))
    config = build_sft_config_from_kwargs(
        {"output_dir": "out", "bf16": True, "removed_in_this_trl": "x"}
    )
    assert config == FakeSFTConfig(output_dir="out", bf16=True)


def test_adapter_provenance_hashes_dataset_and_records_versions(tmp_path: Path) -> None:
    traces = tmp_path / "traces.jsonl"
    traces.write_text('{"prompt": [], "completion": []}\n', encoding="utf-8")
    expected_sha = hashlib.sha256(traces.read_bytes()).hexdigest()

    provenance = build_adapter_provenance(
        base_model="Qwen/Qwen2.5-Coder-3B-Instruct",
        base_model_revision="abc123",
        hyperparams=SftHyperparams(),
        trace_dataset_path=traces,
        n_traces=1,
        commit="deadbeef",
        dirty=False,
    )
    assert provenance.trace_dataset_sha256 == expected_sha
    assert provenance.base_model_revision == "abc123"
    assert provenance.hyperparams.lora_rank == 32
    assert provenance.git_commit == "deadbeef"
    # Keys always present; values are real versions where installed, 'not-installed' otherwise.
    for package in ("trl", "peft", "transformers", "torch"):
        assert package in provenance.library_versions


def test_adapter_provenance_round_trips(tmp_path: Path) -> None:
    from llm_rlvr.train.sft import AdapterProvenance

    traces = tmp_path / "traces.jsonl"
    traces.write_text("{}\n", encoding="utf-8")
    provenance = build_adapter_provenance(
        base_model="m",
        hyperparams=SftHyperparams(),
        trace_dataset_path=traces,
        n_traces=0,
        commit="c",
        dirty=True,
    )
    assert AdapterProvenance.model_validate_json(provenance.model_dump_json()) == provenance
