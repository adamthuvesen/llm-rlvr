"""Requested training settings must survive TRL version drift."""

from __future__ import annotations

import dataclasses
import sys
import types

import pytest

from llm_rlvr.train.grpo import build_grpo_config_from_kwargs


@dataclasses.dataclass
class _FakeGrpoConfig:
    learning_rate: float = 1e-5
    seed: int = 0


def test_build_grpo_config_rejects_unknown_fields(monkeypatch) -> None:
    fake_trl = types.ModuleType("trl")
    fake_trl.GRPOConfig = _FakeGrpoConfig
    monkeypatch.setitem(sys.modules, "trl", fake_trl)

    with pytest.raises(ValueError, match="removed_field") as exc:
        build_grpo_config_from_kwargs({"learning_rate": 2e-5, "seed": 1, "removed_field": True})
    assert "trl=" in str(exc.value)
    assert "Install" in str(exc.value)
    assert "No fields were dropped" in str(exc.value)


def test_build_grpo_config_preserves_supported_fields(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(GRPOConfig=_FakeGrpoConfig))
    config = build_grpo_config_from_kwargs({"learning_rate": 2e-5, "seed": 1})
    assert config == _FakeGrpoConfig(learning_rate=2e-5, seed=1)
