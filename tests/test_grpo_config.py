"""Tests for GRPOConfig kwargs filtering against TRL drift."""

from __future__ import annotations

import dataclasses
import logging
import sys
import types

from llm_rlvr.train.grpo import build_grpo_config_from_kwargs


@dataclasses.dataclass
class _FakeGrpoConfig:
    learning_rate: float = 1e-5
    seed: int = 0


def test_build_grpo_config_from_kwargs_drops_unknown_and_warns(monkeypatch, caplog) -> None:
    fake_trl = types.ModuleType("trl")
    fake_trl.GRPOConfig = _FakeGrpoConfig
    monkeypatch.setitem(sys.modules, "trl", fake_trl)

    with caplog.at_level(logging.WARNING):
        cfg = build_grpo_config_from_kwargs(
            {"learning_rate": 2e-5, "seed": 1, "removed_field": True}
        )

    assert isinstance(cfg, _FakeGrpoConfig)
    assert cfg.learning_rate == 2e-5
    assert cfg.seed == 1
    assert any("removed_field" in record.message for record in caplog.records)
