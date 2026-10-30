"""Entrypoint validation runs without creating or calling a real Modal service."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def modal_module(monkeypatch: pytest.MonkeyPatch):
    class Remote:
        def __init__(self, function):
            self.local = function
            self.calls = []

        def spawn(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(object_id="fake-call")

        def remote(self, **kwargs):
            self.calls.append(kwargs)
            return "fake-output"

    class App:
        def __init__(self, _name):
            pass

        def function(self, **_kwargs):
            return Remote

        def local_entrypoint(self):
            return lambda function: function

    class Image:
        @classmethod
        def from_registry(cls, *_args, **_kwargs):
            return cls()

        def __getattr__(self, _name):
            return lambda *_args, **_kwargs: self

    class Volume:
        @classmethod
        def from_name(cls, *_args, **_kwargs):
            return cls()

        def commit(self):
            pass

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(App=App, Image=Image, Volume=Volume))
    spec = importlib.util.spec_from_file_location(
        "modal_preflight_fixture", Path(__file__).parents[1] / "modal_app.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "git_kwargs", lambda: {"commit": "abc", "dirty": False})
    return module


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dataset": "bird", "few_shot_k": 1},
        {"limit": 0},
        {"max_new_tokens": 0},
        {"adapter": "/outside"},
        {"model_revision": ""},
    ],
)
def test_invalid_generation_never_dispatches(modal_module, kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        modal_module.main(**kwargs)
    assert modal_module.baseline.calls == []


@pytest.mark.parametrize(
    "entrypoint,kwargs,remote",
    [
        ("traces", {"n": 0}, "sample_traces"),
        ("sft", {"learning_rate": 0}, "sft_train"),
        ("sft", {"traces": "../outside"}, "sft_train"),
        ("grpo", {"run_id": "../outside"}, "grpo_train"),
        ("grpo", {"reward_arm": "invalid"}, "grpo_train"),
        ("grpo", {"limit": 32, "dev_size": 32}, "grpo_train"),
    ],
)
def test_invalid_training_never_dispatches(
    modal_module, entrypoint: str, kwargs: dict[str, object], remote: str
) -> None:
    with pytest.raises(ValueError):
        getattr(modal_module, entrypoint)(**kwargs)
    assert getattr(modal_module, remote).calls == []


def test_valid_dispatch_carries_requested_values_and_source_identity(modal_module) -> None:
    with pytest.raises(SystemExit, match="--approved"):
        modal_module.main(max_new_tokens=777)
    assert modal_module.baseline.calls == []
    modal_module.main(max_new_tokens=777, model_revision="abc", spawn=True, approved=True)
    assert modal_module.baseline.calls[0]["max_new_tokens"] == 777
    assert modal_module.baseline.calls[0]["model_revision"] == "abc"
    assert modal_module.baseline.calls[0]["commit"] == "abc"
    modal_module.traces(spawn=False, approved=True)
    assert modal_module.sample_traces.calls[0]["commit"] == "abc"


def test_existing_remote_generation_run_rejected_before_context_and_model(
    modal_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(modal_module, "RUNS_DIR", str(tmp_path))
    (tmp_path / "generation" / "existing").mkdir(parents=True)
    monkeypatch.setattr(
        modal_module, "load_eval_context", lambda **_kwargs: pytest.fail("context loaded")
    )
    monkeypatch.setattr(
        "llm_rlvr.train.configuration.resolve_model_revision",
        lambda *_: pytest.fail("model inspected"),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        modal_module.baseline.local(run_id="existing", approved=True)


@pytest.mark.parametrize("function", ["baseline", "sample_traces", "sft_train", "grpo_train"])
def test_direct_gpu_function_call_refuses_without_approval(modal_module, function: str) -> None:
    with pytest.raises(SystemExit, match=f"modal_app.py::{function} <same arguments> --approved"):
        getattr(modal_module, function).local()


def test_remote_request_validation_also_rejects_invalid_paths(modal_module) -> None:
    with pytest.raises(ValueError, match="relative Volume path"):
        modal_module.baseline.local(adapter="../outside", approved=True)


@pytest.mark.parametrize("requested_revision", [None, "matching-tag"])
def test_modal_adapter_generation_pins_original_base_and_records_effective_request(
    modal_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, requested_revision: str | None
) -> None:
    import json

    from llm_rlvr.artifacts.io import sha256_of
    from llm_rlvr.artifacts.runs import RunRequestManifest
    from llm_rlvr.data.models import DatabaseSchema, Example
    from llm_rlvr.eval.completions import read_completion_set
    from llm_rlvr.train.configuration import inspect_adapter

    pinned = "a" * 40
    adapter = tmp_path / "sft" / "checkpoint"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {"base_model_name_or_path": "org/model", "r": 8, "lora_alpha": 16, "lora_dropout": 0}
        )
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter weights")
    (adapter / "provenance.json").write_text(
        json.dumps({"base_model": "org/model", "base_model_revision": pinned})
    )
    identity = inspect_adapter(adapter, base_model="org/model", base_model_revision=None)
    calls = []
    seen = {}

    def resolve(_model, revision):
        calls.append(revision)
        return pinned if revision in {pinned, "matching-tag"} else "b" * 40

    def backend(_prompts, **kwargs):
        seen.update(kwargs)
        return ["```sql\nSELECT 1\n```"]

    context = SimpleNamespace(
        examples=[Example(db_id="concert", question="q", gold_sql="SELECT 1", split="dev")],
        schemas={"concert": DatabaseSchema(db_id="concert", tables=())},
        dataset="spider",
        dataset_revision="revision",
        db_archive_sha256=None,
        data_source="override",
        train_pool=[],
    )
    monkeypatch.setattr(modal_module, "RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(modal_module, "load_eval_context", lambda **_kwargs: context)
    monkeypatch.setattr("llm_rlvr.train.configuration.resolve_model_revision", resolve)
    monkeypatch.setattr("llm_rlvr.eval.generate._run_backend", backend)
    out = Path(
        modal_module.baseline.local(
            adapter="sft/checkpoint",
            base_model="org/model",
            model_revision=requested_revision,
            limit=1,
            approved=True,
        )
    )
    assert None not in calls
    assert seen["model_revision"] == pinned
    result = read_completion_set(out)
    assert result.provenance.adapter_sha256 == identity.sha256
    assert result.provenance.adapter_provenance_sha256 == sha256_of(adapter / "provenance.json")
    request = RunRequestManifest.model_validate_json((out.parent / "request.json").read_text())
    assert request.request["generation"]["model_revision"] == requested_revision
    assert request.request["resolved_model_revision"] == pinned
    assert request.request["adapter_sha256"] == identity.sha256
    assert request.request["selection_policy"] == "prefix"


def test_modal_adapter_revision_mismatch_rejected_before_context_or_backend(
    modal_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    adapter = tmp_path / "sft" / "checkpoint"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "org/model",
                "revision": "a" * 40,
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0,
            }
        )
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter weights")
    monkeypatch.setattr(modal_module, "RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(
        modal_module, "load_eval_context", lambda **_kwargs: pytest.fail("context loaded")
    )
    monkeypatch.setattr(
        "llm_rlvr.train.configuration.resolve_model_revision",
        lambda _model, revision: "b" * 40 if revision == "different" else "a" * 40,
    )
    with pytest.raises(ValueError, match="base revision"):
        modal_module.baseline.local(
            adapter="sft/checkpoint",
            base_model="org/model",
            model_revision="different",
            approved=True,
        )
    assert not (tmp_path / "generation").exists()
