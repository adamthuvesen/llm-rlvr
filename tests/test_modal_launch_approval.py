"""Paid Modal launches refuse to run until Adam approves them with `--approved`."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import pytest

from llm_rlvr.modal_helpers import GenerationRequest as Request
from llm_rlvr.modal_helpers import dispatch_remote

MODAL_APP = Path(__file__).resolve().parents[1] / "modal_app.py"


class FakeRemote:
    object_id = "fc-test"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def spawn(self, **kwargs: Any) -> FakeRemote:
        self.calls.append("spawn")
        return self

    def remote(self, **kwargs: Any) -> str:
        self.calls.append("remote")
        return "done"


def test_unapproved_launch_refuses_before_any_modal_call(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["modal", "run", "--detach", "modal_app.py::grpo"])
    fn = FakeRemote()
    with pytest.raises(SystemExit) as refusal:
        dispatch_remote(
            fn, Request(base_model="m"), approved=False, spawn=True, label="GRPO", app_name="a"
        )
    message = str(refusal.value.code)
    assert "spends money" in message and "ask Adam" in message
    assert "modal run --detach modal_app.py::grpo --approved" in message
    assert fn.calls == []


def test_approved_launch_dispatches() -> None:
    fn = FakeRemote()
    dispatch_remote(
        fn, Request(base_model="m"), approved=True, spawn=True, label="GRPO", app_name="a"
    )
    assert fn.calls == ["spawn"]


def test_every_modal_entrypoint_passes_its_approval_flag() -> None:
    tree = ast.parse(MODAL_APP.read_text(encoding="utf-8"))
    remote_calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"remote", "spawn", "map", "starmap", "for_each"}
    ]
    assert not remote_calls, (
        f"modal_app.py calls Modal directly at lines {remote_calls}; "
        "use dispatch_remote(..., approved=approved)"
    )
    entrypoints = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any("local_entrypoint" in ast.unparse(d) for d in node.decorator_list)
    ]
    assert entrypoints
    for entry in entrypoints:
        names = [a.arg for a in entry.args.args][-len(entry.args.defaults) :]
        defaults = dict(zip(names, entry.args.defaults, strict=True))
        approved = defaults.get("approved")
        assert isinstance(approved, ast.Constant) and approved.value is False, (
            f"{entry.name} needs an `approved: bool = False` parameter"
        )
        passes_flag = any(
            isinstance(node, ast.Call)
            and ast.unparse(node.func) == "dispatch_remote"
            and any(
                k.arg == "approved" and ast.unparse(k.value) == "approved" for k in node.keywords
            )
            for node in ast.walk(entry)
        )
        assert passes_flag, f"{entry.name} must call dispatch_remote(..., approved=approved)"
