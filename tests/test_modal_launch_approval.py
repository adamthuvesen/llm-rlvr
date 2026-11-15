"""Paid Modal launches refuse to run until Adam approves them with `--approved`."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import pytest

from llm_rlvr.modal_helpers import GenerationRequest as Request
from llm_rlvr.modal_helpers import dispatch_remote

ROOT = Path(__file__).resolve().parents[1]
MODAL_APP = ROOT / "modal_app.py"
MODAL_SOURCES = (MODAL_APP, ROOT / "src/llm_rlvr/modal_helpers.py")
MODAL_CALLS = {"remote", "spawn", "map", "starmap", "for_each", "spawn_map"}


class FakeRemote:
    object_id = "fc-test"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def spawn(self, **kwargs: Any) -> FakeRemote:
        self.calls.append(f"spawn approved={kwargs['approved']}")
        return self

    def remote(self, **kwargs: Any) -> str:
        self.calls.append(f"remote approved={kwargs['approved']}")
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
    assert fn.calls == ["spawn approved=True"]


def _approved_default(function: ast.FunctionDef) -> bool:
    names = [a.arg for a in function.args.args][-len(function.args.defaults) :]
    default = dict(zip(names, function.args.defaults, strict=True)).get("approved")
    return isinstance(default, ast.Constant) and default.value is False


def _is_gate(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and ast.unparse(statement.value.func) == "require_approval"
        and ast.unparse(statement.value.args[0]) == "approved"
    )


def test_every_gpu_function_gates_on_approval_first() -> None:
    tree = ast.parse(MODAL_APP.read_text(encoding="utf-8"))
    gpu_functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            "modal_gpu_entrypoint" in ast.unparse(d)
            or (isinstance(d, ast.Call) and any(k.arg == "gpu" for k in d.keywords))
            for d in node.decorator_list
        )
    ]
    assert gpu_functions
    for function in gpu_functions:
        body = function.body[1:] if ast.get_docstring(function) else function.body
        assert _approved_default(function) and _is_gate(body[0]), (
            f"GPU function {function.name} needs `approved: bool = False` and "
            "`require_approval(approved, ...)` as its first statement"
        )


def test_modal_calls_follow_the_gate_and_forward_approval() -> None:
    for source in MODAL_SOURCES:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef):
                continue
            gates = [
                n.lineno for n in ast.walk(function) if isinstance(n, ast.stmt) and _is_gate(n)
            ]
            for node in ast.walk(function):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in MODAL_CALLS
                ):
                    continue
                forwards = any(
                    k.arg == "approved" and ast.unparse(k.value) == "True" for k in node.keywords
                )
                assert forwards and any(line < node.lineno for line in gates), (
                    f"{source.name}:{node.lineno} calls Modal without a prior "
                    "require_approval(approved, ...) and approved=True; use dispatch_remote"
                )


def test_every_modal_entrypoint_passes_its_approval_flag() -> None:
    tree = ast.parse(MODAL_APP.read_text(encoding="utf-8"))
    entrypoints = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any("local_entrypoint" in ast.unparse(d) for d in node.decorator_list)
    ]
    assert entrypoints
    for entry in entrypoints:
        assert _approved_default(entry), f"{entry.name} needs an `approved: bool = False` parameter"
        passes_flag = any(
            isinstance(node, ast.Call)
            and ast.unparse(node.func) == "dispatch_remote"
            and any(
                k.arg == "approved" and ast.unparse(k.value) == "approved" for k in node.keywords
            )
            for node in ast.walk(entry)
        )
        assert passes_flag, f"{entry.name} must call dispatch_remote(..., approved=approved)"
