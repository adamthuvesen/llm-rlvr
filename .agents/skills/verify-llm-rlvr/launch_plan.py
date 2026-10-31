"""Resolve the paid Modal launch plan and every unapproved launch through the real modal package.

Run by `verify launch-plan` from the repo root. Prints one JSON object. Nothing dispatches.
"""

import json
import sys
from collections.abc import Callable

import modal

import modal_app as m

FUNCTIONS = ["baseline", "sample_traces", "sft_train", "grpo_train"]
ENTRYPOINTS = {
    "main": {"dataset": "spider", "limit": 8, "spawn": False},
    "traces": {"limit": 8, "spawn": False},
    "sft": {},
    "grpo": {"run_id": "grpo-small", "limit": 32, "dev_size": 8, "spawn": False},
}
ARGV = {
    "main": ["--dataset", "spider", "--limit", "8", "--no-spawn"],
    "traces": ["--limit", "8", "--no-spawn"],
    "sft": [],
    "grpo": ["--run-id", "grpo-small", "--limit", "32", "--dev-size", "8", "--no-spawn"],
}


def outcome(call: Callable[[], object]) -> str:
    try:
        call()
    except SystemExit as exc:
        return str(exc)
    except Exception as exc:  # the drive reports a crash as the outcome text
        return f"CRASH {type(exc).__name__}: {exc}"
    return "DISPATCHED"


refusals = {}
for name, kwargs in ENTRYPOINTS.items():
    sys.argv = ["/usr/local/bin/modal", "run", f"modal_app.py::{name}", *ARGV[name]]
    refusals[name] = outcome(lambda name=name, kwargs=kwargs: getattr(m, name)(**kwargs))
for name in FUNCTIONS:
    refusals[name] = outcome(getattr(m, name).local)
invalid = outcome(lambda: m.grpo(limit=8, dev_size=8))

print(
    json.dumps(
        {
            "entrypoints_are_modal": all(
                isinstance(getattr(m, n), modal.app.LocalEntrypoint) for n in ENTRYPOINTS
            ),
            "image": m.CUDA_IMAGE,
            "model": m.MODEL,
            "gpus": {name: getattr(m, name).spec.gpus for name in FUNCTIONS},
            "refusals": refusals,
            "invalid_grpo": invalid,
        }
    )
)
