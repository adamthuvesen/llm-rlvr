"""Manifest integrity: every registered artifact path exists on disk."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "results") not in sys.path:
    sys.path.insert(0, str(_REPO / "results"))

from manifest import all_manifest_paths, repro_manifest  # noqa: E402


def test_repro_manifest_completion_sets_exist() -> None:
    missing = [p for p in all_manifest_paths() if not p.is_file()]
    assert not missing, f"missing manifest paths: {missing}"


def test_repro_manifest_is_non_empty() -> None:
    assert len(repro_manifest()) >= 20
