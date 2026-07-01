"""docs/internal/artifacts.md pin strings must match ``data/artifacts.py`` constants."""

from __future__ import annotations

import re
from pathlib import Path

from llm_rlvr.data.artifacts import BIRD_ARCHIVE, SPIDER_DB_ARCHIVE, SPIDER_HF_REVISION

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS_MD = REPO / "docs/internal/artifacts.md"


def _extract_backtick_tokens(text: str) -> set[str]:
    return set(re.findall(r"`([0-9a-f]{40,64})`", text))


def test_artifacts_md_contains_code_pins() -> None:
    md = ARTIFACTS_MD.read_text(encoding="utf-8")
    tokens = _extract_backtick_tokens(md)

    assert SPIDER_HF_REVISION in tokens
    assert SPIDER_DB_ARCHIVE.revision in tokens
    assert SPIDER_DB_ARCHIVE.sha256 in tokens
    assert BIRD_ARCHIVE.revision in tokens
    assert BIRD_ARCHIVE.sha256 in tokens
