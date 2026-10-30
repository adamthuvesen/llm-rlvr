"""Repository rules outside the code: tracked files and the AGENTS.md rule table."""

from __future__ import annotations

import re
import socket
import subprocess
from pathlib import Path

import pytest
from pytest_socket import SocketBlockedError

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_SUFFIXES = {
    *(".db", ".sqlite", ".sqlite3", ".parquet", ".arrow", ".jsonl", ".csv", ".zip", ".gz"),
    *(".pt", ".pth", ".safetensors", ".bin", ".ckpt"),
}
MAX_TRACKED_BYTES = 512 * 1024
SIZE_EXEMPT = {"uv.lock"}


def test_no_data_files_are_tracked() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split("\0")
    bad = [
        name
        for name in filter(None, tracked)
        if (ROOT / name).exists()
        and (
            name.startswith("data/")
            or Path(name).suffix in BLOCKED_SUFFIXES
            or (name not in SIZE_EXEMPT and (ROOT / name).stat().st_size > MAX_TRACKED_BYTES)
        )
    ]
    assert not bad, (
        f"data, weights, or files over {MAX_TRACKED_BYTES} bytes are tracked: {bad}. "
        "Pin datasets in src/llm_rlvr/data/artifacts.py and keep raw outputs under gitignored "
        "results/; then `git rm --cached` them."
    )


def _defines(text: str, name: str) -> bool:
    """True when `text` defines `name`: def/class, line-start key or make target, TOML table."""
    escaped = re.escape(name)
    return any(
        re.search(pattern, text, re.MULTILINE)
        for pattern in (
            rf"^\s*(?:async\s+)?(?:def|class)\s+{escaped}\b",
            rf"^{escaped}\s*[:=]",
            rf"^\[{{1,2}}{escaped}\]{{1,2}}\s*$",
        )
    )


def test_enforced_rules_cite_existing_targets() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "\n## Enforced Rules\n" in agents, "AGENTS.md must keep its `## Enforced Rules` section"
    table = agents.split("\n## Enforced Rules\n", 1)[1].split("\n## ", 1)[0]
    rows = [line.split("|") for line in table.splitlines() if line.startswith("| ")][2:]
    assert rows, "AGENTS.md needs an `## Enforced Rules` table with at least one row"
    for row in rows:
        refs = re.findall(r"`([^`]+)`", row[2])
        assert refs, f"rule cites no `path::name`: {row[1].strip()}"
        for ref in refs:
            path, separator, name = ref.partition("::")
            target = ROOT / path
            assert separator and name, f"AGENTS.md cites {ref!r}; use `path::name`"
            assert target.is_file(), f"AGENTS.md cites missing file {path!r}; fix the row"
            assert _defines(target.read_text(encoding="utf-8"), name), (
                f"AGENTS.md cites {name!r}, which {path} does not define; fix the row"
            )


@pytest.mark.filterwarnings("ignore:A test tried to use socket")
def test_unit_tests_cannot_open_tcp_sockets() -> None:
    with pytest.raises(SocketBlockedError):
        socket.create_connection(("127.0.0.1", 9))
