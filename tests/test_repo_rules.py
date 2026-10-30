"""Repository rules outside the code: tracked files and the AGENTS.md rule table."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

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


def test_enforced_rules_cite_existing_targets() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    table = agents.split("## Enforced Rules", 1)[1].split("\n## ", 1)[0]
    rows = [line.split("|") for line in table.splitlines() if line.startswith("| ")][2:]
    assert rows, "AGENTS.md needs an `## Enforced Rules` table with at least one row"
    for row in rows:
        refs = re.findall(r"`([^`]+)`", row[2])
        assert refs, f"rule has no `path` or `path::name` citation: {row[1].strip()}"
        for ref in refs:
            path, _, name = ref.partition("::")
            target = ROOT / path
            assert target.exists(), f"AGENTS.md cites missing path {path!r}; fix the row"
            assert name in target.read_text(encoding="utf-8"), (
                f"AGENTS.md cites {name!r}, which is not in {path}; fix the row"
            )
