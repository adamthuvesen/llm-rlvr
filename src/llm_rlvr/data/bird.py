"""BIRD-dev data layer: typed examples (with `evidence`) + databases.

BIRD's `dev_tables.json` is the same format as Spider's `tables.json`, so schema parsing,
serialization, the sandbox, and the verifier are all reused verbatim. Only the example format
(`dev.json`: question / evidence / SQL) and the bundle layout are BIRD-specific. Self-hosted +
checksummed like Spider (design M4); BIRD is CC-BY-NC-4.0.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from llm_rlvr._base import Frozen
from llm_rlvr.data.artifacts import BIRD_ARCHIVE
from llm_rlvr.data.installation import extract_zip, install_archive
from llm_rlvr.data.models import Example
from llm_rlvr.data.paths import resolve_database
from llm_rlvr.data.spider import load_schemas

# Points at an already-extracted BIRD dev bundle root (the dir holding dev.json) — offline hatch.
_BIRD_ROOT_ENV = "TEXT2SQL_BIRD_ROOT"
_DEFAULT_CACHE = Path("data/bird")


class BirdDev(Frozen):
    """The local BIRD-dev tree: the bundle root, its dev.json, dev_tables.json, and database dir."""

    root: Path
    dev_json: Path
    tables_json: Path
    database_root: Path
    source: Literal["archive", "override", "legacy"] = "override"
    archive_sha256: str | None = None


def load_bird_examples(dev_json: str | Path) -> tuple[Example, ...]:
    """Parse BIRD `dev.json` into typed `Example`s, preserving the `evidence` hint."""
    rows = json.loads(Path(dev_json).read_text(encoding="utf-8"))
    ids = [row["question_id"] for row in rows if "question_id" in row]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate BIRD question_id")
    return tuple(
        Example(
            db_id=row["db_id"],
            question=row["question"],
            gold_sql=row["SQL"],
            split="dev",
            dataset="bird",
            evidence=(row.get("evidence") or None),
        )
        for row in rows
    )


def ensure_bird_dev(cache_dir: str | Path | None = None) -> BirdDev:
    """Return the local BIRD-dev tree, fetching + verifying + extracting if needed.

    Resolution order: ``$TEXT2SQL_BIRD_ROOT`` (an extracted bundle root), else the pinned,
    checksummed ``BIRD_ARCHIVE`` from HF (extracted, including the nested `dev_databases.zip`).
    """
    override = os.environ.get(_BIRD_ROOT_ENV)
    if override:
        return _bird_dev_at(Path(override))

    cache = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE

    if not (cache / ".installed").exists() and any(cache.glob("**/dev.json")):
        return _bird_dev_at(_find_bundle_root(cache)).model_copy(update={"source": "legacy"})

    def prepare(root: Path) -> None:
        nested = list(root.glob("**/dev_databases.zip"))
        if len(nested) > 1:
            raise ValueError(f"multiple dev_databases.zip files under {root}")
        if nested:
            extract_zip(nested[0], nested[0].parent)

    installed = install_archive(
        cache,
        BIRD_ARCHIVE,
        prepare,
        validate=lambda root: _bird_dev_at(_find_bundle_root(root)),
    )
    verified = (installed / ".archive.sha256").is_file()
    return _bird_dev_at(_find_bundle_root(installed)).model_copy(
        update={
            "source": "archive" if verified else "legacy",
            "archive_sha256": BIRD_ARCHIVE.sha256 if verified else None,
        }
    )


def _find_bundle_root(cache: Path) -> Path:
    matches = list(cache.glob("**/dev.json"))
    if len(matches) > 1:
        raise ValueError(f"multiple dev.json files under {cache}")
    if not matches:
        raise FileNotFoundError(f"no dev.json found under {cache}")
    return matches[0].parent


def _bird_dev_at(root: Path) -> BirdDev:
    dev_json = root / "dev.json"
    tables_json = root / "dev_tables.json"
    database_root = root / "dev_databases"
    for required in (dev_json, tables_json, database_root):
        if not (required.is_dir() if required == database_root else required.is_file()):
            raise FileNotFoundError(
                f"expected a BIRD dev layout at {root} (dev.json + dev_tables.json + "
                f"dev_databases/); missing {required.name}"
            )
    schemas = load_schemas(tables_json)
    for db_id in schemas:
        resolve_database(db_id, database_root)
    for example in load_bird_examples(dev_json):
        if example.db_id not in schemas:
            raise ValueError(f"unknown database id {example.db_id!r} in BIRD examples")
    return BirdDev(
        root=root, dev_json=dev_json, tables_json=tables_json, database_root=database_root
    )
