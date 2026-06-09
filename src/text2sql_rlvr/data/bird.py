"""BIRD-dev data layer: typed examples (with `evidence`) + databases.

BIRD's `dev_tables.json` is the same format as Spider's `tables.json`, so schema parsing,
serialization, the sandbox, and the verifier are all reused verbatim. Only the example format
(`dev.json`: question / evidence / SQL) and the bundle layout are BIRD-specific. Self-hosted +
checksummed like Spider (design M4); BIRD is CC-BY-NC-4.0.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

from text2sql_rlvr._base import Frozen
from text2sql_rlvr.data.artifacts import BIRD_ARCHIVE
from text2sql_rlvr.data.models import Example
from text2sql_rlvr.data.spider import verify_sha256  # reused checksum gate

# Points at an already-extracted BIRD dev bundle root (the dir holding dev.json) — offline hatch.
_BIRD_ROOT_ENV = "TEXT2SQL_BIRD_ROOT"
_DEFAULT_CACHE = Path("data/bird")


class BirdDev(Frozen):
    """The local BIRD-dev tree: the bundle root, its dev.json, dev_tables.json, and database dir."""

    root: Path
    dev_json: Path
    tables_json: Path
    database_root: Path


def load_bird_examples(dev_json: str | Path) -> tuple[Example, ...]:
    """Parse BIRD `dev.json` into typed `Example`s, preserving the `evidence` hint."""
    rows = json.loads(Path(dev_json).read_text(encoding="utf-8"))
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

    if BIRD_ARCHIVE is None:
        raise RuntimeError(f"BIRD_ARCHIVE is not pinned; set ${_BIRD_ROOT_ENV} or finalize the pin")

    cache = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE
    marker = cache / ".extracted"
    if not marker.exists():
        from huggingface_hub import hf_hub_download

        archive = hf_hub_download(
            repo_id=BIRD_ARCHIVE.repo,
            filename=BIRD_ARCHIVE.filename,
            revision=BIRD_ARCHIVE.revision,
            repo_type="dataset",
        )
        verify_sha256(archive, BIRD_ARCHIVE.sha256)
        cache.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(cache)
        nested = next(cache.glob("**/dev_databases.zip"), None)
        if nested is not None:
            with zipfile.ZipFile(nested) as dbs:
                dbs.extractall(nested.parent)
        marker.write_text("ok", encoding="utf-8")

    return _bird_dev_at(_find_bundle_root(cache))


def _find_bundle_root(cache: Path) -> Path:
    dev = next(cache.glob("**/dev.json"), None)
    if dev is None:
        raise FileNotFoundError(f"no dev.json found under {cache}")
    return dev.parent


def _bird_dev_at(root: Path) -> BirdDev:
    dev_json = root / "dev.json"
    tables_json = root / "dev_tables.json"
    database_root = root / "dev_databases"
    for required in (dev_json, tables_json, database_root):
        if not required.exists():
            raise FileNotFoundError(
                f"expected a BIRD dev layout at {root} (dev.json + dev_tables.json + "
                f"dev_databases/); missing {required.name}"
            )
    return BirdDev(
        root=root, dev_json=dev_json, tables_json=tables_json, database_root=database_root
    )
