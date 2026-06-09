"""Pinned external data artifacts (revisions + checksums) for reproducibility.

Two Spider artifacts are pinned separately (design R1/D2):
  1. the HF parquet revision — question/SQL pairs only (verified present);
  2. the SQLite database archive — NOT shipped by `load_dataset`, fetched + checksum-gated.

The DB archive mirror + SHA256 is finalized when the integration download is first run (R1);
until then `SPIDER_DB_ARCHIVE` is None and the resolver requires an explicit, checksummed source.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

# Resolved 2026-06 from the HF API; parquet train+validation, no databases (confirms D2).
SPIDER_HF_REPO = "xlangai/spider"
SPIDER_HF_REVISION = "0c350918f3f29ec754f1181c65cdce76cd6c133c"


class ArchivePin(BaseModel):
    """A pinned, checksummed downloadable archive (the trust anchor is the SHA256)."""

    model_config = ConfigDict(frozen=True)

    # Where to fetch from. `repo` is an HF dataset id and `filename` a file within it; a bare
    # local path may be passed to the resolver instead for an offline/Yale-sourced archive.
    repo: str
    filename: str
    revision: str
    sha256: str  # computed against the canonical Yale bytes; mismatch fails loudly


# Self-hosted (design R1): no third-party mirror ships the Spider SQLite databases, so we re-host
# the canonical Yale `spider_data.zip` (CC-BY-SA-4.0) on our own HF dataset repo, pinned by the
# immutable commit SHA. The SHA256 is the hard trust anchor (computed from the Yale download,
# 2026-06). Yale Drive id 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J is the documented provenance origin.
SPIDER_DB_ARCHIVE: ArchivePin | None = ArchivePin(
    repo="HAL-9001/spider-databases",
    filename="spider_data.zip",
    revision="4a01bbac6520cd35b216db9e1724e5e1ada60aa4",
    sha256="00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b",
)
