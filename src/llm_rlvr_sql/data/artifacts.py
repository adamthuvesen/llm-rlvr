"""Pinned external data artifacts (revisions + checksums) for reproducibility.

Two Spider artifacts are pinned separately (design R1/D2):
  1. the HF parquet revision — question/SQL pairs only (verified present);
  2. the SQLite database archive — NOT shipped by `load_dataset`, fetched + checksum-gated.
"""

from __future__ import annotations

from llm_rlvr_sql._base import Frozen

# Resolved 2026-06 from the HF API; parquet train+validation, no databases (confirms D2).
SPIDER_HF_REPO = "xlangai/spider"
SPIDER_HF_REVISION = "0c350918f3f29ec754f1181c65cdce76cd6c133c"


class ArchivePin(Frozen):
    """A pinned, checksummed downloadable archive (the trust anchor is the SHA256)."""

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
SPIDER_DB_ARCHIVE = ArchivePin(
    repo="HAL-9001/spider-databases",
    filename="spider_data.zip",
    revision="4a01bbac6520cd35b216db9e1724e5e1ada60aa4",
    sha256="00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b",
)

# BIRD-dev, self-hosted like Spider (canonical BIRD `dev.zip`, 2024-06-27: dev.json +
# dev_tables.json + nested dev_databases.zip). NOTE: BIRD is CC-BY-NC-4.0 (non-commercial), unlike
# Spider's CC-BY-SA. SHA256 is the trust anchor; pin `revision` to the upload commit SHA.
BIRD_ARCHIVE = ArchivePin(
    repo="HAL-9001/bird-dev",
    filename="dev.zip",
    revision="837584317bf7e3d86a3f6d257bd5553b7e03e171",
    sha256="cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630",
)
