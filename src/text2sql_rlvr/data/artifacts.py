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


# Finalize in R1: point at the chosen HF mirror commit that ships `database/*.sqlite`, with the
# SHA256 computed from the canonical Yale archive. Left None so nothing silently fetches an
# unverified archive.
SPIDER_DB_ARCHIVE: ArchivePin | None = None
