"""Forced-revision episode artifacts shared by eval, metrics, and reports."""

from __future__ import annotations

from llm_rlvr._base import Frozen
from llm_rlvr.eval.provenance import GenerationProvenance


class ForcedRevisionEpisode(Frozen):
    """One evaluated episode: attempt, feedback, and the selected final answer."""

    db_id: str
    question: str
    attempt_text: str
    feedback: str
    final_text: str
    revision_requested: bool = True
    revision_text: str | None = None


class ForcedRevisionTranscriptSet(Frozen):
    """All episodes of one forced-revision eval, sharing its CompletionSet's provenance."""

    provenance: GenerationProvenance
    items: tuple[ForcedRevisionEpisode, ...]
