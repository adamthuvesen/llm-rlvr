"""The answer-parsing protocol.

Models emit free text; a verifier needs the *answer* extracted from it (the SQL query, the final
number, the code block). An `AnswerParser` owns that task-specific extraction plus a notion of
"well-formed" output — the format the training reward and the SFT format-rate metric reward.

Extraction is defensive on purpose: the pre-SFT base model is unreliable, so a parser should recover
a usable answer where it can and return `None` only when there genuinely is none.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AnswerParser(Protocol):
    """Pull the answer out of a raw completion and judge whether it is well-formed."""

    def extract(self, text: str) -> str | None:
        """Return the answer found in `text`, or `None` if it contains none."""
        ...

    def well_formed(self, text: str) -> bool:
        """True iff `text` follows the trained output format exactly (the format-rate metric)."""
        ...
