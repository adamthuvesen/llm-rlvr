"""The verifier protocol — the heart of RLVR.

A `Verifier` decides whether a model's answer is correct against a reference, by *checking* (running
code, comparing denotations, executing tests) rather than judging with another model. The verdict it
returns drives both the training reward and the evaluation score, so a single implementation keeps
the two honest: a bug in eval is a bug in training.

`reference` and `context` are deliberately untyped (`object`): a task defines what they mean. For
text-to-SQL the reference is the gold query and the context is the database path; for a math task
the reference is the gold answer and the context is unused; for code, the reference is a test
suite. When the *reference itself* is broken (gold fails to run), a verifier raises
`ReferenceUnverifiable` — that is the example's fault, not the model's, and callers handle it
distinctly from a wrong answer.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from llm_rlvr._base import Frozen


class Verdict(Frozen):
    """The outcome of verifying one answer.

    - `matched`: the answer is correct (the reward-bearing signal).
    - `valid`: the answer was well-formed enough to *check* — it parsed/executed without error.
      A wrong-but-runnable answer is `valid=True, matched=False`; an answer that errors is
      `valid=False`. Shaped rewards give partial credit for `valid` as a path toward `matched`.
    - `reason`: a short, stable tag for diagnostics and per-reason tallies (task-defined).
    """

    matched: bool
    valid: bool
    reason: str


class ReferenceUnverifiable(RuntimeError):  # noqa: N818  — deliberate API name (not an *Error)
    """The reference (gold) could not be verified against — a broken example, not a wrong answer.

    Carries a stable `reason` tag so callers can record it without string-matching the message.
    """

    def __init__(self, message: str, *, reason: str = "reference_error") -> None:
        super().__init__(message)
        self.reason = reason


@runtime_checkable
class Verifier(Protocol):
    """Check one answer against one reference in one context, returning a `Verdict`.

    Implementations must be pure and side-effect-free with respect to the context (e.g. read-only
    database access). Raise `ReferenceUnverifiable` when the reference itself fails to evaluate.
    """

    def verify(self, answer: str, reference: object, context: object) -> Verdict: ...
