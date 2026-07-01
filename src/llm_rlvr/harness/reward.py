"""Generic shaped / near-binary reward over a verifier.

`score_answer` composes one per-completion reward from an `AnswerParser` + a `Verifier`:

  - **format** — credit for the trained reasoning-then-answer shape; full credit only when the
    `<think>` block *leads* (no prose preamble), partial credit otherwise. Text-only, task-agnostic.
  - **valid** — the answer parsed/executed without error (verdict `valid`); a path toward correct.
  - **matched** — the answer is correct (verdict `matched`); the dominant term.
  - **length** — a conciseness proxy in characters (the reward layer has no tokenizer).

Two arms, selected by `RewardConfig.arm`:

  - **shaped** (SQL-R1 style): the weighted sum of the four components above.
  - **near_binary** (Arctic style): `1.0` correct / `0.1` valid-but-wrong / `0` else — nothing else.

Task-specific concerns live behind the parser/verifier: what counts as an "answer", what "valid"
means, and any guards (e.g. the SQL empty-result guard) are configured on the `Verifier` the caller
passes in, not here. This keeps the shaping reusable across tasks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from llm_rlvr._base import Frozen
from llm_rlvr.harness.answer import AnswerParser
from llm_rlvr.harness.verifier import ReferenceUnverifiable, Verifier

RewardArm = Literal["shaped", "near_binary"]


def completion_text(completion: object) -> str:
    """Extract assistant text from a TRL completion (a string, a message, or a message list).

    TRL hands reward functions completions in several shapes depending on the trainer config; this
    normalizes them to the raw assistant text the parser/verifier expect. Task-agnostic.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, Sequence):
        for message in reversed(completion):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content", ""))
        return completion_text(completion[-1]) if completion else ""
    return str(completion)


class RewardConfig(Frozen):
    """Arm selection + shaped-arm component weights. Task-agnostic shaping only.

    The empty-result guard and other verifier knobs are NOT here — they belong to the `Verifier`
    the caller constructs and passes to `score_answer`.
    """

    arm: RewardArm = "shaped"
    # Shaped-arm component weights. The match term dominates; the rest shape the path to it.
    w_format: float = 0.2
    w_executable: float = 0.2
    w_result_match: float = 1.0
    w_length: float = 0.1
    # format component: credit for a well-formed completion whose <think> does NOT lead (has a
    # prose preamble). Full credit (1.0) requires a leading <think>; this caps the preamble case.
    preamble_credit: float = 0.3
    # length component: full credit up to `length_target_chars`, linear decay to 0 at
    # `length_max_chars`.
    length_target_chars: int = 400
    length_max_chars: int = 1600


class RewardBreakdown(Frozen):
    """The components behind a reward — surfaced for transparency and unit tests.

    `executable` and `result_match` mirror the verdict's `valid` and `matched` fields (the names are
    kept for continuity with the text-to-SQL reward and its tests).
    """

    total: float
    format: float
    executable: float
    result_match: float
    length: float
    reason: str


def _format_score(text: str, parser: AnswerParser, cfg: RewardConfig) -> float:
    """1.0 for a well-formed completion led by <think>; `preamble_credit` if a preamble precedes
    it; 0.0 if malformed."""
    if not parser.well_formed(text):
        return 0.0
    leads = text.lstrip().lower().startswith("<think>")
    return 1.0 if leads else cfg.preamble_credit


def _length_score(text: str, cfg: RewardConfig) -> float:
    """Conciseness term: full credit up to the target, linear decay to 0 at the max."""
    n = len(text.strip())
    if n <= cfg.length_target_chars:
        return 1.0
    if n >= cfg.length_max_chars:
        return 0.0
    span = cfg.length_max_chars - cfg.length_target_chars
    return (cfg.length_max_chars - n) / span


def score_answer(
    text: str,
    reference: object,
    context: object,
    *,
    parser: AnswerParser,
    verifier: Verifier,
    config: RewardConfig | None = None,
    no_answer_reason: str = "no_answer",
) -> RewardBreakdown:
    """Score one completion: parse the answer, verify it, and compose the shaped/near-binary reward.

    Pure and side-effect-free (the verifier touches its context read-only). `reference`/`context`
    are passed through to the verifier unchanged. When no answer can be parsed the reward is the
    format+length shaping only, tagged `no_answer_reason`; when the *reference* is unverifiable the
    verifier's `ReferenceUnverifiable.reason` is surfaced and no match/valid credit is given.
    """
    cfg = config or RewardConfig()
    fmt = _format_score(text, parser, cfg)
    length = _length_score(text, cfg)

    answer = parser.extract(text)
    if answer is None:
        return _assemble(
            cfg, fmt=fmt, valid=0.0, matched=0.0, length=length, reason=no_answer_reason
        )

    try:
        verdict = verifier.verify(answer, reference, context)
    except ReferenceUnverifiable as exc:
        # The reference is broken, not the prediction; record it without rewarding the model.
        return _assemble(cfg, fmt=fmt, valid=0.0, matched=0.0, length=length, reason=exc.reason)

    return _assemble(
        cfg,
        fmt=fmt,
        valid=1.0 if verdict.valid else 0.0,
        matched=1.0 if verdict.matched else 0.0,
        length=length,
        reason=verdict.reason,
    )


def _assemble(
    cfg: RewardConfig,
    *,
    fmt: float,
    valid: float,
    matched: float,
    length: float,
    reason: str,
) -> RewardBreakdown:
    if cfg.arm == "near_binary":
        total = 1.0 if matched else (0.1 if valid else 0.0)
    else:
        total = (
            cfg.w_format * fmt
            + cfg.w_executable * valid
            + cfg.w_result_match * matched
            + cfg.w_length * length
        )
    return RewardBreakdown(
        total=total,
        format=fmt,
        executable=valid,
        result_match=matched,
        length=length,
        reason=reason,
    )
