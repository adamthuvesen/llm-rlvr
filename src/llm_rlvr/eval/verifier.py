"""Compatibility imports for the SQL execution-match verifier."""

from llm_rlvr.tasks.sql.verifier import (
    DEFAULT_VERIFY_ROW_CAP,
    SCORER_POLICY_VERSION,
    GoldExecutionError,
    MatchReason,
    MatchResult,
    VerifierConfig,
    execution_match,
    validate_reference,
)

__all__ = [
    "DEFAULT_VERIFY_ROW_CAP",
    "SCORER_POLICY_VERSION",
    "GoldExecutionError",
    "MatchReason",
    "MatchResult",
    "VerifierConfig",
    "execution_match",
    "validate_reference",
]
