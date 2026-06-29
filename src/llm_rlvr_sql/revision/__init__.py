"""Stage-3b forced-revision support (aborted pilot; kept tested for future attempts)."""

from llm_rlvr_sql.revision.episode import ForcedRevisionEpisode, ForcedRevisionTranscriptSet
from llm_rlvr_sql.revision.feedback import (
    REVISE_INSTRUCTION,
    UNPARSEABLE_FEEDBACK,
    attempt_feedback,
    is_error_feedback,
    revision_user_message,
)
from llm_rlvr_sql.revision.rollout import (
    build_db_id_lookup,
    build_suffix_ids,
    make_forced_revision_reward_fn,
    make_forced_revision_rollout,
    splice_episode,
)
from llm_rlvr_sql.revision.smoke import run_forced_revision_smoke
from llm_rlvr_sql.revision.train import run_forced_revision_train

__all__ = [
    "REVISE_INSTRUCTION",
    "UNPARSEABLE_FEEDBACK",
    "ForcedRevisionEpisode",
    "ForcedRevisionTranscriptSet",
    "attempt_feedback",
    "build_db_id_lookup",
    "build_suffix_ids",
    "is_error_feedback",
    "make_forced_revision_reward_fn",
    "make_forced_revision_rollout",
    "revision_user_message",
    "run_forced_revision_smoke",
    "run_forced_revision_train",
    "splice_episode",
]
