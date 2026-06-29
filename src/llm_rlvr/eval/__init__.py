"""Evaluation layer: execution-match verifier, generation, scoring, provenance."""

from llm_rlvr.eval.aggregate import SeedLevelStat, rl_vs_sft_gain, seed_level_ci
from llm_rlvr.eval.completions import (
    CompletionSet,
    GenerationSample,
    ProblemCompletions,
    assemble_completion_set,
    read_completion_set,
    write_completion_set,
)
from llm_rlvr.eval.parsing import extract_sql, well_formed
from llm_rlvr.eval.prompt import FewShotExemplar, build_messages, select_exemplars
from llm_rlvr.eval.provenance import (
    DecodingConfig,
    GenerationProvenance,
    build_generation_provenance,
)
from llm_rlvr.eval.score import ScoreReport, score_completion_set
from llm_rlvr.eval.verifier import VerifierConfig, execution_match

__all__ = [
    "CompletionSet",
    "DecodingConfig",
    "FewShotExemplar",
    "GenerationProvenance",
    "GenerationSample",
    "ProblemCompletions",
    "ScoreReport",
    "SeedLevelStat",
    "VerifierConfig",
    "assemble_completion_set",
    "build_generation_provenance",
    "build_messages",
    "execution_match",
    "extract_sql",
    "read_completion_set",
    "rl_vs_sft_gain",
    "score_completion_set",
    "seed_level_ci",
    "select_exemplars",
    "well_formed",
    "write_completion_set",
]
