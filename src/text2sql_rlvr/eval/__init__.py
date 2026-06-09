"""Evaluation layer: execution-match verifier, generation, scoring, provenance."""

from text2sql_rlvr.eval.completions import (
    CompletionSet,
    GenerationSample,
    ProblemCompletions,
    assemble_completion_set,
    read_completion_set,
    write_completion_set,
)
from text2sql_rlvr.eval.parsing import extract_sql
from text2sql_rlvr.eval.prompt import FewShotExemplar, build_messages, select_exemplars
from text2sql_rlvr.eval.provenance import (
    DecodingConfig,
    GenerationProvenance,
    build_generation_provenance,
)
from text2sql_rlvr.eval.score import ScoreReport, score_completion_set
from text2sql_rlvr.eval.verifier import VerifierConfig, execution_match

__all__ = [
    "CompletionSet",
    "DecodingConfig",
    "FewShotExemplar",
    "GenerationProvenance",
    "GenerationSample",
    "ProblemCompletions",
    "ScoreReport",
    "VerifierConfig",
    "assemble_completion_set",
    "build_generation_provenance",
    "build_messages",
    "execution_match",
    "extract_sql",
    "read_completion_set",
    "score_completion_set",
    "select_exemplars",
    "write_completion_set",
]
