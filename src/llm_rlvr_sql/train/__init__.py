"""Training layer (Stage 1): self-distilled trace dataset + LoRA SFT cold-start."""

from llm_rlvr_sql.train.sft import (
    AdapterProvenance,
    SftHyperparams,
    build_adapter_provenance,
    build_lora_config,
    build_sft_config,
    lora_kwargs,
    run_sft,
    sft_config_kwargs,
)
from llm_rlvr_sql.train.traces import (
    CandidateSet,
    SftTrace,
    TraceCandidates,
    TraceCounts,
    TraceDatasetManifest,
    TraceSamplingConfig,
    build_trace_manifest,
    filter_traces,
    read_candidate_set,
    read_trace_dataset,
    sample_candidates,
    write_trace_dataset,
)

__all__ = [
    "AdapterProvenance",
    "CandidateSet",
    "SftHyperparams",
    "SftTrace",
    "TraceCandidates",
    "TraceCounts",
    "TraceDatasetManifest",
    "TraceSamplingConfig",
    "build_adapter_provenance",
    "build_lora_config",
    "build_sft_config",
    "build_trace_manifest",
    "filter_traces",
    "lora_kwargs",
    "read_candidate_set",
    "read_trace_dataset",
    "run_sft",
    "sample_candidates",
    "sft_config_kwargs",
    "write_trace_dataset",
]
