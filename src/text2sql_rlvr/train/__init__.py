"""Training layer (Stage 1): self-distilled trace dataset + LoRA SFT cold-start."""

from text2sql_rlvr.train.sft import (
    AdapterProvenance,
    SftHyperparams,
    build_adapter_provenance,
    build_lora_config,
    build_sft_config,
    lora_kwargs,
    run_sft,
    sft_config_kwargs,
)
from text2sql_rlvr.train.traces import (
    SftTrace,
    TraceCandidates,
    TraceCounts,
    TraceDatasetManifest,
    TraceSamplingConfig,
    build_trace_manifest,
    filter_traces,
    read_trace_dataset,
    sample_candidates,
    write_trace_dataset,
)

__all__ = [
    "AdapterProvenance",
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
    "read_trace_dataset",
    "run_sft",
    "sample_candidates",
    "sft_config_kwargs",
    "write_trace_dataset",
]
