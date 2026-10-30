"""Training entrypoints with lazy compatibility exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_rlvr.train.sft import (
        AdapterProvenance as AdapterProvenance,
    )
    from llm_rlvr.train.sft import (
        SftHyperparams as SftHyperparams,
    )
    from llm_rlvr.train.sft import (
        build_adapter_provenance as build_adapter_provenance,
    )
    from llm_rlvr.train.sft import (
        build_lora_config as build_lora_config,
    )
    from llm_rlvr.train.sft import (
        build_sft_config as build_sft_config,
    )
    from llm_rlvr.train.sft import (
        lora_kwargs as lora_kwargs,
    )
    from llm_rlvr.train.sft import (
        run_sft as run_sft,
    )
    from llm_rlvr.train.sft import (
        sft_config_kwargs as sft_config_kwargs,
    )
    from llm_rlvr.train.traces import (
        CandidateSet as CandidateSet,
    )
    from llm_rlvr.train.traces import (
        SftTrace as SftTrace,
    )
    from llm_rlvr.train.traces import (
        TraceCandidates as TraceCandidates,
    )
    from llm_rlvr.train.traces import (
        TraceCounts as TraceCounts,
    )
    from llm_rlvr.train.traces import (
        TraceDatasetManifest as TraceDatasetManifest,
    )
    from llm_rlvr.train.traces import (
        TraceSamplingConfig as TraceSamplingConfig,
    )
    from llm_rlvr.train.traces import (
        build_trace_manifest as build_trace_manifest,
    )
    from llm_rlvr.train.traces import (
        filter_traces as filter_traces,
    )
    from llm_rlvr.train.traces import (
        read_candidate_set as read_candidate_set,
    )
    from llm_rlvr.train.traces import (
        read_trace_dataset as read_trace_dataset,
    )
    from llm_rlvr.train.traces import (
        sample_candidates as sample_candidates,
    )
    from llm_rlvr.train.traces import (
        write_trace_dataset as write_trace_dataset,
    )

_EXPORT_MODULES = {
    "AdapterProvenance": "llm_rlvr.train.sft",
    "SftHyperparams": "llm_rlvr.train.sft",
    "build_adapter_provenance": "llm_rlvr.train.sft",
    "build_lora_config": "llm_rlvr.train.sft",
    "build_sft_config": "llm_rlvr.train.sft",
    "lora_kwargs": "llm_rlvr.train.sft",
    "run_sft": "llm_rlvr.train.sft",
    "sft_config_kwargs": "llm_rlvr.train.sft",
    "CandidateSet": "llm_rlvr.train.traces",
    "SftTrace": "llm_rlvr.train.traces",
    "TraceCandidates": "llm_rlvr.train.traces",
    "TraceCounts": "llm_rlvr.train.traces",
    "TraceDatasetManifest": "llm_rlvr.train.traces",
    "TraceSamplingConfig": "llm_rlvr.train.traces",
    "build_trace_manifest": "llm_rlvr.train.traces",
    "filter_traces": "llm_rlvr.train.traces",
    "read_candidate_set": "llm_rlvr.train.traces",
    "read_trace_dataset": "llm_rlvr.train.traces",
    "sample_candidates": "llm_rlvr.train.traces",
    "write_trace_dataset": "llm_rlvr.train.traces",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> object:
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORT_MODULES[name]), name)
    globals()[name] = value
    return value
