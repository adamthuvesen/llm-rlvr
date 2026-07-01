"""Cold-start trace dataset: sample the base model, keep execution-correct `<think>`+SQL traces.

Sample N completions per Spider train question with the `<think>` prompt on a GPU, then filter on
CPU. Keep only well-formed candidates whose SQL execution-matches the reference query, one trace per
question: the shortest correct candidate after exact deduplication. The project scorer is the data
filter.

The dataset serializes as TRL conversational prompt-completion JSONL
(`{"prompt": [...messages], "completion": [assistant turn]}`), which is what gives us
completion-only loss in `SFTTrainer` without relying on the chat template's generation tags.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from llm_rlvr._base import Frozen
from llm_rlvr.data.jsonl import write_jsonl
from llm_rlvr.data.models import DatabaseSchema, Example
from llm_rlvr.data.paths import resolve_database
from llm_rlvr.data.splits import assert_train_split
from llm_rlvr.eval.generate import build_prompts
from llm_rlvr.eval.parsing import extract_sql, well_formed
from llm_rlvr.eval.provenance import DecodingConfig, git_commit, git_is_dirty
from llm_rlvr.eval.verifier import GoldExecutionError, VerifierConfig, execution_match


class TraceSamplingConfig(DecodingConfig):
    """Trace-generation decoding: small-N sampling at modest temperature."""

    n: int = 8
    temperature: float = 0.8
    top_p: float = 0.95


class TraceCandidates(Frozen):
    """One train question with the prompt actually used and its N raw completions."""

    example: Example
    prompt: tuple[dict[str, str], ...]
    texts: tuple[str, ...]


class CandidateSet(Frozen):
    """Sampled candidates plus the sampling provenance the CPU filter step needs.

    The GPU sample step writes this to the Volume; `filter_traces` + `build_trace_manifest`
    consume it locally — the same generate-on-GPU / score-on-CPU split as the baseline.
    """

    base_model: str
    base_model_revision: str | None = None
    sampling: TraceSamplingConfig
    dataset: str
    dataset_revision: str
    db_archive_sha256: str | None = None
    items: tuple[TraceCandidates, ...]


class SftTrace(Frozen):
    """One kept SFT record: the prompt messages and the selected `<think>`+SQL completion."""

    db_id: str
    question: str
    prompt: tuple[dict[str, str], ...]
    completion: str


class TraceCounts(Frozen):
    """Filter-funnel counts: questions in, candidates sampled, eligible after filtering, kept."""

    n_questions: int
    n_candidates: int
    # well-formed AND execution-correct AND deduped — the pool the per-question selection sees
    n_eligible: int
    n_kept: int
    n_gold_errors: int


class TraceDatasetManifest(Frozen):
    """Provenance for a trace dataset: enough to reproduce and audit the build."""

    base_model: str
    base_model_revision: str | None = None
    sampling: TraceSamplingConfig
    dataset: str
    dataset_revision: str
    db_archive_sha256: str | None = None
    selection: str = "shortest"
    counts: TraceCounts
    git_commit: str = "unknown"
    git_dirty: bool = False


def sample_candidates(
    examples: Sequence[Example],
    schemas: Mapping[str, DatabaseSchema],
    *,
    model: str,
    sampling: TraceSamplingConfig,
) -> tuple[TraceCandidates, ...]:
    """Sample N `<think>`-prompt completions per train question (vLLM; GPU).

    Raises loudly if any example is not from the train split — the cold-start data must never
    be built from dev/test (the R2 no-leakage invariant, applied to training data).
    """
    assert_train_split(examples, context="trace dataset")
    prompts = build_prompts(examples, schemas, style="cot")
    sampled = _sample_vllm(prompts, model=model, sampling=sampling)
    return tuple(
        TraceCandidates(example=example, prompt=tuple(prompt), texts=tuple(texts))
        for example, prompt, texts in zip(examples, prompts, sampled, strict=True)
    )


def _sample_vllm(
    prompts: Sequence[list[dict[str, str]]],
    *,
    model: str,
    sampling: TraceSamplingConfig,
) -> list[list[str]]:
    """N completions per prompt (unlike the baseline backend's single greedy output)."""
    from vllm import LLM, SamplingParams  # lazy import: GPU backend only

    llm = LLM(model=model)
    params = SamplingParams(
        n=sampling.n,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        max_tokens=sampling.max_new_tokens,
        seed=sampling.seed,
    )
    results = llm.chat([list(p) for p in prompts], params)
    return [[output.text for output in result.outputs] for result in results]


def filter_traces(
    candidates: Sequence[TraceCandidates],
    db_root: str | Path,
    *,
    config: VerifierConfig | None = None,
) -> tuple[tuple[SftTrace, ...], TraceCounts]:
    """Keep one execution-correct, well-formed trace per question (shortest; exact-deduped).

    The default verifier config turns the trivial-empty guard ON: this is training data, so an
    empty-result "match" must not qualify a trace. A question whose reference query fails
    contributes nothing and is counted in `n_gold_errors`.
    """
    cfg = config or VerifierConfig(guard_trivial_empty=True)
    kept: list[SftTrace] = []
    n_candidates = 0
    n_eligible = 0
    n_gold_errors = 0

    for candidate in candidates:
        n_candidates += len(candidate.texts)
        db_path = resolve_database(candidate.example.db_id, db_root)
        eligible: list[str] = []
        seen: set[str] = set()
        try:
            for text in candidate.texts:
                if text in seen or not well_formed(text):
                    continue
                seen.add(text)
                sql = extract_sql(text)
                if sql is None:
                    continue
                result = execution_match(sql, candidate.example.gold_sql, db_path, config=cfg)
                if result.matched:
                    eligible.append(text)
        except GoldExecutionError:
            n_gold_errors += 1
            continue

        n_eligible += len(eligible)
        if eligible:
            kept.append(
                SftTrace(
                    db_id=candidate.example.db_id,
                    question=candidate.example.question,
                    prompt=candidate.prompt,
                    completion=min(eligible, key=len),  # shortest; min is stable on ties
                )
            )

    counts = TraceCounts(
        n_questions=len(candidates),
        n_candidates=n_candidates,
        n_eligible=n_eligible,
        n_kept=len(kept),
        n_gold_errors=n_gold_errors,
    )
    return tuple(kept), counts


def read_candidate_set(path: str | Path) -> CandidateSet:
    return CandidateSet.model_validate_json(Path(path).read_text(encoding="utf-8"))


def build_trace_manifest(
    candidates: CandidateSet,
    counts: TraceCounts,
    *,
    commit: str | None = None,
    dirty: bool | None = None,
) -> TraceDatasetManifest:
    """Assemble the manifest from the candidate set's sampling provenance + the filter counts."""
    return TraceDatasetManifest(
        base_model=candidates.base_model,
        base_model_revision=candidates.base_model_revision,
        sampling=candidates.sampling,
        dataset=candidates.dataset,
        dataset_revision=candidates.dataset_revision,
        db_archive_sha256=candidates.db_archive_sha256,
        counts=counts,
        git_commit=commit if commit is not None else git_commit(),
        git_dirty=dirty if dirty is not None else git_is_dirty(),
    )


def write_trace_dataset(traces: Sequence[SftTrace], path: str | Path) -> Path:
    """Write TRL conversational prompt-completion JSONL (one record per line)."""
    records = [
        {
            "prompt": list(trace.prompt),
            "completion": [{"role": "assistant", "content": trace.completion}],
            "db_id": trace.db_id,
            "question": trace.question,
        }
        for trace in traces
    ]
    return write_jsonl(records, path)


def read_trace_dataset(path: str | Path) -> tuple[SftTrace, ...]:
    """Read a trace-dataset JSONL back into typed records (strict; loud on malformed lines)."""
    traces = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        (completion,) = record["completion"]  # exactly one assistant turn
        traces.append(
            SftTrace(
                db_id=record["db_id"],
                question=record["question"],
                prompt=tuple(record["prompt"]),
                completion=completion["content"],
            )
        )
    return tuple(traces)
