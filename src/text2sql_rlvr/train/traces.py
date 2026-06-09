"""Cold-start trace dataset: sample the base model, keep execution-correct `<think>`+SQL traces.

The Stage-1 self-distillation pipeline (design D1/D2): sample N completions per Spider-*train*
question with the `<think>` prompt on a GPU (`sample_candidates`, vLLM), then filter on CPU
(`filter_traces`) — keep only candidates that are well-formed AND whose SQL execution-matches
gold under the Stage-0 verifier, one trace per question (the shortest correct one, exact
duplicates removed). The verifier that scores the project is the data filter.

The dataset serializes as TRL conversational prompt-completion JSONL
(`{"prompt": [...messages], "completion": [assistant turn]}`), which is what gives us
completion-only loss in `SFTTrainer` without relying on the chat template's generation tags.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from text2sql_rlvr._base import Frozen
from text2sql_rlvr.data.models import DatabaseSchema, Example
from text2sql_rlvr.data.spider import resolve_database
from text2sql_rlvr.eval.generate import build_prompts
from text2sql_rlvr.eval.parsing import extract_sql, well_formed
from text2sql_rlvr.eval.provenance import DecodingConfig, git_commit, git_is_dirty
from text2sql_rlvr.eval.verifier import GoldExecutionError, VerifierConfig, execution_match


class TraceSamplingConfig(DecodingConfig):
    """Trace-generation decoding (design D2): small-N sampling at modest temperature."""

    n: int = 8
    temperature: float = 0.8
    top_p: float = 0.95


class TraceCandidates(Frozen):
    """One train question with the prompt actually used and its N raw completions."""

    example: Example
    prompt: tuple[dict[str, str], ...]
    texts: tuple[str, ...]


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
    leaked = sorted({example.split for example in examples if example.split != "train"})
    if leaked:
        raise ValueError(
            f"trace dataset must be built from the train split only; got splits {leaked}"
        )
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
    empty-result "match" must not qualify a trace (the same guard the Stage-2 reward will use).
    A question whose gold query fails contributes nothing and is counted in `n_gold_errors`.
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


def build_trace_manifest(
    *,
    base_model: str,
    sampling: TraceSamplingConfig,
    dataset: str,
    dataset_revision: str,
    counts: TraceCounts,
    base_model_revision: str | None = None,
    db_archive_sha256: str | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> TraceDatasetManifest:
    """Assemble the manifest, capturing git state unless `commit`/`dirty` are passed in."""
    return TraceDatasetManifest(
        base_model=base_model,
        base_model_revision=base_model_revision,
        sampling=sampling,
        dataset=dataset,
        dataset_revision=dataset_revision,
        db_archive_sha256=db_archive_sha256,
        counts=counts,
        git_commit=commit if commit is not None else git_commit(),
        git_dirty=dirty if dirty is not None else git_is_dirty(),
    )


def write_trace_dataset(traces: Sequence[SftTrace], path: str | Path) -> Path:
    """Write TRL conversational prompt-completion JSONL (one record per line)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "prompt": list(trace.prompt),
                "completion": [{"role": "assistant", "content": trace.completion}],
                "db_id": trace.db_id,
                "question": trace.question,
            },
            ensure_ascii=False,
        )
        for trace in traces
    ]
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out


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
