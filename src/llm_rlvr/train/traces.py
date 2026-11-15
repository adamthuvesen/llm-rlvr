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

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from llm_rlvr._base import Frozen
from llm_rlvr.artifacts.provenance import DecodingConfig, git_commit, git_is_dirty
from llm_rlvr.data.models import DatabaseSchema, Example
from llm_rlvr.data.paths import resolve_database
from llm_rlvr.data.splits import assert_train_split
from llm_rlvr.harness.messages import ChatMessage as TraceMessage
from llm_rlvr.harness.messages import validate_prompt as _validate_prompt
from llm_rlvr.tasks.sql.parsing import extract_sql, well_formed
from llm_rlvr.tasks.sql.prompt import build_prompts
from llm_rlvr.tasks.sql.verifier import GoldExecutionError, VerifierConfig, execution_match


class TraceSamplingConfig(DecodingConfig):
    """Trace-generation decoding: small-N sampling at modest temperature."""

    n: int = Field(default=8, gt=0)
    temperature: float = 0.8
    top_p: float = 0.95


class TraceCandidates(Frozen):
    """One train question with its validated prompt and raw completions."""

    example: Example
    prompt: tuple[TraceMessage, ...]
    texts: tuple[str, ...] = Field(min_length=1)

    _prompt = field_validator("prompt")(_validate_prompt)

    @model_validator(mode="after")
    def train_only(self) -> Self:
        assert_train_split([self.example], context="trace candidates")
        return self


class CandidateSet(Frozen):
    """Sampled candidates with consistent dataset and sampling metadata."""

    base_model: str
    base_model_revision: str | None = None
    sampling: TraceSamplingConfig
    dataset: str
    dataset_revision: str
    db_archive_sha256: str | None = None
    n_questions: int | None = Field(default=None, ge=0)
    n_candidates: int | None = Field(default=None, ge=0)
    items: tuple[TraceCandidates, ...]

    @model_validator(mode="after")
    def consistent_sampling(self) -> Self:
        for item in self.items:
            if item.example.dataset != self.dataset:
                raise ValueError("candidate dataset differs from dataset metadata")
            if len(item.texts) != self.sampling.n:
                raise ValueError("candidate completion count differs from sampling.n")
        if self.n_questions is None:
            object.__setattr__(self, "n_questions", len(self.items))
        if self.n_candidates is None:
            object.__setattr__(self, "n_candidates", sum(len(item.texts) for item in self.items))
        if self.n_questions != len(self.items):
            raise ValueError("n_questions differs from candidate items")
        if self.n_candidates is not None and self.n_candidates != sum(
            len(item.texts) for item in self.items
        ):
            raise ValueError("n_candidates differs from candidate texts")
        return self


class SftTrace(Frozen):
    """One selected completion with optional source lineage for legacy files."""

    db_id: str
    question: str
    prompt: tuple[TraceMessage, ...]
    completion: str = Field(min_length=1)
    source_example: Example | None = None

    _prompt = field_validator("prompt")(_validate_prompt)

    @model_validator(mode="after")
    def valid_source(self) -> Self:
        _validate_source(self.source_example, self.db_id, self.question)
        if not self.completion.strip():
            raise ValueError("completion must not be empty")
        return self


def _validate_source(example: Example | None, db_id: str, question: str) -> None:
    if example is not None:
        assert_train_split([example], context="trace source")
        if example.db_id != db_id or example.question != question:
            raise ValueError("trace source differs from row database or question")


class TraceDatasetRow(Frozen):
    prompt: tuple[TraceMessage, ...]
    completion: tuple[TraceMessage, ...] = Field(min_length=1, max_length=1)
    db_id: str
    question: str
    source_example: Example | None = None

    _prompt = field_validator("prompt")(_validate_prompt)

    @model_validator(mode="after")
    def valid_completion_and_source(self) -> Self:
        if self.completion[0].role != "assistant":
            raise ValueError("completion role must be assistant")
        _validate_source(self.source_example, self.db_id, self.question)
        return self


class TraceCounts(Frozen):
    """Filter-funnel counts: questions in, candidates sampled, eligible after filtering, kept."""

    n_questions: int = Field(ge=0)
    n_candidates: int = Field(ge=0)
    # well-formed AND execution-correct AND deduped — the pool the per-question selection sees
    n_eligible: int = Field(ge=0)
    n_kept: int = Field(ge=0)
    n_gold_errors: int = Field(ge=0)
    gold_error_examples: tuple[Example, ...] | None = None

    @model_validator(mode="after")
    def valid_funnel(self) -> Self:
        if not self.n_kept <= self.n_eligible <= self.n_candidates:
            raise ValueError("invalid trace filter candidate counts")
        if self.gold_error_examples is not None:
            assert_train_split(self.gold_error_examples, context="trace gold errors")
            if len(self.gold_error_examples) != self.n_gold_errors:
                raise ValueError("gold-error example count differs from n_gold_errors")
        if self.n_kept + self.n_gold_errors > self.n_questions:
            raise ValueError("invalid trace filter question counts")
        return self


class TraceDatasetManifest(Frozen):
    """Provenance for a trace dataset: enough to reproduce and audit the build."""

    base_model: str
    base_model_revision: str | None = None
    sampling: TraceSamplingConfig
    dataset: str
    dataset_revision: str
    db_archive_sha256: str | None = None
    source_examples: tuple[Example, ...] | None = None
    trace_dataset_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    selection: Literal["shortest"] = "shortest"
    counts: TraceCounts
    git_commit: str = "unknown"
    git_dirty: bool = False

    @model_validator(mode="after")
    def valid_lineage(self) -> Self:
        if self.source_examples is not None:
            assert_train_split(self.source_examples, context="trace manifest")
            if len(self.source_examples) != self.counts.n_questions:
                raise ValueError("manifest source count differs from filter question count")
            if any(example.dataset != self.dataset for example in self.source_examples):
                raise ValueError("manifest source dataset differs from metadata")
        return self


def sample_candidates(
    examples: Sequence[Example],
    schemas: Mapping[str, DatabaseSchema],
    *,
    model: str,
    model_revision: str | None = None,
    sampling: TraceSamplingConfig,
) -> tuple[TraceCandidates, ...]:
    """Sample N `<think>`-prompt completions per train question (vLLM; GPU).

    Raises loudly if any example is not from the train split — the cold-start data must never
    be built from dev/test (the R2 no-leakage invariant, applied to training data).
    """
    assert_train_split(examples, context="trace dataset")
    prompts = build_prompts(examples, schemas, style="cot")
    sampled = _sample_vllm(prompts, model=model, model_revision=model_revision, sampling=sampling)
    if any(len(texts) != sampling.n for texts in sampled):
        raise ValueError("sampled completion count differs from sampling.n")
    return tuple(
        TraceCandidates(
            example=example,
            prompt=tuple(TraceMessage.model_validate(message) for message in prompt),
            texts=tuple(texts),
        )
        for example, prompt, texts in zip(examples, prompts, sampled, strict=True)
    )


def _sample_vllm(
    prompts: Sequence[list[dict[str, str]]],
    *,
    model: str,
    model_revision: str | None = None,
    sampling: TraceSamplingConfig,
) -> list[list[str]]:
    """N completions per prompt (unlike the baseline backend's single greedy output)."""
    from llm_rlvr.generation.vllm_backend import make_vllm_engine, vllm_sampling_params

    llm, chat_kwargs = make_vllm_engine(model, model_revision=model_revision)
    params = vllm_sampling_params(sampling, n=sampling.n)
    results = llm.chat([list(p) for p in prompts], params, **chat_kwargs)
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
    assert_train_split([item.example for item in candidates], context="trace filter")
    cfg = config or VerifierConfig(guard_trivial_empty=True)
    if not cfg.guard_trivial_empty:
        raise ValueError("trace filtering requires guard_trivial_empty=True")
    kept: list[SftTrace] = []
    n_candidates = 0
    n_eligible = 0
    gold_error_examples: list[Example] = []

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
            gold_error_examples.append(candidate.example)
            continue

        n_eligible += len(eligible)
        if eligible:
            kept.append(
                SftTrace(
                    db_id=candidate.example.db_id,
                    question=candidate.example.question,
                    source_example=candidate.example,
                    prompt=candidate.prompt,
                    completion=min(eligible, key=len),  # shortest; min is stable on ties
                )
            )

    counts = TraceCounts(
        n_questions=len(candidates),
        n_candidates=n_candidates,
        n_eligible=n_eligible,
        n_kept=len(kept),
        n_gold_errors=len(gold_error_examples),
        gold_error_examples=tuple(gold_error_examples),
    )
    return tuple(kept), counts


def read_candidate_set(path: str | Path) -> CandidateSet:
    from llm_rlvr.artifacts.runs import require_complete

    require_complete(path)
    return CandidateSet.model_validate_json(Path(path).read_text(encoding="utf-8"))


def build_trace_manifest(
    candidates: CandidateSet,
    counts: TraceCounts,
    *,
    commit: str | None = None,
    dirty: bool | None = None,
) -> TraceDatasetManifest:
    """Assemble the manifest from the candidate set's sampling provenance + the filter counts."""
    if counts.n_questions != len(candidates.items) or counts.n_candidates != sum(
        len(item.texts) for item in candidates.items
    ):
        raise ValueError("filter counts differ from candidate set")
    return TraceDatasetManifest(
        base_model=candidates.base_model,
        base_model_revision=candidates.base_model_revision,
        sampling=candidates.sampling,
        dataset=candidates.dataset,
        dataset_revision=candidates.dataset_revision,
        db_archive_sha256=candidates.db_archive_sha256,
        counts=counts,
        source_examples=tuple(item.example for item in candidates.items),
        git_commit=commit if commit is not None else git_commit(),
        git_dirty=dirty if dirty is not None else git_is_dirty(),
    )


def write_trace_dataset(traces: Sequence[SftTrace], path: str | Path) -> Path:
    """Write validated TRL conversational prompt-completion JSONL."""
    rows = [
        TraceDatasetRow(
            prompt=trace.prompt,
            completion=(TraceMessage(role="assistant", content=trace.completion),),
            db_id=trace.db_id,
            question=trace.question,
            source_example=trace.source_example,
        )
        for trace in traces
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [row.model_dump_json(exclude_none=True) for row in rows]
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out


def read_trace_rows(
    path: str | Path, *, require_train_lineage: bool = False
) -> tuple[TraceDatasetRow, ...]:
    """Read typed conversational rows, optionally requiring explicit training lineage."""
    rows = tuple(
        TraceDatasetRow.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
    )
    if require_train_lineage and any(row.source_example is None for row in rows):
        raise ValueError("training requires source_example lineage for every trace row")
    return rows
