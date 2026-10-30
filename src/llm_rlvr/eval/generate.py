"""Generation orchestrator: prompts → model completions → a `CompletionSet`.

The only model-loading step. Backends are imported lazily so the package (and the whole
data/verifier/scoring layer) installs and runs without `torch`/`transformers`/`vllm`. The actual
model call (`_run_backend`) is exercised only by the real baseline run / integration tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NotRequired, Protocol, TypedDict, cast

if TYPE_CHECKING:
    import torch

from llm_rlvr.artifacts.provenance import DecodingConfig, build_generation_provenance
from llm_rlvr.data.models import DatabaseSchema, Example
from llm_rlvr.eval.completions import CompletionSet, assemble_completion_set
from llm_rlvr.eval.vllm_backend import make_vllm_engine, sample_vllm_completions
from llm_rlvr.tasks.sql.prompt import Messages as Messages
from llm_rlvr.tasks.sql.prompt import build_prompts as build_prompts
from llm_rlvr.tasks.sql.prompt import select_exemplars


class _GenerationModel(Protocol):
    @property
    def device(self) -> torch.device: ...

    def generate(
        self,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float = ...,
        top_p: float = ...,
        **model_inputs: torch.Tensor,
    ) -> torch.Tensor: ...


class _GenerationKwargs(TypedDict):
    max_new_tokens: int
    do_sample: bool
    temperature: NotRequired[float]
    top_p: NotRequired[float]


def generate(
    examples: Sequence[Example],
    schemas: Mapping[str, DatabaseSchema],
    *,
    model: str,
    backend: str,
    decoding: DecodingConfig,
    dataset: str,
    dataset_revision: str,
    split: str,
    model_revision: str | None = None,
    db_archive_sha256: str | None = None,
    train_pool: Sequence[Example] = (),
    few_shot_k: int = 0,
    few_shot_seed: int = 0,
    prompt_style: str = "cot",
    selection_policy: Literal["all", "prefix", "seeded"] | None = None,
    selection_seed: int | None = None,
    adapter_path: str | Path | None = None,
    adapter_sha256: str | None = None,
    adapter_provenance_sha256: str | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> CompletionSet:
    """Generate one completion per example and assemble a provenance-stamped `CompletionSet`.

    `adapter_path` loads a LoRA adapter over the base model (the SFT checkpoint); the base
    path with no adapter is byte-identical to before.
    """
    exemplars = select_exemplars(train_pool, schemas, k=few_shot_k, seed=few_shot_seed)
    prompts = build_prompts(examples, schemas, exemplars, style=prompt_style)
    raw_texts = _run_backend(
        prompts,
        model=model,
        model_revision=model_revision,
        backend=backend,
        decoding=decoding,
        adapter_path=adapter_path,
    )
    provenance = build_generation_provenance(
        model=model,
        backend=backend,
        dataset=dataset,
        dataset_revision=dataset_revision,
        split=split,
        n_examples=len(examples),
        decoding=decoding,
        model_revision=model_revision,
        db_archive_sha256=db_archive_sha256,
        few_shot_k=few_shot_k,
        few_shot_seed=few_shot_seed if few_shot_k else None,
        few_shot_exemplars=tuple((e.db_id, e.question) for e in exemplars),
        prompt_style=prompt_style,
        selection_policy=selection_policy,
        selection_seed=selection_seed,
        adapter=str(adapter_path) if adapter_path is not None else None,
        adapter_sha256=adapter_sha256,
        adapter_provenance_sha256=adapter_provenance_sha256,
        commit=commit,
        dirty=dirty,
    )
    return assemble_completion_set(examples, raw_texts, provenance)


def _run_backend(
    prompts: Sequence[Messages],
    *,
    model: str,
    backend: str,
    decoding: DecodingConfig,
    model_revision: str | None = None,
    adapter_path: str | Path | None = None,
) -> list[str]:
    """Dispatch to a generation backend (lazy imports). Returns one raw completion per prompt."""
    if backend == "transformers":
        return _run_transformers(
            prompts,
            model=model,
            model_revision=model_revision,
            decoding=decoding,
            adapter_path=adapter_path,
        )
    if backend == "vllm":
        return _run_vllm(
            prompts,
            model=model,
            model_revision=model_revision,
            decoding=decoding,
            adapter_path=adapter_path,
        )
    raise ValueError(f"unknown backend {backend!r}; expected 'transformers' or 'vllm'")


def _run_transformers(
    prompts: Sequence[Messages],
    *,
    model: str,
    model_revision: str | None = None,
    decoding: DecodingConfig,
    adapter_path: str | Path | None = None,
) -> list[str]:
    # Lazy imports: the GPU/CPU model stack is only needed at generation time.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding, set_seed

    set_seed(decoding.seed)  # match the vllm path: the recorded seed must actually apply
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    revision_kwargs = {"revision": model_revision} if model_revision else {}
    tokenizer = AutoTokenizer.from_pretrained(model, **revision_kwargs)
    lm: torch.nn.Module = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype="auto", **revision_kwargs
    )
    lm.to(device=device)
    if adapter_path is not None:
        from peft import PeftModel  # lazy import: train extra only

        lm = PeftModel.from_pretrained(lm, str(adapter_path))
    lm.eval()
    generation_model = cast(_GenerationModel, lm)

    sampling = decoding.temperature > 0
    gen_kwargs: _GenerationKwargs = {
        "max_new_tokens": decoding.max_new_tokens,
        "do_sample": sampling,
    }
    if sampling:  # only pass sampling params when sampling (greedy + top_p warns and is ignored)
        gen_kwargs["temperature"] = decoding.temperature
        gen_kwargs["top_p"] = decoding.top_p

    outputs: list[str] = []
    for messages in prompts:
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        if not isinstance(inputs, BatchEncoding):
            raise ValueError("tokenizer must return BatchEncoding when return_dict=True")
        inputs = inputs.to(generation_model.device)
        with torch.no_grad():
            generated = generation_model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        text = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)
        if not isinstance(text, str):
            raise ValueError("tokenizer.decode must return one string per completion")
        outputs.append(text)
    return outputs


def _run_vllm(
    prompts: Sequence[Messages],
    *,
    model: str,
    model_revision: str | None = None,
    decoding: DecodingConfig,
    adapter_path: str | Path | None = None,
) -> list[str]:
    llm, chat_kwargs = make_vllm_engine(
        model, model_revision=model_revision, adapter_path=adapter_path
    )
    return sample_vllm_completions(llm, prompts, decoding, chat_kwargs=chat_kwargs)
