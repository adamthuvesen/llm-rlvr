"""Generation orchestrator: prompts → model completions → a `CompletionSet`.

The only model-loading step. Backends are imported lazily so the package (and the whole
data/verifier/scoring layer) installs and runs without `torch`/`transformers`/`vllm`. The actual
model call (`_run_backend`) is exercised only by the real baseline run / integration tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from text2sql_rlvr.data.models import DatabaseSchema, Example
from text2sql_rlvr.data.schema import serialize_schema
from text2sql_rlvr.eval.completions import CompletionSet, assemble_completion_set
from text2sql_rlvr.eval.prompt import FewShotExemplar, build_messages, select_exemplars
from text2sql_rlvr.eval.provenance import DecodingConfig, build_generation_provenance

Messages = list[dict[str, str]]


def build_prompts(
    examples: Sequence[Example],
    schemas: Mapping[str, DatabaseSchema],
    exemplars: Sequence[FewShotExemplar] = (),
    style: str = "cot",
) -> list[Messages]:
    """Render chat messages for every example with the shared few-shot exemplars. Pure/testable."""
    return [
        build_messages(
            serialize_schema(schemas[example.db_id]),
            example.question,
            exemplars,
            evidence=example.evidence,
            style=style,
        )
        for example in examples
    ]


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
    commit: str | None = None,
    dirty: bool | None = None,
) -> CompletionSet:
    """Generate one completion per example and assemble a provenance-stamped `CompletionSet`."""
    exemplars = select_exemplars(train_pool, schemas, k=few_shot_k, seed=few_shot_seed)
    prompts = build_prompts(examples, schemas, exemplars, style=prompt_style)
    raw_texts = _run_backend(prompts, model=model, backend=backend, decoding=decoding)
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
) -> list[str]:
    """Dispatch to a generation backend (lazy imports). Returns one raw completion per prompt."""
    if backend == "transformers":
        return _run_transformers(prompts, model=model, decoding=decoding)
    if backend == "vllm":
        return _run_vllm(prompts, model=model, decoding=decoding)
    raise ValueError(f"unknown backend {backend!r}; expected 'transformers' or 'vllm'")


def _run_transformers(
    prompts: Sequence[Messages], *, model: str, decoding: DecodingConfig
) -> list[str]:
    # Lazy imports: the GPU/CPU model stack is only needed at generation time.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(decoding.seed)  # match the vllm path: the recorded seed must actually apply
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(model)
    lm = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto").to(device)
    lm.eval()

    sampling = decoding.temperature > 0
    gen_kwargs: dict[str, object] = {
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
        ).to(lm.device)
        with torch.no_grad():
            generated = lm.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        outputs.append(tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True))
    return outputs


def _run_vllm(prompts: Sequence[Messages], *, model: str, decoding: DecodingConfig) -> list[str]:
    from vllm import LLM, SamplingParams  # lazy import: GPU backend only

    llm = LLM(model=model)
    params = SamplingParams(
        temperature=decoding.temperature,
        top_p=decoding.top_p,
        max_tokens=decoding.max_new_tokens,
        seed=decoding.seed,
    )
    results = llm.chat(list(prompts), params)
    return [result.outputs[0].text for result in results]
