"""Generation orchestrator: prompts → model completions → a `CompletionSet`.

The only model-loading step. Backends are imported lazily so the package (and the whole
data/verifier/scoring layer) installs and runs without `torch`/`transformers`/`vllm`. The actual
model call (`_run_backend`) is exercised only by the real baseline run / integration tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from llm_rlvr.data.models import DatabaseSchema, Example
from llm_rlvr.data.schema import serialize_schema
from llm_rlvr.eval.completions import CompletionSet
from llm_rlvr.eval.prompt import FewShotExemplar, build_messages, select_exemplars
from llm_rlvr.eval.provenance import DecodingConfig
from llm_rlvr.eval.runner import finish_eval_run
from llm_rlvr.eval.vllm_backend import make_vllm_engine, sample_vllm_completions

# Values are `object`, not `str`: the `mt_demo` style carries structured `tool_calls` entries.
Messages = list[dict[str, object]]


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
    adapter_path: str | Path | None = None,
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
    return finish_eval_run(
        examples,
        raw_texts,
        model=model,
        backend=backend,
        dataset=dataset,
        dataset_revision=dataset_revision,
        split=split,
        decoding=decoding,
        model_revision=model_revision,
        db_archive_sha256=db_archive_sha256,
        few_shot_k=few_shot_k,
        few_shot_seed=few_shot_seed if few_shot_k else None,
        few_shot_exemplars=tuple((e.db_id, e.question) for e in exemplars),
        prompt_style=prompt_style,
        adapter_path=adapter_path,
        commit=commit,
        dirty=dirty,
    )


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
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

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
    lm = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", **revision_kwargs).to(
        device
    )
    if adapter_path is not None:
        from peft import PeftModel  # lazy import: train extra only

        lm = PeftModel.from_pretrained(lm, str(adapter_path))
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
