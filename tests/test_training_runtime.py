"""Training drivers validate inputs before loading models and persist effective settings."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import make_dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_rlvr.data.models import Column, DatabaseSchema, Example, Table
from llm_rlvr.train import configuration
from llm_rlvr.train.configuration import inspect_adapter
from llm_rlvr.train.grpo import GrpoHyperparams, GrpoProvenance, grpo_config_kwargs, run_grpo
from llm_rlvr.train.sft import AdapterProvenance, SftHyperparams, run_sft, sft_config_kwargs


def _adapter(path: Path, *, base: str = "model", rank: int = 8) -> Path:
    path.mkdir(exist_ok=True)
    (path / "adapter_config.json").write_text(
        json.dumps(
            {"base_model_name_or_path": base, "r": rank, "lora_alpha": 16, "lora_dropout": 0.0}
        )
    )
    (path / "adapter_model.safetensors").write_bytes(b"fake weights")
    return path


def _trace_input(tmp_path: Path) -> tuple[Path, Path]:
    source = Example(
        db_id="concert", question="question", gold_sql="SELECT a FROM t", split="train"
    )
    traces = tmp_path / "traces.jsonl"
    traces.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "question"}],
                "completion": [{"role": "assistant", "content": "answer"}],
                "db_id": source.db_id,
                "question": source.question,
                "source_example": source.model_dump(),
            }
        )
        + "\n"
    )
    manifest = tmp_path / "traces.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "base_model": "model",
                "sampling": {},
                "dataset": "spider",
                "dataset_revision": "data-commit",
                "trace_dataset_sha256": hashlib.sha256(traces.read_bytes()).hexdigest(),
                "source_examples": [source.model_dump()],
                "counts": {
                    "n_questions": 1,
                    "n_candidates": 8,
                    "n_eligible": 1,
                    "n_kept": 1,
                    "n_gold_errors": 0,
                },
            }
        )
    )
    return traces, manifest


@pytest.fixture
def fake_training(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, object]]:
    events: list[tuple[str, object]] = []

    def config_type(name: str, values: dict[str, object]):
        return make_dataclass(name, [(key, object, None) for key in values])

    def save_adapter(path: str) -> None:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        trainer = next(value for name, value in events if name == "trainer")
        assert isinstance(trainer, dict)
        lora = trainer.get("peft_config")
        (output / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "model",
                    "r": lora.r if lora else 8,
                    "lora_alpha": lora.lora_alpha if lora else 16,
                    "lora_dropout": lora.lora_dropout if lora else 0.0,
                }
            )
        )
        (output / "adapter_model.safetensors").write_bytes(b"trained weights")
        events.append(("save", path))

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            events.append(("trainer", kwargs))

        def train(self) -> None:
            kwargs = self.kwargs
            if "reward_funcs" in kwargs:
                rows = kwargs["train_dataset"]
                columns = {key: [row[key] for row in rows] for key in rows[0] if key != "prompt"}
                completions = [
                    [
                        {
                            "role": "assistant",
                            "content": "<think>x</think>\n```sql\nSELECT a FROM t\n```",
                        }
                    ]
                    for _row in rows
                ]
                values = kwargs["reward_funcs"][0](completions=completions, **columns)
                events.append(("rewards", values))
            events.append(("train", None))

        save_model = staticmethod(save_adapter)

    def load_base(model: str, **kwargs):
        events.append(("base", {"model": model, **kwargs}))
        return "base-model"

    def load_adapter(base, path, **kwargs):
        events.append(("adapter", {"base": base, "path": path, **kwargs}))
        return "policy"

    monkeypatch.setitem(
        sys.modules,
        "trl",
        SimpleNamespace(
            GRPOConfig=config_type("GRPOConfig", grpo_config_kwargs(GrpoHyperparams(), "out")),
            SFTConfig=config_type(
                "SFTConfig",
                sft_config_kwargs(SftHyperparams(), "out") | {"model_init_kwargs": None},
            ),
            GRPOTrainer=FakeTrainer,
            SFTTrainer=FakeTrainer,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(Dataset=SimpleNamespace(from_list=list))
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            LoraConfig=SimpleNamespace,
            PeftModel=SimpleNamespace(from_pretrained=load_adapter),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_base)),
    )

    def resolve_revision(_model, revision):
        return revision if revision in {"old-commit", "requested-commit"} else "resolved-commit"

    monkeypatch.setattr(configuration, "resolve_model_revision", resolve_revision)
    monkeypatch.setattr("llm_rlvr.train.prepared.resolve_model_revision", resolve_revision)
    return events


def test_sft_runtime_uses_typed_rows_and_saves_effective_configuration(
    tmp_path, fake_training
) -> None:
    traces, manifest = _trace_input(tmp_path)
    output = run_sft(
        traces,
        tmp_path / "output",
        base_model="model",
        trace_manifest_path=manifest,
        commit="commit",
        dirty=False,
    )
    trainer = next(value for name, value in fake_training if name == "trainer")
    assert set(trainer["train_dataset"][0]) == {"prompt", "completion"}
    assert trainer["args"].model_init_kwargs["revision"] == "resolved-commit"
    assert trainer["args"].completion_only_loss is True
    saved = AdapterProvenance.model_validate_json((output / "provenance.json").read_text())
    assert saved.base_model_revision == "resolved-commit"
    assert saved.runtime_config is not None
    assert saved.train_source_ids is not None
    assert saved.runtime_config.effective_lora["r"] == 32
    assert saved.runtime_config.effective_trl["completion_only_loss"] is True
    assert len(saved.train_source_ids) == 1
    assert saved.trace_manifest_sha256
    assert ("save", str(output)) in fake_training


@pytest.mark.parametrize(
    "problem", ["empty", "legacy", "manifest", "gold_error", "count", "digest", "manifest_lineage"]
)
def test_sft_rejects_bad_training_inputs_before_loading(tmp_path, fake_training, problem) -> None:
    traces, manifest = _trace_input(tmp_path)
    if problem == "empty":
        traces.write_text("")
    elif problem == "legacy":
        row = json.loads(traces.read_text())
        row.pop("source_example")
        traces.write_text(json.dumps(row))
    elif problem in {"gold_error", "count", "digest", "manifest_lineage"}:
        payload = json.loads(manifest.read_text())
        if problem == "gold_error":
            payload["counts"].update(n_questions=2, n_gold_errors=1)
            payload["source_examples"] *= 2
        elif problem == "count":
            payload["counts"]["n_kept"] = 0
        elif problem == "digest":
            payload["trace_dataset_sha256"] = "0" * 64
        else:
            payload["source_examples"] = None
        manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        run_sft(
            traces,
            tmp_path / "output",
            base_model="model",
            trace_manifest_path=None if problem == "manifest" else manifest,
        )
    assert fake_training == []


def test_sft_rejects_trl_drift_before_trainer_load(tmp_path, fake_training, monkeypatch) -> None:
    traces, manifest = _trace_input(tmp_path)
    monkeypatch.setattr(sys.modules["trl"], "SFTConfig", make_dataclass("SFTConfig", []))
    with pytest.raises(ValueError, match="completion_only_loss"):
        run_sft(traces, tmp_path / "output", base_model="model", trace_manifest_path=manifest)
    assert fake_training == []


def test_grpo_runtime_uses_sql_reward_and_inherited_adapter(
    tmp_path, scoring_root, fake_training
) -> None:
    adapter = _adapter(tmp_path / "warm")
    examples = [
        Example(
            db_id="concert", question=f"question {index}", gold_sql="SELECT a FROM t", split="train"
        )
        for index in range(3)
    ]
    schema = DatabaseSchema(
        db_id="concert", tables=(Table(name="t", columns=(Column(name="a", type="int"),)),)
    )
    output = run_grpo(
        examples,
        {"concert": schema},
        scoring_root,
        tmp_path / "output",
        base_model="model",
        warm_start_adapter=adapter,
        dataset="spider",
        dataset_revision="data-commit",
        hyperparams=GrpoHyperparams(eval_slice_size=1, lora_rank=64),
        commit="commit",
        dirty=False,
    )
    rewards = next(value for name, value in fake_training if name == "rewards")
    assert len(rewards) == 2
    assert all(value > 1 for value in rewards)
    base_load = next(value for name, value in fake_training if name == "base")
    assert base_load["revision"] == "resolved-commit"
    provenance = GrpoProvenance.model_validate_json((output / "provenance.json").read_text())
    assert provenance.hyperparams.lora_rank == 64
    assert provenance.runtime_config is not None
    assert provenance.warm_start_identity is not None
    assert provenance.train_source_ids is not None
    assert provenance.dev_source_ids is not None
    assert provenance.runtime_config.requested_lora == {}
    assert provenance.runtime_config.effective_lora["r"] == 8
    assert provenance.warm_start_identity.sha256
    assert len(provenance.train_source_ids) == 2
    assert len(provenance.dev_source_ids) == 1
    assert not set(provenance.train_source_ids) & set(provenance.dev_source_ids)
    assert provenance.sft_dev_exposure == "unknown"


@pytest.mark.parametrize("problem", ["base", "revision", "weights", "empty", "trl"])
def test_prepared_grpo_fails_before_model_load(
    tmp_path, fake_training, monkeypatch, problem
) -> None:
    from llm_rlvr.train.prepared import execute_prepared_grpo

    adapter = _adapter(tmp_path / "warm", base="different" if problem == "base" else "model")
    if problem == "revision":
        (adapter / "provenance.json").write_text(
            json.dumps({"base_model": "model", "base_model_revision": "old-commit"})
        )
    if problem == "weights":
        (adapter / "adapter_model.safetensors").unlink()
    if problem == "trl":
        monkeypatch.setattr(sys.modules["trl"], "GRPOConfig", make_dataclass("GRPOConfig", []))
    with pytest.raises(ValueError):
        execute_prepared_grpo(
            [] if problem == "empty" else [{"prompt": "question"}],
            lambda **_kwargs: [1.0],
            tmp_path / "output",
            base_model="model",
            base_model_revision="requested-commit",
            warm_start_adapter=adapter,
            requested_config=grpo_config_kwargs(GrpoHyperparams(), tmp_path / "output"),
        )
    assert fake_training == []


def test_adapter_digest_changes_when_weights_change(tmp_path) -> None:
    adapter = _adapter(tmp_path / "warm")
    first = inspect_adapter(adapter, base_model="model", base_model_revision=None)
    (adapter / "adapter_model.safetensors").write_bytes(b"other weights")
    second = inspect_adapter(adapter, base_model="model", base_model_revision=None)
    assert first.sha256 != second.sha256


def test_sft_failure_leaves_managed_run_incomplete(tmp_path, fake_training, monkeypatch) -> None:
    from llm_rlvr.artifacts.runs import require_complete

    traces, manifest = _trace_input(tmp_path)
    output = tmp_path / "output"

    def fail_training(_self):
        raise RuntimeError("training failed")

    monkeypatch.setattr(sys.modules["trl"].SFTTrainer, "train", fail_training)
    with pytest.raises(RuntimeError, match="training failed"):
        run_sft(traces, output, base_model="model", trace_manifest_path=manifest)
    assert (output / "request.json").exists()
    assert not (output / "complete.json").exists()
    assert not (output / "provenance.json").exists()
    with pytest.raises(ValueError, match="incomplete"):
        require_complete(output)


def test_sft_start_callback_sees_request_before_trainer(tmp_path, fake_training) -> None:
    from llm_rlvr.artifacts.runs import require_complete

    traces, manifest = _trace_input(tmp_path)
    output = tmp_path / "output"

    def on_begin():
        assert (output / "request.json").is_file()
        assert not fake_training

    run_sft(traces, output, base_model="model", trace_manifest_path=manifest, on_begin=on_begin)
    require_complete(output)


def test_existing_training_output_rejected_before_loading(tmp_path, fake_training) -> None:
    traces, manifest = _trace_input(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError, match="output already exists"):
        run_sft(traces, output, base_model="model", trace_manifest_path=manifest)
    assert not fake_training


def test_incomplete_managed_adapter_rejected_before_loading(tmp_path, fake_training) -> None:
    from llm_rlvr.artifacts.runs import begin_run
    from llm_rlvr.train.prepared import execute_prepared_grpo

    adapter = tmp_path / "warm"
    begin_run(adapter, kind="sft", request=SftHyperparams())
    _adapter(adapter)
    with pytest.raises(ValueError, match="incomplete"):
        execute_prepared_grpo(
            [{"prompt": "question"}],
            lambda **_kwargs: [1.0],
            tmp_path / "output",
            base_model="model",
            base_model_revision=None,
            warm_start_adapter=adapter,
            requested_config=grpo_config_kwargs(GrpoHyperparams(), tmp_path / "output"),
        )
    assert not fake_training


@pytest.mark.parametrize("requested_revision", [None, "matching-tag"])
def test_grpo_resolves_default_adapter_revision_and_equivalent_explicit_tag(
    tmp_path, fake_training, monkeypatch, requested_revision
) -> None:
    from llm_rlvr.train.prepared import execute_prepared_grpo

    adapter = _adapter(tmp_path / "warm")
    (adapter / "provenance.json").write_text(
        json.dumps({"base_model": "model", "base_model_revision": "adapter-commit"})
    )
    calls = []

    def resolve(_model, revision):
        calls.append(revision)
        return "adapter-commit" if revision in {"adapter-commit", "matching-tag"} else "moving-main"

    monkeypatch.setattr(configuration, "resolve_model_revision", resolve)
    rewards = []

    def bound_reward(*, completions, expected):
        assert len(completions) == len(expected) == 1
        rewards.append(expected[0])
        return [float(expected[0])]

    execution = execute_prepared_grpo(
        [{"prompt": "question", "expected": 7}],
        bound_reward,
        tmp_path / "output",
        base_model="model",
        base_model_revision=requested_revision,
        warm_start_adapter=adapter,
        requested_config=grpo_config_kwargs(GrpoHyperparams(), tmp_path / "other"),
    )
    assert calls[0] == (requested_revision or "adapter-commit")
    assert execution.base_model_revision == "adapter-commit"
    assert rewards == [7]
    trainer = next(value for name, value in fake_training if name == "trainer")
    assert trainer["args"].output_dir == str(tmp_path / "output")
    assert execution.configuration.effective_trl["output_dir"] == str(tmp_path / "output")
    assert not (tmp_path / "other").exists()


def test_sql_grpo_gold_failure_blocks_model_loading(tmp_path, scoring_root, fake_training) -> None:
    source = Example(
        db_id="concert", question="question", gold_sql="SELECT bad FROM t", split="train"
    )
    schema = DatabaseSchema(
        db_id="concert", tables=(Table(name="t", columns=(Column(name="a", type="int"),)),)
    )
    with pytest.raises(ValueError, match="SQL GRPO reference failed for source"):
        run_grpo(
            [source],
            {"concert": schema},
            scoring_root,
            tmp_path / "output",
            base_model="model",
            warm_start_adapter=_adapter(tmp_path / "warm"),
            dataset="spider",
            dataset_revision="revision",
        )
    assert not fake_training


def test_adapter_provenance_object_required(tmp_path) -> None:
    adapter = _adapter(tmp_path / "warm")
    (adapter / "provenance.json").write_text("[]")
    with pytest.raises(ValueError):
        inspect_adapter(adapter, base_model="model", base_model_revision=None)


def test_adapter_weight_index_requires_all_shards(tmp_path) -> None:
    adapter = _adapter(tmp_path / "warm")
    (adapter / "adapter_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "adapter_model-00002.safetensors"}})
    )
    with pytest.raises(ValueError, match="missing weights"):
        inspect_adapter(adapter, base_model="model", base_model_revision=None)


def test_grpo_records_known_sft_dev_exposure_without_excluding_it() -> None:
    from llm_rlvr.tasks.sql import SqlRewardConfig
    from llm_rlvr.train.grpo import build_grpo_provenance

    provenance = build_grpo_provenance(
        base_model="model",
        warm_start_adapter="warm",
        hyperparams=GrpoHyperparams(),
        reward=SqlRewardConfig(),
        dataset="spider",
        dataset_revision="revision",
        n_train=1,
        n_dev=2,
        train_source_ids=("train",),
        dev_source_ids=("previously_seen", "unseen"),
        sft_train_source_ids=("previously_seen",),
        commit="commit",
        dirty=False,
    )
    assert provenance.sft_dev_exposure == "known"
    assert provenance.sft_exposed_dev_source_ids == ("previously_seen",)
    assert provenance.n_dev == 2


def test_adapter_tracks_sft_lineage_and_provenance_digest_separately(tmp_path) -> None:
    adapter = _adapter(tmp_path / "warm")
    first = inspect_adapter(adapter, base_model="model", base_model_revision=None)
    (adapter / "provenance.json").write_text(
        json.dumps(
            {
                "base_model": "model",
                "trace_dataset_path": "traces.jsonl",
                "train_source_ids": ["source"],
            }
        )
    )
    second = inspect_adapter(adapter, base_model="model", base_model_revision=None)
    assert first.sha256 == second.sha256
    assert second.provenance_sha256
    assert second.sft_train_source_ids == ("source",)


def test_runtime_configuration_rejects_nonfinite_nested_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        configuration.RuntimeConfigRecord(
            requested_trl={"nested": [float("nan")]},
            effective_trl={},
            requested_lora={},
            effective_lora={},
        )


def test_non_sql_prepared_sft_runs_and_publishes_with_few_shot_prompt(
    tmp_path, fake_training
) -> None:
    from llm_rlvr.artifacts.runs import require_complete
    from llm_rlvr.train.prepared import (
        PreparedSftRow,
        TrainingEligibility,
        execute_prepared_sft,
        publish_training_run,
    )

    row = PreparedSftRow.model_validate(
        {
            "prompt": [
                {"role": "system", "content": "Double the integer."},
                {"role": "user", "content": "2"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "3"},
            ],
            "completion": [{"role": "assistant", "content": "6"}],
        }
    )
    output = tmp_path / "output"
    execution = execute_prepared_sft(
        [row],
        output,
        base_model="model",
        base_model_revision=None,
        requested_config=sft_config_kwargs(SftHyperparams(), tmp_path / "other"),
        lora_config=SimpleNamespace(r=8, lora_alpha=16, lora_dropout=0.0),
        requested_lora={"r": 8, "lora_alpha": 16, "lora_dropout": 0.0},
        eligibility=TrainingEligibility(
            verification_policy="integer-doubling-v1",
            source_artifact_sha256="a" * 64,
            proof_sha256="b" * 64,
            source_ids=("integer-3",),
        ),
    )
    publish_training_run(
        output,
        execution,
        base_model="model",
        base_model_revision=execution.base_model_revision,
        training_population_ids=("integer-3",),
    )
    require_complete(output)
    trainer = next(value for name, value in fake_training if name == "trainer")
    assert trainer["train_dataset"] == [row.model_dump(mode="json")]
    assert len(trainer["train_dataset"][0]["prompt"]) == 4
    assert set(trainer["train_dataset"][0]) == {"prompt", "completion"}
    assert trainer["args"].output_dir == str(output)
    assert execution.configuration.effective_trl["output_dir"] == str(output)
    assert not (tmp_path / "other").exists()
    assert inspect_adapter(output, base_model="model", base_model_revision="resolved-commit")


def test_sft_publishes_checkpoint_with_its_actual_identity(
    tmp_path, fake_training, monkeypatch
) -> None:
    from llm_rlvr.artifacts.runs import require_complete
    from llm_rlvr.train.checkpoints import CheckpointProvenance

    traces, manifest = _trace_input(tmp_path)
    save_model = sys.modules["trl"].SFTTrainer.save_model

    def save_with_checkpoint(path):
        save_model(path)
        checkpoint = _adapter(Path(path) / "checkpoint-1", rank=8)
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1, "epoch": 0.5}))

    monkeypatch.setattr(
        sys.modules["trl"].SFTTrainer, "save_model", staticmethod(save_with_checkpoint)
    )
    output = run_sft(traces, tmp_path / "output", base_model="model", trace_manifest_path=manifest)
    checkpoint = output / "checkpoint-1"
    require_complete(output)
    require_complete(checkpoint)
    saved = CheckpointProvenance.model_validate_json((checkpoint / "provenance.json").read_text())
    final = AdapterProvenance.model_validate_json((output / "provenance.json").read_text())
    assert saved.saved_step == 1
    assert saved.training_exposure == "unknown"
    assert saved.output_adapter_identity is not None
    assert final.output_adapter_identity is not None
    assert saved.output_adapter_identity.configuration["r"] == 8
    assert final.output_adapter_identity.configuration["r"] == 32


def test_generic_publication_rejects_incompatible_provenance(tmp_path, fake_training) -> None:
    from llm_rlvr.train.configuration import RuntimeConfigRecord
    from llm_rlvr.train.prepared import publish_training_run

    output = _adapter(tmp_path / "output")
    with pytest.raises(ValueError, match="base_model_revision"):
        publish_training_run(
            output,
            RuntimeConfigRecord(
                requested_trl={}, effective_trl={}, requested_lora={}, effective_lora={}
            ),
            base_model="model",
            base_model_revision="resolved-commit",
            training_population_ids=None,
        )
    assert not (output / "complete.json").exists()


@pytest.mark.parametrize(
    "model,field,value",
    [
        (SftHyperparams, "learning_rate", float("nan")),
        (SftHyperparams, "lora_dropout", 1),
        (SftHyperparams, "epochs", 0),
        (SftHyperparams, "lora_rank", 1.5),
        (GrpoHyperparams, "beta", -1),
        (GrpoHyperparams, "top_p", 1.1),
        (GrpoHyperparams, "temperature", 0),
        (GrpoHyperparams, "num_generations", 1),
        (GrpoHyperparams, "learning_rate", float("inf")),
    ],
)
def test_invalid_hyperparameters_rejected(model, field, value) -> None:
    with pytest.raises(ValueError):
        model.model_validate({field: value})
