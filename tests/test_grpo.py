"""grpo-training: hyperparameter contract, disjoint dev split, dataset rows, provenance."""

from __future__ import annotations

from llm_rlvr_sql.data.models import Column, DatabaseSchema, Example, Table
from llm_rlvr_sql.tasks.sql import RewardConfig
from llm_rlvr_sql.train.grpo import (
    GrpoHyperparams,
    GrpoProvenance,
    build_grpo_provenance,
    build_grpo_rows,
    grpo_config_kwargs,
    lora_kwargs,
    split_train_dev,
)


def _schema(db_id: str) -> DatabaseSchema:
    return DatabaseSchema(
        db_id=db_id, tables=(Table(name="t", columns=(Column(name="a", type="int"),)),)
    )


SCHEMAS = {f"db{i}": _schema(f"db{i}") for i in range(10)}
EXAMPLES = [
    Example(db_id=f"db{i}", question=f"q{i}", gold_sql=f"SELECT {i} FROM t", split="train")
    for i in range(10)
]


def test_grpo_config_carries_the_decided_dials() -> None:
    kw = grpo_config_kwargs(GrpoHyperparams(), "/out")
    assert kw["num_generations"] == 4
    assert kw["max_completion_length"] == 1024
    assert kw["use_vllm"] is True
    assert kw["vllm_mode"] == "colocate"
    assert kw["bf16"] is True
    assert kw["beta"] == GrpoHyperparams().beta
    assert kw["learning_rate"] == 1e-5  # the D4 default, locked off the pilot


def test_lora_targets_all_linear_at_rank_32() -> None:
    kw = lora_kwargs(GrpoHyperparams())
    assert kw["target_modules"] == "all-linear"
    assert kw["r"] == 32
    assert kw["lora_alpha"] == 64


def test_dev_split_is_disjoint_and_deterministic() -> None:
    train_a, dev_a = split_train_dev(EXAMPLES, dev_size=3, seed=0)
    _train_b, dev_b = split_train_dev(EXAMPLES, dev_size=3, seed=0)
    # deterministic in seed
    assert [e.db_id for e in dev_a] == [e.db_id for e in dev_b]
    # disjoint: the dev slice never appears in train
    train_ids = {e.db_id for e in train_a}
    dev_ids = {e.db_id for e in dev_a}
    assert train_ids.isdisjoint(dev_ids)
    assert len(train_a) + len(dev_a) == len(EXAMPLES)
    assert len(dev_a) == 3


def test_dev_split_clamps_so_train_is_never_empty() -> None:
    # a tiny smoke set must still leave something to train on.
    train, dev = split_train_dev(EXAMPLES[:2], dev_size=500, seed=0)
    assert len(train) >= 1
    assert {e.db_id for e in train}.isdisjoint(e.db_id for e in dev)


def test_grpo_rows_carry_prompt_gold_and_db() -> None:
    rows = build_grpo_rows(EXAMPLES[:3], SCHEMAS)
    assert len(rows) == 3
    for example, row in zip(EXAMPLES[:3], rows, strict=True):
        assert set(row) == {"prompt", "gold_sql", "db_id"}
        assert row["gold_sql"] == example.gold_sql
        assert row["db_id"] == example.db_id
        assert isinstance(row["prompt"], list)  # chat messages for the TRL chat-template path


def test_provenance_records_reward_warmstart_and_seed_and_roundtrips() -> None:
    prov = build_grpo_provenance(
        base_model="Qwen/Qwen2.5-Coder-3B-Instruct",
        warm_start_adapter="sft/sft-r32",
        hyperparams=GrpoHyperparams(seed=7),
        reward=RewardConfig(arm="near_binary"),
        dataset="spider",
        dataset_revision="rev123",
        n_train=7,
        n_dev=3,
        commit="abc",
        dirty=False,
    )
    assert prov.warm_start_adapter == "sft/sft-r32"
    assert prov.reward.arm == "near_binary"
    assert prov.hyperparams.seed == 7
    # round-trips through the strict typed artifact contract
    assert GrpoProvenance.model_validate_json(prov.model_dump_json()) == prov
