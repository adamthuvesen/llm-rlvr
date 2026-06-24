"""CPU tests for the forced-revision rollout (grpo-forced-revision task 2.1): feedback
building, suffix differencing, episode splicing, the rollout against fake trainer pieces, the
TRL-facing reward, the smoke verdict, best-of-N selection, and the eval message shape. No GPU,
no trl import."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from llm_rlvr_sql.data.models import Column, DatabaseSchema, Example, Table
from llm_rlvr_sql.env.text2sql_env import FeedbackConfig
from llm_rlvr_sql.eval.bestofn import select_best_of_n
from llm_rlvr_sql.eval.completions import (
    CompletionSet,
    GenerationSample,
    ProblemCompletions,
)
from llm_rlvr_sql.eval.forced_revision_generate import (
    build_revision_messages,
    forced_revision_artifact_stem,
    generate_forced_revision,
)
from llm_rlvr_sql.eval.provenance import DecodingConfig
from llm_rlvr_sql.rewards.multiturn import MultiTurnRewardConfig, score_trajectory
from llm_rlvr_sql.train.forced_revision import (
    REVISE_INSTRUCTION,
    UNPARSEABLE_FEEDBACK,
    ForcedRevisionProvenance,
    attempt_feedback,
    build_db_id_lookup,
    build_forced_revision_provenance,
    build_suffix_ids,
    count_nan_log_values,
    fr_smoke_failures,
    make_forced_revision_reward_fn,
    make_forced_revision_rollout,
    revision_user_message,
    splice_episode,
)
from llm_rlvr_sql.train.grpo import GrpoHyperparams
from llm_rlvr_sql.train.multiturn import (
    FEEDBACK_MARKER,
    IS_RATIO_HEALTHY_BAND,
    IsRatioCheck,
    MaskCheck,
)

GOLD = "SELECT a FROM t"
ATTEMPT_OK = f"<think>try</think>\n```sql\n{GOLD}\n```"
ATTEMPT_BROKEN = "<think>try</think>\n```sql\nSELECT nope FROM missing\n```"
REVISION = f"<think>fixed</think>\n```sql\n{GOLD}\n```"
DB_PATH_KEY = "concert"


def _db(scoring_root: Path) -> Path:
    return scoring_root / DB_PATH_KEY / f"{DB_PATH_KEY}.sqlite"


def _schema() -> DatabaseSchema:
    return DatabaseSchema(
        db_id=DB_PATH_KEY,
        tables=(
            Table(
                name="t",
                columns=(Column(name="a", type="integer"), Column(name="b", type="text")),
            ),
        ),
    )


# --- feedback ---


def test_attempt_feedback_success_and_error_and_unparseable(scoring_root: Path) -> None:
    ok = attempt_feedback(ATTEMPT_OK, _db(scoring_root))
    assert ok.startswith(FEEDBACK_MARKER) and "2 rows" in ok

    err = attempt_feedback(ATTEMPT_BROKEN, _db(scoring_root))
    assert err.startswith(f"{FEEDBACK_MARKER}: error")

    assert attempt_feedback("no sql here", _db(scoring_root)) == UNPARSEABLE_FEEDBACK
    assert UNPARSEABLE_FEEDBACK.startswith(FEEDBACK_MARKER)


def test_revision_user_message_is_gold_free_and_keeps_the_format_contract() -> None:
    message = revision_user_message("Execution feedback: error (syntax): no such column: x")
    assert message.startswith(FEEDBACK_MARKER)
    assert REVISE_INSTRUCTION in message
    assert "<think>" in message and "```sql" in message  # the Stage-2 final-answer format
    assert "gold" not in message.lower()


# --- suffix differencing ---


class _FakeTemplate:
    """A chat-template stand-in: each message renders to [HDR, hash(content)], plus GEN.

    Mimics transformers v5: returns a dict unless `return_dict=False` is passed — so any call
    site that forgets the flag gets string keys instead of token ids, exactly like the real
    tokenizer (this bit the first GPU smoke).
    """

    HDR, GEN = 1000, 2000

    def __call__(self, messages, add_generation_prompt=False, tokenize=True, return_dict=True):
        ids: list[int] = []
        for message in messages:
            ids += [self.HDR, abs(hash((message["role"], message["content"]))) % 997]
        if add_generation_prompt:
            ids.append(self.GEN)
        if return_dict:
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return ids


def test_build_suffix_ids_is_the_appended_turn_plus_generation_prompt() -> None:
    template = _FakeTemplate()
    suffix = build_suffix_ids(template, "feedback text")
    expected_tail = template(
        [{"role": "user", "content": "feedback text"}],
        add_generation_prompt=True,
        return_dict=False,
    )
    assert suffix == expected_tail
    assert all(isinstance(t, int) for t in suffix)  # ids, not v5 dict keys


def test_build_suffix_ids_rejects_non_prefix_templates() -> None:
    def broken(messages, **_kwargs):
        # Renders differently depending on conversation length: differencing is unsound.
        return [len(messages)]

    with pytest.raises(ValueError, match="not a prefix"):
        build_suffix_ids(broken, "feedback")


# --- splicing ---


def test_splice_episode_masks_exactly_the_suffix() -> None:
    ids, mask, lps = splice_episode(
        attempt_ids=[1, 2],
        attempt_logprobs=[-0.1, -0.2],
        suffix_ids=[8, 9, 10],
        revision_ids=[3],
        revision_logprobs=[-0.3],
    )
    assert ids == [1, 2, 8, 9, 10, 3]
    assert mask == [1, 1, 0, 0, 0, 1]
    assert lps == [-0.1, -0.2, 0.0, 0.0, 0.0, -0.3]


def test_splice_episode_rejects_misaligned_logprobs() -> None:
    with pytest.raises(ValueError, match="misaligned"):
        splice_episode([1, 2], [-0.1], [], [3], [-0.3])


# --- the rollout against fake trainer pieces ---


class _FakeTokenizer:
    def __init__(self) -> None:
        self.apply_chat_template = _FakeTemplate()

    def batch_decode(self, ids_lists, skip_special_tokens=True):
        # Map token-id rows back to canned texts by row length parity — the fake generator
        # below encodes which text it "generated" by row identity.
        return [_FAKE_DECODES[tuple(ids)] for ids in ids_lists]


_FAKE_DECODES: dict[tuple[int, ...], str] = {}


class _FakeVllmGeneration:
    """Returns canned (ids, logprobs); records the prompts it was called with."""

    def __init__(self, responses: list[tuple[list[int], str]]) -> None:
        self._responses = responses
        self.calls: list[list[list[int]]] = []

    def generate(self, prompts, images, num_generations):
        self.calls.append([list(p) for p in prompts])
        ids_batch, logprobs_batch = [], []
        for i in range(len(prompts)):
            ids, text = self._responses[(len(self.calls) - 1) * len(prompts) + i]
            _FAKE_DECODES[tuple(ids)] = text
            ids_batch.append(list(ids))
            logprobs_batch.append([[-0.5] for _ in ids])  # top-1 lists, vLLM shape
        return [list(p) for p in prompts], ids_batch, logprobs_batch, None


class _FakeTrainer:
    def __init__(self, responses) -> None:
        self.processing_class = _FakeTokenizer()
        self.vllm_generation = _FakeVllmGeneration(responses)


def _prompt(question: str = "How many rows?") -> list[dict[str, object]]:
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": f"Database schema:\nS\n\nQuestion: {question}"},
    ]


def test_rollout_returns_aligned_contract_with_masked_feedback(scoring_root: Path) -> None:
    prompts = [_prompt(), _prompt()]
    lookup = {str(p[-1]["content"]): DB_PATH_KEY for p in prompts}
    rollout = make_forced_revision_rollout(scoring_root, lookup)
    trainer = _FakeTrainer(
        responses=[
            ([11, 12], ATTEMPT_OK),  # episode-0 attempt
            ([13], ATTEMPT_BROKEN),  # episode-1 attempt
            ([21], REVISION),  # episode-0 revision
            ([22, 23], REVISION),  # episode-1 revision
        ]
    )

    output = rollout(prompts, trainer)

    assert set(output) >= {"prompt_ids", "completion_ids", "logprobs", "env_mask"}
    for ids, mask, lps in zip(
        output["completion_ids"], output["env_mask"], output["logprobs"], strict=True
    ):
        assert len(ids) == len(mask) == len(lps)
    # Episode 0: attempt [11,12] + suffix + revision [21]; the suffix is fully masked.
    mask0 = output["env_mask"][0]
    assert mask0[:2] == [1, 1] and mask0[-1] == 1 and 0 in mask0
    # Masked positions carry placeholder logprobs; model positions keep the sampled ones.
    lps0 = output["logprobs"][0]
    assert all(lp == 0.0 for lp, m in zip(lps0, mask0, strict=True) if m == 0)
    assert all(lp == -0.5 for lp, m in zip(lps0, mask0, strict=True) if m == 1)
    # Feedback content reaches the reward as extra fields; the broken attempt got an error.
    assert output["fr_attempt_text"] == [ATTEMPT_OK, ATTEMPT_BROKEN]
    assert output["fr_feedback"][0].startswith(f"{FEEDBACK_MARKER}: query executed")
    assert output["fr_feedback"][1].startswith(f"{FEEDBACK_MARKER}: error")
    # The revision pass saw prompt + attempt + suffix as its prompt.
    revision_call = trainer.vllm_generation.calls[1]
    assert revision_call[0][: len(output["prompt_ids"][0])] == output["prompt_ids"][0]
    assert len(revision_call[0]) > len(output["prompt_ids"][0]) + 2


def test_rollout_unknown_prompt_is_loud(scoring_root: Path) -> None:
    rollout = make_forced_revision_rollout(scoring_root, {})
    trainer = _FakeTrainer(responses=[([11], ATTEMPT_OK)])
    with pytest.raises(KeyError, match="no db_id"):
        rollout([_prompt()], trainer)


def test_db_id_lookup_from_grpo_rows() -> None:
    rows = [
        {"prompt": _prompt("q1"), "gold_sql": GOLD, "db_id": "concert"},
        {"prompt": _prompt("q2"), "gold_sql": GOLD, "db_id": "concert"},
    ]
    lookup = build_db_id_lookup(rows)
    assert lookup[str(_prompt("q1")[-1]["content"])] == "concert"
    assert len(lookup) == 2


# --- the TRL-facing reward ---


def test_reward_scores_attempt_plus_revision_as_a_trajectory(scoring_root: Path) -> None:
    episodes: list[tuple[list[str | None], object]] = []
    reward_fn = make_forced_revision_reward_fn(
        scoring_root, on_episode=lambda sqls, breakdown: episodes.append((sqls, breakdown))
    )

    rewards = reward_fn(
        completions=["ignored-spliced-text", "ignored"],
        gold_sql=[GOLD, GOLD],
        db_id=[DB_PATH_KEY, DB_PATH_KEY],
        fr_attempt_text=[ATTEMPT_BROKEN, "no sql at all"],
        fr_revision_text=[REVISION, REVISION],
    )

    expected_fixed = score_trajectory(
        ["SELECT nope FROM missing"], REVISION, GOLD, _db(scoring_root)
    )
    expected_unparseable = score_trajectory([], REVISION, GOLD, _db(scoring_root))
    assert rewards == [
        pytest.approx(expected_fixed.total),
        pytest.approx(expected_unparseable.total),
    ]
    # The broken->fixed episode earns the improvement-delta bonus; sqls feed diagnostics.
    assert expected_fixed.delta_bonus > 0
    assert episodes[0][0] == ["SELECT nope FROM missing", GOLD]
    assert episodes[1][0] == [None, GOLD]


def test_reward_without_rollout_extra_fields_is_loud(scoring_root: Path) -> None:
    reward_fn = make_forced_revision_reward_fn(scoring_root)
    with pytest.raises(ValueError, match="fr_attempt_text"):
        reward_fn(completions=["x"], gold_sql=[GOLD], db_id=[DB_PATH_KEY])


def test_reward_is_named_for_the_credit_scheme(scoring_root: Path) -> None:
    reward_fn = make_forced_revision_reward_fn(
        scoring_root, config=MultiTurnRewardConfig(credit="trajectory")
    )
    assert reward_fn.__name__ == "forced_revision_reward_trajectory"


# --- the smoke verdict ---


def _passing_mask(n_rows: int = 4) -> MaskCheck:
    return MaskCheck(
        n_rows=n_rows,
        rows_with_masked_tokens=n_rows,
        masked_tokens_total=40,
        feedback_in_masked_text=True,
        feedback_in_unmasked_text=False,
    )


def _passing_is_ratio() -> IsRatioCheck:
    return IsRatioCheck(
        n_logged=2,
        mean_of_means=1.0,
        overall_min=0.5,
        overall_max=1.8,
        logp_diff_mean=0.01,
        band=IS_RATIO_HEALTHY_BAND,
        in_band=True,
    )


def test_fr_smoke_failures_empty_when_all_assertions_hold() -> None:
    failures = fr_smoke_failures(
        n_episodes=96,
        mask=_passing_mask(),
        is_ratio=_passing_is_ratio(),
        nan_log_values=0,
        none_logprob_tokens=0,
    )
    assert failures == ()


def test_fr_smoke_failures_require_masked_feedback_in_every_row() -> None:
    mask = MaskCheck(
        n_rows=4,
        rows_with_masked_tokens=3,  # one episode lost its feedback segment
        masked_tokens_total=30,
        feedback_in_masked_text=True,
        feedback_in_unmasked_text=False,
    )
    failures = fr_smoke_failures(
        n_episodes=4,
        mask=mask,
        is_ratio=_passing_is_ratio(),
        nan_log_values=0,
        none_logprob_tokens=0,
    )
    assert any("every episode" in failure for failure in failures)


def test_fr_smoke_failures_name_every_broken_assertion() -> None:
    failures = fr_smoke_failures(
        n_episodes=0,
        mask=MaskCheck(
            n_rows=4,
            rows_with_masked_tokens=0,
            masked_tokens_total=0,
            feedback_in_masked_text=False,
            feedback_in_unmasked_text=True,
        ),
        is_ratio=IsRatioCheck(
            n_logged=0,
            mean_of_means=None,
            overall_min=None,
            overall_max=None,
            logp_diff_mean=None,
            band=IS_RATIO_HEALTHY_BAND,
            in_band=False,
        ),
        nan_log_values=2,
        none_logprob_tokens=1,
    )
    assert len(failures) == 8


def test_count_nan_log_values() -> None:
    history = [
        {"loss": 0.5, "grad_norm": 1.0},
        {"loss": float("nan")},
        {"grad_norm": float("nan"), "loss": 0.4},
    ]
    assert count_nan_log_values(history) == 2


# --- provenance ---


def test_forced_revision_provenance_records_rollout_and_instruction() -> None:
    provenance = build_forced_revision_provenance(
        base_model="m",
        warm_start_adapter="grpo/grpo-s0",
        hyperparams=GrpoHyperparams(),
        environment=FeedbackConfig(),
        multiturn_reward=MultiTurnRewardConfig(),
        dataset="spider",
        dataset_revision="rev",
        n_train=10,
        n_dev=2,
        commit="abc",
        dirty=False,
    )
    assert isinstance(provenance, ForcedRevisionProvenance)
    assert provenance.rollout == "forced_revision"
    assert provenance.revise_instruction == REVISE_INSTRUCTION
    assert provenance.grpo.warm_start_adapter == "grpo/grpo-s0"


# --- best-of-N selection (the D6 control) ---


def _completion_set(provenance, texts_and_sqls: list[tuple[str, str | None]]) -> CompletionSet:
    items = tuple(
        ProblemCompletions(
            example=Example(
                dataset="spider",
                split="dev",
                db_id=DB_PATH_KEY,
                question=f"q{i}",
                gold_sql=GOLD,
            ),
            samples=(GenerationSample(text=text, sql=sql),),
        )
        for i, (text, sql) in enumerate(texts_and_sqls)
    )
    return CompletionSet(provenance=provenance, items=items)


def test_best_of_n_prefers_nonempty_then_executes_then_first(
    scoring_root: Path, provenance
) -> None:
    # item 0: set A errors, set B returns rows -> B. item 1: A empty result, B errors -> A.
    # item 2: both error -> fallback to A (first).
    set_a = _completion_set(
        provenance,
        [
            ("a0", "SELECT nope FROM missing"),
            ("a1", "SELECT a FROM t WHERE a = 999"),
            ("a2", None),
        ],
    )
    set_b = _completion_set(
        provenance,
        [
            ("b0", GOLD),
            ("b1", "SELECT nope FROM missing"),
            ("b2", "SELECT nope FROM missing"),
        ],
    )
    selection, report = select_best_of_n([set_a, set_b], scoring_root)

    assert [item.samples[0].text for item in selection.items] == ["b0", "a1", "a2"]
    assert report.n_selected_nonempty == 1
    assert report.n_selected_executes == 1
    assert report.n_fallback_first == 1
    assert report.n_items_with_disagreeing_candidates == 3
    assert selection.provenance.backend == "bestof2-select"


def test_best_of_n_rejects_misaligned_sets(scoring_root: Path, provenance) -> None:
    set_a = _completion_set(provenance, [("a0", GOLD)])
    set_b = _completion_set(provenance, [("b0", GOLD), ("b1", GOLD)])
    with pytest.raises(ValueError, match="different numbers"):
        select_best_of_n([set_a, set_b], scoring_root)


# --- the eval revision-message shape ---


def test_eval_revision_messages_continue_the_conversation() -> None:
    prompt = _prompt()
    messages = build_revision_messages(prompt, ATTEMPT_OK, "Execution feedback: error (x): y")
    assert messages[: len(prompt)] == prompt
    assert messages[-2] == {"role": "assistant", "content": ATTEMPT_OK}
    assert messages[-1]["role"] == "user"
    assert REVISE_INSTRUCTION in str(messages[-1]["content"])


def test_error_only_fresh_generation_batches_only_error_feedback(
    monkeypatch: pytest.MonkeyPatch, scoring_root: Path
) -> None:
    class _Output:
        def __init__(self, text: str) -> None:
            self.outputs = [SimpleNamespace(text=text)]

    class _FakeLLM:
        instances: ClassVar[list[_FakeLLM]] = []

        def __init__(self, **_kwargs) -> None:
            self.calls: list[list[object]] = []
            _FakeLLM.instances.append(self)

        def chat(self, prompts, _params, **_kwargs):
            self.calls.append(list(prompts))
            if len(self.calls) == 1:
                return [_Output(ATTEMPT_BROKEN), _Output(ATTEMPT_OK)]
            return [_Output(REVISION)]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=_FakeLLM, SamplingParams=lambda **kwargs: kwargs),
    )
    examples = [
        Example(
            dataset="spider",
            split="dev",
            db_id=DB_PATH_KEY,
            question=f"q{i}",
            gold_sql=GOLD,
        )
        for i in range(2)
    ]

    completion_set, transcripts = generate_forced_revision(
        examples,
        {DB_PATH_KEY: _schema()},
        scoring_root,
        model="fake",
        decoding=DecodingConfig(max_new_tokens=16),
        dataset="spider",
        dataset_revision="rev",
        split="dev",
        policy="error_only",
        commit="abc",
        dirty=False,
    )

    fake_llm = _FakeLLM.instances[-1]
    assert [len(call) for call in fake_llm.calls] == [2, 1]
    assert [item.samples[0].text for item in completion_set.items] == [REVISION, ATTEMPT_OK]
    assert completion_set.provenance.backend == "vllm-forced-revision-error-only"
    assert [episode.revision_requested for episode in transcripts.items] == [True, False]
    assert [episode.revision_text for episode in transcripts.items] == [REVISION, None]


def test_error_only_fresh_generation_skips_empty_revision_batch(
    monkeypatch: pytest.MonkeyPatch, scoring_root: Path
) -> None:
    class _Output:
        def __init__(self, text: str) -> None:
            self.outputs = [SimpleNamespace(text=text)]

    class _FakeLLM:
        instances: ClassVar[list[_FakeLLM]] = []

        def __init__(self, **_kwargs) -> None:
            self.calls: list[list[object]] = []
            _FakeLLM.instances.append(self)

        def chat(self, prompts, _params, **_kwargs):
            self.calls.append(list(prompts))
            return [_Output(ATTEMPT_OK)]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=_FakeLLM, SamplingParams=lambda **kwargs: kwargs),
    )
    example = Example(
        dataset="spider",
        split="dev",
        db_id=DB_PATH_KEY,
        question="q",
        gold_sql=GOLD,
    )

    completion_set, transcripts = generate_forced_revision(
        [example],
        {DB_PATH_KEY: _schema()},
        scoring_root,
        model="fake",
        decoding=DecodingConfig(max_new_tokens=16),
        dataset="spider",
        dataset_revision="rev",
        split="dev",
        policy="error_only",
    )

    assert [len(call) for call in _FakeLLM.instances[-1].calls] == [1]
    assert completion_set.items[0].samples[0].text == ATTEMPT_OK
    assert transcripts.items[0].revision_requested is False


def test_forced_revision_artifact_stem_separates_policies() -> None:
    always = forced_revision_artifact_stem(
        dataset="spider",
        adapter="grpo-fr/grpo-fr-pilot",
        limit=200,
        temperature=0.0,
        sampling_seed=0,
        policy="always",
    )
    error_only = forced_revision_artifact_stem(
        dataset="spider",
        adapter="grpo-fr/grpo-fr-pilot",
        limit=200,
        temperature=0.0,
        sampling_seed=0,
        policy="error_only",
    )

    assert always == "spider-dev-fr-grpo-fr-pilot-n200"
    assert error_only == "spider-dev-fr-error-only-grpo-fr-pilot-n200"
    assert always != error_only


# --- fr-metrics (fix/harm decomposition) ---


def test_fr_metrics_decomposes_fix_and_harm(scoring_root: Path, provenance) -> None:
    from llm_rlvr_sql.eval.forced_revision_generate import (
        ForcedRevisionEpisode,
        ForcedRevisionTranscriptSet,
    )
    from llm_rlvr_sql.eval.fr_metrics import compute_fr_metrics

    wrong_sql = "SELECT a FROM t WHERE a = 999"  # executes, empty result -> wrong
    err_sql = "SELECT nope FROM missing"

    def _item(question: str, final_text: str, final_sql: str | None) -> ProblemCompletions:
        return ProblemCompletions(
            example=Example(
                dataset="spider",
                split="dev",
                db_id=DB_PATH_KEY,
                question=question,
                gold_sql=GOLD,
            ),
            samples=(GenerationSample(text=final_text, sql=final_sql),),
        )

    def _episode(question: str, attempt_sql: str, feedback: str) -> ForcedRevisionEpisode:
        return ForcedRevisionEpisode(
            db_id=DB_PATH_KEY,
            question=question,
            attempt_text=f"<think>x</think>\n```sql\n{attempt_sql}\n```",
            feedback=feedback,
            final_text="ignored",
        )

    completion_set = CompletionSet(
        provenance=provenance,
        items=(
            _item("q0", REVISION, GOLD),  # error attempt -> fixed (error feedback)
            _item("q1", REVISION, GOLD),  # wrong-rows attempt -> fixed (result feedback)
            _item("q2", "broken", wrong_sql),  # right attempt -> harmed
            _item("q3", REVISION, GOLD),  # right attempt -> kept (identical SQL)
        ),
    )
    transcripts = ForcedRevisionTranscriptSet(
        provenance=provenance,
        items=(
            _episode("q0", err_sql, "Execution feedback: error (syntax): no such column"),
            _episode("q1", wrong_sql, "Execution feedback: query executed, returned 0 rows."),
            _episode("q2", GOLD, "Execution feedback: query executed, returned 2 rows."),
            _episode("q3", GOLD, "Execution feedback: query executed, returned 2 rows."),
        ),
    )

    report = compute_fr_metrics(completion_set, transcripts, scoring_root)

    assert report.n == 4 and report.n_skipped_gold_error == 0
    assert report.attempt_ex == pytest.approx(0.5)
    assert report.final_ex == pytest.approx(0.75)
    assert report.n_fixed == 2 and report.fix_rate == pytest.approx(1.0)
    assert report.n_fixed_from_error_feedback == 1
    assert report.n_fixed_from_result_feedback == 1
    assert report.n_harmed == 1 and report.harm_rate == pytest.approx(0.5)
    assert report.n_revised == 3  # q3 kept its SQL
    assert report.net_delta == pytest.approx(0.25)


def test_forced_revision_eval_report_accounts_for_error_only_policy(
    scoring_root: Path, provenance
) -> None:
    from llm_rlvr_sql.eval.forced_revision_generate import (
        ForcedRevisionEpisode,
        ForcedRevisionTranscriptSet,
    )
    from llm_rlvr_sql.eval.fr_metrics import compute_forced_revision_eval_report

    wrong_sql = "SELECT a FROM t WHERE a = 999"
    err_sql = "SELECT nope FROM missing"

    completion_set = CompletionSet(
        provenance=provenance,
        items=(
            ProblemCompletions(
                example=Example(
                    dataset="spider",
                    split="dev",
                    db_id=DB_PATH_KEY,
                    question="q0",
                    gold_sql=GOLD,
                ),
                samples=(GenerationSample(text=REVISION, sql=GOLD),),
            ),
            ProblemCompletions(
                example=Example(
                    dataset="spider",
                    split="dev",
                    db_id=DB_PATH_KEY,
                    question="q1",
                    gold_sql=GOLD,
                ),
                samples=(GenerationSample(text=ATTEMPT_OK, sql=GOLD),),
            ),
        ),
    )
    transcripts = ForcedRevisionTranscriptSet(
        provenance=provenance,
        items=(
            ForcedRevisionEpisode(
                db_id=DB_PATH_KEY,
                question="q0",
                attempt_text=f"<think>x</think>\n```sql\n{err_sql}\n```",
                feedback="Execution feedback: error (syntax): no such column",
                final_text=REVISION,
                revision_requested=True,
                revision_text=REVISION,
            ),
            ForcedRevisionEpisode(
                db_id=DB_PATH_KEY,
                question="q1",
                attempt_text=f"<think>x</think>\n```sql\n{wrong_sql}\n```",
                feedback="Execution feedback: query executed, returned 0 rows.",
                final_text=ATTEMPT_OK,
                revision_requested=False,
            ),
        ),
    )

    report = compute_forced_revision_eval_report(
        completion_set, transcripts, scoring_root, policy="error_only"
    )

    assert report.policy == "error_only"
    assert report.attempt_ex == pytest.approx(0.0)
    assert report.final_ex == pytest.approx(1.0)
    assert report.n_error_feedback == 1
    assert report.n_second_generation == 1
    assert report.second_generation_rate == pytest.approx(0.5)
    assert report.n_kept_attempt == 1
    assert report.n_fixed == 2
    assert report.n_harmed == 0
    assert any(note == "second_generation_count=1" for note in report.cost_notes)


def test_fr_metrics_rejects_misaligned_inputs(scoring_root: Path, provenance) -> None:
    from llm_rlvr_sql.eval.forced_revision_generate import ForcedRevisionTranscriptSet
    from llm_rlvr_sql.eval.fr_metrics import compute_fr_metrics

    completion_set = CompletionSet(provenance=provenance, items=())
    transcripts = ForcedRevisionTranscriptSet(provenance=provenance, items=())
    assert compute_fr_metrics(completion_set, transcripts, scoring_root).n == 0


# --- error-only forced-revision replay ---


def test_error_only_replay_revises_only_on_error_feedback(scoring_root: Path, provenance) -> None:
    from llm_rlvr_sql.eval.forced_revision_generate import (
        ForcedRevisionEpisode,
        ForcedRevisionTranscriptSet,
        replay_error_only_revision,
    )

    err_attempt = "<think>x</think>\n```sql\nSELECT nope FROM missing\n```"
    non_error_attempt = REVISION
    harmful_revision = "<think>x</think>\n```sql\nSELECT a FROM t WHERE a = 999\n```"
    completion_set = CompletionSet(
        provenance=provenance,
        items=(
            ProblemCompletions(
                example=Example(
                    dataset="spider",
                    split="dev",
                    db_id=DB_PATH_KEY,
                    question="q0",
                    gold_sql=GOLD,
                ),
                samples=(GenerationSample(text=REVISION, sql=GOLD),),
            ),
            ProblemCompletions(
                example=Example(
                    dataset="spider",
                    split="dev",
                    db_id=DB_PATH_KEY,
                    question="q1",
                    gold_sql=GOLD,
                ),
                samples=(
                    GenerationSample(
                        text=harmful_revision,
                        sql="SELECT a FROM t WHERE a = 999",
                    ),
                ),
            ),
        ),
    )
    transcripts = ForcedRevisionTranscriptSet(
        provenance=provenance,
        items=(
            ForcedRevisionEpisode(
                db_id=DB_PATH_KEY,
                question="q0",
                attempt_text=err_attempt,
                feedback="Execution feedback: error (syntax): no such table",
                final_text=REVISION,
            ),
            ForcedRevisionEpisode(
                db_id=DB_PATH_KEY,
                question="q1",
                attempt_text=non_error_attempt,
                feedback="Execution feedback: query executed, returned 2 rows.",
                final_text=harmful_revision,
            ),
        ),
    )

    replay, report = replay_error_only_revision(completion_set, transcripts, scoring_root)

    assert [item.samples[0].text for item in replay.items] == [REVISION, non_error_attempt]
    assert replay.provenance.backend == "forced-revision-error-only-replay"
    assert report.attempt_ex == pytest.approx(0.5)
    assert report.always_revise_ex == pytest.approx(0.5)
    assert report.error_only_ex == pytest.approx(1.0)
    assert report.n_revised == 1 and report.revised_rate == pytest.approx(0.5)
    assert report.n_fixed == 1 and report.n_harmed == 0


def test_error_only_replay_keeps_parse_error_attempts_on_error_feedback(
    scoring_root: Path, provenance
) -> None:
    from llm_rlvr_sql.eval.forced_revision_generate import (
        ForcedRevisionEpisode,
        ForcedRevisionTranscriptSet,
        replay_error_only_revision,
    )

    completion_set = CompletionSet(
        provenance=provenance,
        items=(
            ProblemCompletions(
                example=Example(
                    dataset="spider",
                    split="dev",
                    db_id=DB_PATH_KEY,
                    question="q",
                    gold_sql=GOLD,
                ),
                samples=(GenerationSample(text=REVISION, sql=GOLD),),
            ),
        ),
    )
    transcripts = ForcedRevisionTranscriptSet(
        provenance=provenance,
        items=(
            ForcedRevisionEpisode(
                db_id=DB_PATH_KEY,
                question="q",
                attempt_text="no sql",
                feedback=UNPARSEABLE_FEEDBACK,
                final_text=REVISION,
            ),
        ),
    )

    replay, report = replay_error_only_revision(completion_set, transcripts, scoring_root)

    assert replay.items[0].samples[0].text == REVISION
    assert report.n_revised == 1
    assert report.n_fixed == 1


def test_error_only_replay_rejects_misaligned_inputs(scoring_root: Path, provenance) -> None:
    from llm_rlvr_sql.eval.forced_revision_generate import (
        ForcedRevisionEpisode,
        ForcedRevisionTranscriptSet,
        replay_error_only_revision,
    )

    completion_set = _completion_set(provenance, [(REVISION, GOLD)])
    transcripts = ForcedRevisionTranscriptSet(
        provenance=provenance,
        items=(
            ForcedRevisionEpisode(
                db_id=DB_PATH_KEY,
                question="different",
                attempt_text=REVISION,
                feedback="Execution feedback: query executed, returned 2 rows.",
                final_text=REVISION,
            ),
        ),
    )

    with pytest.raises(ValueError, match="not aligned"):
        replay_error_only_revision(completion_set, transcripts, scoring_root)
