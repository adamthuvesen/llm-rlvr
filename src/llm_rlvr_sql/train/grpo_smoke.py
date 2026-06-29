"""TRL GRPO smoke checks shared by multi-turn and forced-revision drivers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from llm_rlvr_sql._base import Frozen

FEEDBACK_MARKER = "Execution feedback"

# Stage-2 colocate smoke band for TRL importance-sampling ratio (#4543 guard).
IS_RATIO_HEALTHY_BAND = (0.8, 1.25)


def logged_mean(log_history: Sequence[Mapping[str, object]], key: str) -> float | None:
    """Mean of a numeric TRL log-history field across the steps that recorded it (None if none)."""
    values = [float(entry[key]) for entry in log_history if key in entry]
    return (sum(values) / len(values)) if values else None


class MaskCheck(Frozen):
    n_rows: int
    rows_with_masked_tokens: int
    masked_tokens_total: int
    feedback_in_masked_text: bool
    feedback_in_unmasked_text: bool


def check_tool_masking(
    completion_ids: Sequence[Sequence[int]],
    completion_mask: Sequence[Sequence[int]],
    tool_mask: Sequence[Sequence[int]],
    decode: Callable[[Sequence[int]], str],
) -> MaskCheck:
    rows_with_masked = 0
    masked_total = 0
    masked_texts: list[str] = []
    unmasked_texts: list[str] = []
    for ids, cmask, tmask in zip(completion_ids, completion_mask, tool_mask, strict=True):
        masked = [i for i, c, t in zip(ids, cmask, tmask, strict=True) if c == 1 and t == 0]
        unmasked = [i for i, c, t in zip(ids, cmask, tmask, strict=True) if c == 1 and t == 1]
        if masked:
            rows_with_masked += 1
            masked_total += len(masked)
            masked_texts.append(decode(masked))
        unmasked_texts.append(decode(unmasked))
    return MaskCheck(
        n_rows=len(completion_ids),
        rows_with_masked_tokens=rows_with_masked,
        masked_tokens_total=masked_total,
        feedback_in_masked_text=any(FEEDBACK_MARKER in text for text in masked_texts),
        feedback_in_unmasked_text=any(FEEDBACK_MARKER in text for text in unmasked_texts),
    )


class IsRatioCheck(Frozen):
    n_logged: int
    mean_of_means: float | None
    overall_min: float | None
    overall_max: float | None
    logp_diff_mean: float | None
    band: tuple[float, float]
    in_band: bool


def check_is_ratio(
    log_history: Sequence[Mapping[str, object]],
    *,
    band: tuple[float, float] = IS_RATIO_HEALTHY_BAND,
) -> IsRatioCheck:
    means = [
        float(entry["sampling/importance_sampling_ratio/mean"])
        for entry in log_history
        if "sampling/importance_sampling_ratio/mean" in entry
    ]
    mins = [
        float(entry["sampling/importance_sampling_ratio/min"])
        for entry in log_history
        if "sampling/importance_sampling_ratio/min" in entry
    ]
    maxes = [
        float(entry["sampling/importance_sampling_ratio/max"])
        for entry in log_history
        if "sampling/importance_sampling_ratio/max" in entry
    ]
    logp_diffs = [
        float(entry["sampling/sampling_logp_difference/mean"])
        for entry in log_history
        if "sampling/sampling_logp_difference/mean" in entry
    ]
    low, high = band
    return IsRatioCheck(
        n_logged=len(means),
        mean_of_means=(sum(means) / len(means)) if means else None,
        overall_min=min(mins) if mins else None,
        overall_max=max(maxes) if maxes else None,
        logp_diff_mean=(sum(logp_diffs) / len(logp_diffs)) if logp_diffs else None,
        band=band,
        in_band=bool(means) and all(low <= m <= high for m in means),
    )


def capturing_grpo_trainer(base_trainer_cls):
    """Return a GRPOTrainer subclass that records rollout mask batches."""

    class CapturingTrainer(base_trainer_cls):
        def __init__(self, *args, captured: list[dict[str, list[list[int]]]], **kwargs):
            super().__init__(*args, **kwargs)
            self._captured = captured

        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)
            if "tool_mask" in output:
                self._captured.append(
                    {
                        key: output[key].detach().cpu().tolist()
                        for key in ("completion_ids", "completion_mask", "tool_mask")
                    }
                )
            return output

    return CapturingTrainer
