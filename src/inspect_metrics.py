"""Inspect metric helpers for multi-field haiku scores."""

from __future__ import annotations

from inspect_ai.scorer import Metric, SampleScore, metric

# Keys stored in Score.value; means are reported in the Inspect results table.
HAIKU_SCORE_KEYS = (
    "subject_cosine_full",
    "subject_cosine_word_sum",
    "syllable_l1_error",
    "syllable_perfect",
    "line_count_ok",
    "subject_mentioned",
)

INVALID_HAIKU_SCORE = {
    "subject_cosine_full": 0.0,
    "subject_cosine_word_sum": 0.0,
    "syllable_l1_error": 99,
    "syllable_perfect": 0,
    "line_count_ok": 0,
    "subject_mentioned": 0,
}


def _score_dict(sample: SampleScore) -> dict:
    value = sample.score.value
    if isinstance(value, dict):
        return value
    if sample.score.metadata:
        return {k: sample.score.metadata[k] for k in HAIKU_SCORE_KEYS if k in sample.score.metadata}
    return {}


def _mean_key(key: str):
    def compute(scores: list[SampleScore]) -> float:
        import numpy as np

        vals = [float(d[key]) for s in scores if key in (d := _score_dict(s))]
        return float(np.mean(vals)) if vals else 0.0

    return compute


@metric(name="subject_cosine_full_mean")
def subject_cosine_full_mean() -> Metric:
    return _mean_key("subject_cosine_full")


@metric(name="subject_cosine_word_sum_mean")
def subject_cosine_word_sum_mean() -> Metric:
    return _mean_key("subject_cosine_word_sum")


@metric(name="syllable_l1_error_mean")
def syllable_l1_error_mean() -> Metric:
    return _mean_key("syllable_l1_error")


@metric(name="syllable_perfect_rate")
def syllable_perfect_rate() -> Metric:
    return _mean_key("syllable_perfect")


@metric(name="line_count_ok_rate")
def line_count_ok_rate() -> Metric:
    return _mean_key("line_count_ok")


@metric(name="subject_mentioned_rate")
def subject_mentioned_rate() -> Metric:
    return _mean_key("subject_mentioned")


def haiku_scorer_metrics() -> list:
    return [
        subject_cosine_full_mean(),
        subject_cosine_word_sum_mean(),
        syllable_l1_error_mean(),
        syllable_perfect_rate(),
        line_count_ok_rate(),
        subject_mentioned_rate(),
    ]
