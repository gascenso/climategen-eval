from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import ReferenceStatus


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


def reference_status(
    model_value: float,
    reference_boundary: float,
) -> ReferenceStatus:
    if model_value > reference_boundary:
        return ReferenceStatus.ABOVE_REFERENCE_RANGE
    return ReferenceStatus.WITHIN_REFERENCE_RANGE


def empirical_reference_percentile(value: float, reference_values: np.ndarray) -> float:
    """Mid-rank percentile of ``value`` within the empirical real reference."""

    values = np.asarray(reference_values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("reference_values must contain finite values.")
    lower = float(np.sum(values < value))
    equal = float(np.sum(values == value))
    return float((lower + 0.5 * equal + 0.5) / (values.size + 1.0))


def reference_summary_uncertainty(
    *,
    reference_values: np.ndarray,
    model_values: np.ndarray,
    reference_quantile: float,
    interval: float,
    n_resamples: int,
    seed: int,
) -> dict[str, tuple[float | None, float | None]]:
    """Conditional Monte Carlo intervals for reference summaries and the gap.

    The paired resampling preserves the shared held-out-real comparison in each
    real-reference/model-reference repetition. These intervals quantify
    numerical uncertainty conditional on the fixed archive and generated bank;
    they are not population confidence intervals.
    """

    reference = np.asarray(reference_values, dtype=np.float64).reshape(-1)
    model = np.asarray(model_values, dtype=np.float64).reshape(-1)
    if reference.shape != model.shape or reference.size < 2:
        raise ValueError("Paired reference and model values must have equal length >= 2.")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(model)):
        raise ValueError("Paired reference summaries require finite values.")
    if not 0.5 < reference_quantile < 1.0 or not 0.5 < interval < 1.0:
        raise ValueError("reference_quantile and interval must lie in (0.5, 1).")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive.")

    rng = np.random.default_rng(seed)
    summaries = np.empty((n_resamples, 5), dtype=np.float64)
    summaries[:, 4] = np.nan
    for index in range(n_resamples):
        selected = rng.integers(0, reference.size, size=reference.size)
        sampled_reference = reference[selected]
        q50 = float(np.quantile(sampled_reference, 0.50))
        q95 = float(np.quantile(sampled_reference, 0.95))
        upper = float(np.quantile(sampled_reference, reference_quantile))
        model_median = float(np.median(model[selected]))
        summaries[index, :4] = (q50, q95, upper, model_median)
        width = upper - q50
        if width > np.finfo(float).eps:
            summaries[index, 4] = (model_median - q50) / width

    tail = (1.0 - interval) / 2.0

    def bounds(values: np.ndarray) -> tuple[float | None, float | None]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return (None, None)
        return (
            float(np.quantile(finite, tail)),
            float(np.quantile(finite, 1.0 - tail)),
        )

    return {
        "reference_q50": bounds(summaries[:, 0]),
        "reference_q95": bounds(summaries[:, 1]),
        "reference_upper": bounds(summaries[:, 2]),
        "model_median": bounds(summaries[:, 3]),
        "reference_gap": bounds(summaries[:, 4]),
    }
