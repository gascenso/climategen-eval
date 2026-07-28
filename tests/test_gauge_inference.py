from __future__ import annotations

import numpy as np

from gauge_climate.inference import (
    empirical_reference_percentile,
    reference_summary_uncertainty,
)


def test_empirical_reference_percentile_uses_a_finite_sample_midrank():
    rng = np.random.default_rng(32)
    references = rng.gamma(shape=2.0, size=2_000)
    low = empirical_reference_percentile(-1.0, references)
    middle = empirical_reference_percentile(float(np.median(references)), references)
    high = empirical_reference_percentile(1e6, references)
    assert 0.0 < low < middle < high < 1.0


def test_reference_summary_uncertainty_is_paired_and_contains_point_summaries():
    rng = np.random.default_rng(11)
    reference = rng.gamma(shape=2.0, size=500)
    model = reference + rng.normal(scale=0.03, size=500)
    result = reference_summary_uncertainty(
        reference_values=reference,
        model_values=model,
        reference_quantile=0.95,
        interval=0.95,
        n_resamples=500,
        seed=2,
    )
    q50 = float(np.quantile(reference, 0.5))
    q95 = float(np.quantile(reference, 0.95))
    gap = (float(np.median(model)) - q50) / (q95 - q50)
    assert result["reference_q50"][0] <= q50 <= result["reference_q50"][1]
    assert result["reference_q95"][0] <= q95 <= result["reference_q95"][1]
    assert result["reference_upper"][0] <= q95 <= result["reference_upper"][1]
    assert result["reference_gap"][0] <= gap <= result["reference_gap"][1]


def test_reference_summary_uncertainty_returns_null_gap_for_degenerate_width():
    result = reference_summary_uncertainty(
        reference_values=np.zeros(40),
        model_values=np.zeros(40),
        reference_quantile=0.95,
        interval=0.95,
        n_resamples=100,
        seed=2,
    )
    assert result["reference_q50"] == (0.0, 0.0)
    assert result["reference_q95"] == (0.0, 0.0)
    assert result["reference_upper"] == (0.0, 0.0)
    assert result["reference_gap"] == (None, None)
