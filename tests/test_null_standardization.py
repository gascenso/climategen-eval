from __future__ import annotations

import numpy as np

from climategen_eval.standardization import (
    null_standardization_ranges,
    standardized_metric_value,
    standardize_metric_values,
)
from climategen_eval.tc_bt import (
    TCMetricConfig,
    build_class_metric_cache,
    compute_real_vs_real_class_summary,
)


def test_lower_metric_null_standardized_score() -> None:
    ranges = null_standardization_ranges(
        {
            "pixel_hist_w1": {
                "q05": 0.5,
                "q50": 1.0,
                "q95": 3.0,
            }
        }
    )

    node = ranges["pixel_hist_w1"]
    assert node["usable_for_standardized_gap"]
    assert node["good"] == 1.0
    assert node["scale"] == 2.0

    score = standardized_metric_value(value=5.0, standardization=node, display_null_widths=4.0)
    assert score["standardized_gap"] == 2.0
    assert score["score"] == 0.5


def test_lower_metric_better_than_null_median_has_zero_gap() -> None:
    ranges = null_standardization_ranges(
        {
            "pixel_hist_w1": {
                "q05": 0.5,
                "q50": 1.0,
                "q95": 3.0,
            }
        }
    )

    score = standardized_metric_value(
        value=0.75,
        standardization=ranges["pixel_hist_w1"],
        display_null_widths=4.0,
    )
    assert score["standardized_gap"] == 0.0
    assert score["score"] == 1.0


def test_zero_null_width_is_marked_unusable() -> None:
    ranges = null_standardization_ranges(
        {
            "gen_exceedance_rate_total": {
                "q05": 0.0,
                "q50": 0.0,
                "q95": 0.0,
            }
        }
    )

    assert not ranges["gen_exceedance_rate_total"]["usable_for_standardized_gap"]
    assert standardize_metric_values({"gen_exceedance_rate_total": 0.1}, ranges) == {}


def test_real_vs_real_summary_includes_feature_metrics() -> None:
    rng = np.random.default_rng(42)
    real = rng.normal(240.0, 10.0, size=(6, 16, 16)).astype(np.float32)
    pseudo = rng.normal(240.0, 10.0, size=(5, 16, 16)).astype(np.float32)
    cache = build_class_metric_cache(
        real_k=real,
        gen_k_raw=pseudo,
        cfg=TCMetricConfig(image_size=16, profile_bins=8, psd_bins=8),
    )

    summary = compute_real_vs_real_class_summary(
        cache,
        left_idx=np.arange(6),
        right_idx=np.arange(5),
    )

    assert np.isfinite(summary["feature_mmd2"])
    assert np.isfinite(summary["diversity_feature_space_abs_gap"])
    assert np.isfinite(summary["psd_l2"])
