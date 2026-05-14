from __future__ import annotations

import numpy as np

from climategen_eval.standardization import (
    null_standardization_ranges,
    standardized_metric_value,
    standardize_metric_sample_distributions,
    standardize_metric_values,
    summarize_control_response_shape,
)
from climategen_eval.tc_bt import (
    TCMetricConfig,
    build_class_metric_cache,
    compute_real_vs_real_class_summary,
    evaluate_model_by_class,
    mean_profile_mse_gap,
)


def test_lower_metric_null_standardized_gap() -> None:
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

    standardized = standardized_metric_value(value=5.0, standardization=node)
    assert standardized["standardized_gap"] == 2.0


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

    standardized = standardized_metric_value(
        value=0.75,
        standardization=ranges["pixel_hist_w1"],
    )
    assert standardized["standardized_gap"] == 0.0


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


def test_psd_gap_uses_mean_squared_profile_difference() -> None:
    left = np.asarray([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    right = np.asarray([[1.0, 4.0], [3.0, 6.0]], dtype=np.float32)

    assert mean_profile_mse_gap(left, right) == 2.5


def test_model_metric_bootstrap_report_includes_intervals_and_samples() -> None:
    rng = np.random.default_rng(7)
    cfg = TCMetricConfig(image_size=16, profile_bins=8, psd_bins=8)
    real_by_class = {
        0: rng.normal(240.0, 10.0, size=(6, 16, 16)).astype(np.float32),
        1: rng.normal(230.0, 12.0, size=(6, 16, 16)).astype(np.float32),
    }
    generated_by_class = {
        0: rng.normal(241.0, 10.0, size=(5, 16, 16)).astype(np.float32),
        1: rng.normal(231.0, 12.0, size=(5, 16, 16)).astype(np.float32),
    }

    report = evaluate_model_by_class(
        real_by_class=real_by_class,
        generated_by_class=generated_by_class,
        cfg=cfg,
        bootstrap_reps=4,
        bootstrap_seed=11,
        include_bootstrap_samples=True,
    )

    bootstrap = report["aggregate_primary_bootstrap"]
    assert bootstrap["enabled"]
    assert bootstrap["method"] == "within_class_two_sample_nonparametric_bootstrap"
    assert len(bootstrap["samples"]["pixel_hist_w1"]) == 4

    summary = bootstrap["values"]["pixel_hist_w1"]
    assert summary["point"] == report["aggregate_primary_raw"]["pixel_hist_w1"]
    assert summary["interval_95"]["low"] <= summary["interval_95"]["high"]
    assert summary["min"] <= summary["q50"] <= summary["max"]


def test_standardized_bootstrap_report_includes_gap_intervals() -> None:
    ranges = {
        "pixel_hist_w1": {
            "direction": "lower",
            "good": 1.0,
            "scale": 2.0,
            "usable_for_standardized_gap": True,
        }
    }

    report = standardize_metric_sample_distributions(
        {"pixel_hist_w1": [1.0, 3.0, 5.0]},
        ranges,
        point_values={"pixel_hist_w1": 3.0},
    )

    node = report["pixel_hist_w1"]
    assert node["point"]["standardized_gap"] == 1.0
    assert "score" not in node["point"]
    assert node["standardized_gap"]["interval_95"]["low"] <= 1.0
    assert "score" not in node


def test_control_response_shape_reports_monotonicity_and_justification() -> None:
    points = [
        {"severity": 0.0, "value": 1.0, "power": 0.0},
        {"severity": 1.0, "value": 1.5, "power": 0.4},
        {"severity": 2.0, "value": 3.5, "power": 0.9},
    ]

    shape = summarize_control_response_shape(
        points,
        direction="lower",
        good=1.0,
        threshold=3.0,
        target_power=0.8,
    )

    assert shape["expected_raw_correlation_sign"] == "positive"
    assert np.isclose(shape["spearman_severity_value"], 1.0)
    assert np.isclose(shape["spearman_severity_worsening_response"], 1.0)
    assert shape["directionally_consistent"]
    assert shape["monotone_worsening"]
    assert shape["first_target_power_severity"] == 2.0
    assert shape["first_justified_severity"] == 2.0
    assert shape["justifies_inclusion"]
    assert shape["max_response"]["directional_response"] == 2.5
    assert shape["max_response"]["directional_response_in_null_widths"] == 1.25


def test_control_response_shape_handles_higher_is_better_metrics() -> None:
    points = [
        {"severity": 0.0, "value": 10.0, "power": 0.0},
        {"severity": 1.0, "value": 9.0, "power": 0.5},
        {"severity": 2.0, "value": 7.0, "power": 0.9},
    ]

    shape = summarize_control_response_shape(
        points,
        direction="higher",
        good=10.0,
        threshold=8.0,
        target_power=0.8,
    )

    assert shape["expected_raw_correlation_sign"] == "negative"
    assert np.isclose(shape["spearman_severity_value"], -1.0)
    assert np.isclose(shape["spearman_severity_worsening_response"], 1.0)
    assert shape["directionally_consistent"]
    assert shape["justifies_inclusion"]
    assert shape["first_justified_point"]["directional_response"] == 3.0
