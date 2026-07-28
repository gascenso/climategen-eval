from __future__ import annotations

import numpy as np

from gauge_climate.features import TrainFittedFeatureSpace
from gauge_climate.lanes import (
    additive_shift_lane,
    center_jitter_lane,
    gaussian_blur_lane,
    phase_shuffle,
    phase_shuffle_fields,
    run_perturbation_sweep,
)
from gauge_climate.memorization import localized_copying_test
from gauge_climate.metrics import (
    admissibility_metric,
    coverage_metric,
    density_coverage,
    density_metric,
    diversity_metric,
    functional_w1,
    mmd2_biased,
    spearman_pair_metrics,
    spearman_matrix_mae,
)
from gauge_climate.types import FunctionalSpec


def test_mmd_identical_and_shifted():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(100, 3))
    same = mmd2_biased(x, x.copy(), bandwidth=1.0)
    shifted = mmd2_biased(x + 2.0, x, bandwidth=1.0)
    assert same == 0.0
    assert shifted > same


def test_density_coverage_identical():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(80, 2))
    density, coverage = density_coverage(x, x.copy(), k=5)
    assert density > 0.5
    assert coverage == 1.0


def test_mode_collapse_and_excessive_diversity_are_both_penalized():
    rng = np.random.default_rng(21)
    real = rng.normal(size=(100, 3))
    collapsed = np.repeat(real[:1], 100, axis=0)
    excessive = real * 5.0
    metric = diversity_metric(lambda fields: fields)
    assert metric.estimator(collapsed, real) > 0.8
    assert metric.estimator(excessive, real) > 2.0
    _, collapsed_coverage = density_coverage(real, collapsed, k=5)
    assert collapsed_coverage < 0.2


def test_registry_metrics_have_zero_copy_discrepancy():
    values = np.arange(24, dtype=float).reshape(8, 3)
    identity = lambda x: x
    assert np.isfinite(density_metric(identity, k=2).estimator(values, values))
    assert coverage_metric(identity, k=2).estimator(values, values) == 0.0
    assert diversity_metric(identity).estimator(values, values) == 0.0
    rate = admissibility_metric(lambda x: x[:, 0] < 0)
    assert rate.estimator(rate.transform(values), rate.transform(values)) == 0.0


def test_train_fitted_features_are_deterministic():
    rng = np.random.default_rng(5)
    fields = rng.normal(size=(40, 4, 4))
    functional = FunctionalSpec("mean", lambda x: x.mean(axis=(1, 2)), "K", "mean")
    first = TrainFittedFeatureSpace.fit(fields, [functional], n_eof=3, seed=11)
    second = TrainFittedFeatureSpace.fit(fields, [functional], n_eof=3, seed=11)
    assert first.provenance_hash == second.provenance_hash
    assert np.allclose(first.transform(fields), second.transform(fields))
    registered = first.registered_metrics(fields, density_k=3)
    assert {metric.id for metric in registered} == {
        "feature_mmd2",
        "feature_density",
        "feature_coverage_deficit",
        "feature_diversity_departure",
    }
    assert all(metric.provenance["fit_source"] == "train" for metric in registered)
    assert all(
        metric.provenance["reference_draw_max_samples_per_side"] == 256 for metric in registered
    )


def test_registered_feature_metrics_accept_cached_training_features_and_cap_draws():
    rng = np.random.default_rng(52)
    fields = rng.normal(size=(80, 4, 4))
    functional = FunctionalSpec("mean", lambda x: x.mean(axis=(1, 2)), "K", "mean")
    fitted = TrainFittedFeatureSpace.fit(fields, [functional], n_eof=3, seed=11)
    train_features = fitted.transform(fields)
    registered = fitted.registered_metrics(
        train_features=train_features,
        density_k=3,
        max_samples_per_draw=16,
    )
    generated = train_features[:50] + 0.1
    real = train_features[20:70]
    assert all(np.isfinite(metric.estimator(generated, real)) for metric in registered)
    assert all(
        metric.provenance["reference_draw_max_samples_per_side"] == 16 for metric in registered
    )


def test_train_fitted_features_do_not_materialize_lazy_archive():
    class LazyArchive:
        def __init__(self, values):
            self.values = values
            self.shape = values.shape
            self.dtype = values.dtype

        def __getitem__(self, key):
            return self.values[key]

        def __array__(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("Full lazy archive was materialized.")

    rng = np.random.default_rng(51)
    fields = LazyArchive(rng.normal(size=(30, 4, 4)).astype(np.float32))
    functional = FunctionalSpec("mean", lambda x: x.mean(axis=(1, 2)), "K", "mean")
    fitted = TrainFittedFeatureSpace.fit(
        fields,
        [functional],
        n_eof=2,
        max_fit_samples=8,
        batch_size=5,
    )
    assert fitted.transform(fields).shape == (30, 3)


def test_feature_space_rejects_same_size_wrong_dimension_order():
    fields = np.zeros((10, 2, 3, 4))
    fitted = TrainFittedFeatureSpace.fit(fields, (), n_eof=2)
    try:
        fitted.transform(np.zeros((5, 3, 2, 4)))
    except ValueError as exc:
        assert "shape" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("Expected a dimension-order error.")


def test_spearman_closure_detects_scrambling():
    x = np.column_stack([np.arange(30), np.arange(30) ** 2])
    y = x.copy()
    y[:, 1] = y[::-1, 1]
    assert spearman_matrix_mae(x, x) == 0.0
    assert spearman_matrix_mae(y, x) > 0.5
    pair = spearman_pair_metrics(lambda fields: fields, ("linear", "quadratic"))[0]
    assert pair.estimator(x, x) == 0.0
    assert pair.estimator(y, x) > 1.0


def test_copyist_positive_control_triggers_localized_test():
    rng = np.random.default_rng(3)
    train = rng.normal(size=(120, 4))
    blocks = np.repeat(np.arange(30), 3)
    real = rng.normal(size=(90, 4))
    generated = train[:90].copy()
    result = localized_copying_test(
        train_features=train,
        real_features=real,
        generated_features=generated,
        real_blocks=blocks,
        n_regions=4,
        n_resamples=200,
        min_real_blocks=3,
        seed=7,
    )
    assert result["status"] == "LOCALIZED_CLOSENESS_OBSERVED"


def test_copying_test_matches_queries_within_supplied_strata():
    rng = np.random.default_rng(31)
    train = rng.normal(size=(160, 4))
    real = rng.normal(size=(120, 4))
    blocks = np.repeat(np.arange(40), 3)
    real_strata = np.repeat(["common", "rare"], [90, 30])
    generated_strata = np.repeat(["common", "rare"], [40, 80])
    generated = rng.normal(size=(120, 4))
    generated[:40] = train[:40]
    result = localized_copying_test(
        train_features=train,
        real_features=real,
        generated_features=generated,
        real_blocks=blocks,
        real_strata=real_strata,
        generated_strata=generated_strata,
        n_regions=1,
        n_resamples=200,
        min_real_blocks=3,
        seed=9,
    )
    assert result["query_stratification"] == "supplied_labels"
    assert result["n_supported_cells"] == 2
    assert {item["stratum"] for item in result["regions"]} == {"common", "rare"}
    common = next(item for item in result["regions"] if item["stratum"] == "common")
    assert common["status"] == "CLOSER_TO_TRAINING"


def test_localized_test_reports_clear_farther_from_training_generation():
    rng = np.random.default_rng(33)
    train = rng.normal(size=(120, 3))
    real = rng.normal(size=(90, 3))
    blocks = np.repeat(np.arange(30), 3)
    generated = rng.normal(size=(90, 3)) + 15.0
    result = localized_copying_test(
        train_features=train,
        real_features=real,
        generated_features=generated,
        real_blocks=blocks,
        n_regions=1,
        n_resamples=200,
        min_real_blocks=5,
        seed=4,
    )
    assert result["status"] == "FARTHER_FROM_TRAINING"


def test_localized_test_reports_insufficient_support_without_power_language():
    rng = np.random.default_rng(34)
    train = rng.normal(size=(40, 3))
    real = rng.normal(size=(8, 3))
    generated = rng.normal(size=(8, 3))
    blocks = np.repeat(np.arange(4), 2)
    result = localized_copying_test(
        train_features=train,
        real_features=real,
        generated_features=generated,
        real_blocks=blocks,
        n_regions=2,
        n_resamples=20,
        min_real_blocks=5,
        seed=4,
    )
    assert result["status"] == "INSUFFICIENT_LOCAL_SUPPORT"
    assert result["n_supported_cells"] == 0


def test_predeclared_sensitivity_lane_has_monotone_response():
    source = np.zeros((20, 2, 2))
    reference = np.zeros((20, 2, 2))
    functional = FunctionalSpec(
        "mean", lambda fields: fields.mean(axis=(1, 2)), "K", "mean field value"
    )
    metric = functional_w1(functional)
    lane = additive_shift_lane(severities=(0.0, 0.5, 1.0), expected_metrics=(metric.id,))
    result = run_perturbation_sweep(
        lane,
        source_fields=source,
        reference_fields=reference,
        metrics=(metric,),
        reference_boundaries={metric.id: 0.1},
    )
    assert result["passed"] is True
    assert result["metrics"][metric.id]["values"] == [0.0, 0.5, 1.0]
    assert result["metrics"][metric.id]["first_reference_crossing_severity"] == 0.5


def test_sweep_separates_response_validity_from_detection_resolution():
    source = np.zeros((20, 2, 2))
    functional = FunctionalSpec(
        "mean", lambda fields: fields.mean(axis=(1, 2)), "K", "mean field value"
    )
    metric = functional_w1(functional)
    lane = additive_shift_lane(severities=(0.0, 0.1, 0.2), expected_metrics=(metric.id,))
    result = run_perturbation_sweep(
        lane,
        source_fields=source,
        reference_fields=source,
        metrics=(metric,),
        reference_boundaries={metric.id: 1.0},
    )
    cell = result["metrics"][metric.id]
    assert result["passed"] is True
    assert cell["response_contract_passed"] is True
    assert cell["first_reference_crossing_severity"] is None
    assert cell["resolution_status"] == "REMAINS_WITHIN_REFERENCE_AT_MAX_SEVERITY"


def test_phase_shuffle_preserves_each_discrete_power_spectrum():
    rng = np.random.default_rng(18)
    fields = rng.normal(size=(3, 16, 18))
    shuffled = phase_shuffle(fields, n=3, seed=9)
    # Reconstruct the selected source order used internally.
    selected = fields[np.random.default_rng(9).choice(3, 3, replace=True)]
    original_power = np.abs(np.fft.fft2(selected, axes=(-2, -1))) ** 2
    shuffled_power = np.abs(np.fft.fft2(shuffled, axes=(-2, -1))) ** 2
    assert np.allclose(original_power, shuffled_power, rtol=1e-5, atol=1e-5)


def test_phase_shuffle_fields_preserves_order_and_power():
    rng = np.random.default_rng(28)
    fields = rng.normal(size=(4, 12, 14))
    shuffled = phase_shuffle_fields(fields, seed=19)
    original_power = np.abs(np.fft.fft2(fields, axes=(-2, -1))) ** 2
    shuffled_power = np.abs(np.fft.fft2(shuffled, axes=(-2, -1))) ** 2
    assert shuffled.shape == fields.shape
    assert np.allclose(original_power, shuffled_power, rtol=1e-5, atol=1e-5)


def test_spatial_lanes_are_identity_at_zero_and_paired_across_severity():
    fields = np.zeros((3, 10, 10), dtype=np.float32)
    fields[:, 3:7, 4:6] = 1.0
    blur = gaussian_blur_lane(severities=(0.0, 0.5, 1.0), expected_metrics=("metric",))
    jitter = center_jitter_lane(severities=(0.0, 1.0, 2.0), expected_metrics=("metric",))
    assert np.array_equal(blur.apply(fields, 0.0, 7), fields)
    assert np.array_equal(jitter.apply(fields, 0.0, 7), fields)
    assert not np.array_equal(blur.apply(fields, 1.0, 7), fields)
    first = jitter.apply(fields, 2.0, 7)
    second = jitter.apply(fields, 2.0, 7)
    assert np.array_equal(first, second)
