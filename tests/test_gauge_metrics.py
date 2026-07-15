from __future__ import annotations

import numpy as np

from gauge_climate.features import TrainFittedFeatureSpace
from gauge_climate.lanes import additive_shift_lane, phase_shuffle, run_perturbation_sweep
from gauge_climate.memorization import localized_copying_test
from gauge_climate.metrics import (
    admissibility_metric,
    coverage_metric,
    density_coverage,
    density_metric,
    diversity_metric,
    functional_w1,
    mmd2_unbiased,
    spearman_pair_metrics,
    spearman_matrix_mae,
)
from gauge_climate.types import FunctionalSpec


def test_mmd_identical_and_shifted():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(100, 3))
    same = mmd2_unbiased(x, x.copy(), bandwidth=1.0)
    shifted = mmd2_unbiased(x + 2.0, x, bandwidth=1.0)
    assert same < 0.01
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
    assert result["status"] == "COPYING_DETECTED"


def test_localized_test_flags_clear_off_manifold_generation():
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
    assert result["status"] == "OFF_MANIFOLD"


def test_predeclared_sensitivity_lane_has_monotone_response():
    source = np.zeros((20, 2, 2))
    reference = np.zeros((20, 2, 2))
    functional = FunctionalSpec(
        "mean", lambda fields: fields.mean(axis=(1, 2)), "K", "mean field value"
    )
    metric = functional_w1(functional)
    lane = additive_shift_lane(
        severities=(0.0, 0.5, 1.0), expected_metrics=(metric.id,)
    )
    result = run_perturbation_sweep(
        lane,
        source_fields=source,
        reference_fields=reference,
        metrics=(metric,),
        reference_boundaries={metric.id: 0.1},
    )
    assert result["passed"] is True
    assert result["metrics"][metric.id]["values"] == [0.0, 0.5, 1.0]


def test_phase_shuffle_preserves_each_discrete_power_spectrum():
    rng = np.random.default_rng(18)
    fields = rng.normal(size=(3, 16, 18))
    shuffled = phase_shuffle(fields, n=3, seed=9)
    # Reconstruct the selected source order used internally.
    selected = fields[np.random.default_rng(9).choice(3, 3, replace=True)]
    original_power = np.abs(np.fft.fft2(selected, axes=(-2, -1))) ** 2
    shuffled_power = np.abs(np.fft.fft2(shuffled, axes=(-2, -1))) ** 2
    assert np.allclose(original_power, shuffled_power, rtol=1e-5, atol=1e-5)
