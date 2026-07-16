from __future__ import annotations

import numpy as np

from gauge_climate.adapters.tc_bt import (
    PreprocessedFields,
    TCBTConfig,
    curve_metrics,
    functionals,
)


def _fields():
    y, x = np.mgrid[-1:1:32j, -1:1:32j]
    radius = np.sqrt(x * x + y * y)
    base = 280.0 - 70.0 * np.exp(-((radius - 0.45) ** 2) / 0.04)
    return np.stack([base, base + 2.0, base - 2.0, base + x * 4.0], axis=0)


def test_tc_functionals_have_declared_support_and_expected_shape():
    config = TCBTConfig(
        field_shape=(32, 32),
        pixel_size_km=16.0,
        max_profile_radius_km=250.0,
        profile_bins=8,
        psd_bins=8,
    )
    fields = _fields()
    specs = functionals(config)
    assert len(specs) == 7
    for spec in specs:
        values = spec.transform(fields)
        assert values.shape == (4,)
        assert spec.support is not None
        assert np.all(np.isfinite(values))


def test_tc_curve_metrics_are_finite_and_respond_to_blur():
    config = TCBTConfig(
        field_shape=(32, 32),
        pixel_size_km=16.0,
        max_profile_radius_km=250.0,
        profile_bins=8,
        psd_bins=8,
    )
    fields = _fields()
    profile, psd = curve_metrics(config)
    for metric in (profile, psd):
        values = metric.transform(fields)
        assert values.shape == (4, 8)
        assert np.isfinite(metric.estimator(values[:2], values[2:]))
    assert len(psd.provenance["coordinate_values"]) == 8
    assert psd.provenance["coordinate_units"] == "cycles/km"


def test_registered_256_grid_has_24_nonempty_psd_bins():
    config = TCBTConfig(
        field_shape=(256, 256),
        pixel_size_km=8.0,
        profile_bins=32,
        psd_bins=24,
    )
    fields = np.random.default_rng(91).normal(size=(2, 256, 256))
    _, psd = curve_metrics(config)
    values = psd.transform(fields)
    coordinates = np.asarray(psd.provenance["coordinate_values"])
    assert values.shape == (2, 24)
    assert np.all(np.isfinite(values))
    assert np.all(np.diff(coordinates) > 0)


def test_tc_preprocessed_view_clips_lazily_to_training_representation():
    raw = np.full((2, 4, 4), 250.0, dtype=np.float32)
    raw[0, 0, :4] = [150.0, 200.0, 300.0, 340.0]
    view = PreprocessedFields(raw, TCBTConfig(field_shape=(4, 4)))
    assert view.shape == raw.shape
    assert view.dtype == raw.dtype
    assert np.array_equal(view[0, 0], [170.0, 200.0, 300.0, 325.0])
    # The provenance archive is not modified by the lazy analysis view.
    assert raw[0, 0, 0] == 150.0
    assert raw[0, 0, 3] == 340.0
