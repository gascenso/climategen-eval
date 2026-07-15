from __future__ import annotations

import numpy as np

from gauge_climate.adapters.tc_bt import TCBTConfig, curve_metrics, functionals


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
