from __future__ import annotations

import numpy as np

from gauge_climate.inference import (
    Interval,
    dkw_w1_interval,
    practical_status,
    simultaneous_reference_boundaries,
)
from gauge_climate.types import PracticalStatus, SensitivityStatus


def test_dkw_interval_contains_known_bounded_shift():
    rng = np.random.default_rng(12)
    blocks = np.repeat(np.arange(120), 3)
    block_effect = rng.uniform(-0.2, 0.2, size=120)
    real = block_effect.repeat(3) + rng.uniform(-0.8, 0.8, size=blocks.size)
    shift = 0.4
    generated = (
        rng.uniform(-0.2, 0.2, size=300)
        + rng.uniform(-0.8, 0.8, size=300)
        + shift
    )
    interval = dkw_w1_interval(
        generated,
        real,
        blocks,
        n_eval=200,
        support=(-1.1, 1.5),
        alpha=0.05,
        seed=4,
    )
    assert interval.low <= shift <= interval.high


def test_dkw_interval_rejects_values_outside_declared_support():
    generated = np.asarray([0.0, 2.0, 0.5])
    real = np.asarray([0.0, 0.1, 0.2, 0.3])
    blocks = np.arange(real.size)
    try:
        dkw_w1_interval(
            generated,
            real,
            blocks,
            n_eval=3,
            support=(-1.0, 1.0),
            alpha=0.05,
            seed=1,
        )
    except ValueError as exc:
        assert "support" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("Expected a support violation.")


def test_tolerance_statuses_cover_within_exceeds_boundary_and_underpowered():
    def interval(low, high):
        return Interval((low + high) / 2, low, high, np.asarray([]))

    assert (
        practical_status(interval(0.1, 0.4), tolerance=0.5, shift_resolution=0.1)[0]
        is PracticalStatus.WITHIN_TOLERANCE
    )
    assert (
        practical_status(interval(0.6, 0.8), tolerance=0.5, shift_resolution=0.1)[0]
        is PracticalStatus.EXCEEDS_TOLERANCE
    )
    boundary = practical_status(
        interval(0.4, 0.6), tolerance=0.5, shift_resolution=0.1
    )
    assert boundary == (PracticalStatus.INCONCLUSIVE, SensitivityStatus.ADEQUATE)
    underpowered = practical_status(
        interval(0.4, 0.6), tolerance=0.5, shift_resolution=0.8
    )
    assert underpowered == (PracticalStatus.UNDERPOWERED, SensitivityStatus.INADEQUATE)


def test_simultaneous_reference_boundaries_are_cellwise_and_more_conservative():
    rng = np.random.default_rng(32)
    references = rng.gamma(shape=2.0, size=(5, 2_000))
    boundaries = simultaneous_reference_boundaries(references, alpha=0.05)
    assert boundaries.shape == (5,)
    assert np.all(boundaries >= 1.0)


def test_simultaneous_reference_boundaries_withhold_when_monte_carlo_is_too_small():
    references = np.arange(50 * 200, dtype=float).reshape(50, 200)
    boundaries = simultaneous_reference_boundaries(references, alpha=0.05)
    assert np.all(np.isnan(boundaries))


def test_simultaneous_reference_boundary_withholds_a_degenerate_cell():
    references = np.vstack([np.zeros(200), np.linspace(0.0, 1.0, 200)])
    boundaries = simultaneous_reference_boundaries(references, alpha=0.05)
    assert np.isnan(boundaries[0])
    assert np.isfinite(boundaries[1])
