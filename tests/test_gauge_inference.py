from __future__ import annotations

import numpy as np

from gauge_climate.inference import (
    shift_resolution,
    simultaneous_reference_boundaries,
)
from gauge_climate.reference import ReferenceDraw, ReferencePlan
from gauge_climate.types import SamplingEstimand


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


def test_shift_resolution_max_draws_limits_distance_evaluations(monkeypatch):
    draw = ReferenceDraw(
        pseudo_indices=np.arange(10),
        real_indices=np.arange(10, 20),
        generated_indices=np.arange(10),
        blocks_a=tuple(f"a{i}" for i in range(10)),
        blocks_b=tuple(f"b{i}" for i in range(10)),
        frame_count_a=10,
        frame_count_b=10,
    )
    plan = ReferencePlan(
        draws=tuple(draw for _ in range(30)),
        n_eval=10,
        estimand=SamplingEstimand.BLOCK_BALANCED,
        seed=17,
        n_blocks=20,
    )
    calls = 0

    def counted_w1(left, right):
        nonlocal calls
        calls += 1
        return float(abs(np.mean(left) - np.mean(right)))

    monkeypatch.setattr("gauge_climate.inference.wasserstein_1d", counted_w1)
    shift_resolution(
        real_values=np.linspace(0.0, 1.0, 20),
        plan=plan,
        reference_boundary=0.1,
        target_power=0.8,
        grid_size=8,
        max_draws=5,
    )
    assert calls == 8 * 5


def test_shift_resolution_evaluates_required_shift_exactly():
    draw = ReferenceDraw(
        pseudo_indices=np.arange(10),
        real_indices=np.arange(10, 20),
        generated_indices=np.arange(10),
        blocks_a=tuple(f"a{i}" for i in range(10)),
        blocks_b=tuple(f"b{i}" for i in range(10)),
        frame_count_a=10,
        frame_count_b=10,
    )
    plan = ReferencePlan(
        draws=tuple(draw for _ in range(8)),
        n_eval=10,
        estimand=SamplingEstimand.BLOCK_BALANCED,
        seed=17,
        n_blocks=20,
    )
    requested_shift = 0.123456789

    _, shifts, powers = shift_resolution(
        real_values=np.linspace(0.0, 1.0, 20),
        plan=plan,
        reference_boundary=0.1,
        target_power=0.8,
        grid_size=5,
        required_shifts=np.asarray([requested_shift]),
        max_draws=4,
    )

    matches = np.flatnonzero(shifts == requested_shift)
    assert matches.size == 1
    assert np.isfinite(powers[int(matches[0])])
