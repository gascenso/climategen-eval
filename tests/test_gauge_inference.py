from __future__ import annotations

import numpy as np

from gauge_climate.inference import (
    Interval,
    cluster_w1_influence_interval,
    dkw_w1_interval,
    practical_status,
    shift_resolution,
    simultaneous_reference_boundaries,
)
from gauge_climate.reference import ReferenceDraw, ReferencePlan
from gauge_climate.types import SamplingEstimand
from gauge_climate.types import PracticalStatus, SensitivityStatus


def test_dkw_interval_contains_known_bounded_shift():
    rng = np.random.default_rng(12)
    blocks = np.repeat(np.arange(120), 3)
    block_effect = rng.uniform(-0.2, 0.2, size=120)
    real = block_effect.repeat(3) + rng.uniform(-0.8, 0.8, size=blocks.size)
    shift = 0.4
    generated = rng.uniform(-0.2, 0.2, size=300) + rng.uniform(-0.8, 0.8, size=300) + shift
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


def test_cluster_w1_influence_interval_is_invariant_to_within_block_duplication():
    rng = np.random.default_rng(91)
    blocks = np.repeat(np.arange(40), 3)
    real = rng.normal(size=blocks.size)
    generated = rng.normal(loc=0.3, size=160)
    first = cluster_w1_influence_interval(
        generated,
        real,
        blocks,
        n_eval=120,
        alpha=0.05,
        seed=4,
        fallback_support=(-5.0, 5.0),
    )
    second = cluster_w1_influence_interval(
        generated,
        np.repeat(real, 2),
        np.repeat(blocks, 2),
        n_eval=120,
        alpha=0.05,
        seed=4,
        fallback_support=(-5.0, 5.0),
    )

    assert first.method == "cluster_w1_influence_t"
    assert np.isclose(first.estimate, second.estimate)
    assert np.isclose(first.standard_error, second.standard_error)
    assert np.isclose(first.low, second.low)
    assert np.isclose(first.high, second.high)


def test_cluster_w1_influence_interval_contains_clear_location_shift():
    rng = np.random.default_rng(92)
    block_effects = rng.normal(scale=0.25, size=120)
    blocks = np.repeat(np.arange(120), 4)
    real = block_effects.repeat(4) + rng.normal(size=blocks.size)
    shift = 0.4
    generated = rng.normal(scale=np.sqrt(1.0 + 0.25**2), size=800) + shift
    interval = cluster_w1_influence_interval(
        generated,
        real,
        blocks,
        n_eval=600,
        alpha=0.05,
        seed=5,
        fallback_support=(-5.0, 5.0),
    )

    assert interval.low <= shift <= interval.high
    assert interval.standard_error is not None and interval.standard_error > 0.0
    assert interval.degrees_of_freedom is not None


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
    boundary = practical_status(interval(0.4, 0.6), tolerance=0.5, shift_resolution=0.1)
    assert boundary == (PracticalStatus.INCONCLUSIVE, SensitivityStatus.ADEQUATE)
    underpowered = practical_status(interval(0.0, 1.2), tolerance=0.5, shift_resolution=0.8)
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
    tolerance = 0.123456789

    _, shifts, powers = shift_resolution(
        real_values=np.linspace(0.0, 1.0, 20),
        plan=plan,
        reference_boundary=0.1,
        target_power=0.8,
        grid_size=5,
        required_shifts=np.asarray([tolerance]),
        max_draws=4,
    )

    matches = np.flatnonzero(shifts == tolerance)
    assert matches.size == 1
    assert np.isfinite(powers[int(matches[0])])
