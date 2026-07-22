from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .metrics import wasserstein_1d
from .reference import ReferencePlan
from .types import DetectionStatus


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


def detection_status(model_value: float, reference_boundary: float) -> DetectionStatus:
    if model_value > reference_boundary:
        return DetectionStatus.DEPARTURE_DETECTED
    return DetectionStatus.NO_DEPARTURE_DETECTED


def shift_resolution(
    *,
    real_values: np.ndarray,
    plan: ReferencePlan,
    reference_boundary: float,
    target_power: float,
    grid_size: int,
    required_shifts: np.ndarray | None = None,
    max_draws: int | None = None,
) -> tuple[float | None, np.ndarray, np.ndarray]:
    real_values = np.asarray(real_values, dtype=np.float64).reshape(-1)
    spread = float(np.std(real_values))
    if not np.isfinite(spread) or spread <= np.finfo(float).eps:
        return None, np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    shifts = np.linspace(0.0, 4.0 * spread, grid_size)
    if required_shifts is not None:
        required = np.asarray(required_shifts, dtype=np.float64).reshape(-1)
        if np.any(~np.isfinite(required)) or np.any(required < 0.0):
            raise ValueError("required_shifts must contain finite non-negative values.")
        # Explicit diagnostic severities must be evaluated exactly rather than
        # approximated by the nearest point on the plotting grid.
        shifts = np.unique(np.concatenate((shifts, required)))
    powers = np.empty(shifts.size, dtype=np.float64)
    all_draws = plan.draws
    if not all_draws:
        raise ValueError("At least one sensitivity draw is required.")
    rng = np.random.default_rng(plan.seed + 7919)
    if max_draws is None:
        draws = all_draws
    else:
        requested = int(max_draws)
        if requested < 1:
            raise ValueError("max_draws must be positive when supplied.")
        count = min(requested, len(all_draws))
        selected = np.sort(rng.choice(len(all_draws), size=count, replace=False))
        draws = tuple(all_draws[int(index)] for index in selected)
    bootstrap_replicates = len(draws)
    for shift_index, shift in enumerate(shifts):
        draw_values = np.asarray(
            [
                wasserstein_1d(
                    real_values[draw.pseudo_indices] + shift,
                    real_values[draw.real_indices],
                )
                for draw in draws
            ],
            dtype=np.float64,
        )
        hits = 0
        for _ in range(bootstrap_replicates):
            statistic = float(
                np.median(rng.choice(draw_values, size=draw_values.size, replace=True))
            )
            hits += int(statistic > reference_boundary)
        powers[shift_index] = hits / bootstrap_replicates
    powers = np.maximum.accumulate(powers)
    passing = np.flatnonzero(powers >= target_power)
    resolution = None if passing.size == 0 else float(shifts[int(passing[0])])
    return resolution, shifts, powers


def simultaneous_reference_boundaries(reference_matrix: np.ndarray, *, alpha: float) -> np.ndarray:
    """Bonferroni reference-position boundaries for a registered cell family.

    Conditional strata need not admit a scientifically valid shared block
    partition.  Cell-wise Bonferroni boundaries therefore provide the v0.1
    simultaneous claim without inventing a dependence coupling between their
    Monte Carlo draws.  A less conservative max-statistic method may be
    registered when all cells genuinely share resampling units and draws.
    """

    reference_matrix = np.asarray(reference_matrix, dtype=np.float64)
    if reference_matrix.ndim != 2:
        raise ValueError("reference_matrix must have shape (cells, resamples).")
    if reference_matrix.shape[0] == 0 or reference_matrix.shape[1] < 2:
        raise ValueError("reference_matrix must contain cells and at least two resamples.")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5).")
    family_size = reference_matrix.shape[0]
    adjusted_alpha = alpha / family_size
    n_resamples = reference_matrix.shape[1]
    # Under exchangeability of the reference draws and one future null
    # statistic, this order statistic has exceedance probability at most the
    # adjusted alpha.  Linear interpolation between order statistics does not.
    rank = math.ceil((n_resamples + 1) * (1.0 - adjusted_alpha))
    if rank > n_resamples:
        return np.full(family_size, np.nan, dtype=np.float64)
    boundaries = np.empty(family_size, dtype=np.float64)
    for index, values in enumerate(reference_matrix):
        q50 = float(np.quantile(values, 0.50))
        pointwise = float(np.quantile(values, 1.0 - alpha))
        simultaneous = float(np.sort(values)[rank - 1])
        width = pointwise - q50
        boundaries[index] = np.nan if width <= np.finfo(float).eps else (simultaneous - q50) / width
    return boundaries
