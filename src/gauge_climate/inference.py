from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t

from .metrics import wasserstein_1d
from .reference import ReferencePlan
from .types import (
    DetectionStatus,
    PracticalStatus,
    SamplingEstimand,
    SensitivityStatus,
)


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    bootstrap_values: np.ndarray
    method: str = "unspecified"
    standard_error: float | None = None
    degrees_of_freedom: float | None = None
    cdf_zero_fraction: float | None = None


def _empirical_quantile_with_support(
    sample: np.ndarray,
    probability: float,
    support: tuple[float, float],
) -> float:
    if probability <= 0:
        return float(support[0])
    if probability >= 1:
        return float(support[1])
    return float(np.quantile(sample, probability, method="inverted_cdf"))


def dkw_w1_interval(
    generated: np.ndarray,
    real: np.ndarray,
    real_blocks: np.ndarray,
    *,
    n_eval: int,
    support: tuple[float, float],
    alpha: float,
    seed: int,
) -> Interval:
    """Finite-sample DKW interval for block-balanced one-dimensional W1.

    One frame is sampled uniformly from each independent real block, yielding
    an i.i.d. sample from the block-balanced target. Quantile-function bands
    follow Manole, Balakrishnan & Wasserman (2022), Proposition 1. Known finite
    support extends the interval to the untrimmed W1 distance.
    """

    generated = np.asarray(generated, dtype=np.float64).reshape(-1)
    real = np.asarray(real, dtype=np.float64).reshape(-1)
    blocks = np.asarray([str(value) for value in np.asarray(real_blocks)], dtype=object)
    if real.size != blocks.size:
        raise ValueError("real and real_blocks must align.")
    if generated.size < n_eval:
        raise ValueError("Generated functional pool is smaller than n_eval.")
    lower, upper = float(support[0]), float(support[1])
    if np.any(generated < lower) or np.any(generated > upper):
        raise ValueError("Generated functional values fall outside the declared support.")
    if np.any(real < lower) or np.any(real > upper):
        raise ValueError("Real functional values fall outside the declared support.")
    rng = np.random.default_rng(seed)
    x = generated[rng.choice(generated.size, size=n_eval, replace=False)]
    unique_blocks = np.asarray(list(dict.fromkeys(blocks.tolist())), dtype=object)
    y = np.asarray(
        [real[int(rng.choice(np.flatnonzero(blocks == block)))] for block in unique_blocks],
        dtype=np.float64,
    )
    n, m = x.size, y.size
    beta_n = float(np.sqrt(np.log(4.0 / alpha) / (2.0 * n)))
    beta_m = float(np.sqrt(np.log(4.0 / alpha) / (2.0 * m)))
    breakpoints = {0.0, 1.0}
    for size, beta in ((n, beta_n), (m, beta_m)):
        for rank in range(size + 1):
            base = rank / size
            breakpoints.add(float(np.clip(base - beta, 0.0, 1.0)))
            breakpoints.add(float(np.clip(base + beta, 0.0, 1.0)))
    ordered = np.asarray(sorted(breakpoints), dtype=np.float64)
    widths = np.diff(ordered)
    midpoints = ordered[:-1] + widths / 2.0
    lower_integral = 0.0
    upper_integral = 0.0
    for u, width in zip(midpoints, widths, strict=True):
        x_lo = _empirical_quantile_with_support(x, u - beta_n, support)
        x_hi = _empirical_quantile_with_support(x, u + beta_n, support)
        y_lo = _empirical_quantile_with_support(y, u - beta_m, support)
        y_hi = _empirical_quantile_with_support(y, u + beta_m, support)
        point_low = max(x_lo - y_hi, y_lo - x_hi, 0.0)
        point_high = max(x_hi - y_lo, y_hi - x_lo)
        lower_integral += point_low * width
        upper_integral += point_high * width
    estimate = wasserstein_1d(x, y)
    return Interval(
        estimate=float(estimate),
        low=float(max(0.0, lower_integral)),
        high=float(max(0.0, upper_integral)),
        bootstrap_values=np.asarray([], dtype=np.float64),
        method="finite_sample_dkw_quantile_band",
    )


def _real_weights(blocks: np.ndarray, estimand: SamplingEstimand) -> np.ndarray:
    blocks = np.asarray([str(value) for value in np.asarray(blocks)], dtype=object)
    if estimand is SamplingEstimand.FRAME_WEIGHTED:
        return np.full(blocks.size, 1.0 / blocks.size, dtype=np.float64)
    unique, counts = np.unique(blocks, return_counts=True)
    count_by_block = {str(block): int(count) for block, count in zip(unique, counts, strict=True)}
    weight = np.asarray(
        [1.0 / (len(unique) * count_by_block[str(block)]) for block in blocks],
        dtype=np.float64,
    )
    return weight / weight.sum()


def _cluster_bootstrap_weights(
    blocks: np.ndarray,
    estimand: SamplingEstimand,
    rng: np.random.Generator,
) -> np.ndarray:
    blocks = np.asarray([str(value) for value in np.asarray(blocks)], dtype=object)
    unique = np.asarray(list(dict.fromkeys(blocks.tolist())), dtype=object)
    selected = rng.choice(unique, size=unique.size, replace=True)
    selected_counts = {str(block): int(np.sum(selected == block)) for block in unique}
    frame_counts = {str(block): int(np.sum(blocks == block)) for block in unique}
    if estimand is SamplingEstimand.BLOCK_BALANCED:
        weights = np.asarray(
            [
                selected_counts[str(block)] / (unique.size * frame_counts[str(block)])
                for block in blocks
            ],
            dtype=np.float64,
        )
    else:
        denominator = sum(
            selected_counts[str(block)] * frame_counts[str(block)] for block in unique
        )
        weights = np.asarray(
            [selected_counts[str(block)] / denominator for block in blocks],
            dtype=np.float64,
        )
    if weights.sum() <= 0:
        raise RuntimeError("Cluster bootstrap produced zero total weight.")
    return weights / weights.sum()


def functional_w1_interval(
    generated: np.ndarray,
    real: np.ndarray,
    real_blocks: np.ndarray,
    *,
    n_eval: int,
    estimand: SamplingEstimand,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> Interval:
    generated = np.asarray(generated, dtype=np.float64).reshape(-1)
    real = np.asarray(real, dtype=np.float64).reshape(-1)
    real_blocks = np.asarray(real_blocks)
    if generated.size < n_eval:
        raise ValueError("Generated functional pool is smaller than n_eval.")
    if real.size != real_blocks.size:
        raise ValueError("real and real_blocks must align.")
    rng = np.random.default_rng(seed)
    fixed_indices = rng.choice(generated.size, size=n_eval, replace=False)
    fixed_generated = generated[fixed_indices]
    estimate = wasserstein_1d(
        fixed_generated,
        real,
        y_weights=_real_weights(real_blocks, estimand),
    )
    boot = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        generated_boot = fixed_generated[rng.choice(n_eval, size=n_eval, replace=True)]
        weights = _cluster_bootstrap_weights(real_blocks, estimand, rng)
        boot[index] = wasserstein_1d(generated_boot, real, y_weights=weights)
    tail = (1.0 - confidence) / 2.0
    return Interval(
        estimate=float(estimate),
        low=float(np.quantile(boot, tail)),
        high=float(np.quantile(boot, 1.0 - tail)),
        bootstrap_values=boot,
        method="cluster_percentile_bootstrap_diagnostic",
    )


def cluster_w1_influence_interval(
    generated: np.ndarray,
    real: np.ndarray,
    real_blocks: np.ndarray,
    *,
    n_eval: int,
    alpha: float,
    seed: int,
    fallback_support: tuple[float, float] | None = None,
) -> Interval:
    """Cluster-aware confidence interval for scalar W1.

    In one dimension, W1 is the integrated absolute difference between two
    CDFs.  The derivative away from equality is therefore an integral against
    the sign of that CDF difference.  Real-data influence values are computed
    for whole blocks; generated samples are treated as mutually independent.
    A Welch--Satterthwaite t critical value accounts for the two variance
    components.  The resulting interval has asymptotic confidence
    ``1 - alpha``; using its upper endpoint for equivalence is therefore
    conservative relative to a standalone one-sided alpha-level test.

    The method targets the block-balanced distribution and uses every frame to
    estimate each block's CDF without treating within-block frames as
    independent population replicates.  If the empirical influence variance is
    degenerate, the function falls back to the finite-sample DKW interval when
    a support is supplied.
    """

    generated = np.asarray(generated, dtype=np.float64).reshape(-1)
    real = np.asarray(real, dtype=np.float64).reshape(-1)
    blocks = np.asarray([str(value) for value in np.asarray(real_blocks)], dtype=object)
    if real.size != blocks.size:
        raise ValueError("real and real_blocks must align.")
    if generated.size < n_eval:
        raise ValueError("Generated functional pool is smaller than n_eval.")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be between zero and one half.")
    if not np.all(np.isfinite(generated)) or not np.all(np.isfinite(real)):
        raise ValueError("W1 influence inference requires finite functional values.")

    rng = np.random.default_rng(seed)
    generated_sample = generated[
        rng.choice(generated.size, size=n_eval, replace=False)
    ]
    unique_blocks, block_inverse = np.unique(blocks, return_inverse=True)
    n_blocks = unique_blocks.size
    if n_blocks < 2:
        raise ValueError("Cluster W1 inference requires at least two real blocks.")

    real_weights = _real_weights(blocks, SamplingEstimand.BLOCK_BALANCED)
    pooled = np.unique(np.concatenate((real, generated_sample)))
    if pooled.size < 2:
        if fallback_support is None:
            raise ValueError(
                "Degenerate W1 influence inference requires a finite fallback support."
            )
        return dkw_w1_interval(
            generated_sample,
            real,
            blocks,
            n_eval=n_eval,
            support=fallback_support,
            alpha=alpha,
            seed=seed,
        )

    left = pooled[:-1]
    widths = np.diff(pooled)

    def cdf(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        cumulative = np.cumsum(weights[order])
        positions = np.searchsorted(sorted_values, left, side="right") - 1
        result = np.zeros(left.size, dtype=np.float64)
        present = positions >= 0
        result[present] = cumulative[positions[present]]
        return result

    real_cdf = cdf(real, real_weights)
    generated_weights = np.full(
        generated_sample.size, 1.0 / generated_sample.size, dtype=np.float64
    )
    generated_cdf = cdf(generated_sample, generated_weights)
    difference = real_cdf - generated_cdf
    signs = np.sign(difference)
    estimate = float(np.sum(np.abs(difference) * widths))

    signed_widths = signs * widths
    area_from_value = np.concatenate(
        (np.cumsum(signed_widths[::-1])[::-1], np.asarray([0.0]))
    )
    real_positions = np.searchsorted(pooled, real)
    generated_positions = np.searchsorted(pooled, generated_sample)
    real_scores = area_from_value[real_positions]
    generated_scores = -area_from_value[generated_positions]

    block_sums = np.bincount(
        block_inverse, weights=real_scores, minlength=n_blocks
    )
    block_counts = np.bincount(block_inverse, minlength=n_blocks)
    block_influence = block_sums / block_counts
    block_influence -= np.mean(block_influence)
    generated_influence = generated_scores - np.mean(generated_scores)

    real_component = float(np.var(block_influence, ddof=1) / n_blocks)
    generated_component = float(
        np.var(generated_influence, ddof=1) / generated_sample.size
    )
    variance = real_component + generated_component
    zero_fraction = float(
        np.sum(widths[difference == 0.0]) / np.sum(widths)
    )
    if not np.isfinite(variance) or variance <= np.finfo(float).eps:
        if fallback_support is None:
            raise ValueError(
                "Degenerate W1 influence variance requires a finite fallback support."
            )
        return dkw_w1_interval(
            generated_sample,
            real,
            blocks,
            n_eval=n_eval,
            support=fallback_support,
            alpha=alpha,
            seed=seed,
        )

    denominator = 0.0
    if real_component > 0.0:
        denominator += real_component**2 / (n_blocks - 1)
    if generated_component > 0.0:
        denominator += generated_component**2 / (generated_sample.size - 1)
    degrees_of_freedom = variance**2 / denominator
    standard_error = math.sqrt(variance)
    critical = float(student_t.ppf(1.0 - alpha / 2.0, degrees_of_freedom))
    margin = critical * standard_error
    return Interval(
        estimate=estimate,
        low=float(max(0.0, estimate - margin)),
        high=float(estimate + margin),
        bootstrap_values=np.concatenate((block_influence, generated_influence)),
        method="cluster_w1_influence_t",
        standard_error=standard_error,
        degrees_of_freedom=float(degrees_of_freedom),
        cdf_zero_fraction=zero_fraction,
    )


def detection_status(model_value: float, reference_boundary: float) -> DetectionStatus:
    if model_value > reference_boundary:
        return DetectionStatus.DEPARTURE_DETECTED
    return DetectionStatus.NO_DEPARTURE_DETECTED


def practical_status(
    interval: Interval,
    *,
    tolerance: float | None,
    shift_resolution: float | None,
) -> tuple[PracticalStatus, SensitivityStatus]:
    if tolerance is None:
        return PracticalStatus.NOT_APPLICABLE, SensitivityStatus.NOT_APPLICABLE
    sensitivity = (
        SensitivityStatus.NOT_EVALUATED
        if shift_resolution is None
        else (
            SensitivityStatus.ADEQUATE
            if shift_resolution <= tolerance
            else SensitivityStatus.INADEQUATE
        )
    )
    if interval.high < tolerance:
        return PracticalStatus.WITHIN_TOLERANCE, sensitivity
    if interval.low > tolerance:
        return PracticalStatus.EXCEEDS_TOLERANCE, sensitivity
    uncertainty_margin = max(interval.high - interval.estimate, 0.0)
    if uncertainty_margin >= tolerance:
        return PracticalStatus.UNDERPOWERED, sensitivity
    return PracticalStatus.INCONCLUSIVE, sensitivity


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
        # Sensitivity decisions at an application tolerance must be based on
        # power evaluated at that exact tolerance, not on the nearest point of
        # an otherwise arbitrary plotting grid.
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
