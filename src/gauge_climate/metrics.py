from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr, wasserstein_distance

from .types import FunctionalSpec, MetricKind, MetricSpec


class MetricNotEvaluable(ValueError):
    """The metric is mathematically undefined for the supplied finite sample."""


def wasserstein_1d(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_weights: np.ndarray | None = None,
    y_weights: np.ndarray | None = None,
) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0:
        raise ValueError("W1 requires non-empty samples.")
    return float(wasserstein_distance(x, y, u_weights=x_weights, v_weights=y_weights))


def functional_w1(functional: FunctionalSpec) -> MetricSpec:
    return MetricSpec(
        id=f"w1:{functional.id}",
        transform=functional.transform,
        estimator=lambda x, y: wasserstein_1d(x, y),
        kind=MetricKind.FUNCTIONAL_W1,
        units=functional.units,
        description=f"Wasserstein-1 distance for {functional.description}",
        functional=functional,
        direction=lambda generated, real: functional.direction(generated, real),
        provenance={
            "functional": functional.id,
            "support": functional.support,
            **dict(functional.provenance),
        },
    )


def _squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return cdist(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), metric="sqeuclidean")


def median_bandwidth(features: np.ndarray, *, max_samples: int = 2000, seed: int = 0) -> float:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("Bandwidth selection requires a 2-D feature array with at least two rows.")
    rng = np.random.default_rng(seed)
    if features.shape[0] > max_samples:
        features = features[rng.choice(features.shape[0], max_samples, replace=False)]
    distances = pdist(features, metric="euclidean")
    positive = distances[distances > 0]
    if positive.size == 0:
        raise ValueError("Cannot select an MMD bandwidth from identical training features.")
    return float(np.median(positive))


def mmd2_unbiased(x: np.ndarray, y: np.ndarray, *, bandwidth: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("MMD requires two 2-D arrays with the same feature dimension.")
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValueError("Unbiased MMD requires at least two samples per side.")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")
    scale = 2.0 * bandwidth * bandwidth
    kxx = np.exp(-_squared_distances(x, x) / scale)
    kyy = np.exp(-_squared_distances(y, y) / scale)
    kxy = np.exp(-_squared_distances(x, y) / scale)
    nx, ny = x.shape[0], y.shape[0]
    xx = (float(kxx.sum()) - float(np.trace(kxx))) / (nx * (nx - 1))
    yy = (float(kyy.sum()) - float(np.trace(kyy))) / (ny * (ny - 1))
    return float(xx + yy - 2.0 * float(kxy.mean()))


def mmd_metric(
    transform: Callable[[np.ndarray], np.ndarray],
    *,
    bandwidth: float,
    id: str = "feature_mmd2",
    max_samples: int = 5000,
    seed: int = 1789,
) -> MetricSpec:
    if max_samples < 2:
        raise ValueError("max_samples must be at least two.")

    def estimator(x: np.ndarray, y: np.ndarray) -> float:
        rng = np.random.default_rng(seed)
        x = np.asarray(x)
        y = np.asarray(y)
        if x.shape[0] > max_samples:
            x = x[rng.choice(x.shape[0], max_samples, replace=False)]
        if y.shape[0] > max_samples:
            y = y[rng.choice(y.shape[0], max_samples, replace=False)]
        return mmd2_unbiased(x, y, bandwidth=bandwidth)

    return MetricSpec(
        id=id,
        transform=transform,
        estimator=estimator,
        kind=MetricKind.SET,
        units="MMD²",
        description="Unbiased Gaussian-kernel MMD² in a training-fitted feature space.",
        pillar="diversity_and_coverage",
        provenance={
            "bandwidth": float(bandwidth),
            "max_samples_per_side": int(max_samples),
            "subsample_seed": int(seed),
        },
    )


def density_metric(
    transform: Callable[[np.ndarray], np.ndarray],
    *,
    k: int = 5,
    id: str = "feature_density",
) -> MetricSpec:
    """Absolute departure of manifold density from its ideal value of one."""

    return MetricSpec(
        id=id,
        transform=transform,
        estimator=lambda generated, real: abs(
            1.0 - density_coverage(real, generated, k=k)[0]
        ),
        kind=MetricKind.SET,
        units="absolute density departure",
        description="Absolute departure from Naeem-style feature-space density=1.",
        pillar="diversity_and_coverage",
        provenance={"k": int(k), "orientation": "lower_is_better"},
    )


def coverage_metric(
    transform: Callable[[np.ndarray], np.ndarray],
    *,
    k: int = 5,
    id: str = "feature_coverage_deficit",
) -> MetricSpec:
    """Feature-manifold coverage deficit, with zero as the ideal value."""

    return MetricSpec(
        id=id,
        transform=transform,
        estimator=lambda generated, real: 1.0
        - density_coverage(real, generated, k=k)[1],
        kind=MetricKind.SET,
        units="fraction",
        description="Naeem-style real-manifold coverage deficit (1 - coverage).",
        pillar="diversity_and_coverage",
        provenance={"k": int(k), "orientation": "lower_is_better"},
    )


def diversity_metric(
    transform: Callable[[np.ndarray], np.ndarray],
    *,
    id: str = "feature_diversity_departure",
) -> MetricSpec:
    return MetricSpec(
        id=id,
        transform=transform,
        estimator=diversity_discrepancy,
        kind=MetricKind.SET,
        units="relative departure",
        description="Absolute relative departure in mean feature-space pairwise distance.",
        pillar="diversity_and_coverage",
        direction=lambda generated, real: (
            "excessive diversity"
            if mean_pairwise_distance(generated) > mean_pairwise_distance(real)
            else "insufficient diversity"
        ),
        provenance={"orientation": "lower_is_better"},
    )


def dependence_metric(
    transform: Callable[[np.ndarray], np.ndarray],
    *,
    id: str = "spearman_dependence_mae",
) -> MetricSpec:
    return MetricSpec(
        id=id,
        transform=transform,
        estimator=spearman_matrix_mae,
        kind=MetricKind.MATRIX,
        units="correlation MAE",
        description="Mean absolute departure of the Spearman dependence matrix.",
        provenance={"orientation": "lower_is_better"},
    )


def spearman_pair_metrics(
    transform: Callable[[np.ndarray], np.ndarray],
    names: Sequence[str],
    *,
    id_prefix: str = "spearman_pair",
) -> tuple[MetricSpec, ...]:
    """Register interpretable entry-wise dependence discrepancies."""

    names = tuple(str(name) for name in names)
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError("names must contain at least two unique functional ids.")
    metrics = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            left_index, right_index = left, right

            def correlation(values: np.ndarray, i=left_index, j=right_index) -> float:
                values = np.asarray(values, dtype=np.float64)
                if values.ndim != 2 or values.shape[1] != len(names):
                    raise ValueError(
                        f"Dependence transform must return shape (N, {len(names)})."
                    )
                if np.unique(values[:, i]).size < 2 or np.unique(values[:, j]).size < 2:
                    raise MetricNotEvaluable(
                        "Spearman correlation is undefined for a constant entry."
                    )
                statistic = float(spearmanr(values[:, i], values[:, j]).statistic)
                if not np.isfinite(statistic):
                    raise MetricNotEvaluable(
                        "Spearman correlation is undefined for a constant entry."
                    )
                return statistic

            metrics.append(
                MetricSpec(
                    id=f"{id_prefix}:{names[left]}:{names[right]}",
                    transform=transform,
                    estimator=lambda generated, real, corr=correlation: abs(
                        corr(generated) - corr(real)
                    ),
                    kind=MetricKind.MATRIX,
                    units="absolute correlation difference",
                    description=(
                        f"Absolute Spearman-correlation departure for "
                        f"{names[left]} versus {names[right]}."
                    ),
                    provenance={
                        "left": names[left],
                        "right": names[right],
                        "orientation": "lower_is_better",
                    },
                )
            )
    return tuple(metrics)


def admissibility_metric(
    predicate: Callable[[np.ndarray], np.ndarray],
    *,
    id: str = "admissibility_rate_departure",
    description: str = "Absolute departure in hard admissibility-violation rate.",
) -> MetricSpec:
    """Create a registered rate discrepancy from a per-sample predicate.

    ``predicate`` must return one Boolean per sample, where ``True`` denotes a
    violation. The reference engine then calibrates finite-sample rate noise.
    """

    def transform(fields: np.ndarray) -> np.ndarray:
        values = np.asarray(predicate(fields))
        if values.shape != (fields.shape[0],):
            raise ValueError("Admissibility predicate must return shape (N,).")
        return values.astype(np.float64)

    return MetricSpec(
        id=id,
        transform=transform,
        estimator=lambda generated, real: abs(float(np.mean(generated) - np.mean(real))),
        kind=MetricKind.RATE,
        units="absolute rate difference",
        description=description,
        pillar="admissibility",
        provenance={"orientation": "lower_is_better", "predicate": "user_supplied"},
    )


def density_coverage(real: np.ndarray, generated: np.ndarray, *, k: int = 5) -> tuple[float, float]:
    """Naeem-style density and coverage with real-manifold radii."""

    real = np.asarray(real, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if real.ndim != 2 or generated.ndim != 2 or real.shape[1] != generated.shape[1]:
        raise ValueError("density_coverage requires matching 2-D feature arrays.")
    if not 1 <= k < real.shape[0]:
        raise ValueError("k must be below the number of real samples.")
    real_dist = cdist(real, real)
    np.fill_diagonal(real_dist, np.inf)
    radii = np.partition(real_dist, kth=k - 1, axis=1)[:, k - 1]
    gen_to_real = cdist(generated, real)
    density = float(np.mean(np.sum(gen_to_real <= radii[None, :], axis=1) / k))
    coverage = float(np.mean(np.min(gen_to_real, axis=0) <= radii))
    return density, coverage


def mean_pairwise_distance(values: np.ndarray, *, max_samples: int = 5000) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Diversity requires at least two feature vectors.")
    if values.shape[0] > max_samples:
        values = values[:max_samples]
    return float(np.mean(pdist(values)))


def diversity_discrepancy(generated: np.ndarray, real: np.ndarray) -> float:
    real_div = mean_pairwise_distance(real)
    if real_div <= 0:
        raise ValueError("Real feature diversity is zero.")
    ratio = mean_pairwise_distance(generated) / real_div
    return float(abs(1.0 - ratio))


def hard_violation_rate(fields: np.ndarray, *, lower: float, upper: float) -> float:
    fields = np.asarray(fields)
    if lower >= upper:
        raise ValueError("lower must be below upper.")
    per_sample = np.any((fields < lower) | (fields > upper), axis=tuple(range(1, fields.ndim)))
    return float(np.mean(per_sample))


def spearman_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Spearman closure requires shape (N, at least 2 functionals).")
    n_features = values.shape[1]
    matrix = np.eye(n_features, dtype=np.float64)
    for left in range(n_features):
        for right in range(left + 1, n_features):
            if (
                np.unique(values[:, left]).size < 2
                or np.unique(values[:, right]).size < 2
            ):
                raise MetricNotEvaluable(
                    "Spearman correlation is undefined for a constant entry."
                )
            value = float(spearmanr(values[:, left], values[:, right]).statistic)
            if not np.isfinite(value):
                raise MetricNotEvaluable(
                    "Spearman correlation is undefined for a constant entry."
                )
            matrix[left, right] = matrix[right, left] = value
    return matrix


def spearman_matrix_mae(generated: np.ndarray, real: np.ndarray) -> float:
    return float(np.mean(np.abs(spearman_matrix(generated) - spearman_matrix(real))))
