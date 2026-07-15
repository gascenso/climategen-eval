from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

from .types import MetricSpec


Perturbation = Callable[[np.ndarray, float, int], np.ndarray]


@dataclass(frozen=True)
class PerturbationSpec:
    """A predeclared metric sensitivity lane."""

    id: str
    apply: Perturbation
    severities: tuple[float, ...]
    expected_metrics: tuple[str, ...]
    expected_response: str = "nondecreasing"
    minimum_rank_correlation: float = 0.8
    provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if len(self.severities) < 3 or self.severities[0] != 0:
            raise ValueError("A perturbation lane requires at least three severities starting at 0.")
        if any(right <= left for left, right in zip(self.severities, self.severities[1:])):
            raise ValueError("Perturbation severities must be strictly increasing.")
        if self.expected_response != "nondecreasing":
            raise ValueError("v0.1 supports the nondecreasing response contract only.")
        if not 0 <= self.minimum_rank_correlation <= 1:
            raise ValueError("minimum_rank_correlation must be in [0, 1].")


def run_perturbation_sweep(
    spec: PerturbationSpec,
    *,
    source_fields: np.ndarray,
    reference_fields: np.ndarray,
    metrics: Sequence[MetricSpec],
    reference_boundaries: Mapping[str, float] | None = None,
    seed: int = 1789,
) -> dict[str, Any]:
    """Evaluate a registered severity-response curve on reserved real data."""

    source_fields = np.asarray(source_fields)
    reference_fields = np.asarray(reference_fields)
    if source_fields.shape[1:] != reference_fields.shape[1:]:
        raise ValueError("Sensitivity source and reference sample shapes differ.")
    registry = {metric.id: metric for metric in metrics}
    missing = sorted(set(spec.expected_metrics) - set(registry))
    if missing:
        raise ValueError(f"Sensitivity lane references unregistered metrics: {missing}.")
    boundaries = dict(reference_boundaries or {})
    responses = {metric_id: [] for metric_id in spec.expected_metrics}
    for severity_index, severity in enumerate(spec.severities):
        perturbed = np.asarray(spec.apply(source_fields, severity, seed + severity_index))
        if perturbed.shape != source_fields.shape:
            raise ValueError("A sensitivity perturbation changed the field shape.")
        for metric_id in spec.expected_metrics:
            metric = registry[metric_id]
            generated_values = np.asarray(metric.transform(perturbed))
            real_values = np.asarray(metric.transform(reference_fields))
            responses[metric_id].append(
                float(metric.estimator(generated_values, real_values))
            )
    results = {}
    severity_ranks = np.arange(len(spec.severities), dtype=np.float64)
    for metric_id, values in responses.items():
        value_array = np.asarray(values, dtype=np.float64)
        if np.std(value_array) <= np.finfo(float).eps:
            rank_correlation = 0.0
        else:
            rank_correlation = float(spearmanr(severity_ranks, value_array).statistic)
        monotone = bool(np.all(np.diff(value_array) >= -1e-12))
        boundary = boundaries.get(metric_id)
        reaches_boundary = None if boundary is None else bool(value_array[-1] > boundary)
        passed = (
            monotone
            and rank_correlation >= spec.minimum_rank_correlation
            and reaches_boundary is not False
        )
        results[metric_id] = {
            "values": value_array.tolist(),
            "rank_correlation": rank_correlation,
            "minimum_rank_correlation": spec.minimum_rank_correlation,
            "monotone_nondecreasing": monotone,
            "reference_boundary": boundary,
            "reaches_reference_boundary": reaches_boundary,
            "passed": passed,
        }
    return {
        "schema": "gauge_climate.sensitivity_lane.v1",
        "id": spec.id,
        "severities": list(spec.severities),
        "expected_response": spec.expected_response,
        "provenance": dict(spec.provenance or {}),
        "metrics": results,
        "passed": all(result["passed"] for result in results.values()),
    }


def additive_shift_lane(
    *,
    severities: Sequence[float],
    expected_metrics: Sequence[str],
) -> PerturbationSpec:
    return PerturbationSpec(
        id="additive_field_shift",
        apply=lambda fields, severity, seed: np.asarray(fields) + severity,
        severities=tuple(float(value) for value in severities),
        expected_metrics=tuple(expected_metrics),
        provenance={"operation": "field + severity", "fit_source": "reserved_real"},
    )


def copyist(train_fields: np.ndarray, *, n: int, seed: int = 1789) -> np.ndarray:
    train_fields = np.asarray(train_fields)
    rng = np.random.default_rng(seed)
    return np.asarray(train_fields[rng.choice(train_fields.shape[0], n, replace=True)]).copy()


def climatology_noise(train_fields: np.ndarray, *, n: int, seed: int = 1789) -> np.ndarray:
    train_fields = np.asarray(train_fields, dtype=np.float64)
    rng = np.random.default_rng(seed)
    mean = train_fields.mean(axis=0)
    scale = train_fields.std(axis=0)
    output = mean + rng.normal(size=(n, *mean.shape)) * scale
    return output.astype(np.float32)


def phase_shuffle(train_fields: np.ndarray, *, n: int, seed: int = 1789) -> np.ndarray:
    train_fields = np.asarray(train_fields, dtype=np.float64)
    if train_fields.ndim not in (3, 4):
        raise ValueError("phase_shuffle supports (N,H,W) or (N,C,H,W).")
    rng = np.random.default_rng(seed)
    selected = train_fields[rng.choice(train_fields.shape[0], n, replace=True)]
    leading = selected.shape[:-2]
    flat = selected.reshape(-1, *selected.shape[-2:])
    output = np.empty_like(flat)
    for index, image in enumerate(flat):
        spectrum = np.fft.fft2(image)
        # The FFT of real random noise supplies a Hermitian-symmetric phase
        # field, including valid real-valued DC/Nyquist coefficients. Combining
        # it with the original amplitude preserves the complete discrete PSD.
        phase = np.angle(np.fft.fft2(rng.normal(size=image.shape)))
        shuffled = np.abs(spectrum) * np.exp(1j * phase)
        shuffled[0, 0] = spectrum[0, 0]
        output[index] = np.fft.ifft2(shuffled).real
    return output.reshape(*leading, *selected.shape[-2:]).astype(np.float32)


def blurred(train_fields: np.ndarray, *, n: int, sigma: float, seed: int = 1789) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    selected = copyist(train_fields, n=n, seed=seed).astype(np.float64)
    if selected.ndim == 3:
        sigmas = (0.0, sigma, sigma)
    elif selected.ndim == 4:
        sigmas = (0.0, 0.0, sigma, sigma)
    else:
        raise ValueError("blurred supports (N,H,W) or (N,C,H,W).")
    return gaussian_filter(selected, sigma=sigmas).astype(np.float32)


def cross_class(
    train_fields: np.ndarray,
    train_labels: np.ndarray,
    target_labels: np.ndarray,
    *,
    seed: int = 1789,
) -> np.ndarray:
    train_fields = np.asarray(train_fields)
    train_labels = np.asarray(train_labels)
    target_labels = np.asarray(target_labels)
    if train_labels.shape != (train_fields.shape[0],):
        raise ValueError("train_labels must align with train_fields.")
    rng = np.random.default_rng(seed)
    rows = []
    for target in target_labels:
        candidates = np.flatnonzero(train_labels != target)
        if candidates.size == 0:
            raise ValueError("cross_class requires at least two training labels.")
        rows.append(train_fields[int(rng.choice(candidates))])
    return np.asarray(rows).copy()
