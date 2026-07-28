from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, shift as ndimage_shift
from scipy.stats import spearmanr

from .types import MetricSpec


Perturbation = Callable[[np.ndarray, float, int], np.ndarray]


class ControlRole(str, Enum):
    SENSITIVITY = "sensitivity"
    INVARIANCE = "invariance"
    COPYING = "copying"


@dataclass(frozen=True)
class ControlSpec:
    """Typed registry entry linking metrics to an expected control response."""

    id: str
    role: ControlRole | str
    metrics: tuple[str, ...]
    expected_response: str
    provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", ControlRole(self.role))
        if not self.id or not self.metrics or len(self.metrics) != len(set(self.metrics)):
            raise ValueError("A control needs an identifier and unique metric identifiers.")
        if not self.expected_response:
            raise ValueError("A control must declare its expected response.")


class ControlRegistry:
    """Validated collection of named metric-control contracts."""

    def __init__(self, controls: Sequence[ControlSpec]):
        self.controls = tuple(controls)
        identifiers = [control.id for control in self.controls]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Control identifiers must be unique.")

    def for_metric(self, metric_id: str) -> tuple[ControlSpec, ...]:
        return tuple(control for control in self.controls if metric_id in control.metrics)

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "id": control.id,
                "role": control.role.value,
                "metrics": list(control.metrics),
                "expected_response": control.expected_response,
                "provenance": dict(control.provenance or {}),
            }
            for control in self.controls
        ]


@dataclass(frozen=True)
class PerturbationSpec:
    """A predeclared metric sensitivity lane."""

    id: str
    apply: Perturbation
    severities: tuple[float, ...]
    expected_metrics: tuple[str, ...]
    expected_response: str = "nondecreasing"
    minimum_rank_correlation: float = 0.8
    require_reference_boundary: bool = False
    provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if len(self.severities) < 3 or self.severities[0] != 0:
            raise ValueError(
                "A perturbation lane requires at least three severities starting at 0."
            )
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
    reference_values = {
        metric_id: np.asarray(registry[metric_id].transform(reference_fields))
        for metric_id in spec.expected_metrics
    }
    for severity in spec.severities:
        # Keep the stochastic realization paired across severities. Otherwise a
        # severity-response curve can be confounded by a new random direction or
        # noise field at every point.
        perturbed = np.asarray(spec.apply(source_fields, severity, seed))
        if perturbed.shape != source_fields.shape:
            raise ValueError("A sensitivity perturbation changed the field shape.")
        for metric_id in spec.expected_metrics:
            metric = registry[metric_id]
            generated_values = np.asarray(metric.transform(perturbed))
            responses[metric_id].append(
                float(metric.estimator(generated_values, reference_values[metric_id]))
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
        first_crossing_severity = None
        if boundary is not None:
            crossing_indices = np.flatnonzero(value_array > boundary)
            if crossing_indices.size:
                first_crossing_severity = float(spec.severities[int(crossing_indices[0])])
        trend_passed = bool(
            rank_correlation > spec.minimum_rank_correlation
            or np.isclose(
                rank_correlation,
                spec.minimum_rank_correlation,
                rtol=0.0,
                atol=1e-12,
            )
        )
        resolution_status = (
            "NOT_EVALUATED"
            if reaches_boundary is None
            else (
                "CROSSES_REFERENCE_AT_MAX_SEVERITY"
                if reaches_boundary
                else "REMAINS_WITHIN_REFERENCE_AT_MAX_SEVERITY"
            )
        )
        passed = bool(
            trend_passed and (not spec.require_reference_boundary or reaches_boundary is True)
        )
        results[metric_id] = {
            "values": value_array.tolist(),
            "rank_correlation": rank_correlation,
            "minimum_rank_correlation": spec.minimum_rank_correlation,
            "monotone_nondecreasing": monotone,
            "response_contract_passed": trend_passed,
            "reference_boundary": boundary,
            "reaches_reference_boundary": reaches_boundary,
            "first_reference_crossing_severity": first_crossing_severity,
            "resolution_status": resolution_status,
            "reference_boundary_required_for_pass": spec.require_reference_boundary,
            "passed": passed,
        }
    return {
        "schema": "gauge_climate.sensitivity_lane.v1",
        "id": spec.id,
        "severities": list(spec.severities),
        "expected_response": spec.expected_response,
        "pass_rule": (
            "rank trend plus reference-boundary crossing"
            if spec.require_reference_boundary
            else "rank trend; boundary crossing reported separately as resolution"
        ),
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


def gaussian_blur_lane(
    *,
    severities: Sequence[float],
    expected_metrics: Sequence[str],
) -> PerturbationSpec:
    """Create a spatial Gaussian-blur lane without blurring sample/channel axes."""

    def apply(fields: np.ndarray, severity: float, seed: int) -> np.ndarray:
        del seed
        values = np.asarray(fields)
        if severity == 0:
            return values.copy()
        if severity < 0:
            raise ValueError("Gaussian-blur severity may not be negative.")
        if values.ndim == 3:
            sigmas = (0.0, severity, severity)
        elif values.ndim == 4:
            sigmas = (0.0, 0.0, severity, severity)
        else:
            raise ValueError("Gaussian blur supports (N,H,W) or (N,C,H,W).")
        return gaussian_filter(values.astype(np.float64), sigma=sigmas).astype(
            values.dtype, copy=False
        )

    return PerturbationSpec(
        id="gaussian_blur",
        apply=apply,
        severities=tuple(float(value) for value in severities),
        expected_metrics=tuple(expected_metrics),
        provenance={
            "operation": "spatial Gaussian filter",
            "boundary": "scipy.ndimage gaussian_filter default reflect",
            "fit_source": "reserved_real",
        },
    )


def center_jitter_lane(
    *,
    severities: Sequence[float],
    expected_metrics: Sequence[str],
) -> PerturbationSpec:
    """Create a paired random-direction center-displacement lane."""

    def apply(fields: np.ndarray, severity: float, seed: int) -> np.ndarray:
        values = np.asarray(fields)
        if severity == 0:
            return values.copy()
        if severity < 0:
            raise ValueError("Center-jitter severity may not be negative.")
        if values.ndim not in (3, 4):
            raise ValueError("Center jitter supports (N,H,W) or (N,C,H,W).")
        rng = np.random.default_rng(seed)
        angles = rng.uniform(0.0, 2.0 * np.pi, size=values.shape[0])
        output = np.empty_like(values)
        for index, angle in enumerate(angles):
            spatial_shift = (
                float(severity * np.sin(angle)),
                float(severity * np.cos(angle)),
            )
            shift_vector = spatial_shift if values.ndim == 3 else (0.0, *spatial_shift)
            output[index] = ndimage_shift(
                values[index],
                shift=shift_vector,
                order=1,
                mode="reflect",
                prefilter=False,
            )
        return output

    return PerturbationSpec(
        id="center_jitter",
        apply=apply,
        severities=tuple(float(value) for value in severities),
        expected_metrics=tuple(expected_metrics),
        provenance={
            "operation": "paired random-direction spatial displacement",
            "interpolation": "bilinear",
            "boundary": "reflect",
            "fit_source": "reserved_real",
        },
    )


def threshold_band_warming_lane(
    *,
    severities: Sequence[float],
    expected_metrics: Sequence[str],
    threshold: float,
    half_width: float,
) -> PerturbationSpec:
    """Warm pixels near a declared threshold while leaving other pixels fixed."""

    def apply(fields: np.ndarray, severity: float, seed: int) -> np.ndarray:
        del seed
        values = np.asarray(fields)
        output = values.copy()
        mask = np.abs(values - threshold) <= half_width
        output[mask] = output[mask] + severity
        return output

    return PerturbationSpec(
        id="threshold_band_warming",
        apply=apply,
        severities=tuple(float(value) for value in severities),
        expected_metrics=tuple(expected_metrics),
        provenance={
            "operation": "add severity to pixels within threshold band",
            "threshold": float(threshold),
            "half_width": float(half_width),
            "fit_source": "reserved_real",
        },
    )


def radial_contrast_lane(
    *,
    severities: Sequence[float],
    expected_metrics: Sequence[str],
    pixel_size_km: float,
    eye_radius_km: float,
    annulus_inner_km: float,
    annulus_outer_km: float,
) -> PerturbationSpec:
    """Increase inner-eye minus annulus contrast by a known temperature amount."""

    def apply(fields: np.ndarray, severity: float, seed: int) -> np.ndarray:
        del seed
        values = np.asarray(fields)
        height, width = values.shape[-2:]
        y, x = np.mgrid[0:height, 0:width]
        radius = (
            np.sqrt((y - (height - 1) / 2.0) ** 2 + (x - (width - 1) / 2.0) ** 2) * pixel_size_km
        )
        output = values.copy()
        output[..., radius <= eye_radius_km] += severity
        annulus = (radius > annulus_inner_km) & (radius <= annulus_outer_km)
        output[..., annulus] -= 0.5 * severity
        return output

    return PerturbationSpec(
        id="radial_eye_contrast",
        apply=apply,
        severities=tuple(float(value) for value in severities),
        expected_metrics=tuple(expected_metrics),
        provenance={
            "operation": "warm eye and cool surrounding annulus",
            "pixel_size_km": float(pixel_size_km),
            "eye_radius_km": float(eye_radius_km),
            "annulus_inner_km": float(annulus_inner_km),
            "annulus_outer_km": float(annulus_outer_km),
            "fit_source": "reserved_real",
        },
    )


def radial_warp_lane(
    *,
    severities: Sequence[float],
    expected_metrics: Sequence[str],
) -> PerturbationSpec:
    """Radially dilate fields by a fractional scale around the image centre."""

    def apply(fields: np.ndarray, severity: float, seed: int) -> np.ndarray:
        del seed
        values = np.asarray(fields)
        if values.ndim != 3:
            raise ValueError("Radial warp currently supports fields shaped (N,H,W).")
        height, width = values.shape[-2:]
        y, x = np.mgrid[0:height, 0:width]
        cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
        scale = 1.0 + float(severity)
        source_y = cy + (y - cy) / scale
        source_x = cx + (x - cx) / scale
        output = np.empty_like(values)
        for index, field in enumerate(values):
            output[index] = map_coordinates(
                field,
                [source_y, source_x],
                order=1,
                mode="reflect",
                prefilter=False,
            )
        return output

    return PerturbationSpec(
        id="radial_warp",
        apply=apply,
        severities=tuple(float(value) for value in severities),
        expected_metrics=tuple(expected_metrics),
        provenance={
            "operation": "radial coordinate dilation about image centre",
            "interpolation": "bilinear",
            "fit_source": "reserved_real",
        },
    )


def checkerboard_artifact_lane(
    *,
    severities: Sequence[float],
    expected_metrics: Sequence[str],
    wavelength_pixels: int = 4,
) -> PerturbationSpec:
    """Add a deterministic high-frequency off-manifold spatial artifact."""

    def apply(fields: np.ndarray, severity: float, seed: int) -> np.ndarray:
        del seed
        values = np.asarray(fields)
        height, width = values.shape[-2:]
        y, x = np.mgrid[0:height, 0:width]
        pattern = ((y // wavelength_pixels + x // wavelength_pixels) % 2) * 2.0 - 1.0
        return values + severity * pattern

    return PerturbationSpec(
        id="checkerboard_artifact",
        apply=apply,
        severities=tuple(float(value) for value in severities),
        expected_metrics=tuple(expected_metrics),
        provenance={
            "operation": "add deterministic checkerboard artifact",
            "wavelength_pixels": int(wavelength_pixels),
            "fit_source": "reserved_real",
        },
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


def phase_shuffle_fields(fields: np.ndarray, *, seed: int = 1789) -> np.ndarray:
    """Randomize phase while preserving each field's discrete Fourier amplitude."""

    fields = np.asarray(fields, dtype=np.float64)
    if fields.ndim not in (3, 4):
        raise ValueError("phase_shuffle supports (N,H,W) or (N,C,H,W).")
    rng = np.random.default_rng(seed)
    leading = fields.shape[:-2]
    flat = fields.reshape(-1, *fields.shape[-2:])
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
    return output.reshape(*leading, *fields.shape[-2:]).astype(np.float32)


def phase_shuffle(train_fields: np.ndarray, *, n: int, seed: int = 1789) -> np.ndarray:
    train_fields = np.asarray(train_fields)
    rng = np.random.default_rng(seed)
    selected = train_fields[rng.choice(train_fields.shape[0], n, replace=True)]
    return phase_shuffle_fields(selected, seed=seed + 1)


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
