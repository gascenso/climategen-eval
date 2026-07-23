from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable.")


class SamplingEstimand(str, Enum):
    """Distribution represented by the real archive."""

    BLOCK_BALANCED = "block_balanced"
    FRAME_WEIGHTED = "frame_weighted"


class MetricKind(str, Enum):
    FUNCTIONAL_W1 = "functional_w1"
    SET = "set"
    CURVE = "curve"
    RATE = "rate"
    MATRIX = "matrix"


class DetectionStatus(str, Enum):
    NO_DEPARTURE_DETECTED = "NO_DEPARTURE_DETECTED"
    DEPARTURE_DETECTED = "DEPARTURE_DETECTED"
    NOT_EVALUATED = "NOT_EVALUATED"


ArrayTransform = Callable[[np.ndarray], np.ndarray]
MetricEstimator = Callable[[np.ndarray, np.ndarray], float]
DirectionFunction = Callable[[np.ndarray, np.ndarray], str]


@dataclass(frozen=True)
class FunctionalSpec:
    """A predeclared scalar physical functional."""

    id: str
    function: ArrayTransform
    units: str
    description: str
    support: tuple[float, float] | None = None
    tail_referenced: bool = False
    direction_negative: str = "lower"
    direction_positive: str = "higher"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or any(char.isspace() for char in self.id):
            raise ValueError("FunctionalSpec.id must be a non-empty identifier without spaces.")
        if not self.units:
            raise ValueError(f"Functional {self.id!r} must declare units.")
        if not self.description:
            raise ValueError(f"Functional {self.id!r} must have a description.")
        if self.support is not None:
            if len(self.support) != 2 or not self.support[0] < self.support[1]:
                raise ValueError(f"Functional {self.id!r} support must be (lower, upper).")

    def transform(self, fields: np.ndarray) -> np.ndarray:
        values = np.asarray(self.function(fields), dtype=np.float64)
        if values.ndim != 1 or values.shape[0] != fields.shape[0]:
            raise ValueError(
                f"Functional {self.id!r} must return shape (N,), got {values.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Functional {self.id!r} returned non-finite values.")
        if self.support is not None:
            lower, upper = self.support
            if np.any(values < lower) or np.any(values > upper):
                raise ValueError(
                    f"Functional {self.id!r} returned values outside its declared "
                    f"support [{lower}, {upper}]."
                )
        return values

    def direction(self, generated: np.ndarray, real: np.ndarray) -> str:
        delta = float(np.median(generated) - np.median(real))
        if np.isclose(delta, 0.0):
            return "no median shift"
        return self.direction_positive if delta > 0 else self.direction_negative


@dataclass(frozen=True)
class MetricSpec:
    """A scalar distance together with the projection it operates on."""

    id: str
    transform: ArrayTransform
    estimator: MetricEstimator
    kind: MetricKind
    units: str
    description: str
    pillar: str = "physical_fidelity"
    functional: FunctionalSpec | None = None
    direction: DirectionFunction | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or any(char.isspace() for char in self.id):
            raise ValueError("MetricSpec.id must be a non-empty identifier without spaces.")
        if not self.description:
            raise ValueError(f"Metric {self.id!r} must have a description.")
        allowed_pillars = {
            "physical_fidelity",
            "conditional_fidelity",
            "diversity_and_coverage",
            "memorization",
            "admissibility",
        }
        if self.pillar not in allowed_pillars:
            raise ValueError(
                f"Metric {self.id!r} has unknown report pillar {self.pillar!r}."
            )
        if self.kind is MetricKind.FUNCTIONAL_W1 and self.functional is None:
            raise ValueError("functional_w1 metrics must retain their FunctionalSpec.")


@dataclass(frozen=True)
class EvaluationConfig:
    """Frozen statistical and computational choices for one evaluation."""

    n_eval: int | None = None
    n_resamples: int = 200
    diagnostic_resamples: int | None = None
    diagnostic_resamples_by_metric: Mapping[str, int] = field(default_factory=dict)
    sensitivity_resamples: int = 100
    alpha: float = 0.05
    confidence: float = 0.95
    estimand: SamplingEstimand | str = SamplingEstimand.BLOCK_BALANCED
    seed: int = 1789
    min_blocks_per_half: int = 10
    warn_blocks_per_half: int = 30
    shift_power: float = 0.80
    shift_grid_size: int = 33
    reference_stability_threshold: float = 0.25
    simultaneous_family: tuple[str, ...] = ()
    max_materialize_bytes: int | None = None
    transform_batch_size: int = 128
    include_partitions: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "estimand", SamplingEstimand(self.estimand))
        if self.n_eval is not None and self.n_eval <= 1:
            raise ValueError("n_eval must be greater than one.")
        if self.n_resamples < 20:
            raise ValueError("n_resamples must be at least 20.")
        if self.diagnostic_resamples is None:
            object.__setattr__(self, "diagnostic_resamples", min(self.n_resamples, 200))
        assert self.diagnostic_resamples is not None
        if self.diagnostic_resamples < 20:
            raise ValueError("diagnostic_resamples must be at least 20.")
        if self.diagnostic_resamples > self.n_resamples:
            raise ValueError("diagnostic_resamples cannot exceed n_resamples.")
        diagnostic_overrides = {
            str(metric_id): int(value)
            for metric_id, value in self.diagnostic_resamples_by_metric.items()
        }
        if any(not metric_id or any(char.isspace() for char in metric_id) for metric_id in diagnostic_overrides):
            raise ValueError(
                "diagnostic_resamples_by_metric keys must be metric identifiers."
            )
        if any(
            value < 20 or value > self.n_resamples
            for value in diagnostic_overrides.values()
        ):
            raise ValueError(
                "Per-metric diagnostic resamples must be between 20 and n_resamples."
            )
        object.__setattr__(
            self,
            "diagnostic_resamples_by_metric",
            diagnostic_overrides,
        )
        if self.sensitivity_resamples < 20:
            raise ValueError("sensitivity_resamples must be at least 20.")
        if not 0 < self.alpha < 0.5:
            raise ValueError("alpha must be between 0 and 0.5.")
        if not 0.5 < self.confidence < 1:
            raise ValueError("confidence must be between 0.5 and 1.")
        if not np.isclose(self.confidence, 1.0 - self.alpha):
            raise ValueError("confidence must equal 1 - alpha for one coherent report level.")
        if self.min_blocks_per_half < 2:
            raise ValueError("min_blocks_per_half must be at least two.")
        if self.warn_blocks_per_half < self.min_blocks_per_half:
            raise ValueError("warn_blocks_per_half cannot be below the hard minimum.")
        if not 0.5 < self.shift_power < 1:
            raise ValueError("shift_power must be between 0.5 and 1.")
        if self.shift_grid_size < 8:
            raise ValueError("shift_grid_size must be at least eight.")
        if not 0 < self.reference_stability_threshold <= 1:
            raise ValueError("reference_stability_threshold must be in (0, 1].")
        if self.transform_batch_size < 1:
            raise ValueError("transform_batch_size must be positive.")
        if isinstance(self.simultaneous_family, str):
            raise ValueError(
                "simultaneous_family must be a list/tuple of exact 'stratum:metric' keys."
            )
        family = tuple(str(value) for value in self.simultaneous_family)
        if len(family) != len(set(family)) or any(":" not in value for value in family):
            raise ValueError(
                "simultaneous_family must contain unique exact 'stratum:metric' keys."
            )
        object.__setattr__(self, "simultaneous_family", family)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["estimand"] = self.estimand.value
        return _jsonable(payload)


@dataclass(frozen=True)
class CellReport:
    metric: str
    stratum: str
    kind: str
    pillar: str
    units: str
    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    reference_q025: float | None
    reference_q50: float | None
    reference_q95: float | None
    reference_q975: float | None
    reference_detection_boundary: float | None
    reference_position: float | None
    reference_position_low: float | None
    reference_position_high: float | None
    shift_resolution: float | None
    detection_status: DetectionStatus
    n_eval: int
    n_real: int
    n_generated: int
    n_blocks: int
    estimand: SamplingEstimand
    direction: str | None = None
    simultaneous_boundary: float | None = None
    simultaneous_detection_status: DetectionStatus = DetectionStatus.NOT_EVALUATED
    warnings: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["detection_status"] = self.detection_status.value
        payload["simultaneous_detection_status"] = self.simultaneous_detection_status.value
        payload["estimand"] = self.estimand.value
        payload["warnings"] = list(self.warnings)
        return _jsonable(payload)
