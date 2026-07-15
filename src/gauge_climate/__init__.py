"""GAUGE: evaluation of generated Earth-system fields."""

from .data import Dataset
from .evaluation import evaluate
from . import adapters, downstream, features, lanes, memorization, metrics
from .report import EvaluationReport
from .types import (
    DetectionStatus,
    EvaluationConfig,
    FunctionalSpec,
    MetricKind,
    MetricSpec,
    PracticalStatus,
    SamplingEstimand,
    SensitivityStatus,
)

__all__ = [
    "Dataset",
    "DetectionStatus",
    "EvaluationConfig",
    "EvaluationReport",
    "FunctionalSpec",
    "MetricKind",
    "MetricSpec",
    "PracticalStatus",
    "SamplingEstimand",
    "SensitivityStatus",
    "evaluate",
    "downstream",
    "adapters",
    "features",
    "lanes",
    "memorization",
    "metrics",
]

__version__ = "0.1.0"
