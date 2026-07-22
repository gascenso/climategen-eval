"""GAUGE: evaluation of generated Earth-system fields."""

from .data import Dataset
from .evaluation import evaluate
from . import adapters, features, lanes, memorization, metrics
from .report import EvaluationReport
from .presentation import PaperBundleSpec, ReferenceControl
from .types import (
    DetectionStatus,
    EvaluationConfig,
    FunctionalSpec,
    MetricKind,
    MetricSpec,
    SamplingEstimand,
)

__all__ = [
    "Dataset",
    "DetectionStatus",
    "EvaluationConfig",
    "EvaluationReport",
    "FunctionalSpec",
    "MetricKind",
    "MetricSpec",
    "PaperBundleSpec",
    "ReferenceControl",
    "SamplingEstimand",
    "evaluate",
    "adapters",
    "features",
    "lanes",
    "memorization",
    "metrics",
]

__version__ = "0.1.0"
