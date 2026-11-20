"""GAUGE: evaluation of generated Earth-system fields."""

from .data import Dataset
from .evaluation import evaluate
from . import adapters, features, lanes, memorization, metrics
from .report import EvaluationReport
from .presentation import PaperBundleSpec, ReferenceControl
from .lanes import ControlRegistry, ControlRole, ControlSpec
from .types import (
    ControlStatus,
    EvaluationConfig,
    FunctionalSpec,
    MetricKind,
    MetricSpec,
    ReferenceStatus,
    SamplingEstimand,
)

__all__ = [
    "Dataset",
    "ControlStatus",
    "ControlRegistry",
    "ControlRole",
    "ControlSpec",
    "EvaluationConfig",
    "EvaluationReport",
    "FunctionalSpec",
    "MetricKind",
    "MetricSpec",
    "PaperBundleSpec",
    "ReferenceControl",
    "ReferenceStatus",
    "SamplingEstimand",
    "evaluate",
    "adapters",
    "features",
    "lanes",
    "memorization",
    "metrics",
]

__version__ = "0.2.1"
