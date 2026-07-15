from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def _aligned_vector(values: Sequence[Any] | np.ndarray | None, n: int, name: str) -> np.ndarray | None:
    if values is None:
        return None
    raw_values = values.tolist() if isinstance(values, np.ndarray) else list(values)
    for value in raw_values:
        missing_number = isinstance(value, (float, np.floating)) and np.isnan(value)
        missing_text = isinstance(value, str) and not value.strip()
        missing_time = isinstance(value, (np.datetime64, np.timedelta64)) and np.isnat(value)
        if value is None or missing_number or missing_text or missing_time:
            raise ValueError(f"{name} may not contain missing values.")
    array = np.asarray(values)
    if array.ndim != 1 or array.shape[0] != n:
        raise ValueError(f"{name} must have shape ({n},), got {array.shape}.")
    return array


@dataclass(frozen=True)
class Dataset:
    """Fields and aligned metadata used by GAUGE.

    The first axis is always the sample axis. Real training and test datasets
    require block identifiers when passed to :func:`evaluate`; generated
    datasets do not.
    """

    fields: Any
    blocks: Sequence[Any] | np.ndarray | None = None
    labels: Sequence[Any] | np.ndarray | None = None
    sample_ids: Sequence[Any] | np.ndarray | None = None
    name: str = "dataset"
    units: str | None = None
    allow_nan: bool = False
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        shape = getattr(self.fields, "shape", None)
        if shape is None or len(shape) < 2:
            raise ValueError("fields must expose shape (N, ...sample dimensions...).")
        n = int(shape[0])
        if n < 2:
            raise ValueError("A Dataset must contain at least two samples.")
        object.__setattr__(self, "blocks", _aligned_vector(self.blocks, n, "blocks"))
        object.__setattr__(self, "labels", _aligned_vector(self.labels, n, "labels"))
        object.__setattr__(self, "sample_ids", _aligned_vector(self.sample_ids, n, "sample_ids"))
        if self.sample_ids is not None:
            normalized = [str(value) for value in self.sample_ids.tolist()]
            if len(normalized) != len(set(normalized)):
                raise ValueError("sample_ids must be unique within a Dataset.")
        if self.units is not None and not str(self.units).strip():
            raise ValueError("units must be a non-empty string when provided.")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def n_samples(self) -> int:
        return int(self.fields.shape[0])

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.fields.shape[1:])

    def as_numpy(self, *, max_bytes: int | None = None) -> np.ndarray:
        source = self.fields
        if hasattr(source, "data") and source.__class__.__module__.startswith("xarray"):
            source = source.data
        shape = tuple(int(value) for value in source.shape)
        dtype = np.dtype(getattr(source, "dtype", np.float64))
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if max_bytes is not None and expected > max_bytes:
            raise MemoryError(
                f"Materializing {self.name!r} requires approximately {expected} bytes, "
                f"above max_materialize_bytes={max_bytes}."
            )
        if hasattr(source, "compute"):
            source = source.compute()
        array = np.asarray(source)
        if array.shape[0] != self.n_samples:
            raise ValueError("Materialized fields changed sample-axis length.")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("fields must have a numeric dtype.")
        if not self.allow_nan and not np.all(np.isfinite(array)):
            raise ValueError(
                f"Dataset {self.name!r} contains NaN or infinite fields; provide an "
                "explicit imputation/masking adapter or set allow_nan=True."
            )
        return array

    def validate_fields(self, *, batch_size: int = 64) -> dict[str, Any]:
        """Scan numeric/finite validity without materializing the full dataset."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        source = self.fields
        if hasattr(source, "data") and source.__class__.__module__.startswith("xarray"):
            source = source.data
        dtype = np.dtype(getattr(source, "dtype", np.float64))
        if not np.issubdtype(dtype, np.number):
            raise TypeError("fields must have a numeric dtype.")
        for start in range(0, self.n_samples, batch_size):
            batch = source[start : start + batch_size]
            if hasattr(batch, "compute"):
                batch = batch.compute()
            batch = np.asarray(batch)
            if not self.allow_nan and not np.all(np.isfinite(batch)):
                raise ValueError(
                    f"Dataset {self.name!r} contains NaN or infinite fields in "
                    f"sample batch starting at {start}."
                )
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "sample_shape": list(self.sample_shape),
            "dtype": str(dtype),
            "finite_scan": "skipped_by_allow_nan" if self.allow_nan else "passed",
        }


def normalized_blocks(blocks: np.ndarray) -> np.ndarray:
    """Normalize heterogeneous block ids to deterministic strings."""

    array = np.asarray(blocks)
    if array.ndim != 1:
        raise ValueError("blocks must be one-dimensional.")
    if any(value is None for value in array.tolist()):
        raise ValueError("blocks may not contain missing values.")
    return np.asarray([str(value) for value in array], dtype=object)


def validate_split(train: Dataset, real: Dataset) -> dict[str, Any]:
    if train.blocks is None or real.blocks is None:
        raise ValueError("Training and held-out real datasets must provide block ids.")
    train_blocks = normalized_blocks(train.blocks)
    real_blocks = normalized_blocks(real.blocks)
    overlap = sorted(set(train_blocks.tolist()) & set(real_blocks.tolist()))
    if overlap:
        preview = ", ".join(overlap[:5])
        raise ValueError(
            f"Train/test block leakage: {len(overlap)} overlapping blocks ({preview})."
        )
    sample_overlap: list[str] = []
    if train.sample_ids is not None and real.sample_ids is not None:
        train_ids = {str(value) for value in train.sample_ids.tolist()}
        real_ids = {str(value) for value in real.sample_ids.tolist()}
        sample_overlap = sorted(train_ids & real_ids)
        if sample_overlap:
            preview = ", ".join(sample_overlap[:5])
            raise ValueError(
                f"Train/test sample-id leakage: {len(sample_overlap)} overlapping ids "
                f"({preview})."
            )
    if train.sample_shape != real.sample_shape:
        raise ValueError(
            f"Train and real sample shapes differ: {train.sample_shape} vs {real.sample_shape}."
        )
    if train.units is not None and real.units is not None and train.units != real.units:
        raise ValueError(f"Train and real field units differ: {train.units!r} vs {real.units!r}.")
    return {
        "n_train_blocks": int(np.unique(train_blocks).size),
        "n_test_blocks": int(np.unique(real_blocks).size),
        "overlap_count": 0,
        "sample_id_overlap_count": 0,
    }


def validate_generated(real: Dataset, generated: Dataset) -> None:
    if real.sample_shape != generated.sample_shape:
        raise ValueError(
            "Real and generated sample shapes differ: "
            f"{real.sample_shape} vs {generated.sample_shape}."
        )
    if real.units is not None and generated.units is not None and real.units != generated.units:
        raise ValueError(
            f"Real and generated field units differ: {real.units!r} vs {generated.units!r}."
        )


def label_indices(dataset: Dataset, label: Any | None) -> np.ndarray:
    if label is None:
        return np.arange(dataset.n_samples, dtype=np.int64)
    if dataset.labels is None:
        raise ValueError(f"Dataset {dataset.name!r} has no labels for conditional evaluation.")
    return np.flatnonzero(np.asarray(dataset.labels) == label).astype(np.int64)
