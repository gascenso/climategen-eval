from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .types import FunctionalSpec


@dataclass
class TrainFittedFeatureSpace:
    """Standardized physical functionals plus training-fit EOF coefficients."""

    functionals: tuple[FunctionalSpec, ...]
    pixel_mean: np.ndarray
    eof_components: np.ndarray
    functional_mean: np.ndarray
    functional_scale: np.ndarray
    eof_scale: np.ndarray
    fit_indices: np.ndarray
    sample_shape: tuple[int, ...]
    fit_source: str
    source_fingerprint: str
    batch_size: int = 128

    @classmethod
    def fit(
        cls,
        fields: np.ndarray,
        functionals: Sequence[FunctionalSpec],
        *,
        n_eof: int = 20,
        max_fit_samples: int = 512,
        seed: int = 1789,
        batch_size: int = 128,
        fit_source: str = "train",
    ) -> "TrainFittedFeatureSpace":
        fields = np.asarray(fields)
        if fields.ndim < 2 or fields.shape[0] < 2:
            raise ValueError("Feature fitting requires shape (N, ...), N >= 2.")
        if n_eof < 0:
            raise ValueError("n_eof may not be negative.")
        if fit_source != "train":
            raise ValueError("GAUGE feature artifacts must be fitted on the training split.")
        rng = np.random.default_rng(seed)
        fit_n = min(int(max_fit_samples), fields.shape[0])
        fit_indices = np.sort(rng.choice(fields.shape[0], size=fit_n, replace=False))
        fit_fields = np.asarray(fields[fit_indices], dtype=np.float64)
        source_digest = hashlib.sha256()
        source_digest.update(str(fields.shape).encode())
        source_digest.update(str(fields.dtype).encode())
        source_digest.update(fit_indices.tobytes())
        source_digest.update(fit_fields.tobytes())
        flat = fit_fields.reshape(fit_n, -1)
        pixel_mean = flat.mean(axis=0)
        centered = flat - pixel_mean
        max_rank = min(centered.shape[0] - 1, centered.shape[1])
        n_components = min(int(n_eof), max_rank)
        if n_components:
            _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
            components = vt[:n_components]
            eof_scale = singular_values[:n_components] / np.sqrt(max(fit_n - 1, 1))
            eof_scale = np.where(eof_scale > 0, eof_scale, 1.0)
        else:
            components = np.empty((0, centered.shape[1]), dtype=np.float64)
            eof_scale = np.empty((0,), dtype=np.float64)
        functional_tuple = tuple(functionals)
        if functional_tuple:
            matrix = np.column_stack([spec.transform(fit_fields) for spec in functional_tuple])
            functional_mean = matrix.mean(axis=0)
            functional_scale = matrix.std(axis=0)
            functional_scale = np.where(functional_scale > 0, functional_scale, 1.0)
        else:
            functional_mean = np.empty((0,), dtype=np.float64)
            functional_scale = np.empty((0,), dtype=np.float64)
        return cls(
            functionals=functional_tuple,
            pixel_mean=pixel_mean.astype(np.float64),
            eof_components=components.astype(np.float64),
            functional_mean=functional_mean.astype(np.float64),
            functional_scale=functional_scale.astype(np.float64),
            eof_scale=eof_scale.astype(np.float64),
            fit_indices=fit_indices.astype(np.int64),
            sample_shape=tuple(int(value) for value in fields.shape[1:]),
            fit_source=fit_source,
            source_fingerprint=source_digest.hexdigest()[:16],
            batch_size=int(batch_size),
        )

    def transform(self, fields: np.ndarray) -> np.ndarray:
        fields = np.asarray(fields)
        if tuple(fields.shape[1:]) != self.sample_shape:
            raise ValueError("Fields do not match the sample shape used to fit the feature space.")
        parts: list[np.ndarray] = []
        if self.functionals:
            matrix = np.column_stack([spec.transform(fields) for spec in self.functionals])
            parts.append((matrix - self.functional_mean) / self.functional_scale)
        if self.eof_components.shape[0]:
            coefficients = np.empty(
                (fields.shape[0], self.eof_components.shape[0]), dtype=np.float64
            )
            for start in range(0, fields.shape[0], self.batch_size):
                stop = min(start + self.batch_size, fields.shape[0])
                flat = np.asarray(fields[start:stop], dtype=np.float64).reshape(stop - start, -1)
                coefficients[start:stop] = (flat - self.pixel_mean) @ self.eof_components.T
            parts.append(coefficients / self.eof_scale)
        if not parts:
            raise ValueError("The feature space has neither functionals nor EOF components.")
        output = np.concatenate(parts, axis=1)
        if not np.all(np.isfinite(output)):
            raise ValueError("Feature transform produced non-finite values.")
        return output

    @property
    def provenance_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.pixel_mean.tobytes())
        digest.update(self.eof_components.tobytes())
        digest.update(self.functional_mean.tobytes())
        digest.update(self.functional_scale.tobytes())
        digest.update(self.eof_scale.tobytes())
        digest.update(self.fit_indices.tobytes())
        digest.update(str(self.sample_shape).encode())
        digest.update(self.fit_source.encode())
        digest.update(self.source_fingerprint.encode())
        return digest.hexdigest()[:16]

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "gauge_climate.feature_space.v1",
            "fit_source": self.fit_source,
            "source_fingerprint": self.source_fingerprint,
            "sample_shape": list(self.sample_shape),
            "fit_indices": self.fit_indices.tolist(),
            "functional_ids": [functional.id for functional in self.functionals],
            "n_eof": int(self.eof_components.shape[0]),
            "provenance_hash": self.provenance_hash,
        }

    def registered_metrics(
        self,
        train_fields: np.ndarray,
        *,
        density_k: int = 5,
        bandwidth_seed: int = 1789,
    ):
        """Build core feature metrics with every fit choice learned on train."""

        from .metrics import (
            coverage_metric,
            density_metric,
            diversity_metric,
            median_bandwidth,
            mmd_metric,
        )

        train_features = self.transform(train_fields)
        bandwidth = median_bandwidth(train_features, seed=bandwidth_seed)
        provenance = {
            "feature_space_hash": self.provenance_hash,
            "fit_source": self.fit_source,
            "bandwidth_fit_source": "train",
        }
        feature_transform = self.transform
        metrics = (
            mmd_metric(feature_transform, bandwidth=bandwidth),
            density_metric(feature_transform, k=density_k),
            coverage_metric(feature_transform, k=density_k),
            diversity_metric(feature_transform),
        )
        # Dataclasses are frozen; construct copies with merged provenance.
        from dataclasses import replace

        return tuple(
            replace(metric, provenance={**dict(metric.provenance), **provenance})
            for metric in metrics
        )
