from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

from .types import SamplingEstimand


@dataclass(frozen=True)
class RegionCopyingResult:
    region: int
    stratum: str | None
    n_generated: int
    n_real: int
    n_real_blocks: int
    effect: float
    ci_low: float
    ci_high: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _fit_kmeans(values: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    if k < 1 or k > values.shape[0]:
        raise ValueError("Invalid number of feature-space regions.")
    centers = values[rng.choice(values.shape[0], size=k, replace=False)].copy()
    for _ in range(50):
        labels = np.argmin(((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
        updated = centers.copy()
        for region in range(k):
            members = values[labels == region]
            if members.size:
                updated[region] = members.mean(axis=0)
        if np.allclose(updated, centers, rtol=1e-7, atol=1e-9):
            break
        centers = updated
    return centers


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = np.asarray(values)[order]
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    sorted_weights = sorted_weights / sorted_weights.sum()
    return float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), 0.5, side="left")])


def localized_copying_test(
    *,
    train_features: np.ndarray,
    real_features: np.ndarray,
    generated_features: np.ndarray,
    real_blocks: np.ndarray,
    real_strata: np.ndarray | None = None,
    generated_strata: np.ndarray | None = None,
    n_regions: int = 8,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    min_real_blocks: int = 5,
    seed: int = 1789,
    feature_provenance: Mapping[str, Any] | None = None,
    estimand: SamplingEstimand | str = SamplingEstimand.BLOCK_BALANCED,
) -> dict[str, Any]:
    """Localized generated/real nearest-to-training comparison.

    The effect is median(generated distance) - median(real distance). Negative
    values indicate generated queries lie closer to training data. Confidence
    intervals resample generated queries and whole held-out blocks. When query
    strata are supplied, comparisons are made within joint stratum-by-region
    cells so a label-mixture mismatch cannot masquerade as copying or separation.
    A Bonferroni simultaneous interval is used across all supported cells.
    """

    train = np.asarray(train_features, dtype=np.float64)
    real = np.asarray(real_features, dtype=np.float64)
    generated = np.asarray(generated_features, dtype=np.float64)
    blocks = np.asarray([str(value) for value in np.asarray(real_blocks)], dtype=object)
    estimand = SamplingEstimand(estimand)
    if train.ndim != 2 or real.ndim != 2 or generated.ndim != 2:
        raise ValueError("Copying test inputs must be 2-D feature matrices.")
    if len({train.shape[1], real.shape[1], generated.shape[1]}) != 1:
        raise ValueError("Copying test feature dimensions differ.")
    if real.shape[0] != blocks.size:
        raise ValueError("real_features and real_blocks must align.")
    if (real_strata is None) != (generated_strata is None):
        raise ValueError("real_strata and generated_strata must be supplied together.")
    real_stratum_values: np.ndarray | None = None
    generated_stratum_values: np.ndarray | None = None
    strata: list[str | None] = [None]
    if real_strata is not None and generated_strata is not None:
        real_stratum_values = np.asarray(
            [str(value) for value in np.asarray(real_strata)], dtype=object
        )
        generated_stratum_values = np.asarray(
            [str(value) for value in np.asarray(generated_strata)], dtype=object
        )
        if real_stratum_values.shape != (real.shape[0],):
            raise ValueError("real_strata and real_features must align.")
        if generated_stratum_values.shape != (generated.shape[0],):
            raise ValueError("generated_strata and generated_features must align.")
        real_support = set(real_stratum_values.tolist())
        generated_support = set(generated_stratum_values.tolist())
        if real_support != generated_support:
            raise ValueError("Real and generated copying-query strata have different support.")
        strata = sorted(real_support)
    rng = np.random.default_rng(seed)
    centers = _fit_kmeans(train, min(n_regions, train.shape[0]), rng)
    train_regions = np.argmin(((train[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
    real_regions = np.argmin(((real[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
    generated_regions = np.argmin(
        ((generated[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1
    )
    tree = cKDTree(train)
    real_dist = np.asarray(tree.query(real, k=1)[0], dtype=np.float64)
    generated_dist = np.asarray(tree.query(generated, k=1)[0], dtype=np.float64)
    supported: list[tuple[int, str | None]] = []
    for region in range(centers.shape[0]):
        for stratum in strata:
            real_mask = real_regions == region
            generated_mask = generated_regions == region
            if stratum is not None:
                assert real_stratum_values is not None
                assert generated_stratum_values is not None
                real_mask &= real_stratum_values == stratum
                generated_mask &= generated_stratum_values == stratum
            block_count = np.unique(blocks[real_mask]).size
            if real_mask.sum() and generated_mask.sum() and block_count >= min_real_blocks:
                supported.append((region, stratum))
    if not supported:
        return {
            "status": "UNDERPOWERED",
            "reason": "No feature-space region has enough real blocks and both query types.",
            "regions": [],
            "n_regions": int(centers.shape[0]),
            "n_supported_cells": 0,
            "query_stratification": "none" if strata == [None] else "supplied_labels",
            "feature_provenance": dict(feature_provenance or {}),
            "estimand": estimand.value,
        }
    simultaneous_alpha = alpha / len(supported)
    region_results: list[RegionCopyingResult] = []
    for region, stratum in supported:
        real_mask = real_regions == region
        generated_mask = generated_regions == region
        if stratum is not None:
            assert real_stratum_values is not None
            assert generated_stratum_values is not None
            real_mask &= real_stratum_values == stratum
            generated_mask &= generated_stratum_values == stratum
        r_dist = real_dist[real_mask]
        g_dist = generated_dist[generated_mask]
        r_blocks = blocks[real_mask]
        unique_blocks = np.asarray(list(dict.fromkeys(r_blocks.tolist())), dtype=object)
        frame_counts = {
            str(block): int(np.sum(r_blocks == block)) for block in unique_blocks
        }
        if estimand is SamplingEstimand.BLOCK_BALANCED:
            observed_weights = np.asarray(
                [
                    1.0 / (unique_blocks.size * frame_counts[str(block)])
                    for block in r_blocks
                ],
                dtype=np.float64,
            )
        else:
            observed_weights = np.full(r_dist.size, 1.0 / r_dist.size)
        effect = float(np.median(g_dist) - _weighted_median(r_dist, observed_weights))
        bootstrap = np.empty(n_resamples, dtype=np.float64)
        for index in range(n_resamples):
            selected_blocks = rng.choice(unique_blocks, size=unique_blocks.size, replace=True)
            generated_indices = rng.choice(g_dist.size, size=g_dist.size, replace=True)
            multiplicity = {
                str(block): int(np.sum(selected_blocks == block))
                for block in unique_blocks
            }
            if estimand is SamplingEstimand.BLOCK_BALANCED:
                real_weights = np.asarray(
                    [
                        multiplicity[str(block)]
                        / (unique_blocks.size * frame_counts[str(block)])
                        for block in r_blocks
                    ],
                    dtype=np.float64,
                )
            else:
                denominator = sum(
                    multiplicity[str(block)] * frame_counts[str(block)]
                    for block in unique_blocks
                )
                real_weights = np.asarray(
                    [multiplicity[str(block)] / denominator for block in r_blocks],
                    dtype=np.float64,
                )
            bootstrap[index] = float(
                np.median(g_dist[generated_indices])
                - _weighted_median(r_dist, real_weights)
            )
        low = float(np.quantile(bootstrap, simultaneous_alpha / 2.0))
        high = float(np.quantile(bootstrap, 1.0 - simultaneous_alpha / 2.0))
        status = "COPYING_DETECTED" if high < 0 else ("OFF_MANIFOLD" if low > 0 else "INCONCLUSIVE")
        region_results.append(
            RegionCopyingResult(
                region=region,
                stratum=stratum,
                n_generated=int(g_dist.size),
                n_real=int(r_dist.size),
                n_real_blocks=int(unique_blocks.size),
                effect=effect,
                ci_low=low,
                ci_high=high,
                status=status,
            )
        )
    statuses = {result.status for result in region_results}
    overall = (
        "COPYING_DETECTED"
        if "COPYING_DETECTED" in statuses
        else ("OFF_MANIFOLD" if "OFF_MANIFOLD" in statuses else "NO_LOCAL_DEPARTURE_DETECTED")
    )
    return {
        "status": overall,
        "effect_definition": "median generated-to-train NN minus median real-to-train NN",
        "status_semantics": {
            "COPYING_DETECTED": "generated queries are significantly closer to training",
            "OFF_MANIFOLD": (
                "legacy schema label: generated queries are significantly farther from "
                "training; this does not establish physical inadmissibility"
            ),
            "INCONCLUSIVE": "the simultaneous interval crosses zero",
        },
        "multiplicity": (
            "Bonferroni simultaneous intervals across supported query-stratum-by-region cells"
        ),
        "alpha": alpha,
        "regions": [result.to_dict() for result in region_results],
        "n_regions": int(centers.shape[0]),
        "n_supported_cells": len(supported),
        "query_stratification": "none" if strata == [None] else "supplied_labels",
        "seed": int(seed),
        "feature_provenance": dict(feature_provenance or {}),
        "estimand": estimand.value,
        "train_region_counts": np.bincount(train_regions, minlength=centers.shape[0]).tolist(),
    }
