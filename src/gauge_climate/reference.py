from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .types import EvaluationConfig, MetricSpec, SamplingEstimand


@dataclass(frozen=True)
class ReferenceDraw:
    pseudo_indices: np.ndarray
    real_indices: np.ndarray
    generated_indices: np.ndarray
    blocks_a: tuple[str, ...]
    blocks_b: tuple[str, ...]
    frame_count_a: int
    frame_count_b: int


@dataclass(frozen=True)
class ReferencePlan:
    draws: tuple[ReferenceDraw, ...]
    n_eval: int
    estimand: SamplingEstimand
    seed: int
    n_blocks: int

    def partition_manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "blocks_a": list(draw.blocks_a),
                "blocks_b": list(draw.blocks_b),
                "frame_count_a": draw.frame_count_a,
                "frame_count_b": draw.frame_count_b,
                "pseudo_indices": draw.pseudo_indices.tolist(),
                "real_indices": draw.real_indices.tolist(),
                "generated_indices": draw.generated_indices.tolist(),
            }
            for draw in self.draws
        ]


@dataclass(frozen=True)
class ReferenceEvaluation:
    reference_values: np.ndarray
    model_values: np.ndarray

    @property
    def reference_q025(self) -> float:
        return float(np.quantile(self.reference_values, 0.025))

    @property
    def reference_q50(self) -> float:
        return float(np.quantile(self.reference_values, 0.50))

    @property
    def reference_q95(self) -> float:
        return float(np.quantile(self.reference_values, 0.95))

    @property
    def reference_q975(self) -> float:
        return float(np.quantile(self.reference_values, 0.975))

    @property
    def model_median(self) -> float:
        return float(np.median(self.model_values))

    def quantile(self, probability: float) -> float:
        if not 0 < probability < 1:
            raise ValueError("Reference probability must be between zero and one.")
        return float(np.quantile(self.reference_values, probability))


def _stable_unique(values: np.ndarray) -> list[str]:
    return list(dict.fromkeys(str(value) for value in np.asarray(values).tolist()))


def draw_real_indices(
    eligible_indices: np.ndarray,
    blocks: np.ndarray,
    allowed_blocks: list[str] | tuple[str, ...],
    *,
    n: int,
    estimand: SamplingEstimand,
    rng: np.random.Generator,
) -> np.ndarray:
    eligible_indices = np.asarray(eligible_indices, dtype=np.int64)
    block_strings = np.asarray([str(value) for value in np.asarray(blocks)], dtype=object)
    allowed = set(str(value) for value in allowed_blocks)
    candidates = eligible_indices[np.asarray([block_strings[i] in allowed for i in eligible_indices])]
    if candidates.size == 0:
        raise ValueError("No eligible frames remain in the selected block half.")
    if estimand is SamplingEstimand.FRAME_WEIGHTED:
        return rng.choice(candidates, size=n, replace=True).astype(np.int64)

    by_block: dict[str, list[int]] = {}
    for index in candidates.tolist():
        by_block.setdefault(str(block_strings[index]), []).append(int(index))
    block_ids = list(by_block)
    if not block_ids:
        raise ValueError("No eligible blocks remain for block-balanced sampling.")
    selected_blocks = rng.choice(np.asarray(block_ids, dtype=object), size=n, replace=True)
    output = [int(rng.choice(by_block[str(block)])) for block in selected_blocks]
    return np.asarray(output, dtype=np.int64)


def build_reference_plan(
    *,
    real_indices: np.ndarray,
    real_blocks: np.ndarray,
    n_generated: int,
    config: EvaluationConfig,
    seed: int,
) -> ReferencePlan:
    real_indices = np.asarray(real_indices, dtype=np.int64)
    if real_indices.size < 2:
        raise ValueError("At least two real samples are required.")
    block_strings = np.asarray([str(value) for value in np.asarray(real_blocks)], dtype=object)
    eligible_blocks = _stable_unique(block_strings[real_indices])
    minimum = 2 * config.min_blocks_per_half
    if len(eligible_blocks) < minimum:
        raise ValueError(
            f"Reference construction requires at least {minimum} eligible blocks; "
            f"found {len(eligible_blocks)}."
        )
    n_eval = n_generated if config.n_eval is None else int(config.n_eval)
    if n_eval > n_generated:
        raise ValueError(
            f"n_eval={n_eval} exceeds the generated pool of {n_generated}; "
            "GAUGE does not silently resample generated outputs."
        )
    rng = np.random.default_rng(seed)
    block_counts = {
        block: int(np.sum(block_strings[real_indices] == block))
        for block in eligible_blocks
    }
    draws: list[ReferenceDraw] = []
    for _ in range(config.n_resamples):
        # For the frame-weighted estimand, select the best of several random
        # whole-block partitions by total frame balance. This keeps dependence
        # blocks intact while preventing one half from representing a radically
        # different frame-mass distribution merely because block sizes differ.
        candidates = 1 if config.estimand is SamplingEstimand.BLOCK_BALANCED else 32
        best: tuple[float, np.ndarray] | None = None
        for _candidate in range(candidates):
            shuffled = np.asarray(eligible_blocks, dtype=object).copy()
            rng.shuffle(shuffled)
            midpoint = len(shuffled) // 2
            if config.estimand is SamplingEstimand.BLOCK_BALANCED:
                score = 0.0
            else:
                mass_a = sum(block_counts[str(value)] for value in shuffled[:midpoint])
                mass_b = sum(block_counts[str(value)] for value in shuffled[midpoint:])
                score = abs(mass_a - mass_b) / max(mass_a + mass_b, 1)
            if best is None or score < best[0]:
                best = (score, shuffled)
        assert best is not None
        selected = best[1]
        midpoint = len(selected) // 2
        blocks_a = tuple(str(value) for value in selected[:midpoint])
        blocks_b = tuple(str(value) for value in selected[midpoint:])
        if min(len(blocks_a), len(blocks_b)) < config.min_blocks_per_half:
            raise RuntimeError("A generated block split violated the configured minimum.")
        pseudo = draw_real_indices(
            real_indices,
            block_strings,
            blocks_a,
            n=n_eval,
            estimand=config.estimand,
            rng=rng,
        )
        reference = draw_real_indices(
            real_indices,
            block_strings,
            blocks_b,
            n=n_eval,
            estimand=config.estimand,
            rng=rng,
        )
        generated = rng.choice(n_generated, size=n_eval, replace=False).astype(np.int64)
        draws.append(
            ReferenceDraw(
                pseudo_indices=pseudo,
                real_indices=reference,
                generated_indices=generated,
                blocks_a=blocks_a,
                blocks_b=blocks_b,
                frame_count_a=sum(block_counts[value] for value in blocks_a),
                frame_count_b=sum(block_counts[value] for value in blocks_b),
            )
        )
    return ReferencePlan(
        draws=tuple(draws),
        n_eval=n_eval,
        estimand=config.estimand,
        seed=int(seed),
        n_blocks=len(eligible_blocks),
    )


def evaluate_reference_plan(
    metric: MetricSpec,
    *,
    real_values: np.ndarray,
    generated_values: np.ndarray,
    plan: ReferencePlan,
) -> ReferenceEvaluation:
    reference_values = np.empty(len(plan.draws), dtype=np.float64)
    model_values = np.empty(len(plan.draws), dtype=np.float64)
    for index, draw in enumerate(plan.draws):
        reference = real_values[draw.real_indices]
        reference_values[index] = metric.estimator(real_values[draw.pseudo_indices], reference)
        model_values[index] = metric.estimator(
            generated_values[draw.generated_indices], reference
        )
    if not np.all(np.isfinite(reference_values)) or not np.all(np.isfinite(model_values)):
        raise ValueError(f"Metric {metric.id!r} produced non-finite reference values.")
    return ReferenceEvaluation(reference_values=reference_values, model_values=model_values)


def reference_position(value: float, q50: float, q95: float) -> float | None:
    width = float(q95 - q50)
    if not np.isfinite(width) or width <= np.finfo(float).eps:
        return None
    return float((value - q50) / width)
