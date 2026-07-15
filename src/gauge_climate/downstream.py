from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .data import Dataset, normalized_blocks
from .types import _jsonable


class Arm(str, Enum):
    A0_NO_AUG = "A0_no_aug"
    A1_WEIGHT = "A1_class_weight"
    A2_COPY = "A2_copy"
    A3_COPY_GEO = "A3_copy_geo"
    A4_GENERATED = "A4_generated"
    A5_GENERATED_GEO = "A5_generated_geo"
    A6_RESTORED_REAL = "A6_restored_real"
    A7_RESTORED_REAL_GEO = "A7_restored_real_geo"


@dataclass(frozen=True)
class Holdout:
    id: str
    blocks: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Holdout.id may not be empty.")
        normalized = tuple(str(value) for value in self.blocks)
        if not normalized:
            raise ValueError("A holdout must contain at least one block.")
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"Holdout {self.id!r} contains duplicate blocks.")
        object.__setattr__(self, "blocks", normalized)


@dataclass(frozen=True)
class ArmBatch:
    fields: np.ndarray
    labels: np.ndarray
    matching_strata: np.ndarray
    sample_weights: np.ndarray | None
    arm: Arm
    holdout_id: str
    metadata: Mapping[str, Any]


TaskCallback = Callable[[ArmBatch, int, str], Mapping[str, Any]]
TransformCallback = Callable[[np.ndarray, np.ndarray, np.random.Generator], np.ndarray]


@dataclass(frozen=True)
class LadderConfig:
    seeds: tuple[int, ...] = (101, 102, 103, 104, 105)
    bootstrap_resamples: int = 10_000
    confidence: float = 0.95
    alpha: float = 0.05
    random_seed: int = 1789
    audit_callback_determinism: bool = False

    def __post_init__(self) -> None:
        if len(self.seeds) < 2:
            raise ValueError("At least two paired downstream seeds are required.")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("Downstream seeds must be unique.")
        if self.bootstrap_resamples < 100:
            raise ValueError("bootstrap_resamples must be at least 100.")
        if not 0.5 < self.confidence < 1:
            raise ValueError("confidence must be between 0.5 and 1.")
        if not 0 < self.alpha < 0.5:
            raise ValueError("alpha must be between 0 and 0.5.")
        if not np.isclose(self.confidence, 1.0 - self.alpha):
            raise ValueError("confidence must equal 1 - alpha.")


@dataclass(frozen=True)
class LadderRecord:
    holdout: str
    seed: int
    arm: Arm
    metrics: Mapping[str, float]
    block_metrics: Mapping[str, Mapping[str, float]]
    counts_by_label: Mapping[str, int]
    counts_by_stratum: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "holdout": self.holdout,
            "seed": self.seed,
            "arm": self.arm.value,
            "metrics": {str(key): float(value) for key, value in self.metrics.items()},
            "block_metrics": {
                str(metric): {str(block): float(value) for block, value in blocks.items()}
                for metric, blocks in self.block_metrics.items()
            },
            "counts_by_label": dict(self.counts_by_label),
            "counts_by_stratum": dict(self.counts_by_stratum),
        }


@dataclass
class LadderReport:
    records: Sequence[LadderRecord]
    plans: Mapping[str, Any]
    metadata: Mapping[str, Any]
    schema: str = "gauge_climate.downstream.v1"

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({
            "schema": self.schema,
            "metadata": dict(self.metadata),
            "records": [record.to_dict() for record in self.records],
            "plans": dict(self.plans),
        })

    def to_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return output

    def contrast(
        self,
        *,
        metric: str,
        treatment: Arm,
        reference: Arm,
        delta_task: float,
        higher_is_better: bool,
        seed: int = 1789,
        bootstrap_resamples: int | None = None,
    ) -> dict[str, Any]:
        if delta_task <= 0:
            raise ValueError("delta_task must be positive.")
        lookup = {
            (record.holdout, record.seed, record.arm): record
            for record in self.records
        }
        holdouts = sorted({record.holdout for record in self.records})
        paired: dict[str, dict[int, list[float]]] = {}
        for holdout in holdouts:
            seeds = sorted(
                {
                    record.seed
                    for record in self.records
                    if record.holdout == holdout
                    and record.arm in {treatment, reference}
                }
            )
            values: dict[int, list[float]] = {}
            for paired_seed in seeds:
                left = lookup.get((holdout, paired_seed, treatment))
                right = lookup.get((holdout, paired_seed, reference))
                if left is None or right is None:
                    raise ValueError(
                        f"Missing paired arm for holdout={holdout}, seed={paired_seed}."
                    )
                if metric not in left.metrics or metric not in right.metrics:
                    raise KeyError(f"Metric {metric!r} is missing from a paired record.")
                left_blocks = left.block_metrics.get(metric, {})
                right_blocks = right.block_metrics.get(metric, {})
                if left_blocks or right_blocks:
                    if set(left_blocks) != set(right_blocks):
                        raise ValueError(
                            f"Paired test-block support differs for {holdout}, seed={paired_seed}."
                        )
                    raw_values = [
                        float(left_blocks[block]) - float(right_blocks[block])
                        for block in sorted(left_blocks)
                    ]
                else:
                    raw_values = [float(left.metrics[metric]) - float(right.metrics[metric])]
                values[paired_seed] = [
                    raw if higher_is_better else -raw for raw in raw_values
                ]
            paired[holdout] = values
        estimate = float(
            np.mean(
                [
                    np.mean([np.mean(values) for values in seeds.values()])
                    for seeds in paired.values()
                ]
            )
        )
        rng = np.random.default_rng(seed)
        count = bootstrap_resamples or int(self.metadata.get("bootstrap_resamples", 10_000))
        boot = np.empty(count, dtype=np.float64)
        holdout_array = np.asarray(holdouts, dtype=object)
        for index in range(count):
            selected_holdouts = rng.choice(holdout_array, size=len(holdouts), replace=True)
            holdout_means = []
            for holdout in selected_holdouts:
                seed_values = paired[str(holdout)]
                available_seeds = np.asarray(sorted(seed_values), dtype=np.int64)
                selected_seeds = rng.choice(
                    available_seeds, size=available_seeds.size, replace=True
                )
                seed_means = []
                for selected_seed in selected_seeds:
                    block_values = np.asarray(
                        seed_values[int(selected_seed)], dtype=np.float64
                    )
                    seed_means.append(
                        float(
                            np.mean(
                                rng.choice(
                                    block_values, size=block_values.size, replace=True
                                )
                            )
                        )
                    )
                holdout_means.append(float(np.mean(seed_means)))
            boot[index] = float(np.mean(holdout_means))
        confidence = float(self.metadata.get("confidence", 0.95))
        tail = (1.0 - confidence) / 2.0
        low, high = (float(value) for value in np.quantile(boot, [tail, 1.0 - tail]))
        alpha = float(self.metadata.get("alpha", 0.05))
        equivalence_low, equivalence_high = (
            float(value) for value in np.quantile(boot, [alpha, 1.0 - alpha])
        )
        equivalent = equivalence_low > -delta_task and equivalence_high < delta_task
        superior = equivalence_low > 0
        inferior = equivalence_high < 0
        if equivalent and superior:
            status = "SMALL_POSITIVE_WITHIN_MARGIN"
        elif equivalent and inferior:
            status = "SMALL_NEGATIVE_WITHIN_MARGIN"
        elif equivalent:
            status = "EQUIVALENT"
        elif superior:
            status = "SUPERIOR"
        elif inferior:
            status = "INFERIOR"
        else:
            status = "INCONCLUSIVE"
        return {
            "metric": metric,
            "treatment": treatment.value,
            "reference": reference.value,
            "orientation": "positive_is_better",
            "estimate": estimate,
            "ci_low": low,
            "ci_high": high,
            "equivalence_ci_low": equivalence_low,
            "equivalence_ci_high": equivalence_high,
            "equivalence_alpha": alpha,
            "delta_task": float(delta_task),
            "equivalent": equivalent,
            "superior": superior,
            "inferior": inferior,
            "status": status,
            "paired_values": paired,
        }


def _seed(base: int, *parts: str) -> int:
    digest = hashlib.sha256((str(base) + "|" + "|".join(parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def _counts(labels: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {str(label): int(count) for label, count in zip(unique, counts, strict=True)}


def _normalize_task_metrics(
    raw_metrics: Mapping[str, Any], holdout: Holdout, arm: Arm
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    metrics: dict[str, float] = {}
    block_metrics: dict[str, dict[str, float]] = {}
    for key, value in raw_metrics.items():
        metric_id = str(key)
        if isinstance(value, Mapping):
            block_values = {
                str(block): float(block_value) for block, block_value in value.items()
            }
            if not block_values:
                raise ValueError(f"Task callback returned no test blocks for {metric_id!r}.")
            unexpected = sorted(set(block_values) - set(holdout.blocks))
            missing_blocks = sorted(set(holdout.blocks) - set(block_values))
            if unexpected or missing_blocks:
                raise ValueError(
                    f"Task metric {metric_id!r} test-block support does not equal "
                    f"holdout {holdout.id!r} (missing={missing_blocks}, "
                    f"unexpected={unexpected})."
                )
            block_metrics[metric_id] = block_values
            metrics[metric_id] = float(np.mean(list(block_values.values())))
        else:
            metrics[metric_id] = float(value)
    if not metrics or not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError(f"Task callback returned invalid metrics for {arm.value}.")
    if not all(
        np.isfinite(value)
        for values in block_metrics.values()
        for value in values.values()
    ):
        raise ValueError(f"Task callback returned non-finite block metrics for {arm.value}.")
    return metrics, block_metrics


class Ladder:
    """Framework-agnostic, leakage-audited eight-arm augmentation ladder."""

    def __init__(
        self,
        *,
        task: TaskCallback,
        task_metadata: Mapping[str, Any],
        train: Dataset,
        holdouts: Sequence[Holdout],
        generated_by_holdout: Mapping[str, Dataset],
        generator_training_blocks_by_holdout: Mapping[str, Sequence[Any]],
        transform: TransformCallback | None = None,
        transform_metadata: Mapping[str, Any] | None = None,
        matching_strata: Callable[[np.ndarray], np.ndarray] | None = None,
        matching_metadata: Mapping[str, Any] | None = None,
        config: LadderConfig = LadderConfig(),
    ) -> None:
        if train.blocks is None or train.labels is None:
            raise ValueError("The downstream training Dataset requires blocks and labels.")
        if len(holdouts) < 1:
            raise ValueError("At least one holdout is required.")
        if not task_metadata:
            raise ValueError("task_metadata must freeze the downstream task and training protocol.")
        self.task = task
        self.task_metadata = dict(task_metadata)
        self.train = train
        self.holdouts = tuple(holdouts)
        self.generated_by_holdout = dict(generated_by_holdout)
        self.generator_training_blocks_by_holdout = {
            key: tuple(str(value) for value in values)
            for key, values in generator_training_blocks_by_holdout.items()
        }
        self.transform = transform
        if transform is not None and not transform_metadata:
            if getattr(transform, "__module__", None) == __name__ and getattr(
                transform, "__name__", None
            ) == "geo_default":
                transform_metadata = {
                    "policy": "gauge_geo_default_v1",
                    "rotations": "right_angles_shape_preserving",
                    "translations_pixels": [-2, 2],
                }
            else:
                raise ValueError(
                    "A custom transform callback requires frozen transform_metadata."
                )
        self.transform_metadata = dict(transform_metadata or {"policy": "none"})
        if matching_strata is not None and not matching_metadata:
            raise ValueError(
                "A custom matching_strata callback requires frozen matching_metadata."
            )
        self.matching_strata = matching_strata or (lambda labels: labels)
        self.matching_metadata = dict(
            matching_metadata or {"policy": "identity_task_label"}
        )
        self.config = config
        ids = [holdout.id for holdout in self.holdouts]
        if len(ids) != len(set(ids)):
            raise ValueError("Holdout ids must be unique.")
        train_blocks = set(normalized_blocks(train.blocks).tolist())
        for holdout in self.holdouts:
            if not set(holdout.blocks).issubset(train_blocks):
                raise ValueError(f"Holdout {holdout.id!r} contains blocks absent from train.")
            if holdout.id not in self.generated_by_holdout:
                raise ValueError(f"Missing generated pool for holdout {holdout.id!r}.")
            generated = self.generated_by_holdout[holdout.id]
            if generated.labels is None:
                raise ValueError("Every generated holdout pool must provide labels.")
            trained_on = set(self.generator_training_blocks_by_holdout.get(holdout.id, ()))
            leaked = sorted(trained_on & set(holdout.blocks))
            if leaked:
                raise ValueError(
                    f"Generator leakage for holdout {holdout.id!r}: trained on {leaked[:5]}."
                )
            if not trained_on:
                raise ValueError(
                    f"Generator training-block provenance is required for holdout {holdout.id!r}."
                )
            expected = train_blocks - set(holdout.blocks)
            missing = sorted(expected - trained_on)
            unexpected = sorted(trained_on - expected)
            if missing or unexpected:
                raise ValueError(
                    f"Generator provenance for {holdout.id!r} does not equal the restricted "
                    f"training partition (missing={missing[:5]}, unexpected={unexpected[:5]})."
                )

    def _augmented_rows(
        self,
        source_fields: np.ndarray,
        source_labels: np.ndarray,
        source_strata: np.ndarray,
        target_counts: Mapping[Any, int],
        *,
        rng: np.random.Generator,
        replace: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[int]]]:
        fields = []
        labels = []
        strata = []
        selected: dict[str, list[int]] = {}
        for label, target in target_counts.items():
            candidates = np.flatnonzero(source_strata == label)
            if target and candidates.size == 0:
                raise ValueError(f"No source rows are available for label {label!r}.")
            if target > candidates.size and not replace:
                raise ValueError(
                    f"Generated/restored pool has {candidates.size} rows for label {label!r}, "
                    f"below the required {target}."
                )
            indices = rng.choice(candidates, size=target, replace=replace).astype(np.int64)
            selected[str(label)] = indices.tolist()
            if target:
                fields.append(source_fields[indices])
                labels.append(source_labels[indices])
                strata.append(source_strata[indices])
        if not fields:
            return (
                np.empty((0, *source_fields.shape[1:]), dtype=source_fields.dtype),
                np.empty((0,), dtype=source_labels.dtype),
                np.empty((0,), dtype=source_strata.dtype),
                selected,
            )
        return np.concatenate(fields), np.concatenate(labels), np.concatenate(strata), selected

    def _make_batch(
        self,
        arm: Arm,
        *,
        holdout: Holdout,
        restricted_fields: np.ndarray,
        restricted_labels: np.ndarray,
        restricted_strata: np.ndarray,
        restored_fields: np.ndarray,
        restored_labels: np.ndarray,
        restored_strata: np.ndarray,
        generated_fields: np.ndarray,
        generated_labels: np.ndarray,
        generated_strata: np.ndarray,
        target_counts: Mapping[Any, int],
        arm_seed: int,
    ) -> tuple[ArmBatch, dict[str, Any]]:
        rng = np.random.default_rng(arm_seed)
        metadata: dict[str, Any] = {"arm_seed": arm_seed, "selected_indices": {}}
        weights = None
        if arm is Arm.A0_NO_AUG:
            fields, labels, strata = restricted_fields, restricted_labels, restricted_strata
        elif arm is Arm.A1_WEIGHT:
            fields, labels, strata = restricted_fields, restricted_labels, restricted_strata
            counts = _counts(strata)
            weights = np.asarray(
                [1.0 / counts[str(label)] for label in strata], dtype=np.float64
            )
            weights *= weights.size / weights.sum()
        else:
            if arm in {Arm.A2_COPY, Arm.A3_COPY_GEO}:
                extra_fields, extra_labels, extra_strata, selected = self._augmented_rows(
                    restricted_fields,
                    restricted_labels,
                    restricted_strata,
                    target_counts,
                    rng=rng,
                    replace=True,
                )
                source = "restricted_copy"
            elif arm in {Arm.A4_GENERATED, Arm.A5_GENERATED_GEO}:
                extra_fields, extra_labels, extra_strata, selected = self._augmented_rows(
                    generated_fields,
                    generated_labels,
                    generated_strata,
                    target_counts,
                    rng=rng,
                    replace=False,
                )
                source = "generated"
            else:
                extra_fields, extra_labels, extra_strata, selected = self._augmented_rows(
                    restored_fields,
                    restored_labels,
                    restored_strata,
                    target_counts,
                    rng=rng,
                    replace=False,
                )
                source = "restored_real"
            metadata["selected_indices"] = selected
            metadata["augmentation_source"] = source
            use_geometry = arm in {
                Arm.A3_COPY_GEO,
                Arm.A5_GENERATED_GEO,
                Arm.A7_RESTORED_REAL_GEO,
            }
            if use_geometry:
                if self.transform is None:
                    raise ValueError(f"Arm {arm.value} requires a declared transform callback.")
                if self.transform is geo_default:
                    extra_fields, trace = _geo_default_with_trace(
                        extra_fields, extra_labels, rng
                    )
                    metadata["transform_trace"] = trace
                else:
                    transformed = self.transform(extra_fields, extra_labels, rng)
                    if (
                        isinstance(transformed, tuple)
                        and len(transformed) == 2
                        and isinstance(transformed[1], Mapping)
                    ):
                        extra_fields = np.asarray(transformed[0])
                        metadata["transform_trace"] = {
                            **dict(transformed[1]),
                            "explicit_parameters": True,
                        }
                    else:
                        extra_fields = np.asarray(transformed)
                        metadata["transform_trace"] = {
                            "replay": "selected_indices + arm_seed + frozen callback",
                            "explicit_parameters": False,
                        }
                if extra_fields.shape[0] != extra_labels.shape[0]:
                    raise ValueError("Transform callback changed the number of augmented rows.")
                if not np.issubdtype(extra_fields.dtype, np.number):
                    raise TypeError("Transform callback must return numeric fields.")
                metadata["transform_applied"] = True
                contiguous = np.ascontiguousarray(extra_fields)
                metadata["transform_output_sha256"] = hashlib.sha256(
                    contiguous.view(np.uint8)
                ).hexdigest()
            else:
                metadata["transform_applied"] = False
            fields = np.concatenate([restricted_fields, extra_fields])
            labels = np.concatenate([restricted_labels, extra_labels])
            strata = np.concatenate([restricted_strata, extra_strata])
        batch = ArmBatch(
            fields=np.asarray(fields),
            labels=np.asarray(labels),
            matching_strata=np.asarray(strata),
            sample_weights=weights,
            arm=arm,
            holdout_id=holdout.id,
            metadata=metadata,
        )
        return batch, metadata

    def run(self) -> LadderReport:
        train_fields = self.train.as_numpy()
        train_labels = np.asarray(self.train.labels)
        train_strata = np.asarray(self.matching_strata(train_labels))
        if train_strata.shape != train_labels.shape:
            raise ValueError("matching_strata must return one stratum per task label.")
        if any(value is None for value in train_strata.tolist()):
            raise ValueError("matching_strata returned missing values.")
        train_blocks = normalized_blocks(self.train.blocks)
        records: list[LadderRecord] = []
        plans: dict[str, Any] = {}
        for holdout in self.holdouts:
            withheld_mask = np.isin(train_blocks, np.asarray(holdout.blocks, dtype=object))
            if not np.any(withheld_mask) or np.all(withheld_mask):
                raise ValueError(f"Holdout {holdout.id!r} creates an empty restricted/restored side.")
            restricted_fields = train_fields[~withheld_mask]
            restricted_labels = train_labels[~withheld_mask]
            restricted_strata = train_strata[~withheld_mask]
            restored_fields = train_fields[withheld_mask]
            restored_labels = train_labels[withheld_mask]
            restored_strata = train_strata[withheld_mask]
            unique, counts = np.unique(restored_strata, return_counts=True)
            target_counts = {
                label: int(count) for label, count in zip(unique, counts, strict=True)
            }
            generated = self.generated_by_holdout[holdout.id]
            generated_fields = generated.as_numpy()
            generated_labels = np.asarray(generated.labels)
            generated_strata = np.asarray(self.matching_strata(generated_labels))
            if generated_strata.shape != generated_labels.shape:
                raise ValueError("matching_strata changed the generated label shape.")
            plans[holdout.id] = {
                "withheld_blocks": list(holdout.blocks),
                "target_counts": {str(key): value for key, value in target_counts.items()},
                "generator_training_blocks": list(
                    self.generator_training_blocks_by_holdout[holdout.id]
                ),
                "arms": {},
            }
            for paired_seed in self.config.seeds:
                for arm in Arm:
                    arm_seed = _seed(
                        self.config.random_seed,
                        holdout.id,
                        str(paired_seed),
                        arm.value,
                    )
                    batch, arm_plan = self._make_batch(
                        arm,
                        holdout=holdout,
                        restricted_fields=restricted_fields,
                        restricted_labels=restricted_labels,
                        restricted_strata=restricted_strata,
                        restored_fields=restored_fields,
                        restored_labels=restored_labels,
                        restored_strata=restored_strata,
                        generated_fields=generated_fields,
                        generated_labels=generated_labels,
                        generated_strata=generated_strata,
                        target_counts=target_counts,
                        arm_seed=arm_seed,
                    )
                    raw_metrics = self.task(batch, int(paired_seed), holdout.id)
                    metrics, block_metrics = _normalize_task_metrics(
                        raw_metrics, holdout, arm
                    )
                    if (
                        self.config.audit_callback_determinism
                        and holdout is self.holdouts[0]
                        and paired_seed == self.config.seeds[0]
                        and arm is Arm.A0_NO_AUG
                    ):
                        repeated = _normalize_task_metrics(
                            self.task(batch, int(paired_seed), holdout.id), holdout, arm
                        )
                        if repeated != (metrics, block_metrics):
                            raise ValueError(
                                "Task callback is nondeterministic for an identical batch and seed."
                            )
                    records.append(
                        LadderRecord(
                            holdout=holdout.id,
                            seed=int(paired_seed),
                            arm=arm,
                            metrics=metrics,
                            block_metrics=block_metrics,
                            counts_by_label=_counts(batch.labels),
                            counts_by_stratum=_counts(batch.matching_strata),
                        )
                    )
                    plans[holdout.id]["arms"][f"{paired_seed}:{arm.value}"] = arm_plan
        return LadderReport(
            records=tuple(records),
            plans=plans,
            metadata={
                "seeds": list(self.config.seeds),
                "bootstrap_resamples": self.config.bootstrap_resamples,
                "confidence": self.config.confidence,
                "alpha": self.config.alpha,
                "primary_contrast": "A5_generated_geo - A3_copy_geo",
                "secondary_source_contrast": "A4_generated - A2_copy",
                "restored_real_role": "empirical_headroom_positive_control",
                "matching_policy": self.matching_metadata,
                "transform_policy": self.transform_metadata,
                "task": self.task_metadata,
                "callback_determinism_audited": self.config.audit_callback_determinism,
            },
        )


def _geo_default_with_trace(
    fields: np.ndarray, labels: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    """Shape-preserving right-angle rotations and integer translations.

    Arbitrary-angle interpolation is domain dependent and intentionally not a
    universal default. Users should replace this callback when other symmetries
    are scientifically justified.
    """

    del labels
    fields = np.asarray(fields)
    if fields.ndim not in (3, 4):
        raise ValueError("geo_default supports (N,H,W) or (N,C,H,W).")
    output = np.empty_like(fields)
    rotations = []
    translations = []
    for index, sample in enumerate(fields):
        # Odd quarter turns swap H and W. Keep the callback shape preserving for
        # rectangular grids; users may provide a domain-specific transform if a
        # coordinate-aware 90-degree rotation is scientifically appropriate.
        k = (
            int(rng.integers(0, 4))
            if sample.shape[-2] == sample.shape[-1]
            else int(rng.integers(0, 2)) * 2
        )
        rotated = np.rot90(sample, k=k, axes=(-2, -1))
        shift_y = int(rng.integers(-2, 3))
        shift_x = int(rng.integers(-2, 3))
        output[index] = np.roll(rotated, shift=(shift_y, shift_x), axis=(-2, -1))
        rotations.append(k)
        translations.append([shift_y, shift_x])
    return output, {
        "rotation_quarter_turns": rotations,
        "translation_pixels_yx": translations,
        "explicit_parameters": True,
    }


def geo_default(fields: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply the built-in geometry policy; the ladder persists its exact trace."""

    output, _ = _geo_default_with_trace(fields, labels, rng)
    return output
