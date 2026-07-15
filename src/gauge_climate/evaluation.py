from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import scipy

from .data import Dataset, normalized_blocks, validate_generated, validate_split
from .inference import (
    Interval,
    dkw_w1_interval,
    detection_status,
    functional_w1_interval,
    practical_status,
    shift_resolution,
    simultaneous_reference_boundaries,
)
from .metrics import MetricNotEvaluable, functional_w1
from .reference import (
    ReferenceEvaluation,
    build_reference_plan,
    evaluate_reference_plan,
    reference_position,
)
from .report import EvaluationReport
from .types import (
    CellReport,
    DetectionStatus,
    EvaluationConfig,
    FunctionalSpec,
    MetricKind,
    MetricSpec,
    PracticalStatus,
    SensitivityStatus,
    _jsonable,
)


def _derive_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256((str(seed) + "|" + "|".join(parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def _callable_name(value: Any) -> str:
    module = getattr(value, "__module__", "unknown")
    name = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))
    return f"{module}:{name}"


def _metric_manifest(metric: MetricSpec) -> dict[str, Any]:
    return {
        "id": metric.id,
        "kind": metric.kind.value,
        "pillar": metric.pillar,
        "units": metric.units,
        "description": metric.description,
        "tolerance": metric.tolerance,
        "transform": _callable_name(metric.transform),
        "estimator": _callable_name(metric.estimator),
        "functional": None
        if metric.functional is None
        else {
            "id": metric.functional.id,
            "description": metric.functional.description,
            "units": metric.functional.units,
            "support": metric.functional.support,
            "tail_referenced": metric.functional.tail_referenced,
            "function": _callable_name(metric.functional.function),
        },
        "provenance": dict(metric.provenance),
    }


def _config_hash(
    config: EvaluationConfig,
    manifests: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, Any],
) -> str:
    payload = {
        "config": config.to_dict(),
        "metrics": list(manifests),
        "analysis": dict(analysis),
    }
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _metric_values(
    metric: MetricSpec, fields: Any, *, batch_size: int
) -> np.ndarray:
    shape = getattr(fields, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError("Metric fields must expose shape (N, ...sample dimensions...).")
    batches = []
    for start in range(0, int(shape[0]), batch_size):
        stop = min(start + batch_size, int(shape[0]))
        values = np.asarray(metric.transform(np.asarray(fields[start:stop])))
        if values.ndim < 1 or values.shape[0] != stop - start:
            raise ValueError(
                f"Metric {metric.id!r} transform must preserve the sample axis; "
                f"got {values.shape} for a batch of {stop - start}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Metric {metric.id!r} transform produced non-finite values."
            )
        batches.append(values)
    return np.concatenate(batches, axis=0)


def _dataset_manifest(dataset: Dataset) -> dict[str, Any]:
    labels = None
    if dataset.labels is not None:
        unique, counts = np.unique(np.asarray(dataset.labels), return_counts=True)
        labels = {
            str(label): int(count) for label, count in zip(unique, counts, strict=True)
        }
    return {
        "name": dataset.name,
        "n_samples": dataset.n_samples,
        "sample_shape": list(dataset.sample_shape),
        "units": dataset.units,
        "n_blocks": None
        if dataset.blocks is None
        else int(np.unique(normalized_blocks(dataset.blocks)).size),
        "label_counts": labels,
        "sample_ids_supplied": dataset.sample_ids is not None,
        "metadata": dict(dataset.metadata or {}),
    }


def _generic_interval(reference: ReferenceEvaluation, confidence: float) -> Interval:
    tail = (1.0 - confidence) / 2.0
    return Interval(
        estimate=float(np.median(reference.model_values)),
        low=float(np.quantile(reference.model_values, tail)),
        high=float(np.quantile(reference.model_values, 1.0 - tail)),
        bootstrap_values=reference.model_values,
    )


def _estimand_mean(
    values: np.ndarray, blocks: np.ndarray, estimand: Any
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    blocks = np.asarray(blocks, dtype=object)
    if estimand.value == "frame_weighted":
        return np.mean(values, axis=0)
    unique_blocks = list(dict.fromkeys(str(value) for value in blocks.tolist()))
    block_means = [
        np.mean(values[np.asarray([str(value) == block for value in blocks])], axis=0)
        for block in unique_blocks
    ]
    return np.mean(np.stack(block_means), axis=0)


def _failed_sensitivity_lanes(
    metric_id: str, results: Mapping[str, Any] | None
) -> list[str]:
    failures = []
    for lane_id, lane in (results or {}).items():
        if not isinstance(lane, Mapping):
            continue
        metric_results = lane.get("metrics", {})
        if (
            isinstance(metric_results, Mapping)
            and metric_id in metric_results
            and isinstance(metric_results[metric_id], Mapping)
            and metric_results[metric_id].get("passed") is False
        ):
            failures.append(str(lane_id))
    return failures


def _thin_cell(
    metric: MetricSpec,
    *,
    stratum: str,
    n_eval: int,
    n_real: int,
    n_generated: int,
    n_blocks: int,
    config: EvaluationConfig,
    warning: str,
) -> CellReport:
    return CellReport(
        metric=metric.id,
        stratum=stratum,
        kind=metric.kind.value,
        pillar=metric.pillar,
        units=metric.units,
        estimate=None,
        ci_low=None,
        ci_high=None,
        reference_q025=None,
        reference_q50=None,
        reference_q95=None,
        reference_q975=None,
        reference_detection_boundary=None,
        reference_position=None,
        reference_position_low=None,
        reference_position_high=None,
        tolerance=metric.tolerance,
        shift_resolution=None,
        detection_status=DetectionStatus.NOT_EVALUATED,
        practical_status=(
            PracticalStatus.UNDERPOWERED
            if metric.tolerance is not None
            else PracticalStatus.NOT_APPLICABLE
        ),
        sensitivity_status=(
            SensitivityStatus.INADEQUATE
            if metric.tolerance is not None
            else SensitivityStatus.NOT_APPLICABLE
        ),
        n_eval=n_eval,
        n_real=n_real,
        n_generated=n_generated,
        n_blocks=n_blocks,
        estimand=config.estimand,
        warnings=(warning,),
        provenance=metric.provenance,
    )


def evaluate(
    *,
    train: Dataset,
    real: Dataset,
    generated: Dataset,
    functionals: Sequence[FunctionalSpec] = (),
    metrics: Sequence[MetricSpec] = (),
    config: EvaluationConfig | None = None,
    tolerances: Mapping[str, float] | None = None,
    conditional: bool | str = False,
    memorization_result: Mapping[str, Any] | None = None,
    sensitivity_results: Mapping[str, Any] | None = None,
    conditional_groups: Mapping[str, Sequence[Any]] | None = None,
) -> EvaluationReport:
    """Evaluate generated fields using the two-layer GAUGE Track-1 protocol."""

    config = EvaluationConfig() if config is None else config
    split_audit = validate_split(train, real)
    validate_generated(real, generated)
    if real.blocks is None:
        raise ValueError("Held-out real data must provide dependence-block ids.")
    real_blocks = normalized_blocks(real.blocks)
    requested_tolerances = dict(tolerances or {})
    metric_list: list[MetricSpec] = []
    for functional in functionals:
        tolerance = requested_tolerances.get(functional.id, functional.tolerance)
        if tolerance != functional.tolerance:
            functional = FunctionalSpec(
                id=functional.id,
                function=functional.function,
                units=functional.units,
                description=functional.description,
                tolerance=tolerance,
                support=functional.support,
                tail_referenced=functional.tail_referenced,
                direction_negative=functional.direction_negative,
                direction_positive=functional.direction_positive,
                provenance=functional.provenance,
            )
        metric_list.append(functional_w1(functional))
    metric_list.extend(metrics)
    if not metric_list:
        raise ValueError("At least one functional or metric is required.")
    ids = [metric.id for metric in metric_list]
    if len(ids) != len(set(ids)):
        raise ValueError("Metric ids must be unique.")
    unsupported_tolerances = [
        metric.id
        for metric in metric_list
        if metric.kind is not MetricKind.FUNCTIONAL_W1 and metric.tolerance is not None
    ]
    if unsupported_tolerances:
        raise ValueError(
            "Non-functional tolerance inference is not validated in v0.1; remove "
            f"tolerances for {unsupported_tolerances} or register a future validated method."
        )

    real_fields = real.fields
    generated_fields = generated.fields
    transformed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    transform_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for metric in metric_list:
        cache_key = id(metric.transform)
        if cache_key not in transform_cache:
            transform_cache[cache_key] = (
                _metric_values(
                    metric,
                    real_fields,
                    batch_size=config.transform_batch_size,
                ),
                _metric_values(
                    metric,
                    generated_fields,
                    batch_size=config.transform_batch_size,
                ),
            )
        transformed[metric.id] = transform_cache[cache_key]

    groups: list[tuple[str, np.ndarray, np.ndarray]] = [
        (
            "global",
            np.arange(real.n_samples, dtype=np.int64),
            np.arange(generated.n_samples, dtype=np.int64),
        )
    ]
    normalized_conditional_groups = {
        str(name): tuple(values)
        for name, values in (conditional_groups or {}).items()
    }
    if conditional or normalized_conditional_groups:
        if real.labels is None or generated.labels is None:
            raise ValueError("Conditional evaluation requires labels on real and generated data.")
        real_label_array = np.asarray(real.labels)
        generated_label_array = np.asarray(generated.labels)
        real_labels = set(real_label_array.tolist())
        generated_labels = set(generated_label_array.tolist())
        missing = sorted(str(value) for value in real_labels ^ generated_labels)
        if missing:
            raise ValueError(f"Conditional label support differs: {', '.join(missing)}")
        if conditional:
            groups.extend(
                (
                    str(label),
                    np.flatnonzero(real_label_array == label),
                    np.flatnonzero(generated_label_array == label),
                )
                for label in sorted(real_labels, key=str)
            )
        occupied_names = {name for name, _, _ in groups}
        for name, members in normalized_conditional_groups.items():
            if not name or name in occupied_names:
                raise ValueError(f"Conditional group name {name!r} is empty or duplicated.")
            if not members or len(members) != len(set(members)):
                raise ValueError(
                    f"Conditional group {name!r} needs unique supported labels."
                )
            unsupported = sorted(
                str(value) for value in set(members) - real_labels
            )
            if unsupported:
                raise ValueError(
                    f"Conditional group {name!r} has unsupported labels: {unsupported}."
                )
            groups.append(
                (
                    name,
                    np.flatnonzero(np.isin(real_label_array, members)),
                    np.flatnonzero(np.isin(generated_label_array, members)),
                )
            )
            occupied_names.add(name)

    warnings: list[str] = []
    cells: list[CellReport] = []
    reference_arrays: list[np.ndarray] = []
    reference_cell_indices: list[int] = []
    partitions: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    sensitivity_payload: dict[str, Any] = dict(sensitivity_results or {})
    for group_name, real_idx, generated_idx in groups:
        group_blocks = real_blocks[real_idx]
        n_blocks = int(np.unique(group_blocks).size)
        n_eval = len(generated_idx) if config.n_eval is None else int(config.n_eval)
        if n_eval > len(generated_idx):
            raise ValueError(
                f"Stratum {group_name!r}: n_eval={n_eval} exceeds generated pool "
                f"of {len(generated_idx)}."
            )
        if n_blocks < 2 * config.min_blocks_per_half:
            warning = (
                f"Stratum {group_name!r} has {n_blocks} blocks; at least "
                f"{2 * config.min_blocks_per_half} are required for the reference split."
            )
            warnings.append(warning)
            for metric in metric_list:
                cells.append(
                    _thin_cell(
                        metric,
                        stratum=group_name,
                        n_eval=n_eval,
                        n_real=len(real_idx),
                        n_generated=len(generated_idx),
                        n_blocks=n_blocks,
                        config=config,
                        warning=warning,
                    )
                )
            continue
        if n_blocks < 2 * config.warn_blocks_per_half:
            warnings.append(
                f"Stratum {group_name!r} has only {n_blocks} blocks; reference tails may be unstable."
            )
        plan_seed = _derive_seed(config.seed, "reference", group_name)
        plan = build_reference_plan(
            real_indices=real_idx,
            real_blocks=real_blocks,
            n_generated=len(generated_idx),
            config=config,
            seed=plan_seed,
        )
        if config.include_partitions:
            partitions[group_name] = {
                "seed": plan_seed,
                "algorithm": "gauge-matched-reference-v1",
                "partitions": plan.partition_manifest(),
            }
        for metric in metric_list:
            real_values, all_generated_values = transformed[metric.id]
            generated_values = all_generated_values[generated_idx]
            try:
                reference = evaluate_reference_plan(
                    metric,
                    real_values=real_values,
                    generated_values=generated_values,
                    plan=plan,
                )
            except MetricNotEvaluable as exc:
                warning = f"{group_name}:{metric.id} not evaluated: {exc}"
                warnings.append(warning)
                cells.append(
                    _thin_cell(
                        metric,
                        stratum=group_name,
                        n_eval=n_eval,
                        n_real=len(real_idx),
                        n_generated=len(generated_idx),
                        n_blocks=n_blocks,
                        config=config,
                        warning=warning,
                    )
                )
                continue
            if metric.kind is MetricKind.CURVE:
                if real_values.ndim != 2 or generated_values.ndim != 2:
                    raise ValueError(
                        f"Curve metric {metric.id!r} must transform to shape (N, bins)."
                    )
                reference_curves = np.stack(
                    [np.mean(real_values[draw.real_indices], axis=0) for draw in plan.draws]
                )
                generated_curves = np.stack(
                    [
                        np.mean(generated_values[draw.generated_indices], axis=0)
                        for draw in plan.draws
                    ]
                )
                diagnostics[f"{group_name}:{metric.id}"] = {
                    "kind": "curve",
                    "metric": metric.id,
                    "stratum": group_name,
                    "units": metric.units,
                    "coordinate_label": metric.provenance.get(
                        "coordinate_label", "curve bin"
                    ),
                    "coordinate_values": metric.provenance.get("coordinate_values"),
                    "real_mean": _estimand_mean(
                        real_values[real_idx], real_blocks[real_idx], config.estimand
                    ).tolist(),
                    "generated_mean": np.median(generated_curves, axis=0).tolist(),
                    "generated_q025": np.quantile(
                        generated_curves, 0.025, axis=0
                    ).tolist(),
                    "generated_q975": np.quantile(
                        generated_curves, 0.975, axis=0
                    ).tolist(),
                    "reference_q025": np.quantile(
                        reference_curves, 0.025, axis=0
                    ).tolist(),
                    "reference_q975": np.quantile(
                        reference_curves, 0.975, axis=0
                    ).tolist(),
                    "provenance": dict(metric.provenance),
                }
            model_reference_value = reference.model_median
            reference_boundary = reference.quantile(1.0 - config.alpha)
            midpoint = len(reference.reference_values) // 2
            first_boundary = float(
                np.quantile(reference.reference_values[:midpoint], 1.0 - config.alpha)
            )
            second_boundary = float(
                np.quantile(reference.reference_values[midpoint:], 1.0 - config.alpha)
            )
            reference_scale = max(
                reference_boundary - reference.reference_q50, np.finfo(float).eps
            )
            stability_ratio = abs(first_boundary - second_boundary) / reference_scale
            stability_warning = None
            if stability_ratio > config.reference_stability_tolerance:
                stability_warning = (
                    f"Reference boundary split-half instability={stability_ratio:.3g} "
                    f"exceeds {config.reference_stability_tolerance:.3g}; increase n_resamples."
                )
            inference_method = "reference_resampling_interval"
            interval_warning = None
            if metric.kind is MetricKind.FUNCTIONAL_W1:
                interval_seed = _derive_seed(config.seed, "interval", group_name, metric.id)
                inference_method = "cluster_bootstrap_diagnostic"
                if (
                    config.estimand.value == "block_balanced"
                    and metric.functional is not None
                    and metric.functional.support is not None
                ):
                    interval = dkw_w1_interval(
                        generated_values,
                        real_values[real_idx],
                        real_blocks[real_idx],
                        n_eval=plan.n_eval,
                        support=metric.functional.support,
                        alpha=config.alpha,
                        seed=interval_seed,
                    )
                    inference_method = "finite_sample_dkw_quantile_band"
                else:
                    interval = functional_w1_interval(
                        generated_values,
                        real_values[real_idx],
                        real_blocks[real_idx],
                        n_eval=plan.n_eval,
                        estimand=config.estimand,
                        n_resamples=config.bootstrap_resamples,
                        confidence=config.confidence,
                        seed=interval_seed,
                    )
                    interval_warning = (
                        "Tolerance conclusion withheld: finite-sample W1 inference requires "
                        "the block_balanced estimand and a declared finite functional support."
                    )
                resolution, shifts, powers = shift_resolution(
                    real_values=real_values,
                    plan=plan,
                    reference_boundary=reference_boundary,
                    target_power=config.shift_power,
                    grid_size=config.shift_grid_size,
                    max_draws=config.sensitivity_resamples,
                )
                sensitivity_payload[f"{group_name}:{metric.id}"] = {
                    "perturbation": "uniform_functional_shift",
                    "target_power": config.shift_power,
                    "n_resamples": min(config.sensitivity_resamples, len(plan.draws)),
                    "shift_resolution": resolution,
                    "shifts": shifts.tolist(),
                    "detection_probability": powers.tolist(),
                }
                tolerance = metric.tolerance
                if interval_warning is None:
                    practical, sensitivity = practical_status(
                        interval,
                        tolerance=tolerance,
                        shift_resolution=resolution,
                    )
                else:
                    practical = (
                        PracticalStatus.INCONCLUSIVE
                        if tolerance is not None
                        else PracticalStatus.NOT_APPLICABLE
                    )
                    sensitivity = (
                        SensitivityStatus.NOT_EVALUATED
                        if tolerance is not None
                        else SensitivityStatus.NOT_APPLICABLE
                    )
                # Detection and practical relevance answer different questions.
                # Detection is always calibrated against the finite-N perfect-generator
                # reference; the direct population-W1 bound is reserved for tolerance
                # inference below.
                cell_detection = detection_status(
                    model_reference_value, reference_boundary
                )
            else:
                interval = _generic_interval(reference, config.confidence)
                resolution = None
                tolerance = None
                practical = PracticalStatus.NOT_APPLICABLE
                sensitivity = SensitivityStatus.NOT_EVALUATED
                cell_detection = detection_status(
                    reference.model_median, reference_boundary
                )
            position = reference_position(
                model_reference_value, reference.reference_q50, reference_boundary
            )
            position_tail = (1.0 - config.confidence) / 2.0
            model_position_low = float(
                np.quantile(reference.model_values, position_tail)
            )
            model_position_high = float(
                np.quantile(reference.model_values, 1.0 - position_tail)
            )
            position_low = reference_position(
                model_position_low, reference.reference_q50, reference_boundary
            )
            position_high = reference_position(
                model_position_high, reference.reference_q50, reference_boundary
            )
            direction = (
                None
                if metric.direction is None
                else metric.direction(generated_values, real_values[real_idx])
            )
            metric_provenance = {
                **dict(metric.provenance),
                "inference_method": inference_method,
                "reference_model_median": model_reference_value,
                "reference_plan_seed": plan_seed,
                "reference_stability_ratio": stability_ratio,
                "reference_stability_tolerance": config.reference_stability_tolerance,
                "model_reference_q025": model_position_low,
                "model_reference_q975": model_position_high,
            }
            if metric.kind is MetricKind.RATE:
                metric_provenance.update(
                    {
                        "raw_generated_rate": float(np.mean(generated_values)),
                        "raw_real_rate": float(np.mean(real_values[real_idx])),
                    }
                )
            control_failures = _failed_sensitivity_lanes(metric.id, sensitivity_results)
            cell_warnings = [] if interval_warning is None else [interval_warning]
            if stability_warning is not None:
                cell_warnings.append(stability_warning)
                warnings.append(f"{group_name}:{metric.id}: {stability_warning}")
            if control_failures:
                cell_detection = DetectionStatus.NOT_EVALUATED
                practical = (
                    PracticalStatus.INCONCLUSIVE
                    if tolerance is not None
                    else PracticalStatus.NOT_APPLICABLE
                )
                sensitivity = SensitivityStatus.INADEQUATE
                cell_warnings.append(
                    "Metric interpretation withheld because sensitivity lane(s) failed: "
                    + ", ".join(control_failures)
                )
            cell = CellReport(
                metric=metric.id,
                stratum=group_name,
                kind=metric.kind.value,
                pillar=(
                    "conditional_fidelity"
                    if group_name != "global" and metric.pillar == "physical_fidelity"
                    else metric.pillar
                ),
                units=metric.units,
                estimate=interval.estimate,
                ci_low=interval.low,
                ci_high=interval.high,
                reference_q025=reference.reference_q025,
                reference_q50=reference.reference_q50,
                reference_q95=reference.reference_q95,
                reference_q975=reference.reference_q975,
                reference_detection_boundary=reference_boundary,
                reference_position=position,
                reference_position_low=position_low,
                reference_position_high=position_high,
                tolerance=tolerance,
                shift_resolution=resolution,
                detection_status=cell_detection,
                practical_status=practical,
                sensitivity_status=sensitivity,
                n_eval=plan.n_eval,
                n_real=len(real_idx),
                n_generated=len(generated_idx),
                n_blocks=plan.n_blocks,
                estimand=config.estimand,
                direction=direction,
                warnings=tuple(cell_warnings),
                provenance=metric_provenance,
            )
            cell_key = f"{group_name}:{metric.id}"
            if cell_key in config.simultaneous_family:
                reference_cell_indices.append(len(cells))
                reference_arrays.append(reference.reference_values)
            cells.append(cell)

    evaluated_family = {
        f"{cells[index].stratum}:{cells[index].metric}" for index in reference_cell_indices
    }
    missing_family = sorted(set(config.simultaneous_family) - evaluated_family)
    if missing_family:
        warnings.append(
            "Simultaneous family cells were not evaluable: " + ", ".join(missing_family)
        )
    simultaneous: dict[str, Any] | None = None
    if reference_arrays:
        simultaneous_boundaries = simultaneous_reference_boundaries(
            np.vstack(reference_arrays), alpha=config.alpha
        )
        required_resamples = int(np.ceil(len(reference_arrays) / config.alpha - 1))
        simultaneous = {
            "method": "bonferroni_reference_boundaries",
            "family_alpha": config.alpha,
            "family_size": len(reference_arrays),
            "required_resamples": required_resamples,
            "monte_carlo_tail_resolved": config.n_resamples >= required_resamples,
            "resolved": bool(np.all(np.isfinite(simultaneous_boundaries))),
            "cell_reference_position_boundaries": [
                None if not np.isfinite(value) else float(value)
                for value in simultaneous_boundaries
            ],
        }
        updated: list[CellReport] = []
        reference_indices = set(reference_cell_indices)
        family_index = 0
        for index, cell in enumerate(cells):
            if index not in reference_indices:
                updated.append(cell)
                continue
            raw_boundary = simultaneous_boundaries[family_index]
            family_index += 1
            payload = cell.__dict__.copy()
            if not np.isfinite(raw_boundary):
                payload["simultaneous_boundary"] = None
                payload["simultaneous_detection_status"] = DetectionStatus.NOT_EVALUATED
                if not simultaneous["monte_carlo_tail_resolved"]:
                    family_warning = (
                        "Simultaneous conclusion withheld: increase n_resamples to at least "
                        f"{simultaneous['required_resamples']} for this Bonferroni family."
                    )
                else:
                    family_warning = (
                        "Simultaneous conclusion withheld: the pointwise real-reference "
                        "position scale is degenerate for this cell."
                    )
                payload["warnings"] = tuple([*cell.warnings, family_warning])
                warnings.append(f"{cell.stratum}:{cell.metric}: {family_warning}")
            else:
                boundary = float(raw_boundary)
                payload["simultaneous_boundary"] = boundary
                payload["simultaneous_detection_status"] = (
                    DetectionStatus.DEPARTURE_DETECTED
                    if cell.reference_position is not None
                    and cell.reference_position > boundary
                    else (
                        cell.detection_status
                        if cell.reference_position is None
                        else DetectionStatus.NO_DEPARTURE_DETECTED
                    )
                )
            updated.append(CellReport(**payload))
        cells = updated

    metric_manifests = [_metric_manifest(metric) for metric in metric_list]
    analysis_design = {
        "conditional_individual_labels": bool(conditional),
        "conditional_groups": {
            name: list(values)
            for name, values in normalized_conditional_groups.items()
        },
    }
    hash_value = _config_hash(config, metric_manifests, analysis_design)
    metadata = {
        "gauge_version": "0.1.0",
        "method": "matched-reference-plus-bounded-w1-v1",
        "config_hash": hash_value,
        "config": config.to_dict(),
        "analysis": analysis_design,
        "metric_registry": metric_manifests,
        "estimand": config.estimand.value,
        "n_eval": config.n_eval if config.n_eval is not None else "generated_pool_per_stratum",
        "data": {
            "train": _dataset_manifest(train),
            "real": _dataset_manifest(real),
            "generated": _dataset_manifest(generated),
            **split_audit,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "simultaneous_reference": simultaneous,
        "interpretation": {
            "detection": "Non-detection is not evidence of equality.",
            "practical": "Tolerance conclusions use direct raw-distance intervals.",
            "reference_position": "Detectability at this N; not physical severity.",
        },
    }
    return EvaluationReport(
        metadata=metadata,
        cells=tuple(cells),
        warnings=tuple(warnings),
        memorization=memorization_result,
        sensitivity=sensitivity_payload,
        diagnostics=diagnostics,
        partitions=partitions,
    )
