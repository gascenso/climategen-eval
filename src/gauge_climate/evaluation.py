from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import scipy

from .data import Dataset, normalized_blocks, validate_generated, validate_split
from .inference import (
    Interval,
    empirical_reference_percentile,
    reference_status,
    reference_summary_uncertainty,
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
    ControlStatus,
    EvaluationConfig,
    FunctionalSpec,
    MetricKind,
    MetricSpec,
    ReferenceStatus,
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
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _metric_values(metric: MetricSpec, fields: Any, *, batch_size: int) -> np.ndarray:
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
            raise ValueError(f"Metric {metric.id!r} transform produced non-finite values.")
        batches.append(values)
    return np.concatenate(batches, axis=0)


def _dataset_manifest(dataset: Dataset) -> dict[str, Any]:
    labels = None
    if dataset.labels is not None:
        unique, counts = np.unique(np.asarray(dataset.labels), return_counts=True)
        labels = {str(label): int(count) for label, count in zip(unique, counts, strict=True)}
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


def _generic_interval(reference: ReferenceEvaluation, interval: float) -> Interval:
    tail = (1.0 - interval) / 2.0
    return Interval(
        estimate=float(np.median(reference.model_values)),
        low=float(np.quantile(reference.model_values, tail)),
        high=float(np.quantile(reference.model_values, 1.0 - tail)),
    )


def _estimand_mean(values: np.ndarray, blocks: np.ndarray, estimand: Any) -> np.ndarray:
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


def _metric_control_status(
    metric_id: str, results: Mapping[str, Any] | None
) -> tuple[ControlStatus, list[str]]:
    registered: list[tuple[str, Mapping[str, Any]]] = []
    for lane_id, lane in (results or {}).items():
        if not isinstance(lane, Mapping):
            continue
        metric_results = lane.get("metrics", {})
        if (
            isinstance(metric_results, Mapping)
            and metric_id in metric_results
            and isinstance(metric_results[metric_id], Mapping)
        ):
            registered.append((str(lane_id), metric_results[metric_id]))
    if not registered:
        return ControlStatus.CONTROL_NOT_REGISTERED, []
    failed = [lane_id for lane_id, result in registered if result.get("passed") is False]
    if failed:
        return ControlStatus.CONTROL_FAILED, failed
    if any(
        result.get("reaches_reference_boundary") is True
        or (
            result.get("passed") is True
            and result.get("reference_boundary") is not None
            and result.get("reaches_reference_boundary") is None
        )
        for _, result in registered
    ):
        return ControlStatus.CONTROL_VALIDATED, []
    return ControlStatus.RESPONSIVE_BELOW_REFERENCE_RANGE, []


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
        comparison_low=None,
        comparison_high=None,
        reference_q025=None,
        reference_q50=None,
        reference_q95=None,
        reference_q975=None,
        reference_q50_mc_low=None,
        reference_q50_mc_high=None,
        reference_q95_mc_low=None,
        reference_q95_mc_high=None,
        reference_upper_boundary=None,
        reference_upper_mc_low=None,
        reference_upper_mc_high=None,
        reference_percentile=None,
        reference_gap=None,
        comparison_gap_low=None,
        comparison_gap_high=None,
        reference_gap_mc_low=None,
        reference_gap_mc_high=None,
        reference_status=ReferenceStatus.NOT_EVALUATED,
        control_status=ControlStatus.CONTROL_NOT_REGISTERED,
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
    conditional: bool | str = False,
    memorization_result: Mapping[str, Any] | None = None,
    sensitivity_results: Mapping[str, Any] | None = None,
    conditional_groups: Mapping[str, Sequence[Any]] | None = None,
) -> EvaluationReport:
    """Evaluate generated fields using the reference-aware GAUGE protocol."""

    config = EvaluationConfig() if config is None else config
    split_audit = validate_split(train, real)
    validate_generated(real, generated)
    if real.blocks is None:
        raise ValueError("Held-out real data must provide dependence-block ids.")
    real_blocks = normalized_blocks(real.blocks)
    metric_list = [functional_w1(functional) for functional in functionals]
    metric_list.extend(metrics)
    if not metric_list:
        raise ValueError("At least one functional or metric is required.")
    ids = [metric.id for metric in metric_list]
    if len(ids) != len(set(ids)):
        raise ValueError("Metric ids must be unique.")
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
        str(name): tuple(values) for name, values in (conditional_groups or {}).items()
    }
    label_mixture_audit: dict[str, Any] | None = None
    label_mixture_warning: str | None = None
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
        ordered_labels = sorted(real_labels, key=str)
        real_label_counts = np.asarray(
            [np.sum(real_label_array == label) for label in ordered_labels], dtype=float
        )
        generated_label_counts = np.asarray(
            [np.sum(generated_label_array == label) for label in ordered_labels],
            dtype=float,
        )
        real_label_proportions = real_label_counts / real_label_counts.sum()
        generated_label_proportions = generated_label_counts / generated_label_counts.sum()
        label_mixture_tv = float(
            0.5 * np.sum(np.abs(real_label_proportions - generated_label_proportions))
        )
        label_mixture_audit = {
            "policy": "as_provided",
            "labels": [_jsonable(label) for label in ordered_labels],
            "real_counts": real_label_counts.astype(int).tolist(),
            "generated_counts": generated_label_counts.astype(int).tolist(),
            "total_variation_distance": label_mixture_tv,
            "matched": bool(np.isclose(label_mixture_tv, 0.0)),
            "interpretation": (
                "When mixtures differ, global rows test the supplied dataset "
                "composition and must not be called conditional-model fidelity."
            ),
        }
        if not label_mixture_audit["matched"]:
            label_mixture_warning = (
                "Global real/generated label mixtures differ "
                f"(total-variation distance={label_mixture_tv:.3g}); global rows are "
                "as-provided composition checks, not conditional-fidelity conclusions."
            )
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
                raise ValueError(f"Conditional group {name!r} needs unique supported labels.")
            unsupported = sorted(str(value) for value in set(members) - real_labels)
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
    if label_mixture_warning is not None:
        warnings.append(label_mixture_warning)
    cells: list[CellReport] = []
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
                "draw_count": len(plan.draws),
                "n_eval": plan.n_eval,
                "n_blocks": plan.n_blocks,
                "estimand": plan.estimand.value,
                "manifest_sha256": plan.provenance_hash(),
                "preview": plan.partition_manifest(limit=3),
                "regeneration": (
                    "The complete plan is reproduced deterministically from the "
                    "reported seed, configuration, and ordered input manifests."
                ),
            }
        for metric in metric_list:
            primary_reference = metric.kind is MetricKind.FUNCTIONAL_W1
            diagnostic_draw_count = config.diagnostic_resamples_by_metric.get(
                metric.id,
                config.diagnostic_resamples,
            )
            metric_plan = (
                plan
                if primary_reference
                else replace(plan, draws=plan.draws[:diagnostic_draw_count])
            )
            real_values, all_generated_values = transformed[metric.id]
            generated_values = all_generated_values[generated_idx]
            try:
                reference = evaluate_reference_plan(
                    metric,
                    real_values=real_values,
                    generated_values=generated_values,
                    plan=metric_plan,
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
                    [np.mean(real_values[draw.real_indices], axis=0) for draw in metric_plan.draws]
                )
                generated_curves = np.stack(
                    [
                        np.mean(generated_values[draw.generated_indices], axis=0)
                        for draw in metric_plan.draws
                    ]
                )
                diagnostics[f"{group_name}:{metric.id}"] = {
                    "kind": "curve",
                    "metric": metric.id,
                    "stratum": group_name,
                    "units": metric.units,
                    "coordinate_label": metric.provenance.get("coordinate_label", "curve bin"),
                    "coordinate_values": metric.provenance.get("coordinate_values"),
                    "real_mean": _estimand_mean(
                        real_values[real_idx], real_blocks[real_idx], config.estimand
                    ).tolist(),
                    "generated_mean": np.median(generated_curves, axis=0).tolist(),
                    "generated_q025": np.quantile(generated_curves, 0.025, axis=0).tolist(),
                    "generated_q975": np.quantile(generated_curves, 0.975, axis=0).tolist(),
                    "reference_q025": np.quantile(reference_curves, 0.025, axis=0).tolist(),
                    "reference_q975": np.quantile(reference_curves, 0.975, axis=0).tolist(),
                    "provenance": dict(metric.provenance),
                }
            model_reference_value = reference.model_median
            reference_boundary = reference.quantile(config.reference_quantile)
            midpoint = len(reference.reference_values) // 2
            first_boundary = float(
                np.quantile(
                    reference.reference_values[:midpoint],
                    config.reference_quantile,
                )
            )
            second_boundary = float(
                np.quantile(
                    reference.reference_values[midpoint:],
                    config.reference_quantile,
                )
            )
            reference_scale = max(reference_boundary - reference.reference_q50, np.finfo(float).eps)
            stability_ratio = abs(first_boundary - second_boundary) / reference_scale
            stability_warning = None
            if stability_ratio > config.reference_stability_threshold:
                resample_parameter = (
                    "n_resamples"
                    if primary_reference
                    else (
                        f"diagnostic_resamples_by_metric[{metric.id!r}]"
                        if metric.id in config.diagnostic_resamples_by_metric
                        else "diagnostic_resamples"
                    )
                )
                stability_warning = (
                    f"Reference boundary split-half instability={stability_ratio:.3g} "
                    f"exceeds {config.reference_stability_threshold:.3g}; increase "
                    f"{resample_parameter}."
                )
            inference_method = "empirical_reference_calibration"
            interval = _generic_interval(reference, config.comparison_interval)
            summary_uncertainty = reference_summary_uncertainty(
                reference_values=reference.reference_values,
                model_values=reference.model_values,
                reference_quantile=config.reference_quantile,
                interval=config.comparison_interval,
                n_resamples=config.summary_bootstrap_resamples,
                seed=_derive_seed(config.seed, "summary", group_name, metric.id),
            )
            cell_reference_status = reference_status(model_reference_value, reference_boundary)
            gap = reference_position(
                model_reference_value, reference.reference_q50, reference_boundary
            )
            percentile = empirical_reference_percentile(
                model_reference_value,
                reference.reference_values,
            )
            position_tail = (1.0 - config.comparison_interval) / 2.0
            model_position_low = float(np.quantile(reference.model_values, position_tail))
            model_position_high = float(np.quantile(reference.model_values, 1.0 - position_tail))
            comparison_gap_low = reference_position(
                model_position_low, reference.reference_q50, reference_boundary
            )
            comparison_gap_high = reference_position(
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
                "reference_resamples_used": len(metric_plan.draws),
                "reference_resample_role": (
                    "primary" if primary_reference else "secondary_diagnostic"
                ),
                "reference_stability_ratio": stability_ratio,
                "reference_stability_threshold": config.reference_stability_threshold,
                "model_reference_q025": model_position_low,
                "model_reference_q975": model_position_high,
                "comparison_range": (
                    "central range across admissible held-out-real comparisons; "
                    "not generator-retraining or population uncertainty"
                ),
                "summary_mc_interval": (
                    "paired bootstrap over stored reference repetitions, "
                    "conditional on the fixed archive and generated bank"
                ),
            }
            if group_name == "global" and label_mixture_audit is not None:
                metric_provenance.update(
                    {
                        "global_label_mixture_policy": label_mixture_audit["policy"],
                        "global_label_mixture_total_variation": label_mixture_audit[
                            "total_variation_distance"
                        ],
                    }
                )
            if metric.kind is MetricKind.RATE:
                metric_provenance.update(
                    {
                        "raw_generated_rate": float(np.mean(generated_values)),
                        "raw_real_rate": float(np.mean(real_values[real_idx])),
                    }
                )
            control_status, control_failures = _metric_control_status(
                metric.id,
                sensitivity_results,
            )
            cell_warnings = []
            if stability_warning is not None:
                cell_reference_status = ReferenceStatus.REFERENCE_UNRESOLVED
                cell_warnings.append(stability_warning)
                warnings.append(f"{group_name}:{metric.id}: {stability_warning}")
            if control_failures:
                cell_warnings.append(
                    "Metric control interpretation failed for lane(s): "
                    + ", ".join(control_failures)
                )
            if gap is None:
                cell_reference_status = ReferenceStatus.REFERENCE_UNRESOLVED
                cell_warnings.append(
                    "The q95-minus-q50 real-reference width is numerically degenerate."
                )
            q50_mc_low, q50_mc_high = summary_uncertainty["reference_q50"]
            q95_mc_low, q95_mc_high = summary_uncertainty["reference_q95"]
            upper_mc_low, upper_mc_high = summary_uncertainty["reference_upper"]
            gap_mc_low, gap_mc_high = summary_uncertainty["reference_gap"]
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
                comparison_low=interval.low,
                comparison_high=interval.high,
                reference_q025=reference.reference_q025,
                reference_q50=reference.reference_q50,
                reference_q95=reference.reference_q95,
                reference_q975=reference.reference_q975,
                reference_q50_mc_low=q50_mc_low,
                reference_q50_mc_high=q50_mc_high,
                reference_q95_mc_low=q95_mc_low,
                reference_q95_mc_high=q95_mc_high,
                reference_upper_boundary=reference_boundary,
                reference_upper_mc_low=upper_mc_low,
                reference_upper_mc_high=upper_mc_high,
                reference_percentile=percentile,
                reference_gap=gap,
                comparison_gap_low=comparison_gap_low,
                comparison_gap_high=comparison_gap_high,
                reference_gap_mc_low=gap_mc_low,
                reference_gap_mc_high=gap_mc_high,
                reference_status=cell_reference_status,
                control_status=control_status,
                n_eval=plan.n_eval,
                n_real=len(real_idx),
                n_generated=len(generated_idx),
                n_blocks=plan.n_blocks,
                estimand=config.estimand,
                direction=direction,
                warnings=tuple(cell_warnings),
                provenance=metric_provenance,
            )
            cells.append(cell)

    metric_manifests = [_metric_manifest(metric) for metric in metric_list]
    analysis_design = {
        "conditional_individual_labels": bool(conditional),
        "conditional_groups": {
            name: list(values) for name, values in normalized_conditional_groups.items()
        },
        "global_label_mixture": label_mixture_audit,
    }
    hash_value = _config_hash(config, metric_manifests, analysis_design)
    metadata = {
        "package_version": "0.2.0",
        "method": "event-disjoint-empirical-real-reference-v3",
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
        "interpretation": {
            "reference_status": (
                "A reference-range crossing is descriptive, not a p-value, "
                "level-alpha test, or claim of population inequality."
            ),
            "reference_gap": (
                "Position relative to finite-archive real-reference variability "
                "at this N; not physical severity or a model score."
            ),
            "controls": (
                "Metric-specific controls establish response and invariance; "
                "they are not application-acceptability tolerances."
            ),
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
