#!/usr/bin/env python3
"""Execute the versioned TC v0.2 sensitivity, invariance, and copying controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import yaml
from scipy.stats import spearmanr

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from gauge_climate.adapters import tc_bt
from gauge_climate.adapters.tc_data import load_real_archive
from gauge_climate.features import TrainFittedFeatureSpace
from gauge_climate.lanes import (
    additive_shift_lane,
    checkerboard_artifact_lane,
    ControlRegistry,
    ControlRole,
    ControlSpec,
    center_jitter_lane,
    copyist,
    gaussian_blur_lane,
    phase_shuffle_fields,
    PerturbationSpec,
    radial_contrast_lane,
    radial_warp_lane,
    run_perturbation_sweep,
    threshold_band_warming_lane,
)
from gauge_climate.memorization import localized_copying_test
from gauge_climate.metrics import admissibility_metric, functional_w1
from gauge_climate.reference import build_reference_plan, draw_real_indices
from gauge_climate.types import EvaluationConfig, MetricSpec, SamplingEstimand


ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_sha256(root: Path) -> str:
    """Hash the complete importable GAUGE source tree, including file names."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def _mapping_hash(values: Mapping[str, Any]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stable_blocks(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        list(dict.fromkeys(str(value) for value in np.asarray(values).tolist())),
        dtype=object,
    )


def _materialize(fields: Any, indices: np.ndarray) -> np.ndarray:
    values = np.asarray(fields[np.asarray(indices, dtype=np.int64)], dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("A TC control subset contains non-finite fields.")
    return values


def _batch_transform(
    metric: MetricSpec,
    fields: Any,
    indices: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    batches = []
    indices = np.asarray(indices, dtype=np.int64)
    for start in range(0, indices.size, batch_size):
        batch = _materialize(fields, indices[start : start + batch_size])
        batches.append(np.asarray(metric.transform(batch)))
    return np.concatenate(batches, axis=0)


def _reference_boundaries(
    *,
    fields: Any,
    blocks: np.ndarray,
    pool_indices: np.ndarray,
    metrics: Sequence[MetricSpec],
    n_control: int,
    n_resamples: int,
    quantile: float,
    min_blocks_per_half: int,
    seed: int,
    batch_size: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    local_blocks = np.asarray(blocks)[pool_indices]
    config = EvaluationConfig(
        n_eval=n_control,
        n_resamples=n_resamples,
        reference_quantile=quantile,
        comparison_interval=quantile,
        summary_bootstrap_resamples=200,
        estimand=SamplingEstimand.BLOCK_BALANCED,
        seed=seed,
        min_blocks_per_half=min_blocks_per_half,
        warn_blocks_per_half=min_blocks_per_half,
    )
    plan = build_reference_plan(
        real_indices=np.arange(pool_indices.size, dtype=np.int64),
        real_blocks=local_blocks,
        n_generated=n_control,
        config=config,
        seed=seed,
    )
    boundaries = {}
    summaries = {}
    for metric in metrics:
        values = _batch_transform(metric, fields, pool_indices, batch_size=batch_size)
        discrepancies = np.asarray(
            [
                metric.estimator(values[draw.pseudo_indices], values[draw.real_indices])
                for draw in plan.draws
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(discrepancies)):
            raise ValueError(f"Control reference values are non-finite for {metric.id}.")
        boundaries[metric.id] = float(np.quantile(discrepancies, quantile))
        summaries[metric.id] = {
            "q50": float(np.quantile(discrepancies, 0.50)),
            "boundary": boundaries[metric.id],
            "quantile": quantile,
            "n_resamples": n_resamples,
        }
    return boundaries, summaries


def _binary_metric_control(
    *,
    control_id: str,
    source: np.ndarray,
    perturbed: np.ndarray,
    metrics: Sequence[MetricSpec],
    boundaries: Mapping[str, float],
    invariant: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    invariant_payload = dict(invariant or {})
    invariant_passed = bool(invariant_payload.get("passed", True))
    results = {}
    for metric in metrics:
        source_values = metric.transform(source)
        perturbed_values = metric.transform(perturbed)
        value = float(metric.estimator(perturbed_values, source_values))
        boundary = float(boundaries[metric.id])
        reaches = bool(value > boundary)
        results[metric.id] = {
            "value": value,
            "reference_boundary": boundary,
            "reaches_reference_boundary": reaches,
            "invariant_passed": invariant_passed,
            "passed": bool(reaches and invariant_passed),
        }
    return {
        "schema": "gauge_climate.sensitivity_lane.v1",
        "id": control_id,
        "expected_response": "positive_control_exceeds_real_reference_boundary",
        "invariant": invariant_payload,
        "metrics": results,
        "passed": all(value["passed"] for value in results.values()),
    }


def _phase_power_invariant(
    source: np.ndarray,
    shuffled: np.ndarray,
    *,
    batch_size: int,
    relative_threshold: float,
) -> dict[str, Any]:
    maximum = 0.0
    for start in range(0, source.shape[0], batch_size):
        stop = min(start + batch_size, source.shape[0])
        original_power = np.abs(np.fft.fft2(source[start:stop], axes=(-2, -1))) ** 2
        shuffled_power = np.abs(np.fft.fft2(shuffled[start:stop], axes=(-2, -1))) ** 2
        scale = np.maximum(
            np.max(original_power, axis=(-2, -1), keepdims=True),
            np.finfo(np.float64).tiny,
        )
        maximum = max(
            maximum,
            float(np.max(np.abs(shuffled_power - original_power) / scale)),
        )
    return {
        "id": "per_image_discrete_fourier_power_preserved",
        "maximum_relative_error": maximum,
        "relative_threshold": relative_threshold,
        "passed": bool(maximum <= relative_threshold),
    }


def _phase_safe_pair(
    source: np.ndarray,
    *,
    seed: int,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase-shuffle a range-safe representation without invoking clipping."""

    values = np.asarray(source, dtype=np.float64)
    shuffled = np.asarray(phase_shuffle_fields(values, seed=seed), dtype=np.float64)
    means = np.mean(values, axis=(-2, -1), keepdims=True)
    source_anomaly = values - means
    shuffled_anomaly = shuffled - means
    maximum = np.maximum(
        np.max(np.abs(source_anomaly), axis=(-2, -1), keepdims=True),
        np.max(np.abs(shuffled_anomaly), axis=(-2, -1), keepdims=True),
    )
    allowance = np.minimum(means - lower, upper - means)
    scale = np.minimum(
        1.0,
        0.98 * allowance / np.maximum(maximum, np.finfo(float).eps),
    )
    return means + scale * source_anomaly, means + scale * shuffled_anomaly


def _copyist_control(
    *,
    fields: Any,
    blocks: np.ndarray,
    group_a_indices: np.ndarray,
    group_b_indices: np.ndarray,
    feature_space: TrainFittedFeatureSpace,
    n_control: int,
    n_regions: int,
    n_resamples: int,
    min_real_blocks: int,
    seed: int,
    bt_min_k: float,
    bt_max_k: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    group_a_blocks = _stable_blocks(np.asarray(blocks)[group_a_indices])
    group_b_blocks = _stable_blocks(np.asarray(blocks)[group_b_indices])
    train_local = draw_real_indices(
        np.arange(group_a_indices.size, dtype=np.int64),
        np.asarray(blocks)[group_a_indices],
        group_a_blocks.tolist(),
        n=n_control,
        estimand=SamplingEstimand.BLOCK_BALANCED,
        rng=rng,
    )
    real_local = draw_real_indices(
        np.arange(group_b_indices.size, dtype=np.int64),
        np.asarray(blocks)[group_b_indices],
        group_b_blocks.tolist(),
        n=n_control,
        estimand=SamplingEstimand.BLOCK_BALANCED,
        rng=rng,
    )
    train_indices = group_a_indices[train_local]
    real_indices = group_b_indices[real_local]
    train_fields = np.clip(_materialize(fields, train_indices), bt_min_k, bt_max_k)
    real_fields = np.clip(_materialize(fields, real_indices), bt_min_k, bt_max_k)
    generated_fields = copyist(train_fields, n=n_control, seed=seed + 1)
    result = localized_copying_test(
        train_features=feature_space.transform(train_fields),
        real_features=feature_space.transform(real_fields),
        generated_features=feature_space.transform(generated_fields),
        real_blocks=np.asarray(blocks)[real_indices],
        n_regions=n_regions,
        n_resamples=n_resamples,
        comparison_interval=0.95,
        min_real_blocks=min_real_blocks,
        seed=seed,
        feature_provenance=feature_space.manifest,
        estimand=SamplingEstimand.BLOCK_BALANCED,
    )
    passed = result.get("status") == "LOCALIZED_CLOSENESS_OBSERVED"
    return {
        "schema": "gauge_climate.sensitivity_lane.v1",
        "id": "copyist",
        "expected_response": "localized closeness to training is observed",
        "metrics": {
            "localized_memorization": {
                "observed_status": result.get("status"),
                "passed": passed,
            }
        },
        "result": result,
        "sampling": {
            "n_train_reference": n_control,
            "n_real_query": n_control,
            "n_copyist": n_control,
            "train_index_sha256": _array_hash(train_indices),
            "real_index_sha256": _array_hash(real_indices),
        },
        "passed": passed,
    }


def _copy_fraction_control(
    *,
    train_fields: np.ndarray,
    real_fields: np.ndarray,
    real_blocks: np.ndarray,
    feature_space: TrainFittedFeatureSpace,
    fractions: Sequence[float],
    n_regions: int,
    n_resamples: int,
    min_real_blocks: int,
    seed: int,
) -> dict[str, Any]:
    train_features = feature_space.transform(train_fields)
    real_features = feature_space.transform(real_fields)
    rng = np.random.default_rng(seed)
    rows = []
    resolved_counts = []
    for index, fraction in enumerate(fractions):
        fraction = float(fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("Copy fractions must lie in (0, 1].")
        generated = real_fields.copy()
        n_copy = max(1, int(round(fraction * generated.shape[0])))
        targets = rng.choice(generated.shape[0], size=n_copy, replace=False)
        sources = rng.choice(train_fields.shape[0], size=n_copy, replace=True)
        generated[targets] = train_fields[sources]
        result = localized_copying_test(
            train_features=train_features,
            real_features=real_features,
            generated_features=feature_space.transform(generated),
            real_blocks=real_blocks,
            n_regions=n_regions,
            n_resamples=n_resamples,
            comparison_interval=0.95,
            min_real_blocks=min_real_blocks,
            seed=seed + index,
            feature_provenance=feature_space.manifest,
            estimand=SamplingEstimand.BLOCK_BALANCED,
        )
        closer_cells = sum(
            cell.get("status") == "CLOSER_TO_TRAINING" for cell in result.get("regions", [])
        )
        resolved = result.get("status") == "LOCALIZED_CLOSENESS_OBSERVED"
        resolved_counts.append(closer_cells)
        rows.append(
            {
                "copy_fraction": fraction,
                "n_copied": n_copy,
                "status": result.get("status"),
                "closer_to_training_cells": closer_cells,
                "resolved": resolved,
            }
        )
    rank_response = float(spearmanr(np.asarray(fractions, dtype=float), resolved_counts).statistic)
    monotone = bool(np.isfinite(rank_response) and rank_response >= 0.8)
    full_resolved = bool(rows[-1]["resolved"])
    passing_fractions = [row["copy_fraction"] for row in rows if row["resolved"]]
    return {
        "schema": "gauge_climate.copy_fraction_control.v1",
        "id": "copy_fraction",
        "expected_response": "localized closeness strengthens as exact replay fraction increases",
        "fractions": rows,
        "minimum_resolved_fraction": (
            None if not passing_fractions else float(min(passing_fractions))
        ),
        "rank_monotone_copy_response": monotone,
        "copy_fraction_rank_correlation": rank_response,
        "metrics": {
            "localized_memorization": {
                "passed": bool(monotone and full_resolved),
                "reaches_reference_boundary": None,
                "reference_boundary": 1.0,
            }
        },
        "passed": bool(monotone and full_resolved),
    }


def _resampling_mode_removal(
    fields: np.ndarray,
    severity: float,
    seed: int,
) -> np.ndarray:
    values = np.asarray(fields)
    retained_fraction = 1.0 - float(severity)
    if not 0.0 < retained_fraction <= 1.0:
        raise ValueError("Mode-removal retained fraction must lie in (0, 1].")
    score = np.mean(values, axis=tuple(range(1, values.ndim)))
    retained = np.argsort(score)[-max(2, int(round(retained_fraction * values.shape[0]))) :]
    rng = np.random.default_rng(seed)
    return values[rng.choice(retained, size=values.shape[0], replace=True)]


def _feature_scale(
    fields: np.ndarray,
    severity: float,
    seed: int,
    *,
    expansion: bool,
) -> np.ndarray:
    del seed
    values = np.asarray(fields)
    centre = np.mean(values, axis=0, keepdims=True)
    factor = 1.0 + float(severity) if expansion else 1.0 - float(severity)
    return centre + factor * (values - centre)


def _threshold_band_cooling(
    fields: np.ndarray,
    severity: float,
    seed: int,
    *,
    threshold: float,
    half_width: float,
) -> np.ndarray:
    del seed
    values = np.asarray(fields)
    output = values.copy()
    mask = np.abs(values - threshold) <= half_width
    output[mask] = output[mask] - float(severity)
    return output


def _boundary_spikes(
    fields: np.ndarray,
    severity: float,
    seed: int,
    *,
    lower: float,
    upper: float,
) -> np.ndarray:
    values = np.asarray(fields)
    output = values.copy()
    count = int(round(float(severity) * values.shape[0]))
    if count == 0:
        return output
    rng = np.random.default_rng(seed)
    selected = rng.permutation(values.shape[0])[:count]
    output[selected, 0, 0] = lower
    output[selected[::2], 0, 0] = upper
    return output


def _plot_controls(payload: Mapping[str, Any], path: Path) -> None:
    sweep_ids = [
        key
        for key, value in payload.items()
        if isinstance(value, Mapping) and "severities" in value and "metrics" in value
    ]
    extra_ids = ["phase_shuffle", "copy_fraction"]
    panel_count = len(sweep_ids) + len(extra_ids)
    columns = 3
    rows = int(np.ceil(panel_count / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(13.0, 3.2 * rows),
        constrained_layout=True,
    )
    flat_axes = np.atleast_1d(axes).reshape(-1)
    for axis, lane_id in zip(flat_axes, sweep_ids, strict=False):
        lane = payload[lane_id]
        severities = np.asarray(lane["severities"], dtype=float)
        annotations = []
        for metric_id, result in lane["metrics"].items():
            boundary = result.get("reference_boundary")
            denominator = boundary if boundary is not None and boundary > 0 else 1.0
            line = axis.plot(
                severities,
                np.asarray(result["values"], dtype=float) / denominator,
                marker="o",
                label=metric_id,
            )[0]
            crossing = result.get("first_reference_crossing_severity")
            if crossing is not None:
                axis.axvline(
                    float(crossing),
                    color=line.get_color(),
                    linestyle=":",
                    linewidth=0.9,
                )
            annotations.append(
                f"{metric_id}: ρ={result['rank_correlation']:.2f}; "
                f"monotone={'yes' if result['monotone_nondecreasing'] else 'no'}; "
                f"first cross={'—' if crossing is None else f'{float(crossing):g}'}"
            )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_title(lane_id.replace("_", " "))
        axis.set_xlabel("severity")
        axis.set_ylabel("discrepancy / reference boundary")
        axis.legend(fontsize=7)
        axis.text(
            0.02,
            0.98,
            "\n".join(annotations),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            color="#455a64",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
                "pad": 1.2,
            },
        )
    phase_axis = flat_axes[len(sweep_ids)]
    phase = payload["phase_shuffle"]
    phase_ids = list(phase["metrics"])
    phase_values = [
        phase["metrics"][key]["value"]
        / max(phase["metrics"][key]["reference_boundary"], np.finfo(float).eps)
        for key in phase_ids
    ]
    phase_axis.bar(np.arange(len(phase_ids)), phase_values)
    phase_axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    phase_axis.set_xticks(np.arange(len(phase_ids)), phase_ids, rotation=25, ha="right")
    phase_axis.set_ylabel("discrepancy / reference boundary")
    phase_axis.set_title("phase shuffle")
    copy_axis = flat_axes[len(sweep_ids) + 1]
    copy_rows = payload["copy_fraction"]["fractions"]
    copy_axis.plot(
        [row["copy_fraction"] for row in copy_rows],
        [row["closer_to_training_cells"] for row in copy_rows],
        marker="o",
        color="#7b3f61",
    )
    copy_axis.set_xscale("log")
    copy_axis.set_xlabel("exact replay fraction")
    copy_axis.set_ylabel("closer-to-training regions")
    copy_axis.set_title("memorization resolution")
    for axis in flat_axes[panel_count:]:
        axis.axis("off")
    for axis in flat_axes[:panel_count]:
        axis.grid(color="#dfe6e8", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-train-manifest",
        type=Path,
        default=ROOT / "data/tropical_cyclone_bt/real_train/manifest.json",
    )
    parser.add_argument(
        "--analysis-protocol",
        dest="analysis_protocol",
        type=Path,
        default=ROOT / "case_studies/tropical_cyclone_bt/analysis_protocol.yaml",
    )
    parser.add_argument(
        "--track1-config",
        type=Path,
        default=ROOT / "case_studies/tropical_cyclone_bt/track1_config.template.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(ROOT / "case_studies/tropical_cyclone_bt/control_results/sensitivity_lanes.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=(ROOT / "case_studies/tropical_cyclone_bt/control_results/sensitivity_lanes.pdf"),
    )
    args = parser.parse_args()

    protocol = _load_yaml(args.analysis_protocol.resolve())
    sensitivity = protocol.get("sensitivity_lanes", {})
    if sensitivity.get("status") != "VERSIONED":
        raise ValueError("TC sensitivity choices require a versioned analysis protocol.")
    controls = sensitivity.get("controls", {})
    design = sensitivity.get("design", {})
    required_design = {
        "seed",
        "control_block_count",
        "n_control",
        "reference_resamples",
        "reference_boundary_quantile",
        "minimum_blocks_per_half",
        "feature_fit_samples",
        "feature_mapping",
        "copyist_resamples",
        "copy_fraction_resamples",
        "batch_size",
        "phase_power_relative_threshold",
    }
    missing_design = sorted(required_design - set(design))
    if missing_design:
        raise ValueError(f"Preregistration lacks frozen control design: {missing_design}.")
    if design["feature_mapping"] != "exact_registered_training_fitted_representation_and_bandwidth":
        raise ValueError(
            "Feature controls must use the exact representation and bandwidth "
            "registered for the TC evaluation."
        )
    seed = int(design["seed"])
    n_blocks = int(design["control_block_count"])
    n_control = int(design["n_control"])
    real_train = load_real_archive(args.real_train_manifest.resolve(), target="category")
    train_blocks = np.asarray(real_train.blocks)
    unique_blocks = np.sort(_stable_blocks(train_blocks))
    if n_blocks % 2 or n_blocks > unique_blocks.size:
        raise ValueError("control_block_count must be even and available in training.")
    rng = np.random.default_rng(seed)
    selected_blocks = np.asarray(
        rng.choice(unique_blocks, size=n_blocks, replace=False), dtype=object
    )
    rng.shuffle(selected_blocks)
    half = n_blocks // 2
    group_a = set(selected_blocks[:half].tolist())
    group_b = set(selected_blocks[half:].tolist())
    pool_mask = np.asarray([str(value) in group_a | group_b for value in train_blocks])
    group_a_mask = np.asarray([str(value) in group_a for value in train_blocks])
    group_b_mask = np.asarray([str(value) in group_b for value in train_blocks])
    pool_indices = np.flatnonzero(pool_mask).astype(np.int64)
    group_a_indices = np.flatnonzero(group_a_mask).astype(np.int64)
    group_b_indices = np.flatnonzero(group_b_mask).astype(np.int64)

    track1 = _load_yaml(args.track1_config.resolve())
    adapter_config = tc_bt.TCBTConfig(**track1.get("adapter_options", {}))
    analysis_train_fields = tc_bt.PreprocessedFields(
        real_train.fields,
        adapter_config,
    )
    functionals = tc_bt.functionals(adapter_config)
    functional_metrics = {metric.id: metric for metric in map(functional_w1, functionals)}
    curve_metrics = {metric.id: metric for metric in tc_bt.curve_metrics(adapter_config)}
    lower, upper = adapter_config.bt_min_k, adapter_config.bt_max_k

    def boundary_saturation(fields: np.ndarray) -> np.ndarray:
        return np.any(
            np.isclose(fields, lower, rtol=0.0, atol=1e-6)
            | np.isclose(fields, upper, rtol=0.0, atol=1e-6),
            axis=tuple(range(1, fields.ndim)),
        )

    saturation_metric = admissibility_metric(
        boundary_saturation,
        id="preprocessing_boundary_saturation",
        description="Departure in analysis-boundary saturation rate.",
    )
    registry = {
        **functional_metrics,
        **curve_metrics,
        saturation_metric.id: saturation_metric,
    }
    expected_ids = (
        "w1:min_core_bt",
        "w1:p01_bt",
        "w1:cold_frac_220",
        "w1:eye_contrast",
        "w1:r_maxgrad",
        "w1:axisym",
        "w1:dav",
        "radial_profile_mae",
        "radial_log_psd_mse",
        "preprocessing_boundary_saturation",
    )
    source_local = draw_real_indices(
        np.arange(pool_indices.size, dtype=np.int64),
        train_blocks[pool_indices],
        selected_blocks.tolist(),
        n=n_control,
        estimand=SamplingEstimand.BLOCK_BALANCED,
        rng=np.random.default_rng(seed + 1),
    )
    source_indices = pool_indices[source_local]
    source_fields = np.clip(
        _materialize(analysis_train_fields, source_indices),
        adapter_config.bt_min_k,
        adapter_config.bt_max_k,
    )
    feature_node = track1["feature_metrics"]
    configured_feature_fit_samples = int(feature_node["max_fit_samples"])
    if configured_feature_fit_samples != int(design["feature_fit_samples"]):
        raise ValueError(
            "The control protocol feature-fit size must match the registered "
            "TC evaluation feature-fit size."
        )
    feature_space = TrainFittedFeatureSpace.fit(
        analysis_train_fields,
        functionals,
        n_eof=int(feature_node["n_eof"]),
        max_fit_samples=configured_feature_fit_samples,
        seed=int(feature_node.get("fit_seed", track1["evaluation"]["seed"])),
        batch_size=int(feature_node.get("batch_size", design["batch_size"])),
        fit_source="train",
    )
    feature_fit_fields = analysis_train_fields[feature_space.fit_indices]
    feature_metrics = feature_space.registered_metrics(
        train_features=feature_space.transform(feature_fit_fields),
        density_k=int(feature_node["density_k"]),
        bandwidth_seed=int(feature_node.get("bandwidth_seed", track1["evaluation"]["seed"])),
        max_samples_per_draw=int(feature_node["max_samples_per_draw"]),
    )
    registry.update({metric.id: metric for metric in feature_metrics})
    expected_ids = (
        *expected_ids,
        "feature_mmd2",
        "feature_density",
        "feature_coverage_deficit",
        "feature_diversity_departure",
    )
    expected_metrics = tuple(registry[key] for key in expected_ids)
    boundaries, reference_summary = _reference_boundaries(
        fields=analysis_train_fields,
        blocks=train_blocks,
        pool_indices=pool_indices,
        metrics=expected_metrics,
        n_control=n_control,
        n_resamples=int(design["reference_resamples"]),
        quantile=float(design["reference_boundary_quantile"]),
        min_blocks_per_half=int(design["minimum_blocks_per_half"]),
        seed=seed,
        batch_size=int(design["batch_size"]),
    )

    warm = additive_shift_lane(
        severities=controls["warm_shift"]["kelvin"],
        expected_metrics=("w1:min_core_bt", "w1:p01_bt"),
    )
    threshold_warming = threshold_band_warming_lane(
        severities=controls["threshold_warming"]["kelvin"],
        expected_metrics=("w1:cold_frac_220",),
        threshold=float(controls["threshold_warming"]["threshold_k"]),
        half_width=float(controls["threshold_warming"]["half_width_k"]),
    )
    threshold_cooling = PerturbationSpec(
        id="threshold_band_cooling",
        apply=lambda fields, severity, local_seed: _threshold_band_cooling(
            fields,
            severity,
            local_seed,
            threshold=float(controls["threshold_cooling"]["threshold_k"]),
            half_width=float(controls["threshold_cooling"]["half_width_k"]),
        ),
        severities=tuple(float(value) for value in controls["threshold_cooling"]["kelvin"]),
        expected_metrics=("w1:cold_frac_220",),
        provenance={
            "operation": "subtract severity from pixels within threshold band",
            "threshold": float(controls["threshold_cooling"]["threshold_k"]),
            "half_width": float(controls["threshold_cooling"]["half_width_k"]),
            "fit_source": "reserved_real",
        },
    )
    eye_contrast = radial_contrast_lane(
        severities=controls["eye_contrast"]["kelvin"],
        expected_metrics=("w1:eye_contrast",),
        pixel_size_km=adapter_config.pixel_size_km,
        eye_radius_km=adapter_config.eye_radius_km,
        annulus_inner_km=adapter_config.eyewall_inner_km,
        annulus_outer_km=adapter_config.eyewall_outer_km,
    )
    blur = gaussian_blur_lane(
        severities=controls["blur"]["sigma_pixels"],
        expected_metrics=("radial_log_psd_mse", "w1:r_maxgrad"),
    )
    jitter = center_jitter_lane(
        severities=controls["center_jitter"]["pixels"],
        expected_metrics=("radial_profile_mae", "w1:r_maxgrad"),
    )
    radial_warp = radial_warp_lane(
        severities=controls["radial_warp"]["fractional_dilation"],
        expected_metrics=("radial_profile_mae", "w1:r_maxgrad"),
    )
    checkerboard_unclipped = checkerboard_artifact_lane(
        severities=controls["checkerboard"]["kelvin"],
        expected_metrics=(
            "radial_log_psd_mse",
            "feature_mmd2",
        ),
    )
    checkerboard = PerturbationSpec(
        id=checkerboard_unclipped.id,
        apply=lambda fields, severity, local_seed: np.clip(
            checkerboard_unclipped.apply(fields, severity, local_seed),
            adapter_config.bt_min_k,
            adapter_config.bt_max_k,
        ),
        severities=checkerboard_unclipped.severities,
        expected_metrics=checkerboard_unclipped.expected_metrics,
        provenance={
            **dict(checkerboard_unclipped.provenance or {}),
            "clip_after_perturbation_k": [
                adapter_config.bt_min_k,
                adapter_config.bt_max_k,
            ],
        },
    )
    eof_pattern = (
        feature_space.eof_components[0].reshape(feature_space.sample_shape)
        * feature_space.eof_scale[0]
    )
    eof_excursion = PerturbationSpec(
        id="feature_eof_excursion",
        apply=lambda fields, severity, local_seed: np.clip(
            np.asarray(fields) + float(severity) * eof_pattern,
            adapter_config.bt_min_k,
            adapter_config.bt_max_k,
        ),
        severities=tuple(
            float(value) for value in controls["feature_eof_excursion"]["standard_deviations"]
        ),
        expected_metrics=("feature_mmd2", "feature_density"),
        provenance={
            "operation": "shift all samples along the leading training-fit EOF",
            "eof_index": 0,
        },
    )
    mode_removal = PerturbationSpec(
        id="feature_mode_removal",
        apply=_resampling_mode_removal,
        severities=tuple(
            1.0 - float(value) for value in controls["feature_mode_removal"]["retained_fraction"]
        ),
        expected_metrics=("feature_coverage_deficit",),
        provenance={"operation": "retain warmest field-mean mode and resample"},
    )
    contraction = PerturbationSpec(
        id="feature_contraction",
        apply=lambda fields, severity, local_seed: _feature_scale(
            fields,
            severity,
            local_seed,
            expansion=False,
        ),
        severities=tuple(float(value) for value in controls["feature_contraction"]["contraction"]),
        expected_metrics=("feature_diversity_departure",),
        provenance={"operation": "contract fields about reserved-real mean"},
    )
    expansion = PerturbationSpec(
        id="feature_expansion",
        apply=lambda fields, severity, local_seed: _feature_scale(
            fields,
            severity,
            local_seed,
            expansion=True,
        ),
        severities=tuple(float(value) for value in controls["feature_expansion"]["expansion"]),
        expected_metrics=("feature_diversity_departure",),
        provenance={"operation": "expand fields about reserved-real mean"},
    )
    boundary_spikes = PerturbationSpec(
        id="boundary_spikes",
        apply=lambda fields, severity, local_seed: _boundary_spikes(
            fields,
            severity,
            local_seed,
            lower=adapter_config.bt_min_k,
            upper=adapter_config.bt_max_k,
        ),
        severities=tuple(float(value) for value in controls["boundary_spikes"]["sample_fraction"]),
        expected_metrics=("preprocessing_boundary_saturation",),
        provenance={
            "operation": "set one pixel in a declared fraction of samples to a clip boundary"
        },
    )
    sweep_registry = tuple(registry.values())
    sweep_specs = (
        ("warm_shift", warm),
        ("threshold_warming", threshold_warming),
        ("threshold_cooling", threshold_cooling),
        ("eye_contrast", eye_contrast),
        ("blur", blur),
        ("center_jitter", jitter),
        ("radial_warp", radial_warp),
        ("checkerboard", checkerboard),
        ("feature_eof_excursion", eof_excursion),
        ("feature_mode_removal", mode_removal),
        ("feature_contraction", contraction),
        ("feature_expansion", expansion),
        ("boundary_spikes", boundary_spikes),
    )
    sweep_results = {}
    for lane_id, spec in sweep_specs:
        result = run_perturbation_sweep(
            spec,
            source_fields=source_fields,
            reference_fields=source_fields,
            metrics=sweep_registry,
            reference_boundaries=boundaries,
            seed=seed,
        )
        result["id"] = lane_id
        sweep_results[lane_id] = result
    phase_source_fields, shuffled_fields = _phase_safe_pair(
        source_fields,
        seed=seed,
        lower=adapter_config.bt_min_k,
        upper=adapter_config.bt_max_k,
    )
    phase_invariant = _phase_power_invariant(
        phase_source_fields,
        shuffled_fields,
        batch_size=int(design["batch_size"]),
        relative_threshold=float(design["phase_power_relative_threshold"]),
    )
    phase_result = _binary_metric_control(
        control_id="phase_shuffle",
        source=phase_source_fields,
        perturbed=shuffled_fields,
        metrics=(
            registry["w1:axisym"],
            registry["w1:dav"],
            registry["radial_profile_mae"],
        ),
        boundaries=boundaries,
        invariant=phase_invariant,
    )
    rotated_fields = np.rot90(source_fields, k=1, axes=(-2, -1)).copy()
    psd_values = registry["radial_log_psd_mse"].transform(source_fields)
    rotated_psd_values = registry["radial_log_psd_mse"].transform(rotated_fields)
    psd_invariance_value = float(
        registry["radial_log_psd_mse"].estimator(
            rotated_psd_values,
            psd_values,
        )
    )
    psd_invariance_passed = bool(psd_invariance_value <= boundaries["radial_log_psd_mse"])
    psd_rotation_result = {
        "schema": "gauge_climate.invariance_control.v1",
        "id": "psd_quarter_turn",
        "expected_response": "radial Hann-windowed log-PSD remains within reference range",
        "metrics": {
            "radial_log_psd_mse": {
                "value": psd_invariance_value,
                "reference_boundary": boundaries["radial_log_psd_mse"],
                "reaches_reference_boundary": False,
                "expected_invariance": True,
                "passed": psd_invariance_passed,
            }
        },
        "passed": psd_invariance_passed,
    }
    copyist_result = _copyist_control(
        fields=analysis_train_fields,
        blocks=train_blocks,
        group_a_indices=group_a_indices,
        group_b_indices=group_b_indices,
        feature_space=feature_space,
        n_control=n_control,
        n_regions=int(feature_node["memorization"]["n_regions"]),
        n_resamples=int(design["copyist_resamples"]),
        min_real_blocks=int(feature_node["memorization"]["min_real_blocks"]),
        seed=seed,
        bt_min_k=adapter_config.bt_min_k,
        bt_max_k=adapter_config.bt_max_k,
    )
    copy_rng = np.random.default_rng(seed + 300)
    train_local = draw_real_indices(
        np.arange(group_a_indices.size, dtype=np.int64),
        train_blocks[group_a_indices],
        sorted(group_a),
        n=n_control,
        estimand=SamplingEstimand.BLOCK_BALANCED,
        rng=copy_rng,
    )
    real_local = draw_real_indices(
        np.arange(group_b_indices.size, dtype=np.int64),
        train_blocks[group_b_indices],
        sorted(group_b),
        n=n_control,
        estimand=SamplingEstimand.BLOCK_BALANCED,
        rng=copy_rng,
    )
    copy_train_fields = np.clip(
        _materialize(analysis_train_fields, group_a_indices[train_local]),
        adapter_config.bt_min_k,
        adapter_config.bt_max_k,
    )
    copy_real_indices = group_b_indices[real_local]
    copy_real_fields = np.clip(
        _materialize(analysis_train_fields, copy_real_indices),
        adapter_config.bt_min_k,
        adapter_config.bt_max_k,
    )
    copy_fraction_result = _copy_fraction_control(
        train_fields=copy_train_fields,
        real_fields=copy_real_fields,
        real_blocks=train_blocks[copy_real_indices],
        feature_space=feature_space,
        fractions=controls["copy_fraction"]["fractions"],
        n_regions=int(feature_node["memorization"]["n_regions"]),
        n_resamples=int(design["copy_fraction_resamples"]),
        min_real_blocks=int(feature_node["memorization"]["min_real_blocks"]),
        seed=seed + 400,
    )
    lanes = {
        **sweep_results,
        "phase_shuffle": phase_result,
        "psd_quarter_turn": psd_rotation_result,
        "copyist": copyist_result,
        "copy_fraction": copy_fraction_result,
    }
    registry_contract = ControlRegistry(
        (
            ControlSpec(
                "warm_shift", ControlRole.SENSITIVITY, ("w1:min_core_bt", "w1:p01_bt"), "monotone"
            ),
            ControlSpec(
                "threshold_warming", ControlRole.SENSITIVITY, ("w1:cold_frac_220",), "monotone"
            ),
            ControlSpec(
                "threshold_cooling", ControlRole.SENSITIVITY, ("w1:cold_frac_220",), "monotone"
            ),
            ControlSpec("eye_contrast", ControlRole.SENSITIVITY, ("w1:eye_contrast",), "monotone"),
            ControlSpec(
                "blur", ControlRole.SENSITIVITY, ("radial_log_psd_mse", "w1:r_maxgrad"), "monotone"
            ),
            ControlSpec(
                "center_jitter",
                ControlRole.SENSITIVITY,
                ("radial_profile_mae", "w1:r_maxgrad"),
                "monotone",
            ),
            ControlSpec(
                "radial_warp",
                ControlRole.SENSITIVITY,
                ("radial_profile_mae", "w1:r_maxgrad"),
                "monotone",
            ),
            ControlSpec(
                "phase_shuffle",
                ControlRole.SENSITIVITY,
                ("w1:axisym", "w1:dav", "radial_profile_mae"),
                "exceeds reference range",
            ),
            ControlSpec(
                "psd_quarter_turn",
                ControlRole.INVARIANCE,
                ("radial_log_psd_mse",),
                "remains within reference range",
            ),
            ControlSpec(
                "checkerboard",
                ControlRole.SENSITIVITY,
                ("radial_log_psd_mse", "feature_mmd2"),
                "monotone",
            ),
            ControlSpec(
                "feature_eof_excursion",
                ControlRole.SENSITIVITY,
                ("feature_mmd2", "feature_density"),
                "monotone",
            ),
            ControlSpec(
                "feature_mode_removal",
                ControlRole.SENSITIVITY,
                ("feature_coverage_deficit",),
                "monotone",
            ),
            ControlSpec(
                "feature_contraction",
                ControlRole.SENSITIVITY,
                ("feature_diversity_departure",),
                "monotone",
            ),
            ControlSpec(
                "feature_expansion",
                ControlRole.SENSITIVITY,
                ("feature_diversity_departure",),
                "monotone",
            ),
            ControlSpec(
                "boundary_spikes",
                ControlRole.SENSITIVITY,
                ("preprocessing_boundary_saturation",),
                "monotone",
            ),
            ControlSpec(
                "copy_fraction",
                ControlRole.COPYING,
                ("localized_memorization",),
                "increasing copied fraction",
            ),
        )
    )
    failed_lanes = sorted(key for key, lane in lanes.items() if not lane["passed"])
    suite_complete = not failed_lanes
    payload = {
        "schema": "gauge_climate.tc_bt_metric_controls.v2",
        "status": ("COMPLETE" if suite_complete else "COMPLETE_WITH_CONTROL_FAILURES"),
        "summary": {
            "lane_count": len(lanes),
            "failed_lanes": failed_lanes,
            "control_registry": registry_contract.manifest(),
        },
        "_metadata": {
            "scientific_use": "training-only versioned sensitivity and invariance controls",
            "test_fields_used": False,
            "wind_source": real_train.metadata.get("wind_source"),
            "wind_averaging_period": real_train.metadata.get("wind_averaging_period"),
            "real_train_manifest": _display_path(args.real_train_manifest),
            "real_train_manifest_sha256": _sha256(args.real_train_manifest.resolve()),
            "analysis_protocol": _display_path(args.analysis_protocol),
            "analysis_protocol_sha256": _sha256(args.analysis_protocol.resolve()),
            "sensitivity_contract_sha256": _mapping_hash(sensitivity),
            "track1_config_sha256": _sha256(args.track1_config.resolve()),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "gauge_source_sha256": _source_tree_sha256(ROOT / "src/gauge_climate"),
            "design": dict(design),
            "selected_blocks": selected_blocks.tolist(),
            "source_index_sha256": _array_hash(source_indices),
            "registered_feature_space": feature_space.manifest,
            "reference_envelopes": reference_summary,
        },
        **lanes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _plot_controls(payload, args.figure)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(args.output),
                "figure": str(args.figure),
                "lanes": {
                    key: {
                        "response_contract": value["passed"],
                        "metric_results": len(value.get("metrics", {})),
                    }
                    for key, value in lanes.items()
                },
            },
            indent=2,
        )
    )
    return 0 if suite_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
