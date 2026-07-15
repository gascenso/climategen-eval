#!/usr/bin/env python3
"""Execute the preregistered TC Track-1 sensitivity and copyist controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from gauge_climate.adapters import tc_bt
from gauge_climate.adapters.tc_data import load_real_archive
from gauge_climate.features import TrainFittedFeatureSpace
from gauge_climate.lanes import (
    additive_shift_lane,
    center_jitter_lane,
    copyist,
    gaussian_blur_lane,
    phase_shuffle_fields,
    run_perturbation_sweep,
)
from gauge_climate.memorization import localized_copying_test
from gauge_climate.metrics import functional_w1
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


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


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
        bootstrap_resamples=20,
        sensitivity_resamples=20,
        alpha=1.0 - quantile,
        confidence=quantile,
        estimand=SamplingEstimand.BLOCK_BALANCED,
        seed=seed,
        min_blocks_per_half=min_blocks_per_half,
        warn_blocks_per_half=min_blocks_per_half,
        shift_grid_size=8,
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
        values = _batch_transform(
            metric, fields, pool_indices, batch_size=batch_size
        )
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
    relative_tolerance: float,
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
        "relative_tolerance": relative_tolerance,
        "passed": bool(maximum <= relative_tolerance),
    }


def _copyist_control(
    *,
    fields: Any,
    blocks: np.ndarray,
    group_a_indices: np.ndarray,
    group_b_indices: np.ndarray,
    functionals: Sequence[Any],
    n_control: int,
    n_eof: int,
    max_fit_samples: int,
    n_regions: int,
    n_resamples: int,
    min_real_blocks: int,
    seed: int,
    batch_size: int,
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
    train_fields = _materialize(fields, train_indices)
    real_fields = _materialize(fields, real_indices)
    generated_fields = copyist(train_fields, n=n_control, seed=seed + 1)
    feature_space = TrainFittedFeatureSpace.fit(
        train_fields,
        functionals,
        n_eof=n_eof,
        max_fit_samples=max_fit_samples,
        seed=seed,
        batch_size=batch_size,
        fit_source="train",
    )
    result = localized_copying_test(
        train_features=feature_space.transform(train_fields),
        real_features=feature_space.transform(real_fields),
        generated_features=feature_space.transform(generated_fields),
        real_blocks=np.asarray(blocks)[real_indices],
        n_regions=n_regions,
        n_resamples=n_resamples,
        alpha=0.05,
        min_real_blocks=min_real_blocks,
        seed=seed,
        feature_provenance=feature_space.manifest,
        estimand=SamplingEstimand.BLOCK_BALANCED,
    )
    passed = result.get("status") == "COPYING_DETECTED"
    return {
        "schema": "gauge_climate.sensitivity_lane.v1",
        "id": "copyist",
        "expected_response": "localized_copying_detected",
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


def _plot_controls(payload: Mapping[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
    sweep_ids = ("warm_shift", "blur", "center_jitter")
    for axis, lane_id in zip(axes.flat, sweep_ids, strict=False):
        lane = payload[lane_id]
        severities = np.asarray(lane["severities"], dtype=float)
        for metric_id, result in lane["metrics"].items():
            boundary = result.get("reference_boundary")
            denominator = boundary if boundary is not None and boundary > 0 else 1.0
            axis.plot(
                severities,
                np.asarray(result["values"], dtype=float) / denominator,
                marker="o",
                label=metric_id,
            )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"{lane_id}: {'PASS' if lane['passed'] else 'FAIL'}")
        axis.set_xlabel("severity")
        axis.set_ylabel("discrepancy / reference boundary")
        axis.legend(fontsize=7)
    phase_axis = axes.flat[3]
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
    phase_axis.set_title(f"phase_shuffle: {'PASS' if phase['passed'] else 'FAIL'}")
    copy_axis = axes.flat[4]
    copy_axis.axis("off")
    copy_lane = payload["copyist"]
    copy_axis.text(
        0.05,
        0.85,
        "Copyist positive control",
        fontsize=12,
        weight="bold",
        transform=copy_axis.transAxes,
    )
    copy_axis.text(
        0.05,
        0.65,
        f"Expected: COPYING_DETECTED\n"
        f"Observed: {copy_lane['result'].get('status')}\n"
        f"Result: {'PASS' if copy_lane['passed'] else 'FAIL'}",
        fontsize=10,
        transform=copy_axis.transAxes,
    )
    summary_axis = axes.flat[5]
    summary_axis.axis("off")
    metadata = payload["_metadata"]
    summary_axis.text(
        0.05,
        0.85,
        "Frozen control suite",
        fontsize=12,
        weight="bold",
        transform=summary_axis.transAxes,
    )
    summary_axis.text(
        0.05,
        0.60,
        f"Suite: {payload['status']}\n"
        f"N: {metadata['design']['n_control']}\n"
        f"Blocks: {metadata['design']['control_block_count']}\n"
        f"Reference resamples: {metadata['design']['reference_resamples']}",
        fontsize=10,
        transform=summary_axis.transAxes,
    )
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
        "--preregistration",
        type=Path,
        default=ROOT / "case_studies/tropical_cyclone_bt/preregistration.yaml",
    )
    parser.add_argument(
        "--track1-config",
        type=Path,
        default=ROOT / "case_studies/tropical_cyclone_bt/track1_config.template.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "case_studies/tropical_cyclone_bt/control_results/sensitivity_lanes.json"
        ),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=(
            ROOT
            / "case_studies/tropical_cyclone_bt/control_results/sensitivity_lanes.pdf"
        ),
    )
    args = parser.parse_args()

    preregistration = _load_yaml(args.preregistration.resolve())
    sensitivity = preregistration.get("sensitivity_lanes", {})
    if sensitivity.get("status") != "FROZEN":
        raise ValueError("TC sensitivity choices must be frozen before execution.")
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
        "copyist_resamples",
        "batch_size",
        "phase_power_relative_tolerance",
    }
    missing_design = sorted(required_design - set(design))
    if missing_design:
        raise ValueError(f"Preregistration lacks frozen control design: {missing_design}.")
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
    functionals = tc_bt.functionals(adapter_config)
    functional_metrics = {metric.id: metric for metric in map(functional_w1, functionals)}
    curve_metrics = {metric.id: metric for metric in tc_bt.curve_metrics(adapter_config)}
    registry = {**functional_metrics, **curve_metrics}
    expected_ids = (
        "w1:min_core_bt",
        "w1:p01_bt",
        "w1:r_maxgrad",
        "w1:axisym",
        "w1:dav",
        "radial_profile_mae",
        "radial_log_psd_mse",
    )
    expected_metrics = tuple(registry[key] for key in expected_ids)
    boundaries, reference_summary = _reference_boundaries(
        fields=real_train.fields,
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
    source_local = draw_real_indices(
        np.arange(pool_indices.size, dtype=np.int64),
        train_blocks[pool_indices],
        selected_blocks.tolist(),
        n=n_control,
        estimand=SamplingEstimand.BLOCK_BALANCED,
        rng=np.random.default_rng(seed + 1),
    )
    source_indices = pool_indices[source_local]
    source_fields = _materialize(real_train.fields, source_indices)

    warm = additive_shift_lane(
        severities=controls["warm_shift"]["kelvin"],
        expected_metrics=("w1:min_core_bt", "w1:p01_bt"),
    )
    blur = gaussian_blur_lane(
        severities=controls["blur"]["sigma_pixels"],
        expected_metrics=("radial_log_psd_mse", "w1:r_maxgrad"),
    )
    jitter = center_jitter_lane(
        severities=controls["center_jitter"]["pixels"],
        expected_metrics=("radial_profile_mae", "w1:r_maxgrad"),
    )
    sweep_registry = tuple(registry.values())
    warm_result = run_perturbation_sweep(
        warm,
        source_fields=source_fields,
        reference_fields=source_fields,
        metrics=sweep_registry,
        reference_boundaries=boundaries,
        seed=seed,
    )
    warm_result["id"] = "warm_shift"
    blur_result = run_perturbation_sweep(
        blur,
        source_fields=source_fields,
        reference_fields=source_fields,
        metrics=sweep_registry,
        reference_boundaries=boundaries,
        seed=seed,
    )
    blur_result["id"] = "blur"
    jitter_result = run_perturbation_sweep(
        jitter,
        source_fields=source_fields,
        reference_fields=source_fields,
        metrics=sweep_registry,
        reference_boundaries=boundaries,
        seed=seed,
    )
    shuffled_fields = phase_shuffle_fields(source_fields, seed=seed)
    phase_invariant = _phase_power_invariant(
        source_fields,
        shuffled_fields,
        batch_size=int(design["batch_size"]),
        relative_tolerance=float(design["phase_power_relative_tolerance"]),
    )
    phase_result = _binary_metric_control(
        control_id="phase_shuffle",
        source=source_fields,
        perturbed=shuffled_fields,
        metrics=(
            registry["w1:axisym"],
            registry["w1:dav"],
            registry["radial_profile_mae"],
        ),
        boundaries=boundaries,
        invariant=phase_invariant,
    )
    copyist_result = _copyist_control(
        fields=real_train.fields,
        blocks=train_blocks,
        group_a_indices=group_a_indices,
        group_b_indices=group_b_indices,
        functionals=functionals,
        n_control=n_control,
        n_eof=int(track1["feature_metrics"]["n_eof"]),
        max_fit_samples=int(design["feature_fit_samples"]),
        n_regions=int(track1["feature_metrics"]["memorization"]["n_regions"]),
        n_resamples=int(design["copyist_resamples"]),
        min_real_blocks=int(
            track1["feature_metrics"]["memorization"]["min_real_blocks"]
        ),
        seed=seed,
        batch_size=int(design["batch_size"]),
    )
    lanes = {
        "warm_shift": warm_result,
        "blur": blur_result,
        "center_jitter": jitter_result,
        "phase_shuffle": phase_result,
        "copyist": copyist_result,
    }
    suite_passed = all(lane["passed"] for lane in lanes.values())
    payload = {
        "schema": "gauge_climate.tc_bt_sensitivity_controls.v1",
        "status": "PASSED" if suite_passed else "FAILED",
        "_metadata": {
            "scientific_use": "training-only preregistered positive controls",
            "test_fields_used": False,
            "wind_source": real_train.metadata.get("wind_source"),
            "wind_averaging_period": real_train.metadata.get(
                "wind_averaging_period"
            ),
            "real_train_manifest": _display_path(args.real_train_manifest),
            "real_train_manifest_sha256": _sha256(args.real_train_manifest.resolve()),
            "preregistration_sha256": _sha256(args.preregistration.resolve()),
            "track1_config_sha256": _sha256(args.track1_config.resolve()),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "design": dict(design),
            "selected_blocks": selected_blocks.tolist(),
            "source_index_sha256": _array_hash(source_indices),
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
                "lanes": {key: value["passed"] for key, value in lanes.items()},
            },
            indent=2,
        )
    )
    return 0 if suite_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
