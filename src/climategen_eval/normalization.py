from __future__ import annotations

from typing import Any

import numpy as np

from .corruptions import CONTROL_NAMES, build_calibrated_control_bt_k, build_fixed_control_bt_k
from .tc_bt import (
    TCMetricConfig,
    aggregate_class_summaries,
    build_class_metric_cache,
    build_class_metric_cache_from_real_cache,
    compute_class_scalar_summary,
    compute_real_vs_real_class_summary,
    make_metric_context,
)


RANGE_DIRECTIONS = {
    "pixel_hist_js": "lower",
    "pixel_hist_w1": "lower",
    "dav_abs_gap_deg2": "lower",
    "radial_profile_mae_k": "lower",
    "cold_cloud_fraction_abs_gap": "lower",
    "psd_l2": "lower",
    "eye_contrast_proxy_abs_gap": "lower",
    "feature_mmd2": "lower",
    "diversity_feature_space_abs_gap": "lower",
    "gen_exceedance_rate_total": "lower",
}


def summarize_distribution(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "q50": float(np.quantile(arr, 0.50)),
        "q95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def metric_ranges(good_values: dict[str, float], bad_values: dict[str, float]) -> dict[str, dict[str, Any]]:
    out = {}
    for key in sorted(set(good_values) & set(bad_values)):
        if key not in RANGE_DIRECTIONS:
            continue
        good = float(good_values[key])
        bad = float(bad_values[key])
        direction = RANGE_DIRECTIONS[key]
        usable = (
            bool(np.isfinite(good) and np.isfinite(bad))
            and abs(good - bad) > 1.0e-12
            and ((direction == "lower" and bad > good) or (direction == "higher" and good > bad))
        )
        out[key] = {
            "direction": direction,
            "best": good,
            "worst": bad,
            "usable_for_0_1_score": usable,
        }
    return out


def sample_real_images_for_negative_control(
    real_k: np.ndarray,
    *,
    draw_n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(real_k, dtype=np.float32)
    source_n = int(arr.shape[0])
    draw_n = int(draw_n)
    if draw_n <= 0:
        raise ValueError(f"draw_n must be > 0, got {draw_n}")
    replace = source_n < draw_n
    idx = rng.choice(source_n, size=draw_n, replace=replace)
    return np.asarray(arr[idx], dtype=np.float32), {
        "draw_n": draw_n,
        "source_n": source_n,
        "draw_mode": "with_replacement" if replace else "without_replacement",
    }


def append_macro_samples(samples_by_key: dict[str, list[float]], per_class: dict[int, dict[str, float]]) -> None:
    for key in sorted({key for values in per_class.values() for key in values}):
        values = [float(values[key]) for values in per_class.values() if key in values]
        if values:
            samples_by_key.setdefault(key, []).append(float(np.mean(values)))


def bootstrap_real_vs_real_samples(
    class_caches: dict[int, Any],
    *,
    reps: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    samples_by_key: dict[str, list[float]] = {}
    for _ in range(int(reps)):
        per_class = {}
        for class_id, cache in sorted(class_caches.items()):
            real_n = int(cache.real_hist_counts.shape[0])
            other_n = int(cache.gen_raw_hist_counts.shape[0])
            left_idx = rng.integers(real_n, size=real_n)
            right_idx = rng.integers(real_n, size=other_n)
            per_class[int(class_id)] = compute_real_vs_real_class_summary(
                cache,
                left_idx=left_idx,
                right_idx=right_idx,
            )
        append_macro_samples(samples_by_key, per_class)
    return {key: np.asarray(values, dtype=np.float64) for key, values in samples_by_key.items()}


def bootstrap_control_samples(
    control_caches: dict[int, Any],
    *,
    reps: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    samples_by_key: dict[str, list[float]] = {}
    for _ in range(int(reps)):
        per_class = {}
        for class_id, cache in sorted(control_caches.items()):
            real_n = int(cache.real_hist_counts.shape[0])
            gen_n = int(cache.gen_raw_hist_counts.shape[0])
            per_class[int(class_id)] = compute_class_scalar_summary(
                cache,
                real_idx=rng.integers(real_n, size=real_n),
                gen_idx=rng.integers(gen_n, size=gen_n),
            )
        append_macro_samples(samples_by_key, per_class)
    return {key: np.asarray(values, dtype=np.float64) for key, values in samples_by_key.items()}


def real_vs_real_reference(
    class_caches: dict[int, Any],
    *,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    if int(reps) <= 0:
        return {"enabled": False, "reason": "reps <= 0"}
    samples = bootstrap_real_vs_real_samples(class_caches, reps=int(reps), seed=int(seed))
    summary = {key: summarize_distribution(values) for key, values in sorted(samples.items())}
    return {
        "enabled": True,
        "variant": "matched_count_bootstrap_real_vs_real",
        "draw_mode": "with_replacement",
        "reps": int(reps),
        "seed": int(seed),
        "values": {key: float(node["q50"]) for key, node in summary.items()},
        "summary": summary,
    }


def default_calibration_grids(cfg: TCMetricConfig) -> dict[str, list[float]]:
    size = float(max(int(cfg.image_size), 1))
    span = float(max(float(cfg.bt_max_k) - float(cfg.bt_min_k), 1.0))
    jitter = sorted(
        set(
            int(round(size * f))
            for f in (0.0, 0.005, 0.010, 0.020, 0.035, 0.055, 0.080, 0.120, 0.180)
        )
    )
    return {
        "gaussian_blur": [
            size * f for f in (0.0, 0.005, 0.010, 0.015, 0.020, 0.030, 0.045, 0.065, 0.090)
        ],
        "bt_warm_shift": [
            span * f for f in (0.0, 0.005, 0.010, 0.020, 0.035, 0.055, 0.080, 0.120, 0.180)
        ],
        "center_jitter": [float(v) for v in jitter],
        "pixel_shuffle": [0.0, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.400, 0.700, 1.000],
        "mode_collapse": [0.0, 0.050, 0.100, 0.200, 0.400, 0.600, 0.800, 1.000],
    }


def detection_power(values: np.ndarray, *, threshold: float, direction: str) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if str(direction) == "higher":
        return float(np.mean(arr < float(threshold)))
    return float(np.mean(arr > float(threshold)))


def directionally_valid_bad_anchor(direction: str, good_value: float, bad_value: float) -> bool:
    if str(direction) == "higher":
        return float(bad_value) < float(good_value)
    return float(bad_value) > float(good_value)


def calibrate_bad_anchor_values(
    *,
    class_caches: dict[int, Any],
    pseudo_sources_by_class: dict[int, np.ndarray],
    controls: list[str],
    grids: dict[str, list[float]],
    cfg: TCMetricConfig,
    reps: int,
    seed: int,
    target_power: float,
    null_quantile: float,
    aggregation: str,
) -> dict[str, Any]:
    null_samples = bootstrap_real_vs_real_samples(class_caches, reps=int(reps), seed=int(seed))
    thresholds = {}
    good_values = {}
    for key, values in null_samples.items():
        direction = RANGE_DIRECTIONS.get(key, "lower")
        q = 1.0 - float(null_quantile) if direction == "higher" else float(null_quantile)
        thresholds[key] = float(np.quantile(values, q))
        good_values[key] = float(np.quantile(values, 0.50))

    binner, dav_computer, cold_mask = make_metric_context(cfg)
    curves_by_control: dict[str, dict[str, list[dict[str, Any]]]] = {name: {} for name in controls}
    transforms_by_control: dict[str, list[dict[str, Any]]] = {name: [] for name in controls}
    for control_idx, control_name in enumerate(controls):
        for value_idx, severity in enumerate(grids.get(control_name, [])):
            control_caches = {}
            transform_for_value = None
            for class_id, source_k in sorted(pseudo_sources_by_class.items()):
                control_seed = int(
                    seed + 1009 * (control_idx + 1) + 9176 * (value_idx + 1) + 104729 * (class_id + 1)
                )
                control_k, transform = build_calibrated_control_bt_k(
                    control_name,
                    source_k,
                    severity=float(severity),
                    seed=control_seed,
                )
                transform_for_value = transform_for_value or dict(transform)
                control_caches[class_id] = build_class_metric_cache_from_real_cache(
                    real_cache=class_caches[class_id],
                    gen_k_raw=control_k,
                    cfg=cfg,
                    binner=binner,
                    dav_computer=dav_computer,
                    cold_mask=cold_mask,
                )
            transforms_by_control[control_name].append(
                {"severity": float(severity), "transformation": transform_for_value or {}}
            )
            samples_by_metric = bootstrap_control_samples(
                control_caches,
                reps=int(reps),
                seed=int(seed + 1723 * (control_idx + 1) + 7919 * (value_idx + 1)),
            )
            for key, values in sorted(samples_by_metric.items()):
                if key not in thresholds:
                    continue
                direction = RANGE_DIRECTIONS.get(key, "lower")
                summary = summarize_distribution(values)
                point = {
                    "severity": float(severity),
                    "threshold": float(thresholds[key]),
                    "power": detection_power(values, threshold=thresholds[key], direction=direction),
                    "value": float(summary["q50"]),
                    "summary": summary,
                }
                curves_by_control.setdefault(control_name, {}).setdefault(key, []).append(point)

    selected_by_metric: dict[str, dict[str, Any]] = {}
    excluded_by_metric: dict[str, dict[str, str]] = {}
    selected_values_by_metric: dict[str, list[float]] = {}
    for key in sorted(thresholds):
        direction = RANGE_DIRECTIONS.get(key, "lower")
        good = float(good_values[key])
        excluded_by_metric[key] = {}
        for control_name in controls:
            selected = None
            points = sorted(
                curves_by_control.get(control_name, {}).get(key, []),
                key=lambda node: float(node["severity"]),
            )
            for point in points:
                bad = float(point["value"])
                if (
                    float(point["severity"]) > 0.0
                    and float(point["power"]) >= float(target_power)
                    and directionally_valid_bad_anchor(direction, good, bad)
                ):
                    selected = point
                    break
            if selected is None:
                max_power = max((float(point["power"]) for point in points), default=0.0)
                excluded_by_metric[key][control_name] = (
                    f"no_positive_severity_reached_target_power; max_power={max_power:.3f}"
                )
                continue
            selected_by_metric.setdefault(key, {})[control_name] = selected
            selected_values_by_metric.setdefault(key, []).append(float(selected["value"]))

    bad_values = {}
    for key, values in sorted(selected_values_by_metric.items()):
        arr = np.asarray(values, dtype=np.float64)
        bad_values[key] = float(np.mean(arr) if aggregation == "mean" else np.median(arr))

    return {
        "good_values": good_values,
        "bad_values": bad_values,
        "target_power": float(target_power),
        "real_vs_real_null": {
            "reps": int(reps),
            "seed": int(seed),
            "threshold_quantile_lower_is_better": float(null_quantile),
            "threshold_quantile_higher_is_better": float(1.0 - null_quantile),
            "thresholds": thresholds,
            "summary": {key: summarize_distribution(values) for key, values in sorted(null_samples.items())},
        },
        "bad_anchor_aggregation": aggregation,
        "calibrated_by_metric": {
            key: {
                "direction": RANGE_DIRECTIONS.get(key, "lower"),
                "good_value": float(good_values[key]),
                "null_threshold": float(thresholds[key]),
                "selected_controls": selected_by_metric.get(key, {}),
                "excluded_controls": excluded_by_metric.get(key, {}),
                "bad_anchor_value": bad_values.get(key),
            }
            for key in sorted(thresholds)
        },
        "controls": {
            name: {
                "source": "calibrated_corrupted_real_reference_images",
                "grid": [float(v) for v in grids.get(name, [])],
                "grid_transforms": transforms_by_control.get(name, []),
                "response_curve_by_metric": curves_by_control.get(name, {}),
            }
            for name in controls
        },
    }


def compute_anchor_report(
    *,
    real_by_class: dict[int, np.ndarray],
    cfg: TCMetricConfig,
    generated_n_per_class: int,
    controls: list[str] | None = None,
    anchor_mode: str = "fixed",
    seed: int = 123,
    real_reference_reps: int = 200,
    calibration_reps: int | None = None,
    calibration_target_power: float = 0.80,
    calibration_null_quantile: float = 0.95,
    calibration_bad_anchor_aggregation: str = "median",
) -> dict[str, Any]:
    controls = list(CONTROL_NAMES if controls is None else controls)
    binner, dav_computer, cold_mask = make_metric_context(cfg)
    class_caches = {}
    pseudo_sources_by_class = {}
    source_sampling_by_class = {}
    for class_id, real_k in sorted(real_by_class.items()):
        rng = np.random.default_rng(int(seed + 104729 * (class_id + 1)))
        pseudo_k, source_meta = sample_real_images_for_negative_control(
            real_k,
            draw_n=int(generated_n_per_class),
            rng=rng,
        )
        pseudo_sources_by_class[class_id] = pseudo_k
        source_sampling_by_class[class_id] = source_meta
        class_caches[class_id] = build_class_metric_cache(
            real_k=real_k,
            gen_k_raw=pseudo_k,
            cfg=cfg,
            binner=binner,
            dav_computer=dav_computer,
            cold_mask=cold_mask,
        )

    good_reference = real_vs_real_reference(
        class_caches,
        reps=int(real_reference_reps),
        seed=int(seed),
    )
    good_values = dict(good_reference.get("values", {}))
    bad_values: dict[str, float] = {}
    controls_out: dict[str, Any] = {}
    calibration = None

    if anchor_mode == "calibrated":
        calibration = calibrate_bad_anchor_values(
            class_caches=class_caches,
            pseudo_sources_by_class=pseudo_sources_by_class,
            controls=controls,
            grids=default_calibration_grids(cfg),
            cfg=cfg,
            reps=int(real_reference_reps if calibration_reps is None else calibration_reps),
            seed=int(seed),
            target_power=float(calibration_target_power),
            null_quantile=float(calibration_null_quantile),
            aggregation=str(calibration_bad_anchor_aggregation),
        )
        good_values.update(calibration["good_values"])
        bad_values.update(calibration["bad_values"])
        controls_out = dict(calibration["controls"])
    else:
        control_aggregates = {}
        for control_idx, control_name in enumerate(controls):
            control_caches = {}
            transform = None
            for class_id, source_k in sorted(pseudo_sources_by_class.items()):
                control_k, transform = build_fixed_control_bt_k(
                    control_name,
                    source_k,
                    seed=int(seed + 1009 * (control_idx + 1) + 104729 * (class_id + 1)),
                )
                control_caches[class_id] = build_class_metric_cache_from_real_cache(
                    real_cache=class_caches[class_id],
                    gen_k_raw=control_k,
                    cfg=cfg,
                    binner=binner,
                    dav_computer=dav_computer,
                    cold_mask=cold_mask,
                )
            aggregate = aggregate_class_summaries(
                {class_id: compute_class_scalar_summary(cache) for class_id, cache in control_caches.items()}
            )
            controls_out[control_name] = {
                "source": "corrupted_real_reference_images",
                "transformation": transform or {"name": control_name},
                "aggregate_primary_raw": aggregate,
            }
            control_aggregates[control_name] = aggregate
        for key in sorted({key for agg in control_aggregates.values() for key in agg}):
            values = [float(agg[key]) for agg in control_aggregates.values() if key in agg]
            if values:
                bad_values[key] = float(np.mean(values))

    report = {
        "schema": (
            "climategen_eval.metric_normalization_ranges.v2"
            if anchor_mode == "calibrated"
            else "climategen_eval.metric_normalization_ranges.v1"
        ),
        "metric_ranges": metric_ranges(good_values, bad_values),
        "metric_normalization": {
            "enabled": True,
            "schema": "climategen_eval.metric_normalization.v1",
            "anchor_mode": str(anchor_mode),
            "source": (
                "calibrated_corrupted_real_reference_images"
                if anchor_mode == "calibrated"
                else "precomputed_corrupted_real_reference_images"
            ),
            "controls_requested": controls,
            "match_generated_counts": True,
            "seed": int(seed),
            "real_reference_reps": int(real_reference_reps),
            "bad_anchor_aggregation": (
                str(calibration_bad_anchor_aggregation) if anchor_mode == "calibrated" else "mean"
            ),
            "good_reference": {
                "values": good_values,
                "physical_real_vs_real": good_reference,
            },
            "negative_controls": {
                "controls": controls_out,
                "source_sampling_by_class": {
                    str(c): dict(meta) for c, meta in sorted(source_sampling_by_class.items())
                },
            },
            "bad_anchor": {
                "aggregation": (
                    str(calibration_bad_anchor_aggregation) if anchor_mode == "calibrated" else "mean"
                ),
                "values": bad_values,
            },
        },
    }
    if calibration is not None:
        report["metric_normalization"]["calibration"] = {
            key: value
            for key, value in calibration.items()
            if key not in {"controls", "good_values", "bad_values"}
        }
    return report

