from __future__ import annotations

from typing import Any

import numpy as np

from .corruptions import CONTROL_NAMES, build_validation_control_bt_k
from .tc_bt import (
    TCMetricConfig,
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


DEFAULT_DISPLAY_NULL_WIDTHS = 4.0


def summarize_distribution(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot summarize an empty distribution.")
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "q50": float(np.quantile(arr, 0.50)),
        "q95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def null_standardization_ranges(
    real_vs_real_summary: dict[str, dict[str, float]],
    *,
    null_quantile: float = 0.95,
    min_scale: float = 1.0e-12,
) -> dict[str, dict[str, Any]]:
    """Build metric standardization ranges from real-vs-real bootstrap summaries.

    The scale is a one-sided null width around the finite-sample ideal:
    q95 - q50 for lower-is-better metrics, and q50 - q05 for higher-is-better
    metrics. Corruptions are intentionally not used here; they are validation
    diagnostics, not the display ruler.
    """
    if abs(float(null_quantile) - 0.95) > 1.0e-12:
        raise ValueError("null_standardization_ranges currently expects null_quantile=0.95.")

    out: dict[str, dict[str, Any]] = {}
    for key, node in sorted(real_vs_real_summary.items()):
        if key not in RANGE_DIRECTIONS:
            continue
        direction = RANGE_DIRECTIONS[key]
        good = float(node.get("q50", np.nan))
        if direction == "higher":
            boundary = float(node.get("q05", np.nan))
            scale = good - boundary
            scale_definition = "q50_minus_q05"
        else:
            boundary = float(node.get("q95", np.nan))
            scale = boundary - good
            scale_definition = "q95_minus_q50"
        usable = bool(
            np.isfinite(good)
            and np.isfinite(boundary)
            and np.isfinite(scale)
            and scale > float(min_scale)
        )
        out[key] = {
            "direction": direction,
            "good": good,
            "null_boundary": boundary,
            "scale": scale,
            "scale_definition": scale_definition,
            "null_quantile": float(null_quantile),
            "usable_for_standardized_gap": usable,
        }
    return out


def standardized_metric_value(
    *,
    value: float,
    standardization: dict[str, Any],
    display_null_widths: float = DEFAULT_DISPLAY_NULL_WIDTHS,
) -> dict[str, Any]:
    direction = str(standardization.get("direction", "lower"))
    if direction not in {"lower", "higher"}:
        raise ValueError(f"Unsupported metric direction: {direction!r}")
    good = float(standardization["good"])
    scale = float(standardization["scale"])
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError(f"Standardization scale must be positive and finite, got {scale}.")
    raw_value = float(value)
    if not np.isfinite(raw_value):
        raise ValueError(f"Metric value must be finite, got {value}.")
    if direction == "higher":
        gap = max(0.0, good - raw_value) / scale
    else:
        gap = max(0.0, raw_value - good) / scale
    k = float(display_null_widths)
    if k <= 0.0 or not np.isfinite(k):
        raise ValueError(f"display_null_widths must be > 0, got {display_null_widths}.")
    score = float(np.clip(1.0 - gap / k, 0.0, 1.0))
    return {
        "value": raw_value,
        "direction": direction,
        "good": good,
        "scale": scale,
        "standardized_gap": float(gap),
        "display_null_widths": k,
        "score": score,
    }


def standardize_metric_values(
    values: dict[str, float],
    ranges: dict[str, dict[str, Any]],
    *,
    display_null_widths: float = DEFAULT_DISPLAY_NULL_WIDTHS,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, value in sorted(values.items()):
        standardization = ranges.get(key)
        if not standardization or not standardization.get("usable_for_standardized_gap", False):
            continue
        out[key] = standardized_metric_value(
            value=float(value),
            standardization=standardization,
            display_null_widths=display_null_widths,
        )
    return out


def sample_real_images_for_reference_draw(
    real_k: np.ndarray,
    *,
    draw_n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(real_k, dtype=np.float32)
    source_n = int(arr.shape[0])
    draw_n = int(draw_n)
    if source_n <= 0:
        raise ValueError("Reference draw requires at least one real image.")
    if draw_n <= 0:
        raise ValueError(f"draw_n must be > 0, got {draw_n}")
    replace = source_n < draw_n
    idx = rng.choice(source_n, size=draw_n, replace=replace)
    return np.asarray(arr[idx], dtype=np.float32), {
        "draw_n": draw_n,
        "source_n": source_n,
        "draw_mode": "with_replacement" if replace else "without_replacement",
    }


def append_macro_samples(
    samples_by_key: dict[str, list[float]],
    per_class: dict[int, dict[str, float]],
) -> None:
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


def default_validation_grids(cfg: TCMetricConfig) -> dict[str, list[float]]:
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


def directionally_worse_than_reference(direction: str, reference: float, value: float) -> bool:
    if str(direction) == "higher":
        return float(value) < float(reference)
    return float(value) > float(reference)


def null_threshold_from_summary(
    summary: dict[str, float],
    *,
    direction: str,
    null_quantile: float,
) -> float:
    if abs(float(null_quantile) - 0.95) > 1.0e-12:
        raise ValueError("Corruption validation currently expects null_quantile=0.95.")
    return float(summary["q05"] if str(direction) == "higher" else summary["q95"])


def validate_control_names(controls: list[str] | None) -> list[str]:
    raw = list(CONTROL_NAMES if controls is None else controls)
    out: list[str] = []
    unknown: list[str] = []
    for control in raw:
        name = str(control)
        if name not in CONTROL_NAMES:
            unknown.append(name)
        elif name not in out:
            out.append(name)
    if unknown:
        valid = ", ".join(CONTROL_NAMES)
        raise ValueError(f"Unsupported validation controls {unknown}; valid controls are: {valid}.")
    if not out:
        raise ValueError("At least one validation control is required.")
    return out


def compute_corruption_validation(
    *,
    class_caches: dict[int, Any],
    pseudo_sources_by_class: dict[int, np.ndarray],
    controls: list[str],
    grids: dict[str, list[float]],
    cfg: TCMetricConfig,
    reps: int,
    seed: int,
    real_vs_real_summary: dict[str, dict[str, float]],
    target_power: float,
) -> dict[str, Any]:
    if int(reps) <= 0:
        raise ValueError(f"corruption validation reps must be > 0, got {reps}.")
    if not (0.0 <= float(target_power) <= 1.0):
        raise ValueError(f"target_power must be in [0, 1], got {target_power}.")
    if not real_vs_real_summary:
        raise ValueError("corruption validation requires a non-empty real-vs-real summary.")

    null_quantile = 0.95
    thresholds = {}
    good_values = {}
    for key, summary in sorted(real_vs_real_summary.items()):
        direction = RANGE_DIRECTIONS.get(key, "lower")
        thresholds[key] = null_threshold_from_summary(
            summary,
            direction=direction,
            null_quantile=float(null_quantile),
        )
        good_values[key] = float(summary["q50"])

    binner, dav_computer, cold_mask = make_metric_context(cfg)
    curves_by_control: dict[str, dict[str, list[dict[str, Any]]]] = {name: {} for name in controls}
    transforms_by_control: dict[str, list[dict[str, Any]]] = {name: [] for name in controls}
    for control_idx, control_name in enumerate(controls):
        for value_idx, severity in enumerate(grids.get(control_name, [])):
            control_caches = {}
            transforms_for_value = {}
            for class_id, source_k in sorted(pseudo_sources_by_class.items()):
                control_seed = int(
                    seed
                    + 1009 * (control_idx + 1)
                    + 9176 * (value_idx + 1)
                    + 104729 * (class_id + 1)
                )
                control_k, transform = build_validation_control_bt_k(
                    control_name,
                    source_k,
                    severity=float(severity),
                    seed=control_seed,
                )
                transforms_for_value[str(class_id)] = dict(transform)
                control_caches[class_id] = build_class_metric_cache_from_real_cache(
                    real_cache=class_caches[class_id],
                    gen_k_raw=control_k,
                    cfg=cfg,
                    binner=binner,
                    dav_computer=dav_computer,
                    cold_mask=cold_mask,
                )
            transforms_by_control[control_name].append(
                {
                    "severity": float(severity),
                    "transformations_by_class": transforms_for_value,
                }
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
                    "power": detection_power(
                        values,
                        threshold=thresholds[key],
                        direction=direction,
                    ),
                    "value": float(summary["q50"]),
                    "summary": summary,
                }
                curves_by_control.setdefault(control_name, {}).setdefault(key, []).append(point)

    responsiveness_by_metric: dict[str, dict[str, Any]] = {}
    for key in sorted(thresholds):
        direction = RANGE_DIRECTIONS.get(key, "lower")
        good = float(good_values[key])
        responsive_controls = {}
        max_power_by_control = {}
        for control_name in controls:
            points = sorted(
                curves_by_control.get(control_name, {}).get(key, []),
                key=lambda node: float(node["severity"]),
            )
            max_power_by_control[control_name] = max(
                (float(point["power"]) for point in points if float(point["severity"]) > 0.0),
                default=0.0,
            )
            for point in points:
                value = float(point["value"])
                if (
                    float(point["severity"]) > 0.0
                    and float(point["power"]) >= float(target_power)
                    and directionally_worse_than_reference(direction, good, value)
                ):
                    responsive_controls[control_name] = point
                    break
        responsiveness_by_metric[key] = {
            "direction": direction,
            "good_value": good,
            "null_threshold": float(thresholds[key]),
            "target_power": float(target_power),
            "responsive": bool(responsive_controls),
            "responsive_controls": responsive_controls,
            "max_power_by_control": max_power_by_control,
        }

    return {
        "enabled": True,
        "schema": "climategen_eval.corruption_validation.v1",
        "source": "corrupted_real_reference_images",
        "target_power": float(target_power),
        "reps": int(reps),
        "seed": int(seed),
        "null_quantile": float(null_quantile),
        "thresholds": thresholds,
        "responsiveness_by_metric": responsiveness_by_metric,
        "controls": {
            name: {
                "grid": [float(v) for v in grids.get(name, [])],
                "grid_transforms": transforms_by_control.get(name, []),
                "response_curve_by_metric": curves_by_control.get(name, {}),
            }
            for name in controls
        },
    }


def compute_standardization_report(
    *,
    real_by_class: dict[int, np.ndarray],
    cfg: TCMetricConfig,
    generated_n_per_class: int,
    controls: list[str] | None = None,
    seed: int = 123,
    real_reference_reps: int = 200,
    validation_reps: int | None = None,
    validation_target_power: float = 0.80,
    display_null_widths: float = DEFAULT_DISPLAY_NULL_WIDTHS,
) -> dict[str, Any]:
    if not real_by_class:
        raise ValueError("real_by_class must contain at least one class.")
    if int(generated_n_per_class) <= 0:
        raise ValueError(f"generated_n_per_class must be > 0, got {generated_n_per_class}.")
    if int(real_reference_reps) <= 0:
        raise ValueError(f"real_reference_reps must be > 0, got {real_reference_reps}.")
    if validation_reps is not None and int(validation_reps) <= 0:
        raise ValueError(f"validation_reps must be > 0 when provided, got {validation_reps}.")
    if float(display_null_widths) <= 0.0 or not np.isfinite(float(display_null_widths)):
        raise ValueError(f"display_null_widths must be > 0, got {display_null_widths}.")
    if not (0.0 <= float(validation_target_power) <= 1.0):
        raise ValueError(
            f"validation_target_power must be in [0, 1], got {validation_target_power}."
        )
    controls = validate_control_names(controls)
    binner, dav_computer, cold_mask = make_metric_context(cfg)
    class_caches = {}
    pseudo_sources_by_class = {}
    source_sampling_by_class = {}
    for class_id, real_k in sorted(real_by_class.items()):
        rng = np.random.default_rng(int(seed + 104729 * (class_id + 1)))
        pseudo_k, source_meta = sample_real_images_for_reference_draw(
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
    standardization_ranges = null_standardization_ranges(
        good_reference.get("summary", {}),
        null_quantile=0.95,
    )
    good_values = dict(good_reference.get("values", {}))
    corruption_validation = compute_corruption_validation(
        class_caches=class_caches,
        pseudo_sources_by_class=pseudo_sources_by_class,
        controls=controls,
        grids=default_validation_grids(cfg),
        cfg=cfg,
        reps=int(real_reference_reps if validation_reps is None else validation_reps),
        seed=int(seed),
        real_vs_real_summary=good_reference.get("summary", {}),
        target_power=float(validation_target_power),
    )

    report = {
        "schema": "climategen_eval.metric_standardization_report.v1",
        "metric_standardization": {
            "enabled": True,
            "schema": "climategen_eval.metric_standardization.v1",
            "source": "real_vs_real_null_standardization",
            "match_generated_counts": True,
            "seed": int(seed),
            "real_reference_reps": int(real_reference_reps),
            "good_reference": {
                "values": good_values,
                "physical_real_vs_real": good_reference,
            },
            "null_standardization": {
                "enabled": True,
                "schema": "climategen_eval.null_standardization.v1",
                "source": "real_vs_real_bootstrap",
                "good_statistic": "q50",
                "null_quantile": 0.95,
                "display_transform": "linear_clipped",
                "display_null_widths": float(display_null_widths),
                "definition": (
                    "standardized_gap = max(0, model_value - q50(null)) / "
                    "(q95(null) - q50(null)) for lower-is-better metrics; "
                    "higher-is-better metrics use the mirrored q50 - q05 width."
                ),
                "ranges": standardization_ranges,
            },
        },
        "metric_validation": {
            "corruption_validation": corruption_validation,
            "source_sampling_by_class": {
                str(c): dict(meta) for c, meta in sorted(source_sampling_by_class.items())
            },
        },
    }
    return report
