from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np

from .corruptions import CONTROL_NAMES, build_validation_control_bt_k
from .tc_bt import (
    TCMetricConfig,
    _resample_side,
    build_class_metric_cache,
    build_class_metric_cache_from_real_cache,
    compute_class_scalar_summary,
    compute_real_vs_real_class_summary,
    make_metric_context,
)


RANGE_DIRECTIONS = {
    "pixel_hist_js": "lower",
    "pixel_hist_w1": "lower",
    "dav_w1_deg2": "lower",
    "radial_profile_mae_k": "lower",
    "cold_cloud_fraction_w1": "lower",
    "psd_l2": "lower",
    "eye_contrast_proxy_w1_k": "lower",
    "feature_mmd2": "lower",
    "diversity_feature_space_abs_gap": "lower",
    "gen_exceedance_rate_total": "lower",
}


def progress_bar(*, enabled: bool, **kwargs: Any) -> Any:
    if not enabled:
        return nullcontext(None)
    from tqdm.auto import tqdm

    return tqdm(**kwargs)


def summarize_distribution(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot summarize an empty distribution.")
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "mean": float(np.mean(arr)),
        "std": std,
        "min": float(np.min(arr)),
        "q025": float(np.quantile(arr, 0.025)),
        "q05": float(np.quantile(arr, 0.05)),
        "q50": float(np.quantile(arr, 0.50)),
        "q95": float(np.quantile(arr, 0.95)),
        "q975": float(np.quantile(arr, 0.975)),
        "max": float(np.max(arr)),
        "interval_95": {
            "low_quantile": 0.025,
            "high_quantile": 0.975,
            "low": float(np.quantile(arr, 0.025)),
            "high": float(np.quantile(arr, 0.975)),
        },
    }


def null_standardization_ranges(
    real_vs_real_summary: dict[str, dict[str, float]],
    *,
    null_quantile: float = 0.95,
    min_scale: float = 1.0e-12,
) -> dict[str, dict[str, Any]]:
    """Build metric standardization ranges from real-vs-real bootstrap summaries.

    The scale is the one-sided upper null half-width q95 - q50 for
    lower-is-better metrics, and the mirrored q50 - q05 half-width for
    higher-is-better metrics. With this definition, standardized_gap = 1
    corresponds to the model sitting exactly at the upper (or lower) noise
    floor: anything inside [0, 1] is within the central 90% real-vs-real null
    along the direction of badness. Corruptions are intentionally not used
    here; they are validation diagnostics, not the display ruler.
    """
    if abs(float(null_quantile) - 0.95) > 1.0e-12:
        raise ValueError("null_standardization_ranges currently expects null_quantile=0.95.")

    out: dict[str, dict[str, Any]] = {}
    for key, node in sorted(real_vs_real_summary.items()):
        if key not in RANGE_DIRECTIONS:
            continue
        direction = RANGE_DIRECTIONS[key]
        good = float(node.get("q50", np.nan))
        lower_boundary = float(node.get("q05", np.nan))
        upper_boundary = float(node.get("q95", np.nan))
        if direction == "higher":
            boundary = lower_boundary
            scale = good - lower_boundary
            scale_definition = "q50_minus_q05"
        else:
            boundary = upper_boundary
            scale = upper_boundary - good
            scale_definition = "q95_minus_q50"
        usable = bool(
            np.isfinite(good)
            and np.isfinite(boundary)
            and np.isfinite(lower_boundary)
            and np.isfinite(upper_boundary)
            and np.isfinite(scale)
            and scale > float(min_scale)
        )
        out[key] = {
            "direction": direction,
            "good": good,
            "null_boundary": boundary,
            "null_lower_boundary": lower_boundary,
            "null_upper_boundary": upper_boundary,
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
) -> dict[str, Any]:
    """Standardize a raw metric value against a real-vs-real noise floor.

    The returned ``standardized_gap`` is a one-sided skill-score-like quantity:
    it measures how far the model's metric value lies beyond the real-vs-real
    finite-sample median, in units of the upper (or lower) half-width of the
    real-vs-real null. This mirrors the structure of skill scores in
    operational weather verification (Murphy 1988; Jolliffe & Stephenson,
    *Forecast Verification* 2e), where a score is normalized by the gap
    between a reference and a worst-case anchor. Here the reference is the
    real-vs-real median (the best a model can achieve at finite sample size),
    and the half-width q95-q50 is the unit of "detectable difference under
    sampling noise" rather than a worst-case anchor.

    With this scale, ``gap = 0`` matches the real-vs-real median, ``gap = 1``
    sits exactly at the upper boundary of the central 90% null, and larger
    values indicate progressively stronger departures from real-vs-real
    sampling variation. The companion baseline panel (CLIM, GAUSS-PIX,
    CROSS-CLASS, PHASE-SHUFFLE) anchors the bad end of the same axis.
    """
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
    return {
        "value": raw_value,
        "direction": direction,
        "good": good,
        "scale": scale,
        "standardized_gap": float(gap),
    }


def standardize_metric_values(
    values: dict[str, float],
    ranges: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, value in sorted(values.items()):
        standardization = ranges.get(key)
        if not standardization or not standardization.get("usable_for_standardized_gap", False):
            continue
        out[key] = standardized_metric_value(
            value=float(value),
            standardization=standardization,
        )
    return out


def standardize_metric_sample_distributions(
    samples_by_key: dict[str, Any],
    ranges: dict[str, dict[str, Any]],
    *,
    point_values: dict[str, float],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, raw_samples in sorted(samples_by_key.items()):
        standardization = ranges.get(key)
        if not standardization or not standardization.get("usable_for_standardized_gap", False):
            continue
        values = np.asarray(raw_samples, dtype=np.float64).reshape(-1)
        if values.size == 0:
            continue
        transformed = [
            standardized_metric_value(
                value=float(value),
                standardization=standardization,
            )
            for value in values
        ]
        point = None
        if key in point_values:
            point = standardized_metric_value(
                value=float(point_values[key]),
                standardization=standardization,
            )
        gaps = np.asarray([node["standardized_gap"] for node in transformed], dtype=np.float64)
        out[key] = {
            "point": point,
            "standardized_gap": summarize_distribution(gaps),
        }
    return out


def sample_real_images_for_reference_draw(
    real_k: np.ndarray,
    *,
    draw_n: int,
    rng: np.random.Generator,
    storm_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray | None]:
    arr = np.asarray(real_k, dtype=np.float32)
    source_n = int(arr.shape[0])
    draw_n = int(draw_n)
    if source_n <= 0:
        raise ValueError("Reference draw requires at least one real image.")
    if draw_n <= 0:
        raise ValueError(f"draw_n must be > 0, got {draw_n}")
    replace = source_n < draw_n
    idx = rng.choice(source_n, size=draw_n, replace=replace)
    drawn_ids: np.ndarray | None = None
    if storm_ids is not None:
        ids = np.asarray(storm_ids).reshape(-1)
        if ids.shape[0] != source_n:
            raise ValueError(
                f"storm_ids must align with real_k rows, got {ids.shape[0]} "
                f"ids for {source_n} images."
            )
        drawn_ids = ids[idx]
    return np.asarray(arr[idx], dtype=np.float32), {
        "draw_n": draw_n,
        "source_n": source_n,
        "draw_mode": "with_replacement" if replace else "without_replacement",
    }, drawn_ids


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
    progress: bool = False,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    samples_by_key: dict[str, list[float]] = {}
    with progress_bar(
        enabled=progress,
        total=int(reps),
        desc="Real-vs-real null",
        unit="rep",
    ) as pbar:
        for _ in range(int(reps)):
            per_class = {}
            for class_id, cache in sorted(class_caches.items()):
                real_n = int(cache.real_hist_counts.shape[0])
                other_n = int(cache.gen_raw_hist_counts.shape[0])
                # The "right" side mirrors the model evaluation set size; the
                # "left" side mirrors the available real count. Both draws are
                # cluster-bootstrapped by storm when storm groups are present.
                left_idx = _resample_side(
                    cluster_groups=cache.real_cluster_groups,
                    n=real_n,
                    target_size=real_n,
                    rng=rng,
                )
                right_idx = _resample_side(
                    cluster_groups=cache.real_cluster_groups,
                    n=real_n,
                    target_size=other_n,
                    rng=rng,
                )
                per_class[int(class_id)] = compute_real_vs_real_class_summary(
                    cache,
                    left_idx=left_idx,
                    right_idx=right_idx,
                )
            append_macro_samples(samples_by_key, per_class)
            if pbar is not None:
                pbar.update(1)
    return {key: np.asarray(values, dtype=np.float64) for key, values in samples_by_key.items()}


def bootstrap_control_samples(
    control_caches: dict[int, Any],
    *,
    reps: int,
    seed: int,
    progress_update: Any = None,
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
                real_idx=_resample_side(
                    cluster_groups=cache.real_cluster_groups,
                    n=real_n,
                    target_size=real_n,
                    rng=rng,
                ),
                gen_idx=_resample_side(
                    cluster_groups=cache.gen_cluster_groups,
                    n=gen_n,
                    target_size=gen_n,
                    rng=rng,
                ),
            )
        append_macro_samples(samples_by_key, per_class)
        if progress_update is not None:
            progress_update(1)
    return {key: np.asarray(values, dtype=np.float64) for key, values in samples_by_key.items()}


def compute_bootstrap_calibration(
    class_caches: dict[int, Any],
    *,
    standardization_ranges: dict[str, dict[str, Any]],
    reps: int,
    seed: int,
    progress: bool = False,
) -> dict[str, Any]:
    """Empirical false-positive-rate check for the standardized-gap procedure.

    Generates an *independent* set of real-vs-real bootstrap replicates (with
    a different seed from the primary noise-floor bootstrap) and counts how
    often each one would cross ``gap = 1`` when standardized against the
    noise floor. Under correct calibration the rate should be approximately
    the theoretical ``1 - null_quantile`` (5% for a 0.95 quantile), since
    ``q95`` cuts off 5% of the null distribution by construction.

    A large empirical excess over 5% means the primary bootstrap
    underestimates real-vs-real fluctuation (the q95 it produces is too
    tight); a large deficit means it overestimates. Both are diagnostic of
    a methodological issue in the bootstrap — e.g. the wrong cluster unit,
    or an estimator whose finite-sample bias is interacting with the
    standardization.
    """
    if int(reps) <= 0:
        return {"enabled": False, "reason": "reps <= 0"}
    samples = bootstrap_real_vs_real_samples(
        class_caches,
        reps=int(reps),
        seed=int(seed),
        progress=progress,
    )
    n = int(reps)
    theoretical_fpr = 0.05
    # 2-sigma tolerance band on a binomial proportion at p=0.05, n replicates.
    se = float(np.sqrt(theoretical_fpr * (1.0 - theoretical_fpr) / max(n, 1)))
    tolerance_band = [
        float(max(0.0, theoretical_fpr - 2.0 * se)),
        float(min(1.0, theoretical_fpr + 2.0 * se)),
    ]
    by_metric: dict[str, Any] = {}
    for key, values in sorted(samples.items()):
        standardization = standardization_ranges.get(key)
        if not standardization or not standardization.get("usable_for_standardized_gap", False):
            continue
        gaps = np.asarray(
            [
                standardized_metric_value(value=float(v), standardization=standardization)[
                    "standardized_gap"
                ]
                for v in np.asarray(values, dtype=np.float64)
            ],
            dtype=np.float64,
        )
        n_exceed = int(np.sum(gaps > 1.0))
        emp_fpr = float(n_exceed / max(gaps.size, 1))
        within = bool(tolerance_band[0] <= emp_fpr <= tolerance_band[1])
        by_metric[key] = {
            "n_replicates": int(gaps.size),
            "n_exceeding_gap_one": n_exceed,
            "empirical_fpr": emp_fpr,
            "within_2sigma_tolerance": within,
        }
    return {
        "enabled": True,
        "schema": "climategen_eval.bootstrap_calibration.v1",
        "method": (
            "independent_real_vs_real_cluster_bootstrap_standardized_against_primary_noise_floor"
        ),
        "reps": int(reps),
        "seed": int(seed),
        "theoretical_fpr": theoretical_fpr,
        "tolerance_band_2sigma": tolerance_band,
        "by_metric": by_metric,
    }


def real_vs_real_reference(
    class_caches: dict[int, Any],
    *,
    reps: int,
    seed: int,
    progress: bool = False,
) -> dict[str, Any]:
    if int(reps) <= 0:
        return {"enabled": False, "reason": "reps <= 0"}
    samples = bootstrap_real_vs_real_samples(
        class_caches,
        reps=int(reps),
        seed=int(seed),
        progress=progress,
    )
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


def metric_directional_response(*, direction: str, reference: float, value: float) -> float:
    if str(direction) == "higher":
        return float(reference) - float(value)
    return float(value) - float(reference)


def finite_float_or_none(value: float) -> float | None:
    out = float(value)
    return out if np.isfinite(out) else None


def average_ranks(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    ranks = np.empty(arr.shape[0], dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    sorted_arr = arr[order]
    start = 0
    while start < sorted_arr.shape[0]:
        stop = start + 1
        while stop < sorted_arr.shape[0] and sorted_arr[stop] == sorted_arr[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * float(start + stop - 1)
        start = stop
    return ranks


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    left = np.asarray(x, dtype=np.float64).reshape(-1)
    right = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if left.size < 2:
        return None
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    left_std = float(np.std(left_ranks))
    right_std = float(np.std(right_ranks))
    if left_std <= 0.0 or right_std <= 0.0:
        return None
    rho = float(np.corrcoef(left_ranks, right_ranks)[0, 1])
    return finite_float_or_none(rho)


def response_in_null_widths(*, response: float, null_width: float) -> float | None:
    width = float(null_width)
    if width <= 0.0 or not np.isfinite(width):
        return None
    return finite_float_or_none(float(response) / width)


def compact_response_point(
    point: dict[str, Any],
    *,
    direction: str,
    good: float,
    null_width: float,
) -> dict[str, Any]:
    value = float(point["value"])
    response = metric_directional_response(
        direction=direction,
        reference=float(good),
        value=value,
    )
    return {
        "severity": float(point["severity"]),
        "value": value,
        "power": float(point["power"]),
        "directional_response": response,
        "directional_response_in_null_widths": response_in_null_widths(
            response=response,
            null_width=float(null_width),
        ),
    }


def summarize_control_response_shape(
    points: list[dict[str, Any]],
    *,
    direction: str,
    good: float,
    threshold: float,
    target_power: float,
    tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    """Summarize whether a metric has a coherent response over a degradation sweep."""
    sorted_points = sorted(points, key=lambda node: float(node["severity"]))
    null_width = abs(float(threshold) - float(good))
    positive_points = [
        point for point in sorted_points if float(point.get("severity", 0.0)) > 0.0
    ]
    analysis_points = positive_points if positive_points else sorted_points

    severities = np.asarray([float(point["severity"]) for point in sorted_points])
    values = np.asarray([float(point["value"]) for point in sorted_points])
    responses = np.asarray(
        [
            metric_directional_response(
                direction=direction,
                reference=float(good),
                value=float(point["value"]),
            )
            for point in sorted_points
        ],
        dtype=np.float64,
    )

    positive_responses = np.asarray(
        [
            metric_directional_response(
                direction=direction,
                reference=float(good),
                value=float(point["value"]),
            )
            for point in positive_points
        ],
        dtype=np.float64,
    )
    if positive_responses.size:
        directionally_worse_fraction = float(np.mean(positive_responses > float(tolerance)))
        non_improving_fraction = float(np.mean(positive_responses >= -float(tolerance)))
        max_directional_response = float(np.max(positive_responses))
    else:
        directionally_worse_fraction = 0.0
        non_improving_fraction = 0.0
        max_directional_response = float(np.max(responses)) if responses.size else float("nan")

    monotone_steps = []
    for left, right in zip(range(len(sorted_points) - 1), range(1, len(sorted_points))):
        if severities[right] > severities[left]:
            monotone_steps.append(bool(responses[right] >= responses[left] - float(tolerance)))
    monotone_worsening_fraction = (
        float(np.mean(monotone_steps)) if monotone_steps else None
    )

    target_power_points = [
        point for point in positive_points if float(point.get("power", 0.0)) >= target_power
    ]
    justified_points = [
        point
        for point in target_power_points
        if metric_directional_response(
            direction=direction,
            reference=float(good),
            value=float(point["value"]),
        )
        > float(tolerance)
    ]
    max_power = max(
        (float(point.get("power", 0.0)) for point in positive_points),
        default=0.0,
    )
    max_response_point = max(
        analysis_points,
        key=lambda point: metric_directional_response(
            direction=direction,
            reference=float(good),
            value=float(point["value"]),
        ),
    ) if analysis_points else None
    first_target_power_point = target_power_points[0] if target_power_points else None
    first_justified_point = justified_points[0] if justified_points else None

    return {
        "n_points": int(len(sorted_points)),
        "n_positive_severity_points": int(len(positive_points)),
        "expected_raw_correlation_sign": "negative" if direction == "higher" else "positive",
        "spearman_severity_value": spearman_correlation(severities, values),
        "spearman_severity_worsening_response": spearman_correlation(
            severities,
            responses,
        ),
        "directionally_worse_fraction": directionally_worse_fraction,
        "directionally_non_improving_fraction": non_improving_fraction,
        "directionally_consistent": bool(
            positive_responses.size
            and non_improving_fraction == 1.0
            and max_directional_response > float(tolerance)
        ),
        "monotone_worsening_fraction": monotone_worsening_fraction,
        "monotone_worsening": (
            bool(monotone_steps) and bool(all(monotone_steps))
            if monotone_steps
            else False
        ),
        "first_target_power_severity": (
            float(first_target_power_point["severity"])
            if first_target_power_point is not None
            else None
        ),
        "first_target_power_point": (
            compact_response_point(
                first_target_power_point,
                direction=direction,
                good=float(good),
                null_width=null_width,
            )
            if first_target_power_point is not None
            else None
        ),
        "first_justified_severity": (
            float(first_justified_point["severity"])
            if first_justified_point is not None
            else None
        ),
        "first_justified_point": (
            compact_response_point(
                first_justified_point,
                direction=direction,
                good=float(good),
                null_width=null_width,
            )
            if first_justified_point is not None
            else None
        ),
        "max_power": float(max_power),
        "max_response": (
            compact_response_point(
                max_response_point,
                direction=direction,
                good=float(good),
                null_width=null_width,
            )
            if max_response_point is not None
            else None
        ),
        "justifies_inclusion": bool(first_justified_point is not None),
    }


def response_shape_strength(shape: dict[str, Any]) -> tuple[float, float]:
    max_response = shape.get("max_response", {})
    if not isinstance(max_response, dict):
        return -float("inf"), -float("inf")
    response_widths = max_response.get("directional_response_in_null_widths")
    response_raw = max_response.get("directional_response")
    return (
        float(response_widths) if response_widths is not None else -float("inf"),
        float(response_raw) if response_raw is not None else -float("inf"),
    )


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
    progress: bool = False,
    pseudo_source_ids_by_class: dict[int, np.ndarray] | None = None,
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
    jobs = [
        (control_idx, control_name, value_idx, severity)
        for control_idx, control_name in enumerate(controls)
        for value_idx, severity in enumerate(grids.get(control_name, []))
    ]
    with progress_bar(
        enabled=progress,
        total=len(jobs) * int(reps),
        desc="Corruption validation",
        unit="rep",
    ) as pbar:
        for control_idx, control_name, value_idx, severity in jobs:
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
                gen_ids = None
                if pseudo_source_ids_by_class is not None:
                    gen_ids = pseudo_source_ids_by_class.get(class_id)
                control_caches[class_id] = build_class_metric_cache_from_real_cache(
                    real_cache=class_caches[class_id],
                    gen_k_raw=control_k,
                    cfg=cfg,
                    binner=binner,
                    dav_computer=dav_computer,
                    cold_mask=cold_mask,
                    gen_storm_ids=gen_ids,
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
                progress_update=None if pbar is None else pbar.update,
            )
            for key, values in sorted(samples_by_metric.items()):
                if key not in thresholds:
                    continue
                direction = RANGE_DIRECTIONS.get(key, "lower")
                summary = summarize_distribution(values)
                good = float(good_values[key])
                null_width = abs(float(thresholds[key]) - good)
                directional_response = metric_directional_response(
                    direction=direction,
                    reference=good,
                    value=float(summary["q50"]),
                )
                point = {
                    "severity": float(severity),
                    "threshold": float(thresholds[key]),
                    "power": detection_power(
                        values,
                        threshold=thresholds[key],
                        direction=direction,
                    ),
                    "value": float(summary["q50"]),
                    "directional_response": float(directional_response),
                    "directional_response_in_null_widths": response_in_null_widths(
                        response=directional_response,
                        null_width=null_width,
                    ),
                    "summary": summary,
                }
                curves_by_control.setdefault(control_name, {}).setdefault(key, []).append(point)

    responsiveness_by_metric: dict[str, dict[str, Any]] = {}
    for key in sorted(thresholds):
        direction = RANGE_DIRECTIONS.get(key, "lower")
        good = float(good_values[key])
        responsive_controls = {}
        max_power_by_control = {}
        response_shape_by_control = {}
        justifying_controls = []
        for control_name in controls:
            points = sorted(
                curves_by_control.get(control_name, {}).get(key, []),
                key=lambda node: float(node["severity"]),
            )
            response_shape = summarize_control_response_shape(
                points,
                direction=direction,
                good=good,
                threshold=float(thresholds[key]),
                target_power=float(target_power),
            )
            response_shape_by_control[control_name] = response_shape
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
                    justifying_controls.append(control_name)
                    break
        primary_justifying_control = None
        if justifying_controls:
            primary_justifying_control = max(
                justifying_controls,
                key=lambda name: response_shape_strength(response_shape_by_control[name]),
            )
        responsiveness_by_metric[key] = {
            "direction": direction,
            "good_value": good,
            "null_threshold": float(thresholds[key]),
            "target_power": float(target_power),
            "responsive": bool(responsive_controls),
            "justifying_controls": justifying_controls,
            "primary_justifying_control": primary_justifying_control,
            "responsive_controls": responsive_controls,
            "max_power_by_control": max_power_by_control,
            "response_shape_by_control": response_shape_by_control,
        }

    return {
        "enabled": True,
        "schema": "climategen_eval.corruption_validation.v1",
        "role": "audit_only",
        "role_note": (
            "Since v4 of the framework, the corruption sweep is retained as a "
            "supplementary metric-validation audit (response shape, monotonicity, "
            "Spearman correlations under known degradations). It is no longer the "
            "preferred metric-inclusion gate; that role belongs to the baseline "
            "panel. The fields below remain a defensible legacy fallback gate."
        ),
        "source": "corrupted_real_reference_images",
        "responsiveness_decision": (
            "A metric is responsive when at least one positive-severity degradation "
            "family reaches target power and moves the metric in the expected worse "
            "direction. Response-shape fields are reported as audit evidence."
        ),
        "response_shape_definition": {
            "directional_response": (
                "Metric displacement from the real-vs-real q50 in the expected worse "
                "direction; positive values mean degradation worsened the metric."
            ),
            "directional_response_in_null_widths": (
                "directional_response divided by the one-sided real-vs-real null width"
            ),
            "spearman_severity_worsening_response": (
                "Spearman correlation between degradation severity and directional_response"
            ),
            "directionally_consistent": (
                "All positive-severity points are non-improving relative to the "
                "real-vs-real q50, and at least one point is worse."
            ),
            "monotone_worsening": (
                "directional_response is non-decreasing across increasing severities"
            ),
        },
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
    calibration_reps: int | None = None,
    progress: bool = False,
    real_storm_ids_by_class: dict[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    if not real_by_class:
        raise ValueError("real_by_class must contain at least one class.")
    if int(generated_n_per_class) <= 0:
        raise ValueError(f"generated_n_per_class must be > 0, got {generated_n_per_class}.")
    if int(real_reference_reps) <= 0:
        raise ValueError(f"real_reference_reps must be > 0, got {real_reference_reps}.")
    if validation_reps is not None and int(validation_reps) <= 0:
        raise ValueError(f"validation_reps must be > 0 when provided, got {validation_reps}.")
    if not (0.0 <= float(validation_target_power) <= 1.0):
        raise ValueError(
            f"validation_target_power must be in [0, 1], got {validation_target_power}."
        )
    controls = validate_control_names(controls)
    binner, dav_computer, cold_mask = make_metric_context(cfg)
    storm_ids_by_class = real_storm_ids_by_class or {}
    class_caches = {}
    pseudo_sources_by_class = {}
    pseudo_source_ids_by_class: dict[int, np.ndarray] = {}
    source_sampling_by_class = {}
    class_items = sorted(real_by_class.items())
    with progress_bar(
        enabled=progress,
        total=len(class_items),
        desc="Building reference caches",
        unit="class",
    ) as pbar:
        for class_id, real_k in class_items:
            rng = np.random.default_rng(int(seed + 104729 * (class_id + 1)))
            real_ids = storm_ids_by_class.get(class_id)
            pseudo_k, source_meta, pseudo_ids = sample_real_images_for_reference_draw(
                real_k,
                draw_n=int(generated_n_per_class),
                rng=rng,
                storm_ids=real_ids,
            )
            pseudo_sources_by_class[class_id] = pseudo_k
            source_sampling_by_class[class_id] = source_meta
            if pseudo_ids is not None:
                pseudo_source_ids_by_class[class_id] = pseudo_ids
            class_caches[class_id] = build_class_metric_cache(
                real_k=real_k,
                gen_k_raw=pseudo_k,
                cfg=cfg,
                binner=binner,
                dav_computer=dav_computer,
                cold_mask=cold_mask,
                real_storm_ids=real_ids,
                gen_storm_ids=pseudo_ids,
            )
            if pbar is not None:
                pbar.update(1)

    good_reference = real_vs_real_reference(
        class_caches,
        reps=int(real_reference_reps),
        seed=int(seed),
        progress=progress,
    )
    standardization_ranges = null_standardization_ranges(
        good_reference.get("summary", {}),
        null_quantile=0.95,
    )
    good_values = dict(good_reference.get("values", {}))
    # Independent calibration bootstrap. Uses a disjoint seed so the
    # replicates do not share Monte Carlo state with the noise-floor draw,
    # which is what makes the FPR meaningful.
    bootstrap_calibration = compute_bootstrap_calibration(
        class_caches,
        standardization_ranges=standardization_ranges,
        reps=int(
            real_reference_reps if calibration_reps is None else calibration_reps
        ),
        seed=int(seed) + 1_000_003,
        progress=progress,
    )
    corruption_validation = compute_corruption_validation(
        class_caches=class_caches,
        pseudo_sources_by_class=pseudo_sources_by_class,
        pseudo_source_ids_by_class=pseudo_source_ids_by_class,
        controls=controls,
        grids=default_validation_grids(cfg),
        cfg=cfg,
        reps=int(real_reference_reps if validation_reps is None else validation_reps),
        seed=int(seed),
        real_vs_real_summary=good_reference.get("summary", {}),
        target_power=float(validation_target_power),
        progress=progress,
    )

    report = {
        "schema": "climategen_eval.metric_standardization_report.v4",
        "aggregation": {
            "class_aggregation": "unweighted_macro_average_over_classes",
            "rationale": "Each intensity class contributes equally to aggregate metrics.",
        },
        "metric_standardization": {
            "enabled": True,
            "schema": "climategen_eval.metric_standardization.v4",
            "source": "real_vs_real_null_standardization",
            "match_generated_counts": True,
            "seed": int(seed),
            "real_reference_reps": int(real_reference_reps),
            "real_vs_real_design": {
                "left_sample_size": "available_real_count_per_class",
                "right_sample_size": int(generated_n_per_class),
                "sampling": "bootstrap_with_replacement_within_class",
                "class_aggregation": "unweighted_macro_average_over_classes",
            },
            "good_reference": {
                "values": good_values,
                "physical_real_vs_real": good_reference,
            },
            "null_standardization": {
                "enabled": True,
                "schema": "climategen_eval.null_standardization.v4",
                "source": "real_vs_real_bootstrap",
                "good_statistic": "q50",
                "null_quantile": 0.95,
                "interpretation": {
                    "family": "skill_score_like",
                    "reference_anchor": "real_vs_real_finite_sample_median_q50",
                    "scale_unit": "upper_one_sided_null_half_width_q95_minus_q50",
                    "convention_references": [
                        "Murphy 1988, Mon. Wea. Rev. 116: skill score formalism",
                        "Jolliffe and Stephenson, Forecast Verification 2e (2012)",
                        "WMO No.485 Manual on the GDPFS, Vol. I (verification guidance)",
                    ],
                    "reading_guide": (
                        "gap = 0 matches the real-vs-real median; gap = 1 sits at the "
                        "upper edge of the central 90% real-vs-real null; gap > 1 means "
                        "the model is detectably worse than sampling noise can explain. "
                        "The bad-but-sensible baseline panel (CLIM, GAUSS-PIX, "
                        "CROSS-CLASS, PHASE-SHUFFLE) anchors the bad end of the same axis."
                    ),
                },
                "definition": (
                    "standardized_gap = max(0, model_value - q50(null)) / "
                    "(q95(null) - q50(null)) for lower-is-better metrics; "
                    "higher-is-better metrics use max(0, q50(null) - model_value) / "
                    "(q50(null) - q05(null)). With this scale, gap = 1 places the "
                    "model exactly at the upper (or lower) noise-floor boundary of "
                    "the central 90% real-vs-real null."
                ),
                "metric_inclusion_rule": (
                    "A metric is included in the standardized-gap profile when its "
                    "real-vs-real null width is finite and positive. Downstream "
                    "consumers should further gate by baseline-panel responsiveness "
                    "(preferred: --baseline-responsiveness from "
                    "evaluate_tc_bt_baselines.py). The corruption-validation "
                    "responsiveness block under metric_validation is kept as audit "
                    "evidence only; it is still wired as a legacy fallback gate when "
                    "no baseline-panel file is supplied."
                ),
                "ranges": standardization_ranges,
            },
        },
        "metric_validation": {
            "bootstrap_calibration": bootstrap_calibration,
            "corruption_validation": corruption_validation,
            "source_sampling_by_class": {
                str(c): dict(meta) for c, meta in sorted(source_sampling_by_class.items())
            },
        },
    }
    return report
