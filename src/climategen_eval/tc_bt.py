from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import (
    DAVComputer,
    PolarBinner,
    centered_disk_mask,
    cold_cloud_fraction,
    eye_contrast_proxy,
    flatten_features_for_diversity,
    js_divergence,
    psd_radial_batch,
    radial_profile_batch,
    rbf_mmd2,
    summary_stats,
    wasserstein1_from_hist,
)


PRIMARY_RAW_AGG_KEYS = (
    "pixel_hist_js",
    "pixel_hist_w1",
    "cold_cloud_fraction_abs_gap",
    "dav_abs_gap_deg2",
    "eye_contrast_proxy_abs_gap",
    "psd_l2",
    "feature_mmd2",
    "diversity_feature_space_abs_gap",
    "gen_exceedance_rate_total",
    "radial_profile_mae_k",
)

BOOTSTRAP_QUANTILES = (0.025, 0.05, 0.50, 0.95, 0.975)


def progress_iter(iterable: Any, *, enabled: bool, **kwargs: Any) -> Any:
    if not enabled:
        return iterable
    from tqdm.auto import tqdm

    return tqdm(iterable, **kwargs)


def quantile_key(q: float) -> str:
    value = float(q)
    if abs(value - 0.50) < 1.0e-12:
        return "q50"
    if abs(value - 0.05) < 1.0e-12:
        return "q05"
    if abs(value - 0.95) < 1.0e-12:
        return "q95"
    return f"q{int(round(value * 1000.0)):03d}"


def summarize_metric_samples(
    samples_by_key: dict[str, np.ndarray],
    *,
    point_values: dict[str, float],
) -> dict[str, dict[str, float | dict[str, float]]]:
    out: dict[str, dict[str, float | dict[str, float]]] = {}
    for key, samples in sorted(samples_by_key.items()):
        arr = np.asarray(samples, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            continue
        summary: dict[str, float | dict[str, float]] = {
            "point": float(point_values[key]) if key in point_values else float(np.nan),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
        for q in BOOTSTRAP_QUANTILES:
            summary[quantile_key(q)] = float(np.quantile(arr, q))
        summary["interval_95"] = {
            "low_quantile": 0.025,
            "high_quantile": 0.975,
            "low": float(np.quantile(arr, 0.025)),
            "high": float(np.quantile(arr, 0.975)),
        }
        out[key] = summary
    return out


@dataclass
class TCMetricConfig:
    image_size: int = 256
    bt_min_k: float = 170.0
    bt_max_k: float = 325.0
    profile_bins: int = 96
    psd_bins: int = 96
    dav_radius_km: float = 300.0
    dav_pixel_size_km: float = 8.0
    dav_center_region_size: int = 3
    cold_cloud_threshold_k: float = 208.0
    cold_cloud_radius_km: float = 300.0
    cold_cloud_pixel_size_km: float = 8.0


@dataclass
class ClassMetricCache:
    bins: np.ndarray
    real_hist_counts: np.ndarray
    gen_raw_hist_counts: np.ndarray
    real_mean_profiles: np.ndarray
    gen_raw_mean_profiles: np.ndarray
    real_psd_profiles: np.ndarray
    gen_raw_psd_profiles: np.ndarray
    real_cold: np.ndarray
    gen_raw_cold: np.ndarray
    real_dav: np.ndarray
    gen_raw_dav: np.ndarray
    real_eye: np.ndarray
    gen_raw_eye: np.ndarray
    real_features: np.ndarray
    gen_raw_features: np.ndarray
    feature_scaler: tuple[np.ndarray, np.ndarray]
    feature_gamma_raw: float
    real_pixel_mean: np.ndarray
    gen_raw_pixel_mean: np.ndarray
    gen_raw_exceed_total: np.ndarray
    pixel_stats_real: dict[str, float]
    pixel_stats_gen_raw: dict[str, float]
    gen_exceedance_rate: dict[str, float]


def ensure_metric_bt_batch(name: str, value: np.ndarray, cfg: TCMetricConfig) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    expected_shape = (int(cfg.image_size), int(cfg.image_size))
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape (N,H,W), got {arr.shape}.")
    if arr.shape[0] <= 0:
        raise ValueError(f"{name} must contain at least one image.")
    if tuple(arr.shape[1:]) != expected_shape:
        raise ValueError(
            f"{name} images must have shape {expected_shape}, got {tuple(arr.shape[1:])}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def make_metric_context(cfg: TCMetricConfig) -> tuple[PolarBinner, DAVComputer, np.ndarray]:
    binner = PolarBinner(cfg.image_size, cfg.image_size, int(cfg.profile_bins), 360)
    dav_computer = DAVComputer(
        cfg.image_size,
        cfg.image_size,
        pixel_size_km=float(cfg.dav_pixel_size_km),
        radius_km=float(cfg.dav_radius_km),
        center_region_size=int(cfg.dav_center_region_size),
    )
    cold_mask = centered_disk_mask(
        cfg.image_size,
        cfg.image_size,
        pixel_size_km=float(cfg.cold_cloud_pixel_size_km),
        radius_km=float(cfg.cold_cloud_radius_km),
    )
    return binner, dav_computer, cold_mask


def hist_counts_batch(imgs: np.ndarray, bins: np.ndarray) -> np.ndarray:
    out = np.zeros((imgs.shape[0], bins.shape[0] - 1), dtype=np.float64)
    for i in range(imgs.shape[0]):
        out[i], _ = np.histogram(imgs[i].reshape(-1), bins=bins)
    return out


def hist_density_from_counts(counts: np.ndarray) -> np.ndarray:
    h = np.asarray(counts, dtype=np.float64)
    if h.ndim == 2:
        h = h.sum(axis=0)
    return h / (h.sum() + 1.0e-12)


def mean_pair_dist(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=np.float64)
    n = int(z.shape[0])
    if n < 2:
        return 0.0
    block_size = 128
    total = 0.0
    norms = np.sum(z * z, axis=1)
    for i0 in range(0, n, block_size):
        i1 = min(i0 + block_size, n)
        zi = z[i0:i1]
        for j0 in range(i0, n, block_size):
            j1 = min(j0 + block_size, n)
            d2 = norms[i0:i1, None] + norms[j0:j1][None, :] - 2.0 * (zi @ z[j0:j1].T)
            np.maximum(d2, 0.0, out=d2)
            np.sqrt(d2, out=d2)
            if i0 == j0:
                total += float(d2.sum() - np.trace(d2))
            else:
                total += 2.0 * float(d2.sum())
    return float(total / (n * (n - 1)))


def median_heuristic_gamma(x: np.ndarray, y: np.ndarray) -> float:
    z = np.vstack([x, y]).astype(np.float64)
    if z.shape[0] < 2:
        return 1.0
    rng = np.random.default_rng(0)
    idx = rng.choice(z.shape[0], size=min(200, z.shape[0]), replace=False)
    zs = z[idx]
    d2 = np.sum((zs[:, None, :] - zs[None, :, :]) ** 2, axis=-1)
    positive = d2[d2 > 0]
    if positive.size == 0:
        return 1.0
    return float(1.0 / (2.0 * (np.median(positive) + 1.0e-12)))


def compute_feature_metrics(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    gamma: float,
) -> tuple[float, float, float]:
    return (
        float(rbf_mmd2(real_features, gen_features, gamma=gamma)),
        mean_pair_dist(real_features),
        mean_pair_dist(gen_features),
    )


def mean_profile_mse_gap(left: np.ndarray, right: np.ndarray) -> float:
    left_mean = np.asarray(left, dtype=np.float64).mean(axis=0)
    right_mean = np.asarray(right, dtype=np.float64).mean(axis=0)
    diff = left_mean - right_mean
    return float(np.mean(diff * diff))


def build_class_metric_cache(
    *,
    real_k: np.ndarray,
    gen_k_raw: np.ndarray,
    cfg: TCMetricConfig,
    binner: PolarBinner | None = None,
    dav_computer: DAVComputer | None = None,
    cold_mask: np.ndarray | None = None,
) -> ClassMetricCache:
    real_k = ensure_metric_bt_batch("real_k", real_k, cfg)
    gen_k_raw = ensure_metric_bt_batch("gen_k_raw", gen_k_raw, cfg)
    if binner is None or dav_computer is None or cold_mask is None:
        binner, dav_computer, cold_mask = make_metric_context(cfg)

    bins = np.linspace(float(cfg.bt_min_k), float(cfg.bt_max_k), 129)
    real_hist_counts = hist_counts_batch(real_k, bins)
    gen_raw_hist_counts = hist_counts_batch(gen_k_raw, bins)

    real_mean_profiles, _real_azstd = radial_profile_batch(real_k, binner)
    gen_raw_mean_profiles, _gen_azstd = radial_profile_batch(gen_k_raw, binner)
    real_psd_profiles = psd_radial_batch(real_k, int(cfg.psd_bins))
    gen_raw_psd_profiles = psd_radial_batch(gen_k_raw, int(cfg.psd_bins))

    real_cold = cold_cloud_fraction(
        real_k,
        threshold_k=float(cfg.cold_cloud_threshold_k),
        mask=cold_mask,
    )
    gen_raw_cold = cold_cloud_fraction(
        gen_k_raw,
        threshold_k=float(cfg.cold_cloud_threshold_k),
        mask=cold_mask,
    )
    real_dav = dav_computer.batch(real_k)
    gen_raw_dav = dav_computer.batch(gen_k_raw)
    real_eye = eye_contrast_proxy(real_mean_profiles)
    gen_raw_eye = eye_contrast_proxy(gen_raw_mean_profiles)

    real_features, feature_scaler = flatten_features_for_diversity(
        real_mean_profiles,
        real_psd_profiles,
    )
    gen_raw_features, _ = flatten_features_for_diversity(
        gen_raw_mean_profiles,
        gen_raw_psd_profiles,
        scaler=feature_scaler,
    )
    feature_gamma_raw = median_heuristic_gamma(real_features, real_features)

    gen_raw_exceed_below = np.mean(gen_k_raw < float(cfg.bt_min_k), axis=(1, 2)).astype(np.float32)
    gen_raw_exceed_above = np.mean(gen_k_raw > float(cfg.bt_max_k), axis=(1, 2)).astype(np.float32)
    gen_raw_exceed_total = gen_raw_exceed_below + gen_raw_exceed_above

    return ClassMetricCache(
        bins=bins,
        real_hist_counts=real_hist_counts,
        gen_raw_hist_counts=gen_raw_hist_counts,
        real_mean_profiles=real_mean_profiles,
        gen_raw_mean_profiles=gen_raw_mean_profiles,
        real_psd_profiles=real_psd_profiles,
        gen_raw_psd_profiles=gen_raw_psd_profiles,
        real_cold=real_cold,
        gen_raw_cold=gen_raw_cold,
        real_dav=real_dav,
        gen_raw_dav=gen_raw_dav,
        real_eye=real_eye,
        gen_raw_eye=gen_raw_eye,
        real_features=real_features,
        gen_raw_features=gen_raw_features,
        feature_scaler=feature_scaler,
        feature_gamma_raw=feature_gamma_raw,
        real_pixel_mean=real_k.mean(axis=(1, 2)).astype(np.float32),
        gen_raw_pixel_mean=gen_k_raw.mean(axis=(1, 2)).astype(np.float32),
        gen_raw_exceed_total=gen_raw_exceed_total,
        pixel_stats_real=summary_stats(real_k.reshape(-1)),
        pixel_stats_gen_raw=summary_stats(gen_k_raw.reshape(-1)),
        gen_exceedance_rate={
            "below_bt_min": float(gen_raw_exceed_below.mean()),
            "above_bt_max": float(gen_raw_exceed_above.mean()),
            "total": float(gen_raw_exceed_total.mean()),
            "raw_min": float(np.min(gen_k_raw)),
            "raw_max": float(np.max(gen_k_raw)),
        },
    )


def build_class_metric_cache_from_real_cache(
    *,
    real_cache: ClassMetricCache,
    gen_k_raw: np.ndarray,
    cfg: TCMetricConfig,
    binner: PolarBinner,
    dav_computer: DAVComputer,
    cold_mask: np.ndarray,
) -> ClassMetricCache:
    gen_k_raw = ensure_metric_bt_batch("gen_k_raw", gen_k_raw, cfg)
    gen_raw_hist_counts = hist_counts_batch(gen_k_raw, real_cache.bins)
    gen_raw_mean_profiles, _gen_azstd = radial_profile_batch(gen_k_raw, binner)
    gen_raw_psd_profiles = psd_radial_batch(gen_k_raw, int(cfg.psd_bins))
    gen_raw_cold = cold_cloud_fraction(
        gen_k_raw,
        threshold_k=float(cfg.cold_cloud_threshold_k),
        mask=cold_mask,
    )
    gen_raw_dav = dav_computer.batch(gen_k_raw)
    gen_raw_eye = eye_contrast_proxy(gen_raw_mean_profiles)
    gen_raw_features, _ = flatten_features_for_diversity(
        gen_raw_mean_profiles,
        gen_raw_psd_profiles,
        scaler=real_cache.feature_scaler,
    )
    feature_gamma_raw = float(real_cache.feature_gamma_raw)
    gen_raw_exceed_below = np.mean(gen_k_raw < float(cfg.bt_min_k), axis=(1, 2)).astype(np.float32)
    gen_raw_exceed_above = np.mean(gen_k_raw > float(cfg.bt_max_k), axis=(1, 2)).astype(np.float32)
    gen_raw_exceed_total = gen_raw_exceed_below + gen_raw_exceed_above

    return ClassMetricCache(
        bins=real_cache.bins,
        real_hist_counts=real_cache.real_hist_counts,
        gen_raw_hist_counts=gen_raw_hist_counts,
        real_mean_profiles=real_cache.real_mean_profiles,
        gen_raw_mean_profiles=gen_raw_mean_profiles,
        real_psd_profiles=real_cache.real_psd_profiles,
        gen_raw_psd_profiles=gen_raw_psd_profiles,
        real_cold=real_cache.real_cold,
        gen_raw_cold=gen_raw_cold,
        real_dav=real_cache.real_dav,
        gen_raw_dav=gen_raw_dav,
        real_eye=real_cache.real_eye,
        gen_raw_eye=gen_raw_eye,
        real_features=real_cache.real_features,
        gen_raw_features=gen_raw_features,
        feature_scaler=real_cache.feature_scaler,
        feature_gamma_raw=feature_gamma_raw,
        real_pixel_mean=real_cache.real_pixel_mean,
        gen_raw_pixel_mean=gen_k_raw.mean(axis=(1, 2)).astype(np.float32),
        gen_raw_exceed_total=gen_raw_exceed_total,
        pixel_stats_real=real_cache.pixel_stats_real,
        pixel_stats_gen_raw=summary_stats(gen_k_raw.reshape(-1)),
        gen_exceedance_rate={
            "below_bt_min": float(gen_raw_exceed_below.mean()),
            "above_bt_max": float(gen_raw_exceed_above.mean()),
            "total": float(gen_raw_exceed_total.mean()),
            "raw_min": float(np.min(gen_k_raw)),
            "raw_max": float(np.max(gen_k_raw)),
        },
    )


def select_rows(arr: np.ndarray, idx: np.ndarray | None) -> np.ndarray:
    return arr if idx is None else arr[idx]


def class_radial_profile_mae(
    cache: ClassMetricCache,
    *,
    real_idx: np.ndarray | None = None,
    gen_idx: np.ndarray | None = None,
) -> float:
    real_profiles = select_rows(cache.real_mean_profiles, real_idx)
    gen_profiles = select_rows(cache.gen_raw_mean_profiles, gen_idx)
    return float(np.mean(np.abs(real_profiles.mean(axis=0) - gen_profiles.mean(axis=0))))


def compute_class_scalar_summary(
    cache: ClassMetricCache,
    *,
    real_idx: np.ndarray | None = None,
    gen_idx: np.ndarray | None = None,
) -> dict[str, float]:
    real_hist_counts = select_rows(cache.real_hist_counts, real_idx)
    gen_hist_counts = select_rows(cache.gen_raw_hist_counts, gen_idx)
    rh = hist_density_from_counts(real_hist_counts)
    gh = hist_density_from_counts(gen_hist_counts)

    real_psd = select_rows(cache.real_psd_profiles, real_idx)
    gen_psd = select_rows(cache.gen_raw_psd_profiles, gen_idx)
    mmd2, div_real, div_gen = compute_feature_metrics(
        select_rows(cache.real_features, real_idx),
        select_rows(cache.gen_raw_features, gen_idx),
        cache.feature_gamma_raw,
    )

    real_cold = select_rows(cache.real_cold, real_idx)
    gen_cold = select_rows(cache.gen_raw_cold, gen_idx)
    real_dav = select_rows(cache.real_dav, real_idx)
    gen_dav = select_rows(cache.gen_raw_dav, gen_idx)
    real_eye = select_rows(cache.real_eye, real_idx)
    gen_eye = select_rows(cache.gen_raw_eye, gen_idx)

    return {
        "pixel_hist_js": float(js_divergence(rh, gh)),
        "pixel_hist_w1": float(wasserstein1_from_hist(rh, gh, bin_edges=cache.bins)),
        "cold_cloud_fraction_abs_gap": float(abs(real_cold.mean() - gen_cold.mean())),
        "dav_abs_gap_deg2": float(abs(real_dav.mean() - gen_dav.mean())),
        "eye_contrast_proxy_abs_gap": float(abs(real_eye.mean() - gen_eye.mean())),
        "psd_l2": mean_profile_mse_gap(real_psd, gen_psd),
        "feature_mmd2": float(mmd2),
        "diversity_feature_space_abs_gap": float(abs(div_real - div_gen)),
        "gen_exceedance_rate_total": float(select_rows(cache.gen_raw_exceed_total, gen_idx).mean()),
        "real_pixel_mean": float(select_rows(cache.real_pixel_mean, real_idx).mean()),
        "gen_pixel_mean": float(select_rows(cache.gen_raw_pixel_mean, gen_idx).mean()),
        "radial_profile_mae_k": class_radial_profile_mae(cache, real_idx=real_idx, gen_idx=gen_idx),
    }


def compute_real_vs_real_class_summary(
    cache: ClassMetricCache,
    *,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
) -> dict[str, float]:
    left_hist = select_rows(cache.real_hist_counts, left_idx)
    right_hist = select_rows(cache.real_hist_counts, right_idx)
    left_mass = hist_density_from_counts(left_hist)
    right_mass = hist_density_from_counts(right_hist)
    left_radial = select_rows(cache.real_mean_profiles, left_idx)
    right_radial = select_rows(cache.real_mean_profiles, right_idx)
    left_psd = select_rows(cache.real_psd_profiles, left_idx)
    right_psd = select_rows(cache.real_psd_profiles, right_idx)
    left_cold = select_rows(cache.real_cold, left_idx)
    right_cold = select_rows(cache.real_cold, right_idx)
    left_dav = select_rows(cache.real_dav, left_idx)
    right_dav = select_rows(cache.real_dav, right_idx)
    left_eye = select_rows(cache.real_eye, left_idx)
    right_eye = select_rows(cache.real_eye, right_idx)
    left_features = select_rows(cache.real_features, left_idx)
    right_features = select_rows(cache.real_features, right_idx)
    mmd2, div_left, div_right = compute_feature_metrics(
        left_features,
        right_features,
        cache.feature_gamma_raw,
    )
    return {
        "pixel_hist_js": float(js_divergence(left_mass, right_mass)),
        "pixel_hist_w1": float(wasserstein1_from_hist(left_mass, right_mass, bin_edges=cache.bins)),
        "cold_cloud_fraction_abs_gap": float(abs(left_cold.mean() - right_cold.mean())),
        "dav_abs_gap_deg2": float(abs(left_dav.mean() - right_dav.mean())),
        "eye_contrast_proxy_abs_gap": float(abs(left_eye.mean() - right_eye.mean())),
        "psd_l2": mean_profile_mse_gap(left_psd, right_psd),
        "feature_mmd2": float(mmd2),
        "diversity_feature_space_abs_gap": float(abs(div_left - div_right)),
        "radial_profile_mae_k": float(
            np.mean(np.abs(left_radial.mean(axis=0) - right_radial.mean(axis=0)))
        ),
        "gen_exceedance_rate_total": 0.0,
    }


def aggregate_class_summaries(per_class: dict[int, dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for node in per_class.values() for key in node})
    out = {}
    for key in keys:
        values = [float(node[key]) for node in per_class.values() if key in node]
        if values:
            out[key] = float(np.mean(values))
    return out


def bootstrap_model_metric_samples(
    class_caches: dict[int, ClassMetricCache],
    *,
    reps: int,
    seed: int,
    progress: bool = False,
) -> dict[str, np.ndarray]:
    if int(reps) <= 0:
        return {}
    rng = np.random.default_rng(int(seed))
    samples_by_key: dict[str, list[float]] = {}
    for _ in progress_iter(
        range(int(reps)),
        enabled=progress,
        desc="Model metric bootstrap",
        unit="rep",
    ):
        per_class = {}
        for class_id, cache in sorted(class_caches.items()):
            real_n = int(cache.real_hist_counts.shape[0])
            gen_n = int(cache.gen_raw_hist_counts.shape[0])
            per_class[int(class_id)] = compute_class_scalar_summary(
                cache,
                real_idx=rng.integers(real_n, size=real_n),
                gen_idx=rng.integers(gen_n, size=gen_n),
            )
        for key, value in aggregate_class_summaries(per_class).items():
            samples_by_key.setdefault(key, []).append(float(value))
    return {
        key: np.asarray(values, dtype=np.float64)
        for key, values in sorted(samples_by_key.items())
    }


def evaluate_model_by_class(
    *,
    real_by_class: dict[int, np.ndarray],
    generated_by_class: dict[int, np.ndarray],
    cfg: TCMetricConfig,
    bootstrap_reps: int = 0,
    bootstrap_seed: int = 12345,
    include_bootstrap_samples: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    real_classes = set(real_by_class)
    generated_classes = set(generated_by_class)
    if real_classes != generated_classes:
        raise ValueError(
            f"Generated classes {sorted(generated_classes)} do not match "
            f"real classes {sorted(real_classes)}."
        )
    binner, dav_computer, cold_mask = make_metric_context(cfg)
    per_class: dict[int, dict[str, Any]] = {}
    class_summaries: dict[int, dict[str, float]] = {}
    class_caches: dict[int, ClassMetricCache] = {}
    for class_id in progress_iter(
        sorted(real_by_class),
        enabled=progress,
        desc="Evaluating classes",
        unit="class",
    ):
        if class_id not in generated_by_class:
            continue
        cache = build_class_metric_cache(
            real_k=real_by_class[class_id],
            gen_k_raw=generated_by_class[class_id],
            cfg=cfg,
            binner=binner,
            dav_computer=dav_computer,
            cold_mask=cold_mask,
        )
        class_caches[int(class_id)] = cache
        summary = compute_class_scalar_summary(cache)
        class_summaries[int(class_id)] = summary
        per_class[int(class_id)] = {
            "real_count": int(real_by_class[class_id].shape[0]),
            "generated_count": int(generated_by_class[class_id].shape[0]),
            "primary_raw": summary,
            "pixel_stats": {
                "real": cache.pixel_stats_real,
                "generated": cache.pixel_stats_gen_raw,
            },
            "gen_exceedance_rate": cache.gen_exceedance_rate,
        }
    aggregate_primary_raw = aggregate_class_summaries(class_summaries)
    report: dict[str, Any] = {
        "aggregation": {
            "aggregate_primary_raw": "unweighted_macro_average_over_valid_classes",
        },
        "aggregate_primary_raw": aggregate_primary_raw,
        "per_class": {str(k): v for k, v in sorted(per_class.items())},
        "valid_class_ids": [int(k) for k in sorted(class_summaries)],
    }
    if int(bootstrap_reps) > 0:
        samples = bootstrap_model_metric_samples(
            class_caches,
            reps=int(bootstrap_reps),
            seed=int(bootstrap_seed),
            progress=progress,
        )
        bootstrap_report: dict[str, Any] = {
            "enabled": True,
            "schema": "climategen_eval.model_metric_bootstrap.v1",
            "method": "within_class_two_sample_nonparametric_bootstrap",
            "uncertainty_scope": (
                "finite_sample_uncertainty_from_real_and_generated_evaluation_sets"
            ),
            "reps": int(bootstrap_reps),
            "seed": int(bootstrap_seed),
            "class_aggregation": "unweighted_macro_average_over_classes",
            "resampling": {
                "real": "sample available real images with replacement within class",
                "generated": "sample evaluated generated images with replacement within class",
            },
            "values": summarize_metric_samples(samples, point_values=aggregate_primary_raw),
        }
        if include_bootstrap_samples:
            bootstrap_report["samples"] = {
                key: values.astype(float).tolist()
                for key, values in sorted(samples.items())
            }
        report["aggregate_primary_bootstrap"] = bootstrap_report
    else:
        report["aggregate_primary_bootstrap"] = {
            "enabled": False,
            "reason": "bootstrap_reps <= 0",
        }
    return report
