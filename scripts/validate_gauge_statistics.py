#!/usr/bin/env python3
"""Monte Carlo validation and paper figure for GAUGE's empirical reference calibration."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gauge_climate.inference import reference_summary_uncertainty  # noqa: E402
from gauge_climate.metrics import (  # noqa: E402
    coverage_metric,
    density_metric,
    diversity_metric,
    functional_w1,
    mmd_metric,
)
from gauge_climate.reference import (  # noqa: E402
    ReferencePlan,
    build_reference_plan,
    evaluate_reference_plan,
    reference_position,
)
from gauge_climate.types import (  # noqa: E402
    EvaluationConfig,
    FunctionalSpec,
    MetricKind,
    MetricSpec,
)


SCHEMA = "gauge_climate.statistical_validation.v4"
SHIFT_EFFECTS_SD = (0.0, 0.15, 0.30, 0.45, 0.55)
ALTERNATIVE_LABELS = {
    "variance_narrow": "narrow variance",
    "variance_wide": "excess variance",
    "crossing_cdf": "crossing CDF",
    "tail_contamination": "tail contamination",
    "train_test_shift": "train/test shift",
}


@dataclass(frozen=True)
class Regime:
    id: str
    label: str
    n_blocks: int
    n_eval: int
    block_sizes: str


REGIMES = (
    Regime("balanced_80_n80", "80 balanced blocks, N=80", 80, 80, "balanced"),
    Regime("balanced_80", "80 balanced blocks", 80, 240, "balanced"),
    Regime("balanced_80_n1000", "80 balanced blocks, N=1,000", 80, 1_000, "balanced"),
    Regime("unequal_80", "80 unequal blocks", 80, 240, "severely_unequal"),
    Regime("c5_like_22", "22 blocks, N=1,000", 22, 1_000, "c5_like"),
    Regime("minimum_20", "20-block boundary", 20, 1_000, "c5_like"),
)


def _wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return center - half, center + half


def _rate(successes: int, total: int, *, name: str) -> dict[str, Any]:
    low, high = _wilson(successes, total)
    return {
        name: successes / total,
        "successes": successes,
        "trials": total,
        "wilson_95": [low, high],
    }


def _block_sizes(
    rng: np.random.Generator,
    n_blocks: int,
    model: str,
) -> np.ndarray:
    if model == "balanced":
        return np.full(n_blocks, 4, dtype=int)
    if model == "severely_unequal":
        return np.maximum(
            1,
            np.rint(np.exp(rng.uniform(0.0, np.log(60.0), n_blocks))).astype(int),
        )
    if model == "c5_like":
        # Approximately 3.9 frames per block, with a long but bounded tail:
        # the observed TC C5 test stratum contains 85 frames from 22 storms.
        return np.minimum(rng.geometric(p=0.26, size=n_blocks), 12).astype(int)
    raise ValueError(f"Unknown block-size model {model!r}.")


def _clustered_values(
    rng: np.random.Generator,
    sizes: np.ndarray,
    *,
    location: float = 0.0,
) -> np.ndarray:
    effects = rng.normal(loc=location, scale=0.3, size=sizes.size)
    return effects.repeat(sizes) + rng.normal(size=int(np.sum(sizes)))


def _reference_plan(
    blocks: np.ndarray,
    regime: Regime,
    *,
    resamples: int,
    seed: int,
) -> ReferencePlan:
    config = EvaluationConfig(
        n_eval=regime.n_eval,
        n_resamples=resamples,
        diagnostic_resamples=resamples,
        min_blocks_per_half=10,
        warn_blocks_per_half=10,
        seed=seed,
    )
    return build_reference_plan(
        real_indices=np.arange(blocks.size),
        real_blocks=blocks,
        n_generated=regime.n_eval,
        config=config,
        seed=seed,
    )


def _stack_indices(plan: ReferencePlan, attribute: str) -> np.ndarray:
    return np.stack(
        [np.asarray(getattr(draw, attribute), dtype=np.int64) for draw in plan.draws],
        axis=0,
    )


def _vectorized_w1(
    real: np.ndarray,
    generated: np.ndarray,
    plan: ReferencePlan,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate equal-size scalar W1 draws exactly, but in vectorized form."""

    pseudo_indices = _stack_indices(plan, "pseudo_indices")
    real_indices = _stack_indices(plan, "real_indices")
    generated_indices = _stack_indices(plan, "generated_indices")
    pseudo_sorted = np.sort(np.asarray(real)[pseudo_indices], axis=1)
    real_sorted = np.sort(np.asarray(real)[real_indices], axis=1)
    generated_sorted = np.sort(np.asarray(generated)[generated_indices], axis=1)
    reference = np.mean(np.abs(pseudo_sorted - real_sorted), axis=1)
    model = np.mean(np.abs(generated_sorted - real_sorted), axis=1)
    return reference, model


def _crosses(
    reference: np.ndarray,
    model: np.ndarray,
    reference_quantile: float = 0.95,
) -> bool:
    return bool(np.median(model) > np.quantile(reference, reference_quantile))


def _generated_alternatives(
    rng: np.random.Generator,
    n: int,
    marginal_sd: float,
) -> dict[str, np.ndarray]:
    narrow = rng.normal(scale=0.45 * marginal_sd, size=n)
    wide = rng.normal(scale=1.65 * marginal_sd, size=n)
    # A symmetric, nearly two-point mixture retains the null mean and variance
    # while producing strongly crossing CDFs.  This is deliberately a
    # shape-only alternative rather than a disguised location/scale change.
    mixture_location = 0.99 * marginal_sd
    mixture_noise = math.sqrt(max(marginal_sd**2 - mixture_location**2, 1e-12))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=n)
    crossing = signs * mixture_location + rng.normal(scale=mixture_noise, size=n)
    tail = rng.normal(scale=marginal_sd, size=n)
    tail[rng.choice(n, size=max(1, int(round(0.15 * n))), replace=False)] += 3.0 * marginal_sd
    return {
        "variance_narrow": narrow,
        "variance_wide": wide,
        "crossing_cdf": crossing,
        "tail_contamination": tail,
        # The generator matches a historical/train population centered at zero,
        # while the held-out population has shifted by +0.55 marginal SD.
        "train_test_shift": rng.normal(loc=-0.55 * marginal_sd, scale=marginal_sd, size=n),
    }


def _check_invariants(plan: ReferencePlan) -> bool:
    return all(
        not (set(draw.blocks_a) & set(draw.blocks_b))
        and len(draw.pseudo_indices) == plan.n_eval
        and len(draw.real_indices) == plan.n_eval
        and len(draw.generated_indices) == plan.n_eval
        for draw in plan.draws
    )


def _trial(
    trial: int,
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 104_729 * trial)
    marginal_sd = math.sqrt(1.0 + 0.3**2)
    result: dict[str, Any] = {
        "trial": trial,
        "regimes": {},
        "shape_same_population": {},
        "invariants": True,
        "engine_parity_max_abs": 0.0,
    }
    for regime_index, regime in enumerate(REGIMES):
        regime_rng = np.random.default_rng(rng.integers(0, np.iinfo(np.int64).max))
        sizes = _block_sizes(regime_rng, regime.n_blocks, regime.block_sizes)
        blocks = np.repeat(np.arange(regime.n_blocks), sizes)
        real = _clustered_values(regime_rng, sizes)
        plan = _reference_plan(
            blocks,
            regime,
            resamples=resamples,
            seed=seed + trial * 10_000 + regime_index,
        )
        result["invariants"] &= _check_invariants(plan)

        equal = regime_rng.normal(scale=marginal_sd, size=regime.n_eval)
        reference, model = _vectorized_w1(real, equal, plan)
        reference_q50 = float(np.quantile(reference, 0.50))
        reference_q95 = float(np.quantile(reference, 0.95))
        model_median = float(np.median(model))
        same_population_gap = reference_position(
            model_median,
            reference_q50,
            reference_q95,
        )
        node: dict[str, Any] = {
            "same_population": _crosses(reference, model),
            "same_population_summary": {
                "reference_q50": reference_q50,
                "reference_q95": reference_q95,
                "model_median": model_median,
                "reference_percentile": float(np.mean(reference <= model_median)),
                "reference_gap": same_population_gap,
            },
            "shift": {},
            "alternatives": {},
        }
        for effect in SHIFT_EFFECTS_SD:
            shifted = regime_rng.normal(
                loc=effect * marginal_sd,
                scale=marginal_sd,
                size=regime.n_eval,
            )
            _, shift_model = _vectorized_w1(real, shifted, plan)
            node["shift"][str(effect)] = _crosses(reference, shift_model)
        for name, generated in _generated_alternatives(
            regime_rng,
            regime.n_eval,
            marginal_sd,
        ).items():
            _, alternative_model = _vectorized_w1(real, generated, plan)
            node["alternatives"][name] = _crosses(reference, alternative_model)
        result["regimes"][regime.id] = node

        if regime.id == "balanced_80":
            heavy_real = regime_rng.normal(scale=0.3, size=regime.n_blocks).repeat(
                sizes
            ) + regime_rng.standard_t(df=3, size=blocks.size) / math.sqrt(3.0)
            heavy_generated = regime_rng.normal(
                scale=0.3, size=regime.n_eval
            ) + regime_rng.standard_t(df=3, size=regime.n_eval) / math.sqrt(3.0)
            multimodal_real = (
                regime_rng.normal(scale=0.3, size=regime.n_blocks).repeat(sizes)
                + regime_rng.choice((-1.0, 1.0), size=blocks.size)
                + regime_rng.normal(scale=0.15, size=blocks.size)
            )
            multimodal_generated = (
                regime_rng.normal(scale=0.3, size=regime.n_eval)
                + regime_rng.choice((-1.0, 1.0), size=regime.n_eval)
                + regime_rng.normal(scale=0.15, size=regime.n_eval)
            )
            for name, shape_real, shape_generated in (
                ("heavy_tailed", heavy_real, heavy_generated),
                ("multimodal", multimodal_real, multimodal_generated),
            ):
                shape_reference, shape_model = _vectorized_w1(
                    shape_real,
                    shape_generated,
                    plan,
                )
                result["shape_same_population"][name] = _crosses(
                    shape_reference,
                    shape_model,
                )

        if trial == 0:
            metric = functional_w1(
                FunctionalSpec("value", lambda values: values, "unit", "validation scalar")
            )
            engine = evaluate_reference_plan(
                metric,
                real_values=real,
                generated_values=equal,
                plan=plan,
            )
            result["engine_parity_max_abs"] = max(
                result["engine_parity_max_abs"],
                float(np.max(np.abs(reference - engine.reference_values))),
                float(np.max(np.abs(model - engine.model_values))),
            )

    return result


def _invalid_thin_path(*, resamples: int, seed: int) -> bool:
    regime = Regime("invalid_19", "19 blocks", 19, 100, "balanced")
    blocks = np.repeat(np.arange(regime.n_blocks), 4)
    try:
        _reference_plan(blocks, regime, resamples=resamples, seed=seed)
    except ValueError as exc:
        return "at least 20 eligible blocks" in str(exc)
    return False


def _metric_gap(
    metric: MetricSpec,
    *,
    real: np.ndarray,
    generated: np.ndarray,
    plan: ReferencePlan,
) -> tuple[float | None, bool]:
    evaluation = evaluate_reference_plan(
        metric,
        real_values=real,
        generated_values=generated,
        plan=plan,
    )
    q50 = float(np.quantile(evaluation.reference_values, 0.50))
    q95 = float(np.quantile(evaluation.reference_values, 0.95))
    value = float(np.median(evaluation.model_values))
    return reference_position(value, q50, q95), bool(value > q95)


def _metric_family_validation(*, seed: int, resamples: int) -> dict[str, Any]:
    """Validate non-scalar metric families on controlled synthetic changes."""

    rng = np.random.default_rng(seed)
    sizes = np.full(80, 4, dtype=int)
    blocks = np.repeat(np.arange(80), sizes)
    regime = Regime("metric_families", "metric families", 80, 240, "balanced")
    plan = _reference_plan(blocks, regime, resamples=resamples, seed=seed + 1)

    block_effects = rng.normal(scale=0.3, size=(80, 4))
    real_features = block_effects.repeat(4, axis=0) + rng.normal(size=(blocks.size, 4))
    marginal_sd = math.sqrt(1.0 + 0.3**2)
    same_features = rng.normal(scale=marginal_sd, size=(regime.n_eval, 4))
    centre = same_features.mean(axis=0, keepdims=True)
    direction = np.asarray([1.0, -1.0, 0.75, -0.75])

    def off_manifold(severity: float) -> np.ndarray:
        return same_features + severity * direction

    def mode_removed(retained_fraction: float) -> np.ndarray:
        threshold = np.quantile(same_features[:, 0], 1.0 - retained_fraction)
        retained = same_features[same_features[:, 0] >= threshold]
        return retained[rng.integers(0, len(retained), size=regime.n_eval)]

    def duplicated(unique_fraction: float) -> np.ndarray:
        count = max(1, int(round(regime.n_eval * unique_fraction)))
        retained = same_features[:count]
        return retained[np.arange(regime.n_eval) % count]

    def contracted(severity: float) -> np.ndarray:
        return centre + (1.0 - severity) * (same_features - centre)

    def expanded(severity: float) -> np.ndarray:
        return centre + (1.0 + severity) * (same_features - centre)

    feature_cases = {
        "mmd2_off_manifold": (
            mmd_metric(lambda values: values, bandwidth=2.0, max_samples=240),
            [(value, off_manifold(value)) for value in (0.0, 0.5, 1.0, 2.0, 4.0)],
        ),
        "density_off_manifold": (
            density_metric(lambda values: values, k=5),
            [(value, off_manifold(value)) for value in (0.0, 0.5, 1.0, 2.0, 4.0)],
        ),
        "coverage_mode_removal": (
            coverage_metric(lambda values: values, k=5),
            [(1.0 - value, mode_removed(value)) for value in (1.0, 0.75, 0.50, 0.25, 0.10)],
        ),
        "diversity_duplication": (
            diversity_metric(lambda values: values),
            [
                (1.0 - value, duplicated(value))
                for value in (1.0, 0.50, 0.10, 0.02, 1.0 / regime.n_eval)
            ],
        ),
        "diversity_contraction": (
            diversity_metric(lambda values: values),
            [(value, contracted(value)) for value in (0.0, 0.25, 0.50, 0.75, 0.95)],
        ),
        "diversity_expansion": (
            diversity_metric(lambda values: values),
            [(value, expanded(value)) for value in (0.0, 0.25, 0.50, 1.0, 2.0)],
        ),
    }
    results: dict[str, Any] = {}
    for name, (metric, severity_data) in feature_cases.items():
        same_gap, same_crosses = _metric_gap(
            metric,
            real=real_features,
            generated=same_features,
            plan=plan,
        )
        curve = []
        for severity, changed in severity_data:
            changed_gap, changed_crosses = _metric_gap(
                metric,
                real=real_features,
                generated=changed,
                plan=plan,
            )
            curve.append(
                {
                    "severity": float(severity),
                    "gap": changed_gap,
                    "crosses": changed_crosses,
                }
            )
        valid_gaps = [float(row["gap"]) for row in curve if row["gap"] is not None]
        rank_response = float(
            spearmanr(
                [row["severity"] for row in curve],
                valid_gaps,
            ).statistic
        )
        results[name] = {
            "same_population_gap": same_gap,
            "same_population_crosses": same_crosses,
            "severity_curve": curve,
            "severity_rank_correlation": rank_response,
            "monotonic_response": rank_response >= 0.80,
            "changed_gap": curve[-1]["gap"],
            "changed_crosses": curve[-1]["crosses"],
        }

    # Curves are derived from synthetic 2-D fields rather than generated directly.
    # The radial-profile control widens a Gaussian structure; the PSD control
    # progressively removes small-scale power.
    field_size = 32
    axis = np.linspace(-1.0, 1.0, field_size)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    radius = np.sqrt(xx**2 + yy**2)
    radial_edges = np.linspace(0.0, 1.0, 25)
    radial_masks = tuple(
        (radius >= radial_edges[index]) & (radius < radial_edges[index + 1])
        for index in range(len(radial_edges) - 1)
        if np.any((radius >= radial_edges[index]) & (radius < radial_edges[index + 1]))
    )

    def radial_curves(fields: np.ndarray) -> np.ndarray:
        return np.column_stack([np.mean(fields[:, mask], axis=1) for mask in radial_masks])

    block_scales = np.clip(rng.normal(0.28, 0.02, size=80), 0.20, 0.36)
    real_fields = np.stack(
        [
            np.exp(-(radius**2) / (2.0 * block_scales[block] ** 2))
            + rng.normal(scale=0.03, size=radius.shape)
            for block in blocks
        ]
    )
    same_scales = np.clip(
        rng.normal(0.28, math.sqrt(0.02**2 + 0.005**2), size=regime.n_eval),
        0.20,
        0.36,
    )
    same_fields = np.stack(
        [
            np.exp(-(radius**2) / (2.0 * scale**2)) + rng.normal(scale=0.03, size=radius.shape)
            for scale in same_scales
        ]
    )
    real_curves = radial_curves(real_fields)
    same_curves = radial_curves(same_fields)
    curve_metric = MetricSpec(
        id="curve_mae_validation",
        transform=lambda values: values,
        estimator=lambda left, right: float(
            np.mean(np.abs(np.mean(left, axis=0) - np.mean(right, axis=0)))
        ),
        kind=MetricKind.CURVE,
        units="curve units",
        description="Synthetic curve-mean absolute discrepancy.",
    )
    same_gap, same_crosses = _metric_gap(
        curve_metric,
        real=real_curves,
        generated=same_curves,
        plan=plan,
    )
    curve_severities = (0.0, 0.03, 0.06, 0.12, 0.20)
    curve_response = []
    for severity in curve_severities:
        widened = np.stack(
            [
                np.exp(-(radius**2) / (2.0 * (scale + severity) ** 2)) + noise
                for scale, noise in zip(
                    same_scales,
                    same_fields
                    - np.stack([np.exp(-(radius**2) / (2.0 * value**2)) for value in same_scales]),
                    strict=True,
                )
            ]
        )
        changed_gap, changed_crosses = _metric_gap(
            curve_metric,
            real=real_curves,
            generated=radial_curves(widened),
            plan=plan,
        )
        curve_response.append(
            {
                "severity_scale_increment": severity,
                "gap": changed_gap,
                "crosses": changed_crosses,
            }
        )
    curve_rank = float(
        spearmanr(
            curve_severities,
            [row["gap"] for row in curve_response],
        ).statistic
    )
    results["curve_mae"] = {
        "same_population_gap": same_gap,
        "same_population_crosses": same_crosses,
        "severity_curve": curve_response,
        "severity_rank_correlation": curve_rank,
        "monotonic_response": curve_rank >= 0.80,
        "changed_gap": curve_response[-1]["gap"],
        "changed_crosses": curve_response[-1]["crosses"],
        "source": "radial profiles derived from synthetic 2-D Gaussian fields",
    }

    frequency = np.fft.fftshift(np.fft.fftfreq(field_size))
    fy, fx = np.meshgrid(frequency, frequency, indexing="ij")
    radial_frequency = np.sqrt(fx**2 + fy**2)
    frequency_edges = np.linspace(0.0, float(radial_frequency.max()), 17)
    frequency_masks = tuple(
        (radial_frequency >= frequency_edges[index])
        & (radial_frequency < frequency_edges[index + 1])
        for index in range(len(frequency_edges) - 1)
        if np.any(
            (radial_frequency >= frequency_edges[index])
            & (radial_frequency < frequency_edges[index + 1])
        )
    )

    def log_psd_curves(fields: np.ndarray) -> np.ndarray:
        centered = fields - fields.mean(axis=(-2, -1), keepdims=True)
        power = np.abs(np.fft.fftshift(np.fft.fft2(centered), axes=(-2, -1))) ** 2
        return np.column_stack(
            [
                np.log10(np.maximum(np.mean(power[:, mask], axis=1), 1e-12))
                for mask in frequency_masks
            ]
        )

    real_texture = np.stack(
        [
            gaussian_filter(
                rng.normal(size=(field_size, field_size)),
                sigma=float(np.clip(rng.normal(1.0, 0.08), 0.75, 1.25)),
                mode="wrap",
            )
            for _ in range(blocks.size)
        ]
    )
    same_texture = np.stack(
        [
            gaussian_filter(
                rng.normal(size=(field_size, field_size)),
                sigma=float(np.clip(rng.normal(1.0, 0.08), 0.75, 1.25)),
                mode="wrap",
            )
            for _ in range(regime.n_eval)
        ]
    )
    real_log_psd = log_psd_curves(real_texture)
    same_log_psd = log_psd_curves(same_texture)
    psd_metric = MetricSpec(
        id="log_psd_mse_validation",
        transform=lambda values: values,
        estimator=lambda left, right: float(
            np.mean((np.mean(left, axis=0) - np.mean(right, axis=0)) ** 2)
        ),
        kind=MetricKind.CURVE,
        units="dex²",
        description="Synthetic radial log-power discrepancy.",
    )
    same_gap, same_crosses = _metric_gap(
        psd_metric,
        real=real_log_psd,
        generated=same_log_psd,
        plan=plan,
    )
    blur_severities = (0.0, 0.5, 1.0, 2.0, 4.0)
    psd_response = []
    for severity in blur_severities:
        blurred = (
            same_texture
            if severity == 0.0
            else gaussian_filter(
                same_texture,
                sigma=(0.0, severity, severity),
                mode="wrap",
            )
        )
        changed_gap, changed_crosses = _metric_gap(
            psd_metric,
            real=real_log_psd,
            generated=log_psd_curves(blurred),
            plan=plan,
        )
        psd_response.append(
            {
                "blur_sigma_pixels": severity,
                "gap": changed_gap,
                "crosses": changed_crosses,
            }
        )
    psd_rank = float(
        spearmanr(
            blur_severities,
            [row["gap"] for row in psd_response],
        ).statistic
    )
    results["log_psd"] = {
        "same_population_gap": same_gap,
        "same_population_crosses": same_crosses,
        "severity_curve": psd_response,
        "severity_rank_correlation": psd_rank,
        "monotonic_response": psd_rank >= 0.80,
        "changed_gap": psd_response[-1]["gap"],
        "changed_crosses": psd_response[-1]["crosses"],
        "source": "radial log-power curves derived from synthetic 2-D texture fields",
    }

    scalar_metric = functional_w1(
        FunctionalSpec(
            "shape_validation",
            lambda values: np.asarray(values).reshape(-1),
            "unit",
            "Non-Gaussian validation scalar.",
        )
    )
    for name, real_noise, generated_noise in (
        (
            "scalar_heavy_tail",
            rng.standard_t(df=3, size=blocks.size) / math.sqrt(3.0),
            rng.standard_t(df=3, size=regime.n_eval) / math.sqrt(3.0),
        ),
        (
            "scalar_multimodal",
            rng.choice([-1.0, 1.0], size=blocks.size) + rng.normal(scale=0.15, size=blocks.size),
            rng.choice([-1.0, 1.0], size=regime.n_eval)
            + rng.normal(scale=0.15, size=regime.n_eval),
        ),
    ):
        block_scalar = rng.normal(scale=0.3, size=80).repeat(4)
        real_scalar = block_scalar + real_noise
        same_scalar = rng.normal(scale=0.3, size=regime.n_eval) + generated_noise
        changed_scalar = same_scalar + 1.5
        same_gap, same_crosses = _metric_gap(
            scalar_metric,
            real=real_scalar,
            generated=same_scalar,
            plan=plan,
        )
        changed_gap, changed_crosses = _metric_gap(
            scalar_metric,
            real=real_scalar,
            generated=changed_scalar,
            plan=plan,
        )
        results[name] = {
            "same_population_gap": same_gap,
            "same_population_crosses": same_crosses,
            "changed_gap": changed_gap,
            "changed_crosses": changed_crosses,
        }
    return {
        "results": results,
        "all_same_population_within_reference": all(
            not item["same_population_crosses"] for item in results.values()
        ),
        "all_controls_cross_reference": all(item["changed_crosses"] for item in results.values()),
        "all_severity_curves_monotonic": all(
            item.get("monotonic_response", True) for item in results.values()
        ),
        "mmd2_non_negative": all(
            value >= 0.0
            for value in (
                mmd_metric(
                    lambda values: values,
                    bandwidth=2.0,
                    max_samples=240,
                ).estimator(same_features, real_features[:240]),
                mmd_metric(
                    lambda values: values,
                    bandwidth=2.0,
                    max_samples=240,
                ).estimator(off_manifold(4.0), real_features[:240]),
            )
        ),
    }


def _reference_stability_validation(*, seed: int, resamples: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sizes = np.full(80, 8, dtype=int)
    blocks = np.repeat(np.arange(80), sizes)
    real = rng.normal(scale=1.6, size=80).repeat(sizes) + rng.normal(scale=0.35, size=blocks.size)
    marginal_sd = float(np.std(real))
    generated = rng.normal(scale=marginal_sd, size=240)
    regime = Regime("stability", "stability", 80, 240, "balanced")
    plan = _reference_plan(blocks, regime, resamples=resamples, seed=seed + 1)
    reference, model = _vectorized_w1(real, generated, plan)
    q50 = float(np.quantile(reference, 0.50))
    q95 = float(np.quantile(reference, 0.95))
    gap = reference_position(float(np.median(model)), q50, q95)
    intervals = reference_summary_uncertainty(
        reference_values=reference,
        model_values=model,
        reference_quantile=0.95,
        interval=0.95,
        n_resamples=1000,
        seed=seed + 2,
    )

    delete_group_gaps = []
    for group in range(8):
        retained = ~np.isin(blocks, np.arange(group * 10, (group + 1) * 10))
        local_blocks = blocks[retained]
        local_real = real[retained]
        local_plan = _reference_plan(
            local_blocks,
            regime,
            resamples=resamples,
            seed=seed + 100 + group,
        )
        local_reference, local_model = _vectorized_w1(
            local_real,
            generated,
            local_plan,
        )
        local_q50 = float(np.quantile(local_reference, 0.50))
        local_q95 = float(np.quantile(local_reference, 0.95))
        local_gap = reference_position(
            float(np.median(local_model)),
            local_q50,
            local_q95,
        )
        if local_gap is not None:
            delete_group_gaps.append(local_gap)

    fake_blocks = np.arange(blocks.size)
    iid_plan = _reference_plan(
        fake_blocks,
        regime,
        resamples=resamples,
        seed=seed + 500,
    )
    iid_reference, _ = _vectorized_w1(real, generated, iid_plan)
    correct_width = q95 - q50
    iid_width = float(np.quantile(iid_reference, 0.95) - np.quantile(iid_reference, 0.50))
    return {
        "point": {
            "q50": q50,
            "q95": q95,
            "model_median": float(np.median(model)),
            "gap": gap,
        },
        "conditional_mc_intervals": intervals,
        "summary_points_inside_mc_intervals": bool(
            intervals["reference_q50"][0] <= q50 <= intervals["reference_q50"][1]
            and intervals["reference_upper"][0] <= q95 <= intervals["reference_upper"][1]
            and gap is not None
            and intervals["reference_gap"][0] <= gap <= intervals["reference_gap"][1]
        ),
        "delete_group_gap_range": [
            float(min(delete_group_gaps)),
            float(max(delete_group_gaps)),
        ],
        "delete_group_gaps": [float(value) for value in delete_group_gaps],
        "delete_group_all_finite": len(delete_group_gaps) == 8,
        "correct_block_reference_width": correct_width,
        "frame_iid_reference_width": iid_width,
        "block_misspecification_changes_reference": bool(
            not np.isclose(correct_width, iid_width, rtol=0.10)
        ),
        "degenerate_gap_fails_closed": reference_position(0.0, 0.0, 0.0) is None,
    }


def _independent_seed_interval_validation(
    *,
    seed: int,
    resamples: int,
    repetitions: int = 24,
) -> dict[str, Any]:
    """Check conditional Monte Carlo intervals against a high-repetition target."""

    rng = np.random.default_rng(seed)
    sizes = np.full(80, 8, dtype=int)
    blocks = np.repeat(np.arange(80), sizes)
    real = rng.normal(scale=0.55, size=80).repeat(sizes) + rng.normal(
        scale=0.85,
        size=blocks.size,
    )
    generated = rng.normal(
        loc=0.12,
        scale=float(np.std(real)),
        size=240,
    )
    regime = Regime("mc_seed_stability", "MC seed stability", 80, 240, "balanced")

    target_draws = max(5_000, 20 * resamples)
    target_plan = _reference_plan(
        blocks,
        regime,
        resamples=target_draws,
        seed=seed + 1,
    )
    target_reference, target_model = _vectorized_w1(real, generated, target_plan)
    target_q50 = float(np.quantile(target_reference, 0.50))
    target_q95 = float(np.quantile(target_reference, 0.95))
    target_model_median = float(np.median(target_model))
    target_gap = reference_position(
        target_model_median,
        target_q50,
        target_q95,
    )
    if target_gap is None:  # pragma: no cover - guarded by continuous simulation
        raise RuntimeError("Unexpectedly degenerate high-repetition reference target.")
    targets = {
        "reference_q50": target_q50,
        "reference_q95": target_q95,
        "model_median": target_model_median,
        "reference_gap": float(target_gap),
    }

    coverage_counts = {name: 0 for name in targets}
    replicate_summaries = []
    for repetition in range(repetitions):
        plan = _reference_plan(
            blocks,
            regime,
            resamples=resamples,
            seed=seed + 10_000 + repetition,
        )
        reference, model = _vectorized_w1(real, generated, plan)
        intervals = reference_summary_uncertainty(
            reference_values=reference,
            model_values=model,
            reference_quantile=0.95,
            interval=0.95,
            n_resamples=1_000,
            seed=seed + 20_000 + repetition,
        )
        interval_map = {
            "reference_q50": intervals["reference_q50"],
            "reference_q95": intervals["reference_q95"],
            "model_median": intervals["model_median"],
            "reference_gap": intervals["reference_gap"],
        }
        summary = {
            "seed": seed + 10_000 + repetition,
            "intervals": interval_map,
        }
        replicate_summaries.append(summary)
        for name, target in targets.items():
            low, high = interval_map[name]
            if low is not None and high is not None and low <= target <= high:
                coverage_counts[name] += 1

    coverage = {
        name: {
            **_rate(count, repetitions, name="target_coverage"),
            "target": targets[name],
        }
        for name, count in coverage_counts.items()
    }
    return {
        "conditional_target": targets,
        "target_resamples": target_draws,
        "independent_seed_repetitions": repetitions,
        "resamples_per_repetition": resamples,
        "coverage": coverage,
        "replicates": replicate_summaries,
        "all_target_coverages_at_least_0.80": all(
            node["target_coverage"] >= 0.80 for node in coverage.values()
        ),
    }


def _aggregate(
    trials: list[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
    jobs: int,
) -> dict[str, Any]:
    total = len(trials)
    regimes: dict[str, Any] = {}
    for regime in REGIMES:
        nodes = [trial["regimes"][regime.id] for trial in trials]
        same_population_count = sum(bool(node["same_population"]) for node in nodes)
        same_population_gaps = np.asarray(
            [
                float(node["same_population_summary"]["reference_gap"])
                for node in nodes
                if node["same_population_summary"]["reference_gap"] is not None
            ],
            dtype=float,
        )
        same_population_percentiles = np.asarray(
            [
                float(node["same_population_summary"]["reference_percentile"])
                for node in nodes
            ],
            dtype=float,
        )
        if same_population_gaps.size != total:
            raise RuntimeError(
                f"Degenerate same-population reference gap in regime {regime.id!r}."
            )
        shifts = []
        for effect in SHIFT_EFFECTS_SD:
            count = sum(bool(node["shift"][str(effect)]) for node in nodes)
            shifts.append(
                {
                    "effect_sd": effect,
                    **_rate(count, total, name="boundary_crossing_probability"),
                }
            )
        alternatives = {}
        for name in ALTERNATIVE_LABELS:
            count = sum(bool(node["alternatives"][name]) for node in nodes)
            alternatives[name] = _rate(
                count,
                total,
                name="boundary_crossing_probability",
            )
        regimes[regime.id] = {
            **asdict(regime),
            "same_population": _rate(
                same_population_count,
                total,
                name="boundary_crossing_rate",
            ),
            "same_population_gap_distribution": {
                "q05": float(np.quantile(same_population_gaps, 0.05)),
                "q50": float(np.quantile(same_population_gaps, 0.50)),
                "q95": float(np.quantile(same_population_gaps, 0.95)),
                "minimum": float(np.min(same_population_gaps)),
                "maximum": float(np.max(same_population_gaps)),
            },
            "same_population_reference_percentile_distribution": {
                "q05": float(np.quantile(same_population_percentiles, 0.05)),
                "q50": float(np.quantile(same_population_percentiles, 0.50)),
                "q95": float(np.quantile(same_population_percentiles, 0.95)),
            },
            "location_shift": shifts,
            "alternatives": alternatives,
        }

    same_population_gates = {
        regime.id: (
            regimes[regime.id]["same_population"]["boundary_crossing_rate"] <= 0.05
            and regimes[regime.id]["same_population"]["wilson_95"][1] <= 0.05
        )
        for regime in REGIMES
    }
    same_population_gap_gates = {
        regime.id: regimes[regime.id]["same_population_gap_distribution"]["q95"] <= 1.0
        for regime in REGIMES
    }
    shape_same_population = {
        name: _rate(
            sum(bool(trial["shape_same_population"][name]) for trial in trials),
            total,
            name="boundary_crossing_rate",
        )
        for name in ("heavy_tailed", "multimodal")
    }
    shape_same_population_gates = {
        name: (node["boundary_crossing_rate"] <= 0.05 and node["wilson_95"][1] <= 0.05)
        for name, node in shape_same_population.items()
    }
    shift_response_gates = {
        regime.id: all(
            right["boundary_crossing_probability"] >= left["boundary_crossing_probability"]
            for left, right in zip(
                regimes[regime.id]["location_shift"],
                regimes[regime.id]["location_shift"][1:],
            )
        )
        for regime in REGIMES
    }
    standard_shift = next(
        item
        for item in regimes["balanced_80"]["location_shift"]
        if math.isclose(item["effect_sd"], 0.55)
    )
    thin_standard_shifts = [
        next(
            item
            for item in regimes[regime_id]["location_shift"]
            if math.isclose(item["effect_sd"], 0.55)
        )
        for regime_id in ("c5_like_22", "minimum_20")
    ]
    standard_alternative_gates = {
        name: node["wilson_95"][0] >= 0.80
        for name, node in regimes["balanced_80"]["alternatives"].items()
    }
    metric_families = _metric_family_validation(
        seed=seed + 70_000_000,
        resamples=max(resamples, 200),
    )
    stability = _reference_stability_validation(
        seed=seed + 80_000_000,
        resamples=max(resamples, 200),
    )
    independent_seed_intervals = _independent_seed_interval_validation(
        seed=seed + 85_000_000,
        resamples=max(resamples, 200),
    )
    gates = {
        "reference_plan_invariants": all(bool(trial["invariants"]) for trial in trials),
        "vectorized_engine_parity": max(float(trial["engine_parity_max_abs"]) for trial in trials)
        <= 1e-12,
        "same_population_crossing_upper_at_most_reference_tail": all(
            same_population_gates.values()
        ),
        "same_population_gap_q95_within_reference_range": all(
            same_population_gap_gates.values()
        ),
        "heavy_tail_and_multimodal_crossing_upper_at_most_reference_tail": all(
            shape_same_population_gates.values()
        ),
        "standard_shift_0.55_crossing_probability_lower_at_least_0.80": (
            standard_shift["wilson_95"][0] >= 0.80
        ),
        "thin_strata_have_lower_standard_shift_crossing_probability": all(
            item["boundary_crossing_probability"] < standard_shift["boundary_crossing_probability"]
            for item in thin_standard_shifts
        ),
        "larger_location_shifts_cross_at_least_as_often": all(shift_response_gates.values()),
        "standard_alternative_crossing_probability_lower_at_least_0.80": all(
            standard_alternative_gates.values()
        ),
        "metric_family_same_population_within_reference": metric_families[
            "all_same_population_within_reference"
        ],
        "metric_family_controls_cross_reference": metric_families["all_controls_cross_reference"],
        "metric_family_severity_curves_monotonic": metric_families["all_severity_curves_monotonic"],
        "mmd2_non_negative": metric_families["mmd2_non_negative"],
        "summary_mc_intervals_contain_observed_summaries": stability[
            "summary_points_inside_mc_intervals"
        ],
        "summary_mc_intervals_cover_high_repetition_targets_across_seeds": (
            independent_seed_intervals["all_target_coverages_at_least_0.80"]
        ),
        "delete_group_archive_sensitivity_is_finite": stability["delete_group_all_finite"],
        "incorrect_frame_iid_blocks_change_reference": stability[
            "block_misspecification_changes_reference"
        ],
        "degenerate_reference_gap_fails_closed": stability["degenerate_gap_fails_closed"],
        "fewer_than_twenty_blocks_rejected": _invalid_thin_path(
            resamples=resamples,
            seed=seed + 90_000_000,
        ),
    }
    return {
        "schema": SCHEMA,
        "status": "PASSED" if all(gates.values()) else "FAILED",
        "configuration": {
            "seed": seed,
            "trials": total,
            "resamples": resamples,
            "jobs": jobs,
            "reference_quantile": 0.95,
            "minimum_blocks_per_half": 10,
            "shift_effect_units": "marginal standard deviations",
        },
        "regimes": regimes,
        "shape_same_population": shape_same_population,
        "metric_family_validation": metric_families,
        "reference_stability": stability,
        "independent_seed_interval_validation": independent_seed_intervals,
        "failure_paths": {
            "nineteen_blocks": {
                "expected": "fail closed before inference",
                "passed": gates["fewer_than_twenty_blocks_rejected"],
            }
        },
        "engine_parity_max_abs": max(float(trial["engine_parity_max_abs"]) for trial in trials),
        "gates": gates,
    }


def _validation_figure(result: dict[str, Any], path: Path) -> tuple[Path, Path]:
    regimes = result["regimes"]
    display_labels = {
        "balanced_80_n80": "80 events, N=80",
        "balanced_80": "80 events, N=240",
        "balanced_80_n1000": "80 events, N=1,000",
        "unequal_80": "80 unequal events, N=240",
        "c5_like_22": "22 events, N=1,000",
        "minimum_20": "20 events, N=1,000",
    }
    colors = ("#246a73", "#d2764c", "#a33b20")
    markers = ("o", "s", "D")
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))
    fig.subplots_adjust(
        left=0.145,
        right=0.98,
        top=0.95,
        bottom=0.10,
        wspace=0.42,
        hspace=0.42,
    )

    # Panel a: all six designs, expressed directly on the GAUGE reference scale.
    ax = axes[0, 0]
    labels = [display_labels[regime.id] for regime in REGIMES]
    medians = [
        regimes[regime.id]["same_population_gap_distribution"]["q50"]
        for regime in REGIMES
    ]
    lows = [
        regimes[regime.id]["same_population_gap_distribution"]["q05"]
        for regime in REGIMES
    ]
    highs = [
        regimes[regime.id]["same_population_gap_distribution"]["q95"]
        for regime in REGIMES
    ]
    y = np.arange(len(REGIMES))
    ax.axvspan(0.0, 1.0, color="#eaf3f2", zorder=0)
    ax.axvline(0.0, color="#638b8f", linestyle=":", linewidth=1.0)
    ax.axvline(1.0, color="#a33b20", linestyle="--", linewidth=1.2)
    design_colors = ("#246a73", "#3f7f83", "#638b8f", "#8da5a5", "#d2764c", "#a33b20")
    design_markers = ("o", "s", "D", "^", "v", "P")
    for index, (color, marker) in enumerate(
        zip(design_colors, design_markers, strict=True)
    ):
        ax.errorbar(
            medians[index],
            y[index],
            xerr=np.asarray(
                [
                    [medians[index] - lows[index]],
                    [highs[index] - medians[index]],
                ]
            ),
            fmt=marker,
            color=color,
            ecolor=color,
            capsize=3,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    span = max(highs) - min(lows)
    padding = max(0.10, 0.08 * span)
    ax.set_xlim(min(lows) - padding, max(1.0, max(highs)) + padding)
    ax.set_xlabel("Gap for a matching generator\n(median and 5–95% across datasets)")
    ax.set_title("(a) Matching generator", loc="left", fontweight="semibold")
    ax.legend(
        handles=(
            Line2D(
                [0],
                [0],
                marker="o",
                color="#246a73",
                linewidth=1.4,
                label="Median and 5–95% range",
            ),
            Patch(facecolor="#eaf3f2", edgecolor="none", label="Reference interval"),
            Line2D(
                [0],
                [0],
                color="#a33b20",
                linestyle="--",
                linewidth=1.2,
                label="Upper marker (g=1)",
            ),
        ),
        frameon=False,
        fontsize=7.6,
        loc="lower right",
    )

    # Panel b: use equal-N designs so that the effect of independent-event
    # support is visible without mixing it with field-sample size.
    ax = axes[0, 1]
    shift_regimes = (
        next(regime for regime in REGIMES if regime.id == "balanced_80_n1000"),
        next(regime for regime in REGIMES if regime.id == "c5_like_22"),
        next(regime for regime in REGIMES if regime.id == "minimum_20"),
    )
    for regime, color, marker in zip(
        shift_regimes,
        colors,
        markers,
        strict=True,
    ):
        rows = regimes[regime.id]["location_shift"]
        x = [row["effect_sd"] for row in rows]
        probability = [row["boundary_crossing_probability"] for row in rows]
        ax.plot(
            x,
            probability,
            marker=marker,
            linestyle="-",
            color=color,
            label=f"{regime.n_blocks} events",
        )
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("Known location shift (marginal SD)")
    ax.set_ylabel("Fraction above reference range")
    ax.set_title("(b) Known mean shift", loc="left", fontweight="semibold")
    ax.legend(
        frameon=False,
        fontsize=8.0,
        title="All N=1,000",
        title_fontsize=8.0,
        loc="upper left",
    )

    # Panel c: a conventional paired dot plot replaces the dense heat map.
    # Equal N on both series isolates the loss of resolution from fewer events.
    ax = axes[1, 0]
    alternative_names = (
        "variance_narrow",
        "variance_wide",
        "crossing_cdf",
        "tail_contamination",
        "train_test_shift",
    )
    alternative_labels = {
        "variance_narrow": "SD × 0.45",
        "variance_wide": "SD × 1.65",
        "crossing_cdf": "Shape change (matched mean, SD)",
        "tail_contamination": "15% shifted by +3 SD",
        "train_test_shift": "Train–test shift: 0.55 SD",
    }
    y = np.arange(len(alternative_names))
    full_values = np.asarray(
        [
            regimes["balanced_80_n1000"]["alternatives"][name][
                "boundary_crossing_probability"
            ]
            for name in alternative_names
        ]
    )
    thin_values = np.asarray(
        [
            regimes["c5_like_22"]["alternatives"][name][
                "boundary_crossing_probability"
            ]
            for name in alternative_names
        ]
    )
    for row, (full, thin) in enumerate(zip(full_values, thin_values, strict=True)):
        ax.plot((thin, full), (row, row), color="#c8d1d3", linewidth=1.6, zorder=1)
    ax.scatter(
        full_values,
        y,
        marker="o",
        s=44,
        color="#246a73",
        label="80 events",
        zorder=2,
    )
    ax.scatter(
        thin_values,
        y,
        marker="D",
        s=38,
        color="#d2764c",
        label="22 events",
        zorder=2,
    )
    ax.set_yticks(y, [alternative_labels[name] for name in alternative_names])
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("Fraction above reference range")
    ax.set_title("(c) Other known changes", loc="left", fontweight="semibold")
    ax.legend(
        frameon=False,
        fontsize=8.0,
        title="Both N=1,000",
        title_fontsize=8.0,
        loc="lower left",
    )

    # Panel d: show directly how treating correlated fields as independent
    # contracts the empirical reference width.
    ax = axes[1, 1]
    stability = result["reference_stability"]
    widths = (
        float(stability["correct_block_reference_width"]),
        float(stability["frame_iid_reference_width"]),
    )
    block_labels = ("Correct event groups", "Each field treated\nas independent")
    bars = ax.barh(
        np.arange(2),
        widths,
        color=("#246a73", "#d2764c"),
        height=0.58,
    )
    ax.set_yticks(np.arange(2), block_labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, max(widths) * 1.28)
    ax.set_xlabel(r"Reference width ($q_{0.95}^{RR}-q_{0.50}^{RR}$)")
    ax.set_title("(d) Ignoring event dependence", loc="left", fontweight="semibold")
    for bar, value in zip(bars, widths, strict=True):
        ax.text(
            value + max(widths) * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=9.0,
        )

    for ax in axes.flat:
        ax.grid(color="#e1e7e9", linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.title.set_fontsize(12.5)
        ax.xaxis.label.set_size(10.5)
        ax.yaxis.label.set_size(10.5)
        ax.tick_params(axis="both", labelsize=9.0)
    pdf = Path(f"{path}.pdf")
    png = Path(f"{path}.png")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def _validation_appendix_figure(
    result: dict[str, Any],
    path: Path,
) -> tuple[Path, Path]:
    """Render conditional Monte Carlo and finite-archive stability diagnostics."""

    stability = result["reference_stability"]
    point = stability["point"]
    intervals = stability["conditional_mc_intervals"]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.6))
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.97,
        bottom=0.10,
        wspace=0.30,
        hspace=0.38,
    )

    ax = axes[0, 0]
    interval_rows = (
        ("Reference median", "q50", "reference_q50"),
        ("Reference 95th percentile", "q95", "reference_upper"),
        ("Model median", "model_median", "model_median"),
    )
    for row, (label, point_key, interval_key) in enumerate(interval_rows):
        estimate = float(point[point_key])
        low, high = map(float, intervals[interval_key])
        ax.errorbar(
            estimate,
            row,
            xerr=np.asarray([[estimate - low], [high - estimate]]),
            fmt="o",
            color="#246a73",
            ecolor="#638b8f",
            capsize=4,
        )
    ax.set_yticks(np.arange(len(interval_rows)), [row[0] for row in interval_rows])
    ax.invert_yaxis()
    ax.set_xlabel("Native discrepancy")
    ax.set_title("a  Conditional Monte Carlo intervals", loc="left", fontweight="semibold")

    ax = axes[0, 1]
    delete_values = np.asarray(stability["delete_group_gaps"], dtype=float)
    ax.scatter(
        delete_values,
        np.arange(len(delete_values)),
        marker="D",
        s=34,
        color="#d2764c",
        label="delete-group estimates",
    )
    gap = float(point["gap"])
    gap_low, gap_high = map(float, intervals["reference_gap"])
    ax.errorbar(
        gap,
        -1,
        xerr=np.asarray([[gap - gap_low], [gap_high - gap]]),
        fmt="o",
        color="#246a73",
        capsize=4,
        label="full archive + paired bootstrap",
    )
    ax.axvline(1.0, color="#a33b20", linestyle="--", linewidth=1.1)
    ax.set_yticks(
        [-1, *range(len(delete_values))],
        ["Full", *[f"Delete {i + 1}" for i in range(len(delete_values))]],
    )
    ax.set_xlabel("Real-reference gap")
    ax.set_title("b  Finite-archive sensitivity", loc="left", fontweight="semibold")
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1, 0]
    widths = (
        float(stability["correct_block_reference_width"]),
        float(stability["frame_iid_reference_width"]),
    )
    ax.bar(
        np.arange(2),
        widths,
        color=("#246a73", "#d2764c"),
        width=0.62,
    )
    ax.set_xticks(
        np.arange(2),
        ("Correct event blocks", "Incorrect frame-IID"),
        rotation=10,
        ha="right",
    )
    ax.set_ylabel(r"$q_{0.95}^{RR}-q_{0.50}^{RR}$")
    ax.set_title("c  Consequence of block misspecification", loc="left", fontweight="semibold")

    ax = axes[1, 1]
    families = result["metric_family_validation"]["results"]
    colors = plt.get_cmap("tab10")
    severity_labels = {
        "mmd2_off_manifold": "MMD²: off-manifold",
        "density_off_manifold": "Density: off-manifold",
        "coverage_mode_removal": "Coverage: mode removal",
        "diversity_duplication": "Diversity: duplication",
        "diversity_contraction": "Diversity: contraction",
        "diversity_expansion": "Diversity: expansion",
        "curve_mae": "Curve MAE",
        "log_psd": "Log PSD",
    }
    curve_index = 0
    for name, node in families.items():
        curve = node.get("severity_curve")
        if not curve:
            continue
        gaps = np.asarray([row["gap"] for row in curve], dtype=float)
        normalized_severity = np.linspace(0.0, 1.0, len(curve))
        ax.plot(
            normalized_severity,
            gaps,
            marker=("o", "s", "D", "^", "v", "P", "X", "*")[curve_index % 8],
            linewidth=1.2,
            markersize=4.5,
            color=colors(curve_index % 10),
            label=severity_labels.get(name, name.replace("_", " ")),
        )
        curve_index += 1
    ax.axhline(1.0, color="#a33b20", linestyle="--", linewidth=1.1)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Normalized control severity")
    ax.set_ylabel("Real-reference gap")
    ax.set_title("d  Metric-family severity responses", loc="left", fontweight="semibold")
    ax.legend(frameon=False, fontsize=7.2, ncol=2)

    for ax in axes.flat:
        ax.grid(color="#e1e7e9", linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.title.set_fontsize(12.8)
        ax.xaxis.label.set_size(11.2)
        ax.yaxis.label.set_size(11.2)
        ax.tick_params(axis="both", labelsize=9.3)
    pdf = Path(f"{path}.pdf")
    png = Path(f"{path}.png")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def run_validation(
    *,
    trials: int,
    resamples: int,
    seed: int,
    jobs: int,
) -> dict[str, Any]:
    if jobs == 1:
        trial_results = [_trial(trial, seed=seed, resamples=resamples) for trial in range(trials)]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(_trial, trial, seed=seed, resamples=resamples)
                for trial in range(trials)
            ]
            trial_results = [future.result() for future in futures]
        trial_results.sort(key=lambda item: int(item["trial"]))
    return _aggregate(
        trial_results,
        resamples=resamples,
        seed=seed,
        jobs=jobs,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--resamples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1789)
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(4, os.cpu_count() or 1),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/statistical_validation_v0.2.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("validation/statistical_validation_v0.2"),
        help="Figure base path; .pdf and .png are added.",
    )
    parser.add_argument(
        "--appendix-figure",
        type=Path,
        default=Path("validation/statistical_validation_v0.2_appendix"),
        help="Appendix stability-figure base path; .pdf and .png are added.",
    )
    args = parser.parse_args()
    if args.trials < 100 or args.resamples < 100:
        raise ValueError("Validation requires at least 100 trials and 100 resamples.")
    if args.jobs < 1:
        raise ValueError("--jobs must be positive.")
    result = run_validation(
        trials=args.trials,
        resamples=args.resamples,
        seed=args.seed,
        jobs=args.jobs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    figure_paths = _validation_figure(result, args.figure)
    appendix_figure_paths = _validation_appendix_figure(
        result,
        args.appendix_figure,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "figures": [str(path) for path in (*figure_paths, *appendix_figure_paths)],
                "gates": result["gates"],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
