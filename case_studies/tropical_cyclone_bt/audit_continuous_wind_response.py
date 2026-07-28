#!/usr/bin/env python3
"""Audit whether TC functionals vary with the continuous wind-speed condition.

This is a descriptive secondary analysis.  It deliberately produces no
equivalence or pass/fail verdict: the class-conditional GAUGE evaluation remains
the primary analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from gauge_climate.adapters.tc_data import load_real_archive, load_sample_bank


ROOT = Path(__file__).resolve().parents[2]
CLASS_LABELS = ("TS", "C1", "C2", "C3", "C4", "C5")
FUNCTIONALS = (
    "min_core_bt",
    "cold_frac_220",
    "r_maxgrad",
    "axisym",
    "dav",
    "eye_contrast",
    "p01_bt",
)
FUNCTIONAL_LABELS = {
    "min_core_bt": "Core minimum BT",
    "cold_frac_220": "Cold-cloud area < 220 K",
    "r_maxgrad": "Maximum-gradient radius",
    "axisym": "Axisymmetry departure",
    "dav": "Deviation-angle variance",
    "eye_contrast": "Eye minus eyewall BT",
    "p01_bt": "BT first percentile",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _slope_from_moments(
    mean_x: float,
    mean_y: float,
    mean_x2: float,
    mean_xy: float,
) -> float:
    variance = mean_x2 - mean_x * mean_x
    if variance <= np.finfo(float).eps:
        return float("nan")
    return 10.0 * (mean_xy - mean_x * mean_y) / variance


def _sample_slope(x: np.ndarray, y: np.ndarray) -> float:
    return _slope_from_moments(
        float(np.mean(x)),
        float(np.mean(y)),
        float(np.mean(x * x)),
        float(np.mean(x * y)),
    )


def _storm_balanced_slope(
    x: np.ndarray,
    y: np.ndarray,
    blocks: np.ndarray,
) -> tuple[float, np.ndarray]:
    unique = np.unique(blocks)
    moments = np.empty((len(unique), 4), dtype=np.float64)
    for index, block in enumerate(unique):
        selected = blocks == block
        xb = x[selected]
        yb = y[selected]
        moments[index] = (
            np.mean(xb),
            np.mean(yb),
            np.mean(xb * xb),
            np.mean(xb * yb),
        )
    means = moments.mean(axis=0)
    return _slope_from_moments(*means), moments


def _bootstrap_storm_slope(
    moments: np.ndarray,
    *,
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    count = len(moments)
    output = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        draw = moments[rng.integers(0, count, size=count)].mean(axis=0)
        output[index] = _slope_from_moments(*draw)
    return output


def _bootstrap_sample_slope(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    count = len(x)
    output = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        selected = rng.integers(0, count, size=count)
        output[index] = _sample_slope(x[selected], y[selected])
    return output


def _summary(estimate: float, draws: np.ndarray) -> dict[str, Any]:
    finite = draws[np.isfinite(draws)]
    if not np.isfinite(estimate) or len(finite) < max(20, len(draws) // 2):
        return {
            "estimate_per_10kt": None,
            "interval_95_per_10kt": [None, None],
            "valid_resamples": int(len(finite)),
            "warning": "insufficient wind variation for a stable slope",
        }
    low, high = np.quantile(finite, [0.025, 0.975])
    return {
        "estimate_per_10kt": float(estimate),
        "interval_95_per_10kt": [float(low), float(high)],
        "valid_resamples": int(len(finite)),
        "warning": None,
    }


def _validate_slope_engine(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    blocks = np.repeat(np.arange(40), 20)
    x = rng.uniform(35.0, 80.0, size=len(blocks))
    responsive = 0.5 * x + rng.normal(0.0, 1.5, size=len(x))
    ignoring = rng.normal(0.0, 1.5, size=len(x))
    responsive_estimate, responsive_moments = _storm_balanced_slope(x, responsive, blocks)
    ignoring_estimate, ignoring_moments = _storm_balanced_slope(x, ignoring, blocks)
    responsive_draws = _bootstrap_storm_slope(responsive_moments, n_resamples=500, rng=rng)
    ignoring_draws = _bootstrap_storm_slope(ignoring_moments, n_resamples=500, rng=rng)
    responsive_interval = np.quantile(responsive_draws, [0.025, 0.975])
    ignoring_interval = np.quantile(ignoring_draws, [0.025, 0.975])

    generated_x = rng.uniform(35.0, 80.0, size=800)
    responsive_generated = 0.5 * generated_x + rng.normal(
        0.0,
        1.5,
        size=len(generated_x),
    )
    ignoring_generated = rng.normal(0.0, 1.5, size=len(generated_x))
    responsive_generated_estimate = _sample_slope(
        generated_x,
        responsive_generated,
    )
    ignoring_generated_estimate = _sample_slope(
        generated_x,
        ignoring_generated,
    )
    responsive_generated_draws = _bootstrap_sample_slope(
        generated_x,
        responsive_generated,
        n_resamples=500,
        rng=rng,
    )
    ignoring_generated_draws = _bootstrap_sample_slope(
        generated_x,
        ignoring_generated,
        n_resamples=500,
        rng=rng,
    )
    responsive_generated_interval = np.quantile(
        responsive_generated_draws,
        [0.025, 0.975],
    )
    ignoring_generated_interval = np.quantile(
        ignoring_generated_draws,
        [0.025, 0.975],
    )
    passed = bool(
        responsive_interval[0] > 0.0
        and ignoring_interval[0] <= 0.0 <= ignoring_interval[1]
        and abs(responsive_estimate - 5.0) < 0.5
        and responsive_generated_interval[0] > 0.0
        and ignoring_generated_interval[0] <= 0.0 <= ignoring_generated_interval[1]
        and abs(responsive_generated_estimate - 5.0) < 0.5
    )
    return {
        "event_blocked_real_engine": {
            "responsive_expected_per_10kt": 5.0,
            "responsive_estimate_per_10kt": float(responsive_estimate),
            "responsive_interval_95_per_10kt": responsive_interval.tolist(),
            "wind_ignoring_estimate_per_10kt": float(ignoring_estimate),
            "wind_ignoring_interval_95_per_10kt": ignoring_interval.tolist(),
        },
        "fixed_generated_bank_engine": {
            "responsive_expected_per_10kt": 5.0,
            "responsive_estimate_per_10kt": float(responsive_generated_estimate),
            "responsive_interval_95_per_10kt": responsive_generated_interval.tolist(),
            "wind_ignoring_estimate_per_10kt": float(ignoring_generated_estimate),
            "wind_ignoring_interval_95_per_10kt": ignoring_generated_interval.tolist(),
        },
        # Retain flat fields for read-only compatibility with the earlier audit.
        "responsive_expected_per_10kt": 5.0,
        "responsive_estimate_per_10kt": float(responsive_estimate),
        "responsive_interval_95_per_10kt": responsive_interval.tolist(),
        "wind_ignoring_estimate_per_10kt": float(ignoring_estimate),
        "wind_ignoring_interval_95_per_10kt": ignoring_interval.tolist(),
        "passed": passed,
    }


def _plot(payload: dict[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(4, 2, figsize=(10.6, 12.0), sharex=True)
    axes_flat = axes.ravel()
    colors = {"real": "#263238", "generated": "#c45b3c"}
    offsets = {"real": -0.09, "generated": 0.09}
    x = np.arange(6)
    for axis, functional in zip(axes_flat, FUNCTIONALS, strict=False):
        rows = payload["functionals"][functional]
        for source in ("real", "generated"):
            estimates = []
            lower = []
            upper = []
            for row in rows:
                summary = row[source]
                estimate = summary["estimate_per_10kt"]
                interval = summary["interval_95_per_10kt"]
                estimates.append(np.nan if estimate is None else estimate)
                lower.append(np.nan if interval[0] is None else estimate - interval[0])
                upper.append(np.nan if interval[1] is None else interval[1] - estimate)
            estimates_array = np.asarray(estimates, dtype=float)
            axis.errorbar(
                x + offsets[source],
                estimates_array,
                yerr=np.asarray([lower, upper]),
                fmt="o" if source == "real" else "s",
                ms=4.8,
                capsize=2.5,
                lw=1.2,
                color=colors[source],
                label="Held-out real" if source == "real" else "Generated",
            )
        axis.axhline(0.0, color="#9aa3a8", lw=0.8, zorder=0)
        axis.set_title(FUNCTIONAL_LABELS[functional], fontsize=11, loc="left")
        axis.grid(axis="y", color="#e2e7e9", lw=0.6)
        axis.tick_params(labelsize=9)
    axes_flat[-1].axis("off")
    for axis in axes_flat[:-2]:
        axis.tick_params(labelbottom=False)
    axes_flat[-2].set_xticks(x, CLASS_LABELS)
    axes_flat[-2].set_xlabel("Intensity condition", fontsize=10)
    figure.supylabel("Functional slope per 10 kt", fontsize=11, x=0.02)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=2,
        loc="lower right",
        bbox_to_anchor=(0.97, 0.06),
        frameon=False,
        fontsize=10,
    )
    figure.tight_layout(rect=(0.04, 0.04, 1.0, 1.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "case_studies" / "tropical_cyclone_bt" / "track1_config.template.yaml",
    )
    parser.add_argument(
        "--functional-cache",
        type=Path,
        default=ROOT / "outputs" / "tropical_cyclone_bt" / "presentation" / "functionals.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "tropical_cyclone_bt" / "audits",
    )
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1789)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    real_node = config["data"]["real"]
    generated_node = config["data"]["generated"]
    real = load_real_archive(_resolve(config_path, real_node["tc_real_manifest"]), target="wind")
    generated = load_sample_bank(
        _resolve(config_path, generated_node["tc_sample_bank"]),
        target="wind",
        required_run=generated_node.get("required_run"),
    )
    archive = np.load(args.functional_cache)
    real_categories = np.asarray(archive["real_labels"], dtype=np.int16)
    generated_categories = np.asarray(archive["generated_labels"], dtype=np.int16)
    real_blocks = np.asarray(archive["real_blocks"]).astype(str)
    if not np.array_equal(real_blocks, np.asarray(real.blocks).astype(str)):
        raise ValueError("Functional cache does not align with the real storm archive.")
    if len(generated_categories) != len(generated.labels):
        raise ValueError("Functional cache does not align with the generated bank.")

    rng = np.random.default_rng(args.seed)
    results: dict[str, list[dict[str, Any]]] = {}
    csv_rows: list[dict[str, Any]] = []
    for functional in FUNCTIONALS:
        real_values = np.asarray(archive[f"real_{functional}"], dtype=np.float64)
        generated_values = np.asarray(archive[f"generated_{functional}"], dtype=np.float64)
        rows = []
        for class_id, class_label in enumerate(CLASS_LABELS):
            real_mask = real_categories == class_id
            generated_mask = generated_categories == class_id
            real_estimate, real_moments = _storm_balanced_slope(
                np.asarray(real.labels)[real_mask],
                real_values[real_mask],
                real_blocks[real_mask],
            )
            real_draws = _bootstrap_storm_slope(real_moments, n_resamples=args.resamples, rng=rng)
            generated_estimate = _sample_slope(
                np.asarray(generated.labels)[generated_mask],
                generated_values[generated_mask],
            )
            generated_draws = _bootstrap_sample_slope(
                np.asarray(generated.labels)[generated_mask],
                generated_values[generated_mask],
                n_resamples=args.resamples,
                rng=rng,
            )
            row = {
                "condition": class_label,
                "real": {
                    **_summary(real_estimate, real_draws),
                    "samples": int(real_mask.sum()),
                    "storms": int(len(np.unique(real_blocks[real_mask]))),
                    "weighting": "inverse storm frequency within condition",
                },
                "generated": {
                    **_summary(generated_estimate, generated_draws),
                    "samples": int(generated_mask.sum()),
                    "independent_blocks": None,
                    "weighting": "equal sample weight",
                },
            }
            rows.append(row)
            for source in ("real", "generated"):
                summary = row[source]
                csv_rows.append(
                    {
                        "functional": functional,
                        "condition": class_label,
                        "source": source,
                        "estimate_per_10kt": summary["estimate_per_10kt"],
                        "interval_95_low": summary["interval_95_per_10kt"][0],
                        "interval_95_high": summary["interval_95_per_10kt"][1],
                        "samples": summary["samples"],
                        "blocks": summary.get("storms"),
                    }
                )
        results[functional] = rows

    validation = _validate_slope_engine(args.seed + 1)
    if not validation["passed"]:
        raise RuntimeError("Synthetic continuous-wind slope validation failed.")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "gauge_climate.tc_bt_continuous_wind_audit.v1",
        "interpretation": (
            "Descriptive secondary audit only; no equivalence or fidelity verdict. "
            "Slopes are changes in each functional per 10 kt within the native class."
        ),
        "real_weighting": (
            "Inverse storm-frequency weights, implemented as equal means of "
            "storm-specific first and cross moments."
        ),
        "generated_weighting": "Equal sample weights.",
        "intervals": {
            "level": 0.95,
            "real": "storm bootstrap",
            "generated": "sample bootstrap conditional on the fixed sample bank",
            "resamples": args.resamples,
        },
        "inputs": {
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "functional_cache": str(args.functional_cache.resolve()),
            "functional_cache_sha256": _sha256(args.functional_cache.resolve()),
            "real_manifest": real.metadata["manifest_path"],
            "generated_manifest": generated.metadata["manifest_path"],
        },
        "synthetic_validation": validation,
        "functionals": results,
    }
    json_path = output / "continuous_wind_response.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    csv_path = output / "continuous_wind_response.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    _plot(payload, output / "figure_A_continuous_wind_response")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
