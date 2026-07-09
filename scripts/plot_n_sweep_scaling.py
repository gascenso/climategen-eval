#!/usr/bin/env python
"""Visualise the sample-size scaling of the real-vs-real noise floor.

Reads the JSON written by ``compute_n_sweep_standardization.py`` and produces
a per-metric small-multiples figure showing how ``q50(n)`` and the noise-floor
half-width ``q95(n) - q50(n)`` scale with the evaluation sample size, plus the
ratio ``q50(n) / (q95(n) - q50(n))`` (which is the gap at the population
truth under finite-sample bias, and therefore the quantity that must
stabilise for the standardized gap to be a stable population statistic).

Power-law slopes are fitted by ordinary least squares on log(n) vs log(value)
and annotated on each panel. A summary CSV/JSON of the fitted slopes is
written alongside the figure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from climategen_eval.standardization import RANGE_DIRECTIONS  # noqa: E402


METRIC_LABELS = {
    "pixel_hist_w1": "BT histogram W1 (K)",
    "pixel_hist_js": "BT histogram JS",
    "dav_w1_deg2": "DAV W1 (deg^2)",
    "radial_profile_mae_k": "Radial profile MAE (K)",
    "cold_cloud_fraction_w1": "Cold-cloud W1",
    "psd_l2": "Radial PSD L2",
    "eye_contrast_proxy_w1_k": "Eye contrast W1 (K)",
    "feature_mmd2": "Feature MMD^2",
    "diversity_feature_space_abs_gap": "Diversity gap",
    "gen_exceedance_rate_total": "Exceedance rate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the n_per_class scaling of the noise floor.")
    parser.add_argument("--input", required=True, help="Path to the n-sweep JSON.")
    parser.add_argument("--output", required=True, help="Output figure path (.svg or .png).")
    parser.add_argument(
        "--slopes-output",
        default=None,
        help="Optional JSON path for fitted power-law slopes.",
    )
    parser.add_argument("--max-cols", type=int, default=3)
    return parser.parse_args()


def metric_label(key: str) -> str:
    return METRIC_LABELS.get(key, key)


def gather_series(payload: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    """Return per-metric arrays {n, q50, half_width, q05, q95}."""
    per_n = payload.get("per_n", [])
    metric_keys: set[str] = set()
    for entry in per_n:
        metric_keys.update(entry.get("by_metric", {}).keys())
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in sorted(metric_keys):
        direction = RANGE_DIRECTIONS.get(key, "lower")
        n_vals: list[int] = []
        q50_vals: list[float] = []
        half_widths: list[float] = []
        q05_vals: list[float] = []
        q95_vals: list[float] = []
        oversampled: list[bool] = []
        for entry in per_n:
            row = entry.get("by_metric", {}).get(key)
            if row is None:
                continue
            n_vals.append(int(entry["n_per_class"]))
            q50_vals.append(float(row["q50"]))
            q05_vals.append(float(row["q05"]))
            q95_vals.append(float(row["q95"]))
            if str(direction) == "higher":
                half_widths.append(float(row["lower_half_width_q50_minus_q05"]))
            else:
                half_widths.append(float(row["upper_half_width_q95_minus_q50"]))
            oversampled.append(bool(entry.get("any_class_oversampled", False)))
        out[key] = {
            "n": np.asarray(n_vals, dtype=np.float64),
            "q50": np.asarray(q50_vals, dtype=np.float64),
            "half_width": np.asarray(half_widths, dtype=np.float64),
            "q05": np.asarray(q05_vals, dtype=np.float64),
            "q95": np.asarray(q95_vals, dtype=np.float64),
            "oversampled": np.asarray(oversampled, dtype=bool),
            "direction": direction,
        }
    return out


def fit_power_law(n: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    """OLS fit log10(y) = a + slope * log10(n) over the masked support."""
    valid = mask & (n > 0) & np.isfinite(y) & (y > 0)
    if int(valid.sum()) < 2:
        return {"slope": None, "intercept": None, "r2": None, "n_points": int(valid.sum())}
    log_n = np.log10(n[valid])
    log_y = np.log10(y[valid])
    slope, intercept = np.polyfit(log_n, log_y, 1)
    pred = slope * log_n + intercept
    ss_res = float(np.sum((log_y - pred) ** 2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else None
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": None if r2 is None else float(r2),
        "n_points": int(valid.sum()),
    }


def plot_metric_row(
    axes_row: list[plt.Axes],
    key: str,
    series: dict[str, np.ndarray],
) -> dict[str, Any]:
    n = series["n"]
    q50 = series["q50"]
    half_width = series["half_width"]
    oversampled = series["oversampled"]
    iid_mask = ~oversampled

    # The iid-regime fit is the defensible one. Points where any class was
    # oversampled (n > min_real_count) are plotted with an open marker so the
    # reader can see them but they do not enter the slope estimate.
    fit_q50 = fit_power_law(n, q50, iid_mask)
    fit_hw = fit_power_law(n, half_width, iid_mask)

    ax_q50, ax_hw, ax_ratio = axes_row
    label = metric_label(key)

    def _scatter(ax: plt.Axes, y: np.ndarray) -> None:
        if np.any(iid_mask):
            ax.scatter(
                n[iid_mask],
                y[iid_mask],
                color="C0",
                edgecolor="C0",
                s=28,
                zorder=3,
                label="iid regime",
            )
        if np.any(oversampled):
            ax.scatter(
                n[oversampled],
                y[oversampled],
                facecolor="white",
                edgecolor="C0",
                s=28,
                zorder=3,
                label="with-replacement",
            )

    _scatter(ax_q50, q50)
    _scatter(ax_hw, half_width)

    def _annotate_fit(ax: plt.Axes, fit: dict[str, float | None]) -> None:
        if fit["slope"] is None or fit["intercept"] is None:
            ax.text(0.05, 0.05, "no fit", transform=ax.transAxes, fontsize=8)
            return
        n_lo = float(np.min(n[iid_mask])) if np.any(iid_mask) else float(np.min(n))
        n_hi = float(np.max(n[iid_mask])) if np.any(iid_mask) else float(np.max(n))
        xs = np.geomspace(n_lo, n_hi, num=64)
        ys = 10.0 ** (fit["slope"] * np.log10(xs) + fit["intercept"])
        ax.plot(xs, ys, color="C3", linewidth=1.4, zorder=2)
        ax.text(
            0.05,
            0.05,
            f"slope = {fit['slope']:+.2f}"
            + (f"  R^2={fit['r2']:.2f}" if fit["r2"] is not None else ""),
            transform=ax.transAxes,
            fontsize=8,
        )

    _annotate_fit(ax_q50, fit_q50)
    _annotate_fit(ax_hw, fit_hw)

    for ax in (ax_q50, ax_hw):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
    ax_q50.set_ylabel(f"{label}\nq50(n)", fontsize=9)
    ax_hw.set_ylabel("noise-floor half-width", fontsize=9)
    ax_ratio.set_ylabel("q50 / half-width", fontsize=9)

    ratio = np.full_like(q50, np.nan, dtype=np.float64)
    nonzero = half_width > 0.0
    ratio[nonzero] = q50[nonzero] / half_width[nonzero]
    ax_ratio.plot(n, ratio, color="C0", marker="o", linewidth=1.2)
    ax_ratio.set_xscale("log")
    ax_ratio.grid(True, which="both", alpha=0.3)
    finite_ratio = ratio[np.isfinite(ratio)]
    if finite_ratio.size:
        rmin = float(np.min(finite_ratio))
        rmax = float(np.max(finite_ratio))
        spread = (rmax - rmin) / max(abs(np.mean(finite_ratio)), 1e-12)
        ax_ratio.text(
            0.05,
            0.05,
            f"range = [{rmin:.2f}, {rmax:.2f}]  rel-spread = {spread:.2f}",
            transform=ax_ratio.transAxes,
            fontsize=8,
        )

    return {
        "metric": key,
        "label": label,
        "direction": series["direction"],
        "q50_power_law": fit_q50,
        "half_width_power_law": fit_hw,
        "ratio_q50_over_half_width": {
            "values_by_n": {
                int(nv): (float(rv) if np.isfinite(rv) else None)
                for nv, rv in zip(n.astype(int).tolist(), ratio.tolist())
            },
            "min_finite": float(np.min(finite_ratio)) if finite_ratio.size else None,
            "max_finite": float(np.max(finite_ratio)) if finite_ratio.size else None,
            "relative_spread": (
                float((np.max(finite_ratio) - np.min(finite_ratio))
                      / max(abs(np.mean(finite_ratio)), 1e-12))
                if finite_ratio.size
                else None
            ),
        },
    }


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = _REPO_ROOT / in_path
    with in_path.open("r") as f:
        payload = json.load(f)
    series_by_metric = gather_series(payload)
    if not series_by_metric:
        raise SystemExit(f"No usable metric series found in {in_path}.")

    metric_keys = sorted(series_by_metric)
    rows = len(metric_keys)
    fig, axes = plt.subplots(
        nrows=rows,
        ncols=3,
        figsize=(11.0, max(2.6 * rows, 3.2)),
        squeeze=False,
    )

    fit_summary: list[dict[str, Any]] = []
    for row_idx, key in enumerate(metric_keys):
        axes_row = list(axes[row_idx])
        result = plot_metric_row(axes_row, key, series_by_metric[key])
        fit_summary.append(result)
        if row_idx == 0:
            axes_row[0].set_title("real-vs-real q50(n)")
            axes_row[1].set_title("noise-floor half-width(n)")
            axes_row[2].set_title("ratio q50 / half-width")
            axes_row[0].legend(fontsize=8, loc="upper right")
        if row_idx == rows - 1:
            for ax in axes_row:
                ax.set_xlabel("n_per_class")

    fig.suptitle(
        "Real-vs-real null scaling with evaluation sample size",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote figure {out_path}")

    slopes_path: Path | None = None
    if args.slopes_output is not None:
        slopes_path = Path(args.slopes_output)
        if not slopes_path.is_absolute():
            slopes_path = _REPO_ROOT / slopes_path
    else:
        slopes_path = out_path.with_suffix(".slopes.json")
    slopes_path.parent.mkdir(parents=True, exist_ok=True)
    with slopes_path.open("w") as f:
        json.dump(
            {
                "schema": "climategen_eval.n_sweep_scaling_summary.v1",
                "input": str(in_path),
                "figure": str(out_path),
                "ns": list(payload.get("ns", [])),
                "real_counts_by_class": payload.get("real_counts_by_class", {}),
                "min_real_count_across_classes": payload.get(
                    "min_real_count_across_classes"
                ),
                "interpretation": {
                    "iid_regime": (
                        "Power-law slopes are fitted only on n values where every "
                        "class can be sampled without replacement. Slopes for q50 "
                        "and the noise-floor half-width should be approximately "
                        "equal for the standardized gap to be a stable population "
                        "quantity; theory predicts ~ -1/2 for 1-D W1-based metrics "
                        "and ~ -1 for chi-squared-like or MMD-like metrics under H0."
                    ),
                    "ratio_q50_over_half_width": (
                        "If the power-law slopes match, this ratio stabilises with n "
                        "and is the implied standardized gap of the population truth "
                        "under finite-sample bias."
                    ),
                },
                "by_metric": fit_summary,
            },
            f,
            indent=2,
        )
    print(f"Wrote slopes {slopes_path}")


if __name__ == "__main__":
    main()
