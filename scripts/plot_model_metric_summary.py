#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from climategen_eval.standardization import (  # noqa: E402
    standardized_metric_value,
)


@dataclass(frozen=True)
class InputSpec:
    label: str
    metrics_path: Path
    is_baseline: bool = False


@dataclass(frozen=True)
class MetricSpec:
    key: str
    report_key: str
    label: str
    scale: float = 1.0


METRICS = [
    MetricSpec("hist_w1_k", "pixel_hist_w1", "BT histogram W1"),
    MetricSpec("hist_js", "pixel_hist_js", "BT histogram JS"),
    MetricSpec("dav_gap_deg2", "dav_abs_gap_deg2", "DAV gap"),
    MetricSpec("radial_mae_k", "radial_profile_mae_k", "Radial profile MAE"),
    MetricSpec(
        "cold_gap_pct",
        "cold_cloud_fraction_abs_gap",
        "Cold-cloud gap",
        scale=100.0,
    ),
    MetricSpec("psd_l2", "psd_l2", "Radial PSD MSE"),
    MetricSpec("eye_gap_k", "eye_contrast_proxy_abs_gap", "Eye contrast gap"),
    MetricSpec("feature_mmd2", "feature_mmd2", "Feature MMD"),
    MetricSpec(
        "feature_diversity_gap",
        "diversity_feature_space_abs_gap",
        "Feature diversity gap",
    ),
    MetricSpec(
        "range_exceedance_pct",
        "gen_exceedance_rate_total",
        "Range exceedance",
        scale=100.0,
    ),
]

MODEL_COLORS = [
    "#25567B",
    "#D2745A",
    "#2F8F83",
    "#7B6BA8",
    "#C19A43",
    "#4D737D",
    "#B85C7A",
    "#667085",
]


# Baselines are rendered in grayscale with a distinct marker so they never
# compete visually with the evaluated models.
BASELINE_COLORS = [
    "#3F4A55",
    "#6B7480",
    "#8E96A1",
    "#B0B7BF",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot null-standardized model metric gap profile."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Repeatable LABEL=PATH_TO_METRICS_JSON for evaluated models.",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        help=(
            "Repeatable LABEL=PATH_TO_METRICS_JSON for bad-but-sensible "
            "reference baselines. Rendered in grayscale with diamond markers."
        ),
    )
    parser.add_argument("--standardization-report", required=True)
    parser.add_argument("--output", default="outputs/model_metric_gap_profile.svg")
    parser.add_argument("--title", default="Climate Generative Model Metric Summary")
    parser.add_argument(
        "--x-scale",
        choices=["symlog", "linear"],
        default="symlog",
        help=(
            "x-axis scale. 'symlog' (default) renders [0, linthresh] linearly "
            "and log-compresses larger gaps, so a baseline at gap ~ 200 does "
            "not squash models at gap ~ 2. 'linear' is the original layout."
        ),
    )
    parser.add_argument(
        "--x-linthresh",
        type=float,
        default=1.0,
        help="symlog linear-region threshold; defaults to 1 noise-floor width.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help=(
            "Optional manual cap on the x-axis upper bound. Only honored in "
            "linear mode; symlog auto-fits to observed values."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def parse_input(raw: str, *, is_baseline: bool = False) -> InputSpec:
    if "=" not in str(raw):
        raise ValueError(f"Invalid input {raw!r}; expected LABEL=PATH.")
    label, path = str(raw).split("=", 1)
    label = label.strip()
    metrics_path = Path(path.strip())
    if not metrics_path.is_absolute():
        metrics_path = _REPO_ROOT / metrics_path
    return InputSpec(label=label, metrics_path=metrics_path, is_baseline=is_baseline)


def nested_get(node: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    cur: Any = node
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def extract_values(path: Path) -> dict[str, float]:
    report = load_json(path)
    aggregate = report.get("aggregate_primary_raw", {})
    if not isinstance(aggregate, dict):
        raise ValueError(f"{path} is missing aggregate_primary_raw.")
    values = {}
    for spec in METRICS:
        if aggregate.get(spec.report_key) is None:
            continue
        values[spec.key] = float(aggregate[spec.report_key]) * float(spec.scale)
    return values


def extract_gap_intervals(path: Path) -> dict[str, tuple[float, float]]:
    report = load_json(path)
    values = nested_get(
        report,
        ("aggregate_primary_standardized", "bootstrap", "values"),
        {},
    )
    if not isinstance(values, dict):
        return {}
    intervals = {}
    for spec in METRICS:
        node = values.get(spec.report_key)
        gap = node.get("standardized_gap", {}) if isinstance(node, dict) else {}
        interval = gap.get("interval_95", {}) if isinstance(gap, dict) else {}
        if not isinstance(interval, dict):
            continue
        low = interval.get("low")
        high = interval.get("high")
        if low is None or high is None:
            continue
        intervals[spec.key] = (
            max(0.0, float(low)),
            max(0.0, float(high)),
        )
    return intervals


def null_standardization_for_spec(
    standardization_norm: dict[str, Any],
    spec: MetricSpec,
) -> dict[str, Any] | None:
    standardization = standardization_norm.get("null_standardization", {})
    ranges = standardization.get("ranges", {}) if isinstance(standardization, dict) else {}
    if not isinstance(ranges, dict):
        return None
    node = ranges.get(spec.report_key)
    if not isinstance(node, dict) or not node.get("usable_for_standardized_gap", False):
        return None
    out = dict(node)
    out["good"] = float(out["good"]) * float(spec.scale)
    out["scale"] = float(out["scale"]) * abs(float(spec.scale))
    if out.get("null_boundary") is not None:
        out["null_boundary"] = float(out["null_boundary"]) * float(spec.scale)
    return out


def responsive_validation_for_spec(
    standardization_payload: dict[str, Any],
    spec: MetricSpec,
) -> bool:
    validation = standardization_payload.get("metric_validation", {})
    corruption_validation = (
        validation.get("corruption_validation", {}) if isinstance(validation, dict) else {}
    )
    responsiveness = (
        corruption_validation.get("responsiveness_by_metric", {})
        if isinstance(corruption_validation, dict)
        else {}
    )
    node = responsiveness.get(spec.report_key) if isinstance(responsiveness, dict) else None
    return isinstance(node, dict) and bool(node.get("responsive", False))


def gap_metric(
    *,
    spec: MetricSpec,
    value: float,
    standardization_norm: dict[str, Any],
) -> float:
    standardization = null_standardization_for_spec(standardization_norm, spec)
    if standardization is None:
        raise ValueError(f"Metric {spec.report_key!r} is missing null-standardization ranges.")
    return float(
        standardized_metric_value(
            value=float(value),
            standardization=standardization,
        )["standardized_gap"]
    )


def build_gaps(
    values_by_label: dict[str, dict[str, float]],
    standardization_payload: dict[str, Any],
    standardization_norm: dict[str, Any],
) -> tuple[list[MetricSpec], dict[str, dict[str, float]]]:
    available_specs = [
        spec
        for spec in METRICS
        if all(spec.key in values for values in values_by_label.values())
        and null_standardization_for_spec(standardization_norm, spec) is not None
        and responsive_validation_for_spec(standardization_payload, spec)
    ]
    gaps = {label: {} for label in values_by_label}
    for spec in available_specs:
        for label in values_by_label:
            gaps[label][spec.key] = gap_metric(
                spec=spec,
                value=values_by_label[label][spec.key],
                standardization_norm=standardization_norm,
            )
    return available_specs, gaps


def _symlog_major_ticks(max_observed: float, linthresh: float) -> list[float]:
    """Major tick locations for a symlog axis covering [0, max_observed]."""
    ticks: list[float] = [0.0]
    if linthresh > 0.0:
        ticks.append(float(linthresh))
    candidates = [
        2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0,
        1000.0, 2000.0, 5000.0, 10000.0,
    ]
    upper = max(float(max_observed), float(linthresh))
    for value in candidates:
        if value <= upper * 1.05:
            ticks.append(value)
    return ticks


def _format_tick(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value - round(value)) < 1e-9 and abs(value) < 1e4:
        return f"{int(round(value))}"
    return f"{value:g}"


def plot_summary(
    *,
    specs: list[MetricSpec],
    gaps_by_label: dict[str, dict[str, float]],
    intervals_by_label: dict[str, dict[str, tuple[float, float]]] | None,
    is_baseline_by_label: dict[str, bool],
    output_path: Path,
    title: str,
    x_scale: str = "symlog",
    x_linthresh: float = 1.0,
    x_max_override: float | None = None,
) -> None:
    model_labels = [lab for lab in gaps_by_label if not is_baseline_by_label.get(lab, False)]
    baseline_labels = [lab for lab in gaps_by_label if is_baseline_by_label.get(lab, False)]
    labels = model_labels + baseline_labels
    has_baselines = bool(baseline_labels)

    # Each metric gets a "model lane" (top) and an optional "baseline lane"
    # (bottom). Lanes within the same metric are visually grouped by a light
    # background band; the gap between metrics is wider than the gap between
    # lanes so the eye reads the two-lane block as a single unit.
    lane_gap_within_metric = 0.40
    metric_gap = 1.20
    rows_per_metric = 2 if has_baselines else 1
    metric_height = (
        (rows_per_metric - 1) * lane_gap_within_metric + metric_gap
    )
    y_centers = np.asarray(
        [i * metric_height for i in range(len(specs))],
        dtype=np.float64,
    )
    model_y = y_centers - (lane_gap_within_metric / 2.0 if has_baselines else 0.0)
    baseline_y = y_centers + (lane_gap_within_metric / 2.0 if has_baselines else 0.0)

    # Compress jitter on each lane: at most a small vertical spread within a
    # lane so coincident points stay disambiguated, but no horizontal
    # offsets at all. Color and shape carry the series identity instead.
    model_offsets = {
        label: ((i - 0.5 * (len(model_labels) - 1)) * 0.10) if len(model_labels) > 1 else 0.0
        for i, label in enumerate(model_labels)
    }
    baseline_offsets = {
        label: ((i - 0.5 * (len(baseline_labels) - 1)) * 0.10) if len(baseline_labels) > 1 else 0.0
        for i, label in enumerate(baseline_labels)
    }

    observed_highs: list[float] = []
    for label in labels:
        values = np.asarray([gaps_by_label[label][spec.key] for spec in specs], dtype=np.float64)
        observed_highs.extend(float(v) for v in values if np.isfinite(v))
        intervals = (intervals_by_label or {}).get(label, {})
        observed_highs.extend(
            float(intervals[spec.key][1])
            for spec in specs
            if spec.key in intervals and np.isfinite(intervals[spec.key][1])
        )

    max_observed = max(observed_highs, default=1.0)
    if x_max_override is not None:
        x_max = float(x_max_override)
    elif x_scale == "symlog":
        x_max = float(max(max_observed * 1.25, x_linthresh * 1.2))
    else:
        x_max = max(1.0, float(np.ceil(max_observed * 1.08)))

    fig_height = max(6.0, metric_height * len(specs) * 0.55 + 1.8)
    fig, ax = plt.subplots(figsize=(10.2, fig_height), facecolor="#ffffff")
    ax.set_facecolor("#FBFCFD")

    # Alternating band shading groups each metric's two lanes visually.
    band_color = "#EEF3F6"
    for i, y in enumerate(y_centers):
        if i % 2 == 0:
            ax.axhspan(
                y - 0.5 * metric_height,
                y + 0.5 * metric_height,
                color=band_color,
                zorder=0,
            )

    # Models — top lane per metric.
    for i, label in enumerate(model_labels):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        offset = model_offsets[label]
        for pos, spec in enumerate(specs):
            point = float(gaps_by_label[label][spec.key])
            interval = ((intervals_by_label or {}).get(label, {})).get(spec.key)
            xerr = None
            if interval is not None:
                low = min(float(interval[0]), point)
                high = max(float(interval[1]), point)
                xerr = np.asarray([[point - low], [high - point]], dtype=np.float64)
            ax.errorbar(
                point,
                model_y[pos] + offset,
                xerr=xerr,
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.7,
                capsize=2.0,
                markersize=7.0,
                markerfacecolor=color,
                markeredgecolor=color,
                label=label if pos == 0 else None,
                zorder=3,
            )

    # Baselines — bottom lane per metric, drawn underneath the model lane.
    for i, label in enumerate(baseline_labels):
        color = BASELINE_COLORS[i % len(BASELINE_COLORS)]
        offset = baseline_offsets[label]
        for pos, spec in enumerate(specs):
            point = float(gaps_by_label[label][spec.key])
            interval = ((intervals_by_label or {}).get(label, {})).get(spec.key)
            xerr = None
            if interval is not None:
                low = min(float(interval[0]), point)
                high = max(float(interval[1]), point)
                xerr = np.asarray([[point - low], [high - point]], dtype=np.float64)
            ax.errorbar(
                point,
                baseline_y[pos] + offset,
                xerr=xerr,
                fmt="D",
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=1.6,
                markersize=5.6,
                markerfacecolor="#FFFFFF",
                markeredgecolor=color,
                alpha=0.95,
                label=label if pos == 0 else None,
                zorder=2,
            )

    # Reference lines: only the two that carry framework meaning. Anything
    # beyond gap=1 is "detectably worse than the noise can explain" — extra
    # tick lines past that point were arbitrary and added visual noise.
    ax.axvline(0.0, color="#5C8997", linewidth=1.1, alpha=0.85, zorder=1)
    if 1.0 <= x_max:
        ax.axvline(
            1.0,
            color="#5C8997",
            linewidth=1.1,
            alpha=0.65,
            linestyle="--",
            zorder=1,
        )

    ax.set_yticks(y_centers)
    ax.set_yticklabels([spec.label for spec in specs], fontsize=9.5)
    ax.invert_yaxis()
    y_lo = float(y_centers[0]) - 0.5 * metric_height
    y_hi = float(y_centers[-1]) + 0.5 * metric_height
    # invert_yaxis flips bounds; matplotlib accepts either ordering.
    ax.set_ylim(y_hi, y_lo)
    if x_scale == "symlog":
        ax.set_xscale("symlog", linthresh=float(x_linthresh), linscale=1.0)
        ax.set_xlim(0.0, x_max)
        ticks = _symlog_major_ticks(max_observed, float(x_linthresh))
        ticks = [t for t in ticks if t <= x_max * 1.001]
        ax.set_xticks(ticks)
        ax.set_xticklabels([_format_tick(t) for t in ticks])
        ax.xaxis.set_minor_locator(plt.NullLocator())
    else:
        ax.set_xlim(0.0, x_max)
    ax.set_xlabel(
        "Null-standardized gap beyond real-vs-real median "
        "(upper one-sided null half-widths q95 - q50; "
        "gap = 1 sits at the noise-floor boundary; lower is better)"
    )
    ax.set_title("Null-standardized metric gap profile", pad=14)
    ax.grid(axis="x", color="#D6E7ED", linewidth=0.9)
    # Suppress horizontal gridlines — the alternating bands already separate metrics.
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BFD8E0")
    ax.spines["bottom"].set_color("#BFD8E0")
    fig.suptitle(title, fontsize=15, y=0.98)

    # Group the legend: models first (filled circles), baselines second
    # (open diamonds), with a divider label between them.
    handles, labels_in_legend = ax.get_legend_handles_labels()
    model_handles = [(h, l) for h, l in zip(handles, labels_in_legend) if l in model_labels]
    baseline_handles = [
        (h, l) for h, l in zip(handles, labels_in_legend) if l in baseline_labels
    ]
    ordered = model_handles + baseline_handles
    if ordered:
        ax.legend(
            [h for h, _ in ordered],
            [l for _, l in ordered],
            loc="lower right",
            frameon=False,
            fontsize=8.5,
            ncol=1 if len(ordered) <= 5 else 2,
        )
    note = (
        "Error bars: bootstrap 95% intervals (conditional on the fixed standardization "
        "report). Solid circles = evaluated models; open diamonds = bad-but-sensible "
        "reference baselines (separate lane below each metric)."
    )
    fig.text(
        0.01,
        0.01,
        note,
        fontsize=8.0,
        color="#587082",
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.input and not args.baseline:
        raise ValueError("Must provide at least one --input or --baseline.")
    input_specs: list[InputSpec] = [parse_input(raw) for raw in args.input]
    baseline_specs: list[InputSpec] = [
        parse_input(raw, is_baseline=True) for raw in args.baseline
    ]
    all_specs = input_specs + baseline_specs
    values_by_label = {spec.label: extract_values(spec.metrics_path) for spec in all_specs}
    intervals_by_label = {
        spec.label: extract_gap_intervals(spec.metrics_path)
        for spec in all_specs
    }
    is_baseline_by_label = {spec.label: spec.is_baseline for spec in all_specs}
    standardization_path = Path(args.standardization_report)
    if not standardization_path.is_absolute():
        standardization_path = _REPO_ROOT / standardization_path
    standardization_payload = load_json(standardization_path)
    standardization_norm = standardization_payload.get("metric_standardization")
    if not isinstance(standardization_norm, dict):
        raise ValueError(
            f"{standardization_path} does not contain a metric_standardization object."
        )
    specs, gaps = build_gaps(
        values_by_label,
        standardization_payload,
        standardization_norm,
    )
    if len(specs) < 3:
        raise ValueError(
            "Fewer than three metrics are available across all inputs after "
            "null-width and responsiveness checks; cannot make gap profile."
        )
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _REPO_ROOT / output_path
    plot_summary(
        specs=specs,
        gaps_by_label=gaps,
        intervals_by_label=intervals_by_label,
        is_baseline_by_label=is_baseline_by_label,
        output_path=output_path,
        title=str(args.title),
        x_scale=str(args.x_scale),
        x_linthresh=float(args.x_linthresh),
        x_max_override=args.x_max,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
