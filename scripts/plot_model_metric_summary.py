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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot null-standardized model metric gap profile."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Repeatable LABEL=PATH_TO_METRICS_JSON.",
    )
    parser.add_argument("--standardization-report", required=True)
    parser.add_argument("--output", default="outputs/model_metric_gap_profile.svg")
    parser.add_argument("--title", default="Climate Generative Model Metric Summary")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def parse_input(raw: str) -> InputSpec:
    if "=" not in str(raw):
        raise ValueError(f"Invalid --input {raw!r}; expected LABEL=PATH.")
    label, path = str(raw).split("=", 1)
    label = label.strip()
    metrics_path = Path(path.strip())
    if not metrics_path.is_absolute():
        metrics_path = _REPO_ROOT / metrics_path
    return InputSpec(label=label, metrics_path=metrics_path)


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


def plot_summary(
    *,
    specs: list[MetricSpec],
    gaps_by_label: dict[str, dict[str, float]],
    intervals_by_label: dict[str, dict[str, tuple[float, float]]] | None,
    output_path: Path,
    title: str,
) -> None:
    labels = list(gaps_by_label)
    y_base = np.arange(len(specs), dtype=np.float64)
    offset_step = min(0.18, 0.72 / max(len(labels), 1))
    offsets = {
        label: (idx - 0.5 * (len(labels) - 1)) * offset_step
        for idx, label in enumerate(labels)
    }
    observed_highs: list[float] = []
    for idx, label in enumerate(labels):
        values = np.asarray([gaps_by_label[label][spec.key] for spec in specs], dtype=np.float64)
        observed_highs.extend(float(v) for v in values if np.isfinite(v))
        intervals = (intervals_by_label or {}).get(label, {})
        observed_highs.extend(
            float(intervals[spec.key][1])
            for spec in specs
            if spec.key in intervals and np.isfinite(intervals[spec.key][1])
        )

    max_observed = max(observed_highs, default=1.0)
    x_max = max(1.0, float(np.ceil(max_observed * 1.08)))

    fig_height = max(5.8, 0.48 * len(specs) + 2.0)
    fig, ax = plt.subplots(figsize=(9.6, fig_height), facecolor="#ffffff")
    ax.set_facecolor("#F7FBFC")

    for idx, label in enumerate(labels):
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        y = y_base + offsets[label]
        intervals = (intervals_by_label or {}).get(label, {})
        for pos, spec in enumerate(specs):
            point = float(gaps_by_label[label][spec.key])
            interval = intervals.get(spec.key)
            xerr = None
            if interval is not None:
                low = min(float(interval[0]), point)
                high = max(float(interval[1]), point)
                xerr = np.asarray([[point - low], [high - point]], dtype=np.float64)
            ax.errorbar(
                point,
                y[pos],
                xerr=xerr,
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.6,
                capsize=2.5,
                markersize=5.5,
                label=label if pos == 0 else None,
            )

    for ref, alpha in ((0.0, 0.85), (1.0, 0.55), (2.0, 0.35), (4.0, 0.28), (8.0, 0.22)):
        if ref <= x_max:
            ax.axvline(ref, color="#8DBDCA", linewidth=1.0, alpha=alpha, zorder=0)

    ax.set_yticks(y_base)
    ax.set_yticklabels([spec.label for spec in specs], fontsize=9.0)
    ax.invert_yaxis()
    ax.set_xlim(0.0, x_max)
    ax.set_xlabel(
        "Null-standardized gap beyond real-vs-real median "
        "(one-sided null widths; lower is better)"
    )
    ax.set_title("Null-standardized metric gap profile", pad=14)
    ax.grid(axis="x", color="#D6E7ED", linewidth=0.9)
    ax.grid(axis="y", color="#E7F1F4", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BFD8E0")
    ax.spines["bottom"].set_color("#BFD8E0")
    fig.suptitle(title, fontsize=15, y=0.98)
    ax.legend(loc="upper right", frameon=False)
    fig.text(
        0.01,
        0.01,
        "Error bars show bootstrap 95% intervals when present; standardized gaps are "
        "conditional on the fixed standardization report.",
        fontsize=8.0,
        color="#587082",
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_specs = [parse_input(raw) for raw in args.input]
    values_by_label = {spec.label: extract_values(spec.metrics_path) for spec in input_specs}
    intervals_by_label = {
        spec.label: extract_gap_intervals(spec.metrics_path)
        for spec in input_specs
    }
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
        output_path=output_path,
        title=str(args.title),
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
