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
    DEFAULT_DISPLAY_NULL_WIDTHS,
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
    MetricSpec("psd_l2", "psd_l2", "Radial PSD L2"),
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
        description="Plot null-standardized model metric spider summary."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Repeatable LABEL=PATH_TO_METRICS_JSON.",
    )
    parser.add_argument("--standardization-report", required=True)
    parser.add_argument(
        "--display-null-widths",
        type=float,
        default=None,
        help=(
            "Number of one-sided real-vs-real null widths mapped to zero "
            "for null-standardized scores."
        ),
    )
    parser.add_argument("--output", default="outputs/model_metric_summary.svg")
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


def score_metric(
    *,
    spec: MetricSpec,
    value: float,
    standardization_norm: dict[str, Any],
    display_null_widths: float,
) -> float:
    standardization = null_standardization_for_spec(standardization_norm, spec)
    if standardization is None:
        raise ValueError(f"Metric {spec.report_key!r} is missing null-standardization ranges.")
    return float(
        standardized_metric_value(
            value=float(value),
            standardization=standardization,
            display_null_widths=float(display_null_widths),
        )["score"]
    )


def build_scores(
    values_by_label: dict[str, dict[str, float]],
    standardization_payload: dict[str, Any],
    standardization_norm: dict[str, Any],
    *,
    display_null_widths: float,
) -> tuple[list[MetricSpec], dict[str, dict[str, float]]]:
    available_specs = [
        spec
        for spec in METRICS
        if all(spec.key in values for values in values_by_label.values())
        and null_standardization_for_spec(standardization_norm, spec) is not None
        and responsive_validation_for_spec(standardization_payload, spec)
    ]
    scores = {label: {} for label in values_by_label}
    for spec in available_specs:
        for label in values_by_label:
            scores[label][spec.key] = score_metric(
                spec=spec,
                value=values_by_label[label][spec.key],
                standardization_norm=standardization_norm,
                display_null_widths=display_null_widths,
            )
    return available_specs, scores


def plot_summary(
    *,
    specs: list[MetricSpec],
    scores_by_label: dict[str, dict[str, float]],
    output_path: Path,
    title: str,
    display_null_widths: float,
) -> None:
    labels = list(scores_by_label)
    theta = np.linspace(0.0, 2.0 * np.pi, len(specs), endpoint=False)
    theta_closed = np.concatenate([theta, theta[:1]])

    fig = plt.figure(figsize=(9.0, 7.6), facecolor="#ffffff")
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_facecolor("#F7FBFC")

    for idx, label in enumerate(labels):
        values = np.asarray([scores_by_label[label][spec.key] for spec in specs], dtype=np.float64)
        values_closed = np.concatenate([values, values[:1]])
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        ax.plot(theta_closed, values_closed, color=color, linewidth=2.2, label=label)
        ax.fill(theta_closed, values_closed, color=color, alpha=0.10)

    ax.set_xticks(theta)
    ax.set_xticklabels([spec.label.replace(" ", "\n") for spec in specs], fontsize=8.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7.5, color="#587082")
    ax.grid(color="#D6E7ED", linewidth=0.9)
    ax.spines["polar"].set_color("#A3CEDD")
    ax.set_title(
        f"Null-standardized score profile (0 = {display_null_widths:g} null widths)",
        pad=28,
    )
    fig.suptitle(title, fontsize=15, y=0.97)
    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        frameon=False,
        ncols=min(4, len(labels)),
    )
    fig.subplots_adjust(top=0.81, bottom=0.16, left=0.08, right=0.92)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_specs = [parse_input(raw) for raw in args.input]
    values_by_label = {spec.label: extract_values(spec.metrics_path) for spec in input_specs}
    standardization_path = Path(args.standardization_report)
    if not standardization_path.is_absolute():
        standardization_path = _REPO_ROOT / standardization_path
    standardization_payload = load_json(standardization_path)
    standardization_norm = standardization_payload.get("metric_standardization")
    if not isinstance(standardization_norm, dict):
        raise ValueError(
            f"{standardization_path} does not contain a metric_standardization object."
        )
    display_null_widths = (
        float(args.display_null_widths)
        if args.display_null_widths is not None
        else float(
            nested_get(
                standardization_norm,
                ("null_standardization", "display_null_widths"),
                DEFAULT_DISPLAY_NULL_WIDTHS,
            )
        )
    )
    specs, scores = build_scores(
        values_by_label,
        standardization_payload,
        standardization_norm,
        display_null_widths=display_null_widths,
    )
    if len(specs) < 3:
        raise ValueError(
            "Fewer than three metrics are available across all inputs after "
            "null-width and responsiveness checks; cannot make spider plot."
        )
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _REPO_ROOT / output_path
    plot_summary(
        specs=specs,
        scores_by_label=scores,
        output_path=output_path,
        title=str(args.title),
        display_null_widths=display_null_widths,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
