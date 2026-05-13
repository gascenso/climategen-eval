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


@dataclass(frozen=True)
class InputSpec:
    label: str
    metrics_path: Path


@dataclass(frozen=True)
class MetricSpec:
    key: str
    report_key: str
    label: str
    direction: str
    scale: float = 1.0


METRICS = [
    MetricSpec("hist_w1_k", "pixel_hist_w1", "BT histogram W1", "lower"),
    MetricSpec("dav_gap_deg2", "dav_abs_gap_deg2", "DAV gap", "lower"),
    MetricSpec("radial_mae_k", "radial_profile_mae_k", "Radial profile MAE", "lower"),
    MetricSpec("cold_gap_pct", "cold_cloud_fraction_abs_gap", "Cold-cloud gap", "lower", scale=100.0),
    MetricSpec("psd_l2", "psd_l2", "Radial PSD L2", "lower"),
    MetricSpec("eye_gap_k", "eye_contrast_proxy_abs_gap", "Eye contrast gap", "lower"),
    MetricSpec("evaluator_fd", "evaluator_fd", "Evaluator FD", "lower_empirical"),
    MetricSpec("precision", "evaluator_embedding_precision", "Precision", "higher"),
    MetricSpec("diversity_closeness", "evaluator_embedding_diversity_closeness", "Diversity match", "higher"),
    MetricSpec("coverage", "evaluator_embedding_coverage", "Coverage", "higher"),
    MetricSpec("memorization_q01", "nearest_train_q01_distance", "Train NN q01", "higher_relative"),
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
    parser = argparse.ArgumentParser(description="Plot normalized model metric spider summary.")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Repeatable LABEL=PATH_TO_METRICS_JSON.",
    )
    parser.add_argument("--normalization-anchors", default=None)
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


def anchor_value(anchor_norm: dict[str, Any], spec: MetricSpec, *, kind: str) -> float | None:
    if kind == "good":
        values = nested_get(anchor_norm, ("good_reference", "values"), {})
    elif kind == "bad":
        values = nested_get(anchor_norm, ("bad_anchor", "values"), {})
    else:
        raise ValueError(f"Unsupported anchor kind: {kind!r}")
    if not isinstance(values, dict) or values.get(spec.report_key) is None:
        return None
    return float(values[spec.report_key]) * float(spec.scale)


def score_metric(
    *,
    spec: MetricSpec,
    value: float,
    all_values: np.ndarray,
    anchor_norm: dict[str, Any] | None,
) -> float:
    if anchor_norm is not None:
        good = anchor_value(anchor_norm, spec, kind="good")
        bad = anchor_value(anchor_norm, spec, kind="bad")
        if good is not None and bad is not None and abs(float(bad) - float(good)) > 1.0e-12:
            if spec.direction.startswith("lower") and bad > good:
                return float(np.clip((bad - value) / (bad - good), 0.0, 1.0))
            if spec.direction.startswith("higher") and good > bad:
                return float(np.clip((value - bad) / (good - bad), 0.0, 1.0))

    if spec.direction == "higher":
        if spec.key in {"coverage", "precision", "diversity_closeness"}:
            return float(np.clip(value, 0.0, 1.0))
        best = float(np.max(all_values))
        return float(value / best) if best > 0.0 else 0.0

    if spec.direction == "higher_relative":
        best = float(np.max(all_values))
        return float(value / best) if best > 0.0 else 0.0

    best = float(np.min(all_values))
    if value <= 0.0:
        return 1.0
    return float(np.clip(best / value, 0.0, 1.0))


def build_scores(
    values_by_label: dict[str, dict[str, float]],
    anchor_norm: dict[str, Any] | None,
) -> tuple[list[MetricSpec], dict[str, dict[str, float]]]:
    available_specs = [
        spec
        for spec in METRICS
        if all(spec.key in values for values in values_by_label.values())
    ]
    scores = {label: {} for label in values_by_label}
    for spec in available_specs:
        all_values = np.asarray([values_by_label[label][spec.key] for label in values_by_label])
        for label in values_by_label:
            scores[label][spec.key] = score_metric(
                spec=spec,
                value=values_by_label[label][spec.key],
                all_values=all_values,
                anchor_norm=anchor_norm,
            )
    return available_specs, scores


def plot_summary(
    *,
    specs: list[MetricSpec],
    scores_by_label: dict[str, dict[str, float]],
    output_path: Path,
    title: str,
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
    ax.set_title("Normalized score profile (higher is better)", pad=28)
    fig.suptitle(title, fontsize=15, y=0.97)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.06), frameon=False, ncols=min(4, len(labels)))
    fig.subplots_adjust(top=0.81, bottom=0.16, left=0.08, right=0.92)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_specs = [parse_input(raw) for raw in args.input]
    values_by_label = {spec.label: extract_values(spec.metrics_path) for spec in input_specs}
    anchor_norm = None
    if args.normalization_anchors:
        anchor_path = Path(args.normalization_anchors)
        if not anchor_path.is_absolute():
            anchor_path = _REPO_ROOT / anchor_path
        anchor_payload = load_json(anchor_path)
        anchor_norm = anchor_payload.get("metric_normalization", anchor_payload)
    specs, scores = build_scores(values_by_label, anchor_norm)
    if len(specs) < 3:
        raise ValueError("Fewer than three metrics are available across all inputs; cannot make spider plot.")
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _REPO_ROOT / output_path
    plot_summary(specs=specs, scores_by_label=scores, output_path=output_path, title=str(args.title))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

