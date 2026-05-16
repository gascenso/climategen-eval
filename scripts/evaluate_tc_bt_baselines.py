#!/usr/bin/env python
"""Evaluate the bad-but-sensible reference baselines on the TC BT case study.

This emits one `metrics.json` per baseline in the same schema as the model
evaluator so the plotting script can consume them as additional inputs.

Each baseline is a procedure with a textbook interpretation and no tuneable
hyperparameters: see `climategen_eval.baselines` for definitions. The
panel is meant to be read together — a model with standardized gaps well
below the baseline cluster on every metric beats the no-skill/no-structure/
no-conditioning reference points.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from climategen_eval.baselines import (  # noqa: E402
    BASELINE_DESCRIPTIONS,
    BASELINE_NAMES,
    build_baseline,
)
from climategen_eval.data import RealDataset  # noqa: E402
from climategen_eval.standardization import (  # noqa: E402
    standardize_metric_sample_distributions,
    standardize_metric_values,
)
from climategen_eval.tc_bt import TCMetricConfig, evaluate_model_by_class  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate bad-but-sensible baselines.")
    parser.add_argument("--config", default="configs/tc_bt_case_study.yaml")
    parser.add_argument("--n-per-class", type=int, default=1000)
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=list(BASELINE_NAMES),
        help=f"Baselines to evaluate (subset of {list(BASELINE_NAMES)}).",
    )
    parser.add_argument("--output-root", default="outputs/tropical_cyclone_bt/baselines")
    parser.add_argument(
        "--standardization-report",
        default=None,
        help="Optional metric standardization JSON used to add null-standardized gaps.",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    parser.add_argument("--baseline-seed", type=int, default=20260516)
    parser.add_argument(
        "--store-bootstrap-samples",
        action="store_true",
        help="Keep per-replicate bootstrap samples in metrics.json.",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    with path.open("r") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object in {path}.")
    return payload


def metric_config(cfg: dict[str, Any]) -> TCMetricConfig:
    data = dict(cfg.get("data", {}))
    ev = dict(cfg.get("evaluation", {}))
    return TCMetricConfig(
        image_size=int(data.get("image_size", 256)),
        bt_min_k=float(data.get("bt_min_k", 170.0)),
        bt_max_k=float(data.get("bt_max_k", 325.0)),
        profile_bins=int(ev.get("profile_bins", 96)),
        psd_bins=int(ev.get("psd_bins", 96)),
        dav_radius_km=float(ev.get("dav_radius_km", 300.0)),
        dav_pixel_size_km=float(ev.get("dav_pixel_size_km", 8.0)),
        dav_center_region_size=int(ev.get("dav_center_region_size", 3)),
        cold_cloud_threshold_k=float(ev.get("cold_cloud_threshold_k", 208.0)),
        cold_cloud_radius_km=float(ev.get("cold_cloud_radius_km", 300.0)),
        cold_cloud_pixel_size_km=float(ev.get("cold_cloud_pixel_size_km", 8.0)),
    )


def json_default(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def validated_standardization_ranges(
    payload: dict[str, Any],
    path: Path,
) -> dict[str, dict[str, Any]]:
    standardization_node = payload.get("metric_standardization")
    if not isinstance(standardization_node, dict):
        raise ValueError(f"{path} does not contain a metric_standardization object.")
    standardization = standardization_node.get("null_standardization", {})
    ranges = standardization.get("ranges", {}) if isinstance(standardization, dict) else {}
    if not isinstance(ranges, dict) or not ranges:
        raise ValueError(f"{path} does not contain null_standardization ranges.")

    validation = payload.get("metric_validation", {})
    corruption_validation = (
        validation.get("corruption_validation", {}) if isinstance(validation, dict) else {}
    )
    responsiveness = (
        corruption_validation.get("responsiveness_by_metric", {})
        if isinstance(corruption_validation, dict)
        else {}
    )
    if not isinstance(responsiveness, dict) or not responsiveness:
        raise ValueError(f"{path} does not contain corruption-validation responsiveness results.")

    return {
        key: value
        for key, value in ranges.items()
        if isinstance(value, dict)
        and bool(value.get("usable_for_standardized_gap", False))
        and bool(responsiveness.get(key, {}).get("responsive", False))
    }


def validate_baseline_names(names: list[str]) -> list[str]:
    unknown = sorted(set(names) - set(BASELINE_NAMES))
    if unknown:
        raise ValueError(
            f"Unknown baselines {unknown}; valid options are {list(BASELINE_NAMES)}"
        )
    return list(dict.fromkeys(names))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    uncertainty_cfg = dict(cfg.get("uncertainty", {}))
    bootstrap_reps = int(
        uncertainty_cfg.get("bootstrap_reps", 200)
        if args.bootstrap_reps is None
        else args.bootstrap_reps
    )
    bootstrap_seed = int(
        uncertainty_cfg.get("seed", 456)
        if args.bootstrap_seed is None
        else args.bootstrap_seed
    )
    baselines = validate_baseline_names(list(args.baselines))

    real_manifest = Path(cfg["data"]["real_manifest"])
    if not real_manifest.is_absolute():
        real_manifest = _REPO_ROOT / real_manifest
    real_dataset = RealDataset.from_manifest(real_manifest)
    tc_cfg = metric_config(cfg)
    show_progress = not bool(args.no_progress)

    real_by_class = {}
    real_storm_ids_by_class = {}
    for class_id in tqdm(
        real_dataset.class_ids,
        desc="Loading real classes",
        unit="class",
        disable=not show_progress,
    ):
        real_by_class[class_id] = real_dataset.load_bt_by_class(class_id)
        real_storm_ids_by_class[class_id] = real_dataset.storm_ids_by_class(class_id)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = _REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    standardization_ranges = None
    standardization_report_path = None
    if args.standardization_report is not None:
        norm_path = Path(args.standardization_report)
        if not norm_path.is_absolute():
            norm_path = _REPO_ROOT / norm_path
        with norm_path.open("r") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected standardization report object in {norm_path}.")
        ranges = validated_standardization_ranges(payload, norm_path)
        if not ranges:
            raise ValueError(
                f"{norm_path} has no metrics passing both null-width and responsiveness checks."
            )
        standardization_ranges = ranges
        standardization_report_path = norm_path

    # Per-metric standardized gaps collected across baselines, used at the
    # end to emit a baseline-responsiveness summary that downstream consumers
    # (model evaluator, plotter) treat as the metric inclusion gate.
    standardized_gap_by_metric_by_baseline: dict[str, dict[str, float]] = {}
    for name in tqdm(
        baselines,
        desc="Evaluating baselines",
        unit="baseline",
        disable=not show_progress,
    ):
        result = build_baseline(
            name,
            real_by_class,
            n_per_class=int(args.n_per_class),
            seed=int(args.baseline_seed) + abs(hash(name)) % (2**31),
            storm_ids_by_class=real_storm_ids_by_class,
        )
        report = evaluate_model_by_class(
            real_by_class=real_by_class,
            generated_by_class=result.images_by_class,
            cfg=tc_cfg,
            bootstrap_reps=bootstrap_reps,
            bootstrap_seed=bootstrap_seed,
            include_bootstrap_samples=(
                bool(args.store_bootstrap_samples) or standardization_ranges is not None
            ),
            progress=show_progress,
            real_storm_ids_by_class=real_storm_ids_by_class,
            generated_storm_ids_by_class=result.storm_ids_by_class,
        )
        if standardization_ranges is not None:
            bootstrap_samples = report.get("aggregate_primary_bootstrap", {}).get("samples")
            standardized_values = standardize_metric_values(
                report["aggregate_primary_raw"],
                standardization_ranges,
            )
            report["aggregate_primary_standardized"] = {
                "schema": "climategen_eval.null_standardized_gaps.v1",
                "source": str(standardization_report_path),
                "metric_filter": "finite_positive_null_width",
                "values": standardized_values,
            }
            for metric_key, node in standardized_values.items():
                gap = float(node.get("standardized_gap", 0.0))
                standardized_gap_by_metric_by_baseline.setdefault(metric_key, {})[
                    result.name
                ] = gap
            if bootstrap_samples:
                report["aggregate_primary_standardized"]["bootstrap"] = {
                    "schema": "climategen_eval.null_standardized_gap_bootstrap.v1",
                    "source": "aggregate_primary_bootstrap.samples",
                    "uncertainty_scope": (
                        "baseline_bootstrap_conditional_on_fixed_standardization_report"
                    ),
                    "values": standardize_metric_sample_distributions(
                        bootstrap_samples,
                        standardization_ranges,
                        point_values=report["aggregate_primary_raw"],
                    ),
                }
        if not bool(args.store_bootstrap_samples):
            report.get("aggregate_primary_bootstrap", {}).pop("samples", None)
        report.update(
            {
                "schema": "climategen_eval.tc_bt_metrics.v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "case_study": cfg.get("case_study", "tropical_cyclone_brightness_temperature"),
                "model_label": result.name,
                "input_kind": "bad_but_sensible_baseline",
                "baseline": {
                    "name": result.name,
                    "description": result.description,
                    "provenance": result.provenance,
                },
                "n_per_class": int(args.n_per_class),
                "sample_counts": {
                    "real_by_class": {
                        str(c): int(v.shape[0]) for c, v in sorted(real_by_class.items())
                    },
                    "generated_by_class": {
                        str(c): int(v.shape[0])
                        for c, v in sorted(result.images_by_class.items())
                    },
                },
            }
        )
        slug = result.name.lower().replace("-", "_")
        out_dir = output_root / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "metrics.json"
        with out_path.open("w") as f:
            json.dump(report, f, indent=2, default=json_default)
        print(f"Wrote {out_path}")

    responsiveness_by_metric: dict[str, Any] = {}
    for metric_key, by_baseline in sorted(standardized_gap_by_metric_by_baseline.items()):
        gaps = {label: float(gap) for label, gap in by_baseline.items()}
        activating = sorted(
            label for label, gap in gaps.items() if gap > 1.0
        )
        max_label = max(gaps, key=lambda label: gaps[label]) if gaps else None
        max_gap = float(gaps[max_label]) if max_label is not None else float("nan")
        responsiveness_by_metric[metric_key] = {
            "active": bool(activating),
            "max_baseline_gap": max_gap,
            "max_baseline_gap_label": max_label,
            "activating_baselines": activating,
            "by_baseline": gaps,
        }

    summary_path = output_root / "baseline_panel.json"
    with summary_path.open("w") as f:
        json.dump(
            {
                "schema": "climategen_eval.baseline_panel.v2",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "baselines": [
                    {
                        "name": name,
                        "description": BASELINE_DESCRIPTIONS[name],
                    }
                    for name in baselines
                ],
                "responsiveness": {
                    "schema": "climategen_eval.baseline_responsiveness.v1",
                    "standardization_report": (
                        None
                        if standardization_report_path is None
                        else str(standardization_report_path)
                    ),
                    "decision_rule": (
                        "metric is active iff at least one baseline's class-aggregate "
                        "standardized gap exceeds 1 (one upper noise-floor half-width)."
                    ),
                    "by_metric": responsiveness_by_metric,
                },
            },
            f,
            indent=2,
            default=json_default,
        )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
