#!/usr/bin/env python
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

from climategen_eval.data import RealDataset, SampleBank
from climategen_eval.standardization import (
    standardize_metric_sample_distributions,
    standardize_metric_values,
)
from climategen_eval.tc_bt import TCMetricConfig, evaluate_model_by_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TC BT generated sample banks.")
    parser.add_argument("--config", default="configs/tc_bt_case_study.yaml")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Repeatable LABEL=PATH_TO_SAMPLE_BANK_DIR.",
    )
    parser.add_argument("--n-per-class", type=int, default=1000)
    parser.add_argument("--output-root", default="outputs/tropical_cyclone_bt/eval")
    parser.add_argument(
        "--standardization-report",
        default=None,
        help="Optional metric standardization JSON used to add null-standardized gaps.",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=None,
        help="Override uncertainty.bootstrap_reps from the config; use 0 to disable.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=None,
        help="Override uncertainty.seed from the config.",
    )
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


def parse_model(raw: str) -> tuple[str, Path]:
    if "=" not in str(raw):
        raise ValueError(f"Invalid --model {raw!r}; expected LABEL=PATH.")
    label, path = str(raw).split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Invalid --model {raw!r}; label is empty.")
    out = Path(path.strip())
    if not out.is_absolute():
        out = _REPO_ROOT / out
    return label, out


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

    out = {
        key: value
        for key, value in ranges.items()
        if isinstance(value, dict)
        and bool(value.get("usable_for_standardized_gap", False))
        and bool(responsiveness.get(key, {}).get("responsive", False))
    }
    if not out:
        raise ValueError(
            f"{path} has no metrics passing both null-width and responsiveness checks."
        )
    return out


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
        standardization_ranges = validated_standardization_ranges(payload, norm_path)
        standardization_report_path = norm_path

    models = [parse_model(raw) for raw in args.model]
    for label, bank_dir in tqdm(
        models,
        desc="Evaluating models",
        unit="model",
        disable=not show_progress,
    ):
        bank = SampleBank.from_dir(bank_dir)
        expected_classes = set(real_dataset.class_ids)
        bank_classes = set(bank.class_ids)
        if bank_classes != expected_classes:
            raise ValueError(
                f"Sample bank {label!r} classes {sorted(bank_classes)} do not match "
                f"real dataset classes {sorted(expected_classes)}."
            )
        generated_by_class = {
            class_id: bank.load_bt_k(class_id, limit=int(args.n_per_class), mmap_mode=None)
            for class_id in tqdm(
                bank.class_ids,
                desc=f"Loading {label}",
                unit="class",
                disable=not show_progress,
                leave=False,
            )
        }
        report = evaluate_model_by_class(
            real_by_class=real_by_class,
            generated_by_class=generated_by_class,
            cfg=tc_cfg,
            bootstrap_reps=bootstrap_reps,
            bootstrap_seed=bootstrap_seed,
            include_bootstrap_samples=(
                bool(args.store_bootstrap_samples) or standardization_ranges is not None
            ),
            progress=show_progress,
            real_storm_ids_by_class=real_storm_ids_by_class,
        )
        if standardization_ranges is not None:
            bootstrap_samples = report.get("aggregate_primary_bootstrap", {}).get("samples")
            report["aggregate_primary_standardized"] = {
                "schema": "climategen_eval.null_standardized_gaps.v1",
                "source": str(standardization_report_path),
                "metric_filter": "finite_positive_null_width_and_corruption_responsiveness",
                "values": standardize_metric_values(
                    report["aggregate_primary_raw"],
                    standardization_ranges,
                ),
            }
            if bootstrap_samples:
                report["aggregate_primary_standardized"]["bootstrap"] = {
                    "schema": "climategen_eval.null_standardized_gap_bootstrap.v1",
                    "source": "aggregate_primary_bootstrap.samples",
                    "uncertainty_scope": (
                        "model_evaluation_bootstrap_conditional_on_fixed_standardization_report"
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
                "model_label": label,
                "sample_bank": {
                    "path": str(bank_dir),
                    "run_name": bank.run_name,
                    "bank_name": bank.name,
                    "split": bank.split,
                    "generated_counts_by_class": {
                        str(c): int(n) for c, n in sorted(bank.generated_counts_by_class.items())
                    },
                },
                "n_per_class": int(args.n_per_class),
                "sample_counts": {
                    "real_by_class": {
                        str(c): int(v.shape[0]) for c, v in sorted(real_by_class.items())
                    },
                    "generated_by_class": {
                        str(c): int(v.shape[0]) for c, v in sorted(generated_by_class.items())
                    },
                },
            }
        )
        out_dir = output_root / label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "metrics.json"
        with out_path.open("w") as f:
            json.dump(report, f, indent=2, default=json_default)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
