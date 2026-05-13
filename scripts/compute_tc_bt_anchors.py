#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from climategen_eval.data import RealDataset
from climategen_eval.normalization import compute_anchor_report
from climategen_eval.tc_bt import TCMetricConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute TC BT normalization anchors.")
    parser.add_argument("--config", default="configs/tc_bt_case_study.yaml")
    parser.add_argument("--generated-n-per-class", type=int, default=1000)
    parser.add_argument("--anchor-mode", choices=["fixed", "calibrated"], default="fixed")
    parser.add_argument("--real-reference-reps", type=int, default=None)
    parser.add_argument("--calibration-reps", type=int, default=None)
    parser.add_argument("--calibration-target-power", type=float, default=0.80)
    parser.add_argument("--calibration-null-quantile", type=float, default=0.95)
    parser.add_argument(
        "--calibration-bad-anchor-aggregation",
        choices=["median", "mean"],
        default="median",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", required=True)
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    norm_cfg = dict(cfg.get("metric_normalization", {}))
    seed = int(norm_cfg.get("seed", 123) if args.seed is None else args.seed)
    reps = int(
        norm_cfg.get("real_reference_reps", 200)
        if args.real_reference_reps is None
        else args.real_reference_reps
    )
    real_manifest = Path(cfg["data"]["real_manifest"])
    if not real_manifest.is_absolute():
        real_manifest = _REPO_ROOT / real_manifest
    real_dataset = RealDataset.from_manifest(real_manifest)
    real_by_class = {
        class_id: real_dataset.load_bt_by_class(class_id)
        for class_id in real_dataset.class_ids
    }
    report = compute_anchor_report(
        real_by_class=real_by_class,
        cfg=metric_config(cfg),
        generated_n_per_class=int(args.generated_n_per_class),
        controls=list(norm_cfg.get("controls", [])) or None,
        anchor_mode=str(args.anchor_mode),
        seed=seed,
        real_reference_reps=reps,
        calibration_reps=args.calibration_reps,
        calibration_target_power=float(args.calibration_target_power),
        calibration_null_quantile=float(args.calibration_null_quantile),
        calibration_bad_anchor_aggregation=str(args.calibration_bad_anchor_aggregation),
    )
    report.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "case_study": cfg.get("case_study", "tropical_cyclone_brightness_temperature"),
            "generated_counts_by_class": {
                str(c): int(args.generated_n_per_class) for c in real_dataset.class_ids
            },
            "real_counts_by_class": {
                str(c): int(real_dataset.counts_by_class[c]) for c in real_dataset.class_ids
            },
        }
    )
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2, default=json_default)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

