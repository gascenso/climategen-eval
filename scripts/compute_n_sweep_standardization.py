#!/usr/bin/env python
"""Sample-size sweep of the real-vs-real noise floor.

Runs the real-vs-real bootstrap at several values of ``generated_n_per_class``
and stores, per metric, the bootstrap summary (q05, q50, q95, ...) together
with the implied half-widths used by the standardized-gap denominator. The
output supports asymptotic-scaling diagnostics: how the finite-sample bias
``q50(n)`` and the noise-floor half-width ``q95(n) - q50(n)`` scale with the
evaluation sample size, and whether their ratio stabilises (which is the
condition under which the standardized gap is a stable population quantity).

The corruption sweep and calibration audit are skipped here because they do
not enter the noise-floor scaling argument; the canonical standardization
report from ``compute_tc_bt_standardization.py`` should still be produced for
the main results.
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

from climategen_eval.data import RealDataset  # noqa: E402
from climategen_eval.standardization import (  # noqa: E402
    compute_standardization_report,
)
from climategen_eval.tc_bt import TCMetricConfig  # noqa: E402


DEFAULT_NS = "50,100,200,500,1000,2000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample-size sweep of the real-vs-real noise floor."
    )
    parser.add_argument("--config", default="configs/tc_bt_case_study.yaml")
    parser.add_argument(
        "--ns",
        default=DEFAULT_NS,
        help=(
            "Comma-separated list of generated_n_per_class values to sweep. "
            f"Default: {DEFAULT_NS}."
        ),
    )
    parser.add_argument("--real-reference-reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", required=True)
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


def parse_ns(spec: str) -> list[int]:
    out: list[int] = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"--ns entries must be > 0, got {value}.")
        out.append(value)
    if len(out) < 2:
        raise ValueError("--ns must contain at least two distinct sample sizes.")
    return sorted(set(out))


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


def per_metric_summary_view(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    physical = (
        report.get("metric_standardization", {})
        .get("good_reference", {})
        .get("physical_real_vs_real", {})
    )
    summaries = physical.get("summary", {}) if isinstance(physical, dict) else {}
    out: dict[str, dict[str, float]] = {}
    for key, node in sorted(summaries.items()):
        if not isinstance(node, dict):
            continue
        q05 = node.get("q05")
        q50 = node.get("q50")
        q95 = node.get("q95")
        if q05 is None or q50 is None or q95 is None:
            continue
        upper_half_width = float(q95) - float(q50)
        lower_half_width = float(q50) - float(q05)
        out[key] = {
            "q05": float(q05),
            "q50": float(q50),
            "q95": float(q95),
            "upper_half_width_q95_minus_q50": upper_half_width,
            "lower_half_width_q50_minus_q05": lower_half_width,
            "mean": float(node.get("mean", float("nan"))),
            "std": float(node.get("std", float("nan"))),
        }
    return out


def main() -> None:
    args = parse_args()
    ns = parse_ns(args.ns)
    cfg = load_config(args.config)
    real_manifest = Path(cfg["data"]["real_manifest"])
    if not real_manifest.is_absolute():
        real_manifest = _REPO_ROOT / real_manifest
    real_dataset = RealDataset.from_manifest(real_manifest)
    show_progress = not bool(args.no_progress)
    real_by_class: dict[int, Any] = {}
    real_storm_ids_by_class: dict[int, Any] = {}
    for class_id in tqdm(
        real_dataset.class_ids,
        desc="Loading real classes",
        unit="class",
        disable=not show_progress,
    ):
        real_by_class[class_id] = real_dataset.load_bt_by_class(class_id)
        real_storm_ids_by_class[class_id] = real_dataset.storm_ids_by_class(class_id)

    real_counts_by_class = {
        int(c): int(real_dataset.counts_by_class[c]) for c in real_dataset.class_ids
    }
    min_real_count = min(real_counts_by_class.values()) if real_counts_by_class else 0

    metric_cfg = metric_config(cfg)
    per_n: list[dict[str, Any]] = []
    for n in tqdm(
        ns,
        desc="Sweeping n_per_class",
        unit="n",
        disable=not show_progress,
    ):
        report = compute_standardization_report(
            real_by_class=real_by_class,
            cfg=metric_cfg,
            generated_n_per_class=int(n),
            seed=int(args.seed),
            real_reference_reps=int(args.real_reference_reps),
            progress=show_progress,
            real_storm_ids_by_class=real_storm_ids_by_class,
            skip_corruption_validation=True,
            skip_bootstrap_calibration=True,
        )
        with_replacement_classes = {
            str(c): bool(int(n) > int(count))
            for c, count in real_counts_by_class.items()
        }
        per_n.append(
            {
                "n_per_class": int(n),
                "with_replacement_classes": with_replacement_classes,
                "any_class_oversampled": any(with_replacement_classes.values()),
                "by_metric": per_metric_summary_view(report),
            }
        )

    payload = {
        "schema": "climategen_eval.n_sweep_standardization.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_study": cfg.get("case_study", "tropical_cyclone_brightness_temperature"),
        "seed": int(args.seed),
        "real_reference_reps": int(args.real_reference_reps),
        "real_counts_by_class": real_counts_by_class,
        "min_real_count_across_classes": int(min_real_count),
        "ns": ns,
        "interpretation": {
            "purpose": (
                "Empirical scaling diagnostic for the standardized-gap denominator. "
                "Reports q05, q50, q95 of the real-vs-real null and the implied "
                "noise-floor half-widths at each n_per_class. The standardized gap "
                "is asymptotically a stable population quantity only if q50(n) and "
                "(q95-q50)(n) decay at the same rate as n grows."
            ),
            "fields": {
                "q50": "Finite-sample bias of the real-vs-real metric distance.",
                "upper_half_width_q95_minus_q50": (
                    "Denominator of the standardized gap for lower-is-better metrics."
                ),
                "lower_half_width_q50_minus_q05": (
                    "Denominator of the standardized gap for higher-is-better metrics."
                ),
                "with_replacement_classes": (
                    "True for classes where n_per_class exceeds the available real "
                    "count; sample-size scaling above this point reflects "
                    "bootstrap-with-replacement, not additional iid information."
                ),
            },
        },
        "per_n": per_n,
    }
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=json_default)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
