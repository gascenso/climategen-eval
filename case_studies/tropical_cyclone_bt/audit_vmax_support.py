#!/usr/bin/env python3
"""Audit generated conditioning-wind support against corrected real labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


BANKS = {
    "track1_test": {
        "path": "generated/baseline_ztsnr/test/test_baseline_n1000",
        "reference_split": "test",
        "role": "Track-1 held-out generation",
    },
    "cnn_train": {
        "path": "generated/baseline_ztsnr/train/aug_guidance_train",
        "reference_split": "train",
        "role": "tc-diffusion evaluator synthetic augmentation",
    },
}


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32)
    unique, counts = np.unique(values, return_counts=True)
    order = np.argsort(counts)[::-1]
    return {
        "n": int(values.size),
        "n_unique": int(unique.size),
        "min_kt": float(values.min()),
        "q25_kt": float(np.quantile(values, 0.25)),
        "median_kt": float(np.median(values)),
        "q75_kt": float(np.quantile(values, 0.75)),
        "max_kt": float(values.max()),
        "mean_kt": float(values.mean()),
        "sd_kt": float(values.std()),
        "most_common": {
            "wind_kt": float(unique[order[0]]),
            "count": int(counts[order[0]]),
            "share": float(counts[order[0]] / values.size),
        },
        "support_kt": [float(value) for value in unique],
    }


def _total_variation(left: np.ndarray, right: np.ndarray) -> float:
    support = np.union1d(left, right)
    left_probability = np.asarray([np.mean(left == value) for value in support])
    right_probability = np.asarray([np.mean(right == value) for value in support])
    return float(0.5 * np.sum(np.abs(left_probability - right_probability)))


def _generated_winds(root: Path, manifest: dict[str, Any], class_id: str) -> np.ndarray:
    shards = manifest.get("shards_by_class", {}).get(class_id, [])
    if not shards:
        raise ValueError(f"Generated bank {root} has no class {class_id} shards.")
    return np.concatenate([
        np.asarray(np.load(root / str(shard["wind_kt"]), mmap_mode="r"))
        for shard in shards
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/tropical_cyclone_bt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "case_studies/tropical_cyclone_bt/vmax_support_summary.json"
        ),
    )
    args = parser.parse_args()
    results: dict[str, Any] = {}
    majority_cells = []
    for bank_id, specification in BANKS.items():
        bank_root = args.data / str(specification["path"])
        bank_manifest_path = bank_root / "manifest.json"
        bank_manifest = _json(bank_manifest_path)
        if bank_manifest.get("run_name") != "baseline_ztsnr":
            raise ValueError(f"Unexpected generator for {bank_root}.")
        split = str(specification["reference_split"])
        real_root = args.data / f"real_{split}"
        real_manifest_path = real_root / "manifest.json"
        real_manifest = _json(real_manifest_path)
        if real_manifest.get("wind_source") != "USA_WIND" or real_manifest.get(
            "wind_averaging_period"
        ) != "1-minute":
            raise ValueError(f"Reference split {split} lacks corrected wind provenance.")
        real_wind = np.asarray(
            np.load(real_root / str(real_manifest["wind_kt"]), mmap_mode="r")
        )
        real_class = np.asarray(
            np.load(real_root / str(real_manifest["labels"]), mmap_mode="r")
        )
        class_results = {}
        for class_id in sorted(bank_manifest.get("shards_by_class", {}), key=int):
            generated = _generated_winds(bank_root, bank_manifest, class_id)
            reference = real_wind[real_class == int(class_id)]
            generated_stats = _stats(generated)
            reference_stats = _stats(reference)
            if generated_stats["most_common"]["share"] > 0.5:
                majority_cells.append({
                    "bank": bank_id,
                    "class_id": int(class_id),
                    **generated_stats["most_common"],
                })
            class_results[class_id] = {
                "generated": generated_stats,
                "real_reference": reference_stats,
                "generated_support_subset_of_reference": bool(
                    set(generated_stats["support_kt"])
                    <= set(reference_stats["support_kt"])
                ),
                "discrete_distribution_total_variation": _total_variation(
                    generated, reference
                ),
            }
        results[bank_id] = {
            "role": specification["role"],
            "bank_manifest": str(bank_manifest_path),
            "bank_manifest_sha256": _sha256(bank_manifest_path),
            "reference_manifest": str(real_manifest_path),
            "reference_manifest_sha256": _sha256(real_manifest_path),
            "classes": class_results,
        }
        if bank_id == "cnn_train":
            expected_counts = {
                class_id: node["generated"]["n"]
                for class_id, node in class_results.items()
            }
            evaluator_runs = []
            for seed in (42, 43, 44):
                evidence_path = (
                    args.data
                    / "source_manifests"
                    / f"runs__cm_E3_s{seed}__synthetic_augmentation.json"
                )
                evidence = _json(evidence_path)
                observed_counts = {
                    str(key): int(value)
                    for key, value in evidence.get("per_class_used", {}).items()
                }
                if evidence.get("bank_name") != bank_manifest.get("bank_name"):
                    raise ValueError(
                        f"Evaluator run s{seed} used a different sample bank."
                    )
                if observed_counts != expected_counts:
                    raise ValueError(
                        f"Evaluator run s{seed} counts differ from the audited bank."
                    )
                evaluator_runs.append({
                    "run": f"cm_E3_s{seed}",
                    "evidence": str(evidence_path),
                    "evidence_sha256": _sha256(evidence_path),
                    "per_class_used": observed_counts,
                })
            results[bank_id]["tc_diffusion_evaluator_runs"] = evaluator_runs
    payload = {
        "schema": "gauge_climate.tc_bt_vmax_support_audit.v1",
        "wind_source": "USA_WIND",
        "wind_averaging_period": "1-minute",
        "interpretation": (
            "Repeated conditioning values are assessed against the corresponding "
            "corrected real discrete distribution; repetition alone is not evidence "
            "of generator conditioning collapse."
        ),
        "generated_majority_value_cells": majority_cells,
        "banks": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
