#!/usr/bin/env python3
"""Build a deterministic inventory of the frozen TC submission inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "case_studies"
    / "tropical_cyclone_bt"
    / "submission_inventory.json"
)

DATASETS = {
    "real_train": (
        "data/tropical_cyclone_bt/real_train",
        "data/tropical_cyclone_bt/real_train/manifest.json",
    ),
    "real_validation": (
        "data/tropical_cyclone_bt/real_val",
        "data/tropical_cyclone_bt/real_val/manifest.json",
    ),
    "real_test": (
        "data/tropical_cyclone_bt/real_test",
        "data/tropical_cyclone_bt/real_test/manifest.json",
    ),
    "baseline_ztsnr_test": (
        "data/tropical_cyclone_bt/generated/baseline_ztsnr/test/test_baseline_n1000",
        "data/tropical_cyclone_bt/generated/manifests/baseline_ztsnr_test.json",
    ),
}

EVIDENCE = {
    "preregistration": "case_studies/tropical_cyclone_bt/preregistration.yaml",
    "protocol_freeze": "case_studies/tropical_cyclone_bt/protocol_freeze.md",
    "evaluation_config": "case_studies/tropical_cyclone_bt/track1_config.template.yaml",
    "sensitivity_controls": (
        "case_studies/tropical_cyclone_bt/control_results/sensitivity_lanes.json"
    ),
    "statistical_validation": "validation/statistical_validation_v0.2.json",
    "final_report_manifest": (
        "case_studies/tropical_cyclone_bt/final_results/manifest.json"
    ),
    "paper_bundle_manifest": (
        "case_studies/tropical_cyclone_bt/paper_results/manifest.json"
    ),
    "partition_summary": (
        "case_studies/tropical_cyclone_bt/partition_summary.json"
    ),
    "vmax_support_audit": (
        "case_studies/tropical_cyclone_bt/vmax_support_summary.json"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_summary(path: Path) -> tuple[int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def main() -> int:
    datasets = {}
    for identifier, (relative_path, relative_manifest) in DATASETS.items():
        path = ROOT / relative_path
        manifest = ROOT / relative_manifest
        if not path.is_dir():
            raise FileNotFoundError(path)
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        file_count, logical_bytes = _directory_summary(path)
        datasets[identifier] = {
            "path": relative_path,
            "file_count": file_count,
            "logical_bytes": logical_bytes,
            "checksum_manifest": relative_manifest,
            "checksum_manifest_sha256": _sha256(manifest),
        }

    evidence = {}
    for identifier, relative_path in EVIDENCE.items():
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        evidence[identifier] = {
            "path": relative_path,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    payload = {
        "schema": "gauge_climate.tc_submission_inventory.v1",
        "label_contract": {
            "wind_variable": "USA_WIND",
            "averaging_period": "1-minute",
        },
        "datasets": datasets,
        "evidence": evidence,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
