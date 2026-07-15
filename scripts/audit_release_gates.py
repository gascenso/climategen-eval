#!/usr/bin/env python3
"""Fail closed on the scientific and artifact gates required for a release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _yaml(path: Path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksum_manifest(
    manifest_path: Path, *, files_root: Path, mapping_key: str
) -> list[str]:
    if not manifest_path.exists():
        return [f"Checksum manifest is missing: {manifest_path.relative_to(ROOT)}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for relative, node in manifest.get(mapping_key, {}).items():
        expected = node.get("sha256") if isinstance(node, dict) else node
        path = files_root / relative
        if not path.is_file():
            failures.append(f"Checksum target is missing: {path.relative_to(ROOT)}")
        elif _sha256(path) != expected:
            failures.append(f"Checksum mismatch: {path.relative_to(ROOT)}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-survey",
        action="store_true",
        help="Also enforce manuscript literature-survey verification artifacts.",
    )
    args = parser.parse_args()
    failures = []
    prereg = _yaml(ROOT / "case_studies/tropical_cyclone_bt/preregistration.yaml")
    if prereg.get("status") != "FROZEN" or not prereg.get("final_test_run_permitted"):
        failures.append("TC preregistration is not frozen for a final test run.")
    if "PENDING" in json.dumps(prereg):
        failures.append("TC preregistration still contains PENDING fields.")
    artifacts = _yaml(ROOT / "case_studies/tropical_cyclone_bt/artifacts.yaml")
    if not artifacts.get("release_ready"):
        failures.append("TC artifacts are not archived and release-ready.")
    if "PENDING" in json.dumps(artifacts):
        failures.append("TC artifact registry still contains PENDING fields.")
    fixture = artifacts.get("small_fixture", {})
    fixture_manifest = ROOT / str(fixture.get("manifest", "missing"))
    failures.extend(
        _verify_checksum_manifest(
            fixture_manifest,
            files_root=ROOT / str(fixture.get("path", "missing")),
            mapping_key="files",
        )
    )
    for bank in artifacts.get("generated_sample_banks", []):
        bank_id = str(bank.get("id", "missing"))
        checksum_path = ROOT / str(bank.get("sha256_manifest", "missing"))
        bank_root = ROOT / str(bank.get("local_path", "missing"))
        bank_failures = _verify_checksum_manifest(
            checksum_path,
            files_root=bank_root,
            mapping_key="files",
        )
        failures.extend(f"{bank_id}: {failure}" for failure in bank_failures)
    absolute_symlinks = []
    for path in ROOT.rglob("*"):
        if path.is_symlink() and path.readlink().is_absolute():
            absolute_symlinks.append(str(path.relative_to(ROOT)))
    if absolute_symlinks:
        failures.append(
            "Absolute local symlinks remain: " + ", ".join(absolute_symlinks[:8])
        )
    if args.include_survey:
        summary_path = ROOT / "survey/exports/summary.json"
        if not summary_path.exists():
            failures.append("Survey summary has not been generated.")
        else:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not summary.get("verification", {}).get("headline_use_permitted"):
                failures.append("Survey verification does not permit headline use.")
        survey_manifest = ROOT / "survey/exports/manifest.json"
        failures.extend(
            _verify_checksum_manifest(
                survey_manifest,
                files_root=ROOT / "survey/exports",
                mapping_key="outputs",
            )
        )
        if survey_manifest.exists():
            survey_checksums = json.loads(
                survey_manifest.read_text(encoding="utf-8")
            )
            survey_source = ROOT / "docs/methods/genAI_climate_eval_survey.xlsx"
            if not survey_source.is_file():
                failures.append("Survey source workbook is missing.")
            elif _sha256(survey_source) != survey_checksums.get("source_sha256"):
                failures.append(
                    "Survey source workbook checksum does not match its export manifest."
                )
    validation_path = ROOT / "validation/statistical_validation_v0.1.json"
    if not validation_path.exists():
        failures.append("Statistical validation artifact is missing.")
    else:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if not validation.get("passed"):
            failures.append("Statistical validation gate did not pass.")
    for split in ("real_train", "real_val", "real_test"):
        manifest = ROOT / "data/tropical_cyclone_bt" / split / "manifest.json"
        if not manifest.exists():
            failures.append(f"TC {split} manifest is missing.")
            continue
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("wind_source") != "USA_WIND":
            failures.append(f"TC {split} does not declare USA_WIND provenance.")
        if payload.get("wind_averaging_period") != "1-minute":
            failures.append(f"TC {split} does not declare 1-minute winds.")
    if not (ROOT / "case_studies/tropical_cyclone_bt/partition_summary.json").exists():
        failures.append("TC storm/frame/class partition summary is missing.")
    if failures:
        scope = "paper" if args.include_survey else "software/TC"
        print(f"GAUGE {scope} release gates: FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    scope = "paper" if args.include_survey else "software/TC"
    print(f"GAUGE {scope} release gates: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
