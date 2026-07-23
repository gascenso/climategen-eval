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


def _source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
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


def _verify_submission_inventory(path: Path) -> list[str]:
    if not path.is_file():
        return ["TC submission inventory is missing."]
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    if payload.get("schema") != "gauge_climate.tc_submission_inventory.v1":
        failures.append("TC submission inventory uses an obsolete schema.")
    for identifier, node in payload.get("datasets", {}).items():
        directory = ROOT / str(node.get("path", "missing"))
        manifest = ROOT / str(node.get("checksum_manifest", "missing"))
        if not directory.is_dir():
            failures.append(f"Submission dataset is missing: {identifier}.")
            continue
        files = [item for item in directory.rglob("*") if item.is_file()]
        if len(files) != node.get("file_count"):
            failures.append(f"Submission dataset file count changed: {identifier}.")
        if sum(item.stat().st_size for item in files) != node.get("logical_bytes"):
            failures.append(f"Submission dataset byte count changed: {identifier}.")
        if not manifest.is_file() or _sha256(manifest) != node.get(
            "checksum_manifest_sha256"
        ):
            failures.append(f"Submission dataset manifest changed: {identifier}.")
    for identifier, node in payload.get("evidence", {}).items():
        artifact = ROOT / str(node.get("path", "missing"))
        if not artifact.is_file():
            failures.append(f"Submission evidence is missing: {identifier}.")
        elif artifact.stat().st_size != node.get("bytes") or _sha256(
            artifact
        ) != node.get("sha256"):
            failures.append(f"Submission evidence changed: {identifier}.")
    return failures


def _verify_artifact_manifest(manifest_path: Path, *, files_root: Path) -> list[str]:
    if not manifest_path.is_file():
        return [f"Artifact manifest is missing: {manifest_path.relative_to(ROOT)}"]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for relative, node in payload.get("artifacts", {}).items():
        artifact = files_root / relative
        if not artifact.is_file():
            failures.append(f"Frozen artifact is missing: {artifact.relative_to(ROOT)}")
        elif artifact.stat().st_size != node.get("bytes") or _sha256(
            artifact
        ) != node.get("sha256"):
            failures.append(f"Frozen artifact changed: {artifact.relative_to(ROOT)}")
    return failures


def _verify_frozen_tc_report(path: Path) -> list[str]:
    if not path.is_file():
        return ["Frozen TC evaluation report is missing."]
    report = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    if report.get("schema") != "gauge_climate.evaluation.v2":
        failures.append("Frozen TC evaluation report uses an obsolete schema.")
        return failures
    cells = report.get("cells", [])
    if len(cells) != 280:
        failures.append("Frozen TC evaluation cell count changed.")
    physical = [
        cell
        for cell in cells
        if str(cell.get("metric", "")).startswith("w1:")
        and cell.get("stratum") in {"global", "0", "1", "2", "3", "4", "5", "C4_plus"}
    ]
    departures = [
        cell for cell in physical if cell.get("detection_status") == "DEPARTURE_DETECTED"
    ]
    simultaneous_departures = [
        cell
        for cell in physical
        if cell.get("simultaneous_detection_status") == "DEPARTURE_DETECTED"
    ]
    if len(physical) != 56:
        failures.append("Frozen TC scalar-functional family no longer has 56 cells.")
    if len(departures) != 7 or any(cell.get("stratum") != "global" for cell in departures):
        failures.append("Frozen TC pointwise scalar-functional conclusions changed.")
    if len(simultaneous_departures) != 4 or any(
        cell.get("stratum") != "global" for cell in simultaneous_departures
    ):
        failures.append("Frozen TC simultaneous scalar-functional conclusions changed.")

    profile_metrics = {
        "w1:min_core_bt",
        "w1:cold_frac_220",
        "w1:r_maxgrad",
        "w1:axisym",
        "w1:dav",
        "w1:eye_contrast",
        "w1:p01_bt",
        "radial_profile_mae",
        "radial_log_psd_mse",
        "feature_mmd2",
        "feature_density",
        "feature_coverage_deficit",
        "feature_diversity_departure",
    }
    profile = [
        cell
        for cell in cells
        if cell.get("metric") in profile_metrics
        and cell.get("stratum") in {"global", "0", "1", "2", "3", "4", "5", "C4_plus"}
    ]
    native_profile = [
        cell for cell in profile if cell.get("stratum") in {"0", "1", "2", "3", "4", "5"}
    ]
    if len(profile) != 104 or len(native_profile) != 78:
        failures.append("Frozen TC headline-profile family changed.")
    if any(cell.get("detection_status") == "NOT_EVALUATED" for cell in profile):
        failures.append("A frozen TC headline-profile conclusion is now withheld.")

    regions = report.get("memorization", {}).get("regions", [])
    status_counts = {
        status: sum(region.get("status") == status for region in regions)
        for status in ("COPYING_DETECTED", "FARTHER_FROM_TRAINING", "INCONCLUSIVE")
    }
    if status_counts != {
        "COPYING_DETECTED": 0,
        "FARTHER_FROM_TRAINING": 5,
        "INCONCLUSIVE": 36,
    }:
        failures.append("Frozen TC memorization conclusions changed.")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-survey",
        action="store_true",
        help="Also enforce manuscript literature-survey verification artifacts.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help=(
            "Verify all in-repository software/TC evidence while deliberately "
            "excluding external archive URLs, permissions, and persistent identifiers."
        ),
    )
    args = parser.parse_args()
    failures = []
    prereg = _yaml(ROOT / "case_studies/tropical_cyclone_bt/preregistration.yaml")
    if prereg.get("status") != "FROZEN" or not prereg.get("final_test_run_permitted"):
        failures.append("TC preregistration is not frozen for a final test run.")
    if "PENDING" in json.dumps(prereg):
        failures.append("TC preregistration still contains PENDING fields.")
    control_path = ROOT / "case_studies/tropical_cyclone_bt/control_results/sensitivity_lanes.json"
    control_figure = ROOT / "case_studies/tropical_cyclone_bt/control_results/sensitivity_lanes.pdf"
    if not control_path.exists():
        failures.append("TC Track-1 control artifact is missing.")
    else:
        control = json.loads(control_path.read_text(encoding="utf-8"))
        if control.get("status") != "PASSED":
            failures.append("TC Track-1 response/copyist controls have not passed.")
        metadata = control.get("_metadata", {})
        if metadata.get("test_fields_used") is not False:
            failures.append("TC Track-1 controls do not attest training-only execution.")
        control_sources = {
            "real_train_manifest_sha256": (
                ROOT / "data/tropical_cyclone_bt/real_train/manifest.json"
            ),
            "track1_config_sha256": (
                ROOT / "case_studies/tropical_cyclone_bt/track1_config.template.yaml"
            ),
            "runner_sha256": (ROOT / "case_studies/tropical_cyclone_bt/run_track1_controls.py"),
        }
        for key, source in control_sources.items():
            if not source.is_file() or metadata.get(key) != _sha256(source):
                failures.append(f"TC control provenance mismatch: {key}.")
        if metadata.get("gauge_source_sha256") != _source_tree_sha256(ROOT / "src/gauge_climate"):
            failures.append("TC control provenance mismatch: GAUGE source tree.")
        sensitivity_contract = prereg.get("sensitivity_lanes", {})
        encoded_contract = json.dumps(
            sensitivity_contract, sort_keys=True, separators=(",", ":")
        ).encode()
        contract_hash = hashlib.sha256(encoded_contract).hexdigest()
        if metadata.get("sensitivity_contract_sha256") != contract_hash:
            failures.append("TC control provenance mismatch: sensitivity contract.")
    if not control_figure.is_file():
        failures.append("TC Track-1 control figure is missing.")
    validation_path = ROOT / "validation/statistical_validation_v0.2.json"
    validation_figures = (
        ROOT / "validation/statistical_validation_v0.2.pdf",
        ROOT / "validation/statistical_validation_v0.2.png",
    )
    if not validation_path.is_file():
        failures.append("GAUGE statistical validation artifact is missing.")
    else:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation.get("schema") != "gauge_climate.statistical_validation.v3":
            failures.append("GAUGE statistical validation uses an obsolete schema.")
        if validation.get("status") != "PASSED" or not all(
            validation.get("gates", {}).values()
        ):
            failures.append("GAUGE statistical validation has not passed.")
    if any(not path.is_file() for path in validation_figures):
        failures.append("GAUGE statistical validation paper figure is missing.")
    final_root = ROOT / "case_studies/tropical_cyclone_bt/final_results"
    final_report = final_root / "evaluation.json"
    failures.extend(_verify_frozen_tc_report(final_report))
    failures.extend(
        _verify_artifact_manifest(
            final_root / "manifest.json",
            files_root=final_root,
        )
    )
    paper_root = ROOT / "case_studies/tropical_cyclone_bt/paper_results"
    paper_manifest_path = paper_root / "manifest.json"
    failures.extend(
        _verify_artifact_manifest(
            paper_manifest_path,
            files_root=paper_root,
        )
    )
    if paper_manifest_path.is_file() and final_report.is_file():
        paper_manifest = json.loads(paper_manifest_path.read_text(encoding="utf-8"))
        report = json.loads(final_report.read_text(encoding="utf-8"))
        expected_paper_provenance = {
            "source_report_sha256": _sha256(final_report),
            "config_hash": report.get("metadata", {}).get("config_hash"),
            "source_config_sha256": _sha256(
                ROOT / "case_studies/tropical_cyclone_bt/track1_config.template.yaml"
            ),
            "source_controls_sha256": _sha256(control_path),
            "renderer_sha256": _sha256(ROOT / "scripts/render_tc_track1_paper.py"),
        }
        for key, expected in expected_paper_provenance.items():
            if paper_manifest.get(key) != expected:
                failures.append(f"TC paper-bundle provenance mismatch: {key}.")
    artifacts = _yaml(ROOT / "case_studies/tropical_cyclone_bt/artifacts.yaml")
    if not args.local_only:
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
        failures.append("Absolute local symlinks remain: " + ", ".join(absolute_symlinks[:8]))
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
            survey_checksums = json.loads(survey_manifest.read_text(encoding="utf-8"))
            survey_source = ROOT / "docs/methods/genAI_climate_eval_survey.xlsx"
            if not survey_source.is_file():
                failures.append("Survey source workbook is missing.")
            elif _sha256(survey_source) != survey_checksums.get("source_sha256"):
                failures.append(
                    "Survey source workbook checksum does not match its export manifest."
                )
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
    failures.extend(
        _verify_submission_inventory(
            ROOT / "case_studies/tropical_cyclone_bt/submission_inventory.json"
        )
    )
    if failures:
        scope = (
            "paper"
            if args.include_survey
            else ("local software/TC" if args.local_only else "software/TC")
        )
        print(f"GAUGE {scope} release gates: FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    scope = (
        "paper"
        if args.include_survey
        else ("local software/TC" if args.local_only else "software/TC")
    )
    print(f"GAUGE {scope} release gates: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
