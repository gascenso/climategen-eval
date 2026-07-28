#!/usr/bin/env python3
"""Fail-closed consistency audit for the GAUGE v0.2 GMD submission.

The audit checks that the manuscript and supplement use the empirical
reference-calibration contract and that their generated evidence comes from the
canonical schema-v3 report.  Literature-review reconstruction is an optional,
user-owned gate and is checked only with ``--include-survey``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GMD = ROOT / "manuscript" / "gmd"
MAIN = GMD / "manuscript.tex"
SUPPLEMENT = GMD / "supplement.tex"
BIB = GMD / "references.bib"
REPORT = ROOT / "case_studies" / "tropical_cyclone_bt" / "final_results" / "evaluation.json"
CONTROLS = (
    ROOT / "case_studies" / "tropical_cyclone_bt" / "control_results" / "sensitivity_lanes.json"
)
VALIDATION = ROOT / "validation" / "statistical_validation_v0.2.json"
WIND_AUDIT = ROOT / "outputs" / "tropical_cyclone_bt" / "audits" / "continuous_wind_response.json"
FEATURE_AUDIT = (
    ROOT / "outputs" / "tropical_cyclone_bt" / "audits" / "feature_space_robustness.json"
)
GENERATED_MANIFEST = GMD / "generated_artifacts_manifest.json"
SURVEY = ROOT / "survey" / "exports" / "summary.json"
REFERENCE_CANDIDATES = GMD / "reference_candidates.json"
SURVEY_REFERENCE_OVERRIDES = {"survey23": 2, "survey26": 1}

NATIVE_CONDITIONS = {"0", "1", "2", "3", "4", "5"}
PROFILE_METRICS = {
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
ALLOWED_REFERENCE_STATUSES = {
    "WITHIN_REFERENCE_RANGE",
    "ABOVE_REFERENCE_RANGE",
    "REFERENCE_UNRESOLVED",
    "NOT_EVALUATED",
}
ALLOWED_CONTROL_STATUSES = {
    "CONTROL_VALIDATED",
    "RESPONSIVE_BELOW_REFERENCE_RANGE",
    "CONTROL_FAILED",
    "CONTROL_NOT_REGISTERED",
}
OBSOLETE_PROSE_PATTERNS = {
    r"\bDEPARTURE_DETECTED\b": "obsolete testing status",
    r"\bNO_DEPARTURE_DETECTED\b": "obsolete testing status",
    r"\bBonferroni\b": "retired multiplicity verdict",
    r"\bfalse[- ]positive(?: rate| control| behaviour)?\b": "testing claim",
    r"\bdetection power\b": "testing claim",
    r"\bindistinguishab": "population-equality claim",
    r"\b13\s+(?:of|/)\s*78\b": "aggregate result count",
    r"\b42\s+(?:of|/)\s*42\b": "aggregate result count",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def _labels(source: str) -> set[str]:
    labels = re.findall(r"\\label\{([^}]+)\}", source)
    require(len(labels) == len(set(labels)), "duplicate LaTeX labels")
    require(all(" " not in label for label in labels), "spaces in LaTeX labels")
    return set(labels)


def _citation_keys(source: str) -> set[str]:
    groups = re.findall(
        r"\\(?:cite[tp]?|nocite)(?:\[[^\]]*\]){0,2}\{([^}]+)\}",
        source,
    )
    return {key.strip() for group in groups for key in group.split(",")}


def _included_figures(source: str) -> list[str]:
    return re.findall(r"\\includegraphics(?:\[[^\]]+\])?\{([^}]+)\}", source)


def audit_latex(main: str, supplement: str, bib: str) -> None:
    require(
        r"\documentclass[gmd, manuscript]{copernicus}" in main,
        "main manuscript uses the wrong GMD class",
    )
    require(
        r"\documentclass[gmd, manuscript]{copernicus}" in supplement,
        "supplement uses the wrong GMD class",
    )
    for source_name, source in (("main", main), ("supplement", supplement)):
        require(r"\usepackage" not in source, f"{source_name} adds a package")
        require("??" not in source, f"{source_name} contains an unresolved marker")
        require("TODO" not in source and "TBD" not in source, f"{source_name} is unfinished")
        for pattern, description in OBSOLETE_PROSE_PATTERNS.items():
            require(
                re.search(pattern, source, flags=re.IGNORECASE) is None,
                f"{source_name} contains {description}: {pattern}",
            )

    main_labels = _labels(main)
    supplement_labels = _labels(supplement)
    require(main_labels.isdisjoint(supplement_labels), "labels overlap across PDF files")
    for source_name, source, labels in (
        ("main", main, main_labels),
        ("supplement", supplement, supplement_labels),
    ):
        references = set(re.findall(r"\\(?:ref|pageref)\{([^}]+)\}", source))
        require(
            references <= labels,
            f"{source_name} has undefined references: {sorted(references - labels)}",
        )

    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = _citation_keys(main) | _citation_keys(supplement)
    require(cited <= bib_keys, f"undefined citation keys: {sorted(cited - bib_keys)}")
    require(len(bib_keys) >= 90, "unexpectedly small bibliography")

    main_figures = _included_figures(main)
    supplement_figures = _included_figures(supplement)
    require(len(main_figures) == 7, f"expected 7 main figures, found {len(main_figures)}")
    require(
        len(supplement_figures) == 22,
        f"expected 22 supplement figures, found {len(supplement_figures)}",
    )
    missing = [name for name in (*main_figures, *supplement_figures) if not (GMD / name).is_file()]
    require(not missing, f"missing manuscript figures: {missing}")

    require(r"\Author[1,2]{Andrea}{Castelletti}" in main, "co-author is missing")
    require(
        r"\Author[1][guido.ascenso@cmcc.it]{Guido}{Ascenso}" in main,
        "Guido's CMCC author record is missing",
    )
    require("baseline_ztsnr" not in main, "internal checkpoint name leaked into prose")
    require("downstream" not in main.lower(), "obsolete downstream-task language remains")
    require(r"\maketitle" not in supplement, "supplement must not have a title page")
    require(
        main.index(r"\conclusions") < main.index(r"\codedataavailability"),
        "code/data availability must follow the conclusions",
    )
    require(
        main.index(r"\codedataavailability") < main.index(r"\authorcontribution"),
        "author contributions must follow code/data availability",
    )
    require(
        "not a noise floor" in main.lower() and "not a population-equality" in main.lower(),
        "the empirical-reference limitations are not explicit",
    )
    require(
        "unpaired" in main.lower() and "two-dimensional" in main.lower(),
        "the v0.2 scope is not explicit",
    )


def audit_report() -> dict[str, Any]:
    report = load(REPORT)
    require(report.get("schema") == "gauge_climate.evaluation.v3", "wrong report schema")
    metadata = report.get("metadata", {})
    require(metadata.get("package_version") == "0.2.0", "wrong GAUGE report version")
    cells = report.get("cells", [])
    require(
        "C4_plus" not in {str(cell.get("stratum")) for cell in cells},
        "obsolete C4+ condition remains",
    )
    native = [
        cell
        for cell in cells
        if str(cell.get("stratum")) in NATIVE_CONDITIONS and cell.get("metric") in PROFILE_METRICS
    ]
    require(len(native) == 78, f"native profile has {len(native)} rather than 78 results")
    require(
        {cell.get("metric") for cell in native} == PROFILE_METRICS,
        "native metric registry changed",
    )
    require(
        {str(cell.get("stratum")) for cell in native} == NATIVE_CONDITIONS,
        "native conditions changed",
    )
    required_fields = {
        "estimate",
        "comparison_low",
        "comparison_high",
        "reference_q50",
        "reference_q95",
        "reference_q50_mc_low",
        "reference_q50_mc_high",
        "reference_q95_mc_low",
        "reference_q95_mc_high",
        "reference_upper_boundary",
        "reference_upper_mc_low",
        "reference_upper_mc_high",
        "reference_percentile",
        "reference_gap",
        "reference_gap_mc_low",
        "reference_gap_mc_high",
        "n_eval",
        "n_real",
        "n_generated",
        "n_blocks",
        "estimand",
        "provenance",
        "warnings",
    }
    for cell in native:
        missing = required_fields - set(cell)
        require(
            not missing,
            f"{cell.get('stratum')}:{cell.get('metric')} lacks {sorted(missing)}",
        )
        require(
            cell.get("reference_status") in ALLOWED_REFERENCE_STATUSES,
            "obsolete reference status in report",
        )
        require(
            cell.get("control_status") in ALLOWED_CONTROL_STATUSES,
            "obsolete control status in report",
        )
        require("detection_status" not in cell, "obsolete testing field in report")
        require(
            "simultaneous_detection_status" not in cell,
            "obsolete multiplicity field in report",
        )
        require(cell.get("estimand") == "block_balanced", "TC estimand changed")

    psd = [cell for cell in native if cell["metric"] == "radial_log_psd_mse"]
    require({cell["units"] for cell in psd} == {"dex²"}, "PSD units are not dex²")
    mmd = [cell for cell in native if cell["metric"] == "feature_mmd2"]
    require(
        all(float(cell["estimate"]) >= -1e-12 for cell in mmd),
        "biased MMD² is negative",
    )
    require(
        all(cell.get("provenance", {}).get("estimator") == "biased_v_statistic" for cell in mmd),
        "MMD² estimator provenance changed",
    )

    c5 = [cell for cell in native if str(cell["stratum"]) == "5"]
    require({int(cell["n_real"]) for cell in c5} == {85}, "C5 frame count changed")
    require({int(cell["n_blocks"]) for cell in c5} == {22}, "C5 storm count changed")

    mixture = metadata.get("analysis", {}).get("global_label_mixture", {})
    require(mixture.get("matched") is False, "global mixture unexpectedly matches")
    require(
        mixture.get("policy") == "as_provided",
        "global comparison is not marked as-provided",
    )

    memorization = report.get("memorization") or {}
    require(
        memorization.get("n_train_reference") == 51_890,
        "memorization does not use the complete training archive",
    )
    require(
        memorization.get("copyist_positive_control", {}).get("passed") is True,
        "copyist control failed",
    )
    require(
        memorization.get("status")
        in {
            "LOCALIZED_CLOSENESS_OBSERVED",
            "FARTHER_FROM_TRAINING",
            "ALL_CENTRAL_RANGES_OVERLAP_ZERO",
            "INSUFFICIENT_LOCAL_SUPPORT",
        },
        "memorization uses an obsolete global status",
    )
    return report


def audit_controls_and_validation() -> None:
    controls = load(CONTROLS)
    require(
        controls.get("schema") == "gauge_climate.tc_bt_metric_controls.v2",
        "wrong control schema",
    )
    require(controls.get("status") == "COMPLETE", "controls are incomplete")
    summary = controls.get("summary", {})
    require(summary.get("lane_count") == 17, "expected 17 control contracts")
    require(not summary.get("failed_lanes"), "at least one control contract failed")
    require(
        controls.get("_metadata", {}).get("test_fields_used") is False,
        "controls are not training-only",
    )
    feature_hash = (
        controls.get("_metadata", {}).get("registered_feature_space", {}).get("provenance_hash")
    )
    require(feature_hash, "controls do not record their registered feature space")
    report = load(REPORT)
    report_feature_hashes = {
        cell.get("provenance", {}).get("feature_space_hash")
        for cell in report.get("cells", [])
        if cell.get("metric") == "feature_mmd2"
    }
    require(
        report_feature_hashes == {feature_hash},
        "feature controls and report use different training-fitted representations",
    )
    serialized = json.dumps(controls)
    require(
        "DETECTABLE_AT_MAX_SEVERITY" not in serialized
        and "BELOW_REFERENCE_AT_MAX_SEVERITY" not in serialized,
        "controls contain obsolete resolution labels",
    )

    validation = load(VALIDATION)
    require(
        validation.get("schema") == "gauge_climate.statistical_validation.v4",
        "wrong validation schema",
    )
    require(validation.get("status") == "PASSED", "validation gates failed")
    gates = validation.get("gates", {})
    require(gates and all(gates.values()), "not every validation gate passed")
    require(
        validation.get("metric_family_validation", {}).get("mmd2_non_negative"),
        "synthetic MMD² non-negativity gate failed",
    )


def audit_secondary_analyses() -> None:
    wind = load(WIND_AUDIT)
    require(
        wind.get("schema") == "gauge_climate.tc_bt_continuous_wind_audit.v1",
        "wrong continuous-wind audit schema",
    )
    require(
        wind.get("synthetic_validation", {}).get("passed") is True,
        "continuous-wind slope engine failed its synthetic validation",
    )
    require(len(wind.get("functionals", {})) == 7, "wind audit lacks a functional")

    feature = load(FEATURE_AUDIT)
    require(
        feature.get("schema") == "gauge_climate.tc_bt_feature_robustness.v1",
        "wrong feature-robustness schema",
    )
    require(
        feature.get("status") == "POST_HOC_ROBUSTNESS_ANALYSIS",
        "feature robustness is not labelled post hoc",
    )
    require(len(feature.get("variants", {})) == 12, "feature audit lacks a variant")


def audit_generated_artifacts() -> None:
    manifest = load(GENERATED_MANIFEST)
    require(
        manifest.get("schema") == "gauge_climate.gmd_generated_artifacts.v1",
        "wrong manuscript-artifact manifest schema",
    )
    artifacts = manifest.get("artifacts", {})
    require(
        len(artifacts) == 30,
        "expected 7 main and 23 staged supplement figures",
    )
    for name, node in artifacts.items():
        require((GMD / name).is_file(), f"staged figure is missing: {name}")
        require(
            node.get("staged_sha256") and int(node.get("bytes", 0)) > 0,
            f"staged figure provenance is incomplete: {name}",
        )


def audit_survey() -> None:
    """Optional user-owned review-corpus and bibliography gate."""

    summary = load(SURVEY)
    require(summary.get("n_coded") == 69, "survey denominator changed")
    candidates = load(REFERENCE_CANDIDATES)["results"]
    bib = BIB.read_text(encoding="utf-8")
    entries = {
        key: body
        for key, body in re.findall(
            r"@\w+\{([^,]+),(.*?)(?=\n@\w+\{|\Z)",
            bib,
            flags=re.DOTALL,
        )
    }
    for index in range(1, 70):
        key = f"survey{index:02d}"
        selected = candidates[key]["candidates"][SURVEY_REFERENCE_OVERRIDES.get(key, 0)]
        require(candidates[key]["error"] is None, f"reference lookup failed: {key}")
        require(selected["title_similarity"] >= 0.9, f"weak title match: {key}")
        require(selected["doi"], f"survey reference lacks DOI: {key}")
        require(key in entries, f"survey reference absent from BibTeX: {key}")
        match = re.search(r"\bdoi\s*=\s*\{([^}]+)\}", entries[key], re.IGNORECASE)
        require(match is not None, f"survey BibTeX entry lacks DOI: {key}")
        require(
            match.group(1).replace(r"\_", "_").lower() == selected["doi"].lower(),
            f"survey DOI mismatch: {key}",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-survey",
        action="store_true",
        help="Also enforce the user-owned literature-review reconstruction gate.",
    )
    args = parser.parse_args()

    main_source = MAIN.read_text(encoding="utf-8")
    supplement_source = SUPPLEMENT.read_text(encoding="utf-8")
    bibliography = BIB.read_text(encoding="utf-8")
    audit_latex(main_source, supplement_source, bibliography)
    audit_report()
    audit_controls_and_validation()
    audit_secondary_analyses()
    audit_generated_artifacts()
    if args.include_survey:
        audit_survey()

    print("GMD manuscript audit: PASSED")
    print("  empirical-calibration terminology and scope: consistent")
    print("  schema-v3 TC evidence and training-only controls: consistent")
    print("  validation, wind-response, and feature-robustness audits: consistent")
    print("  7 main and 22 supplement figures: staged with provenance")
    if args.include_survey:
        print("  user-owned structured-review gate: passed")
    else:
        print("  user-owned structured-review gate: deferred")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
