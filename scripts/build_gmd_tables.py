#!/usr/bin/env python3
"""Generate evidence tables included by the GMD manuscript."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LABELS = {
    "w1:min_core_bt": "Core minimum BT W1",
    "w1:cold_frac_220": "Cold-cloud area W1",
    "w1:r_maxgrad": "Max-gradient radius W1",
    "w1:axisym": "Axisymmetry W1",
    "w1:dav": "Deviation-angle variance W1",
    "w1:eye_contrast": "Eye contrast W1",
    "w1:p01_bt": "BT lower-tail W1",
    "radial_profile_mae": "Radial BT profile MAE",
    "radial_log_psd_mse": "Radial power-spectrum MSE",
    "feature_mmd2": "Feature MMD$^2$",
    "feature_density": "Feature-density departure",
    "feature_coverage_deficit": "Feature-coverage deficit",
    "feature_diversity_departure": "Feature-diversity departure",
}


SURVEY_ROWS = (
    ("Held-out test set stated", "held_out_test_set"),
    ("Conditional fidelity", "conditional_fidelity"),
    ("Extremes or tails", "extremes_tails"),
    ("Quantitative distributional comparison", "quantitative_distributional"),
    ("Quantified spatial spectra", "spatial_spectra"),
    ("Probabilistic calibration", "probabilistic_calibration"),
    ("Metric uncertainty", "metric_uncertainty"),
    ("Diversity or coverage", "diversity_coverage"),
    ("Real-vs-real reference", "real_reference"),
    ("Metric-validity control", "metric_validity"),
    ("Physical admissibility", "physical_admissibility"),
    ("Memorization or nearest-neighbour check", "memorization"),
)

FUNCTIONAL_ROWS = (
    ("w1:min_core_bt", "Core minimum BT"),
    ("w1:cold_frac_220", "Cold-cloud area"),
    ("w1:r_maxgrad", "Max-gradient radius"),
    ("w1:axisym", "Axisymmetry"),
    ("w1:dav", "Deviation-angle variance"),
    ("w1:eye_contrast", "Eye contrast"),
    ("w1:p01_bt", "BT lower tail"),
)

CLASS_LABELS = {
    "0": "TS",
    "1": "C1",
    "2": "C2",
    "3": "C3",
    "4": "C4",
    "5": "C5",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native-values",
        type=Path,
        default=Path(
            "case_studies/tropical_cyclone_bt/paper_results/main/table_2_native_metric_values.csv"
        ),
    )
    parser.add_argument(
        "--survey-summary",
        type=Path,
        default=Path("survey/exports/summary.json"),
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path("outputs/tropical_cyclone_bt/gauge_track1/evaluation.json"),
    )
    parser.add_argument(
        "--partitions",
        type=Path,
        default=Path(
            "case_studies/tropical_cyclone_bt/paper_results/appendix/table_A3_partition_summary.csv"
        ),
    )
    parser.add_argument(
        "--vmax-support",
        type=Path,
        default=Path(
            "case_studies/tropical_cyclone_bt/paper_results/appendix/table_A4_vmax_support.csv"
        ),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("validation/statistical_validation_v0.2.json"),
    )
    parser.add_argument(
        "--controls",
        type=Path,
        default=Path("case_studies/tropical_cyclone_bt/control_results/sensitivity_lanes.json"),
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=Path("data/tropical_cyclone_bt/real_val/manifest.json"),
    )
    parser.add_argument(
        "--continuous-wind",
        type=Path,
        default=Path("outputs/tropical_cyclone_bt/audits/continuous_wind_response.json"),
    )
    parser.add_argument(
        "--feature-robustness",
        type=Path,
        default=Path("outputs/tropical_cyclone_bt/audits/feature_space_robustness.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("manuscript/gmd/tables"),
    )
    return parser.parse_args()


def _escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def _number(value: str | float | None, *, significant: int = 3) -> str:
    if value is None or str(value).strip().lower() in {"", "none", "nan"}:
        return "--"
    number = float(value)
    if number == 0:
        return "0"
    if abs(number) < 0.001 or abs(number) >= 10000:
        return f"{number:.{significant - 1}e}"
    return f"{number:.{significant}g}"


def _unit(value: str) -> str:
    return {
        "%": r"\%",
        "deg²": r"deg$^2$",
        "dex²": r"dex$^2$",
        "MMD²": r"MMD$^2$",
    }.get(value, _escape(value))


def _integer(value: str | int) -> str:
    return f"{int(value):,}".replace(",", r"\,")


def _macro(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def _percent(value: float, *, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}\\,\\%"


def _result_macros(
    report_source: Path,
    validation_source: Path,
    controls_source: Path,
    validation_manifest_source: Path,
    continuous_wind_source: Path,
    feature_robustness_source: Path,
    output: Path,
) -> None:
    """Generate prose-facing numbers from the authoritative machine artifacts."""

    report = json.loads(report_source.read_text(encoding="utf-8"))
    validation = json.loads(validation_source.read_text(encoding="utf-8"))
    controls = json.loads(controls_source.read_text(encoding="utf-8"))
    validation_manifest = json.loads(validation_manifest_source.read_text(encoding="utf-8"))
    continuous_wind = json.loads(continuous_wind_source.read_text(encoding="utf-8"))
    feature_robustness = json.loads(feature_robustness_source.read_text(encoding="utf-8"))
    if report.get("schema") != "gauge_climate.evaluation.v3":
        raise ValueError("Result macros require the authoritative schema-v3 report.")
    if validation.get("schema") != "gauge_climate.statistical_validation.v4":
        raise ValueError("Result macros require the schema-v4 validation report.")
    if controls.get("schema") != "gauge_climate.tc_bt_metric_controls.v2":
        raise ValueError("Result macros require the v0.2 TC control artifact.")
    if continuous_wind.get("schema") != "gauge_climate.tc_bt_continuous_wind_audit.v1":
        raise ValueError("Result macros require the continuous-wind audit artifact.")
    if feature_robustness.get("schema") != "gauge_climate.tc_bt_feature_robustness.v1":
        raise ValueError("Result macros require the feature-robustness artifact.")
    if not continuous_wind["synthetic_validation"].get("passed"):
        raise ValueError("Continuous-wind synthetic validation did not pass.")

    native = [
        cell
        for cell in report["cells"]
        if cell["stratum"] in CLASS_LABELS and cell["metric"] in LABELS
    ]
    c5 = [cell for cell in native if cell["stratum"] == "5"]
    generated_per_class = {int(cell["n_generated"]) for cell in native}
    if len(generated_per_class) != 1:
        raise ValueError("Native TC report cells do not share one generated-bank size.")
    generated_n = next(iter(generated_per_class))
    c5_frames = {int(cell["n_real"]) for cell in c5}
    c5_blocks = {int(cell["n_blocks"]) for cell in c5}
    if len(c5_frames) != 1 or len(c5_blocks) != 1:
        raise ValueError("Category 5 support is inconsistent across report cells.")
    mixture = report["metadata"]["analysis"]["global_label_mixture"]
    real_counts = [int(value) for value in mixture["real_counts"]]
    if len(real_counts) != 6 or sum(real_counts) <= 0:
        raise ValueError("Global TC mixture audit does not contain six real counts.")

    regimes = validation["regimes"]
    gaussian_rates = [
        float(node["same_population"]["boundary_crossing_rate"]) for node in regimes.values()
    ]
    shape = validation["shape_same_population"]
    standard_shift = {
        regime_id: next(
            row
            for row in regimes[regime_id]["location_shift"]
            if abs(float(row["effect_sd"]) - 0.55) < 1e-12
        )
        for regime_id in ("balanced_80", "c5_like_22", "minimum_20")
    }
    interval_coverages = [
        float(node["target_coverage"])
        for node in validation["independent_seed_interval_validation"]["coverage"].values()
    ]
    seed_validation = validation["independent_seed_interval_validation"]
    stability = validation["reference_stability"]
    copy_fraction = controls["copy_fraction"].get("minimum_resolved_fraction")
    copy_fraction_text = "--" if copy_fraction is None else _percent(float(copy_fraction), digits=0)
    data_summary = report["metadata"]["data"]
    train_summary = data_summary["train"]
    test_summary = data_summary["real"]
    memorization = report.get("memorization") or {}
    memorization_training_fields = memorization.get("n_train_reference")
    if memorization_training_fields is None:
        raise ValueError("The TC report lacks the memorization training-reference count.")
    control_reference_fields = controls["copyist"]["sampling"]["n_train_reference"]

    wind_rows = [
        row
        for functional_rows in continuous_wind["functionals"].values()
        for row in functional_rows
    ]

    def _excludes_zero(interval: list[float]) -> bool:
        return float(interval[0]) > 0 or float(interval[1]) < 0

    wind_real_resolved = [
        row for row in wind_rows if _excludes_zero(row["real"]["interval_95_per_10kt"])
    ]
    wind_generated_resolved = [
        row for row in wind_rows if _excludes_zero(row["generated"]["interval_95_per_10kt"])
    ]
    wind_both_resolved = [
        row
        for row in wind_rows
        if _excludes_zero(row["real"]["interval_95_per_10kt"])
        and _excludes_zero(row["generated"]["interval_95_per_10kt"])
    ]
    wind_both_same_sign = [
        row
        for row in wind_both_resolved
        if float(row["real"]["estimate_per_10kt"]) * float(row["generated"]["estimate_per_10kt"])
        > 0
    ]
    wind_c5_real_resolved = [row for row in wind_real_resolved if row["condition"] == "C5"]
    wind_c5_generated_resolved = [
        row for row in wind_generated_resolved if row["condition"] == "C5"
    ]

    feature_rows: dict[str, list[float]] = {
        metric: []
        for metric in (
            "feature_mmd2",
            "feature_density",
            "feature_coverage_deficit",
            "feature_diversity_departure",
        )
    }
    for variant in feature_robustness["variants"].values():
        for class_node in variant["evaluation"]["classes"].values():
            for metric in feature_rows:
                feature_rows[metric].append(float(class_node["metrics"][metric]["reference_gap"]))
    feature_counts = {len(values) for values in feature_rows.values()}
    if len(feature_counts) != 1:
        raise ValueError("Feature robustness metrics have inconsistent comparison counts.")
    feature_comparisons = feature_counts.pop()
    feature_variant_count = len(feature_robustness["variants"])

    lines = [
        "% Generated from schema-v3/v4 machine artifacts; do not edit by hand.",
        _macro("ValidationTrials", _integer(validation["configuration"]["trials"])),
        _macro(
            "ValidationReferenceRepetitions",
            _integer(validation["configuration"]["resamples"]),
        ),
        _macro("GaussianCrossingMaximum", _percent(max(gaussian_rates))),
        _macro(
            "HeavyTailCrossingRate",
            _percent(shape["heavy_tailed"]["boundary_crossing_rate"]),
        ),
        _macro(
            "MultimodalCrossingRate",
            _percent(shape["multimodal"]["boundary_crossing_rate"]),
        ),
        _macro(
            "BalancedShiftCrossingRate",
            _percent(standard_shift["balanced_80"]["boundary_crossing_probability"]),
        ),
        _macro(
            "CfiveLikeShiftCrossingRate",
            _percent(standard_shift["c5_like_22"]["boundary_crossing_probability"]),
        ),
        _macro(
            "MinimumShiftCrossingRate",
            _percent(standard_shift["minimum_20"]["boundary_crossing_probability"]),
        ),
        _macro("IntervalCoverageMinimum", _percent(min(interval_coverages))),
        _macro("IntervalCoverageMaximum", _percent(max(interval_coverages))),
        _macro(
            "IntervalSeedRepetitions",
            _integer(seed_validation["independent_seed_repetitions"]),
        ),
        _macro(
            "IntervalTargetResamples",
            _integer(seed_validation["target_resamples"]),
        ),
        _macro(
            "CorrectBlockReferenceWidth",
            _number(stability["correct_block_reference_width"]),
        ),
        _macro(
            "FrameIIDReferenceWidth",
            _number(stability["frame_iid_reference_width"]),
        ),
        _macro("GeneratedPerClass", _integer(generated_n)),
        _macro("GeneratedTotal", _integer(6 * generated_n)),
        _macro("TrainingFrames", _integer(train_summary["n_samples"])),
        _macro("TrainingStorms", _integer(train_summary["n_blocks"])),
        _macro("ValidationFrames", _integer(validation_manifest["num_samples"])),
        _macro("ValidationStorms", _integer(validation_manifest["num_storms"])),
        _macro("TestFrames", _integer(test_summary["n_samples"])),
        _macro("TestStorms", _integer(test_summary["n_blocks"])),
        _macro(
            "MemorizationTrainingFields",
            _integer(memorization_training_fields),
        ),
        _macro("ControlReferenceFields", _integer(control_reference_fields)),
        _macro("CfiveFrames", _integer(c5_frames.pop())),
        _macro("CfiveStorms", _integer(c5_blocks.pop())),
        _macro("RealTropicalStormShare", _percent(real_counts[0] / sum(real_counts))),
        _macro("RealCategoryFiveShare", _percent(real_counts[5] / sum(real_counts))),
        _macro("ControlLaneCount", _integer(controls["summary"]["lane_count"])),
        _macro("MinimumResolvedCopyFraction", copy_fraction_text),
        _macro("WindSlopeComparisons", _integer(len(wind_rows))),
        _macro("WindRealResolved", _integer(len(wind_real_resolved))),
        _macro("WindGeneratedResolved", _integer(len(wind_generated_resolved))),
        _macro("WindBothResolved", _integer(len(wind_both_resolved))),
        _macro("WindBothSameSign", _integer(len(wind_both_same_sign))),
        _macro("WindCfiveRealResolved", _integer(len(wind_c5_real_resolved))),
        _macro(
            "WindCfiveGeneratedResolved",
            _integer(len(wind_c5_generated_resolved)),
        ),
        _macro("FeatureRepresentationVariants", _integer(feature_variant_count)),
        _macro("FeatureRobustnessComparisons", _integer(feature_comparisons)),
        _macro(
            "FeatureMmdWithinCount",
            _integer(sum(gap <= 1 for gap in feature_rows["feature_mmd2"])),
        ),
        _macro(
            "FeatureDensityAboveCount",
            _integer(sum(gap > 1 for gap in feature_rows["feature_density"])),
        ),
        _macro(
            "FeatureCoverageWithinCount",
            _integer(sum(gap <= 1 for gap in feature_rows["feature_coverage_deficit"])),
        ),
        _macro(
            "FeatureDiversityAboveCount",
            _integer(sum(gap > 1 for gap in feature_rows["feature_diversity_departure"])),
        ),
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def _native_table(source: Path, output: Path) -> None:
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lines = [
        r"\tablehead{",
        r"\hline",
        r"Class & Metric & Model discrepancy [central range] & "
        r"Reference $q_{50}\rightarrow q_{95}$ & Gap & Status \\",
        r"\hline",
        r"}",
        r"\tabletail{\hline}",
        r"\begin{supertabular}{p{0.055\linewidth}p{0.23\linewidth}"
        r"p{0.25\linewidth}p{0.22\linewidth}rr}",
    ]
    for row in rows:
        estimate = _number(row["estimate"])
        low = _number(row["comparison_low"])
        high = _number(row["comparison_high"])
        unit = _unit(row["units"])
        reference = (
            f"{_number(row['reference_q50'])} $\\rightarrow$ "
            f"{_number(row['reference_boundary'])} {unit}"
        )
        discrepancy = f"{estimate} [{low}, {high}] {unit}"
        status = {
            "WITHIN_REFERENCE_RANGE": "within",
            "ABOVE_REFERENCE_RANGE": "above",
            "REFERENCE_UNRESOLVED": "unresolved",
            "NOT_EVALUATED": "not evaluated",
        }[row["reference_status"]]
        lines.append(
            " & ".join(
                (
                    _escape(row["stratum_label"]),
                    LABELS[row["metric"]],
                    discrepancy,
                    reference,
                    _number(row["reference_gap"], significant=2),
                    status,
                )
            )
            + r" \\"
        )
    lines.extend((r"\hline", r"\end{supertabular}", ""))
    output.write_text("\n".join(lines), encoding="utf-8")


def _survey_table(source: Path, output: Path) -> None:
    summary = json.loads(source.read_text(encoding="utf-8"))
    n = int(summary["n_coded"])
    lines = [
        r"\begin{tabular}{lrr}",
        r"\hline",
        r"Criterion & Papers & Percentage \\",
        r"\hline",
    ]
    for label, key in SURVEY_ROWS:
        if key == "held_out_test_set":
            count = int(summary["descriptive_distributions"][key]["yes"])
        else:
            count = int(summary["criteria"][key]["strict_n"])
        lines.append(f"{label} & {count}/{n} & {100 * count / n:.1f}\\,\\% \\\\")
    lines.extend((r"\hline", r"\end{tabular}", ""))
    output.write_text("\n".join(lines), encoding="utf-8")


def _global_composition_table(source: Path, output: Path) -> None:
    report = json.loads(source.read_text(encoding="utf-8"))
    index = {
        cell["metric"]: cell
        for cell in report["cells"]
        if cell["stratum"] == "global" and cell["metric"] in dict(FUNCTIONAL_ROWS)
    }
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Physical quantity & Model & Reference $q_{95}$ & Gap & Status \\",
        r"\hline",
    ]
    for metric, label in FUNCTIONAL_ROWS:
        cell = index[metric]
        unit = _unit(cell["units"])
        lines.append(
            " & ".join(
                (
                    label,
                    f"{_number(str(cell['estimate']))} {unit}",
                    f"{_number(str(cell['reference_q95']))} {unit}",
                    _number(str(cell["reference_gap"]), significant=2),
                    {
                        "WITHIN_REFERENCE_RANGE": "within",
                        "ABOVE_REFERENCE_RANGE": "above",
                        "REFERENCE_UNRESOLVED": "unresolved",
                        "NOT_EVALUATED": "not evaluated",
                    }[cell["reference_status"]],
                )
            )
            + r" \\"
        )
    lines.extend((r"\hline", r"\end{tabular}", ""))
    output.write_text("\n".join(lines), encoding="utf-8")


def _partition_table(source: Path, output: Path) -> None:
    with source.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    indexed = {(row["split"], row["class_name"]): row for row in records}
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\tophline",
        r"& \multicolumn{2}{c}{Train} & \multicolumn{2}{c}{Validation} &",
        r"\multicolumn{2}{c}{Test}\\",
        r"\cline{2-7}",
        r"Class & Frames & Storms & Frames & Storms & Frames & Storms\\",
        r"\middlehline",
    ]
    for label in CLASS_LABELS.values():
        values = []
        for split in ("train", "val", "test"):
            row = indexed[(split, label)]
            values.extend((_integer(row["frames"]), _integer(row["storms_with_frames"])))
        lines.append(" & ".join((label, *values)) + r"\\")
    lines.extend((r"\bottomhline", r"\end{tabular}", ""))
    output.write_text("\n".join(lines), encoding="utf-8")


def _vmax_table(source: Path, output: Path) -> None:
    with source.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    lines = [
        r"\begin{tabular}{lrrrrrc}",
        r"\tophline",
        r"Class & Generated values & Real values & Generated median (kt) & "
        r"Real median (kt) & TV & Generated support within real\\",
        r"\middlehline",
    ]
    for row in records:
        lines.append(
            " & ".join(
                (
                    row["class"],
                    _integer(row["generated_unique_vmax"]),
                    _integer(row["real_unique_vmax"]),
                    _number(row["generated_median_kt"]),
                    _number(row["real_median_kt"]),
                    f"{float(row['distribution_total_variation']):.3f}",
                    "yes" if row["generated_support_subset_of_real"].lower() == "true" else "no",
                )
            )
            + r"\\"
        )
    lines.extend((r"\bottomhline", r"\end{tabular}", ""))
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _native_table(args.native_values, args.output_dir / "native_values.tex")
    _survey_table(args.survey_summary, args.output_dir / "survey_counts.tex")
    _global_composition_table(
        args.evaluation,
        args.output_dir / "global_composition.tex",
    )
    _partition_table(args.partitions, args.output_dir / "partitions.tex")
    _vmax_table(args.vmax_support, args.output_dir / "vmax_support.tex")
    _result_macros(
        args.evaluation,
        args.validation,
        args.controls,
        args.validation_manifest,
        args.continuous_wind,
        args.feature_robustness,
        args.output_dir / "results_macros.tex",
    )


if __name__ == "__main__":
    main()
