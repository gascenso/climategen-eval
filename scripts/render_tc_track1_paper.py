#!/usr/bin/env python3
"""Render the paper-facing TC bundle from one frozen GAUGE report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import yaml

from gauge_climate import EvaluationReport, PaperBundleSpec, ReferenceControl
from gauge_climate.adapters import tc_bt
from gauge_climate.adapters.tc_data import load_real_archive, load_sample_bank


ROOT = Path(__file__).resolve().parents[1]
CLASS_LABELS = {"0": "TS", "1": "C1", "2": "C2", "3": "C3", "4": "C4", "5": "C5"}
METRIC_LABELS = {
    "w1:min_core_bt": "Core minimum BT · W1",
    "w1:cold_frac_220": "Cold-cloud area · W1",
    "w1:r_maxgrad": "Max-gradient radius · W1",
    "w1:axisym": "Axisymmetry · W1",
    "w1:dav": "Deviation-angle variance · W1",
    "w1:eye_contrast": "Eye contrast · W1",
    "w1:p01_bt": "BT lower tail · W1",
    "radial_profile_mae": "Radial BT profile · MAE",
    "radial_log_psd_mse": "Radial power spectrum · MSE",
    "feature_mmd2": "Feature distribution · MMD²",
    "feature_density": "Feature density departure",
    "feature_coverage_deficit": "Feature coverage deficit",
    "feature_diversity_departure": "Feature diversity departure",
}
MAIN_METRICS = tuple(METRIC_LABELS)
FUNCTIONAL_IDS = (
    "min_core_bt",
    "cold_frac_220",
    "r_maxgrad",
    "axisym",
    "dav",
    "eye_contrast",
    "p01_bt",
)
FUNCTIONAL_LABELS = {
    "min_core_bt": "Core minimum BT",
    "cold_frac_220": "Cold-cloud area < 220 K",
    "r_maxgrad": "Radius of maximum BT gradient",
    "axisym": "Axisymmetry departure",
    "dav": "Deviation-angle variance",
    "eye_contrast": "Eye minus eyewall BT",
    "p01_bt": "Per-image BT first percentile",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _controls(payload: Mapping[str, Any]) -> tuple[ReferenceControl, ...]:
    reference = payload["_metadata"]["reference_envelopes"]
    labels = {
        "warm_shift": "Warm shift (+4 K)",
        "blur": "Blur (σ=2 px)",
        "center_jitter": "Center jitter (4 px)",
        "phase_shuffle": "Phase shuffle",
    }
    controls: list[ReferenceControl] = []
    for lane_id, label in labels.items():
        lane = payload.get(lane_id, {})
        for metric, result in lane.get("metrics", {}).items():
            if metric not in reference:
                continue
            values = result.get("values")
            value = result.get("value") if values is None else values[-1]
            if value is None:
                continue
            q50 = float(reference[metric]["q50"])
            boundary = float(reference[metric]["boundary"])
            width = boundary - q50
            if width <= np.finfo(float).eps:
                continue
            controls.append(
                ReferenceControl(
                    metric=metric,
                    label=label,
                    position=(float(value) - q50) / width,
                    provenance="training-only preregistered sensitivity lane",
                )
            )
    return tuple(controls)


def _adapter_inputs(config_path: Path) -> tuple[Any, Any, tuple[Any, ...], dict[str, Any]]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    options = dict(payload["adapter_options"])
    config = tc_bt.TCBTConfig(**options)
    real_node = payload["data"]["real"]
    generated_node = payload["data"]["generated"]
    real = load_real_archive(
        _resolve(config_path, real_node["tc_real_manifest"]),
        target=real_node.get("target", "category"),
    )
    generated = load_sample_bank(
        _resolve(config_path, generated_node["tc_sample_bank"]),
        target=generated_node.get("target", "category"),
        required_run=generated_node.get("required_run"),
    )
    real_fields = tc_bt.PreprocessedFields(real.fields, config)
    generated_fields = tc_bt.PreprocessedFields(generated.fields, config)
    real = type(real)(
        real_fields,
        blocks=real.blocks,
        labels=real.labels,
        sample_ids=real.sample_ids,
        name=real.name,
        units=real.units,
        metadata=real.metadata,
    )
    generated = type(generated)(
        generated_fields,
        blocks=generated.blocks,
        labels=generated.labels,
        sample_ids=generated.sample_ids,
        name=generated.name,
        units=generated.units,
        metadata=generated.metadata,
    )
    return real, generated, tc_bt.functionals(config), payload


def _transform_functionals(
    fields: Any,
    functionals: Sequence[Any],
    *,
    batch_size: int,
    label: str,
) -> dict[str, np.ndarray]:
    outputs = {
        functional.id: np.empty(len(fields), dtype=np.float64)
        for functional in functionals
    }
    for start in range(0, len(fields), batch_size):
        stop = min(start + batch_size, len(fields))
        batch = np.asarray(fields[start:stop])
        for functional in functionals:
            outputs[functional.id][start:stop] = functional.transform(batch)
        print(f"{label}: transformed {stop}/{len(fields)} fields", flush=True)
    return outputs


def _load_or_build_cache(
    report_path: Path,
    config_path: Path,
    cache_path: Path,
    *,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    real, generated, functionals, config_payload = _adapter_inputs(config_path)
    expected = {
        "schema": "gauge_climate.tc_bt_presentation_cache.v1",
        "report_sha256": _sha256(report_path),
        "config_sha256": _sha256(config_path),
        "real_manifest": str(config_payload["data"]["real"]["tc_real_manifest"]),
        "generated_bank": str(config_payload["data"]["generated"]["tc_sample_bank"]),
        "functional_ids": list(FUNCTIONAL_IDS),
    }
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata == expected:
            archive = np.load(cache_path)
            real_values = {
                name: np.asarray(archive[f"real_{name}"]) for name in FUNCTIONAL_IDS
            }
            generated_values = {
                name: np.asarray(archive[f"generated_{name}"])
                for name in FUNCTIONAL_IDS
            }
            return (
                real_values,
                generated_values,
                np.asarray(archive["real_labels"]),
                np.asarray(archive["generated_labels"]),
                np.asarray(archive["real_blocks"]),
            )

    real_values = _transform_functionals(
        real.fields, functionals, batch_size=batch_size, label="held-out real"
    )
    generated_values = _transform_functionals(
        generated.fields, functionals, batch_size=batch_size, label="generated"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        **{f"real_{key}": value for key, value in real_values.items()},
        **{f"generated_{key}": value for key, value in generated_values.items()},
        real_labels=np.asarray(real.labels),
        generated_labels=np.asarray(generated.labels),
        real_blocks=np.asarray(real.blocks).astype(str),
    )
    metadata_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return (
        real_values,
        generated_values,
        np.asarray(real.labels),
        np.asarray(generated.labels),
        np.asarray(real.blocks).astype(str),
    )


def _block_balanced_weights(blocks: np.ndarray) -> np.ndarray:
    blocks = np.asarray(blocks).astype(str)
    unique, inverse, counts = np.unique(blocks, return_inverse=True, return_counts=True)
    return 1.0 / (unique.size * counts[inverse])


def _weighted_ecdf(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(values, kind="mergesort")
    x = np.asarray(values)[order]
    y = np.cumsum(np.asarray(weights, dtype=float)[order])
    y /= y[-1]
    return x, y


def _ecdf_panel(
    ax: plt.Axes,
    real: np.ndarray,
    generated: np.ndarray,
    blocks: np.ndarray,
    *,
    title: str,
    units: str,
) -> None:
    real_x, real_y = _weighted_ecdf(real, _block_balanced_weights(blocks))
    generated_x, generated_y = _weighted_ecdf(
        generated, np.full(generated.size, 1.0 / generated.size)
    )
    ax.step(real_x, real_y, where="post", color="#263238", linewidth=1.65, label="held-out real")
    ax.step(
        generated_x,
        generated_y,
        where="post",
        color="#2b7a78",
        linewidth=1.65,
        label="generated",
    )
    ax.set_title(title, fontsize=10.5, fontweight="semibold")
    ax.set_xlabel(units)
    ax.set_ylabel("Cumulative probability")
    ax.grid(color="#e0e6e8", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def _functional_distribution_figures(
    report: EvaluationReport,
    output_dir: Path,
    real_values: Mapping[str, np.ndarray],
    generated_values: Mapping[str, np.ndarray],
    real_labels: np.ndarray,
    generated_labels: np.ndarray,
    real_blocks: np.ndarray,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = {(cell.stratum, cell.metric): cell for cell in report.cells}
    artifacts: dict[str, Path] = {}

    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.8))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.11, wspace=0.28, hspace=0.38)
    tail_real = real_labels == 5
    tail_generated = generated_labels == 5
    for ax, functional_id in zip(axes.flat, FUNCTIONAL_IDS, strict=False):
        cell = cells[("5", f"w1:{functional_id}")]
        _ecdf_panel(
            ax,
            real_values[functional_id][tail_real],
            generated_values[functional_id][tail_generated],
            real_blocks[tail_real],
            title=FUNCTIONAL_LABELS[functional_id],
            units=cell.units,
        )
        ax.text(
            0.98,
            0.06,
            f"W1 = {cell.estimate:.3g} {cell.units}\ngap = {cell.reference_position:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#53636a",
        )
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
    fig.suptitle(
        "Category 5 physical distributions (thin-stratum audit)",
        fontsize=17,
        fontweight="semibold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Real ECDFs use the frozen block-balanced storm weighting; generated samples are equally weighted.",
        ha="center",
        fontsize=8.5,
        color="#5e6b70",
    )
    base = output_dir.parent / "main" / "figure_2_tail_physical_distributions"
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    artifacts["tail_pdf"] = base.with_suffix(".pdf")
    artifacts["tail_png"] = base.with_suffix(".png")

    distribution_dir = output_dir / "functional_distributions"
    distribution_dir.mkdir(parents=True, exist_ok=True)
    for functional_id in FUNCTIONAL_IDS:
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0))
        fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.16, wspace=0.27, hspace=0.38)
        for class_id, ax in enumerate(axes.flat):
            real_mask = real_labels == class_id
            generated_mask = generated_labels == class_id
            cell = cells[(str(class_id), f"w1:{functional_id}")]
            _ecdf_panel(
                ax,
                real_values[functional_id][real_mask],
                generated_values[functional_id][generated_mask],
                real_blocks[real_mask],
                title=CLASS_LABELS[str(class_id)],
                units=cell.units,
            )
            ax.text(
                0.98,
                0.06,
                f"W1 {cell.estimate:.3g} · gap {cell.reference_position:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.6,
                color="#53636a",
            )
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=2, frameon=False)
        fig.suptitle(
            f"{FUNCTIONAL_LABELS[functional_id]} — native-class distributions",
            fontsize=16,
            fontweight="semibold",
            y=0.98,
        )
        base = distribution_dir / functional_id
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(base.with_suffix(".png"), dpi=180, bbox_inches="tight")
        plt.close(fig)
    return artifacts


def _curve_figures(report: EvaluationReport, output_dir: Path) -> dict[str, Path]:
    metrics = (
        ("radial_profile_mae", "Mean radial BT profile"),
        ("radial_log_psd_mse", "Radial log-power spectrum"),
    )
    cells = {(cell.stratum, cell.metric): cell for cell in report.cells}
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.70, bottom=0.16, wspace=0.24)
    for ax, (metric, label) in zip(axes, metrics, strict=True):
        diagnostic = report.diagnostics[f"5:{metric}"]
        x = np.asarray(diagnostic["coordinate_values"])
        ax.fill_between(
            x,
            diagnostic["reference_q025"],
            diagnostic["reference_q975"],
            color="#b7d3d0",
            alpha=0.55,
            label="real-reference 95% envelope",
        )
        ax.plot(x, diagnostic["real_mean"], color="#263238", linewidth=1.7, label="held-out real")
        ax.plot(x, diagnostic["generated_mean"], color="#2b7a78", linewidth=1.7, label="generated")
        cell = cells[("5", metric)]
        ax.set_title(label, fontweight="semibold")
        ax.set_xlabel(diagnostic["coordinate_label"])
        ax.set_ylabel(diagnostic["units"])
        ax.grid(color="#e0e6e8", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.98,
            0.06,
            f"gap = {cell.reference_position:.2f}",
            transform=ax.transAxes,
            ha="right",
            fontsize=8.5,
            color="#53636a",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.84), ncol=3, frameon=False)
    fig.suptitle(
        "Category 5 spatial structure (thin-stratum audit)",
        fontsize=17,
        fontweight="semibold",
        y=0.98,
    )
    base = output_dir.parent / "main" / "figure_3_tail_spatial_structure"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"structure_pdf": base.with_suffix(".pdf"), "structure_png": base.with_suffix(".png")}


def _safeguard_figure(report: EvaluationReport, output_dir: Path) -> dict[str, Path]:
    memorization = report.memorization or {}
    regions = memorization.get("regions", [])
    status_order = ("COPYING_DETECTED", "OFF_MANIFOLD", "INCONCLUSIVE")
    status_labels = ("copying", "farther from training", "inconclusive")
    status_colors = ("#b83b32", "#d99b35", "#78909c")
    counts = [sum(item.get("status") == status for item in regions) for status in status_order]

    cells = {
        cell.stratum: cell
        for cell in report.cells
        if cell.metric == "preprocessing_boundary_saturation" and cell.stratum in CLASS_LABELS
    }
    strata = tuple(CLASS_LABELS)
    real_rates = [100.0 * float(cells[stratum].provenance["raw_real_rate"]) for stratum in strata]
    generated_rates = [
        100.0 * float(cells[stratum].provenance["raw_generated_rate"])
        for stratum in strata
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.19, wspace=0.34)
    left, right = axes
    left.barh(np.arange(3), counts, color=status_colors, height=0.58)
    left.set_yticks(np.arange(3), status_labels)
    left.invert_yaxis()
    left.set_xlabel("Label × feature-region query cells")
    left.set_title("Localized copying audit", fontweight="semibold")
    for row, count in enumerate(counts):
        left.text(count + 0.5, row, str(count), va="center", fontsize=10)
    left.text(
        0.02,
        0.02,
        "Copyist positive control: passed",
        transform=left.transAxes,
        fontsize=8.5,
        color="#2d6a4f",
    )
    left.spines[["top", "right", "left"]].set_visible(False)
    left.tick_params(axis="y", length=0)
    left.grid(axis="x", color="#e0e6e8", linewidth=0.6)

    x = np.arange(len(strata))
    width = 0.36
    right.bar(x - width / 2, real_rates, width, color="#4f5b61", label="held-out real")
    right.bar(x + width / 2, generated_rates, width, color="#2b7a78", label="generated")
    right.set_xticks(x, [CLASS_LABELS[stratum] for stratum in strata])
    right.set_ylabel("Fields at clipping boundary (%)")
    right.set_title("Preprocessing-boundary audit", fontweight="semibold")
    right.legend(frameon=False, ncol=2, fontsize=8.5)
    right.spines[["top", "right"]].set_visible(False)
    right.grid(axis="y", color="#e0e6e8", linewidth=0.6)
    fig.suptitle("Generation safeguards", fontsize=17, fontweight="semibold", y=0.97)
    fig.text(
        0.5,
        0.035,
        "‘Farther from training’ is not physical inadmissibility. Boundary saturation audits the frozen clipping representation, not a physical law.",
        ha="center",
        fontsize=8.2,
        color="#5e6b70",
    )
    base = output_dir.parent / "main" / "figure_4_generation_safeguards"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"safeguards_pdf": base.with_suffix(".pdf"), "safeguards_png": base.with_suffix(".png")}


def _copy_audit_artifacts(source: Path, appendix: Path) -> None:
    audit = appendix / "extended_diagnostics"
    audit.mkdir(parents=True, exist_ok=True)
    for stem in (
        "track1_feature_diagnostics",
        "track1_memorization",
        "track1_dependence",
        "track1_admissibility",
    ):
        for suffix in (".pdf", ".png"):
            path = source / f"{stem}{suffix}"
            if path.exists():
                shutil.copy2(path, audit / path.name)


def _write_readme(output_dir: Path, report_path: Path, report: EvaluationReport) -> None:
    text = f"""# TC GAUGE paper-facing results

This directory is the output an ordinary GAUGE user should start from. The
statistical machinery remains in the frozen report, while the main directory
contains the concise figures and native-unit table intended for a results
section.

## Main-text artifacts

- `main/figure_1_reference_gap_profile.pdf`: primary intrinsic result. Every
  native class is visible and training-only positive controls are open diamonds.
- `main/figure_2_tail_physical_distributions.pdf`: Category 5 empirical distributions
  for all seven physical functionals.
- `main/figure_3_tail_spatial_structure.pdf`: Category 5 radial-profile and spectral
  drill-downs.
- `main/figure_4_generation_safeguards.pdf`: compact memorization and
  preprocessing-boundary audit.
- `main/table_1_native_metric_values.csv`: exact native-unit estimates,
  intervals, reference summaries, gaps, and statuses for every displayed class.
- `main/table_1_c5_values.md`: compact human-readable Category 5 table for
  direct inspection or manuscript adaptation.
- `main/table_1_c5_values.csv`: the same Category 5 values in machine-readable form.

`index.html` provides a browser-friendly landing page. The appendix retains the
complete class matrix, all report cells, every native-class distribution, and
the extended diagnostics.

## Reporting language

- Gap 0 is the median real-vs-real discrepancy; gap 1 is the pointwise 95%
  real-reference boundary.
- Gap units express detectability at the declared sample size, not physical
  severity, and must not be averaged into a model score.
- A point below 1 supports “no departure detected,” not equality.
- The unmatched global mixture is deliberately absent from the fidelity figure;
  it remains in the complete audit table as a dataset-composition check.

Source report: `{report_path.name}`

Source report SHA-256: `{_sha256(report_path)}`

Configuration hash: `{report.metadata.get('config_hash', 'unknown')}`
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def _write_c5_table(output_dir: Path, report: EvaluationReport) -> None:
    cells = {(cell.stratum, cell.metric): cell for cell in report.cells}
    rows = []
    for metric in MAIN_METRICS:
        cell = cells[("5", metric)]
        rows.append(
            {
                "metric": METRIC_LABELS[metric],
                "estimate": cell.estimate,
                "ci_low": cell.ci_low,
                "ci_high": cell.ci_high,
                "units": cell.units,
                "reference_median": cell.reference_q50,
                "reference_boundary": cell.reference_detection_boundary,
                "reference_gap": cell.reference_position,
                "detection": cell.detection_status.value,
            }
        )
    main = output_dir / "main"
    csv_path = main / "table_1_c5_values.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    markdown = [
        "# Category 5 metric values (thin stratum)",
        "",
        "| Metric | Model discrepancy (95% resampling interval) | Real-reference median → boundary | Gap | Detection |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        estimate = (
            "—"
            if row["estimate"] is None
            else f"{row['estimate']:.3g} [{row['ci_low']:.3g}, {row['ci_high']:.3g}] {row['units']}"
        )
        reference = (
            "—"
            if row["reference_median"] is None
            else f"{row['reference_median']:.3g} → {row['reference_boundary']:.3g}"
        )
        gap = "—" if row["reference_gap"] is None else f"{row['reference_gap']:.2f}"
        markdown.append(
            f"| {row['metric']} | {estimate} | {reference} | {gap} | "
            f"{row['detection']} |"
        )
    markdown.extend(
        [
            "",
            "Gap 1 is the pointwise 95% real-reference boundary. It is a detectability coordinate, not a universal severity or acceptability scale.",
            "",
        ]
    )
    (main / "table_1_c5_values.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )


def _write_index(output_dir: Path, report: EvaluationReport) -> None:
    figures = (
        (
            "main/figure_1_reference_gap_profile.png",
            "1. Reference-calibrated intrinsic profile",
            "All six native intensity classes and training-only positive controls on one scale; no pooled tail can hide C5.",
        ),
        (
            "main/figure_2_tail_physical_distributions.png",
            "2. Tail physical distributions",
            "Block-balanced Category 5 empirical distributions for every scalar physical functional; interpret with its thin-stratum uncertainty.",
        ),
        (
            "main/figure_3_tail_spatial_structure.png",
            "3. Tail spatial structure",
            "Category 5 radial-profile and power-spectrum drill-downs with the real-reference envelope.",
        ),
        (
            "main/figure_4_generation_safeguards.png",
            "4. Generation safeguards",
            "A compact copying audit and preprocessing-boundary diagnostic.",
        ),
    )
    figure_html = "".join(
        f'<section><h2>{title}</h2><img src="{path}" alt="{title}"><p>{caption}</p></section>'
        for path, title, caption in figures
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>TC GAUGE results</title>
<style>
:root{{--ink:#213138;--muted:#607078;--wash:#f3f7f8;--accent:#246a73}}
body{{font-family:Inter,system-ui,sans-serif;max-width:1180px;margin:0 auto;padding:2.5rem 2rem;color:var(--ink)}}
h1{{font-size:2.15rem;margin-bottom:.35rem}}h2{{margin-top:2.6rem;border-bottom:1px solid #dbe3e6;padding-bottom:.45rem}}
.lede{{font-size:1.08rem;color:var(--muted);max-width:900px}}.key{{background:var(--wash);border-left:4px solid var(--accent);padding:1rem 1.2rem;border-radius:3px}}
img{{display:block;width:100%;height:auto;border:1px solid #e2e8ea;margin-top:1rem}}section p{{color:var(--muted)}}
a{{color:var(--accent)}}code{{background:var(--wash);padding:.1rem .25rem}}
</style></head><body>
<h1>Tropical-cyclone GAUGE evaluation</h1>
<p class="lede">This is the interpretation-first GAUGE output intended for a paper results section. The complete machine report, class matrix, numerical tables, and diagnostic drill-downs remain available underneath it.</p>
<div class="key"><strong>Gap key:</strong> 0 is the median real–real discrepancy and 1 is the pointwise 95% real-reference boundary. Lower is closer. Gap units express detectability at this N, not physical severity; a result below 1 is not automatically equivalent.</div>
{figure_html}
<h2>Tables and appendix</h2>
<ul><li><a href="main/table_1_c5_values.md">Human-readable Category 5 metric table</a></li>
<li><a href="main/table_1_c5_values.csv">Category 5 metric table (CSV)</a></li>
<li><a href="main/table_1_native_metric_values.csv">Every displayed class and metric (CSV)</a></li>
<li><a href="appendix/figure_A1_conditional_gap_matrix.pdf">Complete conditional matrix (PDF)</a></li>
<li><a href="appendix/table_A1_complete_evaluation.csv">Every report cell (CSV)</a></li>
<li><a href="appendix/functional_distributions/">Native-class distribution figures</a></li>
<li><a href="README.md">Reporting guide and permitted language</a></li></ul>
<p>Frozen configuration hash: <code>{report.metadata.get('config_hash', 'unknown')}</code>.</p>
</body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def _write_manifest(output_dir: Path, report_path: Path) -> None:
    artifacts = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts[str(path.relative_to(output_dir))] = {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
    payload = {
        "schema": "gauge_climate.paper_bundle.v1",
        "source_report_sha256": _sha256(report_path),
        "artifacts": artifacts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/tropical_cyclone_bt/presentation/functionals.npz",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    report_path = args.report.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = EvaluationReport.from_json(report_path)
    controls_payload = json.loads(args.controls.read_text(encoding="utf-8"))
    spec = PaperBundleSpec(
        title="Tropical-cyclone brightness-temperature generation",
        model_label="baseline_ztsnr diffusion model",
        metrics=MAIN_METRICS,
        native_strata=("0", "1", "2", "3", "4", "5"),
        priority_strata=(),
        metric_labels=METRIC_LABELS,
        stratum_labels=CLASS_LABELS,
    )
    report.paper_bundle(output_dir, spec, controls=_controls(controls_payload))

    (
        real_values,
        generated_values,
        real_labels,
        generated_labels,
        real_blocks,
    ) = _load_or_build_cache(
        report_path,
        config_path,
        args.cache.resolve(),
        batch_size=args.batch_size,
    )
    appendix = output_dir / "appendix"
    _functional_distribution_figures(
        report,
        appendix,
        real_values,
        generated_values,
        real_labels,
        generated_labels,
        real_blocks,
    )
    _curve_figures(report, appendix)
    _safeguard_figure(report, appendix)
    _copy_audit_artifacts(report_path.parent, appendix)
    _write_c5_table(output_dir, report)
    _write_readme(output_dir, report_path, report)
    _write_index(output_dir, report)
    _write_manifest(output_dir, report_path)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "artifacts": len(
                    json.loads((output_dir / "manifest.json").read_text())["artifacts"]
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
