#!/usr/bin/env python3
"""Create manuscript-ready TC Track-1 tables and figures from one frozen report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


STRATA = ("global", "0", "1", "2", "3", "4", "5", "C4_plus")
STRATUM_LABELS = {
    "global": "Global*",
    "0": "TS",
    "1": "C1",
    "2": "C2",
    "3": "C3",
    "4": "C4",
    "5": "C5",
    "C4_plus": "C4+",
}
FUNCTIONALS = (
    "w1:min_core_bt",
    "w1:cold_frac_220",
    "w1:r_maxgrad",
    "w1:axisym",
    "w1:dav",
    "w1:eye_contrast",
    "w1:p01_bt",
)
FUNCTIONAL_LABELS = {
    "w1:min_core_bt": "Core min",
    "w1:cold_frac_220": "Cold area",
    "w1:r_maxgrad": "Max-gradient r",
    "w1:axisym": "Axisymmetry",
    "w1:dav": "DAV",
    "w1:eye_contrast": "Eye contrast",
    "w1:p01_bt": "BT p01",
}
PRACTICAL_ORDER = (
    "WITHIN_TOLERANCE",
    "INCONCLUSIVE",
    "UNDERPOWERED",
    "EXCEEDS_TOLERANCE",
)
PRACTICAL_COLORS = {
    "WITHIN_TOLERANCE": "#3a923a",
    "INCONCLUSIVE": "#c7c9cc",
    "UNDERPOWERED": "#e8a83e",
    "EXCEEDS_TOLERANCE": "#c84630",
}
PRACTICAL_SHORT = {
    "WITHIN_TOLERANCE": "W",
    "INCONCLUSIVE": "I",
    "UNDERPOWERED": "U",
    "EXCEEDS_TOLERANCE": "E",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_figure(fig: plt.Figure, output: Path) -> None:
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _cells_by_key(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(cell["stratum"], cell["metric"]): cell for cell in payload["cells"]}


def _write_functional_csv(payload: dict[str, Any], output: Path) -> None:
    fields = (
        "stratum",
        "metric",
        "units",
        "estimate",
        "ci_low",
        "ci_high",
        "tolerance",
        "detection_status",
        "simultaneous_detection_status",
        "practical_status",
        "sensitivity_status",
        "n_eval",
        "n_real",
        "n_generated",
        "n_blocks",
        "estimand",
        "direction",
        "reference_q50",
        "reference_detection_boundary",
        "reference_position",
        "simultaneous_boundary",
        "shift_resolution",
        "reference_resamples_used",
        "inference_method",
        "warning_count",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for cell in payload["cells"]:
            if cell["kind"] != "functional_w1":
                continue
            row = {key: cell.get(key) for key in fields}
            row["reference_resamples_used"] = cell["provenance"].get(
                "reference_resamples_used"
            )
            row["inference_method"] = cell["provenance"].get("inference_method")
            row["warning_count"] = len(cell.get("warnings", []))
            writer.writerow(row)


def _write_secondary_csv(payload: dict[str, Any], output: Path) -> None:
    fields = (
        "stratum",
        "metric",
        "kind",
        "pillar",
        "units",
        "estimate",
        "ci_low",
        "ci_high",
        "detection_status",
        "n_eval",
        "n_blocks",
        "reference_position",
        "reference_resamples_used",
        "reference_stability_ratio",
        "warning_count",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for cell in payload["cells"]:
            if cell["kind"] == "functional_w1":
                continue
            row = {key: cell.get(key) for key in fields}
            row["reference_resamples_used"] = cell["provenance"].get(
                "reference_resamples_used"
            )
            row["reference_stability_ratio"] = cell["provenance"].get(
                "reference_stability_ratio"
            )
            row["warning_count"] = len(cell.get("warnings", []))
            writer.writerow(row)


def _status_counts(cells: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "pointwise_detection": dict(Counter(cell["detection_status"] for cell in cells)),
        "simultaneous_detection": dict(
            Counter(cell["simultaneous_detection_status"] for cell in cells)
        ),
        "practical": dict(Counter(cell["practical_status"] for cell in cells)),
    }


def _make_summary(payload: dict[str, Any], report_path: Path) -> dict[str, Any]:
    functionals = [cell for cell in payload["cells"] if cell["kind"] == "functional_w1"]
    exact_conditionals = [cell for cell in functionals if cell["stratum"] in set("012345")]
    aggregate = [cell for cell in functionals if cell["stratum"] == "C4_plus"]
    global_cells = [cell for cell in functionals if cell["stratum"] == "global"]
    warned_cells = [cell for cell in payload["cells"] if cell.get("warnings")]
    data = payload["metadata"]["data"]
    return {
        "schema": "gauge_climate.tc_bt_track1_summary.v1",
        "source_report": report_path.name,
        "source_report_sha256": _sha256(report_path),
        "config_hash": payload["metadata"]["config_hash"],
        "primary_functional_cells": _status_counts(functionals),
        "exact_class_functional_cells": _status_counts(exact_conditionals),
        "c4_plus_functional_cells": _status_counts(aggregate),
        "global_as_provided_mixture_cells": _status_counts(global_cells),
        "global_mixture_audit": {
            "interpretation": (
                "The global row compares the as-provided class-balanced generated bank "
                "with the natural test class mixture; it is a bank-composition check, "
                "not a conditional-fidelity conclusion."
            ),
            "real_label_counts": data["real"]["label_counts"],
            "generated_label_counts": data["generated"]["label_counts"],
        },
        "memorization": payload.get("memorization"),
        "memorization_interpretation": (
            "No localized copying was detected. Positive intervals mean queries are "
            "farther from training than held-out real data; they do not by themselves "
            "establish physical inadmissibility."
        ),
        "simultaneous_reference": payload["metadata"].get("simultaneous_reference"),
        "warning_audit": {
            "total_report_warnings": len(payload.get("warnings", [])),
            "cells_with_warnings": len(warned_cells),
            "by_kind": dict(Counter(cell["kind"] for cell in warned_cells)),
            "primary_functional_cells_with_warnings": sum(
                cell["kind"] == "functional_w1" for cell in warned_cells
            ),
        },
    }


def _functional_card(payload: dict[str, Any], output: Path) -> None:
    cells = _cells_by_key(payload)
    status_index = {status: index for index, status in enumerate(PRACTICAL_ORDER)}
    values = np.asarray(
        [
            [status_index[cells[(stratum, metric)]["practical_status"]] for metric in FUNCTIONALS]
            for stratum in STRATA
        ]
    )
    cmap = ListedColormap([PRACTICAL_COLORS[value] for value in PRACTICAL_ORDER])
    norm = BoundaryNorm(np.arange(-0.5, len(PRACTICAL_ORDER) + 0.5), cmap.N)
    fig, ax = plt.subplots(figsize=(11.8, 5.8), constrained_layout=True)
    ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(FUNCTIONALS)), [FUNCTIONAL_LABELS[x] for x in FUNCTIONALS])
    ax.set_yticks(np.arange(len(STRATA)), [STRATUM_LABELS[x] for x in STRATA])
    plt.setp(ax.get_xticklabels(), rotation=28, ha="right", rotation_mode="anchor")
    for row, stratum in enumerate(STRATA):
        for column, metric in enumerate(FUNCTIONALS):
            cell = cells[(stratum, metric)]
            practical = cell["practical_status"]
            color = "white" if practical in {"WITHIN_TOLERANCE", "EXCEEDS_TOLERANCE"} else "#202428"
            ax.text(
                column,
                row,
                PRACTICAL_SHORT[practical],
                ha="center",
                va="center",
                color=color,
                fontsize=10,
                fontweight="bold",
            )
            if cell["simultaneous_detection_status"] == "DEPARTURE_DETECTED":
                ax.scatter(
                    column + 0.31,
                    row - 0.27,
                    marker="x",
                    s=75,
                    linewidths=1.8,
                    color="black",
                )
            elif cell["detection_status"] == "DEPARTURE_DETECTED":
                ax.scatter(
                    column + 0.31,
                    row - 0.27,
                    marker="^",
                    s=70,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.4,
                )
    ax.axhline(0.5, color="white", linewidth=3.0)
    ax.set_title(
        "TC Track 1: practical relevance with detection overlay\n"
        "Scalar physical functionals evaluated by Wasserstein-1 distance"
    )
    legend = [
        Patch(facecolor=PRACTICAL_COLORS[status], label=f"{PRACTICAL_SHORT[status]}: {status.replace('_', ' ').title()}")
        for status in PRACTICAL_ORDER
    ]
    legend.extend(
        [
            plt.Line2D([], [], marker="x", linestyle="none", color="black", markersize=8, label="Simultaneous departure"),
            plt.Line2D([], [], marker="^", linestyle="none", markerfacecolor="none", markeredgecolor="black", markersize=7, label="Pointwise-only departure"),
        ]
    )
    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        frameon=False,
    )
    ax.text(
        0.5,
        -0.40,
        "* Global compares the as-provided balanced generated bank with the natural test label mixture; conditional rows assess model fidelity.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.5,
    )
    _save_figure(fig, output)


def _reference_profile(payload: dict[str, Any], output: Path) -> None:
    cells = _cells_by_key(payload)
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 7.0), sharex=True, constrained_layout=True)
    axes_flat = axes.ravel()
    x = np.arange(len(STRATA))
    for index, metric in enumerate(FUNCTIONALS):
        ax = axes_flat[index]
        selected = [cells[(stratum, metric)] for stratum in STRATA]
        estimates = np.asarray([cell["reference_position"] for cell in selected], dtype=float)
        low = np.asarray([cell["reference_position_low"] for cell in selected], dtype=float)
        high = np.asarray([cell["reference_position_high"] for cell in selected], dtype=float)
        ax.errorbar(
            x,
            estimates,
            yerr=np.vstack([np.maximum(estimates - low, 0), np.maximum(high - estimates, 0)]),
            fmt="o",
            color="#29577a",
            ecolor="#7793a8",
            capsize=2,
        )
        simultaneous = np.asarray([cell["simultaneous_boundary"] for cell in selected], dtype=float)
        ax.plot(x, simultaneous, "k_", markersize=10, label="simultaneous boundary")
        ax.axhline(1.0, color="#b4432c", linestyle="--", linewidth=1.0)
        ax.axhline(0.0, color="#8d949a", linewidth=0.7)
        ax.set_title(FUNCTIONAL_LABELS[metric])
        ax.set_xticks(x, [STRATUM_LABELS[value] for value in STRATA], rotation=45, ha="right")
        ax.set_yscale("symlog", linthresh=1.0)
    axes_flat[-1].axis("off")
    axes_flat[0].set_ylabel("Reference position\n(detectability, not severity)")
    axes_flat[4].set_ylabel("Reference position\n(detectability, not severity)")
    fig.suptitle("Functional detectability profile (pointwise boundary = dashed line)")
    _save_figure(fig, output)


def _curve_figure(payload: dict[str, Any], metric: str, output: Path) -> None:
    cells = _cells_by_key(payload)
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.2), constrained_layout=True)
    for ax, stratum in zip(axes.ravel(), STRATA, strict=True):
        diagnostic = payload["diagnostics"][f"{stratum}:{metric}"]
        coordinate = np.asarray(diagnostic["coordinate_values"], dtype=float)
        real = np.asarray(diagnostic["real_mean"], dtype=float)
        generated = np.asarray(diagnostic["generated_mean"], dtype=float)
        ref_low = np.asarray(diagnostic["reference_q025"], dtype=float)
        ref_high = np.asarray(diagnostic["reference_q975"], dtype=float)
        gen_low = np.asarray(diagnostic["generated_q025"], dtype=float)
        gen_high = np.asarray(diagnostic["generated_q975"], dtype=float)
        ax.fill_between(coordinate, ref_low, ref_high, color="#bfc4c8", alpha=0.6, label="real reference 95%")
        ax.fill_between(coordinate, gen_low, gen_high, color="#5c8db5", alpha=0.18, label="generated 95%")
        ax.plot(coordinate, real, color="#202428", linewidth=1.5, label="real")
        ax.plot(coordinate, generated, color="#2c6e9b", linewidth=1.5, label="generated")
        cell = cells[(stratum, metric)]
        warning = "†" if cell.get("warnings") else ""
        short = "D" if cell["detection_status"] == "DEPARTURE_DETECTED" else "ND"
        ax.set_title(f"{STRATUM_LABELS[stratum]} · {short}{warning}")
        if metric == "radial_log_psd_mse":
            ax.set_xscale("log")
        ax.set_xlabel(diagnostic.get("coordinate_label", "coordinate"))
        ax.set_ylabel(diagnostic["units"])
    axes.ravel()[0].legend(loc="best", fontsize=8)
    title = "Radial brightness-temperature profile" if metric == "radial_profile_mae" else "Radial log power spectrum"
    fig.suptitle(f"{title} · † unstable 200-draw diagnostic boundary")
    _save_figure(fig, output)


def _feature_heatmap(payload: dict[str, Any], output: Path) -> None:
    metrics = (
        "feature_mmd2",
        "feature_density",
        "feature_coverage_deficit",
        "feature_diversity_departure",
    )
    labels = ("MMD²", "Density departure", "Coverage deficit", "Diversity departure")
    cells = _cells_by_key(payload)
    values = np.asarray(
        [
            [int(cells[(stratum, metric)]["detection_status"] == "DEPARTURE_DETECTED") for metric in metrics]
            for stratum in STRATA
        ]
    )
    cmap = ListedColormap(["#4d9b57", "#c84630"])
    fig, ax = plt.subplots(figsize=(7.4, 5.4), constrained_layout=True)
    ax.imshow(values, cmap=cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5], 2), aspect="auto")
    ax.set_xticks(np.arange(len(metrics)), labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(STRATA)), [STRATUM_LABELS[x] for x in STRATA])
    for row, stratum in enumerate(STRATA):
        for column, metric in enumerate(metrics):
            cell = cells[(stratum, metric)]
            text = "D" if values[row, column] else "ND"
            if cell.get("warnings"):
                text += "†"
            ax.text(column, row, text, ha="center", va="center", color="white", fontweight="bold")
    ax.axhline(0.5, color="white", linewidth=3.0)
    ax.set_title("Training-fitted feature-space diagnostics")
    ax.legend(
        handles=[
            Patch(facecolor="#4d9b57", label="No departure detected"),
            Patch(facecolor="#c84630", label="Departure detected"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        frameon=False,
    )
    ax.text(
        0.5,
        -0.37,
        "† split-half boundary instability at the frozen 200-draw diagnostic budget",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.5,
    )
    _save_figure(fig, output)


def _dependence_figure(payload: dict[str, Any], output: Path) -> None:
    names = tuple(metric.removeprefix("w1:") for metric in FUNCTIONALS)
    labels = tuple(FUNCTIONAL_LABELS[metric] for metric in FUNCTIONALS)
    cells = _cells_by_key(payload)
    matrices = []
    warnings = []
    for stratum in STRATA:
        matrix = np.zeros((len(names), len(names)), dtype=float)
        warning_matrix = np.zeros_like(matrix, dtype=bool)
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                metric = f"spearman_pair:{names[left]}:{names[right]}"
                cell = cells[(stratum, metric)]
                matrix[left, right] = matrix[right, left] = float(cell["estimate"])
                warning_matrix[left, right] = warning_matrix[right, left] = bool(
                    cell.get("warnings")
                )
        matrices.append(matrix)
        warnings.append(warning_matrix)
    vmax = max(float(np.max(matrix)) for matrix in matrices)
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.2), constrained_layout=True)
    image = None
    for ax, stratum, matrix, warning_matrix in zip(
        axes.ravel(), STRATA, matrices, warnings, strict=True
    ):
        image = ax.imshow(matrix, vmin=0.0, vmax=vmax, cmap="magma")
        ax.set_title(STRATUM_LABELS[stratum])
        ax.set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=7)
        for row, column in np.argwhere(warning_matrix):
            if row < column:
                ax.text(column, row, "†", ha="center", va="center", color="white")
    assert image is not None
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="Absolute Spearman correlation difference")
    fig.suptitle("Physical-functional dependence closure · † unstable diagnostic boundary")
    _save_figure(fig, output)


def _admissibility_figure(payload: dict[str, Any], output: Path) -> None:
    cells = _cells_by_key(payload)
    selected = [cells[(stratum, "preprocessing_boundary_saturation")] for stratum in STRATA]
    real = 100.0 * np.asarray(
        [cell["provenance"]["raw_real_rate"] for cell in selected], dtype=float
    )
    generated = 100.0 * np.asarray(
        [cell["provenance"]["raw_generated_rate"] for cell in selected], dtype=float
    )
    x = np.arange(len(STRATA))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    ax.bar(x - width / 2, real, width, label="held-out real", color="#555d63")
    ax.bar(x + width / 2, generated, width, label="generated", color="#3a7ca5")
    ax.set_xticks(x, [STRATUM_LABELS[value] for value in STRATA])
    ax.set_ylabel("Fields at 170 K or 325 K clipping boundary (%)")
    ax.set_title("Preprocessing-boundary saturation (diagnostic, not a physical-law test)")
    ax.legend(frameon=False)
    for index, cell in enumerate(selected):
        if cell.get("warnings"):
            ax.text(index, max(real[index], generated[index]) + 0.03, "†", ha="center")
    _save_figure(fig, output)


def _memorization_figure(payload: dict[str, Any], output: Path) -> None:
    result = payload["memorization"]
    regions = result["regions"]
    strata = sorted(
        {item.get("stratum") for item in regions if item.get("stratum") is not None},
        key=str,
    )
    farther = sum(item["status"] == "OFF_MANIFOLD" for item in regions)
    if strata:
        rows = 2
        columns = int(np.ceil(len(strata) / rows))
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(4.7 * columns, 8.0),
            sharex=True,
            constrained_layout=True,
        )
        axes_flat = np.asarray(axes).ravel()
        for ax, stratum in zip(axes_flat, strata, strict=False):
            selected = sorted(
                [item for item in regions if item.get("stratum") == stratum],
                key=lambda item: item["region"],
            )
            y = np.arange(len(selected))
            for index, item in enumerate(selected):
                color = (
                    "#c84630"
                    if item["status"] == "COPYING_DETECTED"
                    else "#cb8b2b"
                    if item["status"] == "OFF_MANIFOLD"
                    else "#65727c"
                )
                ax.errorbar(
                    item["effect"],
                    index,
                    xerr=[
                        [item["effect"] - item["ci_low"]],
                        [item["ci_high"] - item["effect"]],
                    ],
                    fmt="o",
                    color=color,
                    ecolor=color,
                    capsize=3,
                )
            ax.axvline(0.0, color="#202428", linewidth=1.0)
            ax.set_yticks(
                y,
                [
                    f"R{item['region']} · G={item['n_generated']} · B={item['n_real_blocks']}"
                    for item in selected
                ],
            )
            ax.invert_yaxis()
            ax.set_title(STRATUM_LABELS.get(stratum, stratum))
            ax.set_xlabel("Generated − held-out-real NN distance")
        for ax in axes_flat[len(strata) :]:
            ax.axis("off")
        fig.suptitle(
            f"Label-matched localized copying test · no copying; "
            f"{farther}/{len(regions)} farther-from-training cells\n"
            "Negative intervals imply copying; positive intervals mean farther "
            "from training, not physical inadmissibility.",
            fontsize=12,
        )
    else:
        y = np.arange(len(regions))
        fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
        for index, item in enumerate(regions):
            color = (
                "#c84630"
                if item["status"] == "COPYING_DETECTED"
                else "#cb8b2b"
                if item["status"] == "OFF_MANIFOLD"
                else "#65727c"
            )
            ax.errorbar(
                item["effect"],
                index,
                xerr=[
                    [item["effect"] - item["ci_low"]],
                    [item["ci_high"] - item["effect"]],
                ],
                fmt="o",
                color=color,
                ecolor=color,
                capsize=3,
            )
        ax.axvline(0.0, color="#202428", linewidth=1.0)
        ax.set_yticks(
            y,
            [
                f"Region {item['region']} (G={item['n_generated']}, B={item['n_real_blocks']})"
                for item in regions
            ],
        )
        ax.invert_yaxis()
        ax.set_xlabel("Median NN-distance effect: generated − held-out real")
        ax.set_title(
            f"Localized three-sample copying test · no copying; {farther}/{len(regions)} farther"
        )
        ax.text(
            0.5,
            -0.14,
            "Negative intervals imply copying; positive intervals mean farther from training, not physical inadmissibility.",
            transform=ax.transAxes,
            ha="center",
            fontsize=8.5,
        )
    _save_figure(fig, output)


def _write_markdown(summary: dict[str, Any], output: Path) -> None:
    overall = summary["primary_functional_cells"]
    exact = summary["exact_class_functional_cells"]
    aggregate = summary["c4_plus_functional_cells"]
    global_counts = summary["global_as_provided_mixture_cells"]
    warning = summary["warning_audit"]
    mem = summary["memorization"]
    lines = [
        "# Frozen TC Track-1 results",
        "",
        f"Source report SHA-256: `{summary['source_report_sha256']}`; config hash: `{summary['config_hash']}`.",
        "",
        "## Primary scalar-function results",
        "",
        f"Across all 56 registered cells, pointwise detection counts are `{overall['pointwise_detection']}` and simultaneous counts are `{overall['simultaneous_detection']}`. Practical statuses are `{overall['practical']}`.",
        "",
        f"Across the 42 exact class-conditioned cells (TS and C1–C5), detection counts are `{exact['pointwise_detection']}` and practical statuses are `{exact['practical']}`. The seven predeclared C4+ cells have `{aggregate['pointwise_detection']}` with practical statuses `{aggregate['practical']}`.",
        "",
        f"The seven global as-provided-mixture cells have pointwise counts `{global_counts['pointwise_detection']}` and simultaneous counts `{global_counts['simultaneous_detection']}`. These rows compare a balanced generated bank with the natural test label mixture and are a composition check, not conditional-generator fidelity.",
        "",
        "## Memorization",
        "",
        f"Across {len(mem['regions'])} supported query cells, {sum(item['status'] == 'COPYING_DETECTED' for item in mem['regions'])} report copying, {sum(item['status'] == 'OFF_MANIFOLD' for item in mem['regions'])} are significantly farther from training than held-out real queries, and {sum(item['status'] == 'INCONCLUSIVE' for item in mem['regions'])} are inconclusive. The legacy aggregate machine enum is `{mem['status']}`; here it means farther-from-training and does not establish physical inadmissibility.",
        "",
        "## Monte Carlo audit",
        "",
        f"Primary functional cells with stability warnings: {warning['primary_functional_cells_with_warnings']}. Secondary cells with warnings: {warning['cells_with_warnings']}, by kind `{warning['by_kind']}`. A dagger in secondary figures marks an unstable 200-draw diagnostic boundary; those detection labels are not headline claims.",
        "",
        "## Figure key",
        "",
        "- `track1_functional_card.pdf`: practical-status heatmap with pointwise/simultaneous detection overlay.",
        "- `track1_reference_profile.pdf`: compact detectability profile for the 56 scalar-functional cells.",
        "- `track1_radial_profiles.pdf` and `track1_radial_psd.pdf`: stored curve diagnostics and real-reference envelopes.",
        "- `track1_feature_diagnostics.pdf`: MMD², density, coverage, and diversity detection map.",
        "- `track1_dependence.pdf`: raw pairwise Spearman-correlation gaps; daggers mark unstable categorical boundaries.",
        "- `track1_admissibility.pdf`: real/generated preprocessing-boundary saturation rates.",
        "- `track1_memorization.pdf`: localized block-aware three-sample copying result.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--superseded-report", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.report.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "gauge_climate.evaluation.v1":
        raise ValueError("Unsupported report schema.")

    summary = _make_summary(payload, args.report.resolve())
    if args.superseded_report is not None:
        summary["superseded_report"] = {
            "path": args.superseded_report.name,
            "sha256": _sha256(args.superseded_report),
            "reason": "pre-label-mixture-audit completed report retained for transparency",
        }
    summary_path = args.output_dir / "track1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_functional_csv(payload, args.output_dir / "track1_functional_cells.csv")
    _write_secondary_csv(payload, args.output_dir / "track1_secondary_cells.csv")
    _write_markdown(summary, args.output_dir / "README.md")
    _functional_card(payload, args.output_dir / "track1_functional_card.pdf")
    _reference_profile(payload, args.output_dir / "track1_reference_profile.pdf")
    _curve_figure(payload, "radial_profile_mae", args.output_dir / "track1_radial_profiles.pdf")
    _curve_figure(payload, "radial_log_psd_mse", args.output_dir / "track1_radial_psd.pdf")
    _feature_heatmap(payload, args.output_dir / "track1_feature_diagnostics.pdf")
    _dependence_figure(payload, args.output_dir / "track1_dependence.pdf")
    _admissibility_figure(payload, args.output_dir / "track1_admissibility.pdf")
    _memorization_figure(payload, args.output_dir / "track1_memorization.pdf")
    shutil.copy2(args.report, args.output_dir / "evaluation.json")
    html_report = args.report.with_suffix(".html")
    if html_report.exists():
        shutil.copy2(html_report, args.output_dir / "evaluation_full.html")
    if args.superseded_report is not None:
        shutil.copy2(
            args.superseded_report,
            args.output_dir / "evaluation_superseded_pre_label_audit.json.gz",
        )

    artifacts = {}
    for path in sorted(args.output_dir.iterdir()):
        if path.name != "manifest.json" and path.is_file():
            artifacts[path.name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    manifest = {
        "schema": "gauge_climate.tc_bt_track1_artifacts.v1",
        "source_report_sha256": summary["source_report_sha256"],
        "config_hash": summary["config_hash"],
        "artifacts": artifacts,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "artifacts": len(artifacts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
