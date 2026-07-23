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
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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


def _save_figure(fig: plt.Figure, base: Path, *, dpi: int = 220) -> tuple[Path, Path]:
    pdf = base.with_suffix(".pdf")
    png = base.with_suffix(".png")
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def _framework_overview(output_dir: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(12.2, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ink = "#213138"
    accent = "#246a73"
    warm = "#c45b3c"
    wash = "#eef5f5"
    pale = "#f7f9fa"

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        title: str,
        lines: Sequence[str],
        *,
        edge: str = accent,
        face: str = pale,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.014",
            linewidth=1.35,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(
            x + 0.018,
            y + height - 0.035,
            title,
            fontsize=11,
            fontweight="bold",
            color=ink,
            va="top",
        )
        ax.text(
            x + 0.018,
            y + height - 0.085,
            "\n".join(lines),
            fontsize=8.8,
            color="#53636a",
            va="top",
            linespacing=1.45,
        )

    def arrow(start: tuple[float, float], end: tuple[float, float], color: str = accent) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.35,
                color=color,
                connectionstyle="arc3,rad=0.0",
            )
        )

    ax.text(
        0.5,
        0.965,
        "GAUGE: reference-aware evaluation of generated Earth-system fields",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color=ink,
    )
    ax.text(
        0.5,
        0.915,
        "Simple paper-facing evidence backed by dependence-aware calibration, controls, and provenance",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#607078",
    )

    box(
        0.025,
        0.64,
        0.22,
        0.20,
        "Real training data",
        ("fit features and scales", "construct sensitivity controls", "reference for copying audit"),
    )
    box(
        0.025,
        0.36,
        0.22,
        0.20,
        "Held-out real data",
        ("dependence blocks declared", "weighting estimand declared", "never used to fit features"),
    )
    box(
        0.025,
        0.08,
        0.22,
        0.20,
        "Generated sample bank",
        ("same representation and units", "conditions and sample IDs retained", "no silent filtering"),
    )

    box(
        0.345,
        0.52,
        0.29,
        0.32,
        "Registered evidence",
        (
            "physical scalar functionals → W1 in native units",
            "spatial / spectral adapters",
            "training-fitted MMD, density, coverage, diversity",
            "dependence, memorization, and admissibility audits",
        ),
        face=wash,
    )
    box(
        0.345,
        0.09,
        0.29,
        0.32,
        "Dependence-matched real reference",
        (
            "split whole real blocks",
            "match N, strata, and weighting policy",
            "real–real envelope + model–real discrepancy",
            "withhold unstable or control-failed conclusions",
        ),
        edge=warm,
        face="#fbf2ef",
    )

    box(
        0.735,
        0.55,
        0.24,
        0.29,
        "Paper-facing result",
        (
            "raw estimate and interval",
            "reference-calibrated gap profile",
            "conditional matrix and drill-downs",
            "detected / not detected / not evaluated",
        ),
    )
    box(
        0.735,
        0.16,
        0.24,
        0.25,
        "Machine audit",
        (
            "native-unit tables",
            "seeds and partition digests",
            "warnings and control resolution",
            "checksummed JSON/figures",
        ),
        face=wash,
    )

    for y in (0.74, 0.46, 0.18):
        arrow((0.255, y), (0.335, 0.68 if y > 0.5 else 0.25))
    arrow((0.49, 0.50), (0.49, 0.42), color=warm)
    arrow((0.645, 0.68), (0.725, 0.68))
    arrow((0.645, 0.25), (0.725, 0.29))
    ax.text(
        0.5,
        0.025,
        "No aggregate score · no universal equivalence claim · non-detection is not equality",
        ha="center",
        fontsize=9.5,
        color=warm,
        fontweight="bold",
    )
    return _save_figure(fig, output_dir / "main" / "figure_1_framework_overview")


def _copy_validation_figure(output_dir: Path) -> tuple[Path, Path]:
    source = ROOT / "validation" / "statistical_validation_v0.2"
    targets = []
    for suffix in (".pdf", ".png"):
        source_path = Path(f"{source}{suffix}")
        target = output_dir / "main" / f"figure_2_statistical_validation{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        targets.append(target)
    return targets[0], targets[1]


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> float:
    order = np.argsort(values, kind="mergesort")
    sorted_values = np.asarray(values, dtype=float)[order]
    cumulative = np.cumsum(np.asarray(weights, dtype=float)[order])
    cumulative /= cumulative[-1]
    return float(sorted_values[np.searchsorted(cumulative, probability, side="left")])


def _representative_montage(
    output_dir: Path,
    config_path: Path,
    real_values: Mapping[str, np.ndarray],
    generated_values: Mapping[str, np.ndarray],
    real_labels: np.ndarray,
    generated_labels: np.ndarray,
    real_blocks: np.ndarray,
) -> tuple[Path, Path]:
    real, generated, _, _ = _adapter_inputs(config_path)
    selected: list[dict[str, Any]] = []
    real_matrix = np.column_stack([real_values[name] for name in FUNCTIONAL_IDS])
    generated_matrix = np.column_stack(
        [generated_values[name] for name in FUNCTIONAL_IDS]
    )
    for class_id in range(6):
        real_idx = np.flatnonzero(real_labels == class_id)
        generated_idx = np.flatnonzero(generated_labels == class_id)
        weights = _block_balanced_weights(real_blocks[real_idx])
        center = np.asarray(
            [
                _weighted_quantile(real_matrix[real_idx, column], weights, 0.5)
                for column in range(real_matrix.shape[1])
            ]
        )
        deviations = np.abs(real_matrix[real_idx] - center)
        scale = np.asarray(
            [
                _weighted_quantile(deviations[:, column], weights, 0.5)
                for column in range(deviations.shape[1])
            ]
        )
        fallback = np.std(real_matrix[real_idx], axis=0)
        scale = np.where(scale > np.finfo(float).eps, scale, fallback)
        scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
        real_distance = np.sqrt(
            np.mean(((real_matrix[real_idx] - center) / scale) ** 2, axis=1)
        )
        generated_distance = np.sqrt(
            np.mean(((generated_matrix[generated_idx] - center) / scale) ** 2, axis=1)
        )
        for source, index, distance, dataset in (
            (
                "held-out real",
                int(real_idx[int(np.argmin(real_distance))]),
                float(np.min(real_distance)),
                real,
            ),
            (
                "generated",
                int(generated_idx[int(np.argmin(generated_distance))]),
                float(np.min(generated_distance)),
                generated,
            ),
        ):
            sample_id = (
                None
                if dataset.sample_ids is None
                else str(np.asarray(dataset.sample_ids)[index])
            )
            selected.append(
                {
                    "source": source,
                    "class_id": class_id,
                    "class_label": CLASS_LABELS[str(class_id)],
                    "sample_index": index,
                    "sample_id": sample_id,
                    "standardized_functional_distance": distance,
                }
            )

    fig, axes = plt.subplots(2, 6, figsize=(13.1, 4.9))
    fig.subplots_adjust(
        left=0.055,
        right=0.94,
        top=0.83,
        bottom=0.14,
        wspace=0.06,
        hspace=0.10,
    )
    for row, source in enumerate(("held-out real", "generated")):
        for class_id in range(6):
            ax = axes[row, class_id]
            item = next(
                node
                for node in selected
                if node["source"] == source and node["class_id"] == class_id
            )
            dataset = real if source == "held-out real" else generated
            image = np.asarray(dataset.fields[item["sample_index"]])
            shown = ax.imshow(
                image,
                cmap="gray_r",
                vmin=170.0,
                vmax=325.0,
                interpolation="nearest",
            )
            if row == 0:
                ax.set_title(CLASS_LABELS[str(class_id)], fontweight="bold", fontsize=11)
            if class_id == 0:
                ax.set_ylabel("Held-out real" if row == 0 else "Generated", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#d6dee1")
                spine.set_linewidth(0.7)
    colorbar_axis = fig.add_axes((0.955, 0.18, 0.012, 0.60))
    colorbar = fig.colorbar(shown, cax=colorbar_axis)
    colorbar.set_label("Brightness temperature (K)")
    fig.suptitle(
        "Deterministically selected typical fields by intensity class",
        fontsize=17,
        fontweight="semibold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.045,
        (
            "Selection was frozen algorithmically: nearest to the block-balanced "
            "real class median across the seven registered functionals."
        ),
        ha="center",
        fontsize=8.4,
        color="#53636a",
    )
    paths = _save_figure(fig, output_dir / "main" / "figure_3_tc_examples")
    table_path = output_dir / "appendix" / "table_A2_example_selection.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(selected[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
    return paths


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
    base = output_dir.parent / "main" / "figure_5_tail_physical_distributions"
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    artifacts["tail_pdf"] = base.with_suffix(".pdf")
    artifacts["tail_png"] = base.with_suffix(".png")

    distribution_dir = output_dir / "functional_distributions"
    distribution_dir.mkdir(parents=True, exist_ok=True)
    for functional_id in FUNCTIONAL_IDS:
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.8))
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
    fig.text(
        0.5,
        0.035,
        (
            "Shading is a pointwise curve envelope, not a simultaneous band. "
            "Detection uses the registered scalar curve discrepancy reported as the gap."
        ),
        ha="center",
        fontsize=8.2,
        color="#53636a",
    )
    base = output_dir.parent / "main" / "figure_6_tail_spatial_structure"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"structure_pdf": base.with_suffix(".pdf"), "structure_png": base.with_suffix(".png")}


def _all_class_curve_figures(
    report: EvaluationReport,
    output_dir: Path,
) -> dict[str, Path]:
    specifications = (
        (
            "radial_profile_mae",
            "Mean radial BT profile",
            "figure_A3_all_class_radial_profiles",
        ),
        (
            "radial_log_psd_mse",
            "Radial log-power spectrum",
            "figure_A4_all_class_radial_psd",
        ),
    )
    cells = {(cell.stratum, cell.metric): cell for cell in report.cells}
    artifacts: dict[str, Path] = {}
    for metric, title, stem in specifications:
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0))
        fig.subplots_adjust(
            left=0.08,
            right=0.99,
            top=0.88,
            bottom=0.23,
            wspace=0.25,
            hspace=0.34,
        )
        for class_id, ax in enumerate(axes.flat):
            stratum = str(class_id)
            diagnostic = report.diagnostics[f"{stratum}:{metric}"]
            coordinate = np.asarray(diagnostic["coordinate_values"])
            ax.fill_between(
                coordinate,
                diagnostic["reference_q025"],
                diagnostic["reference_q975"],
                color="#b7d3d0",
                alpha=0.55,
                label="pointwise real-reference 95% envelope",
            )
            ax.plot(
                coordinate,
                diagnostic["real_mean"],
                color="#263238",
                linewidth=1.45,
                label="held-out real",
            )
            ax.plot(
                coordinate,
                diagnostic["generated_mean"],
                color="#2b7a78",
                linewidth=1.45,
                label="generated",
            )
            cell = cells[(stratum, metric)]
            ax.set_title(
                f"{CLASS_LABELS[stratum]} · gap {cell.reference_position:.2f}",
                fontweight="semibold",
            )
            ax.set_xlabel(diagnostic["coordinate_label"])
            ax.set_ylabel(diagnostic["units"])
            ax.grid(color="#e0e6e8", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.018),
            ncol=3,
            frameon=False,
        )
        fig.suptitle(
            f"{title} — all native intensity classes",
            fontsize=16,
            fontweight="semibold",
            y=0.98,
        )
        fig.text(
            0.5,
            0.105,
            (
                "Shading is pointwise curve variability; categorical detection uses "
                "the registered scalar curve discrepancy shown as the gap."
            ),
            ha="center",
            fontsize=8.1,
            color="#53636a",
        )
        pdf, png = _save_figure(fig, output_dir / stem, dpi=190)
        artifacts[f"{metric}_pdf"] = pdf
        artifacts[f"{metric}_png"] = png
    return artifacts


def _safeguard_figure(report: EvaluationReport, output_dir: Path) -> dict[str, Path]:
    memorization = report.memorization or {}
    regions = memorization.get("regions", [])
    status_order = ("COPYING_DETECTED", "FARTHER_FROM_TRAINING", "INCONCLUSIVE")
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
    base = output_dir / "figure_A5_generation_audits"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"safeguards_pdf": base.with_suffix(".pdf"), "safeguards_png": base.with_suffix(".png")}


def _extended_diagnostic_figures(
    report: EvaluationReport,
    appendix: Path,
) -> None:
    audit = appendix / "extended_diagnostics"
    audit.mkdir(parents=True, exist_ok=True)
    index = {(cell.stratum, cell.metric): cell for cell in report.cells}
    strata = tuple(map(str, range(6)))

    feature_metrics = (
        "feature_mmd2",
        "feature_density",
        "feature_coverage_deficit",
        "feature_diversity_departure",
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.90,
        bottom=0.10,
        wspace=0.25,
        hspace=0.36,
    )
    for ax, metric in zip(axes.flat, feature_metrics, strict=True):
        cells = [index[(stratum, metric)] for stratum in strata]
        x = np.arange(6)
        positions = np.asarray([cell.reference_position for cell in cells], dtype=float)
        colors = [
            "#b64932"
            if cell.detection_status.value == "DEPARTURE_DETECTED"
            else "#246a73"
            for cell in cells
        ]
        ax.bar(x, positions, color=colors, width=0.68)
        ax.axhline(1.0, color="#a33b20", linestyle="--", linewidth=1.0)
        ax.axhline(0.0, color="#78909a", linewidth=0.8)
        ax.set_xticks(x, [CLASS_LABELS[stratum] for stratum in strata])
        ax.set_ylabel("Reference-calibrated gap")
        ax.set_title(METRIC_LABELS[metric], fontweight="semibold")
        ax.grid(axis="y", color="#e0e6e8", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Training-fitted feature diagnostics",
        fontsize=16,
        fontweight="semibold",
    )
    _save_figure(fig, audit / "track1_feature_diagnostics", dpi=190)

    dependence_metrics = [
        item["id"]
        for item in report.metadata["metric_registry"]
        if item["kind"] == "matrix"
    ]
    values = np.full((len(dependence_metrics), 6), np.nan)
    statuses = np.empty((len(dependence_metrics), 6), dtype=object)
    for row, metric in enumerate(dependence_metrics):
        for column, stratum in enumerate(strata):
            cell = index[(stratum, metric)]
            values[row, column] = cell.reference_position
            statuses[row, column] = cell.detection_status.value
    fig, ax = plt.subplots(figsize=(9.2, 10.6))
    shown = ax.imshow(
        np.clip(values, -1.0, 4.0),
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=4.0,
        aspect="auto",
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            status = statuses[row, column]
            symbol = (
                "×"
                if status == "NOT_EVALUATED"
                else ("●" if status == "DEPARTURE_DETECTED" else "")
            )
            if symbol:
                ax.text(
                    column,
                    row,
                    symbol,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#18212b",
                )
    labels = [
        metric.removeprefix("spearman_pair:").replace(":", " × ")
        for metric in dependence_metrics
    ]
    ax.set_xticks(np.arange(6), [CLASS_LABELS[stratum] for stratum in strata])
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=7.2)
    ax.set_title(
        "Pairwise functional-dependence audit\n"
        "● departure detected · × conclusion withheld",
        fontsize=15,
        fontweight="semibold",
    )
    colorbar = fig.colorbar(shown, ax=ax, pad=0.02)
    colorbar.set_label("Reference-calibrated gap (clipped for display)")
    _save_figure(fig, audit / "track1_dependence", dpi=190)

    regions = list((report.memorization or {}).get("regions", []))
    region_order = sorted(
        regions,
        key=lambda item: (int(item["stratum"]), int(item["region"])),
    )
    fig_height = max(6.0, 0.23 * len(region_order) + 1.8)
    fig, ax = plt.subplots(figsize=(10.0, fig_height))
    y = np.arange(len(region_order))
    effects = np.asarray([item["effect"] for item in region_order], dtype=float)
    low = np.asarray([item["ci_low"] for item in region_order], dtype=float)
    high = np.asarray([item["ci_high"] for item in region_order], dtype=float)
    memorization_colors = {
        "COPYING_DETECTED": "#b83b32",
        "FARTHER_FROM_TRAINING": "#d99b35",
        "INCONCLUSIVE": "#78909c",
    }
    colors = [
        memorization_colors.get(item["status"], "#78909c")
        for item in region_order
    ]
    for row, color in enumerate(colors):
        ax.errorbar(
            effects[row],
            y[row],
            xerr=np.asarray(
                [
                    [effects[row] - low[row]],
                    [high[row] - effects[row]],
                ]
            ),
            fmt="o",
            color=color,
            ecolor=color,
            markersize=3.7,
            elinewidth=1.0,
            capsize=1.5,
            zorder=3,
        )
    ax.axvline(0.0, color="#263238", linewidth=1.0)
    ax.set_yticks(
        y,
        [
            f"{CLASS_LABELS[str(item['stratum'])]} · region {item['region']}"
            for item in region_order
        ],
        fontsize=7.2,
    )
    ax.invert_yaxis()
    ax.set_xlabel(
        "Median generated-to-train NN minus held-out-real-to-train NN distance"
    )
    ax.set_title(
        "Localized, label-matched copying audit\n"
        "negative intervals would indicate copying",
        fontsize=15,
        fontweight="semibold",
    )
    ax.grid(axis="x", color="#e0e6e8", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    statuses_present = {
        str(item["status"])
        for item in region_order
        if str(item["status"]) in memorization_colors
    }
    legend_order = (
        ("COPYING_DETECTED", "copying detected"),
        ("FARTHER_FROM_TRAINING", "farther from training"),
        ("INCONCLUSIVE", "inconclusive"),
    )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color=memorization_colors[status],
                label=label,
                markersize=5,
            )
            for status, label in legend_order
            if status in statuses_present
        ],
        loc="lower right",
        frameon=False,
        fontsize=8,
    )
    _save_figure(fig, audit / "track1_memorization", dpi=180)

    rate_cells = [index[(stratum, "preprocessing_boundary_saturation")] for stratum in strata]
    real_rates = [
        100.0 * float(cell.provenance["raw_real_rate"]) for cell in rate_cells
    ]
    generated_rates = [
        100.0 * float(cell.provenance["raw_generated_rate"]) for cell in rate_cells
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    x = np.arange(6)
    width = 0.36
    axes[0].bar(x - width / 2, real_rates, width, color="#4f5b61", label="held-out real")
    axes[0].bar(x + width / 2, generated_rates, width, color="#2b7a78", label="generated")
    axes[0].set_xticks(x, [CLASS_LABELS[stratum] for stratum in strata])
    axes[0].set_ylabel("Fields touching 170/325 K boundary (%)")
    axes[0].set_title("Raw preprocessing-boundary rates", fontweight="semibold")
    axes[0].legend(frameon=False)
    gaps = np.asarray([cell.reference_position for cell in rate_cells], dtype=float)
    axes[1].bar(x, gaps, color="#638b8f")
    axes[1].axhline(1.0, color="#a33b20", linestyle="--", linewidth=1.0)
    for column, cell in enumerate(rate_cells):
        if cell.detection_status.value == "NOT_EVALUATED":
            axes[1].text(column, gaps[column], "×", ha="center", va="bottom")
    axes[1].set_xticks(x, [CLASS_LABELS[stratum] for stratum in strata])
    axes[1].set_ylabel("Reference-calibrated gap")
    axes[1].set_title("Reference audit · × withheld", fontweight="semibold")
    for ax in axes:
        ax.grid(axis="y", color="#e0e6e8", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Frozen-representation admissibility audit",
        fontsize=16,
        fontweight="semibold",
    )
    _save_figure(fig, audit / "track1_admissibility", dpi=190)


def _copy_sensitivity_artifact(appendix: Path) -> Path:
    source = (
        ROOT
        / "case_studies"
        / "tropical_cyclone_bt"
        / "control_results"
        / "sensitivity_lanes.pdf"
    )
    target = appendix / "figure_A2_sensitivity_controls.pdf"
    shutil.copy2(source, target)
    return target


def _write_metric_registry(output_dir: Path, report: EvaluationReport) -> None:
    registry = {
        str(item["id"]): item for item in report.metadata.get("metric_registry", [])
    }
    controls = {
        "w1:min_core_bt": "uniform BT shift",
        "w1:cold_frac_220": "uniform BT shift",
        "w1:p01_bt": "uniform BT shift",
        "w1:r_maxgrad": "blur and center jitter",
        "w1:axisym": "phase shuffle",
        "w1:dav": "phase shuffle",
        "w1:eye_contrast": "uniform BT shift",
        "radial_profile_mae": "center jitter and phase shuffle",
        "radial_log_psd_mse": "blur and phase-power preservation",
        "feature_mmd2": "training-fitted feature audit",
        "feature_density": "training-fitted feature audit",
        "feature_coverage_deficit": "training-fitted feature audit",
        "feature_diversity_departure": "training-fitted feature audit",
    }
    rows = []
    for metric in MAIN_METRICS:
        item = registry[metric]
        functional = item.get("functional")
        rows.append(
            {
                "metric": metric,
                "paper_label": METRIC_LABELS[metric],
                "family": item["kind"],
                "units": item["units"],
                "definition": item["description"],
                "physical_quantity": (
                    "" if functional is None else functional["description"]
                ),
                "training_only_or_fixed_preprocessing": (
                    "fixed TC geometry/preprocessing"
                    if item["kind"] in {"functional_w1", "curve"}
                    else "standardization, EOFs, and bandwidth fitted on training only"
                ),
                "registered_control": controls[metric],
            }
        )
    main = output_dir / "main"
    csv_path = main / "table_1_metric_registry.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    markdown = [
        "# Table 1. Registered TC evaluation metrics",
        "",
        "| Metric | Family | Units | Definition | Control / provenance |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['paper_label']} | {row['family']} | {row['units']} | "
            f"{row['definition']} | {row['registered_control']} |"
        )
    markdown.append("")
    (main / "table_1_metric_registry.md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )


def _write_partition_tables(output_dir: Path) -> None:
    source = (
        ROOT
        / "case_studies"
        / "tropical_cyclone_bt"
        / "partition_summary.csv"
    )
    target = output_dir / "appendix" / "table_A3_partition_summary.csv"
    shutil.copy2(source, target)
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    markdown = [
        "# Dataset partitions",
        "",
        "| Split | Class | Frames | Storms with frames |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['split']} | {row['class_name']} | {int(row['frames']):,} | "
            f"{int(row['storms_with_frames']):,} |"
        )
    markdown.append("")
    (output_dir / "appendix" / "table_A3_partition_summary.md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )


def _write_vmax_table(output_dir: Path) -> None:
    source = (
        ROOT
        / "case_studies"
        / "tropical_cyclone_bt"
        / "vmax_support_summary.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    classes = payload["banks"]["heldout_test"]["classes"]
    rows = []
    for class_id in map(str, range(6)):
        node = classes[class_id]
        rows.append(
            {
                "class": CLASS_LABELS[class_id],
                "generated_n": node["generated"]["n"],
                "generated_unique_vmax": node["generated"]["n_unique"],
                "real_n": node["real_reference"]["n"],
                "real_unique_vmax": node["real_reference"]["n_unique"],
                "generated_median_kt": node["generated"]["median_kt"],
                "real_median_kt": node["real_reference"]["median_kt"],
                "distribution_total_variation": node[
                    "discrete_distribution_total_variation"
                ],
                "generated_support_subset_of_real": node[
                    "generated_support_subset_of_reference"
                ],
            }
        )
    csv_path = output_dir / "appendix" / "table_A4_vmax_support.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    markdown = [
        "# Vmax conditioning-support audit",
        "",
        "| Class | Generated/real unique Vmax | Generated/real median (kt) | TV distance | Generated support ⊆ real |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['class']} | {row['generated_unique_vmax']}/"
            f"{row['real_unique_vmax']} | {row['generated_median_kt']}/"
            f"{row['real_median_kt']} | "
            f"{row['distribution_total_variation']:.3f} | "
            f"{'yes' if row['generated_support_subset_of_real'] else 'no'} |"
        )
    markdown.append("")
    (output_dir / "appendix" / "table_A4_vmax_support.md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )


def _write_readme(output_dir: Path, report_path: Path, report: EvaluationReport) -> None:
    text = f"""# TC GAUGE paper-facing results

This is the complete manuscript figure/table bundle generated from one clean
current-engine report. The main directory contains the framework, statistical
validation, case-study, and headline-result artifacts. The appendix retains the
complete class, curve, sensitivity, copying, dependence, admissibility, data,
and provenance evidence.

## Main-text artifacts

- `main/figure_1_framework_overview.pdf`: GAUGE data, calibration, metric, and
  reporting workflow.
- `main/figure_2_statistical_validation.pdf`: Monte Carlo calibration and power,
  including the 22-block C5-like regime.
- `main/figure_3_tc_examples.pdf`: deterministically selected typical real and
  generated fields for TS and Categories 1--5.
- `main/figure_4_reference_gap_profile.pdf`: primary intrinsic result. Every
  native class is visible and training-only positive controls are open diamonds.
- `main/figure_5_tail_physical_distributions.pdf`: Category 5 empirical distributions
  for all seven physical functionals.
- `main/figure_6_tail_spatial_structure.pdf`: Category 5 radial-profile and spectral
  drill-downs.
- `main/table_1_metric_registry.md`: definitions, units, roles, and registered
  controls for the TC metric suite.
- `main/table_2_native_metric_values.csv`: exact native-unit estimates,
  intervals, reference summaries, gaps, and statuses for every displayed class.

The appendix includes the complete conditional matrix, sensitivity lanes,
all-class radial-profile and PSD figures, copying/preprocessing audits, every
native-class functional distribution, extended feature/dependence diagnostics,
partition and Vmax-support tables, deterministic example selection, and every
report cell.

## Reporting language

- Gap 0 is the median real-vs-real discrepancy; gap 1 is the pointwise 95%
  real-reference boundary.
- Gap units express detectability at the declared sample size, not physical
  severity, and must not be averaged into a model score.
- A point below 1 supports “no departure detected,” not equality.
- A `NOT_EVALUATED` cell retains its raw audit value but supports no categorical
  conclusion.
- The unmatched global mixture is deliberately absent from the fidelity figure;
  it remains in the complete audit table as a dataset-composition check.
- Curve shading is pointwise variability; curve detection uses the registered
  scalar curve discrepancy and is not inferred by visually counting excursions.

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
    appendix = output_dir / "appendix"
    csv_path = appendix / "table_A5_c5_values.csv"
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
    (appendix / "table_A5_c5_values.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )


def _write_index(output_dir: Path, report: EvaluationReport) -> None:
    figures = (
        (
            "main/figure_1_framework_overview.png",
            "1. GAUGE framework",
            "From declared data and dependence blocks to registered evidence, real-reference calibration, and an interpretation-first paper bundle.",
        ),
        (
            "main/figure_2_statistical_validation.png",
            "2. Statistical validation",
            "False-positive behavior, power, alternative distributions, familywise inference, and the explicit resolution loss in C5-like thin strata.",
        ),
        (
            "main/figure_3_tc_examples.png",
            "3. Tropical-cyclone fields",
            "Typical held-out and generated fields selected by a deterministic seven-functional rule rather than visual preference.",
        ),
        (
            "main/figure_4_reference_gap_profile.png",
            "4. Reference-calibrated intrinsic profile",
            "All six native intensity classes and training-only positive controls on one scale; no pooled tail can hide C5.",
        ),
        (
            "main/figure_5_tail_physical_distributions.png",
            "5. Tail physical distributions",
            "Block-balanced Category 5 empirical distributions for every scalar physical functional; interpret with its thin-stratum uncertainty.",
        ),
        (
            "main/figure_6_tail_spatial_structure.png",
            "6. Tail spatial structure",
            "Category 5 radial-profile and power-spectrum drill-downs; pointwise curve shading is distinguished from scalar curve detection.",
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
<ul><li><a href="main/table_1_metric_registry.md">Registered metric definitions</a></li>
<li><a href="main/table_2_native_metric_values.csv">Every displayed class and metric (CSV)</a></li>
<li><a href="appendix/figure_A1_conditional_gap_matrix.pdf">Complete conditional matrix (PDF)</a></li>
<li><a href="appendix/figure_A2_sensitivity_controls.pdf">Training-only sensitivity controls (PDF)</a></li>
<li><a href="appendix/figure_A3_all_class_radial_profiles.pdf">All-class radial profiles (PDF)</a></li>
<li><a href="appendix/figure_A4_all_class_radial_psd.pdf">All-class radial PSDs (PDF)</a></li>
<li><a href="appendix/figure_A5_generation_audits.pdf">Copying and preprocessing audits (PDF)</a></li>
<li><a href="appendix/table_A1_complete_evaluation.csv">Every report cell (CSV)</a></li>
<li><a href="appendix/functional_distributions/">Native-class distribution figures</a></li>
<li><a href="README.md">Reporting guide and permitted language</a></li></ul>
<p>Frozen configuration hash: <code>{report.metadata.get('config_hash', 'unknown')}</code>.</p>
</body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def _write_manifest(
    output_dir: Path,
    report_path: Path,
    report: EvaluationReport,
    *,
    config_path: Path,
    controls_path: Path,
) -> None:
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
        "config_hash": report.metadata.get("config_hash"),
        "source_config_sha256": _sha256(config_path),
        "source_controls_sha256": _sha256(controls_path),
        "renderer_sha256": _sha256(Path(__file__)),
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
    controls_path = args.controls.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = EvaluationReport.from_json(report_path)
    controls_payload = json.loads(controls_path.read_text(encoding="utf-8"))
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
    for suffix in (".pdf", ".png"):
        (
            output_dir / "main" / f"figure_1_reference_gap_profile{suffix}"
        ).replace(
            output_dir / "main" / f"figure_4_reference_gap_profile{suffix}"
        )
    (
        output_dir / "main" / "table_1_native_metric_values.csv"
    ).replace(
        output_dir / "main" / "table_2_native_metric_values.csv"
    )
    _framework_overview(output_dir)
    _copy_validation_figure(output_dir)

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
    _representative_montage(
        output_dir,
        config_path,
        real_values,
        generated_values,
        real_labels,
        generated_labels,
        real_blocks,
    )
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
    _all_class_curve_figures(report, appendix)
    _safeguard_figure(report, appendix)
    _extended_diagnostic_figures(report, appendix)
    _copy_sensitivity_artifact(appendix)
    _write_metric_registry(output_dir, report)
    _write_partition_tables(output_dir)
    _write_vmax_table(output_dir)
    _write_c5_table(output_dir, report)
    _write_readme(output_dir, report_path, report)
    _write_index(output_dir, report)
    _write_manifest(
        output_dir,
        report_path,
        report,
        config_path=config_path,
        controls_path=controls_path,
    )
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
