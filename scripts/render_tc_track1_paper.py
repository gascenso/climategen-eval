#!/usr/bin/env python3
"""Render the paper-facing TC bundle from one finalized GAUGE report."""

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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

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
TAIL_FUNCTIONAL_LABELS = {
    "min_core_bt": "Core minimum\nBT",
    "cold_frac_220": "Cold-cloud area\n< 220 K",
    "r_maxgrad": "Maximum-gradient\nradius",
    "axisym": "Axisymmetry\ndeparture",
    "dav": "Deviation-angle\nvariance",
    "eye_contrast": "Eye minus\neyewall BT",
    "p01_bt": "Per-image BT\nfirst percentile",
}
REAL_CURVE_COLOR = "#263238"
GENERATED_CURVE_COLOR = "#c45b3c"
REAL_ENVELOPE_COLOR = "#b8c5cc"


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
        "warm_shift": "Temperature shift",
        "threshold_warming": "Threshold warming",
        "threshold_cooling": "Threshold cooling",
        "eye_contrast": "Eye/eyewall perturbation",
        "blur": "Gaussian blur",
        "checkerboard": "Spatial artifact",
        "phase_shuffle": "Phase randomisation",
        "feature_eof_excursion": "EOF excursion",
        "feature_mode_removal": "Mode removal",
        "feature_contraction": "Feature contraction",
        "feature_expansion": "Feature expansion",
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
                    provenance="training-only registered control",
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
    outputs = {functional.id: np.empty(len(fields), dtype=np.float64) for functional in functionals}
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
        "schema": "gauge_climate.tc_bt_presentation_cache.v2",
        "real_manifest": str(config_payload["data"]["real"]["tc_real_manifest"]),
        "generated_bank": str(config_payload["data"]["generated"]["tc_sample_bank"]),
        "adapter_options": config_payload["adapter_options"],
        "functional_ids": list(FUNCTIONAL_IDS),
        "functional_source_sha256": _sha256(
            ROOT / "src" / "gauge_climate" / "adapters" / "tc_bt.py"
        ),
    }
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        compatible_legacy = (
            metadata.get("schema") == "gauge_climate.tc_bt_presentation_cache.v1"
            and metadata.get("real_manifest") == expected["real_manifest"]
            and metadata.get("generated_bank") == expected["generated_bank"]
            and metadata.get("functional_ids") == expected["functional_ids"]
        )
        if metadata == expected or compatible_legacy:
            archive = np.load(cache_path)
            real_values = {name: np.asarray(archive[f"real_{name}"]) for name in FUNCTIONAL_IDS}
            generated_values = {
                name: np.asarray(archive[f"generated_{name}"]) for name in FUNCTIONAL_IDS
            }
            result = (
                real_values,
                generated_values,
                np.asarray(archive["real_labels"]),
                np.asarray(archive["generated_labels"]),
                np.asarray(archive["real_blocks"]),
            )
            if compatible_legacy:
                metadata_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
            return result

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
    """Render a reader-first GAUGE flowchart at its intended journal dimensions."""

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ink = "#22323A"
    muted = "#5F7077"
    teal = "#176B73"
    teal_dark = "#0F5660"
    teal_wash = "#EAF3F2"
    orange = "#C65A3A"
    orange_wash = "#FBF0EC"
    purple = "#77628F"
    purple_wash = "#F1EEF6"
    grey = "#85949A"
    grey_wash = "#F2F5F5"
    rule = "#CBD6D9"

    def rounded_box(
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str,
        edge: str,
        radius: float = 0.012,
        shadow: bool = True,
    ) -> None:
        if shadow:
            ax.add_patch(
                FancyBboxPatch(
                    (x + 0.003, y - 0.003),
                    width,
                    height,
                    boxstyle=f"round,pad=0.008,rounding_size={radius}",
                    linewidth=0,
                    facecolor=ink,
                    alpha=0.06,
                    zorder=1,
                )
            )
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle=f"round,pad=0.008,rounding_size={radius}",
                linewidth=0.9,
                edgecolor=edge,
                facecolor=fill,
                zorder=2,
            )
        )

    def section_label(number: str, label: str, y: float) -> None:
        ax.text(
            0.030,
            y,
            number,
            fontsize=7.0,
            fontweight="bold",
            color="#FFFFFF",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "circle,pad=0.30",
                "facecolor": teal_dark,
                "edgecolor": "none",
            },
            zorder=5,
        )
        ax.text(
            0.052,
            y,
            label,
            fontsize=7.2,
            fontweight="bold",
            color=teal_dark,
            ha="left",
            va="center",
            zorder=5,
        )

    def data_card(
        x: float,
        title: str,
        body: Sequence[str],
        *,
        color: str,
        fill: str,
    ) -> None:
        y, width, height = 0.835, 0.280, 0.105
        rounded_box(x, y, width, height, fill=fill, edge=color)
        ax.text(
            x + width / 2,
            y + 0.075,
            title,
            fontsize=8.0,
            fontweight="semibold",
            color=ink,
            ha="center",
            va="center",
            zorder=4,
        )
        ax.plot(
            [x + 0.055, x + width - 0.055],
            [y + 0.057, y + 0.057],
            color=color,
            linewidth=1.1,
            zorder=4,
        )
        ax.text(
            x + width / 2,
            y + 0.035,
            "\n".join(body),
            fontsize=6.5,
            color=muted,
            ha="center",
            va="center",
            linespacing=1.22,
            zorder=4,
        )

    def arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str = teal,
        width: float = 0.95,
        dashed: bool = False,
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=width,
                color=color,
                linestyle=(0, (3, 2)) if dashed else "solid",
                shrinkA=0,
                shrinkB=0,
                zorder=0,
            )
        )

    def route(
        points: Sequence[tuple[float, float]],
        *,
        color: str = teal,
        width: float = 0.95,
    ) -> None:
        xs, ys = zip(*points, strict=True)
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=width,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=0,
        )
        arrow(points[-2], points[-1], color=color, width=width)

    def decision(
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        text: str,
    ) -> None:
        vertices = np.asarray(
            [
                (center_x, center_y + height / 2),
                (center_x + width / 2, center_y),
                (center_x, center_y - height / 2),
                (center_x - width / 2, center_y),
            ]
        )
        ax.add_patch(
            Polygon(
                vertices + np.asarray([0.003, -0.003]),
                closed=True,
                facecolor=ink,
                edgecolor="none",
                alpha=0.06,
                zorder=1,
            )
        )
        ax.add_patch(
            Polygon(
                vertices,
                closed=True,
                facecolor="#FFFFFF",
                edgecolor=teal,
                linewidth=0.95,
                zorder=2,
            )
        )
        ax.text(
            center_x,
            center_y,
            text,
            fontsize=7.0,
            fontweight="semibold",
            color=ink,
            ha="center",
            va="center",
            linespacing=1.15,
            zorder=4,
        )

    def branch_label(x: float, y: float, text: str, *, color: str) -> None:
        ax.text(
            x,
            y,
            text,
            fontsize=6.1,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "#FFFFFF",
                "edgecolor": "none",
            },
            zorder=5,
        )

    section_label("1", "DATA ROLES AND DECLARED SCIENTIFIC CHOICES", 0.972)
    data_card(
        0.035,
        "TRAINING DATA",
        ("fit preprocessing and features", "construct named positive controls"),
        color=purple,
        fill=purple_wash,
    )
    data_card(
        0.360,
        "HELD-OUT REAL DATA",
        ("group repeated frames by event", "use only for final evaluation"),
        color=teal,
        fill=teal_wash,
    )
    data_card(
        0.685,
        "GENERATED FIELDS",
        ("provide the target conditions", "evaluate after the design is frozen"),
        color=orange,
        fill=orange_wash,
    )

    rounded_box(
        0.105,
        0.738,
        0.790,
        0.060,
        fill="#FFFFFF",
        edge=rule,
        radius=0.010,
    )
    ax.text(
        0.500,
        0.779,
        "DECLARE THE ANALYSIS AND RECORD ANY AMENDMENTS",
        fontsize=6.7,
        fontweight="bold",
        color=ink,
        ha="center",
        va="center",
    )
    ax.text(
        0.500,
        0.756,
        "metrics and units  ·  conditions  ·  event/frame weighting  ·  sample size and repetitions",
        fontsize=5.9,
        color=muted,
        ha="center",
        va="center",
    )
    for x, color in ((0.175, purple), (0.500, teal), (0.825, orange)):
        arrow((x, 0.835), (x, 0.798), color=color)

    decision(
        0.500,
        0.675,
        0.300,
        0.060,
        "Enough independent real events\nin this condition?",
    )
    arrow((0.500, 0.738), (0.500, 0.705))

    rounded_box(
        0.025,
        0.615,
        0.165,
        0.062,
        fill=grey_wash,
        edge=rule,
        radius=0.025,
    )
    ax.text(
        0.108,
        0.654,
        "NOT EVALUATED",
        fontsize=6.7,
        fontweight="bold",
        color=ink,
        ha="center",
        va="center",
    )
    ax.text(
        0.108,
        0.631,
        "retain counts\nand keep condition visible",
        fontsize=5.7,
        color=muted,
        ha="center",
        va="center",
        linespacing=1.15,
    )
    route(
        [(0.350, 0.675), (0.235, 0.675), (0.235, 0.646), (0.190, 0.646)],
        color=grey,
    )
    branch_label(0.278, 0.675, "NO", color=grey)

    section_label("2", "TWO MATCHED QUESTIONS FOR EACH METRIC × CONDITION", 0.600)
    route(
        [(0.500, 0.645), (0.500, 0.580), (0.335, 0.580), (0.335, 0.565)],
        color=teal,
    )
    route(
        [(0.500, 0.580), (0.665, 0.580), (0.665, 0.565)],
        color=orange,
    )
    branch_label(0.522, 0.635, "YES", color=teal_dark)

    rounded_box(
        0.185,
        0.420,
        0.300,
        0.145,
        fill=teal_wash,
        edge="#AACBC8",
        radius=0.012,
    )
    ax.text(
        0.335,
        0.540,
        "REAL ↔ REAL",
        fontsize=8.4,
        fontweight="bold",
        color=teal_dark,
        ha="center",
        va="center",
    )
    ax.text(
        0.335,
        0.506,
        "What difference is ordinary\nbetween two real samples?",
        fontsize=6.3,
        fontweight="semibold",
        color=ink,
        ha="center",
        va="center",
        linespacing=1.15,
    )
    ax.text(
        0.335,
        0.466,
        "Real A and Real B share no events\nrepeat with the declared N and weighting",
        fontsize=5.9,
        color=muted,
        ha="center",
        va="center",
        linespacing=1.24,
    )
    ax.text(
        0.335,
        0.435,
        "R = real-reference distribution",
        fontsize=6.1,
        fontweight="semibold",
        color=teal_dark,
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "#FFFFFF",
            "edgecolor": "none",
        },
    )

    rounded_box(
        0.515,
        0.420,
        0.300,
        0.145,
        fill=orange_wash,
        edge="#E1B6A7",
        radius=0.012,
    )
    ax.text(
        0.665,
        0.540,
        "GENERATED ↔ REAL",
        fontsize=8.4,
        fontweight="bold",
        color=orange,
        ha="center",
        va="center",
    )
    ax.text(
        0.665,
        0.506,
        "How different is the generator\nfrom held-out real data?",
        fontsize=6.3,
        fontweight="semibold",
        color=ink,
        ha="center",
        va="center",
        linespacing=1.15,
    )
    ax.text(
        0.665,
        0.466,
        "same metric, N, condition and weighting\nresample real events; keep the bank fixed",
        fontsize=5.9,
        color=muted,
        ha="center",
        va="center",
        linespacing=1.24,
    )
    ax.text(
        0.665,
        0.435,
        "M = model–real discrepancy",
        fontsize=6.1,
        fontweight="semibold",
        color=orange,
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "#FFFFFF",
            "edgecolor": "none",
        },
    )

    rounded_box(
        0.200,
        0.285,
        0.600,
        0.095,
        fill="#FFFFFF",
        edge=rule,
        radius=0.012,
    )
    ax.text(
        0.500,
        0.362,
        "3   INTERPRET M USING THE REAL-DATA REFERENCE R",
        fontsize=7.2,
        fontweight="bold",
        color=teal_dark,
        ha="center",
        va="center",
    )
    ax.plot([0.500, 0.500], [0.298, 0.346], color=rule, linewidth=0.8)
    ax.text(
        0.350,
        0.334,
        "NATIVE EVIDENCE",
        fontsize=6.4,
        fontweight="bold",
        color=ink,
        ha="center",
        va="center",
    )
    ax.text(
        0.350,
        0.310,
        "M [central comparison range]\nin K, km, %, or metric units",
        fontsize=6.0,
        color=muted,
        ha="center",
        va="center",
        linespacing=1.18,
    )
    ax.text(
        0.650,
        0.334,
        "REFERENCE GAP",
        fontsize=6.4,
        fontweight="bold",
        color=ink,
        ha="center",
        va="center",
    )
    ax.text(
        0.650,
        0.311,
        r"$g=(M-\mathrm{median}(R))/(q_{0.95}(R)-\mathrm{median}(R))$",
        fontsize=5.8,
        color=muted,
        ha="center",
        va="center",
    )
    ax.text(
        0.650,
        0.293,
        "0 = median real–real  ·  1 = empirical 95th percentile",
        fontsize=5.5,
        color=muted,
        ha="center",
        va="center",
    )
    route([(0.335, 0.420), (0.335, 0.400), (0.450, 0.400), (0.450, 0.380)])
    route(
        [(0.665, 0.420), (0.665, 0.400), (0.550, 0.400), (0.550, 0.380)],
        color=orange,
    )

    decision(
        0.330,
        0.225,
        0.220,
        0.055,
        "Reference stable\nand non-degenerate?",
    )
    decision(
        0.670,
        0.225,
        0.220,
        0.055,
        "Is M above q95 of the\nempirical real reference?",
    )
    arrow((0.500, 0.285), (0.330, 0.253))

    rounded_box(
        0.055,
        0.128,
        0.270,
        0.045,
        fill=grey_wash,
        edge=rule,
        radius=0.023,
    )
    ax.text(
        0.190,
        0.151,
        "REFERENCE UNRESOLVED",
        fontsize=6.4,
        fontweight="bold",
        color=ink,
        ha="center",
        va="center",
    )
    rounded_box(
        0.365,
        0.128,
        0.270,
        0.045,
        fill=teal_wash,
        edge="#AACBC8",
        radius=0.023,
    )
    ax.text(
        0.500,
        0.151,
        "WITHIN REFERENCE RANGE",
        fontsize=6.4,
        fontweight="bold",
        color=teal_dark,
        ha="center",
        va="center",
    )
    rounded_box(
        0.675,
        0.128,
        0.270,
        0.045,
        fill=orange_wash,
        edge="#E1B6A7",
        radius=0.023,
    )
    ax.text(
        0.810,
        0.151,
        "ABOVE REFERENCE RANGE",
        fontsize=6.4,
        fontweight="bold",
        color=orange,
        ha="center",
        va="center",
    )
    route([(0.220, 0.225), (0.190, 0.225), (0.190, 0.173)], color=grey)
    arrow((0.440, 0.225), (0.560, 0.225), color=teal)
    route([(0.670, 0.198), (0.500, 0.198), (0.500, 0.173)], color=teal)
    route([(0.780, 0.225), (0.810, 0.225), (0.810, 0.173)], color=orange)
    branch_label(0.190, 0.195, "NO", color=grey)
    branch_label(0.500, 0.240, "YES", color=teal_dark)
    branch_label(0.570, 0.188, "NO", color=teal_dark)
    branch_label(0.810, 0.195, "YES", color=orange)

    rounded_box(
        0.145,
        0.015,
        0.710,
        0.090,
        fill=grey_wash,
        edge=rule,
        radius=0.018,
    )
    ax.text(
        0.500,
        0.087,
        "4   GAUGE REPORT — ONE RESULT FOR EVERY METRIC × CONDITION",
        fontsize=6.9,
        fontweight="bold",
        color=ink,
        ha="center",
        va="center",
    )
    ax.text(
        0.500,
        0.061,
        "gap profile + reference status  ·  native values, ranges and curves  ·  condition matrix",
        fontsize=5.9,
        color=muted,
        ha="center",
        va="center",
    )
    ax.text(
        0.500,
        0.038,
        "metric controls  ·  diversity/coverage  ·  preprocessing saturation  ·  localized memorization",
        fontsize=5.9,
        color=muted,
        ha="center",
        va="center",
    )
    ax.text(
        0.500,
        0.019,
        "sample and event counts  ·  reference stability  ·  weighting and provenance  ·  no aggregate score",
        fontsize=5.6,
        color=teal_dark,
        ha="center",
        va="center",
    )
    route([(0.190, 0.128), (0.190, 0.112), (0.380, 0.112), (0.380, 0.105)], color=grey)
    route([(0.500, 0.128), (0.500, 0.105)], color=teal)
    route(
        [(0.810, 0.128), (0.810, 0.112), (0.620, 0.112), (0.620, 0.105)],
        color=orange,
    )
    route(
        [(0.108, 0.615), (0.012, 0.615), (0.012, 0.060), (0.145, 0.060)],
        color=grey,
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
    generated_matrix = np.column_stack([generated_values[name] for name in FUNCTIONAL_IDS])
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
        real_distance = np.sqrt(np.mean(((real_matrix[real_idx] - center) / scale) ** 2, axis=1))
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
                None if dataset.sample_ids is None else str(np.asarray(dataset.sample_ids)[index])
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

    fig, axes = plt.subplots(2, 6, figsize=(11.3, 4.25))
    fig.subplots_adjust(
        left=0.055,
        right=0.94,
        top=0.94,
        bottom=0.06,
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
                ax.set_title(CLASS_LABELS[str(class_id)], fontweight="bold", fontsize=13.5)
            if class_id == 0:
                ax.set_ylabel("Held-out real" if row == 0 else "Generated", fontsize=12.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#d6dee1")
                spine.set_linewidth(0.7)
    colorbar_axis = fig.add_axes((0.955, 0.10, 0.012, 0.76))
    colorbar = fig.colorbar(shown, cax=colorbar_axis)
    colorbar.set_label("Brightness temperature (K)", fontsize=11.5)
    colorbar.ax.tick_params(labelsize=10.5)
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
    ax.step(
        real_x,
        real_y,
        where="post",
        color=REAL_CURVE_COLOR,
        linewidth=1.65,
        label="held-out real",
    )
    ax.step(
        generated_x,
        generated_y,
        where="post",
        color=GENERATED_CURVE_COLOR,
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

    fig, axes = plt.subplots(2, 4, figsize=(10.8, 5.65))
    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        top=0.96,
        bottom=0.08,
        wspace=0.31,
        hspace=0.48,
    )
    tail_real = real_labels == 5
    tail_generated = generated_labels == 5
    for ax, functional_id in zip(axes.flat, FUNCTIONAL_IDS, strict=False):
        cell = cells[("5", f"w1:{functional_id}")]
        _ecdf_panel(
            ax,
            real_values[functional_id][tail_real],
            generated_values[functional_id][tail_generated],
            real_blocks[tail_real],
            title=TAIL_FUNCTIONAL_LABELS[functional_id],
            units=cell.units,
        )
        ax.title.set_fontsize(11.5)
        ax.title.set_linespacing(1.05)
        ax.xaxis.label.set_size(11)
        ax.yaxis.label.set_size(11)
        ax.tick_params(axis="both", labelsize=10.5)
        ax.text(
            0.98,
            0.06,
            f"W1 = {cell.estimate:.3g} {cell.units}\ngap = {cell.reference_gap:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10.5,
            color="#53636a",
        )
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False, fontsize=11)
    base = output_dir.parent / "main" / "figure_5_tail_physical_distributions"
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        base.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    artifacts["tail_pdf"] = base.with_suffix(".pdf")
    artifacts["tail_png"] = base.with_suffix(".png")

    distribution_dir = output_dir / "functional_distributions"
    distribution_dir.mkdir(parents=True, exist_ok=True)
    for functional_id in FUNCTIONAL_IDS:
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.8))
        fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.12, wspace=0.27, hspace=0.38)
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
                f"W1 {cell.estimate:.3g} · gap {cell.reference_gap:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.6,
                color="#53636a",
            )
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=2, frameon=False
        )
        base = distribution_dir / functional_id
        fig.savefig(
            base.with_suffix(".pdf"),
            bbox_inches="tight",
            metadata={"CreationDate": None, "ModDate": None},
        )
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
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.13, wspace=0.24)
    for ax, (metric, label) in zip(axes, metrics, strict=True):
        diagnostic = report.diagnostics[f"5:{metric}"]
        x = np.asarray(diagnostic["coordinate_values"])
        envelope = ax.fill_between(
            x,
            diagnostic["reference_q025"],
            diagnostic["reference_q975"],
            color=REAL_ENVELOPE_COLOR,
            alpha=0.52,
            label="real-reference 95% envelope",
        )
        real_line = ax.plot(
            x,
            diagnostic["real_mean"],
            color=REAL_CURVE_COLOR,
            linewidth=1.8,
            label="held-out real",
        )[0]
        generated_line = ax.plot(
            x,
            diagnostic["generated_mean"],
            color=GENERATED_CURVE_COLOR,
            linewidth=1.8,
            label="generated",
        )[0]
        cell = cells[("5", metric)]
        ax.set_title(label, fontweight="semibold")
        ax.set_xlabel(diagnostic["coordinate_label"])
        ax.set_ylabel(diagnostic["units"])
        ax.grid(color="#e0e6e8", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.98,
            0.96,
            f"gap = {cell.reference_gap:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="#53636a",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        )
    fig.legend(
        (real_line, envelope, generated_line),
        ("held-out real", "real-reference 95% envelope", "generated"),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=False,
    )
    base = output_dir.parent / "main" / "figure_6_tail_spatial_structure"
    fig.savefig(
        base.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
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
            top=0.98,
            bottom=0.13,
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
                color=REAL_ENVELOPE_COLOR,
                alpha=0.52,
                label="real-vs-real 95% envelope",
            )
            ax.plot(
                coordinate,
                diagnostic["real_mean"],
                color=REAL_CURVE_COLOR,
                linewidth=1.45,
                label="held-out real",
            )
            ax.plot(
                coordinate,
                diagnostic["generated_mean"],
                color=GENERATED_CURVE_COLOR,
                linewidth=1.45,
                label="generated",
            )
            cell = cells[(stratum, metric)]
            ax.set_title(
                f"{CLASS_LABELS[stratum]} · gap {cell.reference_gap:.2f}",
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
        pdf, png = _save_figure(fig, output_dir / stem, dpi=190)
        artifacts[f"{metric}_pdf"] = pdf
        artifacts[f"{metric}_png"] = png
    return artifacts


def _safeguard_figure(report: EvaluationReport, output_dir: Path) -> dict[str, Path]:
    memorization = report.memorization or {}
    regions = memorization.get("regions", [])
    status_order = (
        "CLOSER_TO_TRAINING",
        "FARTHER_FROM_TRAINING",
        "CENTRAL_RANGE_OVERLAPS_ZERO",
    )
    status_labels = (
        "closer to training",
        "farther from training",
        "range overlaps zero",
    )
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
        100.0 * float(cells[stratum].provenance["raw_generated_rate"]) for stratum in strata
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.16, wspace=0.34)
    left, right = axes
    left.barh(np.arange(3), counts, color=status_colors, height=0.58)
    left.set_yticks(np.arange(3), status_labels)
    left.invert_yaxis()
    left.set_xlabel("Number of intensity/feature neighbourhoods")
    left.set_title("Localized copying audit", fontweight="semibold")
    for row, count in enumerate(counts):
        left.text(count + 0.5, row, str(count), va="center", fontsize=10)
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
    base = output_dir / "figure_A5_generation_audits"
    fig.savefig(
        base.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
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
        positions = np.asarray([cell.reference_gap for cell in cells], dtype=float)
        colors = [
            "#b64932" if cell.reference_status.value == "ABOVE_REFERENCE_RANGE" else "#246a73"
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
    _save_figure(fig, audit / "track1_feature_diagnostics", dpi=190)

    dependence_metrics = [
        item["id"] for item in report.metadata["metric_registry"] if item["kind"] == "matrix"
    ]
    values = np.full((len(dependence_metrics), 6), np.nan)
    statuses = np.empty((len(dependence_metrics), 6), dtype=object)
    for row, metric in enumerate(dependence_metrics):
        for column, stratum in enumerate(strata):
            cell = index[(stratum, metric)]
            values[row, column] = cell.reference_gap
            statuses[row, column] = cell.reference_status.value
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
                else ("●" if status == "ABOVE_REFERENCE_RANGE" else "")
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
        metric.removeprefix("spearman_pair:").replace(":", " × ") for metric in dependence_metrics
    ]
    ax.set_xticks(np.arange(6), [CLASS_LABELS[stratum] for stratum in strata])
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=7.2)
    ax.set_title(
        "Pairwise functional-dependence audit\n● above reference range · × reference unresolved",
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
    low = np.asarray([item["comparison_low"] for item in region_order], dtype=float)
    high = np.asarray([item["comparison_high"] for item in region_order], dtype=float)
    memorization_colors = {
        "CLOSER_TO_TRAINING": "#b83b32",
        "FARTHER_FROM_TRAINING": "#d99b35",
        "CENTRAL_RANGE_OVERLAPS_ZERO": "#78909c",
    }
    colors = [memorization_colors.get(item["status"], "#78909c") for item in region_order]
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
    ax.set_xlabel("Median generated-to-train NN minus held-out-real-to-train NN distance")
    ax.set_title(
        "Localized, label-matched copying audit\n"
        "negative central ranges are consistent with localized replay",
        fontsize=15,
        fontweight="semibold",
    )
    ax.grid(axis="x", color="#e0e6e8", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    statuses_present = {
        str(item["status"]) for item in region_order if str(item["status"]) in memorization_colors
    }
    legend_order = (
        ("CLOSER_TO_TRAINING", "closer to training"),
        ("FARTHER_FROM_TRAINING", "farther from training"),
        ("CENTRAL_RANGE_OVERLAPS_ZERO", "range overlaps zero"),
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
    real_rates = [100.0 * float(cell.provenance["raw_real_rate"]) for cell in rate_cells]
    generated_rates = [100.0 * float(cell.provenance["raw_generated_rate"]) for cell in rate_cells]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    x = np.arange(6)
    width = 0.36
    axes[0].bar(x - width / 2, real_rates, width, color="#4f5b61", label="held-out real")
    axes[0].bar(x + width / 2, generated_rates, width, color="#2b7a78", label="generated")
    axes[0].set_xticks(x, [CLASS_LABELS[stratum] for stratum in strata])
    axes[0].set_ylabel("Fields touching 170/325 K boundary (%)")
    axes[0].set_title("Raw preprocessing-boundary rates", fontweight="semibold")
    axes[0].legend(frameon=False)
    gaps = np.asarray([cell.reference_gap for cell in rate_cells], dtype=float)
    axes[1].bar(x, gaps, color="#638b8f")
    axes[1].axhline(1.0, color="#a33b20", linestyle="--", linewidth=1.0)
    for column, cell in enumerate(rate_cells):
        if cell.reference_status.value in {"NOT_EVALUATED", "REFERENCE_UNRESOLVED"}:
            axes[1].text(column, gaps[column], "×", ha="center", va="bottom")
    axes[1].set_xticks(x, [CLASS_LABELS[stratum] for stratum in strata])
    axes[1].set_ylabel("Reference-calibrated gap")
    axes[1].set_title("Reference audit · × withheld", fontweight="semibold")
    for ax in axes:
        ax.grid(axis="y", color="#e0e6e8", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    _save_figure(fig, audit / "track1_preprocessing_saturation", dpi=190)


def _copy_sensitivity_artifact(appendix: Path) -> Path:
    source = (
        ROOT / "case_studies" / "tropical_cyclone_bt" / "control_results" / "sensitivity_lanes.pdf"
    )
    target = appendix / "figure_A2_sensitivity_controls.pdf"
    shutil.copy2(source, target)
    return target


def _control_metric_label(metric_id: str) -> str:
    labels = {
        "w1:min_core_bt": "Core minimum BT",
        "w1:p01_bt": "BT first percentile",
        "w1:cold_frac_220": "Cold-cloud area",
        "w1:eye_contrast": "Eye contrast",
        "w1:r_maxgrad": "Maximum-gradient radius",
        "w1:axisym": "Axisymmetry departure",
        "w1:dav": "Deviation-angle variance",
        "radial_profile_mae": "Radial-profile MAE",
        "radial_log_psd_mse": "Log-PSD discrepancy",
        "feature_mmd2": "MMD²",
        "feature_density": "Density departure",
        "feature_coverage_deficit": "Coverage deficit",
        "feature_diversity_departure": "Diversity departure",
        "preprocessing_boundary_saturation": "Boundary saturation",
    }
    return labels.get(metric_id, metric_id)


def _severity_panel(
    axis: plt.Axes,
    lane: Mapping[str, Any],
    *,
    title: str,
) -> None:
    severities = np.asarray(lane["severities"], dtype=float)
    colors = ("#176B73", "#C65A3A", "#77628F")
    displayed_values: list[float] = []
    for index, (metric_id, result) in enumerate(lane["metrics"].items()):
        boundary = float(result["reference_boundary"])
        normalized = np.asarray(result["values"], dtype=float) / max(
            boundary,
            np.finfo(float).eps,
        )
        displayed_values.extend(normalized[np.isfinite(normalized)].tolist())
        axis.plot(
            severities,
            normalized,
            marker=("o", "s", "^")[index % 3],
            color=colors[index % len(colors)],
            linewidth=1.35,
            markersize=4.5,
            label=_control_metric_label(metric_id),
        )
        crossing = result.get("first_reference_crossing_severity")
        if crossing is not None:
            axis.axvline(
                float(crossing),
                color=colors[index % len(colors)],
                linestyle=":",
                linewidth=0.9,
            )
        axis.text(
            0.98,
            0.96 - 0.11 * index,
            rf"$\rho$={float(result['rank_correlation']):.2f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.2,
            color=colors[index % len(colors)],
        )
    axis.axhline(1.0, color="#A33B20", linestyle="--", linewidth=1.0)
    if displayed_values and max(displayed_values) > 30.0:
        axis.set_yscale("symlog", linthresh=1.0)
    axis.set_title(title, loc="left", fontsize=11.2, fontweight="semibold")
    axis.set_xlabel("Declared control severity", fontsize=9.3)
    axis.set_ylabel("Discrepancy / reference q95", fontsize=9.3)
    axis.legend(frameon=False, fontsize=8.2, loc="best")
    axis.grid(color="#DFE6E8", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8.8)


def _binary_control_panel(
    axis: plt.Axes,
    lane: Mapping[str, Any],
    *,
    title: str,
) -> None:
    metric_ids = list(lane["metrics"])
    values = [
        float(lane["metrics"][metric_id]["value"])
        / max(
            float(lane["metrics"][metric_id]["reference_boundary"]),
            np.finfo(float).eps,
        )
        for metric_id in metric_ids
    ]
    axis.bar(
        np.arange(len(metric_ids)),
        values,
        color=("#176B73", "#C65A3A", "#77628F")[: len(metric_ids)],
        width=0.62,
    )
    axis.axhline(1.0, color="#A33B20", linestyle="--", linewidth=1.0)
    axis.set_xticks(
        np.arange(len(metric_ids)),
        [_control_metric_label(metric_id) for metric_id in metric_ids],
        rotation=14,
        ha="right",
    )
    axis.set_ylabel("Discrepancy / reference q95", fontsize=9.3)
    axis.set_title(title, loc="left", fontsize=11.2, fontweight="semibold")
    axis.grid(axis="y", color="#DFE6E8", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8.8)


def _render_control_response_figures(
    appendix: Path,
    controls: Mapping[str, Any],
) -> tuple[Path, ...]:
    """Render legible grouped views of every registered control response."""

    outputs: list[Path] = []
    groups = (
        (
            "figure_A2a_physical_controls",
            (
                ("warm_shift", "Additive temperature shift"),
                ("threshold_warming", "Warming around 220 K"),
                ("threshold_cooling", "Cooling around 220 K"),
                ("eye_contrast", "Inner-eye / eyewall perturbation"),
            ),
            (2, 2),
            (10.8, 7.5),
        ),
        (
            "figure_A2b_spatial_controls",
            (
                ("blur", "Gaussian blur"),
                ("center_jitter", "Centre displacement"),
                ("radial_warp", "Radial deformation"),
                ("checkerboard", "High-frequency artifact"),
            ),
            (2, 2),
            (10.8, 7.5),
        ),
        (
            "figure_A2c_feature_controls",
            (
                ("feature_eof_excursion", "Training-EOF excursion"),
                ("feature_mode_removal", "Feature-mode removal"),
                ("feature_contraction", "Feature contraction"),
                ("feature_expansion", "Feature expansion"),
            ),
            (2, 2),
            (10.8, 7.5),
        ),
    )
    for stem, lanes, shape, size in groups:
        figure, axes = plt.subplots(*shape, figsize=size)
        for axis, (lane_id, title) in zip(axes.flat, lanes, strict=True):
            _severity_panel(axis, controls[lane_id], title=title)
        figure.tight_layout()
        pdf, png = _save_figure(figure, appendix / stem, dpi=220)
        outputs.extend((pdf, png))

    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.5))
    _binary_control_panel(
        axes[0, 0],
        controls["phase_shuffle"],
        title="Fourier-phase randomisation",
    )
    invariant = controls["phase_shuffle"]["invariant"]
    axes[0, 0].text(
        0.03,
        0.95,
        (
            "Unwindowed power preserved\n"
            f"max. relative error = {float(invariant['maximum_relative_error']):.1e}"
        ),
        transform=axes[0, 0].transAxes,
        ha="left",
        va="top",
        fontsize=8.3,
        color="#455A64",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    _binary_control_panel(
        axes[0, 1],
        controls["psd_quarter_turn"],
        title="Quarter-turn PSD invariance",
    )
    rotation_result = controls["psd_quarter_turn"]["metrics"]["radial_log_psd_mse"]
    rotation_ratio = float(rotation_result["value"]) / float(rotation_result["reference_boundary"])
    axes[0, 1].text(
        0.50,
        0.48,
        f"Observed ratio = {rotation_ratio:.1e}\n(expected to remain below 1)",
        transform=axes[0, 1].transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#455A64",
    )
    _severity_panel(
        axes[1, 0],
        controls["boundary_spikes"],
        title="Preprocessing-boundary spikes",
    )
    copy_rows = controls["copy_fraction"]["fractions"]
    fractions = np.asarray([row["copy_fraction"] for row in copy_rows], dtype=float)
    resolved_regions = np.asarray(
        [row["closer_to_training_cells"] for row in copy_rows],
        dtype=float,
    )
    axes[1, 1].plot(
        fractions,
        resolved_regions,
        marker="o",
        color="#77628F",
        linewidth=1.5,
    )
    minimum = controls["copy_fraction"]["minimum_resolved_fraction"]
    if minimum is not None:
        axes[1, 1].axvline(
            float(minimum),
            color="#A33B20",
            linestyle="--",
            linewidth=1.0,
            label=f"first resolved: {100 * float(minimum):g}%",
        )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("Exact training replay fraction", fontsize=9.3)
    axes[1, 1].set_ylabel("Regions closer to training", fontsize=9.3)
    axes[1, 1].set_title(
        "Localized memorization resolution",
        loc="left",
        fontsize=11.2,
        fontweight="semibold",
    )
    axes[1, 1].legend(frameon=False, fontsize=8.3)
    axes[1, 1].grid(color="#DFE6E8", linewidth=0.6)
    axes[1, 1].spines[["top", "right"]].set_visible(False)
    axes[1, 1].tick_params(labelsize=8.8)
    figure.tight_layout()
    pdf, png = _save_figure(
        figure,
        appendix / "figure_A2d_invariance_and_resolution_controls",
        dpi=220,
    )
    outputs.extend((pdf, png))
    return tuple(outputs)


def _copy_secondary_audits(appendix: Path) -> tuple[Path, ...]:
    source_dir = ROOT / "outputs" / "tropical_cyclone_bt" / "audits"
    names = (
        "continuous_wind_response.json",
        "continuous_wind_response.csv",
        "figure_A_continuous_wind_response.pdf",
        "figure_A_continuous_wind_response.png",
        "feature_space_robustness.json",
        "figure_A_feature_space_robustness.pdf",
        "figure_A_feature_space_robustness.png",
    )
    copied = []
    for suffix in (".pdf", ".png"):
        source = ROOT / "validation" / f"statistical_validation_v0.2_appendix{suffix}"
        if not source.is_file():
            raise FileNotFoundError(
                f"Required validation stability figure has not been generated: {source}"
            )
        target = appendix / f"figure_A_reference_stability{suffix}"
        shutil.copy2(source, target)
        copied.append(target)
    for name in names:
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Required secondary TC audit has not been generated: {source}")
        target = appendix / name
        shutil.copy2(source, target)
        copied.append(target)
    return tuple(copied)


def _write_metric_registry(output_dir: Path, report: EvaluationReport) -> None:
    registry = {str(item["id"]): item for item in report.metadata.get("metric_registry", [])}
    controls = {
        "w1:min_core_bt": "additive temperature shift",
        "w1:cold_frac_220": "warming near the 220 K threshold",
        "w1:p01_bt": "additive temperature shift",
        "w1:r_maxgrad": "blur, centre displacement, and radial deformation",
        "w1:axisym": "Fourier-phase randomisation",
        "w1:dav": "Fourier-phase randomisation",
        "w1:eye_contrast": "eye/eyewall temperature perturbation",
        "radial_profile_mae": "centre displacement, radial deformation, and phase randomisation",
        "radial_log_psd_mse": "blur/high-frequency positive controls; quarter-turn invariance",
        "feature_mmd2": "training-EOF excursion",
        "feature_density": "training-EOF excursion",
        "feature_coverage_deficit": "feature-mode removal",
        "feature_diversity_departure": "feature contraction and expansion",
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
                "physical_quantity": ("" if functional is None else functional["description"]),
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
    source = ROOT / "case_studies" / "tropical_cyclone_bt" / "partition_summary.csv"
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
    source = ROOT / "case_studies" / "tropical_cyclone_bt" / "vmax_support_summary.json"
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
                "distribution_total_variation": node["discrete_distribution_total_variation"],
                "generated_support_subset_of_real": node["generated_support_subset_of_reference"],
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
schema-v3 report. The main directory contains the framework, validation,
case-study, and headline-result artifacts. The appendix retains the complete
class, curve, control, copying, dependence, preprocessing, robustness, data,
and provenance evidence.

## Main-text artifacts

- `main/figure_1_framework_overview.pdf`: GAUGE data, calibration, metric, and
  reporting workflow.
- `main/figure_2_statistical_validation.pdf`: empirical-reference validation,
  including same-population crossings, perturbation response, unequal blocks,
  and the 22-block C5-like regime.
- `main/figure_3_tc_examples.pdf`: deterministically selected typical real and
  generated fields for TS and Categories 1--5.
- `main/figure_4_reference_gap_profile.pdf`: primary intrinsic result. Every
  native class is visible and training-only positive controls use distinct open
  symbols.
- `main/figure_5_tail_physical_distributions.pdf`: Category 5 empirical distributions
  for all seven physical functionals.
- `main/figure_6_tail_spatial_structure.pdf`: Category 5 radial-profile and spectral
  drill-downs.
- `main/table_1_metric_registry.md`: definitions, units, roles, and registered
  controls for the TC metric suite.
- `main/table_2_native_metric_values.csv`: exact native-unit estimates,
  intervals, reference summaries, gaps, and statuses for every displayed class.

The appendix includes the complete conditional matrix, four legible grouped
control-response figures plus the raw all-control provenance plot, all-class
radial-profile and PSD figures, copying/preprocessing audits, every native-class
functional distribution, extended feature/dependence diagnostics, partition and
Vmax-support tables, deterministic example selection, and every report cell.

## Reporting language

- Gap 0 is the median event-disjoint real-vs-real discrepancy; gap 1 is its
  empirical 95th percentile.
- Gap units express finite-archive reference position, not physical severity,
  and must not be averaged into a model score.
- `WITHIN_REFERENCE_RANGE` and `ABOVE_REFERENCE_RANGE` are descriptive labels,
  not hypothesis-test decisions.
- A `REFERENCE_UNRESOLVED` or `NOT_EVALUATED` cell supports no affirmative
  interpretation.
- The unmatched global mixture is deliberately absent from the fidelity figure;
  it remains in the complete audit table as a dataset-composition check.
- Curve shading shows radius- or wavenumber-specific variability; the registered scalar curve
  discrepancy is reported separately and is not inferred by visually counting
  excursions.

Source report: `{report_path.name}`

Source report SHA-256: `{_sha256(report_path)}`

Configuration hash: `{report.metadata.get("config_hash", "unknown")}`
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
                "comparison_low": cell.comparison_low,
                "comparison_high": cell.comparison_high,
                "units": cell.units,
                "reference_median": cell.reference_q50,
                "reference_q95": cell.reference_q95,
                "reference_percentile": cell.reference_percentile,
                "reference_gap": cell.reference_gap,
                "reference_status": cell.reference_status.value,
                "control_status": cell.control_status.value,
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
        "| Metric | Model discrepancy (central comparison range) | Real-reference q50 → q95 | Percentile | Gap | Reference / control status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        estimate = (
            "—"
            if row["estimate"] is None
            else f"{row['estimate']:.3g} [{row['comparison_low']:.3g}, {row['comparison_high']:.3g}] {row['units']}"
        )
        reference = (
            "—"
            if row["reference_median"] is None
            else f"{row['reference_median']:.3g} → {row['reference_q95']:.3g}"
        )
        percentile = (
            "—"
            if row["reference_percentile"] is None
            else f"{100.0 * row['reference_percentile']:.1f}%"
        )
        gap = "—" if row["reference_gap"] is None else f"{row['reference_gap']:.2f}"
        markdown.append(
            f"| {row['metric']} | {estimate} | {reference} | {percentile} | "
            f"{gap} | {row['reference_status']} / {row['control_status']} |"
        )
    markdown.extend(
        [
            "",
            "Gap 1 is the empirical 95th percentile of the event-disjoint real reference. It is a descriptive coordinate, not a universal severity or acceptability scale.",
            "",
        ]
    )
    (appendix / "table_A5_c5_values.md").write_text("\n".join(markdown), encoding="utf-8")


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
            "Same-population crossings, known perturbation response, unequal events, reference stability, and explicit resolution loss in C5-like thin strata.",
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
            "Category 5 radial-profile and power-spectrum drill-downs; coordinate-wise curve shading is distinguished from the scalar curve discrepancy.",
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
<div class="key"><strong>Gap key:</strong> 0 is the median event-disjoint real–real discrepancy and 1 is its empirical 95th percentile. Lower is closer. The gap is a finite-archive reference coordinate, not physical severity, a p-value, equivalence, or a model score.</div>
{figure_html}
<h2>Tables and appendix</h2>
<ul><li><a href="main/table_1_metric_registry.md">Registered metric definitions</a></li>
<li><a href="main/table_2_native_metric_values.csv">Every displayed class and metric (CSV)</a></li>
<li><a href="appendix/figure_A1_conditional_gap_matrix.pdf">Complete conditional matrix (PDF)</a></li>
<li><a href="appendix/figure_A2a_physical_controls.pdf">Physical-functional controls (PDF)</a></li>
<li><a href="appendix/figure_A2b_spatial_controls.pdf">Spatial controls (PDF)</a></li>
<li><a href="appendix/figure_A2c_feature_controls.pdf">Feature-space controls (PDF)</a></li>
<li><a href="appendix/figure_A2d_invariance_and_resolution_controls.pdf">Invariance and resolution controls (PDF)</a></li>
<li><a href="appendix/figure_A2_sensitivity_controls.pdf">Raw all-control provenance figure (PDF)</a></li>
<li><a href="appendix/figure_A3_all_class_radial_profiles.pdf">All-class radial profiles (PDF)</a></li>
<li><a href="appendix/figure_A4_all_class_radial_psd.pdf">All-class radial PSDs (PDF)</a></li>
<li><a href="appendix/figure_A5_generation_audits.pdf">Copying and preprocessing audits (PDF)</a></li>
<li><a href="appendix/table_A1_complete_evaluation.csv">Every report cell (CSV)</a></li>
<li><a href="appendix/functional_distributions/">Native-class distribution figures</a></li>
<li><a href="README.md">Reporting guide and permitted language</a></li></ul>
<p>Frozen configuration hash: <code>{report.metadata.get("config_hash", "unknown")}</code>.</p>
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
        model_label="TC brightness-temperature diffusion model",
        metrics=MAIN_METRICS,
        native_strata=("0", "1", "2", "3", "4", "5"),
        priority_strata=(),
        metric_labels=METRIC_LABELS,
        stratum_labels=CLASS_LABELS,
    )
    report.paper_bundle(output_dir, spec, controls=_controls(controls_payload))
    for suffix in (".pdf", ".png"):
        (output_dir / "main" / f"figure_1_reference_gap_profile{suffix}").replace(
            output_dir / "main" / f"figure_4_reference_gap_profile{suffix}"
        )
    (output_dir / "main" / "table_1_native_metric_values.csv").replace(
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
    _render_control_response_figures(appendix, controls_payload)
    _copy_secondary_audits(appendix)
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
