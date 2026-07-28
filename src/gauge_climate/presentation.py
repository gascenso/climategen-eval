"""Interpretation-first figures and tables for GAUGE evaluation reports.

The statistical report remains the source of truth.  This module turns that
report into the small set of artifacts most authors need in a results section,
while retaining every native-unit value and class-specific result in an
appendix table.
"""

from __future__ import annotations

import csv
import html
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

if TYPE_CHECKING:
    from .report import EvaluationReport
    from .types import CellReport


@dataclass(frozen=True)
class ReferenceControl:
    """One predeclared positive-control result on the reference-gap scale."""

    metric: str
    label: str
    position: float
    provenance: str = "reserved-real positive control"

    def __post_init__(self) -> None:
        if not self.metric or not self.label:
            raise ValueError("Reference controls require metric and label names.")
        if not np.isfinite(self.position):
            raise ValueError("Reference-control positions must be finite.")


@dataclass(frozen=True)
class PaperBundleSpec:
    """Declare the compact view of a complete GAUGE report.

    ``native_strata`` are the mutually exclusive conditions that must not be
    hidden by a global mixture. ``priority_strata`` are registered aggregates
    or conditions that deserve emphasis (for example, a physically important
    tail). When there are more than ``max_visible_native_strata`` native strata,
    the main profile shows their median/range and labels the worst observed
    stratum; the appendix matrix still shows every stratum.
    """

    title: str
    model_label: str
    metrics: tuple[str, ...]
    native_strata: tuple[str, ...]
    priority_strata: tuple[str, ...] = ()
    metric_labels: Mapping[str, str] = field(default_factory=dict)
    stratum_labels: Mapping[str, str] = field(default_factory=dict)
    max_visible_native_strata: int = 8

    def __post_init__(self) -> None:
        if not self.title or not self.model_label:
            raise ValueError("Paper bundle title and model label cannot be empty.")
        if not self.metrics or len(self.metrics) != len(set(self.metrics)):
            raise ValueError("Paper bundle metrics must be nonempty and unique.")
        if not self.native_strata or len(self.native_strata) != len(set(self.native_strata)):
            raise ValueError("Native strata must be nonempty and unique.")
        if len(self.priority_strata) != len(set(self.priority_strata)):
            raise ValueError("Priority strata must be unique.")
        if set(self.native_strata) & set(self.priority_strata):
            raise ValueError("Native and priority strata must be distinct.")
        if self.max_visible_native_strata < 2:
            raise ValueError("max_visible_native_strata must be at least two.")


def _label(value: str, labels: Mapping[str, str]) -> str:
    return str(labels.get(value, value))


def _cell_index(report: EvaluationReport) -> dict[tuple[str, str], CellReport]:
    return {(cell.stratum, cell.metric): cell for cell in report.cells}


def _selected_cells(
    report: EvaluationReport, spec: PaperBundleSpec
) -> tuple[dict[tuple[str, str], CellReport], tuple[str, ...]]:
    index = _cell_index(report)
    strata = (*spec.native_strata, *spec.priority_strata)
    missing = [
        f"{stratum}:{metric}"
        for metric in spec.metrics
        for stratum in strata
        if (stratum, metric) not in index
    ]
    if missing:
        raise ValueError(
            "Paper bundle requests cells absent from the report: " + ", ".join(missing)
        )
    undefined = [
        f"{stratum}:{metric}"
        for metric in spec.metrics
        for stratum in strata
        if index[stratum, metric].reference_gap is None
    ]
    if undefined:
        raise ValueError(
            "Paper bundle cells lack reference-calibrated gaps: " + ", ".join(undefined)
        )
    return index, strata


def _save_figure(fig: plt.Figure, path: Path) -> tuple[Path, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = path.with_suffix(".pdf")
    png = path.with_suffix(".png")
    metadata = {"CreationDate": None, "ModDate": None}
    fig.savefig(pdf, bbox_inches="tight", metadata=metadata)
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def _finite_position(cell: CellReport, attribute: str) -> float:
    value = getattr(cell, attribute)
    if value is None:
        assert cell.reference_gap is not None
        return float(cell.reference_gap)
    return float(value)


def figure_stratified_profile(
    report: EvaluationReport,
    spec: PaperBundleSpec,
    path: str | Path,
    *,
    controls: Sequence[ReferenceControl] = (),
) -> tuple[Path, Path]:
    """Render the main-text reference-gap profile.

    Native classes are never replaced by an unmatched global mixture. Small
    class systems show every condition; large systems show a class range and
    median together with all registered priority strata. Positive controls
    occupy the same descriptive reference scale but are visually distinguished.
    """

    index, _ = _selected_cells(report, spec)
    controls_by_metric: dict[str, list[ReferenceControl]] = {}
    for control in controls:
        if control.metric in spec.metrics:
            controls_by_metric.setdefault(control.metric, []).append(control)

    metric_count = len(spec.metrics)
    y = np.arange(metric_count, dtype=float)
    fig_height = max(6.2, 0.53 * metric_count + 2.4)
    fig, ax = plt.subplots(figsize=(11.4, fig_height))
    fig.subplots_adjust(left=0.21, right=0.98, top=0.98, bottom=0.21)

    for row in range(metric_count):
        if row % 2 == 0:
            ax.axhspan(row - 0.48, row + 0.48, color="#f3f7f8", zorder=0)

    native_colors = plt.colormaps["viridis"](np.linspace(0.12, 0.82, len(spec.native_strata)))
    all_x: list[float] = [-1.0, 1.0]
    show_native = len(spec.native_strata) <= spec.max_visible_native_strata

    if show_native:
        offsets = np.linspace(-0.20, 0.20, len(spec.native_strata))
        for stratum, color, offset in zip(spec.native_strata, native_colors, offsets, strict=True):
            cells = [index[stratum, metric] for metric in spec.metrics]
            positions = np.asarray([float(cell.reference_gap) for cell in cells], dtype=float)
            lows = np.asarray([_finite_position(cell, "comparison_gap_low") for cell in cells])
            highs = np.asarray([_finite_position(cell, "comparison_gap_high") for cell in cells])
            errors = np.vstack(
                [
                    np.maximum(positions - np.minimum(lows, positions), 0.0),
                    np.maximum(np.maximum(highs, positions) - positions, 0.0),
                ]
            )
            ax.errorbar(
                positions,
                y + offset,
                xerr=errors,
                fmt="o",
                markersize=5.2,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.45,
                color=color,
                ecolor=(*color[:3], 0.42),
                elinewidth=0.8,
                capsize=1.4,
                zorder=3,
            )
            unresolved = np.asarray(
                [
                    cell.reference_status.value in {"REFERENCE_UNRESOLVED", "NOT_EVALUATED"}
                    for cell in cells
                ]
            )
            if np.any(unresolved):
                ax.scatter(
                    positions[unresolved],
                    (y + offset)[unresolved],
                    marker="x",
                    s=45,
                    linewidths=1.2,
                    color="#263238",
                    zorder=5,
                )
            all_x.extend(lows.tolist())
            all_x.extend(highs.tolist())
    else:
        for row, metric in enumerate(spec.metrics):
            positions = np.asarray(
                [float(index[stratum, metric].reference_gap) for stratum in spec.native_strata]
            )
            minimum = float(np.min(positions))
            maximum = float(np.max(positions))
            median = float(np.median(positions))
            worst = spec.native_strata[int(np.argmax(positions))]
            ax.hlines(row, minimum, maximum, color="#69838f", linewidth=2.2, zorder=2)
            ax.plot(median, row, "o", color="#365c6d", markersize=6.0, zorder=3)
            ax.annotate(
                _label(worst, spec.stratum_labels),
                (maximum, row),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.5,
                color="#365c6d",
            )
            all_x.extend([minimum, maximum])

    priority_colors = ["#d05a3a", "#8c3b76", "#b27425", "#2d6a4f"]
    priority_offsets = np.linspace(-0.30, 0.30, max(len(spec.priority_strata), 1))
    for priority_index, stratum in enumerate(spec.priority_strata):
        color = priority_colors[priority_index % len(priority_colors)]
        cells = [index[stratum, metric] for metric in spec.metrics]
        positions = np.asarray([float(cell.reference_gap) for cell in cells], dtype=float)
        lows = np.asarray([_finite_position(cell, "comparison_gap_low") for cell in cells])
        highs = np.asarray([_finite_position(cell, "comparison_gap_high") for cell in cells])
        errors = np.vstack(
            [
                np.maximum(positions - np.minimum(lows, positions), 0.0),
                np.maximum(np.maximum(highs, positions) - positions, 0.0),
            ]
        )
        offset = priority_offsets[priority_index]
        if show_native:
            offset = 0.31 + 0.08 * priority_index
        ax.errorbar(
            positions,
            y + offset,
            xerr=errors,
            fmt="D",
            markersize=6.2,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.55,
            color=color,
            ecolor=(*plt.matplotlib.colors.to_rgb(color), 0.55),
            elinewidth=1.0,
            capsize=1.8,
            zorder=4,
        )
        all_x.extend(lows.tolist())
        all_x.extend(highs.tolist())

    control_labels = list(dict.fromkeys(control.label for control in controls))
    control_marker_cycle = (
        "D",
        "s",
        "^",
        "X",
        "v",
        "P",
        "h",
        "<",
        ">",
        "*",
        "d",
        "p",
    )
    control_markers = {
        label: control_marker_cycle[index_ % len(control_marker_cycle)]
        for index_, label in enumerate(control_labels)
    }
    control_color = "#4f5b61"
    for row, metric in enumerate(spec.metrics):
        metric_controls = controls_by_metric.get(metric, [])
        if not metric_controls:
            continue
        control_offsets = np.linspace(-0.10, 0.10, len(metric_controls))
        for control, offset in zip(metric_controls, control_offsets, strict=True):
            ax.plot(
                control.position,
                row - 0.34 + offset,
                marker=control_markers[control.label],
                linestyle="none",
                markersize=6.1,
                markerfacecolor="white",
                markeredgecolor=control_color,
                markeredgewidth=1.35,
                zorder=5,
            )
            all_x.append(float(control.position))

    finite_x = np.asarray([value for value in all_x if np.isfinite(value)], dtype=float)
    x_min = min(-1.0, float(np.min(finite_x)))
    x_max = max(2.0, float(np.max(finite_x)))
    x_min -= 0.08 * max(abs(x_min), 1.0)
    x_max *= 1.18
    ax.set_xscale("symlog", linthresh=2.0, linscale=1.0, base=10)
    ax.set_xlim(x_min, x_max)
    ax.axvspan(x_min, 1.0, color="#dcece8", alpha=0.38, zorder=0)
    ax.axvline(0.0, color="#78909a", linewidth=0.85, zorder=1)
    ax.axvline(
        1.0,
        color="#b64932",
        linewidth=1.35,
        linestyle=(0, (4, 3)),
        zorder=1,
    )
    candidate_ticks = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0, 5000.0])
    ticks = candidate_ticks[(candidate_ticks >= x_min) & (candidate_ticks <= x_max)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{value:g}" for value in ticks], fontsize=11)
    ax.set_yticks(y, [_label(metric, spec.metric_labels) for metric in spec.metrics], fontsize=12)
    ax.set_ylim(metric_count - 0.52, -0.52)
    ax.grid(axis="x", color="#d8e0e3", linewidth=0.65, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(
        "Reference-calibrated gap   ·   0 = real–real median   ·   "
        "1 = empirical 95th percentile   ·   lower is closer",
        labelpad=10,
        fontsize=11.5,
    )

    handles: list[Line2D] = []
    if show_native:
        handles.extend(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor="white",
                markersize=6,
                label=_label(stratum, spec.stratum_labels),
            )
            for stratum, color in zip(spec.native_strata, native_colors, strict=True)
        )
    else:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="#69838f",
                label="native-class median and range",
            )
        )
    handles.extend(
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor=priority_colors[index_ % len(priority_colors)],
            markeredgecolor="white",
            markersize=7,
            label=_label(stratum, spec.stratum_labels),
        )
        for index_, stratum in enumerate(spec.priority_strata)
    )
    handles.extend(
        Line2D(
            [0],
            [0],
            marker=control_markers[label],
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=control_color,
            markeredgewidth=1.3,
            markersize=6.5,
            label=label,
        )
        for label in control_labels
    )
    if any(
        index[stratum, metric].reference_status.value in {"REFERENCE_UNRESOLVED", "NOT_EVALUATED"}
        for stratum in spec.native_strata
        for metric in spec.metrics
    ):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="x",
                linestyle="none",
                color="#263238",
                markersize=6.5,
                label="reference unresolved",
            )
        )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(6, max(1, len(handles))),
        frameon=False,
        fontsize=11.5,
        columnspacing=1.3,
        handletextpad=0.4,
    )
    return _save_figure(fig, Path(path))


def figure_conditional_matrix(
    report: EvaluationReport,
    spec: PaperBundleSpec,
    path: str | Path,
) -> tuple[Path, Path]:
    """Render a compact all-class appendix matrix of reference gaps."""

    index, strata = _selected_cells(report, spec)
    values = np.asarray(
        [
            [float(index[stratum, metric].reference_gap) for stratum in strata]
            for metric in spec.metrics
        ],
        dtype=float,
    )
    vmin = min(-1.0, float(np.nanmin(values)))
    vmax = max(2.0, float(np.nanmax(values)))
    cmap = LinearSegmentedColormap.from_list(
        "gauge_gap",
        ["#246a73", "#dcece8", "#fff8e8", "#e9a23b", "#b83b32"],
    )
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    width = max(8.5, 0.88 * len(strata) + 4.0)
    height = max(5.4, 0.58 * len(spec.metrics) + 2.2)
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0.22, right=0.88, top=0.98, bottom=0.10)
    image = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)

    for row, metric in enumerate(spec.metrics):
        for column, stratum in enumerate(strata):
            cell = index[stratum, metric]
            value = float(cell.reference_gap)
            marker = ""
            if cell.reference_status.value == "ABOVE_REFERENCE_RANGE":
                marker += "●"
            if cell.warnings:
                marker += "†"
            rgba = cmap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(
                column,
                row,
                f"{value:.2g}{marker}",
                ha="center",
                va="center",
                fontsize=8.3,
                color="white" if luminance < 0.5 else "#263238",
                fontweight="semibold" if marker else "normal",
            )
            if cell.reference_status.value == "ABOVE_REFERENCE_RANGE":
                ax.add_patch(
                    Rectangle(
                        (column - 0.48, row - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="#6f1d1b",
                        linewidth=1.5,
                    )
                )

    if spec.priority_strata:
        boundary = len(spec.native_strata) - 0.5
        ax.axvline(boundary, color="#263238", linewidth=1.6)
    ax.set_xticks(
        np.arange(len(strata)),
        [_label(stratum, spec.stratum_labels) for stratum in strata],
        rotation=0,
    )
    ax.set_yticks(
        np.arange(len(spec.metrics)),
        [_label(metric, spec.metric_labels) for metric in spec.metrics],
    )
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.035)
    colorbar.set_label("Reference-calibrated gap")
    return _save_figure(fig, Path(path))


_TABLE_FIELDS = (
    "metric",
    "metric_label",
    "stratum",
    "stratum_label",
    "estimate",
    "comparison_low",
    "comparison_high",
    "units",
    "reference_q50",
    "reference_boundary",
    "reference_percentile",
    "reference_gap",
    "reference_status",
    "control_status",
    "n_eval",
    "n_real",
    "n_blocks",
    "estimand",
    "direction",
    "warning_count",
)


def _table_row(
    cell: CellReport,
    metric_labels: Mapping[str, str],
    stratum_labels: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "metric": cell.metric,
        "metric_label": _label(cell.metric, metric_labels),
        "stratum": cell.stratum,
        "stratum_label": _label(cell.stratum, stratum_labels),
        "estimate": cell.estimate,
        "comparison_low": cell.comparison_low,
        "comparison_high": cell.comparison_high,
        "units": cell.units,
        "reference_q50": cell.reference_q50,
        "reference_boundary": cell.reference_upper_boundary,
        "reference_percentile": cell.reference_percentile,
        "reference_gap": cell.reference_gap,
        "reference_status": cell.reference_status.value,
        "control_status": cell.control_status.value,
        "n_eval": cell.n_eval,
        "n_real": cell.n_real,
        "n_blocks": cell.n_blocks,
        "estimand": cell.estimand.value,
        "direction": cell.direction,
        "warning_count": len(cell.warnings),
    }


def write_native_table(
    report: EvaluationReport,
    spec: PaperBundleSpec,
    path: str | Path,
    *,
    all_report_cells: bool = False,
) -> Path:
    """Write a publication-source table without formatted-number loss."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if all_report_cells:
        cells = list(report.cells)
    else:
        index, strata = _selected_cells(report, spec)
        cells = [index[stratum, metric] for metric in spec.metrics for stratum in strata]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_TABLE_FIELDS, lineterminator="\n")
        writer.writeheader()
        for cell in cells:
            writer.writerow(_table_row(cell, spec.metric_labels, spec.stratum_labels))
    return output


def _index_html(
    report: EvaluationReport,
    spec: PaperBundleSpec,
    output: Path,
    artifacts: Mapping[str, Path],
) -> Path:
    rows = []
    index, strata = _selected_cells(report, spec)
    for metric in spec.metrics:
        cells = [index[stratum, metric] for stratum in strata]
        worst = max(cells, key=lambda cell: float(cell.reference_gap))
        priority = cells[-1] if spec.priority_strata else None
        rows.append(
            "<tr>"
            f"<td>{html.escape(_label(metric, spec.metric_labels))}</td>"
            f"<td>{html.escape(_label(worst.stratum, spec.stratum_labels))}</td>"
            f"<td>{float(worst.reference_gap):.2f}</td>"
            + (
                "<td>—</td><td>—</td>"
                if priority is None
                else (
                    f"<td>{priority.estimate:.3g} {html.escape(priority.units)}</td>"
                    f"<td>{float(priority.reference_gap):.2f}</td>"
                )
            )
            + "</tr>"
        )
    source_hash = html.escape(str(report.metadata.get("config_hash", "unknown")))
    profile_caption = (
        (
            "Every native class is shown because K is small; the registered "
            "priority aggregate is emphasized."
            if spec.priority_strata
            else "Every native class is shown because K is small; no pooled stratum replaces it."
        )
        if len(spec.native_strata) <= spec.max_visible_native_strata
        else (
            "The main profile summarizes the native-class median and range, "
            "labels the largest observed gap, and emphasizes every registered "
            "priority stratum."
        )
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(spec.title)}</title>
<style>
:root{{--ink:#213138;--muted:#607078;--paper:#fff;--wash:#f3f7f8;--accent:#246a73;--tail:#d05a3a}}
body{{font-family:Inter,system-ui,sans-serif;max-width:1180px;margin:0 auto;padding:2.5rem 2rem;color:var(--ink);background:var(--paper)}}
h1{{font-size:2rem;margin-bottom:.3rem}}h2{{margin-top:2.4rem;border-bottom:1px solid #dbe3e6;padding-bottom:.45rem}}
.lede{{font-size:1.06rem;color:var(--muted);max-width:880px}}.key{{background:var(--wash);border-left:4px solid var(--accent);padding:1rem 1.2rem;border-radius:3px}}
figure{{margin:1.5rem 0}}figure img{{width:100%;height:auto;border:1px solid #e2e8ea}}figcaption{{color:var(--muted);font-size:.9rem;margin-top:.45rem}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}}th,td{{padding:.55rem .65rem;border-bottom:1px solid #dbe3e6;text-align:left}}th{{background:var(--wash)}}
a{{color:var(--accent)}}code{{background:var(--wash);padding:.1rem .25rem}}
</style></head><body>
<h1>{html.escape(spec.title)}</h1>
<p class="lede">A paper-facing GAUGE report. The concise figures below are backed by the complete block-aware evaluation; native-unit tables and the full machine report remain linked for audit.</p>
<div class="key"><strong>How to read the gap:</strong> 0 is the median discrepancy between event-disjoint real subsets; 1 is their 95th-percentile marker. Lower is closer. The gap describes a finite-archive reference position, not physical severity, a p-value, equivalence, or an aggregate model score.</div>
<h2>Main-text figure</h2>
<figure><img src="{artifacts["profile_png"].relative_to(output.parent)}" alt="Stratified reference-gap profile"><figcaption>{html.escape(profile_caption)} Distinct open symbols are training-only positive controls.</figcaption></figure>
<h2>Conditional audit</h2>
<figure><img src="{artifacts["matrix_png"].relative_to(output.parent)}" alt="Conditional gap matrix"><figcaption>The matrix gives exact class-level lookup without requiring one figure per class.</figcaption></figure>
<h2>Compact numerical reading</h2>
<table><thead><tr><th>Metric</th><th>Largest displayed gap</th><th>Gap</th><th>Priority estimate</th><th>Priority gap</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<h2>Downloadable evidence</h2>
<ul><li><a href="{artifacts["selected_table"].relative_to(output.parent)}">Selected native-unit table (CSV)</a></li>
<li><a href="{artifacts["all_table"].relative_to(output.parent)}">Complete report table (CSV)</a></li>
<li><a href="{artifacts["profile_pdf"].relative_to(output.parent)}">Main profile (vector PDF)</a></li>
<li><a href="{artifacts["matrix_pdf"].relative_to(output.parent)}">Conditional matrix (vector PDF)</a></li></ul>
<p>Frozen configuration hash: <code>{source_hash}</code>.</p>
</body></html>"""
    output.write_text(document, encoding="utf-8")
    return output


def render_paper_bundle(
    report: EvaluationReport,
    output_dir: str | Path,
    spec: PaperBundleSpec,
    *,
    controls: Sequence[ReferenceControl] = (),
) -> dict[str, Path]:
    """Render the generic GAUGE main-text and appendix bundle."""

    root = Path(output_dir)
    main = root / "main"
    appendix = root / "appendix"
    profile_pdf, profile_png = figure_stratified_profile(
        report,
        spec,
        main / "figure_1_reference_gap_profile",
        controls=controls,
    )
    matrix_pdf, matrix_png = figure_conditional_matrix(
        report,
        spec,
        appendix / "figure_A1_conditional_gap_matrix",
    )
    selected_table = write_native_table(report, spec, main / "table_1_native_metric_values.csv")
    all_table = write_native_table(
        report,
        spec,
        appendix / "table_A1_complete_evaluation.csv",
        all_report_cells=True,
    )
    artifacts = {
        "profile_pdf": profile_pdf,
        "profile_png": profile_png,
        "matrix_pdf": matrix_pdf,
        "matrix_png": matrix_png,
        "selected_table": selected_table,
        "all_table": all_table,
    }
    artifacts["index"] = _index_html(report, spec, root / "index.html", artifacts)
    return artifacts
