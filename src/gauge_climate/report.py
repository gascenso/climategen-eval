from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .types import CellReport, _jsonable


REPORT_SCHEMA = "gauge_climate.evaluation.v2"
_RETIRED_TOLERANCE_FIELDS = {
    "tolerance",
    "practical",
    "practical_status",
    "sensitivity_status",
    "detection_probability_at_tolerance",
}
_RENAMED_THRESHOLD_FIELDS = {
    "reference_stability_tolerance": "reference_stability_threshold",
    "phase_power_relative_tolerance": "phase_power_relative_threshold",
    "relative_tolerance": "relative_threshold",
}
_RENAMED_MEMORIZATION_STATUS = {
    "OFF_MANIFOLD": "FARTHER_FROM_TRAINING",
}


def _without_retired_tolerance_fields(value: Any) -> Any:
    """Migrate pre-v2 reports without preserving the retired verdict layer."""

    if isinstance(value, Mapping):
        return {
            _RENAMED_THRESHOLD_FIELDS.get(key, key): _without_retired_tolerance_fields(item)
            for key, item in value.items()
            if key not in _RETIRED_TOLERANCE_FIELDS
        }
    if isinstance(value, list):
        return [_without_retired_tolerance_fields(item) for item in value]
    return value


def _with_current_memorization_statuses(value: Any) -> Any:
    """Read pre-1.0 reports using the current neutral copying vocabulary."""

    if isinstance(value, Mapping):
        return {
            _RENAMED_MEMORIZATION_STATUS.get(str(key), str(key)): (
                _with_current_memorization_statuses(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_with_current_memorization_statuses(item) for item in value]
    if isinstance(value, str):
        return _RENAMED_MEMORIZATION_STATUS.get(value, value)
    return value


def _migrate_v1_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a v1 artifact to the detection-only v2 report contract.

    The v1 artifact already stored the matched model-versus-reference draw
    distribution.  V2 uses its median and central interval directly and drops
    the separate population-equivalence interval and verdict branch.
    """

    migrated = _without_retired_tolerance_fields(payload)
    metadata = dict(migrated.get("metadata", {}))
    metadata["method"] = "dependence-matched-real-reference-v2"
    metadata["interpretation"] = {
        "detection": (
            "Non-detection means only that the model discrepancy did not exceed "
            "the finite-N real-reference boundary."
        ),
        "reference_position": "Detectability at this N; not physical severity.",
    }
    metadata["migration"] = {
        "source_schema": "gauge_climate.evaluation.v1",
        "method": "reuse_stored_matched_model_reference_draws",
    }
    config = dict(metadata.get("config", {}))
    config.pop("bootstrap_resamples", None)
    metadata["config"] = config
    migrated["metadata"] = metadata

    cells = []
    for raw in migrated.get("cells", []):
        node = dict(raw)
        provenance = dict(node.get("provenance", {}))
        median = provenance.get("reference_model_median")
        low = provenance.get("model_reference_q025")
        high = provenance.get("model_reference_q975")
        if median is not None and low is not None and high is not None:
            node["estimate"] = median
            node["ci_low"] = low
            node["ci_high"] = high
            provenance["inference_method"] = "reference_resampling_interval"
            provenance["interval_bound_type"] = "central_resampling_1_minus_alpha"
            provenance.pop("interval_standard_error", None)
            provenance.pop("interval_degrees_of_freedom", None)
            provenance.pop("interval_cdf_zero_fraction", None)
        node["provenance"] = provenance
        cells.append(node)
    migrated["cells"] = cells
    migrated["schema"] = REPORT_SCHEMA
    return migrated


@dataclass
class EvaluationReport:
    metadata: Mapping[str, Any]
    cells: Sequence[CellReport]
    warnings: Sequence[str] = ()
    memorization: Mapping[str, Any] | None = None
    sensitivity: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    partitions: Mapping[str, Any] = field(default_factory=dict)
    schema: str = REPORT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({
            "schema": self.schema,
            "metadata": dict(self.metadata),
            "cells": [cell.to_dict() for cell in self.cells],
            "warnings": list(self.warnings),
            "memorization": None if self.memorization is None else dict(self.memorization),
            "sensitivity": dict(self.sensitivity),
            "diagnostics": dict(self.diagnostics),
            "partitions": dict(self.partitions),
        })

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=indent, allow_nan=False)
            handle.write("\n")
        return output

    @classmethod
    def from_json(cls, path: str | Path) -> "EvaluationReport":
        from .types import DetectionStatus, SamplingEstimand

        with Path(path).open("r", encoding="utf-8") as handle:
            raw_payload = json.load(handle)
        if raw_payload.get("schema") == "gauge_climate.evaluation.v1":
            payload = _migrate_v1_payload(raw_payload)
        else:
            payload = _without_retired_tolerance_fields(raw_payload)
        payload = _with_current_memorization_statuses(payload)
        cells = []
        for raw in payload["cells"]:
            node = dict(raw)
            node.setdefault("pillar", "physical_fidelity")
            node.setdefault("reference_detection_boundary", node.get("reference_q95"))
            node["detection_status"] = DetectionStatus(node["detection_status"])
            node["simultaneous_detection_status"] = DetectionStatus(
                node.get("simultaneous_detection_status", "NOT_EVALUATED")
            )
            node["estimand"] = SamplingEstimand(node["estimand"])
            node["warnings"] = tuple(node.get("warnings", ()))
            cells.append(CellReport(**node))
        return cls(
            metadata=payload["metadata"],
            cells=cells,
            warnings=payload.get("warnings", ()),
            memorization=payload.get("memorization"),
            sensitivity=payload.get("sensitivity", {}),
            diagnostics=payload.get("diagnostics", {}),
            partitions=payload.get("partitions", {}),
            schema=REPORT_SCHEMA,
        )

    def to_html(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        pillar_order = {
            "physical_fidelity": 0,
            "conditional_fidelity": 1,
            "diversity_and_coverage": 2,
            "memorization": 3,
            "admissibility": 4,
        }
        ordered_cells = sorted(
            self.cells,
            key=lambda cell: (pillar_order.get(cell.pillar, 99), cell.metric, cell.stratum),
        )
        previous_pillar = None
        confidence_label = 100.0 * float(
            self.metadata.get("config", {}).get("confidence", 0.95)
        )
        column_count = 8
        for cell in ordered_cells:
            if cell.pillar != previous_pillar:
                title = cell.pillar.replace("_", " ").title()
                rows.append(
                    f'<tr class="section"><th colspan="{column_count}">{html.escape(title)}</th></tr>'
                )
                previous_pillar = cell.pillar
            estimate = "—" if cell.estimate is None else f"{cell.estimate:.4g} {html.escape(cell.units)}"
            if cell.direction:
                estimate += f"<br><small>{html.escape(cell.direction)}</small>"
            interval = (
                "—"
                if cell.ci_low is None or cell.ci_high is None
                else f"[{cell.ci_low:.4g}, {cell.ci_high:.4g}]"
            )
            reference = (
                "—"
                if cell.reference_q50 is None or cell.reference_detection_boundary is None
                else (
                    f"q50={cell.reference_q50:.4g}; "
                    f"boundary={cell.reference_detection_boundary:.4g}"
                )
            )
            detection = html.escape(cell.detection_status.value)
            if cell.simultaneous_detection_status.value != "NOT_EVALUATED":
                detection += (
                    "<br><small>joint: "
                    + html.escape(cell.simultaneous_detection_status.value)
                    + "</small>"
                )
            rows.append(
                "<tr>"
                f"<td>{html.escape(cell.pillar.replace('_', ' '))}</td>"
                f"<td>{html.escape(cell.metric)}</td>"
                f"<td>{html.escape(cell.stratum)}</td>"
                f'<td title="real={cell.n_real}; generated pool={cell.n_generated}">'
                f"N={cell.n_eval}; blocks={cell.n_blocks}</td>"
                f"<td>{estimate}</td><td>{interval}</td><td>{reference}</td>"
                f"<td>{detection}</td>"
                "</tr>"
            )
        warning_html = "".join(f"<li>{html.escape(str(item))}</li>" for item in self.warnings)
        memorization_html = (
            "<p>Not supplied.</p>"
            if self.memorization is None
            else (
                "<p>Status: <code>"
                + html.escape(str(self.memorization.get("status", "unknown")))
                + "</code>; supported regions: <code>"
                + html.escape(str(len(self.memorization.get("regions", []))))
                + "</code>.</p>"
            )
        )
        sensitivity_failures = [
            key
            for key, value in self.sensitivity.items()
            if isinstance(value, Mapping) and value.get("passed") is False
        ]
        sensitivity_html = (
            f"<p>{len(self.sensitivity)} registered result(s); "
            f"{len(sensitivity_failures)} failed response contract(s).</p>"
        )
        status_note = (
            "<strong>Reference-aware result.</strong> ‘No departure detected’ means "
            "only that the model did not exceed the finite-N real-reference boundary; "
            "it is not proof of equality or universal application acceptability."
        )
        document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>GAUGE evaluation card</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#18212b}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}}th,td{{border:1px solid #ccd3da;padding:.5rem;text-align:left;vertical-align:top}}
th{{background:#eef2f5}}tr.section th{{background:#dfeaf1;text-transform:uppercase;letter-spacing:.04em}}code{{background:#eef2f5;padding:.1rem .25rem}}.note{{border-left:4px solid #33658a;padding:.5rem 1rem;background:#f3f8fb}}
</style></head><body>
<h1>GAUGE evaluation card</h1>
<p class="note">{status_note}</p>
<p>Estimand: <code>{html.escape(str(self.metadata.get('estimand', 'unknown')))}</code>; N: <code>{html.escape(str(self.metadata.get('n_eval', 'varies')))}</code>; config: <code>{html.escape(str(self.metadata.get('config_hash', 'unknown')))}</code></p>
<table><thead><tr><th>Pillar</th><th>Metric</th><th>Stratum</th><th>Design</th><th>Raw effect</th><th>{confidence_label:g}% resampling interval</th><th>Real-reference envelope</th><th>Detection</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Memorization</h2>{memorization_html}
<h2>Sensitivity controls</h2>{sensitivity_html}
<h2>Warnings</h2><ul>{warning_html or '<li>None</li>'}</ul>
</body></html>"""
        output.write_text(document, encoding="utf-8")
        return output

    def figure_reference_profile(self, path: str | Path) -> Path:
        usable = [cell for cell in self.cells if cell.reference_position is not None]
        if not usable:
            raise ValueError("No cells have a defined standardized reference position.")
        labels = [
            f"{cell.metric} · {cell.stratum} · N={cell.n_eval} · B={cell.n_blocks}"
            for cell in usable
        ]
        positions = np.asarray([float(cell.reference_position) for cell in usable])
        lows = np.asarray(
            [
                float(cell.reference_position_low)
                if cell.reference_position_low is not None
                else float(cell.reference_position)
                for cell in usable
            ]
        )
        highs = np.asarray(
            [
                float(cell.reference_position_high)
                if cell.reference_position_high is not None
                else float(cell.reference_position)
                for cell in usable
            ]
        )
        error_low = np.maximum(positions - np.minimum(lows, positions), 0.0)
        error_high = np.maximum(np.maximum(highs, positions) - positions, 0.0)
        y = np.arange(len(usable))
        fig_height = max(4.0, 0.38 * len(usable) + 1.5)
        fig, ax = plt.subplots(figsize=(9.0, fig_height), constrained_layout=True)
        ax.errorbar(
            positions,
            y,
            xerr=np.vstack([error_low, error_high]),
            fmt="o",
            color="#244f73",
            ecolor="#6688a3",
            capsize=2,
        )
        ax.axvline(1.0, color="#a33b20", linestyle="--", linewidth=1.2, label="pointwise reference boundary")
        ax.axvline(0.0, color="#7d8993", linewidth=0.8)
        ax.set_xscale("symlog", linthresh=2.0)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Standardized reference position (detectability, not severity)")
        config = self.metadata.get("config", {})
        ax.set_title(
            f"estimand={self.metadata.get('estimand', 'unknown')} · "
            f"K={config.get('n_resamples', 'unknown')} · seed={config.get('seed', 'unknown')}"
        )
        ax.legend(loc="best")
        for index, cell in enumerate(usable):
            if cell.estimate is not None:
                ax.annotate(
                    f"{cell.estimate:.3g} {cell.units}",
                    (positions[index], y[index]),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=8,
                )
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
        plt.close(fig)
        return output

    def figure_curve(self, metric: str, stratum: str, path: str | Path) -> Path:
        key = f"{stratum}:{metric}"
        if key not in self.diagnostics:
            raise KeyError(f"No curve diagnostic is stored for {key!r}.")
        diagnostic = self.diagnostics[key]
        coordinates = diagnostic.get("coordinate_values")
        x = (
            np.arange(len(diagnostic["real_mean"]))
            if coordinates is None
            else np.asarray(coordinates)
        )
        fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
        ax.fill_between(
            x,
            diagnostic["reference_q025"],
            diagnostic["reference_q975"],
            color="#9eb6c7",
            alpha=0.45,
            label="real-reference 95% envelope",
        )
        ax.plot(x, diagnostic["real_mean"], color="#18212b", label="held-out real")
        ax.plot(x, diagnostic["generated_mean"], color="#a33b20", label="generated")
        ax.set_xlabel(str(diagnostic.get("coordinate_label", "curve bin")))
        ax.set_ylabel(str(diagnostic.get("units", "value")))
        ax.set_title(f"{metric} · {stratum}")
        ax.legend(loc="best")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
        plt.close(fig)
        return output

    def paper_bundle(
        self,
        output_dir: str | Path,
        spec: Any,
        *,
        controls: Sequence[Any] = (),
    ) -> dict[str, Path]:
        """Render an interpretation-first main-text and appendix bundle.

        ``spec`` is a :class:`gauge_climate.PaperBundleSpec`; ``controls`` may
        contain :class:`gauge_climate.ReferenceControl` objects. The loose type
        annotations keep this module free of a circular import while the public
        presentation module performs full validation.
        """

        from .presentation import render_paper_bundle

        return render_paper_bundle(self, output_dir, spec, controls=controls)
