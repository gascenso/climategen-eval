#!/usr/bin/env python3
"""Export the source XLSX into reviewable, reproducible survey artifacts.

This reader uses only the Python standard library so survey regeneration does
not depend on a particular spreadsheet engine. Formulas are intentionally not
evaluated: all headline counts are recomputed from the coded-paper sheet.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
CRITERIA = {
    "quantitative_distributional": "Q1a: quant. distributional comparison?",
    "visual_only_distributional": "Q1b: visual-only distribution comparison?",
    "spatial_spectra": "Q1c: spatial spectra reported?",
    "extremes_tails": "Q1d: extremes/tails evaluated?",
    "conditional_fidelity": "Q2a: conditional fidelity evaluated?",
    "probabilistic_calibration": "Q2b: probabilistic calibration?",
    "physical_admissibility": "Q3: physical admissibility checked?",
    "diversity_coverage": "Q4a: diversity/coverage quantified?",
    "memorization": "Q4b: memorization/NN check?",
    "c2st": "Q5a: discriminative two-sample test (C2ST)?",
    "downstream_utility": "Q5b: downstream-task utility (TSTR)?",
    "metric_validity": "V1: metric-validity control?",
    "metric_uncertainty": "U1: UQ on the metrics themselves?",
    "real_reference": "U2: real-vs-real reference computed?",
    "non_ml_baseline": "B1: non-ML baseline included?",
    "code_released": "R1: code released?",
    "protocol_reproducible": "R2: seeds/protocol reproducible?",
}
LOAD_BEARING = (
    "Q4b: memorization/NN check?",
    "Q5b: downstream-task utility (TSTR)?",
    "V1: metric-validity control?",
    "U2: real-vs-real reference computed?",
)


def _column(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference)
    if letters is None:
        raise ValueError(f"Invalid spreadsheet reference {reference!r}.")
    number = 0
    for char in letters.group(0):
        number = number * 26 + ord(char) - ord("A") + 1
    return number - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(node.itertext()) for node in root.findall("m:si", NS)]


def read_sheet(path: Path, sheet_number: int) -> list[list[str]]:
    with zipfile.ZipFile(path) as archive:
        shared = _shared_strings(archive)
        root = ET.fromstring(archive.read(f"xl/worksheets/sheet{sheet_number}.xml"))
    sparse_rows = []
    width = 0
    for row in root.findall(".//m:sheetData/m:row", NS):
        sparse: dict[int, str] = {}
        for cell in row.findall("m:c", NS):
            index = _column(str(cell.attrib["r"]))
            kind = cell.attrib.get("t")
            if kind == "inlineStr":
                inline = cell.find("m:is", NS)
                value = "" if inline is None else "".join(inline.itertext())
            else:
                node = cell.find("m:v", NS)
                value = "" if node is None or node.text is None else node.text
                if kind == "s" and value:
                    value = shared[int(value)]
            sparse[index] = value
            width = max(width, index + 1)
        sparse_rows.append(sparse)
    return [[row.get(index, "") for index in range(width)] for row in sparse_rows]


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)


def _normalized(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _criterion_summary(header: list[str], rows: list[list[str]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, column in CRITERIA.items():
        index = header.index(column)
        distribution = Counter(_normalized(row[index]) or "missing" for row in rows)
        strict = distribution.get("yes", 0) + distribution.get("quantified", 0)
        lenient = strict + distribution.get("partial", 0)
        result[key] = {
            "column": column,
            "distribution": dict(sorted(distribution.items())),
            "strict_n": strict,
            "strict_percent": round(100.0 * strict / len(rows), 1),
            "lenient_n": lenient,
            "lenient_percent": round(100.0 * lenient / len(rows), 1),
        }
    return result


def _distribution(header: list[str], rows: list[list[str]], column: str) -> dict[str, int]:
    index = header.index(column)
    return dict(
        sorted(Counter(_normalized(row[index]) or "missing" for row in rows).items())
    )


def _double_coding_rows(header: list[str], rows: list[list[str]], fraction: float) -> list[list[str]]:
    count = max(1, round(len(rows) * fraction))
    filename = header.index("Filename")
    title = header.index("Title")
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(("gauge-double-code-v1|" + row[filename]).encode()).hexdigest(),
    )[:count]
    coder_columns = [
        "Filename",
        "Title",
        *LOAD_BEARING,
        "coder_id",
        "coder_notes",
    ]
    output = [coder_columns]
    for row in ranked:
        # Original codes are intentionally hidden from the independent coder.
        output.append([row[filename], row[title], *("" for _ in LOAD_BEARING), "", ""])
    return output


def export(source: Path, output: Path) -> dict[str, object]:
    coded = read_sheet(source, 2)
    candidates = read_sheet(source, 3)
    codebook = read_sheet(source, 4)
    if len(coded) < 2 or coded[0][0] != "Filename":
        raise ValueError("The coded survey sheet does not have the expected schema.")
    header, data = coded[0], coded[1:]
    if len({row[0] for row in data}) != len(data):
        raise ValueError("Filename is not unique in the coded survey.")
    extended_header = [
        *header,
        "Downstream baseline strength",
        "Downstream baseline strength verification",
        "Load-bearing fields human verification",
    ]
    extended_rows = [
        [*row, "", "REQUIRES_HUMAN_CODING", "REQUIRES_HUMAN_VERIFICATION"]
        for row in data
    ]
    write_csv(output / "coded_papers.csv", [extended_header, *extended_rows])
    write_csv(output / "additional_candidates.csv", candidates)
    codebook_rows = [
        *codebook,
        [
            "Downstream baseline strength",
            "Human-coded strength of the strongest count/transform-matched non-generated "
            "downstream baseline; vocabulary and decision rules must be frozen before coding.",
        ],
        [
            "Load-bearing fields human verification",
            "Required full-corpus verification status for memorization, real-reference, "
            "downstream-utility, and metric-validity fields.",
        ],
    ]
    write_csv(output / "codebook.csv", codebook_rows)
    write_csv(output / "double_coding_sample.csv", _double_coding_rows(header, data, 0.20))
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    summary = {
        "schema": "gauge_climate.survey_summary.v1",
        "source": str(source),
        "source_sha256": source_hash,
        "n_coded": len(data),
        "n_additional_candidates": max(0, len(candidates) - 1),
        "coding_semantics": {
            "strict": "yes or quantified",
            "lenient": "strict plus partial",
            "visual and other values": "reported separately, not silently treated as yes",
        },
        "criteria": _criterion_summary(header, data),
        "descriptive_distributions": {
            key: _distribution(header, data, column)
            for key, column in {
                "held_out_test_set": "Held-out test set?",
                "leakage_prone": "Leakage-prone?",
                "temporal_separation": "Temporal separation stated?",
                "venue_type": "Venue type",
                "model_class": "Model class (derived)",
                "task_type": "Task type",
            }.items()
        },
        "verification": {
            "double_coding_target_n": max(1, round(len(data) * 0.20)),
            "load_bearing_full_corpus": "PENDING",
            "downstream_baseline_strength": "PENDING",
            "search_protocol_frozen": False,
            "headline_use_permitted": False,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "gauge_climate.survey_manifest.v1",
        "source_sha256": source_hash,
        "outputs": {},
    }
    for path in sorted(output.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        manifest["outputs"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("docs/methods/genAI_climate_eval_survey.xlsx"),
    )
    parser.add_argument("--output", type=Path, default=Path("survey/exports"))
    args = parser.parse_args()
    summary = export(args.source, args.output)
    print(json.dumps({"n_coded": summary["n_coded"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
