#!/usr/bin/env python3
"""Generate provisional survey figures entirely from canonical exports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from export_survey import CRITERIA


def _norm(value: str) -> str:
    return " ".join(value.strip().lower().split())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports", type=Path, default=Path("survey/exports"))
    parser.add_argument("--output", type=Path, default=Path("survey/figures"))
    args = parser.parse_args()
    with (args.exports / "coded_papers.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads((args.exports / "summary.json").read_text(encoding="utf-8"))
    if not rows:
        raise ValueError("No canonical coded papers are available.")
    criteria = list(CRITERIA.items())
    code = {"no": 0, "not stated": 0, "n/a": 0, "unclear": 1, "visual overlay": 1,
            "partial": 2, "yes": 3, "quantified": 3}
    matrix = np.asarray(
        [[code.get(_norm(row[column]), 1) for _, column in criteria] for row in rows],
        dtype=float,
    )
    order = sorted(
        range(len(rows)),
        key=lambda index: (
            rows[index]["Venue type"],
            rows[index]["Model class (derived)"],
            int(float(rows[index]["Year"])),
            rows[index]["Title"],
        ),
    )
    matrix = matrix[order]
    args.output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 13), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=3,
                      cmap=plt.matplotlib.colors.ListedColormap(["#f1f2f3", "#d9b44a", "#6b9ac4", "#256d4a"]))
    ax.set_xticks(np.arange(len(criteria)), [key.replace("_", " ") for key, _ in criteria],
                  rotation=55, ha="right", fontsize=8)
    ax.set_yticks([])
    ax.set_ylabel(f"coded papers (n={len(rows)})")
    ax.set_title("GAUGE evaluation-practice survey — PROVISIONAL, NOT FOR HEADLINE USE")
    colorbar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2, 3], shrink=0.6)
    colorbar.ax.set_yticklabels(["no / n.a.", "unclear / visual", "partial", "yes / quantified"])
    fig.savefig(args.output / "survey_dot_matrix.pdf")
    plt.close(fig)

    keys = list(summary["criteria"])
    strict = [summary["criteria"][key]["strict_percent"] for key in keys]
    lenient = [summary["criteria"][key]["lenient_percent"] for key in keys]
    y = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.barh(y + 0.18, lenient, height=0.34, color="#8eb1c7", label="yes + partial")
    ax.barh(y - 0.18, strict, height=0.34, color="#285f7a", label="strict")
    ax.set_yticks(y, [key.replace("_", " ") for key in keys], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("coded papers (%)")
    ax.set_title("Current coded corpus — PROVISIONAL, NOT FOR HEADLINE USE")
    ax.legend()
    fig.savefig(args.output / "survey_coverage.pdf")
    plt.close(fig)
    print(f"wrote provisional figures to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
