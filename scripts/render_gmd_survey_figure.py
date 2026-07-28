#!/usr/bin/env python3
"""Render the bounded descriptive survey figure used by the GMD manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CRITERIA = (
    ("held_out_test_set", "Held-out test set stated"),
    ("conditional_fidelity", "Conditional fidelity"),
    ("extremes_tails", "Extremes or tails"),
    ("quantitative_distributional", "Quantitative distributional comparison"),
    ("spatial_spectra", "Quantified spatial spectra"),
    ("probabilistic_calibration", "Probabilistic calibration"),
    ("metric_uncertainty", "Metric uncertainty"),
    ("diversity_coverage", "Diversity or coverage"),
    ("real_reference", "Real-vs-real reference"),
    ("metric_validity", "Metric-validity control"),
    ("physical_admissibility", "Physical admissibility"),
    ("memorization", "Memorization or nearest-neighbour check"),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("survey/exports/summary.json"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("manuscript/gmd/fig02"),
    )
    return parser.parse_args()


def _strict_count(summary: dict, key: str) -> int:
    if key == "held_out_test_set":
        return int(summary["descriptive_distributions"][key]["yes"])
    return int(summary["criteria"][key]["strict_n"])


def main() -> None:
    args = _arguments()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    n = int(summary["n_coded"])
    counts = np.asarray([_strict_count(summary, key) for key, _ in CRITERIA])
    percentages = 100.0 * counts / n
    labels = [label for _, label in CRITERIA]

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.7))
    fig.subplots_adjust(left=0.47, right=0.97, top=0.98, bottom=0.13)
    y = np.arange(len(labels))
    for index, percentage in enumerate(percentages):
        if index % 2 == 0:
            ax.axhspan(index - 0.47, index + 0.47, color="#f5f8f8", zorder=0)
        ax.plot(
            (0, percentage),
            (index, index),
            color="#d5e5e3",
            linewidth=6.0,
            solid_capstyle="round",
            zorder=1,
        )
    ax.scatter(
        percentages,
        y,
        s=47,
        color="#246a73",
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 104)
    ax.set_xticks((0, 25, 50, 75, 100))
    ax.set_xlabel("Share of coded corpus meeting the strict criterion (%)")
    ax.grid(axis="x", color="#d9e1e4", linewidth=0.7)
    ax.set_axisbelow(True)
    for index, (count, percentage) in enumerate(zip(counts, percentages, strict=True)):
        ax.text(
            min(percentage + 1.25, 100.2),
            index,
            f"{count}/{n}",
            va="center",
            ha="left",
            fontsize=8.3,
            color="#263238",
        )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="both", labelsize=8.7)
    ax.spines["bottom"].set_color("#6a777d")
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
