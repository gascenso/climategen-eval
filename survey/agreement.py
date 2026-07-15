#!/usr/bin/env python3
"""Compute agreement for the blinded 20% survey recode."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


FIELDS = (
    "Q4b: memorization/NN check?",
    "Q5b: downstream-task utility (TSTR)?",
    "V1: metric-validity control?",
    "U2: real-vs-real reference computed?",
)


def _norm(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _kappa(left: list[str], right: list[str]) -> float | None:
    n = len(left)
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / n
    categories = set(left) | set(right)
    expected = sum(left.count(c) * right.count(c) for c in categories) / (n * n)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def _gwet_ac1(left: list[str], right: list[str]) -> float | None:
    """Gwet AC1/AC2-style prevalence-robust chance correction for nominal codes."""

    n = len(left)
    categories = sorted(set(left) | set(right))
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / n
    if len(categories) < 2:
        return None
    proportions = [
        (left.count(category) + right.count(category)) / (2 * n)
        for category in categories
    ]
    chance = sum(p * (1 - p) for p in proportions) / (len(categories) - 1)
    return None if chance == 1 else (observed - chance) / (1 - chance)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, default=Path("survey/exports/coded_papers.csv"))
    parser.add_argument("--recoded", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("survey/exports/agreement.json"))
    args = parser.parse_args()
    with args.canonical.open(encoding="utf-8", newline="") as handle:
        original = {row["Filename"]: row for row in csv.DictReader(handle)}
    with args.recoded.open(encoding="utf-8", newline="") as handle:
        recoded = list(csv.DictReader(handle))
    if not recoded:
        raise ValueError("The recoded file is empty.")
    results = {}
    for field in FIELDS:
        left, right = [], []
        for row in recoded:
            key = row.get("Filename", "")
            if key not in original:
                raise ValueError(f"Unknown recoded paper {key!r}.")
            a, b = _norm(original[key].get(field, "")), _norm(row.get(field, ""))
            if not a or not b:
                raise ValueError(f"Missing code for {field!r} in {key!r}.")
            left.append(a)
            right.append(b)
        results[field] = {
            "n": len(left),
            "raw_agreement": sum(a == b for a, b in zip(left, right, strict=True)) / len(left),
            "cohen_kappa": _kappa(left, right),
            "gwet_ac1": _gwet_ac1(left, right),
            "original_distribution": dict(Counter(left)),
            "recoded_distribution": dict(Counter(right)),
        }
    payload = {"schema": "gauge_climate.survey_agreement.v1", "fields": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
