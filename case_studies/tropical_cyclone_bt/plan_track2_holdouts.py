#!/usr/bin/env python3
"""Preselect whole-storm TC Track-2 holdouts without inspecting images/results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--storm-column", default="storm_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--eligible-label", action="append", required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1789)
    parser.add_argument("--output", type=Path, default=Path("case_studies/tropical_cyclone_bt/track2_holdouts.json"))
    args = parser.parse_args()
    with args.index_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("The index is empty.")
    by_storm = defaultdict(list)
    for row in rows:
        storm = row.get(args.storm_column, "").strip()
        label = row.get(args.label_column, "").strip()
        if not storm or not label:
            raise ValueError("Every row requires non-empty storm and label values.")
        by_storm[storm].append(label)
    eligible = sorted(
        storm
        for storm, labels in by_storm.items()
        if set(labels) & set(args.eligible_label)
    )
    if len(eligible) < args.count:
        raise ValueError(
            f"Only {len(eligible)} storms meet the declared labels; need {args.count}."
        )
    rng = np.random.default_rng(args.seed)
    selected = sorted(rng.choice(np.asarray(eligible, dtype=object), args.count, replace=False).tolist())
    all_storms = sorted(by_storm)
    payload = {
        "schema": "gauge_climate.tc_bt_track2_holdouts.v1",
        "source": str(args.index_csv),
        "source_sha256": hashlib.sha256(args.index_csv.read_bytes()).hexdigest(),
        "selection_rule": "uniform_without_replacement_among_storms_with_any_eligible_label",
        "eligible_labels": list(args.eligible_label),
        "seed": args.seed,
        "holdouts": [
            {
                "id": f"holdout_{index + 1}",
                "storms": [storm],
                "frame_count": len(by_storm[storm]),
                "labels": dict(sorted(Counter(by_storm[storm]).items())),
                "restricted_training_storms": [value for value in all_storms if value != storm],
            }
            for index, storm in enumerate(selected)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
