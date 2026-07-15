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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows_from_real_manifest(path: Path) -> tuple[list[dict[str, str]], dict[str, object]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": "gauge_climate.tc_bt_real_dataset.v2",
        "split": "train",
        "wind_source": "USA_WIND",
        "wind_averaging_period": "1-minute",
    }
    mismatches = {
        key: manifest.get(key)
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Invalid corrected training manifest {path}: {mismatches}.")
    root = path.parent
    labels_path = root / str(manifest["labels"])
    storms_path = root / str(manifest["storm_ids"])
    labels = np.load(labels_path, mmap_mode="r")
    storms = np.load(storms_path, mmap_mode="r")
    if labels.shape != storms.shape or labels.ndim != 1:
        raise ValueError("Training labels and storm IDs must be aligned vectors.")
    rows = [
        {"storm_id": str(storm), "label": str(int(label))}
        for storm, label in zip(storms, labels, strict=True)
    ]
    return rows, {
        "manifest": str(path),
        "manifest_sha256": _sha256(path),
        "labels_sha256": _sha256(labels_path),
        "storm_ids_sha256": _sha256(storms_path),
        "wind_source": "USA_WIND",
        "wind_averaging_period": "1-minute",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--real-manifest", type=Path)
    source.add_argument("--index-csv", type=Path)
    parser.add_argument("--storm-column", default="storm_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--eligible-label", action="append", required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1789)
    parser.add_argument("--output", type=Path, default=Path("case_studies/tropical_cyclone_bt/track2_holdouts.json"))
    args = parser.parse_args()
    if args.real_manifest is not None:
        rows, source_provenance = _rows_from_real_manifest(args.real_manifest)
    else:
        with args.index_csv.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        source_provenance = {
            "index_csv": str(args.index_csv),
            "index_csv_sha256": _sha256(args.index_csv),
        }
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
        "source": source_provenance,
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
