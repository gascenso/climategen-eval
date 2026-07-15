#!/usr/bin/env python3
"""Generate the auditable TC storm/frame/class split table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CLASS_NAMES = {0: "TS", 1: "C1", 2: "C2", 3: "C3", 4: "C4", 5: "C5"}


def load_split(root: Path, split: str):
    split_root = root / f"real_{split}"
    manifest = json.loads((split_root / "manifest.json").read_text(encoding="utf-8"))
    labels = np.load(split_root / manifest["labels"], mmap_mode="r")
    storms = np.load(split_root / manifest["storm_ids"], mmap_mode="r")
    if labels.shape != storms.shape:
        raise ValueError(f"{split} labels and storm ids do not align.")
    return manifest, np.asarray(labels), np.asarray(storms).astype(str)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/tropical_cyclone_bt"))
    parser.add_argument("--output", type=Path, default=Path("case_studies/tropical_cyclone_bt/partition_summary"))
    args = parser.parse_args()
    loaded = {split: load_split(args.data, split) for split in ("train", "test")}
    overlap = sorted(set(loaded["train"][2]) & set(loaded["test"][2]))
    if overlap:
        raise ValueError(f"Storm leakage across train/test: {overlap[:10]}.")
    rows = []
    cross_category = {}
    for split, (_, labels, storms) in loaded.items():
        unknown = sorted(set(labels.tolist()) - set(CLASS_NAMES))
        if unknown:
            raise ValueError(f"Unknown {split} class ids: {unknown}.")
        for class_id, class_name in CLASS_NAMES.items():
            mask = labels == class_id
            rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "frames": int(mask.sum()),
                    "storms_with_frames": int(np.unique(storms[mask]).size),
                }
            )
        labels_by_storm = {
            storm: sorted(set(labels[storms == storm].tolist())) for storm in np.unique(storms)
        }
        cross_category[split] = sum(len(values) > 1 for values in labels_by_storm.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema": "gauge_climate.tc_bt_partition_summary.v1",
        "storm_overlap_count": 0,
        "cross_category_storms": cross_category,
        "rows": rows,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
