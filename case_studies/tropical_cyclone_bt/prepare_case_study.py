#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from climategen_eval.data import (
    PackedMemmapBackend,
    class_midpoint_wind_kt,
    counts_by_label,
    load_dataset_index,
    load_split_file_set,
)


# usa1min case-study models (refreshed 2026-07): baseline (ZTSNR), physics LOSS, physics GUIDANCE.
DEFAULT_BANKS = {
    "baseline": "outputs/baseline_ztsnr/sample_banks/test/test_baseline_n1000",
    "pinn_loss": "outputs/pinn_cc_lam0.0002/sample_banks/test/test_pinnloss_n1000",
    "pinn_guided": "outputs/baseline_ztsnr/sample_banks/test/test_guidance_n1000",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the TC BT case-study data directory from the tc-diffusion workspace."
    )
    parser.add_argument(
        "--source-repo",
        default="../tc-diffusion",
        help="Path to the original tc-diffusion repo.",
    )
    parser.add_argument("--output", default="data/tropical_cyclone_bt")
    parser.add_argument(
        "--copy-mode",
        choices=["symlink", "copy", "hardlink"],
        default="symlink",
        help=(
            "How to port generated sample-bank files. "
            "Real test data are exported as new NPY arrays."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def link_or_copy_tree(src: Path, dst: Path, *, mode: str, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src.resolve(), target_is_directory=True)
        return
    if mode == "hardlink":
        shutil.copytree(src, dst, copy_function=os.link)
        return
    shutil.copytree(src, dst)


def export_real_test_data(
    *,
    source_repo: Path,
    output_root: Path,
    overwrite: bool,
    progress: bool,
) -> None:
    real_root = output_root / "real_test"
    manifest_path = real_root / "manifest.json"
    if manifest_path.exists() and not overwrite:
        print(f"Real test export already exists: {manifest_path}")
        return

    dataset_index_path = source_repo / "data" / "dataset_index_usa1min.json"
    split_file = source_repo / "data" / "splits_usa1min" / "files_test.txt"
    packed_manifest = source_repo / "data" / "packed_tc_memmap_usa1min" / "manifest.json"

    class_to_files, sample_meta = load_dataset_index(dataset_index_path)
    test_files = load_split_file_set(split_file)
    backend = PackedMemmapBackend(packed_manifest)

    selected: list[tuple[str, int, float]] = []
    for class_id, rels in sorted(class_to_files.items()):
        for rel in rels:
            if rel not in test_files:
                continue
            meta = sample_meta.get(rel, {})
            wind = meta.get("wmo_wind_kt")
            if wind is None or not np.isfinite(float(wind)):
                wind = class_midpoint_wind_kt(class_id)
            selected.append((rel, int(class_id), float(wind)))
    selected = sorted(selected, key=lambda row: row[0])
    if not selected:
        raise ValueError("No test files were selected from the source manifests.")

    real_root.mkdir(parents=True, exist_ok=True)
    bt_path = real_root / "bt_k.npy"
    labels_path = real_root / "labels.npy"
    wind_path = real_root / "wind_kt.npy"
    rel_paths_path = real_root / "rel_paths.txt"

    first = backend.load_bt(selected[0][0])
    bt = np.lib.format.open_memmap(
        bt_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(selected), *first.shape),
    )
    labels = np.empty((len(selected),), dtype=np.int16)
    winds = np.empty((len(selected),), dtype=np.float32)
    with rel_paths_path.open("w") as f:
        for idx, (rel_path, class_id, wind_kt) in enumerate(
            tqdm(
                selected,
                desc="Exporting real test BT",
                unit="image",
                disable=not progress,
            )
        ):
            bt[idx] = backend.load_bt(rel_path)
            labels[idx] = int(class_id)
            winds[idx] = float(wind_kt)
            f.write(f"{rel_path}\n")
    bt.flush()
    np.save(labels_path, labels)
    np.save(wind_path, winds)

    manifest = {
        "schema": "climategen_eval.real_bt_dataset.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": str(source_repo.resolve()),
        "source_dataset_index": str(dataset_index_path),
        "source_split_file": str(split_file),
        "source_packed_manifest": str(packed_manifest),
        "split": "test",
        "bt_k": bt_path.name,
        "labels": labels_path.name,
        "wind_kt": wind_path.name,
        "rel_paths": rel_paths_path.name,
        "num_samples": int(len(selected)),
        "image_shape": [int(first.shape[0]), int(first.shape[1])],
        "counts_by_class": {str(k): int(v) for k, v in counts_by_label(labels).items()},
    }
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {manifest_path}")


def port_generated_banks(
    *,
    source_repo: Path,
    output_root: Path,
    mode: str,
    overwrite: bool,
) -> None:
    generated_root = output_root / "generated"
    for label, rel in DEFAULT_BANKS.items():
        src = source_repo / rel
        if not src.exists():
            raise FileNotFoundError(f"Expected sample bank not found: {src}")
        dst = generated_root / label / src.name
        link_or_copy_tree(src, dst, mode=mode, overwrite=overwrite)
        print(f"{mode}: {src} -> {dst}")


def copy_source_manifests(*, source_repo: Path, output_root: Path, overwrite: bool) -> None:
    dst_root = output_root / "source_manifests"
    dst_root.mkdir(parents=True, exist_ok=True)
    for rel in [
        "data/dataset_index_usa1min.json",
        "data/splits_usa1min/files_test.txt",
        "data/splits_usa1min/split_report.json",
        "data/packed_tc_memmap_usa1min/manifest.json",
        "data/packed_tc_memmap_usa1min/rel_paths.txt",
    ]:
        src = source_repo / rel
        dst = dst_root / rel.replace("/", "__")
        if dst.exists() and not overwrite:
            continue
        if not src.exists():
            print(f"[warn] source manifest not found, skipping: {src}")
            continue
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    source_repo = Path(args.source_repo)
    if not source_repo.is_absolute():
        source_repo = (_REPO_ROOT / source_repo).resolve()
    output_root = Path(args.output)
    if not output_root.is_absolute():
        output_root = _REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    copy_source_manifests(
        source_repo=source_repo,
        output_root=output_root,
        overwrite=bool(args.overwrite),
    )
    export_real_test_data(
        source_repo=source_repo,
        output_root=output_root,
        overwrite=bool(args.overwrite),
        progress=not bool(args.no_progress),
    )
    port_generated_banks(
        source_repo=source_repo,
        output_root=output_root,
        mode=str(args.copy_mode),
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
