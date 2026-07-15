#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]


# usa1min case-study models (refreshed 2026-07): baseline (ZTSNR), physics LOSS, physics GUIDANCE.
DEFAULT_BANKS = {
    "baseline": "outputs/baseline_ztsnr/sample_banks/test/test_baseline_n1000",
    "pinn_loss": "outputs/pinn_cc_lam0.0002/sample_banks/test/test_pinnloss_n1000",
    "pinn_guided": "outputs/baseline_ztsnr/sample_banks/test/test_guidance_n1000",
}


def load_split_file_set(path: Path) -> set[str]:
    with path.open() as handle:
        return {line.strip() for line in handle if line.strip()}


def load_dataset_index(path: Path) -> tuple[dict[int, list[str]], dict[str, dict[str, Any]]]:
    payload = load_json(path)
    raw_classes = payload.get("classes")
    if not isinstance(raw_classes, dict):
        raise ValueError(f"Dataset index {path} is missing classes.")
    class_to_files = {
        int(class_id): [str(value) for value in values]
        for class_id, values in raw_classes.items()
    }
    raw_samples = payload.get("samples", {})
    sample_meta = (
        {
            str(rel_path): dict(node)
            for rel_path, node in raw_samples.items()
            if isinstance(node, dict)
        }
        if isinstance(raw_samples, dict)
        else {}
    )
    return class_to_files, sample_meta


def class_midpoint_wind_kt(class_id: int) -> float:
    return {0: 49.0, 1: 73.0, 2: 89.0, 3: 104.0, 4: 124.5}.get(
        int(class_id), 145.0
    )


def counts_by_label(labels: Iterable[int]) -> dict[int, int]:
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


@dataclass(frozen=True)
class PackedShardInfo:
    path: str
    start: int
    stop: int


class PackedMemmapBackend:
    """Read BT fields from the source repository's sharded NumPy store."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        manifest = load_json(manifest_path)
        if str(manifest.get("backend", "")).lower() != "packed_memmap":
            raise ValueError(f"Expected a packed_memmap manifest: {manifest_path}")
        self.root = manifest_path.parent
        self.num_samples = int(manifest["num_samples"])
        rel_paths_file = self.root / str(manifest["rel_paths_file"])
        with rel_paths_file.open() as handle:
            self.rel_paths = [line.strip() for line in handle if line.strip()]
        if len(self.rel_paths) != self.num_samples:
            raise ValueError("Packed rel_paths count does not match manifest num_samples.")
        self.path_to_index = {value: index for index, value in enumerate(self.rel_paths)}
        self.shards = [
            PackedShardInfo(
                path=str(shard["path"]),
                start=int(shard["start"]),
                stop=int(shard["stop"]),
            )
            for shard in manifest["shards"]
        ]
        self._starts = [shard.start for shard in self.shards]
        self._arrays = [
            np.load(self.root / shard.path, mmap_mode="r") for shard in self.shards
        ]

    def load_bt(self, rel_path: str) -> np.ndarray:
        try:
            global_index = self.path_to_index[str(rel_path)]
        except KeyError as exc:
            raise KeyError(f"Packed manifest does not contain {rel_path!r}.") from exc
        shard_index = bisect_right(self._starts, global_index) - 1
        if shard_index < 0 or global_index >= self.shards[shard_index].stop:
            raise IndexError(f"Could not resolve packed index {global_index}.")
        shard = self.shards[shard_index]
        return np.asarray(
            self._arrays[shard_index][global_index - shard.start], dtype=np.float32
        )


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
        default="copy",
        help=(
            "How to port generated sample-bank files (copy is release-safe; "
            "symlink is local-development only). "
            "Real train/test data are exported as new NPY arrays."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "test"],
        default=["train", "test"],
        help="Real-data splits to export. Both are required for a final GAUGE run.",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def real_file_checksums(real_root: Path, manifest: dict[str, Any]) -> dict[str, str]:
    keys = ("bt_k", "labels", "wind_kt", "storm_ids", "rel_paths")
    return {
        str(manifest[key]): sha256_file(real_root / str(manifest[key]))
        for key in keys
        if manifest.get(key) and (real_root / str(manifest[key])).is_file()
    }


def tree_checksum_manifest(source: Path, output: Path, *, artifact_id: str) -> None:
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Generated bank {source} contains no files.")
    payload = {
        "schema": "gauge_climate.artifact_checksums.v1",
        "artifact_id": artifact_id,
        "source_name": source.name,
        "files": {
            path.relative_to(source).as_posix(): sha256_file(path) for path in files
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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


def storm_id_from_rel_path(value: str) -> str:
    match = re.match(r"^[^-]+-([^_]+)_\d{8}T\d{4}", Path(value).name)
    if match is None:
        raise ValueError(f"Cannot parse storm id from {value!r}.")
    return match.group(1)


def _migrate_existing_real_manifest(real_root: Path, manifest: dict[str, Any]) -> None:
    rel_paths_path = real_root / str(manifest["rel_paths"])
    storm_path = real_root / "storm_ids.npy"
    if not storm_path.exists():
        rel_paths = [line.strip() for line in rel_paths_path.read_text().splitlines() if line.strip()]
        storms = np.asarray([storm_id_from_rel_path(value) for value in rel_paths])
        np.save(storm_path, storms)
    storms = np.load(storm_path)
    manifest.update(
        {
            "schema": "gauge_climate.tc_bt_real_dataset.v1",
            "storm_ids": storm_path.name,
            "num_storms": int(np.unique(storms).size),
            "source_repo": "external:tc-diffusion",
            "source_dataset_index": "data/dataset_index_usa1min.json",
            "source_split_file": f"data/splits_usa1min/files_{manifest['split']}.txt",
            "source_packed_manifest": "data/packed_tc_memmap_usa1min/manifest.json",
        }
    )
    manifest["file_sha256"] = real_file_checksums(real_root, manifest)
    with (real_root / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def export_real_data(
    *,
    source_repo: Path,
    output_root: Path,
    split: str,
    overwrite: bool,
    progress: bool,
) -> None:
    real_root = output_root / f"real_{split}"
    manifest_path = real_root / "manifest.json"
    if manifest_path.exists() and not overwrite:
        _migrate_existing_real_manifest(real_root, load_json(manifest_path))
        print(f"Real {split} export already exists: {manifest_path}")
        return

    dataset_index_path = source_repo / "data" / "dataset_index_usa1min.json"
    split_file = source_repo / "data" / "splits_usa1min" / f"files_{split}.txt"
    packed_manifest = source_repo / "data" / "packed_tc_memmap_usa1min" / "manifest.json"

    class_to_files, sample_meta = load_dataset_index(dataset_index_path)
    split_files = load_split_file_set(split_file)
    backend = PackedMemmapBackend(packed_manifest)

    selected: list[tuple[str, int, float]] = []
    for class_id, rels in sorted(class_to_files.items()):
        for rel in rels:
            if rel not in split_files:
                continue
            meta = sample_meta.get(rel, {})
            wind = meta.get("wmo_wind_kt")
            if wind is None or not np.isfinite(float(wind)):
                wind = class_midpoint_wind_kt(class_id)
            selected.append((rel, int(class_id), float(wind)))
    selected = sorted(selected, key=lambda row: row[0])
    if not selected:
        raise ValueError(f"No {split} files were selected from the source manifests.")

    real_root.mkdir(parents=True, exist_ok=True)
    bt_path = real_root / "bt_k.npy"
    labels_path = real_root / "labels.npy"
    wind_path = real_root / "wind_kt.npy"
    storm_path = real_root / "storm_ids.npy"
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
    storms = []
    with rel_paths_path.open("w") as f:
        for idx, (rel_path, class_id, wind_kt) in enumerate(
            tqdm(
                selected,
                desc=f"Exporting real {split} BT",
                unit="image",
                disable=not progress,
            )
        ):
            bt[idx] = backend.load_bt(rel_path)
            labels[idx] = int(class_id)
            winds[idx] = float(wind_kt)
            storms.append(storm_id_from_rel_path(rel_path))
            f.write(f"{rel_path}\n")
    bt.flush()
    np.save(labels_path, labels)
    np.save(wind_path, winds)
    np.save(storm_path, np.asarray(storms))

    manifest = {
        "schema": "gauge_climate.tc_bt_real_dataset.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": "external:tc-diffusion",
        "source_dataset_index": "data/dataset_index_usa1min.json",
        "source_split_file": f"data/splits_usa1min/files_{split}.txt",
        "source_packed_manifest": "data/packed_tc_memmap_usa1min/manifest.json",
        "split": split,
        "bt_k": bt_path.name,
        "labels": labels_path.name,
        "wind_kt": wind_path.name,
        "storm_ids": storm_path.name,
        "rel_paths": rel_paths_path.name,
        "num_samples": int(len(selected)),
        "num_storms": int(np.unique(storms).size),
        "image_shape": [int(first.shape[0]), int(first.shape[1])],
        "counts_by_class": {str(k): int(v) for k, v in counts_by_label(labels).items()},
    }
    manifest["file_sha256"] = real_file_checksums(real_root, manifest)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
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
        tree_checksum_manifest(
            src,
            generated_root / "manifests" / f"{label}.json",
            artifact_id=label,
        )
        print(f"{mode}: {src} -> {dst}")


def copy_source_manifests(*, source_repo: Path, output_root: Path, overwrite: bool) -> None:
    dst_root = output_root / "source_manifests"
    dst_root.mkdir(parents=True, exist_ok=True)
    for rel in [
        "data/dataset_index_usa1min.json",
        "data/splits_usa1min/files_test.txt",
        "data/splits_usa1min/files_train.txt",
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
    for split in args.splits:
        export_real_data(
            source_repo=source_repo,
            output_root=output_root,
            split=split,
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
