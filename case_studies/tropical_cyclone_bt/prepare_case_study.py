#!/usr/bin/env python
from __future__ import annotations

import argparse
import errno
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
import yaml
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]


# Both banks come from the selected baseline_ztsnr model. The training bank is
# development-only because that generator saw the complete training split.
DEFAULT_BANKS = {
    "baseline_ztsnr_test": {
        "source": "outputs/baseline_ztsnr/sample_banks/test/test_baseline_n1000",
        "destination": "baseline_ztsnr/test/test_baseline_n1000",
        "role": "track1_held_out_generation",
        "split": "test",
    },
    "baseline_ztsnr_train_development": {
        "source": "outputs/baseline_ztsnr/sample_banks/train/aug_guidance_train",
        "destination": "baseline_ztsnr/train/aug_guidance_train",
        "role": "track2_development_only_full_training_generator",
        "split": "train",
    },
}


def load_split_file_set(path: Path) -> set[str]:
    with path.open() as handle:
        return {line.strip() for line in handle if line.strip()}


def load_dataset_index(
    path: Path,
) -> tuple[dict[int, list[str]], dict[str, dict[str, Any]], dict[str, Any]]:
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
    index_meta = payload.get("meta", {})
    if not isinstance(index_meta, dict):
        raise ValueError(f"Dataset index {path} has invalid meta.")
    return class_to_files, sample_meta, dict(index_meta)


def usa_wind_to_class(wind_kt: float) -> int:
    wind = float(wind_kt)
    if not np.isfinite(wind) or wind < 35.0:
        raise ValueError(f"USA_WIND must be finite and at least 35 kt, got {wind_kt!r}.")
    if 35.0 <= wind < 64.0:
        return 0
    if wind < 83.0:
        return 1
    if wind < 96.0:
        return 2
    if wind < 113.0:
        return 3
    if wind < 137.0:
        return 4
    return 5


def validate_usa_1min_index(meta: dict[str, Any], *, path: Path) -> None:
    if meta.get("wind_source") != "USA_WIND":
        raise ValueError(f"Refusing non-USA_WIND dataset index: {path}")
    if meta.get("wind_averaging_period") != "1-minute":
        raise ValueError(f"Refusing dataset without 1-minute wind provenance: {path}")


def validated_reference_counts(source_repo: Path, split: str) -> dict[str, int]:
    """Return corrected class counts after validating the entire declared split."""

    index_path = source_repo / "data" / "dataset_index_usa1min.json"
    split_path = source_repo / "data" / "splits_usa1min" / f"files_{split}.txt"
    class_to_files, sample_meta, index_meta = load_dataset_index(index_path)
    validate_usa_1min_index(index_meta, path=index_path)
    split_files = load_split_file_set(split_path)
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for class_id, rel_paths in class_to_files.items():
        selected = set(rel_paths) & split_files
        for rel_path in selected:
            if rel_path in seen:
                raise ValueError(f"Duplicate TC sample across classes: {rel_path!r}.")
            meta = sample_meta.get(rel_path, {})
            wind = meta.get("usa_wind_kt")
            if (
                meta.get("wind_source") != "USA_WIND"
                or wind is None
                or usa_wind_to_class(float(wind)) != int(class_id)
            ):
                raise ValueError(f"Invalid USA_WIND class provenance for {rel_path!r}.")
            seen.add(rel_path)
        counts[str(class_id)] = len(selected)
    if seen != split_files:
        missing = sorted(split_files - seen)
        raise ValueError(
            f"Corrected {split} split has {len(missing)} unindexed samples: {missing[:5]}."
        )
    return counts


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
            "Real train/validation/test data are exported as new NPY arrays."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
        help="Real-data splits to export. All three are required for the TC workflow.",
    )
    parser.add_argument(
        "--banks",
        nargs="+",
        choices=sorted(DEFAULT_BANKS),
        default=sorted(DEFAULT_BANKS),
        help="baseline_ztsnr sample banks to port.",
    )
    parser.add_argument(
        "--skip-real",
        action="store_true",
        help="Skip real-split export and operate only on source manifests/sample banks.",
    )
    parser.add_argument(
        "--skip-banks",
        action="store_true",
        help="Skip generated sample-bank transfer and export only the real splits.",
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


def tree_checksum_manifest(
    source: Path,
    output: Path,
    *,
    artifact_id: str,
    role: str,
    provenance: dict[str, Any],
) -> None:
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Generated bank {source} contains no files.")
    payload = {
        "schema": "gauge_climate.artifact_checksums.v1",
        "artifact_id": artifact_id,
        "role": role,
        "wind_source": "USA_WIND",
        "wind_averaging_period": "1-minute",
        "source_name": source.name,
        "source_manifest_sha256": sha256_file(source / "manifest.json"),
        "generator_provenance": provenance,
        "files": {
            path.relative_to(source).as_posix(): sha256_file(path) for path in files
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def validate_generated_bank(
    *, source_repo: Path, bank_path: Path, expected_split: str
) -> dict[str, Any]:
    """Validate the bank's USA_WIND conditioning contract and source model."""

    manifest = load_json(bank_path / "manifest.json")
    required = {
        "schema": "tc_diffusion.sample_bank.v2",
        "run_name": "baseline_ztsnr",
        "split": expected_split,
    }
    bad = {
        key: manifest.get(key)
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    if bad:
        raise ValueError(f"Generated bank has incompatible provenance {bank_path}: {bad}.")
    if manifest.get("conditioning", {}).get("use_wind_speed") is not True:
        raise ValueError(f"Generated bank is not wind-conditioned: {bank_path}.")

    config_rel = Path(str(manifest.get("config_path", "")))
    weights_rel = Path(str(manifest.get("weights_path", "")))
    if config_rel.is_absolute() or weights_rel.is_absolute():
        raise ValueError("Generated-bank model paths must be relative to the source repository.")
    config_path = source_repo / config_rel
    weights_path = source_repo / weights_rel
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError("Generated-bank config or checkpoint is missing.")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in generator config {config_path}.")
    data_config = config.get("data", {})
    if not str(data_config.get("dataset_index", "")).endswith(
        "dataset_index_usa1min.json"
    ) or not str(data_config.get("split_dir", "")).endswith("splits_usa1min"):
        raise ValueError(
            f"Generator {config_path} was not configured for the USA_WIND data contract."
        )

    observed_counts: dict[str, int] = {}
    image_shape = tuple(int(value) for value in manifest.get("image_shape", ()))
    for raw_class, shards in manifest.get("shards_by_class", {}).items():
        class_id = int(raw_class)
        total = 0
        for shard in shards:
            count = int(shard["count"])
            bt_path = bank_path / str(shard["bt_k"])
            wind_path = bank_path / str(shard["wind_kt"])
            bt = np.load(bt_path, mmap_mode="r")
            wind = np.asarray(np.load(wind_path, mmap_mode="r"))
            if bt.shape != (count, *image_shape) or wind.shape != (count,):
                raise ValueError(f"Generated shard arrays are misaligned: {bt_path}.")
            if any(usa_wind_to_class(value) != class_id for value in wind):
                raise ValueError(
                    f"Generated conditioning winds disagree with class {class_id}: {wind_path}."
                )
            total += count
        observed_counts[str(class_id)] = total
    declared_counts = {
        str(key): int(value)
        for key, value in manifest.get("generated_counts_by_class", {}).items()
    }
    if observed_counts != declared_counts:
        raise ValueError(
            f"Generated bank counts disagree with its manifest: {bank_path}."
        )
    reference_counts = {
        str(key): int(value)
        for key, value in manifest.get("reference_counts_by_class", {}).items()
    }
    expected_reference = validated_reference_counts(source_repo, expected_split)
    if reference_counts != expected_reference:
        raise ValueError(
            f"Generated bank reference counts do not match the corrected {expected_split} "
            f"partition: {bank_path}."
        )
    return {
        "run_name": "baseline_ztsnr",
        "config_path": config_rel.as_posix(),
        "config_sha256": sha256_file(config_path),
        "weights_path": weights_rel.as_posix(),
        "weights_sha256": sha256_file(weights_path),
        "dataset_index": "data/dataset_index_usa1min.json",
        "split_directory": "data/splits_usa1min",
    }


def link_or_copy_tree(src: Path, dst: Path, *, mode: str, overwrite: bool) -> str:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            raise FileExistsError(
                f"Generated bank destination already exists: {dst}. "
                "Use --overwrite to replace and revalidate it."
            )
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src.resolve(), target_is_directory=True)
        return "symlink"
    if mode == "hardlink":
        warned = False

        def hardlink_or_copy(source: str, destination: str) -> str:
            nonlocal warned
            try:
                os.link(source, destination)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                if not warned:
                    print(
                        "[warn] Source and destination do not share a hardlink-capable "
                        "filesystem; falling back to byte copies."
                    )
                    warned = True
                shutil.copy2(source, destination)
            return destination

        shutil.copytree(src, dst, copy_function=hardlink_or_copy)
        return "copy (hardlink unavailable)" if warned else "hardlink"
    shutil.copytree(src, dst)
    return "copy"


def storm_id_from_rel_path(value: str) -> str:
    match = re.match(r"^[^-]+-([^_]+)_\d{8}T\d{4}", Path(value).name)
    if match is None:
        raise ValueError(f"Cannot parse storm id from {value!r}.")
    return match.group(1)


def validate_existing_real_manifest(
    manifest: dict[str, Any],
    *,
    path: Path,
    expected_split: str,
    source_repo: Path,
) -> None:
    required = {
        "schema": "gauge_climate.tc_bt_real_dataset.v2",
        "wind_source": "USA_WIND",
        "wind_averaging_period": "1-minute",
        "split": expected_split,
    }
    mismatches = {
        key: (manifest.get(key), expected)
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"Refusing stale or mixed-period TC archive {path}: {mismatches}. "
            "Re-run with --overwrite after removing the old archive."
        )
    for relative_key, checksum_key in {
        "source_dataset_index": "source_dataset_index_sha256",
        "source_split_file": "source_split_file_sha256",
        "source_packed_manifest": "source_packed_manifest_sha256",
    }.items():
        source = source_repo / str(manifest.get(relative_key, "missing"))
        if not source.is_file() or sha256_file(source) != manifest.get(checksum_key):
            raise ValueError(
                f"TC archive source provenance changed for {relative_key}: {source}."
            )
    for relative, expected in manifest.get("file_sha256", {}).items():
        artifact = path.parent / str(relative)
        if not artifact.is_file():
            raise ValueError(f"TC archive file is missing: {artifact}")
        observed = sha256_file(artifact)
        if observed != expected:
            raise ValueError(f"TC archive checksum mismatch: {artifact}")


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
        validate_existing_real_manifest(
            load_json(manifest_path),
            path=manifest_path,
            expected_split=split,
            source_repo=source_repo,
        )
        print(f"Real {split} export already exists: {manifest_path}")
        return

    dataset_index_path = source_repo / "data" / "dataset_index_usa1min.json"
    split_file = source_repo / "data" / "splits_usa1min" / f"files_{split}.txt"
    packed_manifest = source_repo / "data" / "packed_tc_memmap_usa1min" / "manifest.json"

    class_to_files, sample_meta, index_meta = load_dataset_index(dataset_index_path)
    validate_usa_1min_index(index_meta, path=dataset_index_path)
    split_files = load_split_file_set(split_file)
    backend = PackedMemmapBackend(packed_manifest)

    selected: list[tuple[str, int, float]] = []
    for class_id, rels in sorted(class_to_files.items()):
        for rel in rels:
            if rel not in split_files:
                continue
            meta = sample_meta.get(rel)
            if meta is None:
                raise ValueError(f"Missing USA_WIND metadata for {rel!r}.")
            if meta.get("wind_source") != "USA_WIND":
                raise ValueError(f"Sample {rel!r} is not explicitly sourced from USA_WIND.")
            wind = meta.get("usa_wind_kt")
            canonical = meta.get("wind_kt")
            if wind is None or not np.isfinite(float(wind)):
                raise ValueError(f"Sample {rel!r} lacks a finite USA_WIND value.")
            if canonical is None or not np.isclose(float(canonical), float(wind)):
                raise ValueError(f"Sample {rel!r} has inconsistent canonical wind metadata.")
            if int(meta.get("ss_cat", -1)) != int(class_id):
                raise ValueError(f"Sample {rel!r} has inconsistent class metadata.")
            if usa_wind_to_class(float(wind)) != int(class_id):
                raise ValueError(f"Sample {rel!r} class is inconsistent with USA_WIND.")
            selected.append((rel, int(class_id), float(wind)))
    selected = sorted(selected, key=lambda row: row[0])
    selected_paths = [row[0] for row in selected]
    if len(selected_paths) != len(set(selected_paths)):
        raise ValueError(f"The corrected {split} index contains duplicate samples.")
    if set(selected_paths) != split_files:
        missing = sorted(split_files - set(selected_paths))
        raise ValueError(
            f"The corrected {split} split has {len(missing)} unindexed samples: "
            f"{missing[:5]}."
        )
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
        "schema": "gauge_climate.tc_bt_real_dataset.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": "external:tc-diffusion",
        "source_dataset_index": "data/dataset_index_usa1min.json",
        "source_split_file": f"data/splits_usa1min/files_{split}.txt",
        "source_packed_manifest": "data/packed_tc_memmap_usa1min/manifest.json",
        "wind_source": "USA_WIND",
        "wind_averaging_period": "1-minute",
        "category_source": "Saffir-Simpson category derived from USA_WIND",
        "source_dataset_index_sha256": sha256_file(dataset_index_path),
        "source_split_file_sha256": sha256_file(split_file),
        "source_packed_manifest_sha256": sha256_file(packed_manifest),
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
    bank_ids: Iterable[str],
) -> None:
    generated_root = output_root / "generated"
    for bank_id in bank_ids:
        spec = DEFAULT_BANKS[bank_id]
        src = source_repo / str(spec["source"])
        if not src.exists():
            raise FileNotFoundError(f"Expected sample bank not found: {src}")
        provenance = validate_generated_bank(
            source_repo=source_repo,
            bank_path=src,
            expected_split=str(spec["split"]),
        )
        dst = generated_root / str(spec["destination"])
        effective_mode = link_or_copy_tree(src, dst, mode=mode, overwrite=overwrite)
        tree_checksum_manifest(
            dst,
            generated_root / "manifests" / f"{bank_id}.json",
            artifact_id=bank_id,
            role=str(spec["role"]),
            provenance=provenance,
        )
        print(f"{effective_mode}: {src} -> {dst}")


def copy_source_manifests(*, source_repo: Path, output_root: Path, overwrite: bool) -> None:
    dst_root = output_root / "source_manifests"
    dst_root.mkdir(parents=True, exist_ok=True)
    for rel in [
        "data/dataset_index_usa1min.json",
        "data/splits_usa1min/files_test.txt",
        "data/splits_usa1min/files_train.txt",
        "data/splits_usa1min/files_val.txt",
        "data/splits_usa1min/split_report.json",
        "data/packed_tc_memmap_usa1min/manifest.json",
        "data/packed_tc_memmap_usa1min/rel_paths.txt",
        "runs/baseline_ztsnr/config.yaml",
        "runs/baseline_ztsnr/meta.json",
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
    if not args.skip_real:
        for split in args.splits:
            export_real_data(
                source_repo=source_repo,
                output_root=output_root,
                split=split,
                overwrite=bool(args.overwrite),
                progress=not bool(args.no_progress),
            )
    if not args.skip_banks:
        port_generated_banks(
            source_repo=source_repo,
            output_root=output_root,
            mode=str(args.copy_mode),
            overwrite=bool(args.overwrite),
            bank_ids=args.banks,
        )


if __name__ == "__main__":
    main()
