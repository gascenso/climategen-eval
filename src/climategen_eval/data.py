from __future__ import annotations

import json
import re
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


# IBTrACS-style SID embedded in patch filenames of the form
# "<seq>-<SID>_<YYYYMMDD>T<HHMM>_ml.nc". The SID itself is the canonical
# IBTrACS storm identifier (e.g. "1994348N04175") and is shared by every
# image extracted from the same storm track. Two images from the same SID
# are not exchangeable observations for bootstrap purposes.
_STORM_SID_RE = re.compile(r"^[0-9]+-(?P<sid>[0-9A-Z]+)_[0-9]{8}T[0-9]{4}")


def storm_id_from_rel_path(rel_path: str) -> str:
    """Extract the IBTrACS storm SID from a packed patch rel_path.

    Falls back to the basename stem so that out-of-format paths still yield a
    deterministic cluster identifier (each path becomes its own singleton
    cluster, which is the most conservative behavior for the bootstrap).
    """
    name = Path(str(rel_path)).name
    match = _STORM_SID_RE.match(name)
    if match is not None:
        return match.group("sid")
    return Path(str(rel_path)).stem


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    out = Path(path)
    if out.is_absolute():
        return out
    return (repo_root() if base is None else Path(base)) / out


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def load_split_file_set(split_file: str | Path) -> set[str]:
    with Path(split_file).open("r") as f:
        return {line.strip() for line in f if line.strip()}


def load_dataset_index(
    index_path: str | Path,
) -> tuple[dict[int, list[str]], dict[str, dict[str, Any]]]:
    index = load_json(index_path)
    raw_classes = index.get("classes")
    if not isinstance(raw_classes, dict):
        raise ValueError(f"Dataset index {index_path} is missing classes.")
    class_to_files = {int(k): [str(v) for v in values] for k, values in raw_classes.items()}

    sample_meta: dict[str, dict[str, Any]] = {}
    raw_samples = index.get("samples", {})
    if isinstance(raw_samples, dict):
        for rel_path, node in raw_samples.items():
            if isinstance(node, dict):
                sample_meta[str(rel_path)] = dict(node)
    return class_to_files, sample_meta


def class_midpoint_wind_kt(class_id: int) -> float:
    cls = int(class_id)
    if cls == 0:
        return 49.0
    if cls == 1:
        return 73.0
    if cls == 2:
        return 89.0
    if cls == 3:
        return 104.0
    if cls == 4:
        return 124.5
    return 145.0


@dataclass(frozen=True)
class PackedShardInfo:
    path: str
    start: int
    stop: int


class PackedMemmapBackend:
    """Read raw BT arrays from sharded `.npy` memmaps."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        manifest = load_json(self.manifest_path)
        if str(manifest.get("backend", "")).lower() != "packed_memmap":
            raise ValueError(f"Expected a packed_memmap manifest: {self.manifest_path}")

        self.root = self.manifest_path.parent
        self.sample_shape = tuple(int(v) for v in manifest["sample_shape"])
        self.dtype = np.dtype(manifest["dtype"])
        self.num_samples = int(manifest["num_samples"])

        rel_paths_file = self.root / str(manifest["rel_paths_file"])
        with rel_paths_file.open("r") as f:
            self.rel_paths = [line.strip() for line in f if line.strip()]
        if len(self.rel_paths) != self.num_samples:
            raise ValueError("Packed rel_paths count does not match manifest num_samples.")
        self.path_to_index = {rel_path: idx for idx, rel_path in enumerate(self.rel_paths)}

        self.shards = [
            PackedShardInfo(
                path=str(shard["path"]),
                start=int(shard["start"]),
                stop=int(shard["stop"]),
            )
            for shard in manifest["shards"]
        ]
        self._shard_starts = [shard.start for shard in self.shards]
        self._memmaps = [np.load(self.root / shard.path, mmap_mode="r") for shard in self.shards]

    def _resolve_shard(self, global_index: int) -> tuple[int, PackedShardInfo]:
        idx = int(global_index)
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Packed sample index out of range: {idx}")
        shard_idx = bisect_right(self._shard_starts, idx) - 1
        if shard_idx < 0:
            raise IndexError(f"Could not resolve shard for packed index {idx}")
        shard = self.shards[shard_idx]
        if idx >= shard.stop:
            raise IndexError(f"Packed index {idx} is outside shard {shard}")
        return shard_idx, shard

    def load_bt_by_index(self, global_index: int) -> np.ndarray:
        shard_idx, shard = self._resolve_shard(int(global_index))
        local_idx = int(global_index) - shard.start
        return np.asarray(self._memmaps[shard_idx][local_idx], dtype=np.float32)

    def load_bt(self, rel_path: str) -> np.ndarray:
        try:
            idx = self.path_to_index[str(rel_path)]
        except KeyError as exc:
            raise KeyError(f"Packed manifest does not contain rel_path {rel_path!r}") from exc
        return self.load_bt_by_index(idx)

    def load_bt_batch(self, rel_paths: Sequence[str]) -> np.ndarray:
        rows = [self.load_bt(str(rel_path)) for rel_path in rel_paths]
        if not rows:
            raise ValueError("load_bt_batch requires at least one rel_path.")
        return np.stack(rows, axis=0).astype(np.float32, copy=False)


@dataclass(frozen=True)
class RealDataset:
    root: Path
    manifest_path: Path
    bt_path: Path
    labels_path: Path
    rel_paths_path: Path
    wind_path: Path | None
    class_ids: tuple[int, ...]
    counts_by_class: dict[int, int]

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "RealDataset":
        manifest_path = Path(manifest_path)
        manifest = load_json(manifest_path)
        root = manifest_path.parent
        if manifest.get("schema") != "climategen_eval.real_bt_dataset.v1":
            raise ValueError(f"Unsupported real dataset schema: {manifest_path}")
        counts = {int(k): int(v) for k, v in manifest.get("counts_by_class", {}).items()}
        wind_rel = manifest.get("wind_kt")
        return cls(
            root=root,
            manifest_path=manifest_path,
            bt_path=root / str(manifest["bt_k"]),
            labels_path=root / str(manifest["labels"]),
            rel_paths_path=root / str(manifest["rel_paths"]),
            wind_path=None if wind_rel is None else root / str(wind_rel),
            class_ids=tuple(sorted(counts)),
            counts_by_class=counts,
        )

    def rel_paths(self) -> list[str]:
        with self.rel_paths_path.open("r") as f:
            return [line.strip() for line in f if line.strip()]

    def labels(self) -> np.ndarray:
        return np.load(self.labels_path, mmap_mode="r")

    def bt(self, *, mmap_mode: str | None = "r") -> np.ndarray:
        return np.load(self.bt_path, mmap_mode=mmap_mode)

    def class_indices(self, class_id: int) -> np.ndarray:
        labels = np.asarray(self.labels())
        return np.flatnonzero(labels == int(class_id))

    def load_bt_by_class(self, class_id: int, *, mmap_mode: str | None = "r") -> np.ndarray:
        bt = self.bt(mmap_mode=mmap_mode)
        return np.asarray(bt[self.class_indices(int(class_id))], dtype=np.float32)

    def storm_ids_by_class(self, class_id: int) -> np.ndarray:
        """IBTrACS-style storm SID aligned with `load_bt_by_class(class_id)`.

        Returned as a string array so the bootstrap can group images by track.
        """
        rel_paths = self.rel_paths()
        indices = self.class_indices(int(class_id))
        sids = [storm_id_from_rel_path(rel_paths[int(i)]) for i in indices]
        return np.asarray(sids, dtype=object)


@dataclass(frozen=True)
class SampleBankShard:
    part_index: int
    count: int
    bt_path: Path
    wind_path: Path | None = None


@dataclass(frozen=True)
class SampleBank:
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    run_name: str
    split: str
    name: str
    class_ids: tuple[int, ...]
    generated_counts_by_class: dict[int, int]
    shards_by_class: dict[int, tuple[SampleBankShard, ...]]
    image_shape: tuple[int, int]

    @classmethod
    def from_dir(cls, root: str | Path) -> "SampleBank":
        root = Path(root)
        manifest_path = root / "manifest.json"
        manifest = load_json(manifest_path)
        if manifest.get("schema") != "tc_diffusion.sample_bank.v2":
            raise ValueError(f"Unsupported sample-bank schema: {manifest_path}")

        shards_by_class: dict[int, tuple[SampleBankShard, ...]] = {}
        generated_counts_by_class: dict[int, int] = {}
        for raw_class_id, raw_shards in manifest.get("shards_by_class", {}).items():
            class_id = int(raw_class_id)
            shards: list[SampleBankShard] = []
            count = 0
            for raw_shard in raw_shards:
                shard = SampleBankShard(
                    part_index=int(raw_shard["part_index"]),
                    count=int(raw_shard["count"]),
                    bt_path=root / str(raw_shard["bt_k"]),
                    wind_path=(
                        None
                        if raw_shard.get("wind_kt") is None
                        else root / str(raw_shard["wind_kt"])
                    ),
                )
                shards.append(shard)
                count += shard.count
            shards_by_class[class_id] = tuple(sorted(shards, key=lambda node: node.part_index))
            generated_counts_by_class[class_id] = int(count)

        shape = tuple(int(v) for v in manifest.get("image_shape", [0, 0]))
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError(f"Sample bank has invalid image_shape: {manifest_path}")
        return cls(
            root=root,
            manifest_path=manifest_path,
            manifest=manifest,
            run_name=str(manifest.get("run_name", "")),
            split=str(manifest.get("split", "")),
            name=str(manifest.get("bank_name", root.name)),
            class_ids=tuple(sorted(generated_counts_by_class)),
            generated_counts_by_class=generated_counts_by_class,
            shards_by_class=shards_by_class,
            image_shape=(shape[0], shape[1]),
        )

    def available_n_per_class(self) -> int:
        return min(self.generated_counts_by_class.values()) if self.generated_counts_by_class else 0

    def load_bt_k(
        self,
        class_id: int,
        *,
        limit: int | None = None,
        mmap_mode: str | None = "r",
    ) -> np.ndarray:
        class_id = int(class_id)
        shards = self.shards_by_class.get(class_id)
        if not shards:
            raise KeyError(f"Class {class_id} is not present in sample bank {self.name!r}.")

        available = int(self.generated_counts_by_class[class_id])
        take_total = available if limit is None else int(limit)
        if take_total <= 0 or take_total > available:
            raise ValueError(f"Invalid sample-bank limit {take_total}; available={available}.")

        chunks = []
        remaining = take_total
        for shard in shards:
            if remaining <= 0:
                break
            arr = np.load(shard.bt_path, mmap_mode=mmap_mode)
            take = min(remaining, int(arr.shape[0]))
            chunks.append(arr[:take])
            remaining -= take
        if remaining:
            raise ValueError(
                f"Sample bank {self.name!r} is missing {remaining} rows for class {class_id}."
            )
        if len(chunks) == 1:
            return np.asarray(chunks[0], dtype=np.float32)
        return np.concatenate([np.asarray(chunk, dtype=np.float32) for chunk in chunks], axis=0)


def counts_by_label(labels: Iterable[int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for label in labels:
        key = int(label)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))
