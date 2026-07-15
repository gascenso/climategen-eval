"""Lazy loaders for the TC case-study real archives and generated sample banks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from ..data import Dataset


Target = Literal["category", "wind"]


def wind_to_category(wind_kt: np.ndarray | Sequence[float]) -> np.ndarray:
    """Map USA one-minute sustained wind (kt) to TS/Categories 1--5."""

    wind = np.asarray(wind_kt, dtype=np.float32)
    if np.any(~np.isfinite(wind)) or np.any(wind < 35.0):
        raise ValueError(
            "TC wind labels must be finite USA_WIND values of at least 35 kt."
        )
    return np.digitize(wind, [64.0, 83.0, 96.0, 113.0, 137.0]).astype(
        np.int16
    )


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


@dataclass(frozen=True)
class _Shard:
    path: Path
    start: int
    stop: int


class SampleBankFields:
    """Sample-axis array view over class-partitioned NumPy shards."""

    def __init__(self, shards: Sequence[Path]):
        arrays = tuple(np.load(path, mmap_mode="r") for path in shards)
        if not arrays:
            raise ValueError("A generated sample bank must contain at least one BT shard.")
        sample_shape = tuple(int(value) for value in arrays[0].shape[1:])
        if any(
            tuple(int(value) for value in array.shape[1:]) != sample_shape
            for array in arrays
        ):
            raise ValueError("Generated BT shards have inconsistent sample shapes.")
        if any(not np.issubdtype(array.dtype, np.number) for array in arrays):
            raise TypeError("Generated BT shards must be numeric.")
        offsets = np.cumsum([0, *(int(array.shape[0]) for array in arrays)], dtype=np.int64)
        self._arrays = arrays
        self._offsets = offsets
        self.shape = (int(offsets[-1]), *sample_shape)
        self.dtype = np.result_type(*(array.dtype for array in arrays))

    def __len__(self) -> int:
        return int(self.shape[0])

    def __getitem__(self, key: Any) -> np.ndarray:
        sample_key, trailing = (key[0], key[1:]) if isinstance(key, tuple) else (key, ())
        selected = np.arange(len(self), dtype=np.int64)[sample_key]
        if np.ndim(selected) == 0:
            index = int(selected)
            shard_index = int(np.searchsorted(self._offsets[1:], index, side="right"))
            value = self._arrays[shard_index][index - int(self._offsets[shard_index])]
            return value[trailing] if trailing else value
        selected = np.asarray(selected, dtype=np.int64).reshape(-1)
        output = np.empty((selected.size, *self.shape[1:]), dtype=self.dtype)
        for shard_index, array in enumerate(self._arrays):
            start = int(self._offsets[shard_index])
            stop = int(self._offsets[shard_index + 1])
            mask = (selected >= start) & (selected < stop)
            if np.any(mask):
                output[mask] = array[selected[mask] - start]
        return output[(slice(None), *trailing)] if trailing else output

    def __array__(
        self, dtype: np.dtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        array = np.concatenate([np.asarray(value) for value in self._arrays], axis=0)
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array.copy() if copy else array


def load_real_archive(manifest_path: str | Path, *, target: Target = "category") -> Dataset:
    """Load a corrected USA_WIND real split without materializing its fields."""

    if target not in {"category", "wind"}:
        raise ValueError("TC archive target must be 'category' or 'wind'.")
    if target not in {"category", "wind"}:
        raise ValueError("target must be 'category' or 'wind'.")
    manifest_path = Path(manifest_path)
    manifest = _json(manifest_path)
    required = {
        "schema": "gauge_climate.tc_bt_real_dataset.v2",
        "wind_source": "USA_WIND",
        "wind_averaging_period": "1-minute",
    }
    bad = {
        key: manifest.get(key)
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    if bad:
        raise ValueError(
            "Refusing TC archive without the USA_WIND/1-minute contract "
            f"{manifest_path}: {bad}."
        )
    root = manifest_path.parent
    fields = np.load(root / str(manifest["bt_k"]), mmap_mode="r")
    categories = np.load(root / str(manifest["labels"]), mmap_mode="r")
    winds = np.load(root / str(manifest["wind_kt"]), mmap_mode="r")
    storms = np.load(root / str(manifest["storm_ids"]), mmap_mode="r")
    rel_paths = np.asarray([
        line.strip()
        for line in (root / str(manifest["rel_paths"])).read_text().splitlines()
        if line.strip()
    ])
    expected_n = int(manifest.get("num_samples", fields.shape[0]))
    if any(
        len(value) != expected_n
        for value in (fields, categories, winds, storms, rel_paths)
    ):
        raise ValueError(
            f"TC archive arrays do not agree with num_samples in {manifest_path}."
        )
    if not np.array_equal(
        wind_to_category(winds), np.asarray(categories, dtype=np.int16)
    ):
        raise ValueError(
            f"TC archive categories are inconsistent with USA_WIND in {manifest_path}."
        )
    labels = categories if target == "category" else winds
    return Dataset(
        fields=fields,
        blocks=storms,
        labels=labels,
        sample_ids=rel_paths,
        name=f"tc_bt_real_{manifest['split']}_{target}",
        units="K",
        metadata={
            **manifest,
            "target": target,
            "manifest_path": str(manifest_path),
        },
    )


def load_sample_bank(
    bank_path: str | Path,
    *,
    target: Target = "category",
    required_run: str | None = "baseline_ztsnr",
) -> Dataset:
    """Load a ``tc_diffusion.sample_bank.v2`` directory as a lazy GAUGE dataset."""

    if target not in {"category", "wind"}:
        raise ValueError("TC sample-bank target must be 'category' or 'wind'.")
    if target not in {"category", "wind"}:
        raise ValueError("target must be 'category' or 'wind'.")
    root = Path(bank_path)
    manifest_path = root / "manifest.json"
    manifest = _json(manifest_path)
    if manifest.get("schema") != "tc_diffusion.sample_bank.v2":
        raise ValueError(f"Unsupported generated sample-bank schema: {manifest_path}.")
    if required_run is not None and manifest.get("run_name") != required_run:
        raise ValueError(
            f"Expected sample bank from {required_run!r}, got {manifest.get('run_name')!r}."
        )
    shard_paths: list[Path] = []
    labels: list[np.ndarray] = []
    sample_ids: list[str] = []
    for raw_class in sorted(manifest.get("shards_by_class", {}), key=int):
        class_id = int(raw_class)
        for raw_shard in manifest["shards_by_class"][raw_class]:
            count = int(raw_shard["count"])
            bt_path = root / str(raw_shard["bt_k"])
            shard_paths.append(bt_path)
            wind_path = raw_shard.get("wind_kt")
            if wind_path is None:
                raise ValueError(
                    f"Generated shard {bt_path} has no continuous wind targets."
                )
            wind_values = np.asarray(
                np.load(root / str(wind_path), mmap_mode="r")
            )
            if wind_values.shape != (count,):
                raise ValueError(f"Generated wind targets do not align for {bt_path}.")
            if not np.all(wind_to_category(wind_values) == class_id):
                raise ValueError(
                    f"Generated wind targets are inconsistent with class {class_id} "
                    f"for {bt_path}."
                )
            if target == "category":
                shard_labels = np.full(count, class_id, dtype=np.int16)
            else:
                shard_labels = wind_values
            labels.append(shard_labels)
            part = int(raw_shard.get("part_index", 0))
            sample_ids.extend(
                f"{manifest['bank_name']}/class_{class_id}/part_{part}/{index}"
                for index in range(count)
            )
    fields = SampleBankFields(shard_paths)
    aligned_labels = np.concatenate(labels)
    if len(aligned_labels) != len(fields):
        raise ValueError(f"Generated labels and fields do not align in {root}.")
    return Dataset(
        fields=fields,
        labels=aligned_labels,
        sample_ids=np.asarray(sample_ids),
        name=f"tc_bt_{manifest['bank_name']}_{target}",
        units="K",
        metadata={**manifest, "target": target, "manifest_path": str(manifest_path)},
    )
