#!/usr/bin/env python3
"""Generate the deterministic, fully synthetic TC-BT smoke-test fixture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def _fields(rng: np.random.Generator, n: int, shift: float = 0.0) -> np.ndarray:
    y, x = np.mgrid[-1:1:32j, -1:1:32j]
    radius = np.sqrt(x * x + y * y)
    base = 282.0 - 62.0 * np.exp(-((radius - 0.42) ** 2) / 0.035)
    fields = np.stack(
        [base + shift + rng.normal(0.0, 2.5, size=base.shape) for _ in range(n)]
    )
    return np.clip(fields, 170.0, 325.0).astype(np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    rng = np.random.default_rng(1789)
    arrays = {
        "train_fields.npy": _fields(rng, 48),
        "train_blocks.npy": np.repeat([f"train_{i:02d}" for i in range(24)], 2),
        "train_labels.npy": np.tile(np.arange(6, dtype=np.int16), 8),
        "real_fields.npy": _fields(rng, 48),
        "real_blocks.npy": np.repeat([f"test_{i:02d}" for i in range(24)], 2),
        "real_labels.npy": np.tile(np.arange(6, dtype=np.int16), 8),
        "generated_fields.npy": _fields(rng, 30, shift=0.5),
        "generated_labels.npy": np.tile(np.arange(6, dtype=np.int16), 5),
    }
    for name, values in arrays.items():
        np.save(ROOT / name, values)
    manifest = {
        "schema": "gauge_climate.synthetic_tc_bt_fixture.v1",
        "generator_seed": 1789,
        "scientific_use": "none; software smoke tests only",
        "license": "BSD-3-Clause",
        "files": {
            name: {"sha256": _sha256(ROOT / name), "shape": list(values.shape)}
            for name, values in arrays.items()
        },
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
