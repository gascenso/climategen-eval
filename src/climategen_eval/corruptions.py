from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter


CONTROL_NAMES = (
    "gaussian_blur",
    "pixel_shuffle",
    "bt_warm_shift",
    "center_jitter",
    "mode_collapse",
)


def apply_gaussian_blur_bt_k(bt_k: np.ndarray, *, sigma_px: float) -> np.ndarray:
    arr = ensure_bt_batch(bt_k)
    sigma = float(sigma_px)
    if sigma <= 0.0:
        return arr.copy()
    out = np.empty_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        out[i] = gaussian_filter(arr[i], sigma=sigma, mode="reflect")
    return out


def apply_partial_pixel_shuffle_bt_k(bt_k: np.ndarray, *, fraction: float, seed: int) -> np.ndarray:
    arr = ensure_bt_batch(bt_k)
    frac = float(np.clip(fraction, 0.0, 1.0))
    if frac <= 0.0:
        return arr.copy()
    flat = arr.reshape(arr.shape[0], -1)
    n_pixels = int(flat.shape[1])
    n_shuffle = int(round(frac * n_pixels))
    if n_shuffle < 2:
        return arr.copy()
    rng = np.random.default_rng(int(seed))
    idx = np.asarray(rng.choice(n_pixels, size=n_shuffle, replace=False), dtype=np.int64)
    permuted_idx = idx[np.asarray(rng.permutation(n_shuffle), dtype=np.int64)]
    out = flat.copy()
    out[:, idx] = flat[:, permuted_idx]
    return np.asarray(out.reshape(arr.shape), dtype=np.float32)


def apply_bt_warm_shift_bt_k(bt_k: np.ndarray, *, shift_k: float) -> np.ndarray:
    return np.asarray(ensure_bt_batch(bt_k) + float(shift_k), dtype=np.float32)


def apply_center_jitter_bt_k(bt_k: np.ndarray, *, max_shift_px: int, seed: int) -> np.ndarray:
    arr = ensure_bt_batch(bt_k)
    max_shift = int(max(0, max_shift_px))
    if max_shift == 0:
        return arr.copy()
    rng = np.random.default_rng(int(seed))
    out = np.empty_like(arr, dtype=np.float32)
    h, w = int(arr.shape[1]), int(arr.shape[2])
    for i in range(arr.shape[0]):
        dy = int(rng.integers(-max_shift, max_shift + 1))
        dx = int(rng.integers(-max_shift, max_shift + 1))
        if dy == 0 and dx == 0:
            dy = max_shift
        padded = np.pad(arr[i], ((max_shift, max_shift), (max_shift, max_shift)), mode="edge")
        y0 = max_shift - dy
        x0 = max_shift - dx
        out[i] = padded[y0 : y0 + h, x0 : x0 + w]
    return out


def apply_mode_collapse_fraction_bt_k(
    bt_k: np.ndarray,
    *,
    fraction: float,
    seed: int,
) -> np.ndarray:
    arr = ensure_bt_batch(bt_k)
    frac = float(np.clip(fraction, 0.0, 1.0))
    if frac <= 0.0:
        return arr.copy()
    rng = np.random.default_rng(int(seed))
    out = arr.copy()
    n_replace = min(max(int(round(frac * arr.shape[0])), 1), arr.shape[0])
    source_idx = int(rng.integers(arr.shape[0]))
    replace_idx = np.asarray(
        rng.choice(arr.shape[0], size=n_replace, replace=False),
        dtype=np.int64,
    )
    out[replace_idx] = arr[source_idx]
    return out


def ensure_bt_batch(bt_k: np.ndarray) -> np.ndarray:
    arr = np.asarray(bt_k, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected BT array shaped [N,H,W] or [H,W], got {arr.shape}.")
    return arr


def build_validation_control_bt_k(
    control_name: str,
    source_k: np.ndarray,
    *,
    severity: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    name = str(control_name)
    value = float(severity)
    if value <= 0.0:
        return ensure_bt_batch(source_k).copy(), {"name": name, "severity": 0.0, "identity": True}
    if name == "gaussian_blur":
        return apply_gaussian_blur_bt_k(source_k, sigma_px=value), {
            "name": name,
            "parameter": "gaussian_sigma_px",
            "severity": value,
        }
    if name == "bt_warm_shift":
        return apply_bt_warm_shift_bt_k(source_k, shift_k=value), {
            "name": name,
            "parameter": "warm_shift_k",
            "severity": value,
        }
    if name == "center_jitter":
        max_shift = int(round(value))
        return apply_center_jitter_bt_k(source_k, max_shift_px=max_shift, seed=seed), {
            "name": name,
            "parameter": "max_shift_px",
            "severity": float(max_shift),
            "seed": int(seed),
        }
    if name == "pixel_shuffle":
        return apply_partial_pixel_shuffle_bt_k(source_k, fraction=value, seed=seed), {
            "name": name,
            "parameter": "pixel_fraction_shuffled",
            "severity": value,
            "seed": int(seed),
        }
    if name == "mode_collapse":
        return apply_mode_collapse_fraction_bt_k(source_k, fraction=value, seed=seed), {
            "name": name,
            "parameter": "sample_fraction_replaced_by_one_image",
            "severity": value,
            "seed": int(seed),
        }
    raise ValueError(f"Unsupported validation control: {control_name!r}")
