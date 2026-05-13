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


def build_shared_pixel_shuffle_permutation(*, image_shape: tuple[int, int], seed: int) -> np.ndarray:
    h, w = int(image_shape[0]), int(image_shape[1])
    rng = np.random.default_rng(int(seed))
    return np.asarray(rng.permutation(h * w), dtype=np.int64)


def apply_shared_pixel_shuffle_bt_k(bt_k: np.ndarray, *, permutation: np.ndarray) -> np.ndarray:
    arr = ensure_bt_batch(bt_k)
    flat = arr.reshape(arr.shape[0], -1)
    perm = np.asarray(permutation, dtype=np.int64).reshape(-1)
    if flat.shape[1] != perm.shape[0]:
        raise ValueError("Pixel-shuffle permutation length does not match image size.")
    return np.asarray(flat[:, perm].reshape(arr.shape), dtype=np.float32)


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


def apply_mode_collapse_bt_k(bt_k: np.ndarray, *, seed: int) -> np.ndarray:
    arr = ensure_bt_batch(bt_k)
    if arr.shape[0] <= 0:
        raise ValueError("Cannot mode-collapse an empty image set.")
    rng = np.random.default_rng(int(seed))
    idx = int(rng.integers(arr.shape[0]))
    return np.repeat(arr[idx : idx + 1], arr.shape[0], axis=0).astype(np.float32, copy=False)


def apply_mode_collapse_fraction_bt_k(bt_k: np.ndarray, *, fraction: float, seed: int) -> np.ndarray:
    arr = ensure_bt_batch(bt_k)
    frac = float(np.clip(fraction, 0.0, 1.0))
    if frac <= 0.0:
        return arr.copy()
    rng = np.random.default_rng(int(seed))
    out = arr.copy()
    n_replace = min(max(int(round(frac * arr.shape[0])), 1), arr.shape[0])
    source_idx = int(rng.integers(arr.shape[0]))
    replace_idx = np.asarray(rng.choice(arr.shape[0], size=n_replace, replace=False), dtype=np.int64)
    out[replace_idx] = arr[source_idx]
    return out


def ensure_bt_batch(bt_k: np.ndarray) -> np.ndarray:
    arr = np.asarray(bt_k, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected BT array shaped [N,H,W] or [H,W], got {arr.shape}.")
    return arr


def build_fixed_control_bt_k(
    control_name: str,
    source_k: np.ndarray,
    *,
    seed: int,
    gaussian_sigma_px: float = 6.0,
    bt_warm_shift_k: float = 15.0,
    center_jitter_max_px: int = 24,
) -> tuple[np.ndarray, dict[str, Any]]:
    name = str(control_name)
    if name == "gaussian_blur":
        return apply_gaussian_blur_bt_k(source_k, sigma_px=gaussian_sigma_px), {
            "name": name,
            "sigma_px": float(gaussian_sigma_px),
        }
    if name == "pixel_shuffle":
        arr = ensure_bt_batch(source_k)
        permutation = build_shared_pixel_shuffle_permutation(
            image_shape=(int(arr.shape[1]), int(arr.shape[2])),
            seed=int(seed),
        )
        return apply_shared_pixel_shuffle_bt_k(arr, permutation=permutation), {
            "name": name,
            "permutation": "shared_flattened_pixel_permutation",
            "seed": int(seed),
        }
    if name == "bt_warm_shift":
        return apply_bt_warm_shift_bt_k(source_k, shift_k=bt_warm_shift_k), {
            "name": name,
            "shift_k": float(bt_warm_shift_k),
        }
    if name == "center_jitter":
        return apply_center_jitter_bt_k(source_k, max_shift_px=center_jitter_max_px, seed=seed), {
            "name": name,
            "max_shift_px": int(center_jitter_max_px),
            "seed": int(seed),
            "padding": "edge",
        }
    if name == "mode_collapse":
        return apply_mode_collapse_bt_k(source_k, seed=seed), {
            "name": name,
            "selection": "one sampled same-class real image repeated to generated count",
            "seed": int(seed),
        }
    raise ValueError(f"Unsupported control: {control_name!r}")


def build_calibrated_control_bt_k(
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
    raise ValueError(f"Unsupported calibrated control: {control_name!r}")

