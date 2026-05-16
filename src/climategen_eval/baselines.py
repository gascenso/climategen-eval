"""Bad-but-sensible reference baselines.

Each baseline produces a per-class array of synthetic "generated" BT images
that can be fed through the same metric pipeline as a diffusion model. The
goal is to anchor the standardized-gap scale on the "decidedly not good
enough" side using procedures with textbook interpretations and no tuneable
hyperparameters:

* CLIM (climatological mean): per-class pixel-wise mean replicated to fill
  the bank. The textbook no-skill forecast. Correct marginal mean, zero
  variance, no spatial structure beyond the class average. Saturates
  diversity- and variance-sensitive metrics.
* GAUSS-PIX (per-pixel Gaussian climatology): per-class pixel-wise mean plus
  iid Gaussian noise at the per-pixel sample standard deviation. The MLE
  under the assumption that pixels are independent Gaussians with
  matching first and second moments. Correct marginal moments by
  construction, wrong spatial dependence structure.
* CROSS-CLASS (wrong-class real images): for class k, sample real images
  uniformly from the OTHER classes. A model that ignored the intensity
  conditioning would look like this. Tests label fidelity at the metric
  level.
* PHASE-SHUFFLE (Theiler-style surrogate): real images of the right class
  with their 2-D Fourier phases randomized under a strict Hermitian
  symmetry constraint. Preserves the per-image radial PSD and pixel
  histogram almost exactly while destroying coherent spatial structure
  (eye, eyewall, asymmetry). The standard surrogate-data null from
  nonlinear-dynamics analysis (Theiler et al. 1992).

The baselines are intentionally complementary: each saturates a different
subset of metrics. Reviewers cannot dismiss the panel by attacking any one
choice — there is no tuneable knob to over- or under-strengthen them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


BASELINE_NAMES: tuple[str, ...] = (
    "CLIM",
    "GAUSS-PIX",
    "CROSS-CLASS",
    "PHASE-SHUFFLE",
)


BASELINE_DESCRIPTIONS: dict[str, str] = {
    "CLIM": (
        "Per-class pixel-wise mean image replicated to fill the bank. "
        "Correct marginal mean, zero variance, no spatial structure beyond the "
        "class average."
    ),
    "GAUSS-PIX": (
        "Per-class pixel-wise mean plus iid Gaussian noise at the per-pixel "
        "sample standard deviation. Correct first and second per-pixel moments, "
        "wrong spatial dependence structure."
    ),
    "CROSS-CLASS": (
        "For each target class, sample real images uniformly from the OTHER "
        "classes. Tests label/conditional fidelity."
    ),
    "PHASE-SHUFFLE": (
        "Real images with 2-D Fourier phases randomized under Hermitian "
        "symmetry, amplitudes preserved. Standard surrogate-data null: keeps "
        "radial PSD and pixel histogram, destroys coherent spatial structure."
    ),
}


@dataclass(frozen=True)
class BaselineResult:
    """One baseline's per-class outputs and provenance metadata."""

    name: str
    images_by_class: dict[int, np.ndarray]
    storm_ids_by_class: dict[int, np.ndarray] | None
    description: str
    provenance: dict[str, Any]


def _validate_real_by_class(real_by_class: dict[int, np.ndarray]) -> None:
    if not real_by_class:
        raise ValueError("real_by_class must contain at least one class.")
    shape: tuple[int, ...] | None = None
    for cid, arr in real_by_class.items():
        if arr.ndim != 3:
            raise ValueError(f"real_by_class[{cid}] must be (N,H,W), got {arr.shape}.")
        if arr.shape[0] <= 0:
            raise ValueError(f"real_by_class[{cid}] has no images.")
        if shape is None:
            shape = arr.shape[1:]
        elif shape != arr.shape[1:]:
            raise ValueError(
                f"Inconsistent image shape across classes: {arr.shape[1:]} vs {shape}."
            )


def build_climatology_baseline(
    real_by_class: dict[int, np.ndarray],
    *,
    n_per_class: int,
) -> BaselineResult:
    _validate_real_by_class(real_by_class)
    images: dict[int, np.ndarray] = {}
    for cid, real in real_by_class.items():
        mean_img = np.asarray(real, dtype=np.float32).mean(axis=0)
        images[cid] = np.broadcast_to(
            mean_img[None, ...],
            (int(n_per_class), *mean_img.shape),
        ).astype(np.float32, copy=True)
    return BaselineResult(
        name="CLIM",
        images_by_class=images,
        storm_ids_by_class=None,
        description=BASELINE_DESCRIPTIONS["CLIM"],
        provenance={
            "construction": "per_class_pixelwise_mean_replicated",
            "deterministic": True,
        },
    )


def build_gauss_pix_baseline(
    real_by_class: dict[int, np.ndarray],
    *,
    n_per_class: int,
    seed: int,
) -> BaselineResult:
    _validate_real_by_class(real_by_class)
    rng = np.random.default_rng(int(seed))
    images: dict[int, np.ndarray] = {}
    for cid in sorted(real_by_class):
        real = np.asarray(real_by_class[cid], dtype=np.float32)
        mean_img = real.mean(axis=0)
        # Unbiased per-pixel sample std; for ddof=1 we need n >= 2.
        std_img = real.std(axis=0, ddof=1 if real.shape[0] > 1 else 0).astype(np.float32)
        noise = rng.standard_normal(
            (int(n_per_class), *mean_img.shape),
        ).astype(np.float32)
        images[cid] = (mean_img[None, ...] + std_img[None, ...] * noise).astype(np.float32)
    return BaselineResult(
        name="GAUSS-PIX",
        images_by_class=images,
        storm_ids_by_class=None,
        description=BASELINE_DESCRIPTIONS["GAUSS-PIX"],
        provenance={
            "construction": "per_class_pixelwise_mean_plus_pixelwise_std_iid_gaussian",
            "moments_matched": ["per_pixel_mean", "per_pixel_std"],
            "deterministic": False,
            "seed": int(seed),
        },
    )


def build_cross_class_baseline(
    real_by_class: dict[int, np.ndarray],
    *,
    n_per_class: int,
    seed: int,
    storm_ids_by_class: dict[int, np.ndarray] | None = None,
) -> BaselineResult:
    _validate_real_by_class(real_by_class)
    if len(real_by_class) < 2:
        raise ValueError("CROSS-CLASS baseline requires at least two classes.")
    rng = np.random.default_rng(int(seed))
    images: dict[int, np.ndarray] = {}
    drawn_ids: dict[int, np.ndarray] | None = (
        {} if storm_ids_by_class is not None else None
    )
    classes = sorted(real_by_class)
    for cid in classes:
        others = [c for c in classes if c != cid]
        pool = np.concatenate(
            [np.asarray(real_by_class[c], dtype=np.float32) for c in others],
            axis=0,
        )
        idx = rng.integers(pool.shape[0], size=int(n_per_class))
        images[cid] = pool[idx].copy()
        if drawn_ids is not None:
            other_ids = np.concatenate(
                [np.asarray(storm_ids_by_class[c]).reshape(-1) for c in others],
                axis=0,
            )
            drawn_ids[cid] = other_ids[idx]
    return BaselineResult(
        name="CROSS-CLASS",
        images_by_class=images,
        storm_ids_by_class=drawn_ids,
        description=BASELINE_DESCRIPTIONS["CROSS-CLASS"],
        provenance={
            "construction": "for_each_class_draw_real_images_from_other_classes",
            "drawn_storm_ids_propagated": drawn_ids is not None,
            "deterministic": False,
            "seed": int(seed),
        },
    )


def _phase_randomize_2d(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-randomized surrogate of a single 2-D image.

    Amplitudes |F[k]| are preserved exactly; phases are replaced by a random
    field that is antisymmetric under k -> -k (mod N). The construction
    guarantees F_new is Hermitian symmetric so the inverse FFT is real to
    machine precision (we still discard the imaginary part to absorb
    floating-point noise).

    DC and Nyquist components have zero antisymmetric phase by construction,
    so the image mean is preserved exactly.
    """
    arr = np.asarray(x, dtype=np.float64)
    F = np.fft.fft2(arr)
    amp = np.abs(F)
    phi = rng.uniform(-np.pi, np.pi, size=arr.shape)
    # Antisymmetric part of phi under (k0,k1) -> (-k0,-k1) mod N
    phi_neg = np.roll(np.roll(phi[::-1, ::-1], shift=1, axis=0), shift=1, axis=1)
    phi_anti = 0.5 * (phi - phi_neg)
    F_new = amp * np.exp(1j * phi_anti)
    out = np.fft.ifft2(F_new).real
    return out.astype(np.float32, copy=False)


def build_phase_shuffle_baseline(
    real_by_class: dict[int, np.ndarray],
    *,
    n_per_class: int,
    seed: int,
    storm_ids_by_class: dict[int, np.ndarray] | None = None,
) -> BaselineResult:
    _validate_real_by_class(real_by_class)
    rng = np.random.default_rng(int(seed))
    images: dict[int, np.ndarray] = {}
    drawn_ids: dict[int, np.ndarray] | None = (
        {} if storm_ids_by_class is not None else None
    )
    for cid in sorted(real_by_class):
        real = np.asarray(real_by_class[cid], dtype=np.float32)
        idx = rng.integers(real.shape[0], size=int(n_per_class))
        sources = real[idx]
        out = np.empty_like(sources, dtype=np.float32)
        for i in range(out.shape[0]):
            out[i] = _phase_randomize_2d(sources[i], rng)
        images[cid] = out
        if drawn_ids is not None:
            drawn_ids[cid] = np.asarray(storm_ids_by_class[cid]).reshape(-1)[idx]
    return BaselineResult(
        name="PHASE-SHUFFLE",
        images_by_class=images,
        storm_ids_by_class=drawn_ids,
        description=BASELINE_DESCRIPTIONS["PHASE-SHUFFLE"],
        provenance={
            "construction": "per_class_real_image_fft_phase_randomization_hermitian_symmetric",
            "preserves": ["radial_psd_per_image", "pixel_histogram_per_image", "image_mean"],
            "drawn_storm_ids_propagated": drawn_ids is not None,
            "deterministic": False,
            "seed": int(seed),
        },
    )


BASELINE_BUILDERS = {
    "CLIM": build_climatology_baseline,
    "GAUSS-PIX": build_gauss_pix_baseline,
    "CROSS-CLASS": build_cross_class_baseline,
    "PHASE-SHUFFLE": build_phase_shuffle_baseline,
}


def build_baseline(
    name: str,
    real_by_class: dict[int, np.ndarray],
    *,
    n_per_class: int,
    seed: int,
    storm_ids_by_class: dict[int, np.ndarray] | None = None,
) -> BaselineResult:
    key = str(name).upper()
    if key not in BASELINE_BUILDERS:
        raise ValueError(
            f"Unknown baseline {name!r}; expected one of {list(BASELINE_BUILDERS)}"
        )
    if key == "CLIM":
        return build_climatology_baseline(real_by_class, n_per_class=int(n_per_class))
    if key == "GAUSS-PIX":
        return build_gauss_pix_baseline(
            real_by_class, n_per_class=int(n_per_class), seed=int(seed),
        )
    if key == "CROSS-CLASS":
        return build_cross_class_baseline(
            real_by_class,
            n_per_class=int(n_per_class),
            seed=int(seed),
            storm_ids_by_class=storm_ids_by_class,
        )
    if key == "PHASE-SHUFFLE":
        return build_phase_shuffle_baseline(
            real_by_class,
            n_per_class=int(n_per_class),
            seed=int(seed),
            storm_ids_by_class=storm_ids_by_class,
        )
    raise AssertionError(f"Unreachable baseline dispatch for {key!r}.")
