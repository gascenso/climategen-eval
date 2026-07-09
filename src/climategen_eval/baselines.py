"""Targeted-null reference baselines.

Each baseline produces a per-class array of synthetic "generated" BT images
that can be fed through the same metric pipeline as a diffusion model. The
panel is a *spanning set of targeted nulls*: each preserves a specific
sufficient statistic of the class-conditional distribution and nothing
beyond it, so the gap of a given baseline on a given metric is informative
about *what that metric measures*. The panel is intentionally constructed
to have no tuneable knobs aligned with the metrics being validated.

* CLIM (climatological mean): per-class pixel-wise mean replicated to fill
  the bank. Preserves the first pixel moment only. Degenerate case of
  GAUSS-FULL with covariance set to zero.
* GAUSS-PIX (per-pixel Gaussian climatology): per-class pixel-wise mean plus
  iid Gaussian noise at the per-pixel sample standard deviation. Preserves
  the first moment and the diagonal of the spatial covariance. Degenerate
  case of GAUSS-FULL with off-diagonal covariance set to zero.
* GAUSS-FULL (maximum-entropy Gaussian): per-class draws from N(mu, Sigma)
  with Sigma the empirical sample covariance of the class. Preserves first
  and second moments exactly, including the full spatial covariance and
  hence the average radial PSD. Fails on higher-than-Gaussian structure:
  the pixel histogram is Gaussian (real BT marginals are not), gradient-
  angle statistics like DAV degrade, and per-image nonlinear features
  (eye, eyewall) are absent. This is the textbook information-theoretic
  null for "everything learnable from the first two moments." Sampled via
  the linear-combination identity x = mu + R^T z / sqrt(N - 1), z ~ N(0, I),
  so the (H*W)^2 covariance is never materialized.
* CROSS-CLASS (wrong-class real images): for class k, sample real images
  uniformly from the OTHER classes. A model that ignored the intensity
  conditioning would look like this. Tests label fidelity at the metric
  level — orthogonal to any unconditional null.
* PHASE-SHUFFLE (Theiler-style surrogate): real images of the right class
  with their 2-D Fourier phases randomized under a strict Hermitian
  symmetry constraint. Preserves the per-image power spectrum EXACTLY —
  the full 2-D periodogram and hence every function of it (radial average,
  azimuthal, slope, band-power) to machine precision — and the pixel mean
  and variance, but NOT the higher-order marginal: phase randomization
  Gaussianizes the one-point distribution (CLT), so skew/kurtosis and the
  pixel histogram are destroyed along with coherent spatial structure
  (eye, eyewall, asymmetry). The standard surrogate-data null from
  nonlinear-dynamics analysis (Theiler et al. 1992). Per-image, not
  ensemble: orthogonal to GAUSS-FULL, which preserves ensemble covariance
  but loses per-image identity.

GAUSS-FULL strictly dominates CLIM and GAUSS-PIX (both are degenerate cases
of it); the latter two are retained as ablations that strip progressively
more covariance information out of the Gaussian null.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


BASELINE_NAMES: tuple[str, ...] = (
    "CLIM",
    "GAUSS-PIX",
    "GAUSS-FULL",
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
    "GAUSS-FULL": (
        "Per-class maximum-entropy Gaussian with empirical mean and full "
        "spatial covariance. Preserves first and second moments exactly, "
        "including the average radial PSD; fails on higher-than-Gaussian "
        "structure (DAV, eye, non-Gaussian pixel tails). Sampled via the "
        "linear-combination identity x = mu + R^T z / sqrt(N - 1), so the "
        "(H*W)^2 covariance is never materialized."
    ),
    "CROSS-CLASS": (
        "For each target class, sample real images uniformly from the OTHER "
        "classes. Tests label/conditional fidelity."
    ),
    "PHASE-SHUFFLE": (
        "Real images with 2-D Fourier phases randomized under Hermitian "
        "symmetry, amplitudes preserved. Standard surrogate-data null: keeps "
        "the full power spectrum (2-D/radial) exactly + pixel mean/variance, "
        "Gaussianizes the marginal (destroys the pixel histogram) and all "
        "coherent spatial structure."
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


def build_gauss_full_baseline(
    real_by_class: dict[int, np.ndarray],
    *,
    n_per_class: int,
    seed: int,
) -> BaselineResult:
    """Class-conditional maximum-entropy Gaussian with empirical covariance.

    Each generated sample is drawn from N(mu_class, Sigma_class) where
    Sigma_class is the empirical sample covariance of the class's real images.
    The covariance is never materialized (it would be (H*W)x(H*W)): instead,
    samples use the linear-combination identity

        x = mu + R^T z / sqrt(N - 1),    z ~ N(0, I_N)

    where R is the (N, H*W) matrix of mean-centered class residuals. A direct
    computation gives Cov(x) = R^T R / (N - 1), which is exactly the unbiased
    sample covariance. Each sample lies in the affine hull of the real class,
    which is the correct interpretation of "the best linear-Gaussian model of
    the data" — no more, no less. Per-sample cost is O(N * H * W) and the
    effective rank of the implied covariance is at most N - 1.

    Skill profile across the BT metrics: first and second moments are exact,
    so the radial profile MAE, the average radial PSD MSE, and per-pixel
    Gaussian marginals are all near the noise floor. Higher-order structure
    is mis-modeled, so DAV (a nonlinear gradient-angle functional) and any
    per-image coherent structure (eye, eyewall) collapse. This is the
    textbook information-theoretic null for "everything that can be learned
    from the first two moments of the field."
    """
    _validate_real_by_class(real_by_class)
    rng = np.random.default_rng(int(seed))
    images: dict[int, np.ndarray] = {}
    for cid in sorted(real_by_class):
        real = np.asarray(real_by_class[cid], dtype=np.float32)
        n = int(real.shape[0])
        if n < 2:
            raise ValueError(
                "GAUSS-FULL requires at least 2 real samples per class to "
                f"estimate a sample covariance; class {cid} has {n}."
            )
        h, w = int(real.shape[1]), int(real.shape[2])
        mean_img = real.mean(axis=0)
        # Centered residuals, (N, H*W). Promoted to float64 for the matmul
        # to keep the linear combination numerically stable when N is large.
        residuals = (real - mean_img).reshape(n, -1).astype(np.float64)
        scale = 1.0 / np.sqrt(float(n - 1))
        z = rng.standard_normal((int(n_per_class), n))
        deltas = (z @ residuals) * scale
        out = mean_img.reshape(1, -1).astype(np.float64) + deltas
        images[cid] = out.reshape(int(n_per_class), h, w).astype(np.float32)
    return BaselineResult(
        name="GAUSS-FULL",
        images_by_class=images,
        storm_ids_by_class=None,
        description=BASELINE_DESCRIPTIONS["GAUSS-FULL"],
        provenance={
            "construction": (
                "per_class_max_entropy_gaussian_with_empirical_covariance_"
                "sampled_via_linear_combination_of_centered_residuals"
            ),
            "moments_matched": ["per_pixel_mean", "full_pixel_covariance"],
            "rank_bound": "at_most_N_minus_1_per_class",
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
            "preserves": [
                "power_spectrum_2d_per_image",  # exact: full 2-D periodogram and every function of it (radial, azimuthal, slope, band-power)
                "radial_psd_per_image",
                "image_mean",
                "image_variance",
            ],
            "does_not_preserve": ["pixel_histogram_shape", "skewness", "kurtosis", "spatial_phase_structure"],
            "drawn_storm_ids_propagated": drawn_ids is not None,
            "deterministic": False,
            "seed": int(seed),
        },
    )


BASELINE_BUILDERS = {
    "CLIM": build_climatology_baseline,
    "GAUSS-PIX": build_gauss_pix_baseline,
    "GAUSS-FULL": build_gauss_full_baseline,
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
    if key == "GAUSS-FULL":
        return build_gauss_full_baseline(
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
