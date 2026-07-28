from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import FunctionalSpec, MetricKind, MetricSpec


@dataclass(frozen=True)
class TCBTConfig:
    field_shape: tuple[int, int] | None = None
    bt_min_k: float = 170.0
    bt_max_k: float = 325.0
    clip_to_analysis_range: bool = True
    pixel_size_km: float = 8.0
    core_radius_km: float = 100.0
    organization_radius_km: float = 300.0
    max_profile_radius_km: float = 500.0
    cold_threshold_k: float = 220.0
    profile_bins: int = 32
    psd_bins: int = 24
    eye_radius_km: float = 50.0
    eyewall_inner_km: float = 50.0
    eyewall_outer_km: float = 150.0
    dav_center_region_size: int = 3

    def __post_init__(self) -> None:
        if self.field_shape is not None:
            shape = tuple(int(value) for value in self.field_shape)
            if len(shape) != 2 or min(shape) < 4:
                raise ValueError("field_shape must contain two dimensions of at least four.")
            object.__setattr__(self, "field_shape", shape)
        if self.bt_min_k >= self.bt_max_k:
            raise ValueError("bt_min_k must be below bt_max_k.")
        if self.pixel_size_km <= 0:
            raise ValueError("pixel_size_km must be positive.")
        if self.profile_bins < 4 or self.psd_bins < 4:
            raise ValueError("profile_bins and psd_bins must be at least four.")
        if not self.eye_radius_km <= self.eyewall_inner_km < self.eyewall_outer_km:
            raise ValueError("Eye and eyewall radii are inconsistent.")
        if self.dav_center_region_size < 1 or self.dav_center_region_size % 2 != 1:
            raise ValueError("dav_center_region_size must be a positive odd integer.")


def _validate(fields: np.ndarray, config: TCBTConfig | None = None) -> np.ndarray:
    fields = np.asarray(fields, dtype=np.float64)
    if fields.ndim != 3:
        raise ValueError(f"TC-BT adapter expects (N,H,W), got {fields.shape}.")
    if not np.all(np.isfinite(fields)):
        raise ValueError("TC-BT fields contain non-finite values.")
    if config is not None and config.field_shape is not None:
        if tuple(fields.shape[-2:]) != config.field_shape:
            raise ValueError(
                f"TC-BT fields have shape {fields.shape[-2:]}, expected {config.field_shape}."
            )
    if config is not None and config.clip_to_analysis_range:
        fields = np.clip(fields, config.bt_min_k, config.bt_max_k)
    return fields


class PreprocessedFields:
    """Lazy TC analysis view with the generator-training BT clipping contract."""

    def __init__(self, fields: object, config: TCBTConfig):
        shape = getattr(fields, "shape", None)
        if shape is None or len(shape) != 3:
            raise ValueError("TC preprocessing expects a field source shaped (N,H,W).")
        if config.field_shape is not None and tuple(shape[1:]) != config.field_shape:
            raise ValueError("TC preprocessing source shape differs from adapter config.")
        self._fields = fields
        self._config = config
        self.shape = tuple(int(value) for value in shape)
        self.dtype = np.dtype(getattr(fields, "dtype", np.float32))

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key: object) -> np.ndarray:
        values = np.asarray(self._fields[key])
        if not self._config.clip_to_analysis_range:
            return values
        return np.clip(values, self._config.bt_min_k, self._config.bt_max_k).astype(
            values.dtype, copy=False
        )

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        values = np.asarray(self._fields)
        if self._config.clip_to_analysis_range:
            values = np.clip(values, self._config.bt_min_k, self._config.bt_max_k)
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        return values.copy() if copy else values


def _radius(
    shape: tuple[int, int], pixel_size_km: float, *, center_shift=(0, 0)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    y, x = np.mgrid[0:h, 0:w]
    cy = (h - 1) / 2.0 + float(center_shift[0])
    cx = (w - 1) / 2.0 + float(center_shift[1])
    dy = y - cy
    dx = x - cx
    return np.sqrt(dx * dx + dy * dy) * pixel_size_km, dx, dy


def radial_profiles(fields: np.ndarray, config: TCBTConfig = TCBTConfig()) -> np.ndarray:
    fields = _validate(fields, config)
    radius, _, _ = _radius(fields.shape[1:], config.pixel_size_km)
    edges = np.linspace(0.0, config.max_profile_radius_km, config.profile_bins + 1)
    bin_index = np.digitize(radius.reshape(-1), edges, right=False) - 1
    output = np.empty((fields.shape[0], config.profile_bins), dtype=np.float64)
    flat = fields.reshape(fields.shape[0], -1)
    for radial_bin in range(config.profile_bins):
        mask = bin_index == radial_bin
        if not np.any(mask):
            raise ValueError(f"Radial bin {radial_bin} contains no pixels.")
        output[:, radial_bin] = flat[:, mask].mean(axis=1)
    return output


def _psd_geometry(shape: tuple[int, int], config: TCBTConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return nonempty radial bins near log-spaced physical targets.

    A naïve geometric sequence can place boundaries between sparse low-frequency
    Fourier shells and create empty bins. We therefore choose distinct observed
    radial wavenumbers nearest the log-spaced targets, then put boundaries at
    geometric midpoints. Every registered bin contains at least one discrete
    Fourier mode while retaining the intended logarithmic ordering.
    """

    h, w = shape
    ky = np.fft.fftfreq(h, d=config.pixel_size_km)
    kx = np.fft.fftfreq(w, d=config.pixel_size_km)
    radial_k = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    k_max = min(0.5 / config.pixel_size_km, float(radial_k.max()))
    support = np.unique(radial_k[(radial_k > 0) & (radial_k <= k_max)])
    if support.size < config.psd_bins:
        raise ValueError(
            f"Only {support.size} positive radial Fourier shells are available "
            f"for {config.psd_bins} PSD bins."
        )
    targets = np.geomspace(float(support[0]), float(support[-1]), config.psd_bins)
    selected = []
    previous = -1
    for index, target in enumerate(targets):
        lower = previous + 1
        upper = support.size - (config.psd_bins - index)
        candidates = support[lower : upper + 1]
        local = int(np.argmin(np.abs(np.log(candidates) - np.log(target))))
        chosen = lower + local
        selected.append(chosen)
        previous = chosen
    centers = support[np.asarray(selected, dtype=np.int64)]
    edges = np.empty(config.psd_bins + 1, dtype=np.float64)
    edges[0] = np.nextafter(centers[0], 0.0)
    edges[-1] = np.nextafter(centers[-1], np.inf)
    edges[1:-1] = np.sqrt(centers[:-1] * centers[1:])
    return edges, centers


def _psd_edges(shape: tuple[int, int], config: TCBTConfig) -> np.ndarray:
    return _psd_geometry(shape, config)[0]


def radial_log_psd(fields: np.ndarray, config: TCBTConfig = TCBTConfig()) -> np.ndarray:
    """Hann-windowed, mean-removed, area-normalized radial log10 PSD.

    Wavenumbers are cycles/km. The zero mode is excluded and positive
    wavenumbers are aggregated into fixed, nonempty, discrete-support-aware bins
    nearest log-spaced targets from the fundamental domain frequency to the
    smaller-axis Nyquist frequency.
    """

    fields = _validate(fields, config)
    n, h, w = fields.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    window_energy = float(np.sum(window * window))
    if window_energy <= 0:
        raise ValueError("Invalid Hann window energy.")
    centered = fields - fields.mean(axis=(1, 2), keepdims=True)
    spectrum = np.fft.fft2(centered * window[None, :, :], axes=(-2, -1))
    power = np.abs(spectrum) ** 2 / window_energy
    ky = np.fft.fftfreq(h, d=config.pixel_size_km)
    kx = np.fft.fftfreq(w, d=config.pixel_size_km)
    radial_k = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    positive = radial_k > 0
    edges = _psd_edges((h, w), config)
    bin_index = np.digitize(radial_k.reshape(-1), edges, right=False) - 1
    valid_flat = positive.reshape(-1)
    output = np.empty((n, config.psd_bins), dtype=np.float64)
    flat_power = power.reshape(n, -1)
    for radial_bin in range(config.psd_bins):
        mask = valid_flat & (bin_index == radial_bin)
        if not np.any(mask):
            raise ValueError(f"PSD bin {radial_bin} contains no Fourier modes.")
        output[:, radial_bin] = np.log10(np.maximum(flat_power[:, mask].mean(axis=1), 1e-20))
    return output


def _dav_one(image: np.ndarray, config: TCBTConfig) -> float:
    h, w = image.shape
    p = np.pad(np.asarray(image, dtype=np.float64), 1, mode="reflect")
    gx = p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2] - p[:-2, 2:] - 2 * p[1:-1, 2:] - p[2:, 2:]
    gy = p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:] - p[2:, :-2] - 2 * p[2:, 1:-1] - p[2:, 2:]
    half = config.dav_center_region_size // 2
    values = []
    for y_shift in range(-half, half + 1):
        for x_shift in range(-half, half + 1):
            radius, dx, dy = _radius((h, w), config.pixel_size_km, center_shift=(y_shift, x_shift))
            radius_px = np.maximum(radius / config.pixel_size_km, 1e-12)
            radial_x = dx / radius_px
            radial_y = dy / radius_px
            grad_norm = np.sqrt(gx * gx + gy * gy + 1e-12)
            cosine = np.clip((gx * radial_x + gy * radial_y) / grad_norm, -1.0, 1.0)
            angle = np.degrees(np.arccos(cosine))
            mask = (radius > 0) & (radius <= config.organization_radius_km)
            values.append(float(np.var(angle[mask])))
    return float(np.mean(values))


def functionals(config: TCBTConfig = TCBTConfig()) -> tuple[FunctionalSpec, ...]:
    def min_core(fields: np.ndarray) -> np.ndarray:
        fields = _validate(fields, config)
        radius, _, _ = _radius(fields.shape[1:], config.pixel_size_km)
        return fields[:, radius < config.core_radius_km].min(axis=1)

    def cold_fraction(fields: np.ndarray) -> np.ndarray:
        fields = _validate(fields, config)
        radius, _, _ = _radius(fields.shape[1:], config.pixel_size_km)
        mask = radius <= config.organization_radius_km
        return 100.0 * np.mean(fields[:, mask] < config.cold_threshold_k, axis=1)

    def radius_max_gradient(fields: np.ndarray) -> np.ndarray:
        profiles = radial_profiles(fields, config)
        centers = (
            (np.arange(config.profile_bins, dtype=np.float64) + 0.5)
            * config.max_profile_radius_km
            / config.profile_bins
        )
        gradient = np.gradient(profiles, centers, axis=1)
        return centers[np.argmax(np.abs(gradient), axis=1)]

    def axisymmetry(fields: np.ndarray) -> np.ndarray:
        fields = _validate(fields, config)
        radius, _, _ = _radius(fields.shape[1:], config.pixel_size_km)
        edges = np.linspace(0.0, config.organization_radius_km, config.profile_bins + 1)
        bin_index = np.digitize(radius.reshape(-1), edges, right=False) - 1
        valid = (bin_index >= 0) & (bin_index < config.profile_bins)
        output = np.empty(fields.shape[0], dtype=np.float64)
        for sample, image in enumerate(fields.reshape(fields.shape[0], -1)):
            residuals = []
            for radial_bin in range(config.profile_bins):
                mask = valid & (bin_index == radial_bin)
                if np.any(mask):
                    residuals.append(image[mask] - image[mask].mean())
            output[sample] = np.sqrt(np.mean(np.concatenate(residuals) ** 2))
        return output

    def dav(fields: np.ndarray) -> np.ndarray:
        fields = _validate(fields, config)
        return np.asarray([_dav_one(image, config) for image in fields], dtype=np.float64)

    def eye_contrast(fields: np.ndarray) -> np.ndarray:
        fields = _validate(fields, config)
        radius, _, _ = _radius(fields.shape[1:], config.pixel_size_km)
        eye = fields[:, radius <= config.eye_radius_km].mean(axis=1)
        eyewall = fields[
            :, (radius > config.eyewall_inner_km) & (radius <= config.eyewall_outer_km)
        ].mean(axis=1)
        return eye - eyewall

    def p01(fields: np.ndarray) -> np.ndarray:
        fields = _validate(fields, config)
        return np.quantile(fields.reshape(fields.shape[0], -1), 0.01, axis=1)

    shared = {
        "pixel_size_km": config.pixel_size_km,
        "adapter": "gauge_climate.adapters.tc_bt.v1",
    }
    return (
        FunctionalSpec(
            "min_core_bt",
            min_core,
            "K",
            f"minimum brightness temperature within {config.core_radius_km:g} km",
            support=(config.bt_min_k, config.bt_max_k),
            direction_negative="colder core",
            direction_positive="warmer core",
            provenance=shared,
        ),
        FunctionalSpec(
            "cold_frac_220",
            cold_fraction,
            "%",
            f"area fraction below {config.cold_threshold_k:g} K within {config.organization_radius_km:g} km",
            support=(0.0, 100.0),
            direction_negative="smaller cold-cloud area",
            direction_positive="larger cold-cloud area",
            provenance=shared,
        ),
        FunctionalSpec(
            "r_maxgrad",
            radius_max_gradient,
            "km",
            "radius of the largest absolute azimuthal-mean BT gradient",
            support=(0.0, config.max_profile_radius_km),
            direction_negative="gradient radius inward",
            direction_positive="gradient radius outward",
            provenance={
                **shared,
                "discrete_support_step_km": (config.max_profile_radius_km / config.profile_bins),
            },
        ),
        FunctionalSpec(
            "axisym",
            axisymmetry,
            "K",
            f"RMS azimuthal departure within {config.organization_radius_km:g} km",
            support=(0.0, config.bt_max_k - config.bt_min_k),
            direction_negative="more axisymmetric",
            direction_positive="less axisymmetric",
            provenance=shared,
        ),
        FunctionalSpec(
            "dav",
            dav,
            "deg²",
            f"deviation-angle variance within {config.organization_radius_km:g} km",
            support=(0.0, 8100.0),
            direction_negative="lower DAV",
            direction_positive="higher DAV",
            provenance=shared,
        ),
        FunctionalSpec(
            "eye_contrast",
            eye_contrast,
            "K",
            "mean eye brightness temperature minus mean eyewall brightness temperature",
            support=(
                -(config.bt_max_k - config.bt_min_k),
                config.bt_max_k - config.bt_min_k,
            ),
            direction_negative="weaker warm-eye contrast",
            direction_positive="stronger warm-eye contrast",
            provenance=shared,
        ),
        FunctionalSpec(
            "p01_bt",
            p01,
            "K",
            "per-image first percentile of brightness temperature",
            support=(config.bt_min_k, config.bt_max_k),
            tail_referenced=True,
            direction_negative="colder lower tail",
            direction_positive="warmer lower tail",
            provenance=shared,
        ),
    )


def curve_metrics(config: TCBTConfig = TCBTConfig()) -> tuple[MetricSpec, MetricSpec]:
    profile_centers = (
        (np.arange(config.profile_bins, dtype=np.float64) + 0.5)
        * config.max_profile_radius_km
        / config.profile_bins
    )
    profile = MetricSpec(
        id="radial_profile_mae",
        transform=lambda fields: radial_profiles(fields, config),
        estimator=lambda generated, real: float(
            np.mean(np.abs(np.mean(generated, axis=0) - np.mean(real, axis=0)))
        ),
        kind=MetricKind.CURVE,
        units="K",
        description="MAE between mean azimuthal BT profiles on fixed physical radial bins.",
        provenance={
            "bins": config.profile_bins,
            "radius_km": config.max_profile_radius_km,
            "pixel_size_km": config.pixel_size_km,
            "coordinate_label": "radius (km)",
            "coordinate_values": profile_centers.tolist(),
        },
    )
    psd_centers = None
    if config.field_shape is not None:
        _, selected_centers = _psd_geometry(config.field_shape, config)
        psd_centers = selected_centers.tolist()
    psd = MetricSpec(
        id="radial_log_psd_mse",
        transform=lambda fields: radial_log_psd(fields, config),
        estimator=lambda generated, real: float(
            np.mean((np.mean(generated, axis=0) - np.mean(real, axis=0)) ** 2)
        ),
        kind=MetricKind.CURVE,
        units="dex²",
        description="Mean squared difference between mean radial log10 power spectra.",
        provenance={
            "bins": config.psd_bins,
            "binning": (
                "distinct discrete radial wavenumbers nearest log-spaced targets; "
                "geometric-midpoint boundaries"
            ),
            "detrending": "per-image spatial mean removed",
            "window": "separable Hann",
            "normalization": "window-energy normalized",
            "discrepancy_units": "squared base-10 log-power difference (dex²)",
            "coordinate_label": "wavenumber (cycles/km)",
            "coordinate_values": psd_centers,
            "coordinate_units": "cycles/km",
            "field_shape": config.field_shape,
        },
    )
    return profile, psd
