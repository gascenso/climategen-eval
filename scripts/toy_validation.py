#!/usr/bin/env python
"""Synthetic ground-truth validation of the skill-score framework (§13a of the
methods document docs/methods/skill_score_framework.md).

The TC case study has no ground-truth "distance from realism". This experiment
builds a synthetic *real* law whose structure is known, constructs "models" with
*planted* failure modes, and the property-preserving baseline panel, then checks
the three claims the framework rests on:

  (1) the noise-floor / studentized score  G(N)  DIVERGES with N for any imperfect
      model, while the skill score  S(N)  CONVERGES to a finite population value;
  (2) the CI-admissibility filter EXCLUDES a baseline on the metrics whose property
      it preserves (e.g. the covariance-matched Gaussian on the spectral metric);
  (3) the recovered per-metric skill profile MATCHES the planted failure modes.

It is deliberately self-contained (numpy only; matplotlib optional for plots) so
it validates the framework logic independently of the TC-specific pipeline. The
baseline constructions mirror src/climategen_eval/baselines.py.

Usage:
    python scripts/toy_validation.py --output runs/toy_validation.json
    python scripts/toy_validation.py --output runs/toy_validation.json --plot runs/toy
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

Sample = np.ndarray  # (N, H, W) float32 stack of fields
Metric = Callable[[Sample, Sample], float]  # lower-is-better, min at equal laws


# --------------------------------------------------------------------------- #
# Synthetic real law: correlated Gaussian field -> non-Gaussian marginal -> eye
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RealParams:
    size: int = 32
    psd_beta: float = 2.6        # power-law slope of the base spectrum
    k0: float = 2.0             # low-wavenumber roll-off
    nonlinearity: float = 0.55  # strength of the marginal-skewing nonlinearity
    eye_amp: float = 1.6        # amplitude of the injected coherent core
    eye_radius: float = 4.0     # core radius (pixels)
    eye_jitter: float = 2.0     # center jitter std (pixels)


def _radial_grid(size: int) -> np.ndarray:
    ky = np.fft.fftfreq(size)[:, None] * size
    kx = np.fft.fftfreq(size)[None, :] * size
    return np.sqrt(ky**2 + kx**2)


def _spectral_filter(size: int, beta: float, k0: float) -> np.ndarray:
    k = _radial_grid(size)
    filt = 1.0 / np.power(k**2 + k0**2, beta / 2.0)
    filt[0, 0] = 0.0  # zero-mean fields
    return filt


def _coherent_core(size: int, amp: float, radius: float, cy: float, cx: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return amp * np.exp(-r2 / (2.0 * radius**2))


def sample_real(
    n: int,
    rng: np.random.Generator,
    p: RealParams,
    *,
    gaussian_only: bool = False,
    with_eye: bool = True,
    eye_amp_scale: float = 1.0,
) -> Sample:
    """Draw n synthetic fields from the (optionally ablated) real law.

    gaussian_only=True drops the marginal-skewing nonlinearity (a covariance-only
    model). with_eye=False / eye_amp_scale drops or attenuates the coherent core.
    """
    size = p.size
    filt = _spectral_filter(size, p.psd_beta, p.k0)
    white = rng.standard_normal((n, size, size))
    spec = np.fft.fft2(white) * filt[None]
    g = np.fft.ifft2(spec).real
    g = (g - g.mean()) / (g.std() + 1e-8)
    field = g
    if not gaussian_only:
        # sinh-arcsinh-style skew: keeps variance ~O(1), adds non-Gaussian tail
        field = np.sinh(p.nonlinearity * np.arcsinh(g)) / max(p.nonlinearity, 1e-3)
    if with_eye and eye_amp_scale != 0.0:
        cy = size / 2 + rng.normal(0, p.eye_jitter, n)
        cx = size / 2 + rng.normal(0, p.eye_jitter, n)
        for i in range(n):
            field[i] = field[i] - _coherent_core(
                size, p.eye_amp * eye_amp_scale, p.eye_radius, cy[i], cx[i]
            )
    return field.astype(np.float32)


# --------------------------------------------------------------------------- #
# Planted-failure "models" and property-preserving baselines (from a real pool)
# --------------------------------------------------------------------------- #
def model_perfect(pool: Sample, n: int, rng, p: RealParams) -> Sample:
    return sample_real(n, rng, p)                                  # truly real law


def model_close(pool: Sample, n: int, rng, p: RealParams) -> Sample:
    return sample_real(n, rng, p, eye_amp_scale=0.85)             # slightly weak core


def model_marginal_only(pool: Sample, n: int, rng, p: RealParams) -> Sample:
    # Correct per-image marginal, destroyed spatial structure (pixel shuffle).
    idx = rng.integers(pool.shape[0], size=n)
    out = pool[idx].copy()
    for i in range(n):
        flat = out[i].ravel()
        rng.shuffle(flat)
        out[i] = flat.reshape(out[i].shape)
    return out


def model_covariance_only(pool: Sample, n: int, rng, p: RealParams) -> Sample:
    return _gauss_full(pool, n, rng)            # right covariance, Gaussian, no eye


MODELS: dict[str, Callable[..., Sample]] = {
    "PERFECT": model_perfect,
    "CLOSE": model_close,
    "MARGINAL_ONLY": model_marginal_only,
    "COVARIANCE_ONLY": model_covariance_only,
}


def _gauss_full(pool: Sample, n: int, rng) -> Sample:
    nb, h, w = pool.shape
    mean = pool.mean(axis=0)
    resid = (pool - mean).reshape(nb, -1).astype(np.float64)
    z = rng.standard_normal((n, nb))
    deltas = (z @ resid) / np.sqrt(nb - 1)
    return (mean.reshape(1, -1) + deltas).reshape(n, h, w).astype(np.float32)


def baseline_clim(pool, n, rng):
    return np.broadcast_to(pool.mean(axis=0)[None], (n, *pool.shape[1:])).astype(np.float32)


def baseline_gauss_pix(pool, n, rng):
    mean = pool.mean(axis=0)
    std = pool.std(axis=0, ddof=1)
    return (mean[None] + std[None] * rng.standard_normal((n, *pool.shape[1:]))).astype(np.float32)


def baseline_gauss_full(pool, n, rng):
    return _gauss_full(pool, n, rng)


def _phase_randomize(x, rng):
    F = np.fft.fft2(x.astype(np.float64))
    amp = np.abs(F)
    phi = rng.uniform(-np.pi, np.pi, size=x.shape)
    phi_neg = np.roll(np.roll(phi[::-1, ::-1], 1, 0), 1, 1)
    phi_anti = 0.5 * (phi - phi_neg)
    return np.fft.ifft2(amp * np.exp(1j * phi_anti)).real.astype(np.float32)


def baseline_phase_shuffle(pool, n, rng):
    idx = rng.integers(pool.shape[0], size=n)
    return np.stack([_phase_randomize(pool[i], rng) for i in idx]).astype(np.float32)


BASELINES: dict[str, Callable[..., Sample]] = {
    "CLIM": baseline_clim,
    "GAUSS-PIX": baseline_gauss_pix,
    "GAUSS-FULL": baseline_gauss_full,
    "PHASE-SHUFFLE": baseline_phase_shuffle,
}


# --------------------------------------------------------------------------- #
# Metric suite (lower-is-better; each minimized when A, B share a law)
# --------------------------------------------------------------------------- #
def _w1_1d(a: np.ndarray, b: np.ndarray, q: int = 512) -> float:
    qs = np.linspace(0.0, 1.0, q)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def metric_marginal_w1(a: Sample, b: Sample) -> float:
    # Pooled pixel-value distribution (subsampled for speed).
    return _w1_1d(a.reshape(-1)[::4], b.reshape(-1)[::4])


def _mean_radial_psd(imgs: Sample, nbins: int = 16) -> np.ndarray:
    size = imgs.shape[-1]
    power = np.mean(np.abs(np.fft.fft2(imgs)) ** 2, axis=0)
    k = _radial_grid(size).ravel()
    p = power.ravel()
    edges = np.linspace(0, k.max() + 1e-6, nbins + 1)
    which = np.clip(np.digitize(k, edges) - 1, 0, nbins - 1)
    prof = np.array([p[which == i].mean() if np.any(which == i) else 0.0 for i in range(nbins)])
    return np.log(prof + 1e-8)


def metric_radial_psd_mse(a: Sample, b: Sample) -> float:
    return float(np.mean((_mean_radial_psd(a) - _mean_radial_psd(b)) ** 2))


def _eye_contrast(imgs: Sample, r_in: float = 4.0, r_out: float = 10.0) -> np.ndarray:
    size = imgs.shape[-1]
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
    core = r <= r_in
    annulus = (r > r_in) & (r <= r_out)
    return imgs[:, core].mean(axis=1) - imgs[:, annulus].mean(axis=1)


def metric_coherent_gap(a: Sample, b: Sample) -> float:
    return _w1_1d(_eye_contrast(a), _eye_contrast(b))


METRICS: dict[str, Metric] = {
    "marginal_w1": metric_marginal_w1,
    "radial_psd_mse": metric_radial_psd_mse,
    "coherent_gap": metric_coherent_gap,
}


# --------------------------------------------------------------------------- #
# Framework core: anchors, admissibility, skill, and the noise-floor gap
# --------------------------------------------------------------------------- #
def _bootstrap_metric(
    sample_fn: Callable[[int], Sample],
    real_ref_fn: Callable[[int], Sample],
    metric: Metric,
    n: int,
    reps: int,
    rng,
) -> np.ndarray:
    return np.array([metric(real_ref_fn(n), sample_fn(n)) for _ in range(reps)])


def evaluate_at_n(
    *,
    real_pool: Sample,
    real_rng_factory: Callable[[], np.random.Generator],
    p: RealParams,
    n: int,
    reps: int,
    eps: float,
    rng: np.random.Generator,
) -> dict:
    """Compute null anchors, baseline admissibility, and per-model skill at size n."""
    def draw_real(_n):  # an independent real reference sample of size _n
        return sample_real(_n, real_rng_factory(), p)

    out: dict[str, dict] = {}
    for mname, metric in METRICS.items():
        # Real-vs-real null -> perfect anchor a1 and ribbon upper edge r.
        null = _bootstrap_metric(draw_real, draw_real, metric, n, reps, rng)
        a1 = float(np.median(null))
        r = float(np.quantile(null, 1.0 - eps))

        # Baselines: bootstrap distribution; CI-separation admissibility.
        base_stats: dict[str, dict] = {}
        for bname, builder in BASELINES.items():
            vals = _bootstrap_metric(draw_real, lambda _n, b=builder: b(real_pool, _n, rng), metric, n, reps, rng)
            lo = float(np.quantile(vals, eps))  # lower confidence edge
            admissible = bool(lo > r)           # CI lies outside the ribbon
            base_stats[bname] = {"median": float(np.median(vals)), "ci_lo": lo, "admissible": admissible}

        admissible = {b: s for b, s in base_stats.items() if s["admissible"]}
        if admissible:
            ref_name = min(admissible, key=lambda b: admissible[b]["median"])
            a0 = admissible[ref_name]["median"]
        else:
            ref_name, a0 = None, None

        # Per-model skill S and noise-floor gap G.
        models: dict[str, dict] = {}
        for mod_name, builder in MODELS.items():
            mvals = _bootstrap_metric(draw_real, lambda _n, b=builder: b(real_pool, _n, rng, p), metric, n, reps, rng)
            mval = float(np.median(mvals))
            denom_floor = r - a1
            g = float((mval - a1) / denom_floor) if denom_floor > 0 else float("inf")
            if a0 is not None and (a0 - a1) != 0:
                s = float((a0 - mval) / (a0 - a1))
            else:
                s = None
            models[mod_name] = {"value": mval, "skill": s, "noise_floor_gap": g}

        out[mname] = {
            "a1_perfect": a1,
            "ribbon_upper": r,
            "reference_baseline": ref_name,
            "a0_reference": a0,
            "baselines": base_stats,
            "models": models,
        }
    return out


def run(ns: list[int], reps: int, eps_grid: list[float], pool_size: int, seed: int) -> dict:
    p = RealParams()
    master = np.random.default_rng(seed)
    real_pool = sample_real(pool_size, np.random.default_rng(seed + 1), p)
    # Fresh independent real reference draws (ground-truth perfect generator).
    real_rng_factory = lambda: np.random.default_rng(master.integers(1 << 31))

    per_eps: dict[str, dict] = {}
    for eps in eps_grid:
        per_n = {}
        for n in ns:
            per_n[str(n)] = evaluate_at_n(
                real_pool=real_pool,
                real_rng_factory=real_rng_factory,
                p=p,
                n=n,
                reps=reps,
                eps=eps,
                rng=np.random.default_rng(seed + 7 * n + int(1000 * eps)),
            )
        per_eps[f"{eps:.2f}"] = per_n

    return {
        "schema": "climategen_eval.toy_validation.v1",
        "description": "Synthetic ground-truth validation of the skill-score framework.",
        "planted_failure_modes": {
            "PERFECT": "draws from the true law (expected skill ~1 on every metric)",
            "CLOSE": "slightly attenuated coherent core (high skill, lower on coherent_gap)",
            "MARGINAL_ONLY": "correct marginal, destroyed spatial structure "
            "(high marginal_w1 skill, low spectral/coherent skill)",
            "COVARIANCE_ONLY": "correct covariance, Gaussian marginal, no core "
            "(high spectral skill, low marginal/coherent skill)",
        },
        "expected_admissibility": {
            "radial_psd_mse": "GAUSS-FULL preserves spectrum -> expected NOT admissible",
            "coherent_gap": "PHASE-SHUFFLE/GAUSS-FULL destroy core -> expected admissible",
        },
        "config": {"ns": ns, "reps": reps, "eps_grid": eps_grid, "pool_size": pool_size, "seed": seed},
        "results_by_eps": per_eps,
    }


# --------------------------------------------------------------------------- #
# Plotting (optional): S(N) convergence vs G(N) divergence, and skill profile
# --------------------------------------------------------------------------- #
def make_plots(payload: dict, out_prefix: str, eps: float = 0.05) -> None:
    import matplotlib.pyplot as plt

    res = payload["results_by_eps"][f"{eps:.2f}"]
    ns = sorted(int(n) for n in res)
    metrics = list(next(iter(res.values())).keys())
    model = "MARGINAL_ONLY"  # an imperfect model: shows G -> inf, S -> const

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for met in metrics:
        s = [res[str(n)][met]["models"][model]["skill"] for n in ns]
        g = [res[str(n)][met]["models"][model]["noise_floor_gap"] for n in ns]
        axes[0].plot(ns, s, marker="o", label=met)
        axes[1].plot(ns, g, marker="o", label=met)
    axes[0].axhline(1.0, ls="--", c="gray"); axes[0].axhline(0.0, ls=":", c="gray")
    axes[0].set_title(f"Skill S(N) — converges  [{model}]"); axes[0].set_xlabel("N"); axes[0].set_ylabel("S")
    axes[1].set_title("Noise-floor gap G(N) — diverges"); axes[1].set_xlabel("N"); axes[1].set_ylabel("G")
    for ax in axes:
        ax.set_xscale("log"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out_prefix}_convergence.png", dpi=130); plt.close(fig)

    # Skill profile at largest N for every model.
    big = res[str(ns[-1])]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(metrics)); width = 0.2
    for i, mod in enumerate(MODELS):
        vals = [big[m]["models"][mod]["skill"] for m in metrics]
        vals = [np.nan if v is None else v for v in vals]
        ax.bar(x + i * width, vals, width, label=mod)
    ax.axhline(1.0, ls="--", c="gray"); ax.axhline(0.0, ls=":", c="gray")
    ax.set_xticks(x + 1.5 * width); ax.set_xticklabels(metrics)
    ax.set_ylabel("skill S"); ax.set_title(f"Skill profile @ N={ns[-1]} (planted failures should show)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{out_prefix}_profile.png", dpi=130); plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ns", default="25,50,100,200,400")
    ap.add_argument("--reps", type=int, default=120)
    ap.add_argument("--eps-grid", default="0.01,0.05,0.10")
    ap.add_argument("--pool-size", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--plot", default=None, help="path prefix for PNG plots (requires matplotlib)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ns = sorted({int(x) for x in args.ns.split(",") if x.strip()})
    eps_grid = [float(x) for x in args.eps_grid.split(",") if x.strip()]
    payload = run(ns, args.reps, eps_grid, args.pool_size, args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}")
    if args.plot:
        make_plots(payload, args.plot)
        print(f"Wrote {args.plot}_convergence.png and {args.plot}_profile.png")


if __name__ == "__main__":
    main()
