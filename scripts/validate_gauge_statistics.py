#!/usr/bin/env python3
"""Monte Carlo validation and paper figure for GAUGE's reference detector."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gauge_climate.inference import simultaneous_reference_boundaries  # noqa: E402
from gauge_climate.metrics import functional_w1  # noqa: E402
from gauge_climate.reference import (  # noqa: E402
    ReferencePlan,
    build_reference_plan,
    evaluate_reference_plan,
    reference_position,
)
from gauge_climate.types import EvaluationConfig, FunctionalSpec  # noqa: E402


SCHEMA = "gauge_climate.statistical_validation.v3"
SHIFT_EFFECTS_SD = (0.0, 0.15, 0.30, 0.45, 0.55)
ALTERNATIVE_LABELS = {
    "variance_narrow": "narrow variance",
    "variance_wide": "excess variance",
    "crossing_cdf": "crossing CDF",
    "tail_contamination": "tail contamination",
    "train_test_shift": "train/test shift",
}


@dataclass(frozen=True)
class Regime:
    id: str
    label: str
    n_blocks: int
    n_eval: int
    block_sizes: str


REGIMES = (
    Regime("balanced_80", "80 balanced blocks", 80, 240, "balanced"),
    Regime("unequal_80", "80 unequal blocks", 80, 240, "severely_unequal"),
    Regime("c5_like_22", "22 blocks, N=1,000", 22, 1_000, "c5_like"),
    Regime("minimum_20", "20-block boundary", 20, 1_000, "c5_like"),
)


def _wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - half, center + half


def _rate(successes: int, total: int, *, name: str) -> dict[str, Any]:
    low, high = _wilson(successes, total)
    return {
        name: successes / total,
        "successes": successes,
        "trials": total,
        "wilson_95": [low, high],
    }


def _block_sizes(
    rng: np.random.Generator,
    n_blocks: int,
    model: str,
) -> np.ndarray:
    if model == "balanced":
        return np.full(n_blocks, 4, dtype=int)
    if model == "severely_unequal":
        return np.maximum(
            1,
            np.rint(np.exp(rng.uniform(0.0, np.log(60.0), n_blocks))).astype(int),
        )
    if model == "c5_like":
        # Approximately 3.9 frames per block, with a long but bounded tail:
        # the observed TC C5 test stratum contains 85 frames from 22 storms.
        return np.minimum(rng.geometric(p=0.26, size=n_blocks), 12).astype(int)
    raise ValueError(f"Unknown block-size model {model!r}.")


def _clustered_values(
    rng: np.random.Generator,
    sizes: np.ndarray,
    *,
    location: float = 0.0,
) -> np.ndarray:
    effects = rng.normal(loc=location, scale=0.3, size=sizes.size)
    return effects.repeat(sizes) + rng.normal(size=int(np.sum(sizes)))


def _reference_plan(
    blocks: np.ndarray,
    regime: Regime,
    *,
    resamples: int,
    seed: int,
) -> ReferencePlan:
    config = EvaluationConfig(
        n_eval=regime.n_eval,
        n_resamples=resamples,
        diagnostic_resamples=resamples,
        sensitivity_resamples=20,
        min_blocks_per_half=10,
        warn_blocks_per_half=10,
        seed=seed,
    )
    return build_reference_plan(
        real_indices=np.arange(blocks.size),
        real_blocks=blocks,
        n_generated=regime.n_eval,
        config=config,
        seed=seed,
    )


def _stack_indices(plan: ReferencePlan, attribute: str) -> np.ndarray:
    return np.stack(
        [np.asarray(getattr(draw, attribute), dtype=np.int64) for draw in plan.draws],
        axis=0,
    )


def _vectorized_w1(
    real: np.ndarray,
    generated: np.ndarray,
    plan: ReferencePlan,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate equal-size scalar W1 draws exactly, but in vectorized form."""

    pseudo_indices = _stack_indices(plan, "pseudo_indices")
    real_indices = _stack_indices(plan, "real_indices")
    generated_indices = _stack_indices(plan, "generated_indices")
    pseudo_sorted = np.sort(np.asarray(real)[pseudo_indices], axis=1)
    real_sorted = np.sort(np.asarray(real)[real_indices], axis=1)
    generated_sorted = np.sort(np.asarray(generated)[generated_indices], axis=1)
    reference = np.mean(np.abs(pseudo_sorted - real_sorted), axis=1)
    model = np.mean(np.abs(generated_sorted - real_sorted), axis=1)
    return reference, model


def _detected(reference: np.ndarray, model: np.ndarray, alpha: float = 0.05) -> bool:
    return bool(np.median(model) > np.quantile(reference, 1.0 - alpha))


def _generated_alternatives(
    rng: np.random.Generator,
    n: int,
    marginal_sd: float,
) -> dict[str, np.ndarray]:
    narrow = rng.normal(scale=0.45 * marginal_sd, size=n)
    wide = rng.normal(scale=1.65 * marginal_sd, size=n)
    # A symmetric, nearly two-point mixture retains the null mean and variance
    # while producing strongly crossing CDFs.  This is deliberately a
    # shape-only alternative rather than a disguised location/scale change.
    mixture_location = 0.99 * marginal_sd
    mixture_noise = math.sqrt(max(marginal_sd**2 - mixture_location**2, 1e-12))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=n)
    crossing = signs * mixture_location + rng.normal(scale=mixture_noise, size=n)
    tail = rng.normal(scale=marginal_sd, size=n)
    tail[rng.choice(n, size=max(1, int(round(0.15 * n))), replace=False)] += (
        3.0 * marginal_sd
    )
    return {
        "variance_narrow": narrow,
        "variance_wide": wide,
        "crossing_cdf": crossing,
        "tail_contamination": tail,
        # The generator matches a historical/train population centered at zero,
        # while the held-out population has shifted by +0.55 marginal SD.
        "train_test_shift": rng.normal(loc=-0.55 * marginal_sd, scale=marginal_sd, size=n),
    }


def _check_invariants(plan: ReferencePlan) -> bool:
    return all(
        not (set(draw.blocks_a) & set(draw.blocks_b))
        and len(draw.pseudo_indices) == plan.n_eval
        and len(draw.real_indices) == plan.n_eval
        and len(draw.generated_indices) == plan.n_eval
        for draw in plan.draws
    )


def _simultaneous_trial(
    *,
    rng: np.random.Generator,
    sizes: np.ndarray,
    plan: ReferencePlan,
    marginal_sd: float,
    family_size: int = 5,
) -> tuple[bool, bool, bool]:
    reference_rows = []
    model_positions = []
    pointwise_any = False
    for _ in range(family_size):
        real = _clustered_values(rng, sizes)
        generated = rng.normal(scale=marginal_sd, size=plan.n_eval)
        reference, model = _vectorized_w1(real, generated, plan)
        q50 = float(np.quantile(reference, 0.50))
        q95 = float(np.quantile(reference, 0.95))
        position = reference_position(float(np.median(model)), q50, q95)
        if position is None:
            return False, False, False
        reference_rows.append(reference)
        model_positions.append(position)
        pointwise_any |= position > 1.0
    boundaries = simultaneous_reference_boundaries(
        np.vstack(reference_rows),
        alpha=0.05,
    )
    resolved = bool(np.all(np.isfinite(boundaries)))
    simultaneous_any = bool(
        resolved and np.any(np.asarray(model_positions) > boundaries)
    )
    return pointwise_any, simultaneous_any, resolved


def _trial(
    trial: int,
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 104_729 * trial)
    marginal_sd = math.sqrt(1.0 + 0.3**2)
    result: dict[str, Any] = {
        "trial": trial,
        "regimes": {},
        "invariants": True,
        "engine_parity_max_abs": 0.0,
        "simultaneous": None,
    }
    for regime_index, regime in enumerate(REGIMES):
        regime_rng = np.random.default_rng(rng.integers(0, np.iinfo(np.int64).max))
        sizes = _block_sizes(regime_rng, regime.n_blocks, regime.block_sizes)
        blocks = np.repeat(np.arange(regime.n_blocks), sizes)
        real = _clustered_values(regime_rng, sizes)
        plan = _reference_plan(
            blocks,
            regime,
            resamples=resamples,
            seed=seed + trial * 10_000 + regime_index,
        )
        result["invariants"] &= _check_invariants(plan)

        equal = regime_rng.normal(scale=marginal_sd, size=regime.n_eval)
        reference, model = _vectorized_w1(real, equal, plan)
        node: dict[str, Any] = {
            "null": _detected(reference, model),
            "shift": {},
            "alternatives": {},
        }
        for effect in SHIFT_EFFECTS_SD:
            shifted = regime_rng.normal(
                loc=effect * marginal_sd,
                scale=marginal_sd,
                size=regime.n_eval,
            )
            _, shift_model = _vectorized_w1(real, shifted, plan)
            node["shift"][str(effect)] = _detected(reference, shift_model)
        for name, generated in _generated_alternatives(
            regime_rng,
            regime.n_eval,
            marginal_sd,
        ).items():
            _, alternative_model = _vectorized_w1(real, generated, plan)
            node["alternatives"][name] = _detected(reference, alternative_model)
        result["regimes"][regime.id] = node

        if trial == 0:
            metric = functional_w1(
                FunctionalSpec("value", lambda values: values, "unit", "validation scalar")
            )
            engine = evaluate_reference_plan(
                metric,
                real_values=real,
                generated_values=equal,
                plan=plan,
            )
            result["engine_parity_max_abs"] = max(
                result["engine_parity_max_abs"],
                float(np.max(np.abs(reference - engine.reference_values))),
                float(np.max(np.abs(model - engine.model_values))),
            )

        if regime.id == "balanced_80":
            pointwise, simultaneous, resolved = _simultaneous_trial(
                rng=regime_rng,
                sizes=sizes,
                plan=plan,
                marginal_sd=marginal_sd,
            )
            result["simultaneous"] = {
                "pointwise_any": pointwise,
                "simultaneous_any": simultaneous,
                "resolved": resolved,
            }
    return result


def _invalid_thin_path(*, resamples: int, seed: int) -> bool:
    regime = Regime("invalid_19", "19 blocks", 19, 100, "balanced")
    blocks = np.repeat(np.arange(regime.n_blocks), 4)
    try:
        _reference_plan(blocks, regime, resamples=resamples, seed=seed)
    except ValueError as exc:
        return "at least 20 eligible blocks" in str(exc)
    return False


def _aggregate(
    trials: list[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
    jobs: int,
) -> dict[str, Any]:
    total = len(trials)
    regimes: dict[str, Any] = {}
    for regime in REGIMES:
        nodes = [trial["regimes"][regime.id] for trial in trials]
        null_count = sum(bool(node["null"]) for node in nodes)
        shifts = []
        for effect in SHIFT_EFFECTS_SD:
            count = sum(bool(node["shift"][str(effect)]) for node in nodes)
            shifts.append(
                {
                    "effect_sd": effect,
                    **_rate(count, total, name="detection_power"),
                }
            )
        alternatives = {}
        for name in ALTERNATIVE_LABELS:
            count = sum(bool(node["alternatives"][name]) for node in nodes)
            alternatives[name] = _rate(count, total, name="detection_power")
        regimes[regime.id] = {
            **asdict(regime),
            "null": _rate(null_count, total, name="detection_rate"),
            "location_shift": shifts,
            "alternatives": alternatives,
        }

    simultaneous_nodes = [
        trial["simultaneous"]
        for trial in trials
        if trial["simultaneous"] is not None
    ]
    pointwise_count = sum(bool(node["pointwise_any"]) for node in simultaneous_nodes)
    simultaneous_count = sum(bool(node["simultaneous_any"]) for node in simultaneous_nodes)
    simultaneous_resolved = all(bool(node["resolved"]) for node in simultaneous_nodes)
    simultaneous = {
        "family_size": 5,
        "pointwise_any": _rate(
            pointwise_count,
            len(simultaneous_nodes),
            name="familywise_detection_rate",
        ),
        "bonferroni_any": _rate(
            simultaneous_count,
            len(simultaneous_nodes),
            name="familywise_detection_rate",
        ),
        "all_boundaries_resolved": simultaneous_resolved,
        "minimum_resamples": math.ceil(5 / 0.05 - 1),
    }

    null_gates = {
        regime.id: regimes[regime.id]["null"]["wilson_95"][1] <= 0.05
        for regime in REGIMES
    }
    standard_shift = next(
        item
        for item in regimes["balanced_80"]["location_shift"]
        if math.isclose(item["effect_sd"], 0.55)
    )
    standard_alternative_gates = {
        name: node["wilson_95"][0] >= 0.80
        for name, node in regimes["balanced_80"]["alternatives"].items()
    }
    gates = {
        "reference_plan_invariants": all(bool(trial["invariants"]) for trial in trials),
        "vectorized_engine_parity": max(
            float(trial["engine_parity_max_abs"]) for trial in trials
        )
        <= 1e-12,
        "pointwise_false_positive_upper_at_most_0.05_all_regimes": all(
            null_gates.values()
        ),
        "standard_shift_0.55_power_lower_at_least_0.80": (
            standard_shift["wilson_95"][0] >= 0.80
        ),
        "standard_alternative_power_lower_at_least_0.80": all(
            standard_alternative_gates.values()
        ),
        "simultaneous_familywise_upper_at_most_0.05": (
            simultaneous["bonferroni_any"]["wilson_95"][1] <= 0.05
        ),
        "simultaneous_boundary_resolution": simultaneous_resolved,
        "fewer_than_twenty_blocks_rejected": _invalid_thin_path(
            resamples=resamples,
            seed=seed + 90_000_000,
        ),
    }
    return {
        "schema": SCHEMA,
        "status": "PASSED" if all(gates.values()) else "FAILED",
        "configuration": {
            "seed": seed,
            "trials": total,
            "resamples": resamples,
            "jobs": jobs,
            "alpha": 0.05,
            "minimum_blocks_per_half": 10,
            "shift_effect_units": "marginal standard deviations",
        },
        "regimes": regimes,
        "simultaneous_reference": simultaneous,
        "failure_paths": {
            "nineteen_blocks": {
                "expected": "fail closed before inference",
                "passed": gates["fewer_than_twenty_blocks_rejected"],
            }
        },
        "engine_parity_max_abs": max(
            float(trial["engine_parity_max_abs"]) for trial in trials
        ),
        "gates": gates,
    }


def _validation_figure(result: dict[str, Any], path: Path) -> tuple[Path, Path]:
    regimes = result["regimes"]
    colors = ("#246a73", "#638b8f", "#d2764c", "#a33b20")
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.7))
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.89,
        bottom=0.18,
        wspace=0.28,
        hspace=0.38,
    )

    ax = axes[0, 0]
    labels = [regimes[regime.id]["label"] for regime in REGIMES]
    rates = [regimes[regime.id]["null"]["detection_rate"] for regime in REGIMES]
    lows = [regimes[regime.id]["null"]["wilson_95"][0] for regime in REGIMES]
    highs = [regimes[regime.id]["null"]["wilson_95"][1] for regime in REGIMES]
    y = np.arange(len(REGIMES))
    ax.errorbar(
        rates,
        y,
        xerr=np.maximum(
            np.vstack(
                [
                    np.asarray(rates) - np.asarray(lows),
                    np.asarray(highs) - np.asarray(rates),
                ]
            ),
            0.0,
        ),
        fmt="o",
        color="#246a73",
        ecolor="#7ca7a8",
        capsize=3,
    )
    ax.axvline(0.05, color="#a33b20", linestyle="--", linewidth=1.2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.065, max(highs) * 1.18))
    ax.set_xlabel("Detection rate under equality")
    ax.set_title("a  Conservative false-positive behavior", loc="left", fontweight="semibold")

    ax = axes[0, 1]
    for regime, color in zip(REGIMES, colors, strict=True):
        rows = regimes[regime.id]["location_shift"]
        x = [row["effect_sd"] for row in rows]
        power = [row["detection_power"] for row in rows]
        ax.plot(x, power, marker="o", color=color, label=regime.label)
    ax.axhline(0.80, color="#a33b20", linestyle="--", linewidth=1.1)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("Location shift (marginal SD)")
    ax.set_ylabel("Detection power")
    ax.set_title("b  Resolution depends on block support", loc="left", fontweight="semibold")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    names = list(ALTERNATIVE_LABELS)
    x = np.arange(len(names))
    width = 0.18
    for index, (regime, color) in enumerate(zip(REGIMES, colors, strict=True)):
        values = [
            regimes[regime.id]["alternatives"][name]["detection_power"]
            for name in names
        ]
        ax.bar(
            x + (index - 1.5) * width,
            values,
            width,
            color=color,
            label=regime.label,
        )
    ax.axhline(0.80, color="#a33b20", linestyle="--", linewidth=1.1)
    ax.set_xticks(x, [ALTERNATIVE_LABELS[name] for name in names], rotation=20, ha="right")
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Detection power")
    ax.set_title("c  Distributional alternatives", loc="left", fontweight="semibold")

    ax = axes[1, 1]
    simultaneous = result["simultaneous_reference"]
    values = [
        simultaneous["pointwise_any"]["familywise_detection_rate"],
        simultaneous["bonferroni_any"]["familywise_detection_rate"],
    ]
    intervals = [
        simultaneous["pointwise_any"]["wilson_95"],
        simultaneous["bonferroni_any"]["wilson_95"],
    ]
    xx = np.arange(2)
    ax.errorbar(
        xx,
        values,
        yerr=np.maximum(
            np.vstack(
                [
                    np.asarray(values) - np.asarray([item[0] for item in intervals]),
                    np.asarray([item[1] for item in intervals]) - np.asarray(values),
                ]
            ),
            0.0,
        ),
        fmt="o",
        color="#246a73",
        ecolor="#7ca7a8",
        capsize=4,
    )
    ax.axhline(0.05, color="#a33b20", linestyle="--", linewidth=1.2)
    ax.set_xticks(xx, ["Any pointwise\ncell", "Bonferroni\nfamily"])
    ax.set_ylim(0, max(0.065, max(item[1] for item in intervals) * 1.18))
    ax.set_ylabel("Familywise detection under equality")
    ax.set_title("d  Registered joint claims", loc="left", fontweight="semibold")
    ax.text(
        0.02,
        0.96,
        "Fewer than 20 blocks: inference withheld ✓",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        color="#2d6a4f",
    )

    for ax in axes.flat:
        ax.grid(color="#e1e7e9", linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "GAUGE statistical validation",
        fontsize=17,
        fontweight="semibold",
        y=0.97,
    )
    fig.text(
        0.08,
        0.022,
        (
            f"{result['configuration']['trials']} Monte Carlo trials; "
            f"{result['configuration']['resamples']} whole-block reference draws per trial. "
            "Error bars show Wilson 95% intervals; power bars are observed rates."
        ),
        fontsize=8.5,
        color="#53636a",
    )
    pdf = Path(f"{path}.pdf")
    png = Path(f"{path}.png")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def run_validation(
    *,
    trials: int,
    resamples: int,
    seed: int,
    jobs: int,
) -> dict[str, Any]:
    if jobs == 1:
        trial_results = [
            _trial(trial, seed=seed, resamples=resamples)
            for trial in range(trials)
        ]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(_trial, trial, seed=seed, resamples=resamples)
                for trial in range(trials)
            ]
            trial_results = [future.result() for future in futures]
        trial_results.sort(key=lambda item: int(item["trial"]))
    return _aggregate(
        trial_results,
        resamples=resamples,
        seed=seed,
        jobs=jobs,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--resamples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1789)
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(4, os.cpu_count() or 1),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/statistical_validation_v0.2.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("validation/statistical_validation_v0.2"),
        help="Figure base path; .pdf and .png are added.",
    )
    args = parser.parse_args()
    if args.trials < 100 or args.resamples < 100:
        raise ValueError("Validation requires at least 100 trials and 100 resamples.")
    if args.jobs < 1:
        raise ValueError("--jobs must be positive.")
    result = run_validation(
        trials=args.trials,
        resamples=args.resamples,
        seed=args.seed,
        jobs=args.jobs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    figure_paths = _validation_figure(result, args.figure)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "figures": [str(path) for path in figure_paths],
                "gates": result["gates"],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
