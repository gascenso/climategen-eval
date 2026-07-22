#!/usr/bin/env python3
"""Monte Carlo validation gate for GAUGE's real-reference detector."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gauge_climate.inference import simultaneous_reference_boundaries  # noqa: E402
from gauge_climate.metrics import functional_w1  # noqa: E402
from gauge_climate.reference import (  # noqa: E402
    build_reference_plan,
    evaluate_reference_plan,
)
from gauge_climate.types import EvaluationConfig, FunctionalSpec  # noqa: E402


def _wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    ) / denominator
    return center - half, center + half


def _clustered_sample(
    rng: np.random.Generator,
    n_blocks: int,
    *,
    unequal_sizes: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    sizes = (
        np.maximum(1, np.rint(np.exp(rng.uniform(0.0, np.log(60.0), n_blocks))))
        .astype(int)
        if unequal_sizes
        else np.full(n_blocks, 4, dtype=int)
    )
    blocks = np.repeat(np.arange(n_blocks), sizes)
    effects = rng.normal(scale=0.3, size=n_blocks)
    values = effects.repeat(sizes) + rng.normal(size=blocks.size)
    return values, blocks


def _detect(
    real: np.ndarray,
    blocks: np.ndarray,
    generated: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[bool, bool]:
    config = EvaluationConfig(
        n_eval=min(160, generated.size),
        n_resamples=resamples,
        diagnostic_resamples=min(resamples, 200),
        sensitivity_resamples=20,
        min_blocks_per_half=10,
        warn_blocks_per_half=20,
        seed=seed,
    )
    plan = build_reference_plan(
        real_indices=np.arange(real.size),
        real_blocks=blocks,
        n_generated=generated.size,
        config=config,
        seed=seed,
    )
    metric = functional_w1(
        FunctionalSpec("value", lambda values: values, "unit", "validation scalar")
    )
    result = evaluate_reference_plan(
        metric,
        real_values=real,
        generated_values=generated,
        plan=plan,
    )
    detected = result.model_median > result.quantile(0.95)
    invariants = all(
        not (set(draw.blocks_a) & set(draw.blocks_b))
        and len(draw.pseudo_indices) == plan.n_eval
        and len(draw.real_indices) == plan.n_eval
        and len(draw.generated_indices) == plan.n_eval
        for draw in plan.draws
    )
    return detected, invariants


def run_validation(*, trials: int, resamples: int, seed: int) -> dict:
    equality_detections = 0
    unequal_equality_detections = 0
    shift_detections = 0
    variance_detections = 0
    tail_detections = 0
    invariant_failures = 0

    for trial in range(trials):
        rng = np.random.default_rng(seed + trial)
        real, blocks = _clustered_sample(rng, 80)
        marginal_sd = math.sqrt(1.0 + 0.3**2)
        equal = rng.normal(scale=marginal_sd, size=240)
        shifted = rng.normal(loc=0.55, scale=marginal_sd, size=240)
        narrowed = rng.normal(scale=0.45 * marginal_sd, size=240)
        tail = rng.normal(scale=marginal_sd, size=240)
        tail[rng.choice(tail.size, size=48, replace=False)] += 3.0

        for offset, generated, counter in (
            (0, equal, "equality"),
            (1_000_000, shifted, "shift"),
            (2_000_000, narrowed, "variance"),
            (3_000_000, tail, "tail"),
        ):
            detected, valid = _detect(
                real,
                blocks,
                generated,
                resamples=resamples,
                seed=seed + offset + trial,
            )
            invariant_failures += int(not valid)
            if counter == "equality":
                equality_detections += int(detected)
            elif counter == "shift":
                shift_detections += int(detected)
            elif counter == "variance":
                variance_detections += int(detected)
            else:
                tail_detections += int(detected)

        unequal_real, unequal_blocks = _clustered_sample(
            rng, 80, unequal_sizes=True
        )
        unequal_equal = rng.normal(scale=marginal_sd, size=240)
        detected, valid = _detect(
            unequal_real,
            unequal_blocks,
            unequal_equal,
            resamples=resamples,
            seed=seed + 4_000_000 + trial,
        )
        unequal_equality_detections += int(detected)
        invariant_failures += int(not valid)

    family_size = 5
    rng = np.random.default_rng(seed + 9_000_000)
    reference_matrix = rng.gamma(shape=2.0, size=(family_size, resamples))
    simultaneous = simultaneous_reference_boundaries(reference_matrix, alpha=0.05)
    minimum_resamples = math.ceil(family_size / 0.05 - 1)

    equality_rate = equality_detections / trials
    unequal_rate = unequal_equality_detections / trials
    shift_power = shift_detections / trials
    equality_ci = _wilson(equality_detections, trials)
    unequal_ci = _wilson(unequal_equality_detections, trials)
    shift_ci = _wilson(shift_detections, trials)
    gates = {
        "reference_plan_invariants": invariant_failures == 0,
        "equality_false_positive_upper_at_most_0.05": equality_ci[1] <= 0.05,
        "unequal_block_false_positive_upper_at_most_0.05": unequal_ci[1] <= 0.05,
        "shift_power_lower_at_least_0.80": shift_ci[0] >= 0.80,
        "simultaneous_boundary_resolution": bool(
            resamples >= minimum_resamples and np.all(np.isfinite(simultaneous))
        ),
    }
    return {
        "schema": "gauge_climate.statistical_validation.v2",
        "status": "PASSED" if all(gates.values()) else "FAILED",
        "seed": seed,
        "trials": trials,
        "resamples": resamples,
        "scenarios": {
            "equality": {"detection_rate": equality_rate, "wilson_95": equality_ci},
            "unequal_block_equality": {
                "detection_rate": unequal_rate,
                "wilson_95": unequal_ci,
            },
            "location_shift_0.55": {
                "detection_power": shift_power,
                "wilson_95": shift_ci,
            },
            "variance_change": {"detection_power": variance_detections / trials},
            "tail_contamination": {"detection_power": tail_detections / trials},
            "simultaneous_reference": {
                "family_size": family_size,
                "minimum_resamples": minimum_resamples,
                "boundaries": simultaneous.tolist(),
            },
        },
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--resamples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1789)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/statistical_validation_v0.1.json"),
    )
    args = parser.parse_args()
    if args.trials < 50 or args.resamples < 100:
        raise ValueError("Validation requires at least 50 trials and 100 resamples.")
    result = run_validation(
        trials=args.trials, resamples=args.resamples, seed=args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
