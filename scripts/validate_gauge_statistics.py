#!/usr/bin/env python3
"""Monte Carlo validation gate for GAUGE's Track-1 statistical procedures.

This script deliberately reports the classical clustered-bootstrap interval as
diagnostic only. The release gate is based on the finite-sample DKW W1 interval,
reference-plan invariants, and sensitivity monotonicity.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gauge_climate.inference import (  # noqa: E402
    dkw_w1_interval,
    functional_w1_interval,
    simultaneous_reference_boundaries,
)
from gauge_climate.metrics import functional_w1  # noqa: E402
from gauge_climate.reference import build_reference_plan, evaluate_reference_plan  # noqa: E402
from gauge_climate.types import (  # noqa: E402
    EvaluationConfig,
    FunctionalSpec,
    SamplingEstimand,
)


def _bounded_clustered_sample(rng: np.random.Generator, n_blocks: int, frames: int):
    block_effects = rng.uniform(-0.2, 0.2, size=n_blocks)
    blocks = np.repeat(np.arange(n_blocks), frames)
    values = block_effects.repeat(frames) + rng.uniform(-0.8, 0.8, size=blocks.size)
    return values, blocks


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def run_validation(*, trials: int, resamples: int, seed: int) -> dict:
    coverage_hits = 0
    equality_false_positives = 0
    bootstrap_coverage_hits = 0
    simultaneous_false_positives = 0
    simultaneous_resolved = True
    plan_failures = 0
    detection_hits = Counter({
        "equality": 0,
        "location_shift": 0,
        "variance_change": 0,
        "tail_contamination": 0,
        "train_test_shift": 0,
        "unequal_block_equality": 0,
        "frame_weighted_unequal_block_equality": 0,
    })
    shift = 0.4
    for trial in range(trials):
        rng = np.random.default_rng(seed + trial)
        joint_rng = np.random.default_rng(seed + 80_000 + trial)
        joint_reference = joint_rng.gamma(shape=2.0, size=(5, resamples))
        joint_boundaries = simultaneous_reference_boundaries(
            joint_reference, alpha=0.05
        )
        if not np.all(np.isfinite(joint_boundaries)):
            simultaneous_resolved = False
        else:
            joint_q50 = np.quantile(joint_reference, 0.50, axis=1)
            joint_q95 = np.quantile(joint_reference, 0.95, axis=1)
            joint_model = joint_rng.gamma(shape=2.0, size=5)
            joint_position = (joint_model - joint_q50) / np.maximum(
                joint_q95 - joint_q50, np.finfo(float).eps
            )
            simultaneous_false_positives += int(
                np.any(joint_position > joint_boundaries)
            )
        real, blocks = _bounded_clustered_sample(rng, n_blocks=80, frames=3)
        generated_equal = (
            rng.uniform(-0.2, 0.2, size=240) + rng.uniform(-0.8, 0.8, size=240)
        )
        generated_shifted = (
            rng.uniform(-0.2, 0.2, size=240)
            + rng.uniform(-0.8, 0.8, size=240)
            + shift
        )
        generated_variance = (
            rng.uniform(-0.2, 0.2, size=240)
            + 0.35 * rng.uniform(-0.8, 0.8, size=240)
        )
        generated_tail = (
            rng.uniform(-0.2, 0.2, size=240) + rng.uniform(-0.8, 0.8, size=240)
        )
        generated_tail[rng.choice(240, size=48, replace=False)] = 1.45
        equal_interval = dkw_w1_interval(
            generated_equal,
            real,
            blocks,
            n_eval=160,
            support=(-1.1, 1.5),
            alpha=0.05,
            seed=seed + 10_000 + trial,
        )
        equality_false_positives += int(equal_interval.low > 0.0)
        shifted_interval = dkw_w1_interval(
            generated_shifted,
            real,
            blocks,
            n_eval=160,
            support=(-1.1, 1.5),
            alpha=0.05,
            seed=seed + 20_000 + trial,
        )
        coverage_hits += int(shifted_interval.low <= shift <= shifted_interval.high)
        stress_blocks = np.repeat(np.arange(50), 4)
        stress_effects = rng.normal(0.0, 0.25, size=50)
        stress_real = stress_effects.repeat(4) + rng.normal(0.0, 1.0, size=200)
        stress_generated = rng.normal(0.5, np.sqrt(1.0 + 0.25**2), size=120)
        bootstrap = functional_w1_interval(
            stress_generated,
            stress_real,
            stress_blocks,
            n_eval=100,
            estimand=SamplingEstimand.BLOCK_BALANCED,
            n_resamples=resamples,
            confidence=0.95,
            seed=seed + 30_000 + trial,
        )
        bootstrap_coverage_hits += int(bootstrap.low <= 0.5 <= bootstrap.high)

        sizes = np.maximum(
            1, np.rint(np.exp(rng.uniform(0.0, np.log(60.0), size=40)))
        ).astype(int)
        uneven_blocks = np.repeat(np.arange(40), sizes)
        config = EvaluationConfig(
            n_eval=80,
            n_resamples=max(20, resamples // 2),
            bootstrap_resamples=20,
            sensitivity_resamples=20,
            min_blocks_per_half=8,
            warn_blocks_per_half=8,
            seed=seed + trial,
        )
        plan = build_reference_plan(
            real_indices=np.arange(uneven_blocks.size),
            real_blocks=uneven_blocks,
            n_generated=100,
            config=config,
            seed=seed + 40_000 + trial,
        )
        normalized = np.asarray([str(value) for value in uneven_blocks], dtype=object)
        for draw in plan.draws:
            if not set(draw.blocks_a).isdisjoint(draw.blocks_b):
                plan_failures += 1
            if not set(normalized[draw.pseudo_indices]).issubset(draw.blocks_a):
                plan_failures += 1
            if not set(normalized[draw.real_indices]).issubset(draw.blocks_b):
                plan_failures += 1

        unequal_effects = rng.uniform(-0.2, 0.2, size=40)
        unequal_values = unequal_effects.repeat(sizes) + rng.uniform(
            -0.8, 0.8, size=uneven_blocks.size
        )
        unequal_generated = rng.uniform(-0.2, 0.2, size=100) + rng.uniform(
            -0.8, 0.8, size=100
        )
        unequal_metric = functional_w1(
            FunctionalSpec(
                "unequal_value",
                lambda fields: fields[:, 0],
                "unit",
                "unequal-block validation functional",
                support=(-1.1, 1.1),
            )
        )
        unequal_evaluation = evaluate_reference_plan(
            unequal_metric,
            real_values=unequal_values,
            generated_values=unequal_generated,
            plan=plan,
        )
        detection_hits["unequal_block_equality"] += int(
            unequal_evaluation.model_median > unequal_evaluation.reference_q95
        )
        frame_config = EvaluationConfig(
            n_eval=80,
            n_resamples=max(20, resamples // 2),
            bootstrap_resamples=20,
            sensitivity_resamples=20,
            min_blocks_per_half=8,
            warn_blocks_per_half=8,
            estimand=SamplingEstimand.FRAME_WEIGHTED,
            seed=seed + trial,
        )
        frame_plan = build_reference_plan(
            real_indices=np.arange(uneven_blocks.size),
            real_blocks=uneven_blocks,
            n_generated=100,
            config=frame_config,
            seed=seed + 45_000 + trial,
        )
        frame_evaluation = evaluate_reference_plan(
            unequal_metric,
            real_values=unequal_values,
            generated_values=unequal_generated,
            plan=frame_plan,
        )
        detection_hits["frame_weighted_unequal_block_equality"] += int(
            frame_evaluation.model_median > frame_evaluation.reference_q95
        )

        detection_config = EvaluationConfig(
            n_eval=160,
            n_resamples=resamples,
            bootstrap_resamples=20,
            sensitivity_resamples=20,
            min_blocks_per_half=10,
            warn_blocks_per_half=10,
            seed=seed + trial,
        )
        detection_plan = build_reference_plan(
            real_indices=np.arange(real.size),
            real_blocks=blocks,
            n_generated=240,
            config=detection_config,
            seed=seed + 50_000 + trial,
        )
        spec = FunctionalSpec(
            "bounded_value",
            lambda fields: fields[:, 0],
            "unit",
            "bounded validation functional",
            support=(-1.1, 1.5),
        )
        metric = functional_w1(spec)
        scenarios = {
            "equality": generated_equal,
            "location_shift": generated_shifted,
            "variance_change": generated_variance,
            "tail_contamination": generated_tail,
            # The held-out target is shifted relative to the implicit training
            # distribution represented by this unshifted generator.
            "train_test_shift": generated_equal - shift,
        }
        for name, values in scenarios.items():
            evaluated = evaluate_reference_plan(
                metric,
                real_values=real,
                generated_values=values,
                plan=detection_plan,
            )
            detection_hits[name] += int(
                evaluated.model_median > evaluated.reference_q95
            )

    coverage = coverage_hits / trials
    false_positive = equality_false_positives / trials
    bootstrap_coverage = bootstrap_coverage_hits / trials
    coverage_wilson = _wilson_interval(coverage_hits, trials)
    false_positive_wilson = _wilson_interval(equality_false_positives, trials)
    simultaneous_rate = simultaneous_false_positives / trials
    simultaneous_wilson = _wilson_interval(simultaneous_false_positives, trials)
    reference_rates = {key: value / trials for key, value in detection_hits.items()}
    reference_intervals = {
        key: list(_wilson_interval(value, trials)) for key, value in detection_hits.items()
    }
    gate = {
        "dkw_coverage_at_least_nominal_with_mc_uncertainty": coverage_wilson[0] >= 0.95,
        "dkw_false_positive_at_most_alpha_with_mc_uncertainty": false_positive_wilson[1]
        <= 0.05,
        "reference_plan_invariants": plan_failures == 0,
        "reference_detection_false_positive_control": reference_intervals["equality"][1]
        <= 0.05,
        "reference_detection_location_power": reference_intervals["location_shift"][0]
        >= 0.80,
        "reference_detection_variance_power": reference_intervals["variance_change"][0]
        >= 0.80,
        "reference_detection_tail_power": reference_intervals["tail_contamination"][0]
        >= 0.80,
        "reference_detection_train_test_shift_power": reference_intervals["train_test_shift"][0]
        >= 0.80,
        "unequal_block_false_positive_control": reference_intervals[
            "unequal_block_equality"
        ][1]
        <= 0.05,
        "frame_weighted_unequal_block_false_positive_control": reference_intervals[
            "frame_weighted_unequal_block_equality"
        ][1]
        <= 0.05,
        "simultaneous_reference_calibration": simultaneous_resolved
        and simultaneous_wilson[0] <= 0.05 <= simultaneous_wilson[1],
    }
    return {
        "schema": "gauge_climate.statistical_validation.v1",
        "seed": seed,
        "trials": trials,
        "resamples": resamples,
        "scenarios": {
            "bounded_shift_dkw": {
                "true_w1": shift,
                "coverage": coverage,
                "coverage_wilson_95": list(coverage_wilson),
            },
            "bounded_equality_dkw": {
                "false_positive_rate": false_positive,
                "false_positive_wilson_95": list(false_positive_wilson),
            },
            "cluster_bootstrap_diagnostic": {
                "coverage": bootstrap_coverage,
                "release_gate": False,
                "reason": "Classical bootstrap coverage is not guaranteed for empirical W1.",
            },
            "unequal_block_reference_plan": {"invariant_failures": plan_failures},
            "simultaneous_reference": {
                "method": "Bonferroni order-statistic boundaries",
                "family_size": 5,
                "false_positive_rate": simultaneous_rate,
                "false_positive_wilson_95": list(simultaneous_wilson),
                "resolved_at_declared_resamples": simultaneous_resolved,
            },
            "reference_detection": {
                "rule": "median model discrepancy exceeds q95 real-reference discrepancy",
                "calibration_target": "one-sided type-I error no greater than alpha; conservatism is reported",
                "rates": reference_rates,
                "wilson_95": reference_intervals,
                "declared_severities": {
                    "location_shift": 0.4,
                    "variance_scale_on_frame_component": 0.35,
                    "tail_contamination_fraction": 0.20,
                    "tail_value": 1.45,
                    "train_test_location_shift": 0.4,
                },
            },
        },
        "gate": gate,
        "passed": all(gate.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--resamples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1789)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trials < 20 or args.resamples < 20:
        parser.error("--trials and --resamples must both be at least 20")
    result = run_validation(trials=args.trials, resamples=args.resamples, seed=args.seed)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
