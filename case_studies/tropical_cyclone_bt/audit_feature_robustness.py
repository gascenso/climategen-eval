#!/usr/bin/env python3
"""Post-hoc robustness audit for the TC training-fitted feature representation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import svds

from gauge_climate.adapters import tc_bt
from gauge_climate.adapters.tc_data import load_real_archive, load_sample_bank
from gauge_climate.features import TrainFittedFeatureSpace
from gauge_climate.metrics import (
    coverage_metric,
    density_metric,
    diversity_metric,
    median_bandwidth,
    mmd_metric,
)
from gauge_climate.reference import (
    build_reference_plan,
    evaluate_reference_plan,
    reference_position,
)
from gauge_climate.types import EvaluationConfig


ROOT = Path(__file__).resolve().parents[2]
CLASS_LABELS = ("TS", "C1", "C2", "C3", "C4", "C5")
METRIC_LABELS = {
    "feature_mmd2": "MMD²",
    "feature_density": "Density departure",
    "feature_coverage_deficit": "Coverage deficit",
    "feature_diversity_departure": "Diversity departure",
}
FUNCTIONAL_IDS = (
    "min_core_bt",
    "cold_frac_220",
    "r_maxgrad",
    "axisym",
    "dav",
    "eye_contrast",
    "p01_bt",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _preprocessed(dataset: Any, config: tc_bt.TCBTConfig) -> Any:
    fields = tc_bt.PreprocessedFields(dataset.fields, config)
    return type(dataset)(
        fields,
        blocks=dataset.blocks,
        labels=dataset.labels,
        sample_ids=dataset.sample_ids,
        name=dataset.name,
        units=dataset.units,
        metadata=dataset.metadata,
    )


def _functional_matrix(fields: np.ndarray, functionals: tuple[Any, ...]) -> np.ndarray:
    return np.column_stack([functional.transform(fields) for functional in functionals])


def _top_eofs(
    fields: np.ndarray,
    *,
    rank: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = np.asarray(fields, dtype=np.float32).reshape(len(fields), -1)
    mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = flat - mean
    if rank >= min(centered.shape):
        raise ValueError("Requested EOF rank is not below the matrix dimensions.")
    _, singular, vt = svds(
        centered,
        k=rank,
        which="LM",
        return_singular_vectors=True,
        random_state=seed,
    )
    order = np.argsort(singular)[::-1]
    singular = singular[order]
    vt = vt[order]
    scale = singular / np.sqrt(max(len(fields) - 1, 1))
    scale = np.where(scale > 0, scale, 1.0)
    return mean, vt.astype(np.float32), scale.astype(np.float64)


def _project(
    fields: Any,
    *,
    mean: np.ndarray,
    components: np.ndarray,
    scale: np.ndarray,
    batch_size: int,
    spatial_filter: Callable[[np.ndarray], np.ndarray] | None,
) -> np.ndarray:
    output = np.empty((len(fields), len(components)), dtype=np.float64)
    for start in range(0, len(fields), batch_size):
        stop = min(start + batch_size, len(fields))
        batch = np.asarray(fields[start:stop], dtype=np.float32)
        if spatial_filter is not None:
            batch = spatial_filter(batch)
        flat = batch.reshape(stop - start, -1)
        output[start:stop] = ((flat - mean) @ components.T) / scale
    return output


def _identity(values: np.ndarray) -> np.ndarray:
    return values


def _evaluate_features(
    *,
    train_features: np.ndarray,
    real_features: np.ndarray,
    generated_features: np.ndarray,
    real_labels: np.ndarray,
    generated_labels: np.ndarray,
    real_blocks: np.ndarray,
    seed: int,
    bandwidth_seed: int,
    n_resamples: int,
    max_samples: int,
) -> dict[str, Any]:
    bandwidth = median_bandwidth(train_features, seed=bandwidth_seed)
    metrics = (
        mmd_metric(
            _identity,
            bandwidth=bandwidth,
            max_samples=max_samples,
            seed=seed + 1,
        ),
        density_metric(_identity, k=5),
        coverage_metric(_identity, k=5),
        diversity_metric(_identity),
    )
    output: dict[str, Any] = {
        "bandwidth": float(bandwidth),
        "classes": {},
    }
    for class_id, class_label in enumerate(CLASS_LABELS):
        real_mask = real_labels == class_id
        generated_mask = generated_labels == class_id
        local_real = real_features[real_mask]
        local_generated = generated_features[generated_mask]
        local_blocks = real_blocks[real_mask]
        plan = build_reference_plan(
            real_indices=np.arange(len(local_real), dtype=np.int64),
            real_blocks=local_blocks,
            n_generated=len(local_generated),
            config=EvaluationConfig(
                n_eval=len(local_generated),
                n_resamples=n_resamples,
                diagnostic_resamples=n_resamples,
                summary_bootstrap_resamples=200,
                min_blocks_per_half=10,
                warn_blocks_per_half=30,
                seed=seed,
                include_partitions=False,
            ),
            seed=seed + 1009 * (class_id + 1),
        )
        class_result: dict[str, Any] = {
            "real_samples": int(real_mask.sum()),
            "generated_samples": int(generated_mask.sum()),
            "real_storms": int(len(np.unique(local_blocks))),
            "metrics": {},
        }
        for metric_index, metric in enumerate(metrics):
            # Density/coverage can be substantially more expensive than MMD and
            # diversity. All four use the same deterministic matched cap.
            base_estimator = metric.estimator

            def capped(
                x: np.ndarray,
                y: np.ndarray,
                *,
                estimator: Callable[[np.ndarray, np.ndarray], float] = base_estimator,
                metric_seed: int = int((bandwidth_seed + 1009 * (metric_index + 1)) % (2**32 - 1)),
            ) -> float:
                rng = np.random.default_rng(metric_seed)
                if len(x) > max_samples:
                    x = x[rng.choice(len(x), max_samples, replace=False)]
                if len(y) > max_samples:
                    y = y[rng.choice(len(y), max_samples, replace=False)]
                return float(estimator(x, y))

            capped_metric = type(metric)(
                id=metric.id,
                transform=metric.transform,
                estimator=capped,
                kind=metric.kind,
                units=metric.units,
                description=metric.description,
                pillar=metric.pillar,
                functional=metric.functional,
                direction=metric.direction,
                provenance=metric.provenance,
            )
            evaluated = evaluate_reference_plan(
                capped_metric,
                real_values=local_real,
                generated_values=local_generated,
                plan=plan,
            )
            q50 = evaluated.reference_q50
            q95 = evaluated.reference_q95
            model = evaluated.model_median
            class_result["metrics"][metric.id] = {
                "model_reference_median": model,
                "model_reference_interval_95": np.quantile(
                    evaluated.model_values, [0.025, 0.975]
                ).tolist(),
                "real_reference_q50": q50,
                "real_reference_q95": q95,
                "reference_gap": reference_position(model, q50, q95),
                "reference_percentile": float(np.mean(evaluated.reference_values <= model)),
            }
        output["classes"][class_label] = class_result
    return output


def _combined_features(
    *,
    functionals_train: np.ndarray,
    functionals_real: np.ndarray,
    functionals_generated: np.ndarray,
    eof_train: np.ndarray,
    eof_real: np.ndarray,
    eof_generated: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = functionals_train.mean(axis=0)
    scale = functionals_train.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return (
        np.column_stack(((functionals_train - mean) / scale, eof_train[:, :rank])),
        np.column_stack(((functionals_real - mean) / scale, eof_real[:, :rank])),
        np.column_stack(((functionals_generated - mean) / scale, eof_generated[:, :rank])),
    )


def _plot(payload: dict[str, Any], output: Path) -> None:
    variants = payload["variants"]
    baseline = "fit512_rank20_scale0km"
    ordered = [baseline, *[name for name in variants if name != baseline]]
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.4), sharex=True)
    for axis, metric_id in zip(axes.ravel(), METRIC_LABELS, strict=True):
        for index, variant in enumerate(ordered):
            gaps = [
                variants[variant]["evaluation"]["classes"][label]["metrics"][metric_id][
                    "reference_gap"
                ]
                for label in CLASS_LABELS
            ]
            x = np.arange(6) + (index - (len(ordered) - 1) / 2) * 0.035
            color = "#c45b3c" if variant == baseline else "#62757f"
            alpha = 1.0 if variant == baseline else 0.48
            axis.plot(
                x,
                gaps,
                marker="o" if variant == baseline else ".",
                ms=5.0 if variant == baseline else 3.5,
                lw=1.5 if variant == baseline else 0.7,
                color=color,
                alpha=alpha,
            )
        axis.axhspan(0.0, 1.0, color="#dfe7ea", alpha=0.65, zorder=-2)
        axis.axhline(1.0, color="#78909c", lw=0.9, ls="--")
        axis.set_title(METRIC_LABELS[metric_id], loc="left", fontsize=11)
        axis.grid(axis="y", color="#e2e7e9", lw=0.6)
        axis.tick_params(labelsize=9)
    for axis in axes[-1]:
        axis.set_xticks(np.arange(6), CLASS_LABELS)
        axis.set_xlabel("Intensity condition", fontsize=10)
    figure.supylabel("Descriptive real-reference gap", x=0.02, fontsize=11)
    figure.text(
        0.5,
        0.005,
        (
            "Orange: registered 512-field, rank-20 representation. "
            "Grey: EOF fit-size/rank and physical low-pass variants."
        ),
        ha="center",
        fontsize=9,
        color="#455a64",
    )
    figure.tight_layout(rect=(0.035, 0.03, 1.0, 1.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "case_studies" / "tropical_cyclone_bt" / "track1_config.template.yaml",
    )
    parser.add_argument(
        "--functional-cache",
        type=Path,
        default=ROOT / "outputs" / "tropical_cyclone_bt" / "presentation" / "functionals.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "tropical_cyclone_bt" / "audits",
    )
    parser.add_argument("--reference-resamples", type=int, default=100)
    parser.add_argument("--metric-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1789)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    adapter_config = tc_bt.TCBTConfig(**config_payload["adapter_options"])
    train_node = config_payload["data"]["train"]
    real_node = config_payload["data"]["real"]
    generated_node = config_payload["data"]["generated"]
    train = _preprocessed(
        load_real_archive(_resolve(config_path, train_node["tc_real_manifest"]), target="category"),
        adapter_config,
    )
    real = _preprocessed(
        load_real_archive(_resolve(config_path, real_node["tc_real_manifest"]), target="category"),
        adapter_config,
    )
    generated = _preprocessed(
        load_sample_bank(
            _resolve(config_path, generated_node["tc_sample_bank"]),
            target="category",
            required_run=generated_node.get("required_run"),
        ),
        adapter_config,
    )
    cache = np.load(args.functional_cache)
    functionals_real = np.column_stack(
        [np.asarray(cache[f"real_{name}"]) for name in FUNCTIONAL_IDS]
    )
    functionals_generated = np.column_stack(
        [np.asarray(cache[f"generated_{name}"]) for name in FUNCTIONAL_IDS]
    )
    real_labels = np.asarray(cache["real_labels"], dtype=np.int16)
    generated_labels = np.asarray(cache["generated_labels"], dtype=np.int16)
    real_blocks = np.asarray(cache["real_blocks"]).astype(str)
    if not np.array_equal(real_labels, np.asarray(real.labels, dtype=np.int16)):
        raise ValueError("Functional cache real labels do not match the archive.")
    if not np.array_equal(generated_labels, np.asarray(generated.labels, dtype=np.int16)):
        raise ValueError("Functional cache generated labels do not match the bank.")

    functionals = tc_bt.functionals(adapter_config)
    feature_node = config_payload["feature_metrics"]
    bandwidth_seed = int(feature_node.get("bandwidth_seed", args.seed))
    variants: dict[str, Any] = {}
    fit_sizes = (256, 512, 1024)
    ranks = (10, 20, 40)
    spatial_scales_km = (0.0, 16.0, 32.0, 64.0)
    for fit_size in fit_sizes:
        fit_indices = np.sort(
            np.random.default_rng(args.seed).choice(len(train.fields), size=fit_size, replace=False)
        )
        fit_fields = np.asarray(train.fields[fit_indices], dtype=np.float32)
        functionals_train = _functional_matrix(fit_fields, functionals)
        pixel_mean, components, eof_scale = _top_eofs(
            fit_fields, rank=max(ranks), seed=args.seed + fit_size
        )
        eof_train = _project(
            fit_fields,
            mean=pixel_mean,
            components=components,
            scale=eof_scale,
            batch_size=args.batch_size,
            spatial_filter=None,
        )
        eof_real = _project(
            real.fields,
            mean=pixel_mean,
            components=components,
            scale=eof_scale,
            batch_size=args.batch_size,
            spatial_filter=None,
        )
        eof_generated = _project(
            generated.fields,
            mean=pixel_mean,
            components=components,
            scale=eof_scale,
            batch_size=args.batch_size,
            spatial_filter=None,
        )
        for rank in ranks:
            train_features, real_features, generated_features = _combined_features(
                functionals_train=functionals_train,
                functionals_real=functionals_real,
                functionals_generated=functionals_generated,
                eof_train=eof_train,
                eof_real=eof_real,
                eof_generated=eof_generated,
                rank=rank,
            )
            name = f"fit{fit_size}_rank{rank}_scale0km"
            variants[name] = {
                "fit_size": fit_size,
                "rank": rank,
                "low_pass_sigma_km": 0.0,
                "post_hoc": True,
                "evaluation": _evaluate_features(
                    train_features=train_features,
                    real_features=real_features,
                    generated_features=generated_features,
                    real_labels=real_labels,
                    generated_labels=generated_labels,
                    real_blocks=real_blocks,
                    seed=args.seed + fit_size + rank,
                    bandwidth_seed=bandwidth_seed,
                    n_resamples=args.reference_resamples,
                    max_samples=args.metric_samples,
                ),
            }
            print(f"completed {name}", flush=True)

    # Recompute the nominal baseline with the exact public feature-space
    # implementation. The sparse solver above is used only for the post-hoc
    # rank/fit-size perturbations; the orange baseline in the figure therefore
    # represents the same training indices, full-SVD construction, and
    # bandwidth seed as the registered TC evaluation.
    registered_space = TrainFittedFeatureSpace.fit(
        train.fields,
        functionals,
        n_eof=int(feature_node["n_eof"]),
        max_fit_samples=int(feature_node["max_fit_samples"]),
        seed=int(feature_node.get("fit_seed", args.seed)),
        batch_size=args.batch_size,
        fit_source="train",
    )
    registered_fit_fields = train.fields[registered_space.fit_indices]
    variants["fit512_rank20_scale0km"] = {
        "fit_size": int(feature_node["max_fit_samples"]),
        "rank": int(feature_node["n_eof"]),
        "low_pass_sigma_km": 0.0,
        "post_hoc": False,
        "implementation": "exact_registered_train_fitted_feature_space",
        "feature_space_manifest": registered_space.manifest,
        "evaluation": _evaluate_features(
            train_features=registered_space.transform(registered_fit_fields),
            real_features=registered_space.transform(real.fields),
            generated_features=registered_space.transform(generated.fields),
            real_labels=real_labels,
            generated_labels=generated_labels,
            real_blocks=real_blocks,
            seed=args.seed + 512 + 20,
            bandwidth_seed=bandwidth_seed,
            n_resamples=args.reference_resamples,
            max_samples=args.metric_samples,
        ),
    }
    print("completed exact registered baseline", flush=True)

    # Multiscale variants retain the original physical-function features while
    # changing only the spatial EOF component. This isolates scale sensitivity
    # from changes in the definitions of the seven registered functionals.
    fit_size = 512
    rank = 20
    fit_indices = np.sort(
        np.random.default_rng(args.seed).choice(len(train.fields), size=fit_size, replace=False)
    )
    raw_fit_fields = np.asarray(train.fields[fit_indices], dtype=np.float32)
    functionals_train = _functional_matrix(raw_fit_fields, functionals)
    for scale_km in spatial_scales_km[1:]:
        sigma_pixels = scale_km / adapter_config.pixel_size_km

        def low_pass(values: np.ndarray, sigma: float = sigma_pixels) -> np.ndarray:
            return gaussian_filter(values, sigma=(0.0, sigma, sigma), mode="reflect")

        filtered_fit = low_pass(raw_fit_fields)
        pixel_mean, components, eof_scale = _top_eofs(
            filtered_fit,
            rank=rank,
            seed=args.seed + int(scale_km),
        )
        eof_train = _project(
            raw_fit_fields,
            mean=pixel_mean,
            components=components,
            scale=eof_scale,
            batch_size=args.batch_size,
            spatial_filter=low_pass,
        )
        eof_real = _project(
            real.fields,
            mean=pixel_mean,
            components=components,
            scale=eof_scale,
            batch_size=args.batch_size,
            spatial_filter=low_pass,
        )
        eof_generated = _project(
            generated.fields,
            mean=pixel_mean,
            components=components,
            scale=eof_scale,
            batch_size=args.batch_size,
            spatial_filter=low_pass,
        )
        train_features, real_features, generated_features = _combined_features(
            functionals_train=functionals_train,
            functionals_real=functionals_real,
            functionals_generated=functionals_generated,
            eof_train=eof_train,
            eof_real=eof_real,
            eof_generated=eof_generated,
            rank=rank,
        )
        name = f"fit{fit_size}_rank{rank}_scale{int(scale_km)}km"
        variants[name] = {
            "fit_size": fit_size,
            "rank": rank,
            "low_pass_sigma_km": scale_km,
            "low_pass_sigma_pixels": sigma_pixels,
            "post_hoc": True,
            "evaluation": _evaluate_features(
                train_features=train_features,
                real_features=real_features,
                generated_features=generated_features,
                real_labels=real_labels,
                generated_labels=generated_labels,
                real_blocks=real_blocks,
                seed=args.seed + int(scale_km),
                bandwidth_seed=bandwidth_seed,
                n_resamples=args.reference_resamples,
                max_samples=args.metric_samples,
            ),
        }
        print(f"completed {name}", flush=True)

    baseline = variants["fit512_rank20_scale0km"]["evaluation"]["classes"]
    for variant in variants.values():
        comparisons = {}
        for metric_id in METRIC_LABELS:
            baseline_gaps = np.asarray(
                [baseline[label]["metrics"][metric_id]["reference_gap"] for label in CLASS_LABELS],
                dtype=float,
            )
            variant_gaps = np.asarray(
                [
                    variant["evaluation"]["classes"][label]["metrics"][metric_id]["reference_gap"]
                    for label in CLASS_LABELS
                ],
                dtype=float,
            )
            finite = np.isfinite(baseline_gaps) & np.isfinite(variant_gaps)
            if np.sum(finite) < 2:
                correlation = None
            else:
                correlation = float(np.corrcoef(baseline_gaps[finite], variant_gaps[finite])[0, 1])
            comparisons[metric_id] = {
                "class_pattern_correlation_with_registered": correlation,
                "median_absolute_gap_change": float(
                    np.nanmedian(np.abs(variant_gaps - baseline_gaps))
                ),
            }
        variant["comparison_with_registered"] = comparisons

    payload = {
        "schema": "gauge_climate.tc_bt_feature_robustness.v1",
        "status": "POST_HOC_ROBUSTNESS_ANALYSIS",
        "interpretation": (
            "No verdict is attached to these variants. They assess whether the "
            "class-wise feature-metric pattern depends materially on EOF fit "
            "sample count, rank, or the spatial scales retained by the EOFs."
        ),
        "registered_variant": "fit512_rank20_scale0km",
        "methods": {
            "eof_solver": "scipy sparse truncated SVD (largest singular triplets)",
            "registered_baseline_solver": ("exact gauge_climate TrainFittedFeatureSpace full SVD"),
            "fit_source": "training split only",
            "physical_features": (
                "The seven registered physical functionals are unchanged across "
                "low-pass variants; only the EOF component is low-pass filtered."
            ),
            "reference_resamples": args.reference_resamples,
            "metric_sample_cap": args.metric_samples,
            "low_pass_scale_definition": "Gaussian standard deviation in kilometres",
        },
        "inputs": {
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "functional_cache": str(args.functional_cache.resolve()),
            "functional_cache_sha256": _sha256(args.functional_cache.resolve()),
        },
        "variants": variants,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "feature_space_robustness.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _plot(payload, output_dir / "figure_A_feature_space_robustness")
    print(json_path)


if __name__ == "__main__":
    main()
