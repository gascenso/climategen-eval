#!/usr/bin/env python3
"""Run the frozen training-only TC Track-1 statistical calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gauge_climate.adapters import tc_bt  # noqa: E402
from gauge_climate.adapters.tc_data import load_real_archive  # noqa: E402
from gauge_climate.inference import (  # noqa: E402
    cluster_w1_influence_interval,
    dkw_w1_interval,
    shift_resolution,
    simultaneous_reference_boundaries,
)
from gauge_climate.metrics import functional_w1  # noqa: E402
from gauge_climate.reference import (  # noqa: E402
    build_reference_plan,
    draw_real_indices,
    evaluate_reference_plan,
    reference_position,
)
from gauge_climate.types import EvaluationConfig, SamplingEstimand  # noqa: E402


def _yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return payload


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_sha256(root: Path) -> str:
    """Hash the complete importable GAUGE source tree, including file names."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _array_hash(values: np.ndarray) -> str:
    array = np.asarray(values)
    if array.dtype.kind in {"O", "U", "S"}:
        encoded = json.dumps(
            [str(value) for value in array.tolist()], separators=(",", ":")
        ).encode()
    else:
        encoded = array.tobytes()
    return hashlib.sha256(encoded).hexdigest()


def _derive_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _resolve(protocol_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (protocol_path.parent / path).resolve()


def _stable_blocks(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        list(dict.fromkeys(str(value) for value in np.asarray(values).tolist())),
        dtype=object,
    )


def _partition_counts(summary: Mapping[str, Any]) -> dict[int, int]:
    return {
        int(row["class_id"]): int(row["storms_with_frames"])
        for row in summary.get("rows", [])
        if row.get("split") == "test"
    }


def _select_strata(
    *,
    labels: np.ndarray,
    blocks: np.ndarray,
    specifications: Mapping[str, Any],
    seed: int,
) -> dict[str, dict[str, Any]]:
    selected = {}
    block_strings = np.asarray([str(value) for value in blocks], dtype=object)
    for stratum, node in specifications.items():
        members = node.get("labels")
        label_mask = (
            np.ones(labels.shape[0], dtype=bool)
            if members is None
            else np.isin(labels, np.asarray(members))
        )
        eligible_blocks = np.sort(np.unique(block_strings[label_mask]))
        count = int(node["calibration_blocks"])
        if count < 2 or 2 * count > eligible_blocks.size:
            raise ValueError(
                f"Stratum {stratum} requests two disjoint {count}-block pools "
                f"from {eligible_blocks.size} eligible blocks."
            )
        rng = np.random.default_rng(_derive_seed(seed, f"blocks:{stratum}"))
        chosen = rng.choice(eligible_blocks, size=2 * count, replace=False)
        reference_blocks = np.sort(chosen[:count])
        pseudo_blocks = np.sort(chosen[count:])
        indices = np.flatnonzero(label_mask & np.isin(block_strings, reference_blocks)).astype(
            np.int64
        )
        pseudo_indices = np.flatnonzero(label_mask & np.isin(block_strings, pseudo_blocks)).astype(
            np.int64
        )
        if np.unique(block_strings[indices]).size != count:
            raise RuntimeError(f"Stratum {stratum} block selection lost eligible blocks.")
        if np.unique(block_strings[pseudo_indices]).size != count:
            raise RuntimeError(
                f"Stratum {stratum} pseudo-generator selection lost eligible blocks."
            )
        selected[str(stratum)] = {
            "labels": None if members is None else [int(value) for value in members],
            "blocks": reference_blocks,
            "indices": indices,
            "pseudo_blocks": pseudo_blocks,
            "pseudo_indices": pseudo_indices,
            "n_eval": int(node["n_eval"]),
        }
    return selected


def _transform_union(
    fields: Any,
    union_indices: np.ndarray,
    functionals: tuple[Any, ...],
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    output = {
        functional.id: np.empty(union_indices.size, dtype=np.float64) for functional in functionals
    }
    for start in range(0, union_indices.size, batch_size):
        stop = min(start + batch_size, union_indices.size)
        batch = np.asarray(fields[union_indices[start:stop]], dtype=np.float32)
        if not np.all(np.isfinite(batch)):
            raise ValueError("Calibration training fields contain non-finite values.")
        for functional in functionals:
            output[functional.id][start:stop] = functional.transform(batch)
        if start == 0 or stop == union_indices.size or stop % (10 * batch_size) == 0:
            print(
                f"transformed {stop}/{union_indices.size} calibration frames",
                flush=True,
            )
    return output


def _cached_transforms(
    *,
    cache_path: Path | None,
    cache_key: str,
    fields: Any,
    union_indices: np.ndarray,
    functionals: tuple[Any, ...],
    batch_size: int,
) -> dict[str, np.ndarray]:
    if cache_path is not None and cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            observed_key = str(np.asarray(cached["cache_key"]).item())
            if observed_key == cache_key:
                values = {
                    functional.id: np.asarray(cached[functional.id], dtype=np.float64)
                    for functional in functionals
                }
                if all(value.shape == (union_indices.size,) for value in values.values()):
                    print(f"loaded validated transform cache {cache_path}", flush=True)
                    return values
    values = _transform_union(
        fields,
        union_indices,
        functionals,
        batch_size=batch_size,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, cache_key=np.asarray(cache_key), **values)
        print(f"wrote transform cache {cache_path}", flush=True)
    return values


def _check_partition_summary(protocol: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    if (
        summary.get("wind_source") != "USA_WIND"
        or summary.get("wind_averaging_period") != "1-minute"
    ):
        raise ValueError("Partition summary lacks corrected USA_WIND provenance.")
    counts = _partition_counts(summary)
    strata = protocol["strata"]
    for class_id in range(6):
        declared = int(strata[str(class_id)]["calibration_blocks"])
        if declared != counts[class_id]:
            raise ValueError(
                f"Stratum {class_id} calibration blocks differ from the frozen "
                "published test block count."
            )
    if int(strata["global"]["calibration_blocks"]) != counts[0]:
        raise ValueError("Global conservative block count must equal the TS lower bound.")
    if int(strata["C4_plus"]["calibration_blocks"]) != counts[4]:
        raise ValueError("C4+ conservative block count must equal the C4 lower bound.")


def _plot(payload: Mapping[str, Any], path: Path) -> None:
    strata = payload["family"]["strata"]
    functionals = payload["family"]["functionals"]
    cells = {cell["key"]: cell for cell in payload["cells"]}
    resolution = np.asarray(
        [
            [
                cells[f"{stratum}:w1:{functional}"]["resolution_to_tolerance_ratio"]
                for functional in functionals
            ]
            for stratum in strata
        ],
        dtype=float,
    )
    stability = np.asarray(
        [
            [
                cells[f"{stratum}:w1:{functional}"]["reference_stability_ratio"]
                for functional in functionals
            ]
            for stratum in strata
        ],
        dtype=float,
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    first = axes[0].imshow(resolution, aspect="auto", cmap="viridis")
    axes[0].set_title("Shift resolution / tolerance")
    figure.colorbar(first, ax=axes[0], shrink=0.75)
    second = axes[1].imshow(stability, aspect="auto", cmap="magma", vmin=0.0)
    axes[1].set_title("Reference split-half instability")
    axes[1].axhline(-1, color="none")
    figure.colorbar(second, ax=axes[1], shrink=0.75)
    for axis in axes[:2]:
        axis.set_yticks(np.arange(len(strata)), strata)
        axis.set_xticks(np.arange(len(functionals)), functionals, rotation=40, ha="right")
    positions = np.asarray(
        [cell["pseudo_generator_reference_position"] for cell in payload["cells"]]
    )
    boundaries = np.asarray(payload["family"]["simultaneous_boundaries"])
    order = np.arange(positions.size)
    axes[2].scatter(order, positions, s=14, label="pseudo-generator")
    axes[2].plot(order, boundaries, linewidth=1, label="simultaneous boundary")
    axes[2].set_xlabel("registered cell")
    axes[2].set_ylabel("reference-position diagnostic")
    axes[2].set_title("56-cell simultaneous calibration")
    axes[2].legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)


def run(protocol_path: Path, *, transform_cache: Path | None = None) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    protocol = _yaml(protocol_path)
    if (
        protocol.get("schema") != ("gauge_climate.tc_bt_statistical_calibration_protocol.v1")
        or protocol.get("status") != "FROZEN"
    ):
        raise ValueError("Unsupported or unfrozen TC calibration protocol.")
    source = protocol["source"]
    train_manifest = _resolve(protocol_path, source["real_train_manifest"])
    summary_path = _resolve(protocol_path, source["partition_summary"])
    track1_path = _resolve(protocol_path, source["track1_config"])
    summary = _json(summary_path)
    track1 = _yaml(track1_path)
    _check_partition_summary(protocol, summary)
    train = load_real_archive(train_manifest, target="category")
    labels = np.asarray(train.labels)
    blocks = np.asarray(train.blocks)
    seed = int(protocol["seed"])
    selected = _select_strata(
        labels=labels,
        blocks=blocks,
        specifications=protocol["strata"],
        seed=seed,
    )
    union_indices = np.unique(
        np.concatenate(
            [
                indices
                for node in selected.values()
                for indices in (node["indices"], node["pseudo_indices"])
            ]
        )
    ).astype(np.int64)
    union_position = np.full(train.n_samples, -1, dtype=np.int64)
    union_position[union_indices] = np.arange(union_indices.size, dtype=np.int64)
    adapter = tc_bt.TCBTConfig(**track1.get("adapter_options", {}))
    if adapter.field_shape != train.sample_shape:
        raise ValueError("Track-1 adapter shape differs from the corrected training archive.")
    functionals = tc_bt.functionals(adapter)
    expected_functionals = tuple(protocol["simultaneous_family"]["functionals"])
    if tuple(functional.id for functional in functionals) != expected_functionals:
        raise ValueError("Calibration functional registry differs from the frozen family.")
    gauge_source_sha256 = _source_tree_sha256(SRC / "gauge_climate")
    cache_material = {
        "union_index_sha256": _array_hash(union_indices),
        "real_train_manifest_sha256": _sha256(train_manifest),
        "track1_config_sha256": _sha256(track1_path),
        "gauge_source_sha256": gauge_source_sha256,
        "functionals": list(expected_functionals),
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    values = _cached_transforms(
        cache_path=transform_cache,
        cache_key=cache_key,
        fields=train.fields,
        union_indices=union_indices,
        functionals=functionals,
        batch_size=int(protocol["reference"]["transform_batch_size"]),
    )
    metrics = {functional.id: functional_w1(functional) for functional in functionals}
    track1_tolerances = {
        str(key): float(value) for key, value in track1.get("tolerances", {}).items()
    }
    protocol_tolerances = {str(key): float(value) for key, value in protocol["tolerances"].items()}
    if track1_tolerances != protocol_tolerances:
        raise ValueError("Calibration tolerances differ from the frozen Track-1 config.")
    reference_node = protocol["reference"]
    config_common = {
        "n_resamples": int(reference_node["n_resamples"]),
        "bootstrap_resamples": 20,
        "sensitivity_resamples": int(reference_node["sensitivity_resamples"]),
        "alpha": float(reference_node["alpha"]),
        "confidence": 1.0 - float(reference_node["alpha"]),
        "estimand": SamplingEstimand.BLOCK_BALANCED,
        "seed": seed,
        "min_blocks_per_half": int(reference_node["min_blocks_per_half"]),
        "warn_blocks_per_half": int(reference_node["min_blocks_per_half"]),
        "shift_power": float(reference_node["target_power"]),
        "shift_grid_size": int(reference_node["shift_grid_size"]),
        "reference_stability_tolerance": float(reference_node["reference_stability_tolerance"]),
        "include_partitions": False,
    }
    cells = []
    reference_rows = []
    plan_failures = 0
    pseudo_reference_overlap_failures = 0
    for stratum in protocol["simultaneous_family"]["strata"]:
        stratum = str(stratum)
        node = selected[stratum]
        original_indices = node["indices"]
        local_union = union_position[original_indices]
        if np.any(local_union < 0):
            raise RuntimeError("Calibration union index mapping failed.")
        local_blocks = np.asarray(blocks)[original_indices]
        allowed_blocks = _stable_blocks(local_blocks)
        n_eval = int(node["n_eval"])
        pseudo_original_indices = node["pseudo_indices"]
        pseudo_blocks = np.asarray(blocks)[pseudo_original_indices]
        pseudo_allowed_blocks = _stable_blocks(pseudo_blocks)
        if set(allowed_blocks.tolist()) & set(pseudo_allowed_blocks.tolist()):
            pseudo_reference_overlap_failures += 1
        pseudo_local = draw_real_indices(
            np.arange(pseudo_original_indices.size, dtype=np.int64),
            pseudo_blocks,
            pseudo_allowed_blocks.tolist(),
            n=n_eval,
            estimand=SamplingEstimand.BLOCK_BALANCED,
            rng=np.random.default_rng(_derive_seed(seed, f"pseudo:{stratum}")),
        )
        pseudo_union = union_position[pseudo_original_indices[pseudo_local]]
        if np.any(pseudo_union < 0):
            raise RuntimeError("Calibration pseudo-generator union mapping failed.")
        config = EvaluationConfig(n_eval=n_eval, **config_common)
        plan = build_reference_plan(
            real_indices=np.arange(original_indices.size, dtype=np.int64),
            real_blocks=local_blocks,
            n_generated=n_eval,
            config=config,
            seed=_derive_seed(seed, f"reference:{stratum}"),
        )
        normalized_blocks = np.asarray([str(value) for value in local_blocks], dtype=object)
        for draw in plan.draws:
            if not set(draw.blocks_a).isdisjoint(draw.blocks_b):
                plan_failures += 1
            if not set(normalized_blocks[draw.pseudo_indices]).issubset(draw.blocks_a):
                plan_failures += 1
            if not set(normalized_blocks[draw.real_indices]).issubset(draw.blocks_b):
                plan_failures += 1
        for functional in functionals:
            functional_values = values[functional.id][local_union]
            pseudo_values = values[functional.id][pseudo_union]
            metric = metrics[functional.id]
            evaluated = evaluate_reference_plan(
                metric,
                real_values=functional_values,
                generated_values=pseudo_values,
                plan=plan,
            )
            boundary = evaluated.quantile(1.0 - config.alpha)
            midpoint = len(evaluated.reference_values) // 2
            first = float(np.quantile(evaluated.reference_values[:midpoint], 1.0 - config.alpha))
            second = float(np.quantile(evaluated.reference_values[midpoint:], 1.0 - config.alpha))
            reference_scale = max(boundary - evaluated.reference_q50, np.finfo(float).eps)
            stability = abs(first - second) / reference_scale
            tolerance = float(protocol["tolerances"][functional.id])
            resolution, shifts, powers = shift_resolution(
                real_values=functional_values,
                plan=plan,
                reference_boundary=boundary,
                target_power=config.shift_power,
                grid_size=config.shift_grid_size,
                required_shifts=np.asarray([tolerance], dtype=np.float64),
                max_draws=config.sensitivity_resamples,
            )
            tolerance_index = int(np.flatnonzero(shifts == tolerance)[0])
            tolerance_power = float(powers[tolerance_index])
            if functional.support is None:
                raise ValueError(f"Functional {functional.id} lacks finite support.")
            dkw = dkw_w1_interval(
                pseudo_values,
                functional_values,
                local_blocks,
                n_eval=n_eval,
                support=functional.support,
                alpha=config.alpha,
                seed=_derive_seed(seed, f"dkw:{stratum}:{functional.id}"),
            )
            if (
                functional.provenance.get("practical_inference")
                == "detection_only_discrete_functional"
            ):
                direct = dkw
            else:
                direct = cluster_w1_influence_interval(
                    pseudo_values,
                    functional_values,
                    local_blocks,
                    n_eval=n_eval,
                    alpha=config.alpha,
                    seed=_derive_seed(seed, f"direct:{stratum}:{functional.id}"),
                    fallback_support=functional.support,
                )
            key = f"{stratum}:w1:{functional.id}"
            cell = {
                "key": key,
                "stratum": stratum,
                "functional": functional.id,
                "units": functional.units,
                "n_eval": n_eval,
                "n_real_frames": int(original_indices.size),
                "n_real_blocks": int(allowed_blocks.size),
                "reference_q50": evaluated.reference_q50,
                "reference_q95": boundary,
                "reference_stability_ratio": float(stability),
                "pseudo_generator_model_median": evaluated.model_median,
                "pseudo_generator_reference_position": reference_position(
                    evaluated.model_median, evaluated.reference_q50, boundary
                ),
                "pseudo_generator_pointwise_detection": bool(evaluated.model_median > boundary),
                "dkw_equality_interval": {
                    "estimate": dkw.estimate,
                    "low": dkw.low,
                    "high": dkw.high,
                },
                "direct_equality_interval": {
                    "method": direct.method,
                    "estimate": direct.estimate,
                    "low": direct.low,
                    "high": direct.high,
                    "standard_error": direct.standard_error,
                    "degrees_of_freedom": direct.degrees_of_freedom,
                    "cdf_zero_fraction": direct.cdf_zero_fraction,
                    "within_tolerance": bool(direct.high < tolerance),
                },
                "tolerance": tolerance,
                "shift_resolution": resolution,
                "detection_probability_at_tolerance": tolerance_power,
                "resolution_to_tolerance_ratio": (
                    None if resolution is None else float(resolution / tolerance)
                ),
                "tolerance_resolution_adequate": bool(tolerance_power >= config.shift_power),
                "shift_grid": shifts.tolist(),
                "shift_detection_probability": powers.tolist(),
            }
            cells.append(cell)
            reference_rows.append(evaluated.reference_values)
        print(f"calibrated stratum {stratum}", flush=True)
        del plan
    expected_keys = [
        f"{str(stratum)}:w1:{functional}"
        for stratum in protocol["simultaneous_family"]["strata"]
        for functional in expected_functionals
    ]
    if [cell["key"] for cell in cells] != expected_keys:
        raise RuntimeError("Calibration cells differ from the frozen simultaneous family.")
    reference_matrix = np.stack(reference_rows)
    simultaneous = simultaneous_reference_boundaries(
        reference_matrix, alpha=float(reference_node["alpha"])
    )
    positions = np.asarray(
        [cell["pseudo_generator_reference_position"] for cell in cells], dtype=float
    )
    simultaneous_detections = positions > simultaneous
    for cell, boundary, detected in zip(cells, simultaneous, simultaneous_detections, strict=True):
        cell["simultaneous_boundary"] = float(boundary)
        cell["pseudo_generator_simultaneous_detection"] = bool(detected)
    stability_tolerance = float(reference_node["reference_stability_tolerance"])
    gates = {
        "reference_plan_invariants": plan_failures == 0,
        "pseudo_generator_reference_blocks_disjoint": (pseudo_reference_overlap_failures == 0),
        "finite_reference_boundaries": all(np.isfinite(cell["reference_q95"]) for cell in cells),
        "reference_stability": all(
            cell["reference_stability_ratio"] <= stability_tolerance for cell in cells
        ),
        "finite_shift_resolution": all(
            cell["shift_resolution"] is not None and np.isfinite(cell["shift_resolution"])
            for cell in cells
        ),
        "dkw_equality_lower_bound_zero": all(
            np.isclose(cell["dkw_equality_interval"]["low"], 0.0) for cell in cells
        ),
        "finite_direct_equality_intervals": all(
            np.isfinite(cell["direct_equality_interval"]["estimate"])
            and np.isfinite(cell["direct_equality_interval"]["low"])
            and np.isfinite(cell["direct_equality_interval"]["high"])
            for cell in cells
        ),
        "finite_simultaneous_boundaries": bool(np.all(np.isfinite(simultaneous))),
        "registered_pseudo_generator_no_simultaneous_detection": not bool(
            np.any(simultaneous_detections)
        ),
    }
    return {
        "schema": "gauge_climate.tc_bt_statistical_calibration.v1",
        "status": "PASSED" if all(gates.values()) else "FAILED",
        "_metadata": {
            "scientific_use": "training-only pre-final statistical calibration",
            "test_fields_used": False,
            "test_metadata_used": "published stratum block counts only",
            "protocol": str(protocol_path.relative_to(ROOT)),
            "protocol_sha256": _sha256(protocol_path),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "gauge_source_sha256": gauge_source_sha256,
            "real_train_manifest": str(train_manifest.relative_to(ROOT)),
            "real_train_manifest_sha256": _sha256(train_manifest),
            "partition_summary": str(summary_path.relative_to(ROOT)),
            "partition_summary_sha256": _sha256(summary_path),
            "track1_config": str(track1_path.relative_to(ROOT)),
            "track1_config_sha256": _sha256(track1_path),
            "union_frame_count": int(union_indices.size),
            "union_index_sha256": _array_hash(union_indices),
            "functional_value_sha256": {key: _array_hash(value) for key, value in values.items()},
            "selected_blocks": {
                stratum: {
                    "reference": {
                        "count": int(node["blocks"].size),
                        "ids": node["blocks"].tolist(),
                        "sha256": _array_hash(node["blocks"]),
                    },
                    "pseudo_generator": {
                        "count": int(node["pseudo_blocks"].size),
                        "ids": node["pseudo_blocks"].tolist(),
                        "sha256": _array_hash(node["pseudo_blocks"]),
                    },
                    "overlap_count": len(
                        set(node["blocks"].tolist()) & set(node["pseudo_blocks"].tolist())
                    ),
                }
                for stratum, node in selected.items()
            },
        },
        "family": {
            "method": protocol["simultaneous_family"]["method"],
            "size": len(cells),
            "strata": [str(value) for value in protocol["simultaneous_family"]["strata"]],
            "functionals": list(expected_functionals),
            "simultaneous_boundaries": simultaneous.tolist(),
            "pseudo_generator_simultaneous_detection_count": int(simultaneous_detections.sum()),
            "pseudo_generator_pointwise_detection_count": int(
                sum(cell["pseudo_generator_pointwise_detection"] for cell in cells)
            ),
        },
        "summary": {
            "adequate_tolerance_resolution_cells": int(
                sum(cell["tolerance_resolution_adequate"] for cell in cells)
            ),
            "direct_equality_within_tolerance_cells": int(
                sum(
                    cell["direct_equality_interval"]["within_tolerance"]
                    for cell in cells
                )
            ),
            "total_cells": len(cells),
            "maximum_reference_stability_ratio": float(
                max(cell["reference_stability_ratio"] for cell in cells)
            ),
            "reference_stability_tolerance": stability_tolerance,
            "plan_invariant_failures": plan_failures,
            "pseudo_reference_overlap_failures": (pseudo_reference_overlap_failures),
        },
        "gates": gates,
        "cells": cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "case_studies/tropical_cyclone_bt/calibration_protocol.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT / "case_studies/tropical_cyclone_bt/calibration_results/track1_statistics.json"
        ),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=(
            ROOT / "case_studies/tropical_cyclone_bt/calibration_results/track1_statistics.pdf"
        ),
    )
    parser.add_argument(
        "--transform-cache",
        type=Path,
        default=Path("/tmp/gauge_tc_track1_calibration_transforms.npz"),
        help="Local restart cache; validated against data, indices, config, and registry.",
    )
    args = parser.parse_args()
    payload = run(args.protocol, transform_cache=args.transform_cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _plot(payload, args.figure)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(args.output),
                "figure": str(args.figure),
                "gates": payload["gates"],
                "summary": payload["summary"],
            },
            indent=2,
        )
    )
    return 0 if payload["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
