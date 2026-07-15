#!/usr/bin/env python3
"""Run the leakage-audited eight-arm TC downstream experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from gauge_climate.adapters.tc_data import load_real_archive, load_sample_bank
from gauge_climate.downstream import Arm, Holdout, Ladder, LadderConfig

from .downstream_cnn import TCCNNConfig, TCCNNTask, tc_geometry


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) if path.suffix in {".yaml", ".yml"} else json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return payload


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha256(value: Any, *, field: str, holdout: str) -> str:
    text = str(value or "")
    try:
        valid = len(text) == 64 and int(text, 16) >= 0
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(
            f"Generator provenance for {holdout} has invalid {field}."
        )
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _load(config_path)
    if config.get("schema") != "gauge_climate.tc_track2_config.v1":
        raise ValueError("Unsupported TC Track-2 config schema.")
    base = config_path.parent
    data = config["data"]
    train = load_real_archive(
        _resolve(base, data["real_train_manifest"]), target="wind"
    )
    validation = load_real_archive(
        _resolve(base, data["real_val_manifest"]), target="wind"
    )

    holdout_plan = _load(_resolve(base, config["holdout_plan"]))
    if holdout_plan.get("schema") != "gauge_climate.tc_bt_track2_holdouts.v1":
        raise ValueError("Unsupported TC Track-2 holdout plan schema.")
    source = holdout_plan.get("source", {})
    if source.get("wind_source") != "USA_WIND" or source.get(
        "wind_averaging_period"
    ) != "1-minute":
        raise ValueError("Track-2 holdouts must come from the corrected USA_WIND archive.")
    holdouts = tuple(
        Holdout(str(node["id"]), tuple(str(value) for value in node["storms"]))
        for node in holdout_plan["holdouts"]
    )
    if len(holdouts) != 3:
        raise ValueError("The final TC Track-2 protocol requires exactly three holdouts.")
    configured = config.get("generated_by_holdout", {})
    generated = {}
    provenance = {}
    generator_artifacts = {}
    train_blocks = np.asarray([str(value) for value in train.blocks], dtype=object)
    train_winds = np.asarray(train.labels, dtype=np.float32)
    for holdout in holdouts:
        if holdout.id not in configured:
            raise ValueError(f"Missing generated/provenance artifacts for {holdout.id}.")
        node = configured[holdout.id]
        bank = load_sample_bank(
            _resolve(base, node["bank_path"]), target="wind", required_run=None
        )
        if bank.metadata.get("split") != "train":
            raise ValueError(f"Track-2 bank for {holdout.id} must target the train split.")
        provenance_payload = _load(_resolve(base, node["generator_provenance"]))
        if provenance_payload.get("schema") != "gauge_climate.tc_generator_fit.v1":
            raise ValueError(f"Unsupported generator provenance for {holdout.id}.")
        if provenance_payload.get("holdout_id") != holdout.id:
            raise ValueError(f"Generator provenance does not identify {holdout.id}.")
        if provenance_payload.get("model_family") != "baseline_ztsnr":
            raise ValueError(
                f"Generator provenance for {holdout.id} is not a baseline_ztsnr retrain."
            )
        if provenance_payload.get("run_name") != bank.metadata.get("run_name"):
            raise ValueError(
                f"Generator run name differs from the sample bank for {holdout.id}."
            )
        if provenance_payload.get("wind_source") != "USA_WIND" or provenance_payload.get(
            "wind_averaging_period"
        ) != "1-minute":
            raise ValueError(
                f"Generator provenance for {holdout.id} lacks the USA_WIND/1-minute contract."
            )
        if set(provenance_payload.get("withheld_storms", [])) != set(holdout.blocks):
            raise ValueError(f"Generator provenance has incorrect withheld storms for {holdout.id}.")
        required_hashes = (
            "restricted_split_sha256",
            "generator_config_sha256",
            "checkpoint_sha256",
            "sample_bank_manifest_sha256",
        )
        hashes = {
            key: _require_sha256(
                provenance_payload.get(key), field=key, holdout=holdout.id
            )
            for key in required_hashes
        }
        bank_manifest = Path(str(bank.metadata["manifest_path"]))
        if hashes["sample_bank_manifest_sha256"] != _sha256(bank_manifest):
            raise ValueError(
                f"Sample-bank manifest hash differs from provenance for {holdout.id}."
            )
        restricted_mask = ~np.isin(train_blocks, np.asarray(holdout.blocks, dtype=object))
        if not np.all(np.isin(np.asarray(bank.labels), train_winds[restricted_mask])):
            raise ValueError(
                f"Generated wind conditions for {holdout.id} include values absent "
                "from its permitted restricted training distribution."
            )
        generated[holdout.id] = bank
        provenance[holdout.id] = tuple(
            str(value) for value in provenance_payload.get("training_storms", [])
        )
        generator_artifacts[holdout.id] = hashes

    cnn_node = dict(config.get("cnn", {}))
    if "stage_channels" in cnn_node:
        cnn_node["stage_channels"] = tuple(int(value) for value in cnn_node["stage_channels"])
    cnn_config = TCCNNConfig(**cnn_node)
    holdout_blocks = {holdout.id: holdout.blocks for holdout in holdouts}
    task = TCCNNTask(
        full_training_archive=train,
        validation_archive=validation,
        holdout_blocks=holdout_blocks,
        output_root=_resolve(base, config["outputs"]["runs"]),
        config=cnn_config,
    )
    ladder_node = dict(config.get("ladder", {}))
    ladder_node["seeds"] = tuple(
        int(value)
        for value in ladder_node.get("seeds", (101, 102, 103, 104, 105))
    )
    if len(ladder_node["seeds"]) != 5:
        raise ValueError("The final TC Track-2 protocol requires five paired seeds.")
    margins = config.get("task_margins", {})
    try:
        vmax_margin = float(margins["vmax_mae_kt"])
        category_margin = float(margins["category_accuracy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Freeze numeric Vmax and category task margins before running.") from exc
    if vmax_margin <= 0 or category_margin <= 0:
        raise ValueError("Track-2 task margins must be positive.")
    edges = np.asarray(config["matching"]["wind_bin_edges_kt"], dtype=np.float32)
    if edges.shape != (5,) or not np.all(np.diff(edges) > 0):
        raise ValueError("TC Track-2 requires five increasing wind-bin edges.")

    report = Ladder(
        task=task,
        task_metadata={
            **task.metadata,
            "protocol_sources": {
                "config_sha256": _sha256(config_path),
                "holdout_plan_sha256": _sha256(
                    _resolve(base, config["holdout_plan"])
                ),
                "generator_artifacts": generator_artifacts,
            },
        },
        train=train,
        holdouts=holdouts,
        generated_by_holdout=generated,
        generator_training_blocks_by_holdout=provenance,
        transform=tc_geometry,
        transform_metadata={
            "policy": "tc_rot90_reflect_shift_v1",
            "rotations": "right angles",
            "translations_pixels": [-2, 2],
            "boundary": "reflect padding",
            "applies_to": "matched added samples only",
        },
        matching_strata=lambda values: np.digitize(values, edges),
        matching_metadata={"policy": "frozen_vmax_bins", "edges_kt": edges.tolist()},
        config=LadderConfig(**ladder_node),
    ).run()
    output = _resolve(base, config["outputs"]["report"])
    report.to_json(output)
    contrasts = {
        "vmax_mae_kt": {
            "headline": report.contrast(
                metric="vmax_mae_kt",
                treatment=Arm.A5_GENERATED_GEO,
                reference=Arm.A3_COPY_GEO,
                delta_task=vmax_margin,
                higher_is_better=False,
            ),
            "source_only": report.contrast(
                metric="vmax_mae_kt",
                treatment=Arm.A4_GENERATED,
                reference=Arm.A2_COPY,
                delta_task=vmax_margin,
                higher_is_better=False,
            ),
            "headroom": report.contrast(
                metric="vmax_mae_kt",
                treatment=Arm.A7_RESTORED_REAL_GEO,
                reference=Arm.A3_COPY_GEO,
                delta_task=vmax_margin,
                higher_is_better=False,
            ),
        },
        "category_accuracy": {
            "headline": report.contrast(
                metric="category_accuracy",
                treatment=Arm.A5_GENERATED_GEO,
                reference=Arm.A3_COPY_GEO,
                delta_task=category_margin,
                higher_is_better=True,
            ),
            "source_only": report.contrast(
                metric="category_accuracy",
                treatment=Arm.A4_GENERATED,
                reference=Arm.A2_COPY,
                delta_task=category_margin,
                higher_is_better=True,
            ),
            "headroom": report.contrast(
                metric="category_accuracy",
                treatment=Arm.A7_RESTORED_REAL_GEO,
                reference=Arm.A3_COPY_GEO,
                delta_task=category_margin,
                higher_is_better=True,
            ),
        },
    }
    contrast_path = output.with_name(f"{output.stem}_contrasts.json")
    contrast_path.write_text(json.dumps(contrasts, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "contrasts": str(contrast_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
