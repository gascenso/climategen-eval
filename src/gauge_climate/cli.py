from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .adapters import tc_bt
from .adapters.tc_data import load_real_archive, load_sample_bank
from .data import Dataset, validate_generated, validate_split
from .evaluation import evaluate
from .features import TrainFittedFeatureSpace
from .memorization import localized_copying_test
from .metrics import admissibility_metric, spearman_pair_metrics
from .report import EvaluationReport
from .types import EvaluationConfig, FunctionalSpec


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    return payload


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.parent / path


def _load_array(config_path: Path, value: str | None, *, mmap: bool = True):
    if value is None:
        return None
    path = _resolve(config_path, value)
    if path.suffix != ".npy":
        raise ValueError(f"GAUGE CLI array inputs must be .npy files, got {path}.")
    return np.load(path, mmap_mode="r" if mmap else None)


def _dataset(config_path: Path, node: dict[str, Any], name: str) -> Dataset:
    if not isinstance(node, dict):
        raise ValueError(f"data.{name} must be a mapping.")
    sources = [key for key in ("fields", "tc_real_manifest", "tc_sample_bank") if key in node]
    if len(sources) != 1:
        raise ValueError(
            f"data.{name} must provide exactly one of fields, tc_real_manifest, or tc_sample_bank."
        )
    target = str(node.get("target", "category"))
    if sources[0] == "tc_real_manifest":
        return load_real_archive(
            _resolve(config_path, str(node["tc_real_manifest"])),
            target=target,
        )
    if sources[0] == "tc_sample_bank":
        required_run = node.get("required_run", "baseline_ztsnr")
        return load_sample_bank(
            _resolve(config_path, str(node["tc_sample_bank"])),
            target=target,
            required_run=None if required_run is None else str(required_run),
        )
    return Dataset(
        _load_array(config_path, str(node["fields"])),
        blocks=_load_array(config_path, node.get("blocks")),
        labels=_load_array(config_path, node.get("labels")),
        sample_ids=_load_array(config_path, node.get("sample_ids")),
        name=name,
        units=node.get("units"),
        allow_nan=bool(node.get("allow_nan", False)),
        metadata=node.get("metadata", {}),
    )


def _builtin_functional(node: dict[str, Any]) -> FunctionalSpec:
    identifier = str(node["id"])
    builtin = str(node["builtin"])
    units = str(node["units"])
    if builtin == "mean":
        function = lambda fields: np.mean(fields, axis=tuple(range(1, fields.ndim)))
    elif builtin == "minimum":
        function = lambda fields: np.min(fields, axis=tuple(range(1, fields.ndim)))
    elif builtin == "maximum":
        function = lambda fields: np.max(fields, axis=tuple(range(1, fields.ndim)))
    elif builtin == "quantile":
        quantile = float(node["quantile"])
        if not 0 <= quantile <= 1:
            raise ValueError("Functional quantile must be between zero and one.")
        function = lambda fields, q=quantile: np.quantile(
            fields.reshape(fields.shape[0], -1), q, axis=1
        )
    elif builtin == "fraction_below":
        threshold = float(node["threshold"])
        function = lambda fields, t=threshold: np.mean(
            fields.reshape(fields.shape[0], -1) < t, axis=1
        )
    else:
        raise ValueError(f"Unknown builtin functional {builtin!r}.")
    return FunctionalSpec(
        id=identifier,
        function=function,
        units=units,
        description=str(node.get("description", f"CLI builtin {builtin}")),
        support=(
            None
            if node.get("support") is None
            else (float(node["support"][0]), float(node["support"][1]))
        ),
        tail_referenced=bool(node.get("tail_referenced", False)),
        direction_negative=str(node.get("direction_negative", "lower")),
        direction_positive=str(node.get("direction_positive", "higher")),
        provenance={"builtin": builtin},
    )


def _build(path: Path, payload: dict[str, Any], *, fit_artifacts: bool):
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("Configuration requires a data mapping.")
    train = _dataset(path, data.get("train", {}), "train")
    real = _dataset(path, data.get("real", {}), "real")
    generated = _dataset(path, data.get("generated", {}), "generated")
    functional_node = payload.get("functionals", [])
    if functional_node == "tc_bt" or payload.get("adapter") == "tc_bt":
        adapter_options = payload.get("adapter_options", {})
        adapter_config = tc_bt.TCBTConfig(**adapter_options)
        preprocessing = (
            {
                "operation": "clip",
                "lower_k": adapter_config.bt_min_k,
                "upper_k": adapter_config.bt_max_k,
                "reason": "match generator-training field representation",
            }
            if adapter_config.clip_to_analysis_range
            else {"operation": "none"}
        )
        train = Dataset(
            tc_bt.PreprocessedFields(train.fields, adapter_config),
            blocks=train.blocks,
            labels=train.labels,
            sample_ids=train.sample_ids,
            name=train.name,
            units=train.units,
            allow_nan=train.allow_nan,
            metadata={**train.metadata, "analysis_preprocessing": preprocessing},
        )
        real = Dataset(
            tc_bt.PreprocessedFields(real.fields, adapter_config),
            blocks=real.blocks,
            labels=real.labels,
            sample_ids=real.sample_ids,
            name=real.name,
            units=real.units,
            allow_nan=real.allow_nan,
            metadata={**real.metadata, "analysis_preprocessing": preprocessing},
        )
        generated = Dataset(
            tc_bt.PreprocessedFields(generated.fields, adapter_config),
            blocks=generated.blocks,
            labels=generated.labels,
            sample_ids=generated.sample_ids,
            name=generated.name,
            units=generated.units,
            allow_nan=generated.allow_nan,
            metadata={
                **generated.metadata,
                "analysis_preprocessing": preprocessing,
            },
        )
        functionals = tc_bt.functionals(adapter_config)
        metrics = list(tc_bt.curve_metrics(adapter_config))
    else:
        if not isinstance(functional_node, list):
            raise ValueError("functionals must be a list or the string 'tc_bt'.")
        functionals = tuple(_builtin_functional(node) for node in functional_node)
        adapter_config = None
        metrics = []
    config = EvaluationConfig(**payload.get("evaluation", {}))
    feature_node = payload.get("feature_metrics", {})
    if feature_node is None:
        feature_node = {}
    if not isinstance(feature_node, dict):
        raise ValueError("feature_metrics must be a mapping.")
    feature_enabled = bool(feature_node.get("enabled", False))
    memorization_result = None
    if fit_artifacts and feature_enabled:
        train_fields = train.fields
        real_fields = real.fields
        generated_fields = generated.fields
        feature_space = TrainFittedFeatureSpace.fit(
            train_fields,
            functionals,
            n_eof=int(feature_node.get("n_eof", 20)),
            max_fit_samples=int(feature_node.get("max_fit_samples", 512)),
            seed=int(feature_node.get("fit_seed", config.seed)),
            batch_size=int(feature_node.get("batch_size", 128)),
            fit_source="train",
        )
        # The bandwidth and standardization artifacts are fitted on the same
        # declared training-only subset as the EOF basis.  Materializing all
        # training frames would add substantial cost without changing the data
        # provenance of the fitted representation.
        fit_features = feature_space.transform(train_fields[feature_space.fit_indices])
        metrics.extend(
            feature_space.registered_metrics(
                train_features=fit_features,
                density_k=int(feature_node.get("density_k", 5)),
                bandwidth_seed=int(feature_node.get("bandwidth_seed", config.seed)),
                max_samples_per_draw=int(feature_node.get("max_samples_per_draw", 256)),
            )
        )
        dependence_requested = bool(feature_node.get("dependence_pairs", len(functionals) >= 2))
        if dependence_requested:
            if len(functionals) < 2:
                raise ValueError("Dependence pairs require at least two functionals.")
            names = [functional.id for functional in functionals]

            def functional_matrix(fields):
                return np.column_stack([functional.transform(fields) for functional in functionals])

            metrics.extend(spearman_pair_metrics(functional_matrix, names))
        copying = feature_node.get("memorization", {})
        if copying is None:
            copying = {}
        if not isinstance(copying, dict):
            raise ValueError("feature_metrics.memorization must be a mapping.")
        if bool(copying.get("enabled", True)):
            if real.blocks is None:
                raise ValueError("Memorization inference requires held-out real block ids.")
            match_labels = bool(copying.get("match_labels", True))
            if match_labels and (real.labels is None or generated.labels is None):
                raise ValueError(
                    "Label-matched memorization requires labels on real and generated data; "
                    "set feature_metrics.memorization.match_labels=false only for a "
                    "scientifically unconditional query population."
                )
            # Copying is a different question from fitting the feature basis or
            # kernel bandwidth: every field seen by the generator during
            # training must be eligible as a nearest-neighbour reference.
            memorization_train_features = feature_space.transform(train_fields)
            memorization_result = localized_copying_test(
                train_features=memorization_train_features,
                real_features=feature_space.transform(real_fields),
                generated_features=feature_space.transform(generated_fields),
                real_blocks=np.asarray(real.blocks),
                real_strata=None if not match_labels else np.asarray(real.labels),
                generated_strata=(None if not match_labels else np.asarray(generated.labels)),
                n_regions=int(copying.get("n_regions", 8)),
                n_resamples=int(copying.get("n_resamples", 1000)),
                comparison_interval=config.comparison_interval,
                min_real_blocks=int(copying.get("min_real_blocks", 5)),
                seed=int(copying.get("seed", config.seed)),
                feature_provenance=feature_space.manifest,
                estimand=config.estimand,
            )
    saturation_requested = payload.get(
        "tc_preprocessing_saturation_audit",
        payload.get("tc_admissibility", True),
    )
    if adapter_config is not None and bool(saturation_requested):
        lower, upper = adapter_config.bt_min_k, adapter_config.bt_max_k

        def boundary_saturation(fields):
            axes = tuple(range(1, fields.ndim))
            return np.any(
                np.isclose(fields, lower, rtol=0.0, atol=1e-6)
                | np.isclose(fields, upper, rtol=0.0, atol=1e-6),
                axis=axes,
            )

        metrics.append(
            admissibility_metric(
                boundary_saturation,
                id="preprocessing_boundary_saturation",
                description=(
                    "Departure in the fraction of samples touching the frozen "
                    f"analysis clip boundaries [{lower}, {upper}] K."
                ),
            )
        )
    sensitivity_results = None
    sensitivity_path = payload.get("sensitivity_results")
    if sensitivity_path is not None:
        with _resolve(path, str(sensitivity_path)).open(encoding="utf-8") as handle:
            sensitivity_results = json.load(handle)
        if not isinstance(sensitivity_results, dict):
            raise ValueError("sensitivity_results must contain a JSON object.")
    copying_config = feature_node.get("memorization", {}) or {}
    if memorization_result is not None and bool(
        copying_config.get("require_copyist_control", False)
    ):
        copyist_control = (sensitivity_results or {}).get("copyist")
        if not isinstance(copyist_control, dict):
            raise ValueError(
                "The registered memorization analysis requires a copyist positive control."
            )
        raw_status = memorization_result.get("status")
        memorization_result = {
            **memorization_result,
            "copyist_positive_control": {
                "passed": bool(copyist_control.get("passed", False)),
                "observed_status": copyist_control.get("result", {}).get("status"),
            },
        }
        if not copyist_control.get("passed", False):
            memorization_result = {
                **memorization_result,
                "status": "CONTROL_FAILED",
                "uninterpreted_status": raw_status,
                "reason": (
                    "Memorization interpretation withheld because the registered "
                    "copyist positive control failed."
                ),
            }
    return (
        train,
        real,
        generated,
        functionals,
        tuple(metrics),
        config,
        memorization_result,
        sensitivity_results,
        feature_enabled,
    )


def command_validate(args: argparse.Namespace) -> int:
    path = Path(args.config).resolve()
    payload = _load_yaml(path)
    (
        train,
        real,
        generated,
        functionals,
        metrics,
        config,
        _,
        _,
        feature_enabled,
    ) = _build(path, payload, fit_artifacts=False)
    split = validate_split(train, real)
    validate_generated(real, generated)
    batch_size = int(payload.get("validation_batch_size", 64))
    result = {
        "valid": True,
        "split": split,
        "field_scans": [
            dataset.validate_fields(batch_size=batch_size) for dataset in (train, real, generated)
        ],
        "n_functionals": len(functionals),
        "n_metrics": len(metrics),
        "feature_metrics_requested": feature_enabled,
        "config": config.to_dict(),
    }
    print(json.dumps(result, indent=2))
    return 0


def command_run(args: argparse.Namespace) -> int:
    path = Path(args.config).resolve()
    payload = _load_yaml(path)
    (
        train,
        real,
        generated,
        functionals,
        metrics,
        config,
        memorization_result,
        sensitivity_results,
        _,
    ) = _build(path, payload, fit_artifacts=True)
    report = evaluate(
        train=train,
        real=real,
        generated=generated,
        functionals=functionals,
        metrics=metrics,
        config=config,
        memorization_result=memorization_result,
        sensitivity_results=sensitivity_results,
        conditional=bool(payload.get("conditional", False)),
        conditional_groups=payload.get("conditional_groups", {}),
    )
    outputs = payload.get("outputs", {})
    json_path = _resolve(path, str(outputs.get("json", "evaluation_card.json")))
    report.to_json(json_path)
    if outputs.get("html"):
        report.to_html(_resolve(path, str(outputs["html"])))
    if outputs.get("profile"):
        report.figure_reference_profile(_resolve(path, str(outputs["profile"])))
    for curve in outputs.get("curves", []):
        report.figure_curve(
            str(curve["metric"]),
            str(curve.get("stratum", "global")),
            _resolve(path, str(curve["path"])),
        )
    print(json.dumps({"report": str(json_path), "cells": len(report.cells)}, indent=2))
    return 0


def command_report(args: argparse.Namespace) -> int:
    report = EvaluationReport.from_json(args.input)
    if args.html:
        report.to_html(args.html)
    if args.profile:
        report.figure_reference_profile(args.profile)
    for metric, stratum, output in args.curve or []:
        report.figure_curve(metric, stratum, output)
    if not args.html and not args.profile and not args.curve:
        print(json.dumps(report.to_dict(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gauge-climate")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate", help="Validate data and configuration.")
    validate_parser.add_argument("config")
    validate_parser.set_defaults(function=command_validate)
    run_parser = subparsers.add_parser("run", help="Run a GAUGE evaluation.")
    run_parser.add_argument("config")
    run_parser.set_defaults(function=command_run)
    report_parser = subparsers.add_parser("report", help="Render an existing JSON report.")
    report_parser.add_argument("input")
    report_parser.add_argument("--html")
    report_parser.add_argument("--profile")
    report_parser.add_argument(
        "--curve",
        nargs=3,
        action="append",
        metavar=("METRIC", "STRATUM", "OUTPUT"),
        help="Render a stored curve diagnostic; may be repeated.",
    )
    report_parser.set_defaults(function=command_report)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
