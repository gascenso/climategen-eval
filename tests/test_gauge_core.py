from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

import gauge_climate as gauge
from gauge_climate.data import validate_split
from gauge_climate.reference import build_reference_plan


def _datasets(seed: int = 4):
    rng = np.random.default_rng(seed)
    train_blocks = np.repeat(np.arange(40), 3)
    real_blocks = np.repeat(np.arange(100, 160), 4)
    train = gauge.Dataset(rng.uniform(-1, 1, size=(train_blocks.size, 1)), blocks=train_blocks)
    real = gauge.Dataset(rng.uniform(-1, 1, size=(real_blocks.size, 1)), blocks=real_blocks)
    generated = gauge.Dataset(rng.uniform(-1, 1, size=(150, 1)))
    return train, real, generated


def _config(**overrides):
    values = dict(
        n_eval=100,
        n_resamples=40,
        sensitivity_resamples=40,
        min_blocks_per_half=5,
        warn_blocks_per_half=10,
        shift_grid_size=12,
        seed=12,
    )
    values.update(overrides)
    return gauge.EvaluationConfig(**values)


def test_split_overlap_is_rejected():
    train = gauge.Dataset(np.zeros((4, 1)), blocks=["a", "a", "b", "b"])
    real = gauge.Dataset(np.zeros((4, 1)), blocks=["c", "c", "a", "a"])
    try:
        validate_split(train, real)
    except ValueError as exc:
        assert "leakage" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("Expected block leakage to be rejected.")


def test_split_sample_id_overlap_is_rejected_even_with_relabeled_blocks():
    train = gauge.Dataset(
        np.zeros((4, 1)),
        blocks=["a", "a", "b", "b"],
        sample_ids=["t0", "shared", "t2", "t3"],
    )
    real = gauge.Dataset(
        np.zeros((4, 1)),
        blocks=["c", "c", "d", "d"],
        sample_ids=["r0", "r1", "shared", "r3"],
    )
    try:
        validate_split(train, real)
    except ValueError as exc:
        assert "sample-id leakage" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("Expected sample-id leakage to be rejected.")


def test_dataset_rejects_missing_metadata_duplicate_ids_and_unbounded_materialization():
    for kwargs in (
        {"blocks": ["a", np.nan]},
        {"sample_ids": ["same", "same"]},
    ):
        try:
            gauge.Dataset(np.zeros((2, 2)), **kwargs)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"Expected Dataset validation failure for {kwargs}.")
    dataset = gauge.Dataset(np.zeros((2, 20), dtype=np.float64))
    try:
        dataset.as_numpy(max_bytes=8)
    except MemoryError:
        pass
    else:  # pragma: no cover
        raise AssertionError("Expected controlled materialization failure.")
    with_nan = gauge.Dataset(np.asarray([[0.0, np.nan], [1.0, 2.0]]))
    try:
        with_nan.as_numpy()
    except ValueError as exc:
        assert "nan" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("Expected non-finite field rejection.")


def test_declared_field_unit_mismatch_is_rejected():
    train = gauge.Dataset(np.zeros((4, 2)), blocks=["a", "a", "b", "b"], units="K")
    real = gauge.Dataset(np.zeros((4, 2)), blocks=["c", "c", "d", "d"], units="degC")
    try:
        validate_split(train, real)
    except ValueError as exc:
        assert "units" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("Expected unit mismatch rejection.")


def test_reference_plan_keeps_halves_disjoint_and_whole():
    blocks = np.repeat(np.arange(30), [2 + (i % 5) for i in range(30)])
    plan = build_reference_plan(
        real_indices=np.arange(blocks.size),
        real_blocks=blocks,
        n_generated=120,
        config=_config(n_eval=80),
        seed=9,
    )
    normalized = np.asarray([str(value) for value in blocks], dtype=object)
    for draw in plan.draws:
        assert set(draw.blocks_a).isdisjoint(draw.blocks_b)
        assert set(normalized[draw.pseudo_indices]).issubset(draw.blocks_a)
        assert set(normalized[draw.real_indices]).issubset(draw.blocks_b)
        assert len(draw.pseudo_indices) == len(draw.real_indices) == len(draw.generated_indices) == 80


def test_reference_plan_has_compact_reproducible_provenance():
    blocks = np.repeat(np.arange(30), 3)
    kwargs = dict(
        real_indices=np.arange(blocks.size),
        real_blocks=blocks,
        n_generated=80,
        config=_config(n_eval=80),
    )
    first = build_reference_plan(**kwargs, seed=9)
    second = build_reference_plan(**kwargs, seed=9)
    changed = build_reference_plan(**kwargs, seed=10)
    assert first.provenance_hash() == second.provenance_hash()
    assert first.provenance_hash() != changed.provenance_hash()
    assert len(first.partition_manifest(limit=3)) == 3
    assert all(
        draw.generated_indices is first.draws[0].generated_indices
        for draw in first.draws
    )


def test_perfect_generator_has_reference_aware_detection_and_raw_interval():
    train, real, generated = _datasets()
    functional = gauge.FunctionalSpec(
        "value",
        lambda fields: fields[:, 0],
        "unit",
        "bounded toy value",
        support=(-1.0, 1.0),
    )
    report = gauge.evaluate(
        train=train,
        real=real,
        generated=generated,
        functionals=[functional],
        config=_config(),
    )
    cell = report.cells[0]
    assert cell.detection_status is gauge.DetectionStatus.NO_DEPARTURE_DETECTED
    assert cell.provenance["inference_method"] == "reference_resampling_interval"
    assert cell.provenance["interval_bound_type"] == "central_resampling_1_minus_alpha"
    assert cell.ci_low <= cell.estimate <= cell.ci_high
    assert cell.reference_position is not None


def test_evaluation_transforms_lazy_fields_in_bounded_batches():
    class LazyArchive:
        def __init__(self, values):
            self.values = values
            self.shape = values.shape
            self.dtype = values.dtype
            self.batch_sizes = []

        def __getitem__(self, key):
            result = self.values[key]
            self.batch_sizes.append(int(result.shape[0]))
            return result

        def __array__(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("Full lazy archive was materialized.")

    train, real, generated = _datasets()
    real_fields = LazyArchive(real.fields)
    generated_fields = LazyArchive(generated.fields)
    real = gauge.Dataset(real_fields, blocks=real.blocks)
    generated = gauge.Dataset(generated_fields)
    functional = gauge.FunctionalSpec(
        "value", lambda fields: fields[:, 0], "unit", "bounded toy", support=(-1, 1)
    )
    gauge.evaluate(
        train=train,
        real=real,
        generated=generated,
        functionals=[functional],
        config=_config(transform_batch_size=17),
    )
    assert max(real_fields.batch_sizes) <= 17
    assert max(generated_fields.batch_sizes) <= 17


def test_joint_detection_is_only_issued_for_a_preregistered_family():
    train, real, generated = _datasets()
    functional = gauge.FunctionalSpec(
        "value", lambda fields: fields[:, 0], "unit", "bounded toy", support=(-1, 1)
    )
    no_family = gauge.evaluate(
        train=train,
        real=real,
        generated=generated,
        functionals=[functional],
        config=_config(),
    )
    assert (
        no_family.cells[0].simultaneous_detection_status
        is gauge.DetectionStatus.NOT_EVALUATED
    )
    registered = gauge.evaluate(
        train=train,
        real=real,
        generated=generated,
        functionals=[functional],
        config=_config(simultaneous_family=("global:w1:value",)),
    )
    assert (
        registered.cells[0].simultaneous_detection_status
        is not gauge.DetectionStatus.NOT_EVALUATED
    )


def test_functional_support_is_not_required_for_reference_evaluation():
    train, real, generated = _datasets()
    functional = gauge.FunctionalSpec(
        "value", lambda fields: fields[:, 0], "unit", "unbounded toy"
    )
    report = gauge.evaluate(
        train=train,
        real=real,
        generated=generated,
        functionals=[functional],
        config=_config(),
    )
    cell = report.cells[0]
    assert cell.estimate is not None
    assert cell.detection_status is not gauge.DetectionStatus.NOT_EVALUATED


def test_thin_conditional_stratum_is_reported_without_a_verdict():
    train, real, generated = _datasets()
    real_labels = np.zeros(real.n_samples, dtype=int)
    real_labels[:16] = 1  # four blocks
    generated_labels = np.zeros(generated.n_samples, dtype=int)
    generated_labels[:20] = 1
    real = gauge.Dataset(real.fields, blocks=real.blocks, labels=real_labels)
    generated = gauge.Dataset(generated.fields, labels=generated_labels)
    functional = gauge.FunctionalSpec(
        "value",
        lambda fields: fields[:, 0],
        "unit",
        "bounded toy value",
        support=(-1, 1),
    )
    report = gauge.evaluate(
        train=train,
        real=real,
        generated=generated,
        functionals=[functional],
        config=_config(n_eval=20),
        conditional=True,
    )
    thin = next(cell for cell in report.cells if cell.stratum == "1")
    assert thin.detection_status is gauge.DetectionStatus.NOT_EVALUATED
    assert "block" in thin.warnings[0].lower()


def test_predeclared_conditional_aggregate_is_an_explicit_report_stratum():
    train, real, generated = _datasets()
    real_labels = np.tile([0, 1], real.n_samples // 2)
    generated_labels = np.tile([0, 1], generated.n_samples // 2)
    real = gauge.Dataset(real.fields, blocks=real.blocks, labels=real_labels)
    generated = gauge.Dataset(generated.fields, labels=generated_labels)
    functional = gauge.FunctionalSpec(
        "value", lambda fields: fields[:, 0], "unit", "bounded toy", support=(-1, 1)
    )
    report = gauge.evaluate(
        train=train,
        real=real,
        generated=generated,
        functionals=[functional],
        config=_config(),
        conditional=False,
        conditional_groups={"all_labels": (0, 1)},
    )
    assert {cell.stratum for cell in report.cells} == {"global", "all_labels"}
    assert report.metadata["analysis"]["conditional_groups"] == {
        "all_labels": [0, 1]
    }


def test_global_conditional_mixture_mismatch_is_explicitly_audited():
    train, real, generated = _datasets()
    real_labels = np.zeros(real.n_samples, dtype=int)
    real_labels[: real.n_samples // 4] = 1
    generated_labels = np.tile([0, 1], generated.n_samples // 2)
    real = gauge.Dataset(real.fields, blocks=real.blocks, labels=real_labels)
    generated = gauge.Dataset(generated.fields, labels=generated_labels)
    functional = gauge.FunctionalSpec(
        "value", lambda fields: fields[:, 0], "unit", "bounded toy", support=(-1, 1)
    )
    report = gauge.evaluate(
        train=train,
        real=real,
        generated=generated,
        functionals=[functional],
        config=_config(n_eval=50),
        conditional=True,
    )
    audit = report.metadata["analysis"]["global_label_mixture"]
    assert audit["policy"] == "as_provided"
    assert audit["matched"] is False
    assert audit["total_variation_distance"] > 0
    assert "composition checks" in report.warnings[0]
    global_cell = next(cell for cell in report.cells if cell.stratum == "global")
    assert global_cell.provenance["global_label_mixture_policy"] == "as_provided"


def test_report_round_trip_and_render():
    train, real, generated = _datasets()
    functional = gauge.FunctionalSpec(
        "value", lambda fields: fields[:, 0], "unit", "bounded toy", support=(-1, 1)
    )
    report = gauge.evaluate(
        train=train,
        real=real,
        generated=generated,
        functionals=[functional],
        config=_config(),
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        report.to_json(root / "report.json")
        report.to_html(root / "report.html")
        report.figure_reference_profile(root / "profile.png")
        html = (root / "report.html").read_text(encoding="utf-8")
        loaded = gauge.EvaluationReport.from_json(root / "report.json")
        assert loaded.to_dict() == report.to_dict()
        assert "<th>Tolerance</th>" not in html
        assert "<th>Practical relevance</th>" not in html
        assert "not proof of equality or universal application acceptability" in html
        assert (root / "report.html").stat().st_size > 100
        assert (root / "profile.png").stat().st_size > 100


def test_v1_report_migration_reuses_matched_reference_interval():
    train, real, generated = _datasets()
    functional = gauge.FunctionalSpec(
        "value", lambda fields: fields[:, 0], "unit", "bounded toy", support=(-1, 1)
    )
    report = gauge.evaluate(
        train=train,
        real=real,
        generated=generated,
        functionals=[functional],
        config=_config(),
    )
    payload = report.to_dict()
    payload["schema"] = "gauge_climate.evaluation.v1"
    payload["metadata"]["config"]["bootstrap_resamples"] = 100
    payload["metadata"]["interpretation"]["practical"] = "retired"
    cell = payload["cells"][0]
    cell["tolerance"] = 0.2
    cell["practical_status"] = "WITHIN_TOLERANCE"
    cell["provenance"]["reference_model_median"] = 0.31
    cell["provenance"]["model_reference_q025"] = 0.21
    cell["provenance"]["model_reference_q975"] = 0.44
    cell["provenance"]["interval_standard_error"] = 99.0

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "legacy.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        migrated = gauge.EvaluationReport.from_json(path)
        result = migrated.to_dict()

    migrated_cell = result["cells"][0]
    assert result["schema"] == "gauge_climate.evaluation.v2"
    assert migrated_cell["estimate"] == 0.31
    assert migrated_cell["ci_low"] == 0.21
    assert migrated_cell["ci_high"] == 0.44
    assert migrated_cell["provenance"]["inference_method"] == "reference_resampling_interval"
    assert "tolerance" not in migrated_cell
    assert "practical_status" not in migrated_cell
    assert "bootstrap_resamples" not in result["metadata"]["config"]


def test_paper_bundle_preserves_native_strata_and_renders_controls():
    train, real, generated = _datasets()
    real = gauge.Dataset(
        real.fields,
        blocks=real.blocks,
        labels=np.repeat([0, 1], len(real.fields) // 2),
    )
    generated = gauge.Dataset(
        generated.fields,
        labels=np.repeat([0, 1], len(generated.fields) // 2),
    )
    functional = gauge.FunctionalSpec(
        "value", lambda fields: fields[:, 0], "unit", "bounded toy", support=(-1, 1)
    )
    report = gauge.evaluate(
        train=train,
        real=real,
        generated=generated,
        functionals=[functional],
        config=_config(n_eval=70),
        conditional=True,
    )
    spec = gauge.PaperBundleSpec(
        title="Toy conditional evaluation",
        model_label="toy generator",
        metrics=("w1:value",),
        native_strata=("0", "1"),
        metric_labels={"w1:value": "Value distribution"},
        stratum_labels={"0": "low", "1": "high"},
    )
    controls = (
        gauge.ReferenceControl("w1:value", "shift control", 3.0),
    )
    with tempfile.TemporaryDirectory() as directory:
        artifacts = report.paper_bundle(directory, spec, controls=controls)
        assert set(artifacts) == {
            "profile_pdf",
            "profile_png",
            "matrix_pdf",
            "matrix_png",
            "selected_table",
            "all_table",
            "index",
        }
        assert all(path.stat().st_size > 100 for path in artifacts.values())
        table = artifacts["selected_table"].read_bytes()
        assert b"\r\n" not in table
        assert b"low" in table and b"high" in table


def test_paper_bundle_compresses_large_k_only_in_main_profile():
    train, real, generated = _datasets()
    functional = gauge.FunctionalSpec(
        "value", lambda fields: fields[:, 0], "unit", "bounded toy", support=(-1, 1)
    )
    source = gauge.evaluate(
        train=train,
        real=real,
        generated=generated,
        functionals=[functional],
        config=_config(),
    )
    base = source.cells[0]
    cells = tuple(
        replace(
            base,
            stratum=str(index),
            reference_position=0.2 * index,
            reference_position_low=0.2 * index - 0.1,
            reference_position_high=0.2 * index + 0.1,
        )
        for index in range(10)
    )
    report = gauge.EvaluationReport(metadata=source.metadata, cells=cells)
    spec = gauge.PaperBundleSpec(
        title="Large-K evaluation",
        model_label="toy generator",
        metrics=("w1:value",),
        native_strata=tuple(str(index) for index in range(10)),
        max_visible_native_strata=8,
    )
    with tempfile.TemporaryDirectory() as directory:
        artifacts = report.paper_bundle(directory, spec)
        assert all(path.stat().st_size > 100 for path in artifacts.values())
        table = artifacts["selected_table"].read_text(encoding="utf-8")
        assert ",0,0," in table and ",9,9," in table
        assert sum(1 for line in table.splitlines() if line) == 11


def test_config_hash_captures_metric_contract():
    train, real, generated = _datasets()
    reports = []
    for support in ((-1.0, 1.0), (-2.0, 2.0)):
        functional = gauge.FunctionalSpec(
            "value",
            lambda fields: fields[:, 0],
            "unit",
            "bounded toy",
            support=support,
        )
        reports.append(
            gauge.evaluate(
                train=train,
                real=real,
                generated=generated,
                functionals=[functional],
                config=_config(),
            )
        )
    assert reports[0].metadata["config_hash"] != reports[1].metadata["config_hash"]


def test_failed_predeclared_control_withholds_metric_interpretation():
    train, real, generated = _datasets()
    functional = gauge.FunctionalSpec(
        "value",
        lambda fields: fields[:, 0],
        "unit",
        "bounded toy",
        support=(-1, 1),
    )
    metric_id = "w1:value"
    report = gauge.evaluate(
        train=train,
        real=real,
        generated=generated,
        functionals=[functional],
        config=_config(),
        sensitivity_results={
            "bad_lane": {"metrics": {metric_id: {"passed": False}}, "passed": False}
        },
    )
    cell = report.cells[0]
    assert cell.detection_status is gauge.DetectionStatus.NOT_EVALUATED
    assert "withheld" in cell.warnings[-1].lower()


def test_curve_diagnostic_is_serialized_and_renderable():
    train, real, generated = _datasets()
    curve = gauge.MetricSpec(
        id="toy_curve",
        transform=lambda fields: np.column_stack([fields[:, 0], fields[:, 0] ** 2]),
        estimator=lambda generated_values, real_values: float(
            np.mean(
                np.abs(
                    np.mean(generated_values, axis=0)
                    - np.mean(real_values, axis=0)
                )
            )
        ),
        kind=gauge.MetricKind.CURVE,
        units="toy units",
        description="two-bin toy curve",
        provenance={"coordinate_values": [1.0, 2.0], "coordinate_label": "toy x"},
    )
    report = gauge.evaluate(
        train=train,
        real=real,
        generated=generated,
        metrics=[curve],
        config=_config(),
    )
    assert "global:toy_curve" in report.diagnostics
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "curve.pdf"
        report.figure_curve("toy_curve", "global", path)
        assert path.stat().st_size > 100


def test_secondary_metrics_use_declared_diagnostic_resample_budget():
    train, real, generated = _datasets()
    functional = gauge.FunctionalSpec(
        "value", lambda fields: fields[:, 0], "unit", "bounded toy", support=(-1, 1)
    )
    diagnostic = gauge.MetricSpec(
        id="mean_departure",
        transform=lambda fields: fields[:, 0],
        estimator=lambda generated_values, real_values: abs(
            float(np.mean(generated_values) - np.mean(real_values))
        ),
        kind=gauge.MetricKind.SET,
        units="unit",
        description="toy secondary diagnostic",
    )
    report = gauge.evaluate(
        train=train,
        real=real,
        generated=generated,
        functionals=[functional],
        metrics=[diagnostic],
        config=_config(diagnostic_resamples=20),
    )
    by_metric = {cell.metric: cell for cell in report.cells}
    assert by_metric["w1:value"].provenance["reference_resamples_used"] == 40
    assert by_metric["mean_departure"].provenance["reference_resamples_used"] == 20
    assert (
        by_metric["mean_departure"].provenance["reference_resample_role"]
        == "secondary_diagnostic"
    )
    partition = report.partitions["global"]
    assert partition["draw_count"] == 40
    assert len(partition["preview"]) == 3
    assert len(partition["manifest_sha256"]) == 64


def test_secondary_metric_can_receive_a_predeclared_larger_budget():
    train, real, generated = _datasets()
    diagnostic = gauge.MetricSpec(
        id="mean_departure",
        transform=lambda fields: fields[:, 0],
        estimator=lambda generated_values, real_values: abs(
            float(np.mean(generated_values) - np.mean(real_values))
        ),
        kind=gauge.MetricKind.SET,
        units="unit",
        description="toy secondary diagnostic",
    )
    report = gauge.evaluate(
        train=train,
        real=real,
        generated=generated,
        metrics=[diagnostic],
        config=_config(
            diagnostic_resamples=20,
            diagnostic_resamples_by_metric={"mean_departure": 30},
        ),
    )
    cell = report.cells[0]
    assert cell.provenance["reference_resamples_used"] == 30


def test_invalid_per_metric_diagnostic_budget_is_rejected():
    for budget in (19, 41):
        try:
            _config(diagnostic_resamples_by_metric={"mean_departure": budget})
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("Expected invalid per-metric diagnostic budget rejection.")


def test_unstable_reference_boundary_withholds_categorical_detection():
    train, real, generated = _datasets()
    diagnostic = gauge.MetricSpec(
        id="mean_departure",
        transform=lambda fields: fields[:, 0],
        estimator=lambda generated_values, real_values: abs(
            float(np.mean(generated_values) - np.mean(real_values))
        ),
        kind=gauge.MetricKind.SET,
        units="unit",
        description="toy secondary diagnostic",
    )
    report = gauge.evaluate(
        train=train,
        real=real,
        generated=generated,
        metrics=[diagnostic],
        config=_config(
            diagnostic_resamples=20,
            reference_stability_threshold=1e-12,
        ),
    )
    cell = report.cells[0]
    assert cell.detection_status is gauge.DetectionStatus.NOT_EVALUATED
    assert any("instability" in warning for warning in cell.warnings)
