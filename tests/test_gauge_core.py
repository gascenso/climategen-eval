from __future__ import annotations

import tempfile
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
        bootstrap_resamples=60,
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


def test_perfect_generator_has_two_separate_conclusions():
    train, real, generated = _datasets()
    functional = gauge.FunctionalSpec(
        "value",
        lambda fields: fields[:, 0],
        "unit",
        "bounded toy value",
        tolerance=0.8,
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
    assert cell.practical_status in {
        gauge.PracticalStatus.WITHIN_TOLERANCE,
        gauge.PracticalStatus.INCONCLUSIVE,
        gauge.PracticalStatus.UNDERPOWERED,
    }
    assert cell.provenance["inference_method"] == "finite_sample_dkw_quantile_band"
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


def test_missing_support_withholds_tolerance_conclusion():
    train, real, generated = _datasets()
    functional = gauge.FunctionalSpec(
        "value", lambda fields: fields[:, 0], "unit", "unbounded toy", tolerance=0.2
    )
    report = gauge.evaluate(
        train=train,
        real=real,
        generated=generated,
        functionals=[functional],
        config=_config(),
    )
    cell = report.cells[0]
    assert cell.practical_status is gauge.PracticalStatus.INCONCLUSIVE
    assert "withheld" in cell.warnings[0].lower()


def test_thin_conditional_stratum_is_reported_underpowered():
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
        tolerance=0.5,
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
    assert thin.practical_status is gauge.PracticalStatus.UNDERPOWERED


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
        loaded = gauge.EvaluationReport.from_json(root / "report.json")
        assert loaded.to_dict() == report.to_dict()
        assert (root / "report.html").stat().st_size > 100
        assert (root / "profile.png").stat().st_size > 100


def test_config_hash_captures_metric_contract():
    train, real, generated = _datasets()
    reports = []
    for tolerance in (0.2, 0.4):
        functional = gauge.FunctionalSpec(
            "value",
            lambda fields: fields[:, 0],
            "unit",
            "bounded toy",
            tolerance=tolerance,
            support=(-1, 1),
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
        tolerance=0.4,
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
    assert cell.sensitivity_status is gauge.SensitivityStatus.INADEQUATE
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
