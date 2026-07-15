import numpy as np
import pytest

from gauge_climate import Dataset
from gauge_climate.downstream import Arm, Holdout, Ladder, LadderConfig, geo_default


def _fixture(leak=False):
    # Every storm contains both labels, exercising whole-storm removal when a
    # block spans categories while preserving copy candidates in training.
    fields = np.arange(12 * 4 * 6, dtype=np.float32).reshape(12, 4, 6)
    blocks = np.repeat(["s0", "s1", "s2", "s3", "s4", "s5"], 2)
    labels = np.tile([0, 1], 6)
    train = Dataset(fields, blocks=blocks, labels=labels, name="train")
    holdouts = (Holdout("h0", ("s0",)), Holdout("h1", ("s1",)))
    generated = {
        holdout.id: Dataset(
            np.full((6, 4, 6), 5.0 + index, dtype=np.float32),
            labels=np.tile([0, 1], 3),
            name=f"generated_{holdout.id}",
        )
        for index, holdout in enumerate(holdouts)
    }
    provenance = {}
    for holdout in holdouts:
        allowed = [block for block in np.unique(blocks) if block not in holdout.blocks]
        if leak and holdout.id == "h0":
            allowed.append("s0")
        provenance[holdout.id] = allowed
    return train, holdouts, generated, provenance


def test_ladder_count_transform_and_pairing_contract():
    train, holdouts, generated, provenance = _fixture()
    seen = {}

    def task(batch, seed, holdout_id):
        seen[(holdout_id, seed, batch.arm)] = batch
        return {"utility": float(np.mean(batch.fields))}

    ladder = Ladder(
        task=task,
        task_metadata={"task": "mean_toy"},
        train=train,
        holdouts=holdouts,
        generated_by_holdout=generated,
        generator_training_blocks_by_holdout=provenance,
        transform=geo_default,
        config=LadderConfig(seeds=(11, 12), bootstrap_resamples=100),
    )
    report = ladder.run()
    assert len(report.records) == 2 * 2 * 8
    for holdout in holdouts:
        for seed in (11, 12):
            counts = [
                seen[(holdout.id, seed, arm)].fields.shape[0]
                for arm in (
                    Arm.A2_COPY,
                    Arm.A3_COPY_GEO,
                    Arm.A4_GENERATED,
                    Arm.A5_GENERATED_GEO,
                    Arm.A6_RESTORED_REAL,
                    Arm.A7_RESTORED_REAL_GEO,
                )
            ]
            assert len(set(counts)) == 1
            for arm in Arm:
                batch = seen[(holdout.id, seed, arm)]
                assert batch.metadata["restricted_count"] == 10
            for arm in (Arm.A3_COPY_GEO, Arm.A5_GENERATED_GEO, Arm.A7_RESTORED_REAL_GEO):
                assert seen[(holdout.id, seed, arm)].metadata["transform_applied"] is True
                plan = report.plans[holdout.id]["arms"][f"{seed}:{arm.value}"]
                assert plan["transform_output_sha256"]
                assert plan["transform_trace"]["explicit_parameters"] is True
    contrast = report.contrast(
        metric="utility",
        treatment=Arm.A5_GENERATED_GEO,
        reference=Arm.A3_COPY_GEO,
        delta_task=100.0,
        higher_is_better=True,
        bootstrap_resamples=100,
    )
    assert set(contrast["paired_values"]) == {"h0", "h1"}


def test_ladder_rejects_generator_holdout_leakage():
    train, holdouts, generated, provenance = _fixture(leak=True)
    with pytest.raises(ValueError, match="Generator leakage"):
        Ladder(
            task=lambda batch, seed, holdout: {"x": 0.0},
            task_metadata={"task": "leakage_toy"},
            train=train,
            holdouts=holdouts,
            generated_by_holdout=generated,
            generator_training_blocks_by_holdout=provenance,
            transform=geo_default,
        )


def test_ladder_determinism_audit_rejects_stateful_callback():
    train, holdouts, generated, provenance = _fixture()
    counter = {"value": 0}

    def nondeterministic(batch, seed, holdout):
        counter["value"] += 1
        return {"score": float(counter["value"])}

    ladder = Ladder(
        task=nondeterministic,
        task_metadata={"task": "intentional_nondeterminism"},
        train=train,
        holdouts=(holdouts[0],),
        generated_by_holdout={holdouts[0].id: generated[holdouts[0].id]},
        generator_training_blocks_by_holdout={holdouts[0].id: provenance[holdouts[0].id]},
        transform=geo_default,
        config=LadderConfig(
            seeds=(1, 2), bootstrap_resamples=100, audit_callback_determinism=True
        ),
    )
    with pytest.raises(ValueError, match="nondeterministic"):
        ladder.run()


def test_geo_default_preserves_rectangular_shape():
    fields = np.arange(3 * 4 * 6).reshape(3, 4, 6)
    transformed = geo_default(fields, np.zeros(3), np.random.default_rng(1))
    assert transformed.shape == fields.shape


def test_continuous_targets_can_use_frozen_matching_bins():
    train, holdouts, generated, provenance = _fixture()
    train = Dataset(
        train.fields,
        blocks=train.blocks,
        labels=np.linspace(35.0, 145.0, train.n_samples),
    )
    generated = {
        key: Dataset(value.fields, labels=np.linspace(35.0, 145.0, value.n_samples))
        for key, value in generated.items()
    }

    def wind_bin(values):
        return np.digitize(values, [64.0, 83.0, 96.0, 113.0, 137.0])

    report = Ladder(
        task=lambda batch, seed, holdout: {"mae": float(np.mean(batch.labels))},
        task_metadata={"task": "continuous_label_toy"},
        train=train,
        holdouts=holdouts,
        generated_by_holdout=generated,
        generator_training_blocks_by_holdout=provenance,
        transform=geo_default,
        matching_strata=wind_bin,
        matching_metadata={"policy": "wind_bins", "edges_kt": [64, 83, 96, 113, 137]},
        config=LadderConfig(seeds=(1, 2), bootstrap_resamples=100),
    ).run()
    for holdout in holdouts:
        records = [
            record
            for record in report.records
            if record.holdout == holdout.id and record.seed == 1
        ]
        augmented = [record for record in records if record.arm not in {Arm.A0_NO_AUG, Arm.A1_WEIGHT}]
        assert len({sum(record.counts_by_stratum.values()) for record in augmented}) == 1


def test_toy_ladder_recovers_useful_generation_and_saturation():
    values = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    train = Dataset(
        values[:, None],
        blocks=np.repeat(["left", "holdout", "right"], 2),
        labels=values,
    )
    holdout = Holdout("h", ("holdout",))
    generated = {"h": Dataset(np.asarray([[2.0], [3.0]]), labels=np.asarray([2.0, 3.0]))}
    provenance = {"h": ("left", "right")}

    def identity(fields, labels, rng):
        return fields

    test_x = np.asarray([2.0, 3.0])
    test_y = np.asarray([2.0, 3.0])

    def nearest_neighbor_task(batch, seed, holdout_id):
        del seed, holdout_id
        train_x = batch.fields[:, 0]
        nearest = np.argmin(np.abs(test_x[:, None] - train_x[None, :]), axis=1)
        return {
            "mae": {
                "holdout": float(np.mean(np.abs(batch.labels[nearest] - test_y)))
            }
        }

    common = dict(
        task_metadata={"task": "nearest_neighbor_toy"},
        train=train,
        holdouts=(holdout,),
        generated_by_holdout=generated,
        generator_training_blocks_by_holdout=provenance,
        transform=identity,
        transform_metadata={"policy": "identity_toy_transform"},
        matching_strata=lambda labels: np.zeros(labels.shape, dtype=int),
        matching_metadata={"policy": "single_toy_stratum"},
        config=LadderConfig(seeds=(1, 2), bootstrap_resamples=100),
    )
    useful = Ladder(task=nearest_neighbor_task, **common).run().contrast(
        metric="mae",
        treatment=Arm.A5_GENERATED_GEO,
        reference=Arm.A3_COPY_GEO,
        delta_task=0.25,
        higher_is_better=False,
        bootstrap_resamples=100,
    )
    assert useful["estimate"] > 0
    assert useful["superior"] is True

    saturated_common = {**common, "task_metadata": {"task": "saturation_toy"}}
    saturated = Ladder(
        task=lambda batch, seed, holdout_id: {"score": 1.0}, **saturated_common
    )
    saturation_contrast = saturated.run().contrast(
        metric="score",
        treatment=Arm.A5_GENERATED_GEO,
        reference=Arm.A3_COPY_GEO,
        delta_task=0.1,
        higher_is_better=True,
        bootstrap_resamples=100,
    )
    assert saturation_contrast["status"] == "EQUIVALENT"
