import json

import numpy as np
import pytest

from gauge_climate.adapters.tc_data import (
    load_real_archive,
    load_sample_bank,
    wind_to_category,
)
from case_studies.tropical_cyclone_bt.downstream_cnn import tc_geometry


def test_wind_categories_use_usa_one_minute_thresholds():
    winds = np.asarray([35, 63, 64, 82, 83, 95, 96, 112, 113, 136, 137, 170])
    assert wind_to_category(winds).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    with pytest.raises(ValueError, match="USA_WIND"):
        wind_to_category([34.0])


def test_real_archive_rejects_mixed_period_provenance(tmp_path):
    root = tmp_path / "real_train"
    root.mkdir()
    winds = np.asarray([40.0, 70.0, 90.0, 100.0, 120.0, 145.0], dtype=np.float32)
    np.save(root / "bt_k.npy", np.ones((6, 4, 4), dtype=np.float32) * 240.0)
    np.save(root / "labels.npy", wind_to_category(winds))
    np.save(root / "wind_kt.npy", winds)
    np.save(root / "storm_ids.npy", np.asarray(["a", "a", "b", "b", "c", "c"]))
    (root / "rel_paths.txt").write_text("\n".join(f"sample_{i}" for i in range(6)) + "\n")
    manifest = {
        "schema": "gauge_climate.tc_bt_real_dataset.v2",
        "split": "train",
        "wind_source": "USA_WIND",
        "wind_averaging_period": "1-minute",
        "bt_k": "bt_k.npy",
        "labels": "labels.npy",
        "wind_kt": "wind_kt.npy",
        "storm_ids": "storm_ids.npy",
        "rel_paths": "rel_paths.txt",
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    archive = load_real_archive(path, target="wind")
    assert archive.labels.tolist() == winds.tolist()

    manifest["wind_source"] = "WMO_WIND"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="USA_WIND"):
        load_real_archive(path)


def test_baseline_sample_bank_is_loaded_lazily(tmp_path):
    root = tmp_path / "bank"
    root.mkdir()
    np.save(root / "c0.npy", np.ones((2, 4, 4), dtype=np.float32))
    np.save(root / "c0_w.npy", np.asarray([40.0, 50.0], dtype=np.float32))
    np.save(root / "c5.npy", np.ones((3, 4, 4), dtype=np.float32) * 2)
    np.save(root / "c5_w.npy", np.asarray([140.0, 145.0, 150.0], dtype=np.float32))
    manifest = {
        "schema": "tc_diffusion.sample_bank.v2",
        "run_name": "baseline_ztsnr",
        "bank_name": "fixture",
        "split": "test",
        "shards_by_class": {
            "0": [
                {
                    "part_index": 0,
                    "count": 2,
                    "bt_k": "c0.npy",
                    "wind_kt": "c0_w.npy",
                }
            ],
            "5": [
                {
                    "part_index": 0,
                    "count": 3,
                    "bt_k": "c5.npy",
                    "wind_kt": "c5_w.npy",
                }
            ],
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    archive = load_sample_bank(root)
    assert archive.fields.shape == (5, 4, 4)
    assert archive.fields[[4, 0], 0, 0].tolist() == [2.0, 1.0]
    assert archive.labels.tolist() == [0, 0, 5, 5, 5]


def test_tc_geometry_uses_shape_preserving_reflect_shifts():
    fields = np.arange(2 * 5 * 5, dtype=np.float32).reshape(2, 5, 5)
    transformed, trace = tc_geometry(fields, np.asarray([40.0, 140.0]), np.random.default_rng(4))
    assert transformed.shape == fields.shape
    assert trace["policy"] == "tc_rot90_reflect_shift_v1"
    assert trace["explicit_parameters"] is True
