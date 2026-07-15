import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from gauge_climate.xarray_adapter import from_xarray


def test_xarray_adapter_reorders_sample_dimension_and_coordinates():
    data = xr.DataArray(
        np.zeros((3, 4, 5)),
        dims=("y", "sample", "x"),
        coords={"storm": ("sample", ["a", "b", "c", "d"])},
        attrs={"units": "K"},
    )
    dataset = from_xarray(data, sample_dim="sample", block_coord="storm")
    assert dataset.sample_shape == (3, 5)
    assert dataset.units == "K"
    assert dataset.metadata["xarray_dims"] == ["sample", "y", "x"]


def test_xarray_adapter_rejects_wrong_coordinate_and_controls_lazy_materialization():
    da = pytest.importorskip("dask.array")
    data = xr.DataArray(
        da.zeros((4, 8, 8), chunks=(2, 8, 8)),
        dims=("sample", "y", "x"),
        coords={"bad": ("y", np.arange(8))},
    )
    with pytest.raises(ValueError, match="vary only"):
        from_xarray(data, sample_dim="sample", block_coord="bad")
    dataset = from_xarray(data, sample_dim="sample")
    with pytest.raises(MemoryError):
        dataset.as_numpy(max_bytes=8)
    assert dataset.as_numpy(max_bytes=10_000).shape == (4, 8, 8)
