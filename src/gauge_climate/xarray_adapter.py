from __future__ import annotations

from typing import Any

from .data import Dataset


def from_xarray(
    data: Any,
    *,
    sample_dim: str,
    block_coord: str | None = None,
    label_coord: str | None = None,
    sample_id_coord: str | None = None,
    name: str | None = None,
    units: str | None = None,
    allow_nan: bool = False,
) -> Dataset:
    """Create a GAUGE Dataset from an xarray DataArray without eager loading."""

    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - exercised without the optional extra
        raise ImportError("Install gauge-climate[xarray] to use the xarray adapter.") from exc
    if not isinstance(data, xr.DataArray):
        raise TypeError("from_xarray expects an xarray.DataArray.")
    if sample_dim not in data.dims:
        raise ValueError(f"sample_dim {sample_dim!r} is not present in {data.dims}.")
    ordered = data.transpose(sample_dim, *(dim for dim in data.dims if dim != sample_dim))

    def coordinate(coord: str | None):
        if coord is None:
            return None
        if coord not in ordered.coords:
            raise ValueError(f"Coordinate {coord!r} is not present.")
        values = ordered.coords[coord]
        if values.dims != (sample_dim,):
            raise ValueError(f"Coordinate {coord!r} must vary only along {sample_dim!r}.")
        return values.values

    return Dataset(
        ordered,
        blocks=coordinate(block_coord),
        labels=coordinate(label_coord),
        sample_ids=coordinate(sample_id_coord),
        name=name or str(data.name or "xarray_dataset"),
        units=units if units is not None else data.attrs.get("units"),
        allow_nan=allow_nan,
        metadata={"xarray_dims": list(ordered.dims)},
    )
