# src/firelens/slow_context/writer.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
import zarr


def _assert_same_xy(ds: xr.Dataset, template: xr.Dataset) -> None:
    for coord in ["x", "y"]:
        if coord not in ds.coords:
            raise KeyError(f"Dataset missing {coord!r} coordinate.")
        if coord not in template.coords:
            raise KeyError(f"Template slow_context group missing {coord!r} coordinate.")

        a = np.asarray(ds[coord].values)
        b = np.asarray(template[coord].values)

        if a.shape != b.shape or not np.allclose(a, b):
            raise ValueError(f"{coord} coordinate does not match slow_context group.")


def _chunks_for_var(
    da: xr.DataArray,
    *,
    chunks_yx: tuple[int, int],
    vintage_chunk: int,
) -> tuple[int, ...]:
    chunks = []

    for dim in da.dims:
        if dim == "vintage":
            chunks.append(vintage_chunk)
        elif dim == "y":
            chunks.append(chunks_yx[0])
        elif dim == "x":
            chunks.append(chunks_yx[1])
        else:
            chunks.append(1)

    return tuple(chunks)


def write_slow_context_dataset(
    *,
    cube_zarr: str | Path,
    ds: xr.Dataset,
    overwrite_vars: bool = False,
    chunks_yx: tuple[int, int] = (512, 512),
    vintage_chunk: int = 1,
    zarr_format: int = 2,
) -> xr.Dataset:
    """
    Write versioned slow-context variables into /slow_context.
    """
    cube_zarr = Path(cube_zarr)

    if "vintage" not in ds.dims:
        raise ValueError("Landfire slow-context dataset must have a 'vintage' dimension.")

    slow = xr.open_zarr(cube_zarr, group="slow_context", consolidated=True)
    _assert_same_xy(ds, slow)

    if "spatial_ref" in slow.coords and "spatial_ref" not in ds.coords:
        ds = ds.assign_coords(spatial_ref=slow["spatial_ref"])
    elif "spatial_ref" in slow and "spatial_ref" not in ds:
        ds["spatial_ref"] = slow["spatial_ref"]

    root = zarr.open_group(str(cube_zarr), mode="a")
    zg = root["slow_context"]

    existing = [name for name in ds.data_vars if name in zg]

    if existing and not overwrite_vars:
        raise ValueError(
            "These variables already exist in /slow_context: "
            + ", ".join(existing)
            + ". Pass overwrite_vars=True to replace them."
        )

    if existing and overwrite_vars:
        for name in existing:
            print(f"Deleting existing /slow_context/{name}")
            del zg[name]

    encoding = {
        name: {
            "dtype": "float32",
            "chunks": _chunks_for_var(
                da,
                chunks_yx=chunks_yx,
                vintage_chunk=vintage_chunk,
            ),
        }
        for name, da in ds.data_vars.items()
    }

    print(f"Writing Landfire variables to /slow_context")
    print(f"Variables: {list(ds.data_vars)}")

    ds.to_zarr(
        cube_zarr,
        group="slow_context",
        mode="a",
        zarr_format=zarr_format,
        consolidated=False,
        encoding=encoding,
        align_chunks=True,
    )

    zarr.consolidate_metadata(str(cube_zarr))
    return ds