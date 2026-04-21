# src/firelens/met/writer.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import xarray as xr
import zarr


DEFAULT_CHUNKS = {
    "time": 24,
    "y": 512,
    "x": 512,
}


def read_group_chunk_policy(
    store_zarr: str | Path,
    *,
    group: str,
) -> dict[str, int]:
    """
    Read chunk policy from a Zarr group attrs.

    Falls back to one day hourly and 512x512 spatial chunks.
    """
    ds = xr.open_zarr(store_zarr, group=group, consolidated=True)
    raw = ds.attrs.get("chunk_policy_json")

    if raw is None:
        return dict(DEFAULT_CHUNKS)

    loaded = json.loads(raw)

    return {
        "time": int(loaded.get("time") or DEFAULT_CHUNKS["time"]),
        "y": int(loaded.get("y") or DEFAULT_CHUNKS["y"]),
        "x": int(loaded.get("x") or DEFAULT_CHUNKS["x"]),
    }


def get_time_region_for_year(
    store_zarr: str | Path,
    *,
    group: str,
    year: int,
) -> tuple[slice, xr.DataArray]:
    """
    Return the contiguous time slice for a calendar year in a group.
    """
    ds = xr.open_zarr(store_zarr, group=group, consolidated=True)

    if "time" not in ds.coords:
        raise KeyError(f"{group!r} group has no time coordinate")

    times = ds["time"].values

    start = np.datetime64(f"{year}-01-01T00:00:00")
    end = np.datetime64(f"{year + 1}-01-01T00:00:00")

    idx = np.where((times >= start) & (times < end))[0]

    if len(idx) == 0:
        raise ValueError(f"No target times found for year={year} in group={group}")

    if not np.all(np.diff(idx) == 1):
        raise ValueError(f"Target time indices for year={year} are not contiguous")

    region = slice(int(idx[0]), int(idx[-1]) + 1)
    target_time = ds["time"].isel(time=region)

    return region, target_time


def ensure_time_yx_arrays(
    *,
    store_zarr: str | Path,
    group: str,
    var_specs: Mapping[str, Mapping],
    dtype: str = "float32",
    chunks: Mapping[str, int] | None = None,
    overwrite: bool = False,
) -> None:
    """
    Ensure variables with dims (time, y, x) exist in a Zarr group.

    This creates empty Zarr arrays with xarray-compatible _ARRAY_DIMENSIONS.
    """
    store_zarr = Path(store_zarr)
    template = xr.open_zarr(store_zarr, group=group, consolidated=True)

    for dim in ["time", "y", "x"]:
        if dim not in template.sizes:
            raise KeyError(f"{group!r} group is missing dimension {dim!r}")

    ntime = int(template.sizes["time"])
    ny = int(template.sizes["y"])
    nx = int(template.sizes["x"])

    if chunks is None:
        chunks = read_group_chunk_policy(store_zarr, group=group)

    chunk_tuple = (
        min(int(chunks["time"]), ntime),
        min(int(chunks["y"]), ny),
        min(int(chunks["x"]), nx),
    )

    root = zarr.open_group(str(store_zarr), mode="a")
    zg = root[group]

    for name, attrs in var_specs.items():
        if name in zg:
            if overwrite:
                print(f"Deleting existing /{group}/{name}")
                del zg[name]
            else:
                continue

        print(f"Creating /{group}/{name} with shape={(ntime, ny, nx)}, chunks={chunk_tuple}")

        arr = zg.create_dataset(
            name,
            shape=(ntime, ny, nx),
            chunks=chunk_tuple,
            dtype=dtype,
            fill_value=np.nan,
            overwrite=False,
        )

        arr.attrs.update(
            {
                "_ARRAY_DIMENSIONS": ["time", "y", "x"],
                **dict(attrs),
            }
        )


def write_time_yx_region(
    *,
    store_zarr: str | Path,
    group: str,
    ds_region: xr.Dataset,
    time_region: slice,
    var_names: list[str],
    chunks: Mapping[str, int] | None = None,
) -> None:
    """
    Write selected variables into an existing Zarr group over a time region.

    Variables must already exist in the target group.
    """
    store_zarr = Path(store_zarr)

    template = xr.open_zarr(store_zarr, group=group, consolidated=True)

    ny = int(template.sizes["y"])
    nx = int(template.sizes["x"])

    if chunks is None:
        chunks = read_group_chunk_policy(store_zarr, group=group)

    payload_vars = {}

    for name in var_names:
        if name not in ds_region:
            raise KeyError(f"{name!r} missing from ds_region")

        da = ds_region[name].transpose("time", "y", "x")

        if da.sizes["y"] != ny or da.sizes["x"] != nx:
            raise ValueError(
                f"{name} shape mismatch: got y/x={da.sizes['y']}/{da.sizes['x']}, "
                f"expected {ny}/{nx}"
            )

        payload_vars[name] = (
            ("time", "y", "x"),
            da.astype("float32").data,
        )

    payload = xr.Dataset(payload_vars)

    payload = payload.chunk(
        {
            "time": int(chunks["time"]),
            "y": int(chunks["y"]),
            "x": int(chunks["x"]),
        }
    )

    payload.to_zarr(
        store_zarr,
        group=group,
        mode="r+",
        consolidated=False,
        region={
            "time": time_region,
            "y": slice(0, ny),
            "x": slice(0, nx),
        },
    )