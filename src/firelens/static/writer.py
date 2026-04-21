# src/firelens/static/writer.py
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import xarray as xr
import zarr

from firelens.grid import transform_from_xy, _affine_transform_coeffs


#def _affine6(transform) -> tuple[float, float, float, float, float, float]:
    #"""Return Affine coefficients as a,b,c,d,e,f."""
    #return (
    #    float(transform.a),
    #    float(transform.b),
    #    float(transform.c),
    #    float(transform.d),
    #    float(transform.e),
    #    float(transform.f),
    #)


def _validate_static_group(static: xr.Dataset) -> None:
    if "x" not in static.coords or "y" not in static.coords:
        raise KeyError("Static group must contain 1-D x and y coordinates.")

    x = np.asarray(static["x"].values)
    y = np.asarray(static["y"].values)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Expected 1-D x/y coordinates in static group.")

    if len(x) < 2 or len(y) < 2:
        raise ValueError("Static group x/y coordinates are too short.")

    if not np.all(np.diff(x) > 0):
        raise ValueError("Expected x coordinate to be increasing.")

    if not np.all(np.diff(y) < 0):
        raise ValueError("Expected y coordinate to be decreasing.")


def infer_static_yx_chunks(
    *,
    cube_zarr: str | Path,
    preferred_vars: tuple[str, ...] = ("domain_mask", "lon", "lat"),
    fallback: tuple[int, int] = (512, 512),
) -> tuple[int, int]:
    """
    Infer y/x chunks from an existing variable in /static.
    """
    root = zarr.open_group(str(cube_zarr), mode="a")
    zg = root["static"]

    for name in preferred_vars:
        if name in zg:
            chunks = tuple(zg[name].chunks)
            if len(chunks) >= 2:
                return tuple(int(v) for v in chunks[-2:])

    return fallback


def load_aligned_tif_as_dataarray(
    *,
    path: str | Path,
    static: xr.Dataset,
    var_name: str,
    long_name: str,
    units: str,
    expected_epsg: int = 5070,
    transform_atol: float = 1e-6,
) -> xr.DataArray:
    """
    Load one aligned GeoTIFF and validate it against the /static grid.

    Checks:
    - CRS is expected_epsg
    - shape equals static y/x shape
    - transform equals static transform
    - nodata is converted to NaN
    """
    _validate_static_group(static)

    path = Path(path)

    expected_shape = (static.sizes["y"], static.sizes["x"])
    expected_transform = transform_from_xy(static["x"].values, static["y"].values)

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"{path} has no CRS.")

        if src.crs.to_epsg() != expected_epsg:
            raise ValueError(f"{path} CRS is not EPSG:{expected_epsg}: {src.crs}")

        if (src.height, src.width) != expected_shape:
            raise ValueError(
                f"{path} shape {(src.height, src.width)} != expected {expected_shape}"
            )

        if not np.allclose(
            _affine_transform_coeffs(src.transform),
            _affine_transform_coeffs(expected_transform),
            atol=transform_atol,
        ):
            raise ValueError(
                f"{path} transform does not match static grid.\n"
                f"src={src.transform}\n"
                f"expected={expected_transform}"
            )

        arr = src.read(1).astype(np.float32)

        if src.nodata is not None:
            try:
                nodata_is_nan = bool(np.isnan(src.nodata))
            except TypeError:
                nodata_is_nan = False

            if nodata_is_nan:
                arr = np.where(np.isnan(arr), np.nan, arr)
            else:
                arr = np.where(arr == src.nodata, np.nan, arr)

    return xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={
            "y": static["y"],
            "x": static["x"],
        },
        name=var_name,
        attrs={
            "long_name": long_name,
            "units": units,
            "grid_mapping": "spatial_ref",
            "source_file": str(path),
            "crs": f"EPSG:{expected_epsg}",
        },
    )


def build_static_dataset_from_tifs(
    *,
    cube_zarr: str | Path,
    var_specs: Mapping[str, Mapping[str, Any]],
    consolidated: bool | None = True,
    expected_epsg: int = 5070,
) -> xr.Dataset:
    """
    Build an xarray Dataset from aligned static GeoTIFFs.

    var_specs format:
        {
          "dem_m": {
            "path": "...",
            "long_name": "...",
            "units": "m"
          },
          ...
        }
    """
    static = xr.open_zarr(cube_zarr, group="static", consolidated=consolidated)
    _validate_static_group(static)

    data_vars = {}

    for var_name, spec in var_specs.items():
        print(f"Loading {var_name} <- {spec['path']}")

        data_vars[var_name] = load_aligned_tif_as_dataarray(
            path=spec["path"],
            static=static,
            var_name=var_name,
            long_name=spec.get("long_name", var_name),
            units=spec.get("units", "unknown"),
            expected_epsg=expected_epsg,
        )

    ds = xr.Dataset(data_vars)

    if "spatial_ref" in static.coords:
        ds = ds.assign_coords(spatial_ref=static["spatial_ref"])
    elif "spatial_ref" in static:
        ds["spatial_ref"] = static["spatial_ref"]

    return ds


def write_static_group(
    *,
    cube_zarr: str | Path,
    var_specs: Mapping[str, Mapping[str, Any]],
    consolidated: bool | None = True,
    expected_epsg: int = 5070,
    chunks_yx: tuple[int, int] | None = None,
    overwrite_vars: bool = False,
    zarr_format: int = 2,
) -> xr.Dataset:
    """
    Validate aligned static TIFFs and write them into the /static Zarr group.
    """
    cube_zarr = Path(cube_zarr)

    ds_new = build_static_dataset_from_tifs(
        cube_zarr=cube_zarr,
        var_specs=var_specs,
        consolidated=consolidated,
        expected_epsg=expected_epsg,
    )

    if chunks_yx is None:
        chunks_yx = infer_static_yx_chunks(cube_zarr=cube_zarr)

    root = zarr.open_group(str(cube_zarr), mode="a")
    zg = root["static"]

    existing = [name for name in ds_new.data_vars if name in zg]

    if existing and not overwrite_vars:
        raise ValueError(
            "These variables already exist in /static: "
            + ", ".join(existing)
            + ". Pass overwrite_vars=True to replace them."
        )

    if existing and overwrite_vars:
        for name in existing:
            print(f"Deleting existing /static/{name}")
            del zg[name]

    encoding = {
        name: {
            "dtype": "float32",
            "chunks": chunks_yx,
        }
        for name in ds_new.data_vars
        if name != "spatial_ref"
    }

    print(f"Writing variables to /static with chunks={chunks_yx}")

    ds_new.to_zarr(
        cube_zarr,
        group="static",
        mode="a",
        consolidated=False,
        zarr_format=zarr_format,
        encoding=encoding,
    )

    zarr.consolidate_metadata(str(cube_zarr))

    print("Done. Variables written:")
    print(list(ds_new.data_vars))

    return ds_new