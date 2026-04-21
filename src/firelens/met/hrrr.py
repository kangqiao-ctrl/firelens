# src/firelens/met/hrrr.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe


HRRR_FIELD_SPECS = {
    "gust_surface": {
        "source_var": "gust",
        "filters": {"shortName": "gust", "typeOfLevel": "surface"},
        "attrs": {
            "long_name": "HRRR surface wind speed gust",
            "source_variable": "gust",
            "source_grib_shortName": "gust",
            "source_grib_name": "Wind speed (gust)",
            "source_grib_typeOfLevel": "surface",
            "units": "m s**-1",
            "source_dataset": "HRRR wrfsfcf00",
        },
    },
    "blh": {
        "source_var": "blh",
        "filters": {"shortName": "blh", "typeOfLevel": "surface"},
        "attrs": {
            "long_name": "HRRR planetary boundary layer height",
            "source_variable": "blh",
            "source_grib_shortName": "blh",
            "source_grib_name": "Boundary layer height",
            "source_grib_typeOfLevel": "surface",
            "units": "m",
            "source_dataset": "HRRR wrfsfcf00",
        },
    },
}

HRRR_VAR_ATTRS = {
    name: spec["attrs"]
    for name, spec in HRRR_FIELD_SPECS.items()
}
@dataclass(frozen=True)
class HrrrSubset:
    y_dim: str
    x_dim: str
    y_slice: slice
    x_slice: slice


def configure_eccodes_env(eccodes_root: str | Path | None = None) -> None:
    """
    Best-effort ecCodes environment configuration.

    Safest practice is still to export these before launching Python, but this
    helps when cfgrib/eccodes has not yet been imported in the current process.
    """
    if eccodes_root is None:
        eccodes_root = Path(os.environ["HOME"]) / "local" / "eccodes"

    eccodes_root = Path(eccodes_root)

    os.environ.setdefault("ECCODES_DIR", str(eccodes_root))
    os.environ.setdefault(
        "ECCODES_DEFINITION_PATH",
        str(eccodes_root / "share" / "eccodes" / "definitions"),
    )
    os.environ.setdefault(
        "ECCODES_SAMPLES_PATH",
        str(eccodes_root / "share" / "eccodes" / "samples"),
    )

    lib64 = str(eccodes_root / "lib64")
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if lib64 not in existing.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{lib64}:{existing}" if existing else lib64


def parse_hrrr_valid_time(path: str | Path) -> np.datetime64:
    """
    Parse valid time from paths like:

      .../hrrr.20170101/conus/hrrr.t00z.wrfsfcf00.grib2

    For f00, valid time equals initialization time. For fXX, valid time is
    init time + XX hours.
    """
    path = Path(path)

    date_match = re.search(r"hrrr\.(\d{8})", str(path))
    file_match = re.search(r"hrrr\.t(\d{2})z\.wrfsfcf(\d{2})", path.name)

    if date_match is None:
        raise ValueError(f"Could not parse HRRR date from path: {path}")

    if file_match is None:
        raise ValueError(f"Could not parse HRRR cycle/forecast hour from filename: {path.name}")

    ymd = date_match.group(1)
    cycle_hour = int(file_match.group(1))
    forecast_hour = int(file_match.group(2))

    init = pd.Timestamp(
        year=int(ymd[0:4]),
        month=int(ymd[4:6]),
        day=int(ymd[6:8]),
        hour=cycle_hour,
    )

    valid = init + pd.Timedelta(hours=forecast_hour)

    return np.datetime64(valid.to_datetime64(), "ns")


def find_hrrr_files(
    *,
    hrrr_root: str | Path,
    years: list[int],
    forecast_hour: int = 0,
    stream_dir: str | None = None,
) -> dict[np.datetime64, Path]:
    """
    Build valid_time -> HRRR file path map.

    Expected layout:
      hrrr_root/sfc_f00/hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcf00.grib2
    """
    hrrr_root = Path(hrrr_root)

    if stream_dir is None:
        stream_dir = f"sfc_f{forecast_hour:02d}"

    base = hrrr_root / stream_dir

    if not base.exists():
        raise FileNotFoundError(f"Missing HRRR stream directory: {base}")

    out: dict[np.datetime64, Path] = {}

    pattern = f"hrrr.*/conus/hrrr.t*z.wrfsfcf{forecast_hour:02d}.grib2"

    for path in sorted(base.glob(pattern)):
        valid_time = parse_hrrr_valid_time(path)
        year = pd.Timestamp(valid_time).year

        if year not in years:
            continue

        if valid_time in out:
            raise ValueError(f"Duplicate HRRR valid time {valid_time}: {out[valid_time]} and {path}")

        out[valid_time] = path

    return out


def open_cfgrib_field(
    grib_path: str | Path,
    *,
    output_name: str,
) -> xr.DataArray:
    spec = HRRR_FIELD_SPECS[output_name]

    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": spec["filters"],
            "indexpath": "",
        },
    )

    source_var = spec["source_var"]

    if source_var not in ds.data_vars:
        raise KeyError(
            f"Expected source variable {source_var!r} for output {output_name!r}. "
            f"Got data_vars={list(ds.data_vars)} from {grib_path}. "
            f"Filter was {spec['filters']}."
        )

    da = ds[source_var]
    da.name = output_name
    da.attrs.update(spec["attrs"])

    return da

def _rename_hrrr_latlon(da: xr.DataArray) -> xr.DataArray:
    rename = {}

    if "latitude" in da.coords:
        rename["latitude"] = "lat"

    if "longitude" in da.coords:
        rename["longitude"] = "lon"

    if rename:
        da = da.rename(rename)

    if "lat" not in da.coords or "lon" not in da.coords:
        raise KeyError("HRRR field must contain latitude/longitude or lat/lon coordinates.")

    return da


def _normalize_hrrr_lon_to_target(
    da: xr.DataArray,
    *,
    target_lon: xr.DataArray,
) -> xr.DataArray:
    """
    Normalize HRRR longitudes to match target convention, usually -180..180.
    """
    lon = da["lon"]

    target_min = float(target_lon.min())
    src_min = float(lon.min())
    src_max = float(lon.max())

    target_uses_negative = target_min < 0
    source_uses_0360 = src_min >= 0 and src_max > 180

    if target_uses_negative and source_uses_0360:
        lon_new = ((lon + 180) % 360) - 180
        da = da.assign_coords(lon=lon_new)

    return da


def _drop_non_spatial_scalar_coords(da: xr.DataArray) -> xr.DataArray:
    """
    Drop scalar GRIB coordinates before adding our own time dimension.

    Keep only 2-D lat/lon spatial coordinates.
    """
    keep = {"lat", "lon"}
    drop = [name for name in da.coords if name not in keep]

    if drop:
        da = da.drop_vars(drop, errors="ignore")

    return da


def open_hrrr_file_dataset(
    grib_path: str | Path,
    *,
    target_grid: xr.Dataset,
    subset: HrrrSubset | None = None,
    variables: list[str] | None = None,
) -> xr.Dataset:
    if variables is None:
        variables = ["gust_surface", "blh"]

    data_vars = {}
    coords_to_keep = None

    for out_name in variables:
        if out_name not in HRRR_FIELD_SPECS:
            raise KeyError(
                f"Unknown HRRR output variable {out_name!r}. "
                f"Valid: {list(HRRR_FIELD_SPECS)}"
            )

        da = open_cfgrib_field(
            grib_path,
            output_name=out_name,
        )

        da = _rename_hrrr_latlon(da)
        da = _normalize_hrrr_lon_to_target(da, target_lon=target_grid["lon"])

        if subset is not None:
            da = da.isel(
                {
                    subset.y_dim: subset.y_slice,
                    subset.x_dim: subset.x_slice,
                }
            )

        lat_dims = da["lat"].dims
        if len(lat_dims) != 2:
            raise ValueError(f"Expected 2-D HRRR lat/lon. Got lat dims={lat_dims}")

        y_dim, x_dim = lat_dims

        field = da.transpose(y_dim, x_dim).astype("float32")
        field = _drop_non_spatial_scalar_coords(field)

        field.name = out_name
        field.attrs.update(HRRR_FIELD_SPECS[out_name]["attrs"])

        # Optional but useful: load after subsetting so we do not keep many
        # GRIB file handles open across a block.
        field = field.load()

        data_vars[out_name] = field

        if coords_to_keep is None:
            coords_to_keep = {
                "lat": field["lat"],
                "lon": field["lon"],
            }

    ds = xr.Dataset(data_vars)

    if coords_to_keep is not None:
        ds = ds.assign_coords(coords_to_keep)

    valid_time = parse_hrrr_valid_time(grib_path)
    ds = ds.expand_dims(time=np.asarray([valid_time], dtype="datetime64[ns]"))

    return ds


def compute_hrrr_subset(
    *,
    sample_ds: xr.Dataset,
    target_grid: xr.Dataset,
    buffer_deg: float = 1.0,
) -> HrrrSubset:
    """
    Compute a rectangular native HRRR subset covering the target grid.
    """
    if "lat" not in sample_ds.coords or "lon" not in sample_ds.coords:
        raise KeyError("sample_ds must contain lat/lon coordinates")

    lat = sample_ds["lat"]
    lon = sample_ds["lon"]

    if lat.ndim != 2 or lon.ndim != 2:
        raise ValueError("Expected 2-D HRRR lat/lon coordinates")

    y_dim, x_dim = lat.dims

    lon_min = float(target_grid["lon"].min()) - buffer_deg
    lon_max = float(target_grid["lon"].max()) + buffer_deg
    lat_min = float(target_grid["lat"].min()) - buffer_deg
    lat_max = float(target_grid["lat"].max()) + buffer_deg

    mask = (
        (lon >= lon_min)
        & (lon <= lon_max)
        & (lat >= lat_min)
        & (lat <= lat_max)
    )

    idx = np.argwhere(mask.values)

    if idx.size == 0:
        raise ValueError(
            "HRRR subset is empty. "
            f"lon range=({lon_min}, {lon_max}), lat range=({lat_min}, {lat_max})"
        )

    y0, x0 = idx.min(axis=0)
    y1, x1 = idx.max(axis=0)

    # Add a small native-grid halo.
    y0 = max(0, int(y0) - 2)
    x0 = max(0, int(x0) - 2)
    y1 = min(lat.sizes[y_dim] - 1, int(y1) + 2)
    x1 = min(lat.sizes[x_dim] - 1, int(x1) + 2)

    return HrrrSubset(
        y_dim=y_dim,
        x_dim=x_dim,
        y_slice=slice(y0, y1 + 1),
        x_slice=slice(x0, x1 + 1),
    )


def open_met_grid(met_zarr: str | Path) -> xr.Dataset:
    """
    Open FireLens met store /grid group and load 2-D lon/lat.
    """
    grid = xr.open_zarr(
        met_zarr,
        group="grid",
        consolidated=True,
    )

    for name in ["x", "y", "lon", "lat"]:
        if name not in grid.coords and name not in grid:
            raise KeyError(
                f"met store /grid must contain {name!r}. "
                "If lon/lat are missing, reinitialize context/met stores without --no-latlon."
            )

    # xESMF weight generation works more predictably with concrete 2-D lon/lat.
    grid["lon"].load()
    grid["lat"].load()

    return grid


def build_or_load_hrrr_regridder(
    *,
    sample_file: str | Path,
    met_zarr: str | Path,
    weights_nc: str | Path,
    buffer_deg: float = 1.0,
    method: str = "bilinear",
) -> tuple[xr.Dataset, HrrrSubset, xe.Regridder]:
    """
    Build/reuse xESMF weights from native HRRR grid to FireLens /grid.
    """
    target_grid = open_met_grid(met_zarr)

    sample_full = open_hrrr_file_dataset(
        sample_file,
        target_grid=target_grid,
        subset=None,
        variables=["gust_surface"],
    )

    subset = compute_hrrr_subset(
        sample_ds=sample_full,
        target_grid=target_grid,
        buffer_deg=buffer_deg,
    )

    sample_sub = sample_full.isel(
        {
            subset.y_dim: subset.y_slice,
            subset.x_dim: subset.x_slice,
        }
    )

    weights_nc = Path(weights_nc)
    weights_nc.parent.mkdir(parents=True, exist_ok=True)

    grid_in = xr.Dataset(
        coords={
            "lon": sample_sub["lon"],
            "lat": sample_sub["lat"],
        }
    )

    grid_out = xr.Dataset(
        coords={
            "lon": target_grid["lon"],
            "lat": target_grid["lat"],
        }
    )

    reuse = weights_nc.exists()

    regridder = xe.Regridder(
        grid_in,
        grid_out,
        method,
        filename=str(weights_nc),
        reuse_weights=reuse,
    )

    return target_grid, subset, regridder


def read_hrrr_block(
    *,
    paths: list[Path],
    target_grid: xr.Dataset,
    subset: HrrrSubset,
    variables: list[str],
) -> xr.Dataset:
    """
    Read and concatenate a block of HRRR files.
    """
    pieces = []

    for path in paths:
        ds = open_hrrr_file_dataset(
            path,
            target_grid=target_grid,
            subset=subset,
            variables=variables,
        )
        pieces.append(ds)

    if not pieces:
        raise ValueError("No HRRR files provided for block")

    out = xr.concat(pieces, dim="time")
    out = out.sortby("time")

    return out


def regrid_hrrr_block_to_met_grid(
    *,
    hrrr_block: xr.Dataset,
    target_grid: xr.Dataset,
    regridder: xe.Regridder,
    target_time: xr.DataArray,
    variables: list[str],
    time_chunk: int = 24,
    output_chunks: dict[str, int] | None = None,
) -> xr.Dataset:
    """
    Regrid a HRRR block to FireLens met grid.

    Missing target hours become NaN after reindexing.
    """
    if len(target_time) == 0:
        raise ValueError("target_time block is empty")

    ds = hrrr_block[variables].reindex(time=target_time.values)

    chunk_spec = {"time": time_chunk}
    if "y" in ds.dims:
        chunk_spec["y"] = -1
    if "x" in ds.dims:
        chunk_spec["x"] = -1

    ds = ds.chunk(chunk_spec)

    if output_chunks is None:
        output_chunks = {"y": 512, "x": 512}

    out = regridder(
        ds,
        output_chunks=output_chunks,
    )

    out = out.assign_coords(
        time=target_time.values,
        y=target_grid["y"],
        x=target_grid["x"],
    )

    out = out.transpose("time", "y", "x")

    return out