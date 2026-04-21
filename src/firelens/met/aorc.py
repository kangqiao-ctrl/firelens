# src/firelens/met/aorc.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
import xesmf as xe


AORC_RENAME_DEFAULT = {
    "TMP_2maboveground": "tmp_2m",
    "SPFH_2maboveground": "spfh_2m",
    "PRES_surface": "pres_surface",
    "UGRD_10maboveground": "ugrd_10m",
    "VGRD_10maboveground": "vgrd_10m",
    "DSWRF_surface": "dswrf_surface",
    "DLWRF_surface": "dlwrf_surface",
    "APCP_surface": "apcp_surface",
}


AORC_VAR_ATTRS = {
    "tmp_2m": {
        "long_name": "AORC 2 m air temperature",
        "source_variable": "TMP_2maboveground",
    },
    "spfh_2m": {
        "long_name": "AORC 2 m specific humidity",
        "source_variable": "SPFH_2maboveground",
    },
    "pres_surface": {
        "long_name": "AORC surface pressure",
        "source_variable": "PRES_surface",
    },
    "ugrd_10m": {
        "long_name": "AORC 10 m U wind component",
        "source_variable": "UGRD_10maboveground",
    },
    "vgrd_10m": {
        "long_name": "AORC 10 m V wind component",
        "source_variable": "VGRD_10maboveground",
    },
    "dswrf_surface": {
        "long_name": "AORC downward shortwave radiation flux",
        "source_variable": "DSWRF_surface",
    },
    "dlwrf_surface": {
        "long_name": "AORC downward longwave radiation flux",
        "source_variable": "DLWRF_surface",
    },
    "apcp_surface": {
        "long_name": "AORC hourly precipitation accumulation",
        "source_variable": "APCP_surface",
        "time_semantics": "hourly accumulation; no sub-hourly splitting in met_500m_1h store",
    },
}


def _ordered_slice(coord_1d: xr.DataArray, lo: float, hi: float) -> slice:
    vals = np.asarray(coord_1d.values)

    if vals.ndim != 1:
        raise ValueError("Expected 1-D coordinate.")

    if len(vals) < 2:
        raise ValueError("Coordinate is too short.")

    if vals[0] < vals[-1]:
        return slice(lo, hi)

    return slice(hi, lo)


def open_met_grid(met_zarr: str | Path) -> xr.Dataset:
    """
    Open FireLens met store /grid group.

    Requires 2-D lon/lat for xESMF target grid construction.
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
                "If lon/lat are missing, reinitialize the context/met stores without --no-latlon."
            )

    # Loading lon/lat is reasonable here: about tens of MB, and xESMF needs them for weights.
    grid = grid[["domain_mask"]] if "domain_mask" in grid else grid
    grid = grid.assign_coords(
        x=xr.open_zarr(met_zarr, group="grid", consolidated=True)["x"],
        y=xr.open_zarr(met_zarr, group="grid", consolidated=True)["y"],
        lon=xr.open_zarr(met_zarr, group="grid", consolidated=True)["lon"].load(),
        lat=xr.open_zarr(met_zarr, group="grid", consolidated=True)["lat"].load(),
    )

    return grid


def open_aorc_year(aorc_year_zarr: str | Path) -> xr.Dataset:
    """
    Open one AORC year Zarr and normalize coordinate names.
    """
    ds = xr.open_zarr(
        aorc_year_zarr,
        consolidated=None,
        decode_cf=True,
        mask_and_scale=True,
        decode_times=True,
    )

    rename = {}

    if "latitude" in ds.coords:
        rename["latitude"] = "lat"

    if "longitude" in ds.coords:
        rename["longitude"] = "lon"

    if rename:
        ds = ds.rename(rename)

    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise KeyError("AORC dataset must contain lat/lon or latitude/longitude coordinates.")

    if "time" not in ds.coords:
        raise KeyError("AORC dataset must contain a time coordinate.")

    return ds


def normalize_source_longitude_to_target(
    src: xr.Dataset,
    *,
    target_lon: xr.DataArray,
) -> xr.Dataset:
    """
    Convert source lon convention to match target lon convention.

    FireLens target lon is usually -180..180. Some source products may use 0..360.
    """
    src_lon = src["lon"]

    target_min = float(target_lon.min())
    src_min = float(src_lon.min())
    src_max = float(src_lon.max())

    target_uses_negative = target_min < 0
    source_uses_0360 = src_min >= 0 and src_max > 180

    if target_uses_negative and source_uses_0360:
        lon_new = ((src_lon + 180) % 360) - 180
        src = src.assign_coords(lon=lon_new).sortby("lon")

    return src


def subset_aorc_to_target_extent(
    src: xr.Dataset,
    *,
    target_grid: xr.Dataset,
    buffer_deg: float = 1.0,
) -> xr.Dataset:
    """
    Spatially subset AORC source around the FireLens target grid.
    """
    if "lon" not in target_grid.coords and "lon" not in target_grid:
        raise KeyError("target_grid must contain 2-D lon.")

    if "lat" not in target_grid.coords and "lat" not in target_grid:
        raise KeyError("target_grid must contain 2-D lat.")

    src = normalize_source_longitude_to_target(
        src,
        target_lon=target_grid["lon"],
    )

    lon_min = float(target_grid["lon"].min()) - buffer_deg
    lon_max = float(target_grid["lon"].max()) + buffer_deg
    lat_min = float(target_grid["lat"].min()) - buffer_deg
    lat_max = float(target_grid["lat"].max()) + buffer_deg

    lon_sel = _ordered_slice(src["lon"], lon_min, lon_max)
    lat_sel = _ordered_slice(src["lat"], lat_min, lat_max)

    src_sub = src.sel(lon=lon_sel, lat=lat_sel)

    if src_sub.sizes.get("lon", 0) == 0 or src_sub.sizes.get("lat", 0) == 0:
        raise ValueError(
            "AORC spatial subset is empty. "
            f"lon range=({lon_min}, {lon_max}), lat range=({lat_min}, {lat_max})"
        )

    return src_sub


def build_or_load_aorc_regridder_from_subset(
    *,
    src_sub: xr.Dataset,
    target_grid: xr.Dataset,
    weights_nc: str | Path,
    method: str = "bilinear",
) -> xe.Regridder:
    """
    Build or load an xESMF regridder from an already-subset AORC grid to FireLens /grid.
    """
    weights_nc = Path(weights_nc)
    weights_nc.parent.mkdir(parents=True, exist_ok=True)

    grid_in = xr.Dataset(
        coords={
            "lon": src_sub["lon"],
            "lat": src_sub["lat"],
        }
    )

    grid_out = xr.Dataset(
        coords={
            "lon": target_grid["lon"],
            "lat": target_grid["lat"],
        }
    )

    reuse = weights_nc.exists()

    return xe.Regridder(
        grid_in,
        grid_out,
        method,
        filename=str(weights_nc),
        reuse_weights=reuse,
    )


def open_aorc_year_subset_and_regridder(
    *,
    aorc_year_zarr: str | Path,
    met_zarr: str | Path,
    weights_nc: str | Path,
    buffer_deg: float = 1.0,
    method: str = "bilinear",
) -> tuple[xr.Dataset, xr.Dataset, xe.Regridder]:
    """
    Open one AORC year, spatially subset it, and build/load the regridder.
    """
    target_grid = open_met_grid(met_zarr)
    src = open_aorc_year(aorc_year_zarr)

    src_sub = subset_aorc_to_target_extent(
        src,
        target_grid=target_grid,
        buffer_deg=buffer_deg,
    )

    regridder = build_or_load_aorc_regridder_from_subset(
        src_sub=src_sub,
        target_grid=target_grid,
        weights_nc=weights_nc,
        method=method,
    )

    return src_sub, target_grid, regridder


def standardize_aorc_variables(
    ds: xr.Dataset,
    *,
    variables: list[str],
    rename_map: dict[str, str] | None = None,
) -> xr.Dataset:
    """
    Keep selected AORC variables and rename them to FireLens-friendly names.
    """
    rename_map = rename_map or AORC_RENAME_DEFAULT

    missing = [v for v in variables if v not in ds.data_vars]

    if missing:
        raise KeyError(f"AORC variables missing from source dataset: {missing}")

    out = ds[variables]

    rename = {
        v: rename_map.get(v, v.lower())
        for v in variables
    }

    out = out.rename(rename)

    for src_name, dst_name in rename.items():
        attrs = dict(out[dst_name].attrs)
        attrs.update(AORC_VAR_ATTRS.get(dst_name, {}))
        attrs["source_variable"] = src_name
        attrs["source_dataset"] = "AORC"
        out[dst_name].attrs = attrs

    return out

def regrid_aorc_block_to_met_grid(
    *,
    src_sub: xr.Dataset,
    target_grid: xr.Dataset,
    regridder: xe.Regridder,
    target_time: xr.DataArray,
    variables: list[str],
    time_chunk: int = 24,
    output_chunks: dict[str, int] | None = None,
) -> xr.Dataset:
    """
    Regrid a block of AORC times to FireLens met grid.

    Output dims: time, y, x
    """
    if len(target_time) == 0:
        raise ValueError("target_time block is empty.")

    # Align source to target hourly times. Missing source hours become NaN.
    src_block = src_sub.reindex(time=target_time.values)

    src_vars = standardize_aorc_variables(
        src_block,
        variables=variables,
    )

    src_vars = standardize_aorc_variables(
        src_block,
        variables=variables,
    )

    # Keep time chunked, but avoid many small source spatial chunks.
    # xESMF otherwise tries to preserve source lat/lon chunk sizes on the output grid.
    chunk_spec = {"time": time_chunk}
    if "lat" in src_vars.dims:
        chunk_spec["lat"] = -1
    if "lon" in src_vars.dims:
        chunk_spec["lon"] = -1

    src_vars = src_vars.chunk(chunk_spec)

    if output_chunks is None:
        output_chunks = {"y": 512, "x": 512}

    out = regridder(
        src_vars,
        output_chunks=output_chunks,
    )

    return out