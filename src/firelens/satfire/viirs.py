# src/firelens/satfire/viirs.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from pyproj import Transformer

from firelens.grid import TargetGrid, build_target_grid_from_zarr


VIIRS_VAR_SPECS: dict[str, dict[str, Any]] = {
    "viirs_detect_count": {
        "dtype": "uint16",
        "fill_value": 0,
        "attrs": {
            "long_name": "VIIRS active-fire detection count",
            "units": "1",
        },
    },
    "viirs_frp_sum": {
        "dtype": "float32",
        "fill_value": 0.0,
        "attrs": {
            "long_name": "Sum of VIIRS fire radiative power detections",
            "units": "MW",
        },
    },
    "viirs_frp_max": {
        "dtype": "float32",
        "fill_value": 0.0,
        "attrs": {
            "long_name": "Maximum VIIRS fire radiative power detection",
            "units": "MW",
        },
    },
    "viirs_confidence_max": {
        "dtype": "uint8",
        "fill_value": 0,
        "attrs": {
            "long_name": "Maximum VIIRS detection confidence score",
            "units": "1",
            "confidence_encoding": "0=missing/no detection; numeric source values preserved when numeric; string confidence mapped low/nominal/high to 1/2/3",
        },
    },
    "viirs_detected_any": {
        "dtype": "uint8",
        "fill_value": 0,
        "attrs": {
            "long_name": "Any VIIRS active-fire detection in cell/time bin",
            "units": "1",
            "flag_values": [0, 1],
            "flag_meanings": "not_detected detected",
        },
    },
}


def log(msg: str) -> None:
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def task_id_to_year_month(
    task_id: int,
    *,
    start_year: int,
    end_year: int,
) -> tuple[int, int]:
    years = list(range(start_year, end_year + 1))
    n_months = len(years) * 12

    if task_id < 0 or task_id >= n_months:
        raise ValueError(f"task_id={task_id} out of range [0, {n_months - 1}]")

    year = years[task_id // 12]
    month = (task_id % 12) + 1

    return year, month


def infer_time_frequency(time_index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(time_index) < 2:
        raise ValueError("Time index too short to infer frequency")

    dt = pd.Timedelta(time_index[1] - time_index[0])

    if dt <= pd.Timedelta(0):
        raise ValueError(f"Expected increasing time index, got dt={dt}")

    return dt


def month_interval(year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=year, month=month, day=1)

    if month == 12:
        end_exclusive = pd.Timestamp(year=year + 1, month=1, day=1)
    else:
        end_exclusive = pd.Timestamp(year=year, month=month + 1, day=1)

    return start, end_exclusive


def time_region_for_interval(
    time_index: pd.DatetimeIndex,
    *,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> slice:
    i0 = int(time_index.searchsorted(start, side="left"))
    i1 = int(time_index.searchsorted(end_exclusive, side="left"))

    if i1 <= i0:
        raise ValueError(f"Empty time region for interval {start} -> {end_exclusive}")

    return slice(i0, i1)


def read_events_month(
    events_root: str | Path,
    *,
    year: int,
    month: int,
) -> pd.DataFrame:
    events_root = Path(events_root)

    df = pd.read_parquet(
        events_root,
        filters=[
            ("year", "=", year),
            ("month", "=", month),
        ],
    )

    return df


def _pick_column(
    df: pd.DataFrame,
    *,
    requested: str | None,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    if requested is not None:
        if requested not in df.columns:
            raise KeyError(f"Requested column {requested!r} not found. Columns: {list(df.columns)}")
        return requested

    lower_to_real = {c.lower(): c for c in df.columns}

    for c in candidates:
        if c.lower() in lower_to_real:
            return lower_to_real[c.lower()]

    if required:
        raise KeyError(f"Could not find any of {candidates}. Columns: {list(df.columns)}")

    return None


def confidence_to_uint8(s: pd.Series) -> pd.Series:
    """
    Convert VIIRS confidence to uint8-compatible numeric scores.

    Supports either numeric confidence or common string categories.
    """
    numeric = pd.to_numeric(s, errors="coerce")

    if numeric.notna().any():
        return numeric.fillna(0).clip(lower=0, upper=255).astype(np.uint8)

    mapping = {
        "l": 1,
        "low": 1,
        "n": 2,
        "nominal": 2,
        "m": 2,
        "medium": 2,
        "h": 3,
        "high": 3,
    }

    out = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(0)
        .astype(np.uint8)
    )

    return out


def standardize_viirs_events(
    df: pd.DataFrame,
    *,
    obs_time_col: str | None = None,
    lat_col: str | None = None,
    lon_col: str | None = None,
    frp_col: str | None = None,
    confidence_col: str | None = None,
) -> pd.DataFrame:
    """
    Standardize event columns to:

      obs_time
      lat
      lon
      frp
      confidence_score
    """
    if len(df) == 0:
        return df.copy()

    obs_time_col = _pick_column(
        df,
        requested=obs_time_col,
        candidates=["obs_time", "acq_datetime", "datetime", "time"],
    )

    lat_col = _pick_column(
        df,
        requested=lat_col,
        candidates=["lat", "latitude"],
    )

    lon_col = _pick_column(
        df,
        requested=lon_col,
        candidates=["lon", "longitude"],
    )

    frp_col = _pick_column(
        df,
        requested=frp_col,
        candidates=["frp", "power", "bright_ti4_power"],
    )

    confidence_col = _pick_column(
        df,
        requested=confidence_col,
        candidates=["confidence", "conf", "confidence_score"],
        required=False,
    )

    out = pd.DataFrame(
        {
            "obs_time": pd.to_datetime(df[obs_time_col], utc=True),
            "lat": pd.to_numeric(df[lat_col], errors="coerce"),
            "lon": pd.to_numeric(df[lon_col], errors="coerce"),
            "frp": pd.to_numeric(df[frp_col], errors="coerce").fillna(0.0),
        }
    )

    if confidence_col is None:
        out["confidence_score"] = np.uint8(0)
    else:
        out["confidence_score"] = confidence_to_uint8(df[confidence_col])

    out = out.dropna(subset=["obs_time", "lat", "lon"])

    return out


def assign_time_bins(
    events: pd.DataFrame,
    *,
    time_index: pd.DatetimeIndex,
    dt: pd.Timedelta,
    bin_policy: str = "interval_end",
) -> pd.DataFrame:
    """
    Assign store time indices to VIIRS events.

    bin_policy:
      - floor: event at 12:07 goes to 12:00
      - ceil:  event at 12:07 goes to 12:15
      - nearest: nearest store timestamp

    Use ceil if your store timestamp represents the end of the aggregation interval.
    Use floor to preserve the behavior of the previous script.
    """
    if len(events) == 0:
        return events.copy()

    out = events.copy()

    obs_naive = (
        out["obs_time"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )

    if bin_policy == "floor":
        bin_time = obs_naive.dt.floor(dt)
    elif bin_policy == "ceil":
        bin_time = obs_naive.dt.ceil(dt)
    elif bin_policy == "nearest":
        bin_time = obs_naive.dt.round(dt)
    elif bin_policy == "interval_end":
        bin_time = obs_naive.dt.floor(dt) + dt
    else:
        raise ValueError(f"Unknown bin_policy={bin_policy!r}")

    time_idx = time_index.get_indexer(bin_time)

    keep = time_idx >= 0

    out = out.loc[keep].copy()
    out["bin_time"] = bin_time.loc[keep].to_numpy(dtype="datetime64[ns]")
    out["time_idx"] = time_idx[keep].astype(np.int64)

    return out


def attach_grid_indices_from_lonlat(
    events: pd.DataFrame,
    *,
    target: TargetGrid,
) -> pd.DataFrame:
    """
    Map VIIRS lon/lat points to target-grid row/col using the target CRS and affine transform.
    """
    if len(events) == 0:
        return events.copy()

    transformer = Transformer.from_crs(
        "EPSG:4326",
        target.crs,
        always_xy=True,
    )

    lon = events["lon"].to_numpy(dtype=np.float64)
    lat = events["lat"].to_numpy(dtype=np.float64)

    x_proj, y_proj = transformer.transform(lon, lat)

    inv = ~target.transform

    # Vectorized affine inverse.
    col_f = inv.a * x_proj + inv.b * y_proj + inv.c
    row_f = inv.d * x_proj + inv.e * y_proj + inv.f

    col = np.floor(col_f).astype(np.int64)
    row = np.floor(row_f).astype(np.int64)

    keep = (
        np.isfinite(col_f)
        & np.isfinite(row_f)
        & (row >= 0)
        & (row < target.height)
        & (col >= 0)
        & (col < target.width)
    )

    if target.mask is not None:
        keep_mask = np.zeros(len(events), dtype=bool)
        valid_pos = np.where(keep)[0]
        keep_mask[valid_pos] = target.mask[row[valid_pos], col[valid_pos]]
        keep = keep & keep_mask

    out = events.loc[keep].copy()
    out["row"] = row[keep].astype(np.int64)
    out["col"] = col[keep].astype(np.int64)

    return out


def read_group_chunk_policy(
    store_zarr: str | Path,
    *,
    group: str,
) -> dict[str, int]:
    ds = xr.open_zarr(store_zarr, group=group, consolidated=True)

    raw = ds.attrs.get("chunk_policy_json")

    if raw is None:
        return {"time": 96, "y": 512, "x": 512}

    loaded = json.loads(raw)

    return {
        "time": int(loaded.get("time") or 96),
        "y": int(loaded.get("y") or 512),
        "x": int(loaded.get("x") or 512),
    }


def ensure_viirs_arrays(
    *,
    satfire_zarr: str | Path,
    group: str = "viirs",
    overwrite_vars: bool = False,
) -> None:
    """
    Ensure VIIRS variables exist in the target Zarr group.
    """
    satfire_zarr = Path(satfire_zarr)

    template = xr.open_zarr(satfire_zarr, group=group, consolidated=True)

    for dim in ["time", "y", "x"]:
        if dim not in template.sizes:
            raise KeyError(f"/{group} is missing dimension {dim!r}")

    ntime = int(template.sizes["time"])
    ny = int(template.sizes["y"])
    nx = int(template.sizes["x"])

    chunks = read_group_chunk_policy(satfire_zarr, group=group)
    chunk_tuple = (
        min(int(chunks["time"]), ntime),
        min(int(chunks["y"]), ny),
        min(int(chunks["x"]), nx),
    )

    root = zarr.open_group(str(satfire_zarr), mode="a")
    zg = root[group]

    for name, spec in VIIRS_VAR_SPECS.items():
        if name in zg:
            if overwrite_vars:
                log(f"Deleting existing /{group}/{name}")
                del zg[name]
            else:
                continue

        log(f"Creating /{group}/{name}: shape={(ntime, ny, nx)}, chunks={chunk_tuple}")

        arr = zg.create_dataset(
            name,
            shape=(ntime, ny, nx),
            chunks=chunk_tuple,
            dtype=spec["dtype"],
            fill_value=spec["fill_value"],
            overwrite=False,
        )

        arr.attrs.update(
            {
                "_ARRAY_DIMENSIONS": ["time", "y", "x"],
                **spec["attrs"],
            }
        )


def zero_time_region(
    *,
    zgrp,
    time_region: slice,
) -> None:
    """
    Reset a time region to fill values.

    This can be expensive for a full month because it touches every spatial chunk.
    """
    for name, spec in VIIRS_VAR_SPECS.items():
        log(f"Zeroing {name} over time region {time_region}")
        zgrp[name][time_region, :, :] = spec["fill_value"]


def write_viirs_events_by_chunk(
    *,
    zgrp,
    events: pd.DataFrame,
    time_index: pd.DatetimeIndex,
    time_region: slice,
    ny: int,
    nx: int,
) -> None:
    """
    Aggregate sparse VIIRS events and write touched Zarr chunks.

    Writes are clipped to time_region to avoid overwriting adjacent months.
    """
    if len(events) == 0:
        log("No events to write")
        return

    t_chunk, y_chunk, x_chunk = zgrp["viirs_detect_count"].chunks

    log(f"Chunk layout: time={t_chunk}, y={y_chunk}, x={x_chunk}")

    i0 = int(time_region.start)
    i1 = int(time_region.stop)

    events = events.copy()

    events["t_chunk"] = (events["time_idx"] // t_chunk).astype(np.int64)
    events["y_chunk"] = (events["row"] // y_chunk).astype(np.int64)
    events["x_chunk"] = (events["col"] // x_chunk).astype(np.int64)

    grouped = events.groupby(["t_chunk", "y_chunk", "x_chunk"], sort=True)

    n_groups = len(grouped)
    log(f"Non-empty chunk groups to write: {n_groups:,}")

    count_arr = zgrp["viirs_detect_count"]
    frp_sum_arr = zgrp["viirs_frp_sum"]
    frp_max_arr = zgrp["viirs_frp_max"]
    conf_arr = zgrp["viirs_confidence_max"]
    any_arr = zgrp["viirs_detected_any"]

    for j, ((tc, yc, xc), g) in enumerate(grouped, start=1):
        chunk_t0 = int(tc) * t_chunk
        chunk_y0 = int(yc) * y_chunk
        chunk_x0 = int(xc) * x_chunk

        chunk_t1 = min(chunk_t0 + t_chunk, len(time_index))
        chunk_y1 = min(chunk_y0 + y_chunk, ny)
        chunk_x1 = min(chunk_x0 + x_chunk, nx)

        # Clip to this month/time-region.
        t0 = max(chunk_t0, i0)
        t1 = min(chunk_t1, i1)

        y0 = chunk_y0
        y1 = chunk_y1
        x0 = chunk_x0
        x1 = chunk_x1

        if t1 <= t0:
            continue

        g = g.loc[(g["time_idx"] >= t0) & (g["time_idx"] < t1)]

        if len(g) == 0:
            continue

        local_shape = (t1 - t0, y1 - y0, x1 - x0)

        count_block = np.zeros(local_shape, dtype=np.uint16)
        frp_sum_block = np.zeros(local_shape, dtype=np.float32)
        frp_max_block = np.zeros(local_shape, dtype=np.float32)
        conf_block = np.zeros(local_shape, dtype=np.uint8)
        any_block = np.zeros(local_shape, dtype=np.uint8)

        lt = g["time_idx"].to_numpy(np.int64) - t0
        ly = g["row"].to_numpy(np.int64) - y0
        lx = g["col"].to_numpy(np.int64) - x0

        frp = g["frp"].fillna(0).to_numpy(np.float32)
        conf = g["confidence_score"].fillna(0).to_numpy(np.uint8)

        np.add.at(count_block, (lt, ly, lx), 1)
        np.add.at(frp_sum_block, (lt, ly, lx), frp)
        np.maximum.at(frp_max_block, (lt, ly, lx), frp)
        np.maximum.at(conf_block, (lt, ly, lx), conf)
        any_block[lt, ly, lx] = 1

        count_arr[t0:t1, y0:y1, x0:x1] = count_block
        frp_sum_arr[t0:t1, y0:y1, x0:x1] = frp_sum_block
        frp_max_arr[t0:t1, y0:y1, x0:x1] = frp_max_block
        conf_arr[t0:t1, y0:y1, x0:x1] = conf_block
        any_arr[t0:t1, y0:y1, x0:x1] = any_block

        if j % 100 == 0 or j == n_groups:
            log(f"[{j}/{n_groups}] chunk groups written")