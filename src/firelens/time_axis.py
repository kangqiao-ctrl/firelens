# firelens/time_axis.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr


def _as_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    """
    Convert a string or Timestamp to a timezone-aware UTC Timestamp.

    Naive timestamps are interpreted as UTC.
    """
    ts = pd.Timestamp(value)

    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts


def build_time_axis_dataset(
    *,
    start_utc: str | pd.Timestamp,
    end_utc: str | pd.Timestamp,
    freq: str,
    coord_name: str = "time",
    title: str = "Frozen UTC time coordinate",
    time_semantics: str = "timestamp marks the end of the feature availability/aggregation interval",
    inclusive: str = "both",
) -> xr.Dataset:
    """
    Build a frozen UTC time coordinate.

    The coordinate is stored as timezone-naive datetime64[ns], but the attrs
    explicitly declare that the convention is UTC.

    Parameters
    ----------
    start_utc, end_utc
        Start and end timestamps. Naive timestamps are interpreted as UTC.
    freq
        Pandas frequency string, e.g. "15min", "1h".
    coord_name
        Coordinate name. Usually "time".
    title
        Dataset title.
    time_semantics
        Human-readable convention for what timestamps mean.
    inclusive
        Passed to pandas.date_range. Usually "both".

    Returns
    -------
    xr.Dataset
        Dataset containing only the requested time coordinate.
    """
    start = _as_utc_timestamp(start_utc)
    end = _as_utc_timestamp(end_utc)

    if end < start:
        raise ValueError(f"end_utc must be >= start_utc, got {start=} and {end=}")

    idx = pd.date_range(
        start=start,
        end=end,
        freq=freq,
        inclusive=inclusive,
    )

    if len(idx) == 0:
        raise ValueError(
            f"Time axis is empty. Check start={start}, end={end}, freq={freq}"
        )

    # Store timezone-naive numpy datetime64 but document UTC convention.
    idx = idx.tz_convert("UTC").tz_localize(None)

    ds = xr.Dataset(
        coords={
            coord_name: (
                coord_name,
                idx.to_numpy(dtype="datetime64[ns]"),
            )
        },
        attrs={
            "title": title,
            "timezone_convention": "UTC",
            "frequency": freq,
            "time_semantics": time_semantics,
        },
    )

    ds[coord_name].attrs = {
        "long_name": "analysis time",
        "standard_name": "time",
        "timezone_convention": "UTC",
    }

    return ds


def write_time_axis_netcdf(
    out_nc: str | Path,
    *,
    start_utc: str | pd.Timestamp,
    end_utc: str | pd.Timestamp,
    freq: str,
    coord_name: str = "time",
    title: str = "Frozen UTC time coordinate",
    time_semantics: str = "timestamp marks the end of the feature availability/aggregation interval",
    inclusive: str = "both",
) -> xr.Dataset:
    """
    Build and write a frozen time-axis NetCDF.

    Returns the dataset after writing.
    """
    out_nc = Path(out_nc)
    out_nc.parent.mkdir(parents=True, exist_ok=True)

    ds = build_time_axis_dataset(
        start_utc=start_utc,
        end_utc=end_utc,
        freq=freq,
        coord_name=coord_name,
        title=title,
        time_semantics=time_semantics,
        inclusive=inclusive,
    )

    ds.to_netcdf(out_nc)
    return ds