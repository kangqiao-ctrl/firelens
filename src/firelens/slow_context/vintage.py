# src/firelens/slow_context/vintage.py
from __future__ import annotations

import json
from typing import Mapping

import pandas as pd
import xarray as xr


YEAR_TO_VINTAGE = {
    2017: "LF2016",
    2018: "LF2016",
    2019: "LF2016",
    2020: "LF2020",
    2021: "LF2020",
    2022: "LF2022",
    2023: "LF2023",
    2024: "LF2024",
    2025: "LF2024",
}


def vintage_for_timestamp(
    t: str | pd.Timestamp,
    *,
    year_to_vintage: Mapping[int, str] = YEAR_TO_VINTAGE,
) -> str:
    ts = pd.Timestamp(t)
    year = int(ts.year)

    if year not in year_to_vintage:
        raise KeyError(f"No Landfire vintage mapping defined for year {year}")

    return year_to_vintage[year]


def year_to_vintage_json(
    year_to_vintage: Mapping[int, str] = YEAR_TO_VINTAGE,
) -> str:
    return json.dumps({int(k): str(v) for k, v in year_to_vintage.items()})


def read_year_to_vintage_from_attrs(ds: xr.Dataset) -> dict[int, str]:
    raw = ds.attrs.get("year_to_vintage_json")
    if raw is None:
        return dict(YEAR_TO_VINTAGE)

    loaded = json.loads(raw)
    return {int(k): str(v) for k, v in loaded.items()}


def select_landfire_at_time(
    slow_ds: xr.Dataset,
    t: str | pd.Timestamp,
    *,
    vars_keep: list[str] | None = None,
) -> xr.Dataset:
    year_to_vintage = read_year_to_vintage_from_attrs(slow_ds)
    vintage = vintage_for_timestamp(t, year_to_vintage=year_to_vintage)

    out = slow_ds.sel(vintage=vintage, drop=True)

    if vars_keep is not None:
        out = out[vars_keep]

    return out