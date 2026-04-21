# src/firelens/slow_context/landfire_fbfm40.py
from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.enums import Resampling

from firelens.slow_context.outputs import select_output_vars
from firelens.slow_context.raster import (
    TargetGrid,
    read_source_subset,
    regrid_binary_fraction_full_cell,
    regrid_masked_array,
)


FBFM40_FAMILIES = ["NB", "GR", "GS", "SH", "TU", "TL", "SB"]

NB1_URBAN = 91
NB8_WATER = 98

FBFM40_EXPECTED_FILL_CODES = np.asarray(
    [-9999, 32767],
    dtype=np.int32,
)


def build_fbfm40_groups(
    fbfm40_df: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, list[int]]]:
    raw = fbfm40_df.copy()

    raw = raw[pd.to_numeric(raw["VALUE"], errors="coerce").notna()].copy()
    raw["VALUE"] = raw["VALUE"].astype(int)

    raw["FBFM40"] = raw["FBFM40"].astype(str).str.strip().str.upper()

    df = raw.loc[raw["VALUE"] >= 0].copy()
    df["family"] = df["FBFM40"].str.extract(r"^(NB|GR|GS|SH|TU|TL|SB)")

    family_map = {
        fam: sorted(df.loc[df["family"] == fam, "VALUE"].astype(int).tolist())
        for fam in FBFM40_FAMILIES
    }

    official_codes = np.asarray(sorted(df["VALUE"].unique()), dtype=np.int32)

    missing_family = df["family"].isna()
    if missing_family.any():
        bad = df.loc[missing_family, ["VALUE", "FBFM40"]]
        raise ValueError(f"Unrecognized FBFM40 labels:\n{bad}")

    return official_codes, family_map


def fbfm40_dataset(
    *,
    fbfm40_tif: str,
    fbfm40_df: pd.DataFrame,
    target: TargetGrid,
    raise_on_unexpected: bool = True,
    output_mode: str = "full",
    keep_valid_fraction: bool = True,
    keep_family_sum: bool = False,
) -> xr.Dataset:
    official_codes, family_map = build_fbfm40_groups(fbfm40_df)

    out = {}

    with rasterio.open(fbfm40_tif) as src:
        src_ma, src_transform = read_source_subset(src, target, band=1)

        raw_mask = np.ma.getmaskarray(src_ma)
        values = src_ma.data.astype(np.int32)

        is_filled = np.isin(values, FBFM40_EXPECTED_FILL_CODES)
        is_official = np.isin(values, official_codes)

        expected_invalid = raw_mask | is_filled
        valid = (~expected_invalid) & is_official
        unexpected = (~expected_invalid) & (~is_official)

        if raise_on_unexpected and np.any(unexpected):
            uniq, cnt = np.unique(values[unexpected], return_counts=True)
            raise ValueError(
                "Unexpected unmasked FBFM40 codes found: "
                + ", ".join(f"{u}: {c}" for u, c in zip(uniq.tolist(), cnt.tolist()))
            )

        out["fbfm40_valid_fraction"] = regrid_binary_fraction_full_cell(
            valid.astype("float32"),
            src_transform,
            src.crs,
            target,
            name="fbfm40_valid_fraction",
        )

        for fam in FBFM40_FAMILIES:
            fam_lower = fam.lower()
            codes = family_map[fam]

            mask01 = np.isin(values, codes).astype("float32")
            fam_ma = np.ma.array(mask01, mask=~valid)

            frac_name = f"fbfm40_frac_{fam_lower}"
            full_name = f"fbfm40_full_{fam_lower}"

            out[frac_name] = regrid_masked_array(
                fam_ma,
                src_transform,
                src.crs,
                target,
                resampling=Resampling.average,
                name=frac_name,
            )

            out[full_name] = out[frac_name] * out["fbfm40_valid_fraction"]
            out[full_name].name = full_name
            out[full_name].attrs.update(
                {
                    "long_name": f"Whole-cell fraction of FBFM40 family {fam}",
                    "units": "1",
                    "semantics": "fraction of full 500 m cell, not only valid support",
                }
            )

        for code, label in [
            (NB1_URBAN, "nb1_urban"),
            (NB8_WATER, "nb8_water"),
        ]:
            mask01 = (values == code).astype("float32")
            code_ma = np.ma.array(mask01, mask=~valid)

            frac_name = f"fbfm40_frac_{label}"
            full_name = f"fbfm40_full_{label}"

            out[frac_name] = regrid_masked_array(
                code_ma,
                src_transform,
                src.crs,
                target,
                resampling=Resampling.average,
                name=frac_name,
            )

            out[full_name] = out[frac_name] * out["fbfm40_valid_fraction"]
            out[full_name].name = full_name
            out[full_name].attrs.update(
                {
                    "long_name": f"Whole-cell fraction of FBFM40 subclass {label}",
                    "units": "1",
                    "semantics": "fraction of full 500 m cell, not only valid support",
                }
            )

    ds = xr.Dataset(out)

    frac_family_vars = [f"fbfm40_frac_{fam.lower()}" for fam in FBFM40_FAMILIES]
    full_family_vars = [f"fbfm40_full_{fam.lower()}" for fam in FBFM40_FAMILIES]

    ds["fbfm40_family_sum"] = xr.concat(
        [ds[v] for v in frac_family_vars],
        dim="family",
    ).sum("family", skipna=True, min_count=1)

    ds["fbfm40_full_family_sum"] = xr.concat(
        [ds[v] for v in full_family_vars],
        dim="family",
    ).sum("family", skipna=True, min_count=1)

    return select_output_vars(
        ds,
        output_mode=output_mode,
        keep_valid_fraction=keep_valid_fraction,
        keep_family_sum=keep_family_sum,
    )