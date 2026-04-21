# src/firelens/slow_context/landfire_fdist.py
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype

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


FDIST_FAMILIES = [
    "NO_DISTURBANCE",
    "FIRE",
    "MECHANICAL_ADD",
    "MECHANICAL_REMOVE",
    "OTHER",
]

FDIST_EXPECTED_FILL_CODES = np.asarray(
    [-9999, -1111, 32767, -32768],
    dtype=np.int32,
)


FDIST_LABEL_COLUMN_CANDIDATES = [
    "D_TYPE",
    "DIST_TYPE",
    "DISTURBANCE_TYPE",
    "DISTURBANCE",
    "LABEL",
    "CLASS",
    "CLASSNAME",
    "NAME",
]


def _guess_label_column(df: pd.DataFrame) -> str:
    """
    Pick the column that describes the FDIST disturbance type.

    Preferred LANDFIRE FDIST schema uses D_TYPE, e.g.:

        VALUE,D_TYPE,D_SEVERITY,D_TIME,...
        0,No Disturbance,NA,NA,...
        111,Fire,Low,One Year,...

    We prefer known semantic names before falling back to string-like columns.
    """
    # Exact-name or case-insensitive match first.
    columns_by_upper = {c.upper(): c for c in df.columns}

    for candidate in FDIST_LABEL_COLUMN_CANDIDATES:
        if candidate.upper() in columns_by_upper:
            return columns_by_upper[candidate.upper()]

    # Fallback: any string-like non-VALUE column with nonnumeric content.
    fallback_candidates = []

    for c in df.columns:
        if c.upper() == "VALUE":
            continue

        s = df[c]

        if is_string_dtype(s) or s.dtype == object:
            non_null = s.dropna().astype(str).str.strip()
            if len(non_null) == 0:
                continue

            # Prefer columns that are not purely numeric-looking.
            numeric_fraction = pd.to_numeric(non_null, errors="coerce").notna().mean()
            if numeric_fraction < 0.5:
                fallback_candidates.append(c)

    if fallback_candidates:
        return fallback_candidates[0]

    raise ValueError(
        "Could not infer FDIST label column. "
        f"Columns were: {list(df.columns)}. "
        "Expected a column such as 'D_TYPE'."
    )


def _classify_fdist_label(label: str) -> str:
    """
    Map a LANDFIRE FDIST disturbance label to a coarse family.
    """
    s = str(label).strip().upper()

    if s in {"", "NA", "NAN", "NONE"}:
        return "OTHER"

    if ("NO" in s and "DIST" in s) or "UNDIST" in s:
        return "NO_DISTURBANCE"

    if "FIRE" in s or "BURN" in s:
        return "FIRE"

    if "MECHANICAL" in s and (
        "ADD" in s
        or "ADDITION" in s
        or "ADDED" in s
    ):
        return "MECHANICAL_ADD"

    if "MECHANICAL" in s and (
        "REMOVE" in s
        or "REMOVAL" in s
        or "REMOVED" in s
        or "REMOV" in s
    ):
        return "MECHANICAL_REMOVE"

    return "OTHER"


def build_fdist_groups(
    fdist_df: pd.DataFrame,
    *,
    family_map: dict[str, list[int]] | None = None,
    label_col: str | None = None,
) -> tuple[np.ndarray, dict[str, list[int]]]:
    """
    Build FDIST official code list and coarse family map.

    Families:
      - NO_DISTURBANCE
      - FIRE
      - MECHANICAL_ADD
      - MECHANICAL_REMOVE
      - OTHER
    """
    raw = fdist_df.copy()

    if "VALUE" not in raw.columns:
        raise KeyError(f"FDIST CSV must contain VALUE column. Columns: {list(raw.columns)}")

    raw = raw[pd.to_numeric(raw["VALUE"], errors="coerce").notna()].copy()
    raw["VALUE"] = raw["VALUE"].astype(int)

    # Keep actual class codes and exclude known fill/nodata codes.
    df = raw.loc[
        (raw["VALUE"] >= 0)
        & (~raw["VALUE"].isin(FDIST_EXPECTED_FILL_CODES))
    ].copy()

    official_codes = np.asarray(sorted(df["VALUE"].unique()), dtype=np.int32)

    if family_map is not None:
        cleaned = {
            fam: sorted([int(v) for v in codes])
            for fam, codes in family_map.items()
        }

        missing = [fam for fam in FDIST_FAMILIES if fam not in cleaned]
        if missing:
            raise ValueError(f"family_map missing families: {missing}")

        return official_codes, cleaned

    if label_col is None:
        label_col = _guess_label_column(df)

    if label_col not in df.columns:
        raise KeyError(
            f"label_col={label_col!r} not found in FDIST CSV. "
            f"Columns: {list(df.columns)}"
        )

    df["family"] = df[label_col].map(_classify_fdist_label)

    inferred = {
        fam: sorted(df.loc[df["family"] == fam, "VALUE"].astype(int).tolist())
        for fam in FDIST_FAMILIES
    }

    # Basic sanity printout. Useful during pipeline runs.
    print("FDIST grouping:")
    print(f"  label_col: {label_col}")
    for fam in FDIST_FAMILIES:
        print(f"  {fam}: {inferred[fam]}")

    return official_codes, inferred

def fdist_dataset(
    *,
    fdist_tif: str,
    fdist_df: pd.DataFrame,
    target: TargetGrid,
    family_map: dict[str, list[int]] | None = None,
    label_col: str | None = None,
    raise_on_unexpected: bool = True,
    output_mode: str = "full",
    keep_valid_fraction: bool = True,
    keep_family_sum: bool = False,
) -> xr.Dataset:
    official_codes, family_map = build_fdist_groups(
        fdist_df,
        family_map=family_map,
        label_col=label_col,
    )

    out = {}

    with rasterio.open(fdist_tif) as src:
        src_ma, src_transform = read_source_subset(src, target, band=1)

        raw_mask = np.ma.getmaskarray(src_ma)
        values = src_ma.data.astype(np.int32)

        is_filled = np.isin(values, FDIST_EXPECTED_FILL_CODES)
        is_official = np.isin(values, official_codes)

        expected_invalid = raw_mask | is_filled
        valid = (~expected_invalid) & is_official
        unexpected = (~expected_invalid) & (~is_official)

        if raise_on_unexpected and np.any(unexpected):
            uniq, cnt = np.unique(values[unexpected], return_counts=True)
            raise ValueError(
                "Unexpected unmasked FDIST codes found: "
                + ", ".join(f"{u}: {c}" for u, c in zip(uniq.tolist(), cnt.tolist()))
            )

        out["fdist_valid_fraction"] = regrid_binary_fraction_full_cell(
            valid.astype("float32"),
            src_transform,
            src.crs,
            target,
            name="fdist_valid_fraction",
        )

        for fam in FDIST_FAMILIES:
            fam_lower = fam.lower()
            codes = family_map[fam]

            mask01 = np.isin(values, codes).astype("float32")
            fam_ma = np.ma.array(mask01, mask=~valid)

            frac_name = f"fdist_frac_{fam_lower}"
            full_name = f"fdist_full_{fam_lower}"

            out[frac_name] = regrid_masked_array(
                fam_ma,
                src_transform,
                src.crs,
                target,
                resampling=Resampling.average,
                name=frac_name,
            )

            out[full_name] = out[frac_name] * out["fdist_valid_fraction"]
            out[full_name].name = full_name
            out[full_name].attrs.update(
                {
                    "long_name": f"Whole-cell fraction of FDIST family {fam}",
                    "units": "1",
                    "semantics": "fraction of full 500 m cell, not only valid support",
                }
            )

            if fam in {"FIRE", "MECHANICAL_REMOVE"}:
                for code in codes:
                    code01 = (values == code).astype("float32")
                    code_ma = np.ma.array(code01, mask=~valid)

                    frac_code_name = f"fdist_frac_{code}"
                    full_code_name = f"fdist_full_{code}"

                    out[frac_code_name] = regrid_masked_array(
                        code_ma,
                        src_transform,
                        src.crs,
                        target,
                        resampling=Resampling.average,
                        name=frac_code_name,
                    )

                    out[full_code_name] = out[frac_code_name] * out["fdist_valid_fraction"]
                    out[full_code_name].name = full_code_name

    ds = xr.Dataset(out)

    frac_family_vars = [f"fdist_frac_{fam.lower()}" for fam in FDIST_FAMILIES]
    full_family_vars = [f"fdist_full_{fam.lower()}" for fam in FDIST_FAMILIES]

    ds["fdist_family_sum"] = xr.concat(
        [ds[v] for v in frac_family_vars],
        dim="family",
    ).sum("family", skipna=True, min_count=1)

    ds["fdist_full_family_sum"] = xr.concat(
        [ds[v] for v in full_family_vars],
        dim="family",
    ).sum("family", skipna=True, min_count=1)

    return select_output_vars(
        ds,
        output_mode=output_mode,
        keep_valid_fraction=keep_valid_fraction,
        keep_family_sum=keep_family_sum,
    )