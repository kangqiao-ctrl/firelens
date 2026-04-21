# src/firelens/slow_context/landfire_canopy.py
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


EXPECTED_FILL_CODES = np.asarray(
    [-9999, 32767],
    dtype=np.int32,
)


def build_numeric_codes(
    df: pd.DataFrame,
    *,
    value_col: str = "VALUE",
    exclude_codes: np.ndarray | list[int] | tuple[int, ...] = (),
    keep_nonnegative_only: bool = True,
) -> np.ndarray:
    """
    Build official integer codes from a LANDFIRE CSV table.

    This intentionally excludes known fill/nodata codes, even if they appear
    in the CSV table.
    """
    if value_col not in df.columns:
        raise KeyError(f"Missing {value_col!r} column in LANDFIRE CSV.")

    values = pd.to_numeric(df[value_col], errors="coerce")
    values = values.dropna().astype(int)

    if keep_nonnegative_only:
        values = values[values >= 0]

    if len(exclude_codes) > 0:
        exclude = np.asarray(exclude_codes, dtype=np.int32)
        values = values[~values.isin(exclude)]

    return np.asarray(sorted(values.unique()), dtype=np.int32)


def build_cc_codes(cc_df: pd.DataFrame) -> np.ndarray:
    return build_numeric_codes(
        cc_df,
        value_col="VALUE",
        exclude_codes=EXPECTED_FILL_CODES,
    )


def build_ch_codes(ch_df: pd.DataFrame) -> np.ndarray:
    return build_numeric_codes(
        ch_df,
        value_col="VALUE",
        exclude_codes=EXPECTED_FILL_CODES,
    )


def cc_dataset(
    *,
    cc_tif: str,
    cc_df: pd.DataFrame,
    target: TargetGrid,
    raise_on_unexpected: bool = True,
    output_mode: str = "full",
    keep_valid_fraction: bool = True,
    keep_qc_vars: bool = False,
) -> xr.Dataset:
    """
    Build canopy-cover variables on the target grid.

    Main outputs:
      cc_valid_fraction
      cc_total_valid_pct
      cc_total_full_pct
    """
    official_codes = build_cc_codes(cc_df)

    out = {}

    with rasterio.open(cc_tif) as src:
        src_ma, src_transform = read_source_subset(src, target, band=1)

        raw_mask = np.ma.getmaskarray(src_ma)
        values = src_ma.data.astype(np.int32)

        is_filled = np.isin(values, EXPECTED_FILL_CODES)
        is_official = np.isin(values, official_codes)

        expected_invalid = raw_mask | is_filled
        valid = (~expected_invalid) & is_official
        unexpected = (~expected_invalid) & (~is_official)

        if raise_on_unexpected and np.any(unexpected):
            uniq, cnt = np.unique(values[unexpected], return_counts=True)
            raise ValueError(
                "Unexpected unmasked CC codes found: "
                + ", ".join(f"{u}: {c}" for u, c in zip(uniq.tolist(), cnt.tolist()))
            )

        out["cc_valid_fraction"] = regrid_binary_fraction_full_cell(
            valid.astype("float32"),
            src_transform,
            src.crs,
            target,
            name="cc_valid_fraction",
        )

        out["cc_total_valid_pct"] = regrid_masked_array(
            np.ma.array(values.astype("float32"), mask=~valid),
            src_transform,
            src.crs,
            target,
            resampling=Resampling.average,
            name="cc_total_valid_pct",
            units="percent",
        )

        out["cc_total_full_pct"] = out["cc_total_valid_pct"] * out["cc_valid_fraction"]
        out["cc_total_full_pct"].name = "cc_total_full_pct"
        out["cc_total_full_pct"].attrs.update(
            {
                "long_name": "Whole-cell canopy cover percentage",
                "units": "percent",
                "semantics": (
                    "valid-support mean canopy cover multiplied by valid support fraction"
                ),
            }
        )

    ds = xr.Dataset(out)

    return select_output_vars(
        ds,
        output_mode=output_mode,
        keep_valid_fraction=keep_valid_fraction,
        keep_qc_vars=keep_qc_vars,
    )


def ch_dataset(
    *,
    ch_tif: str,
    ch_df: pd.DataFrame,
    target: TargetGrid,
    raise_on_unexpected: bool = True,
    output_mode: str = "full",
    keep_valid_fraction: bool = True,
    keep_qc_vars: bool = False,
) -> xr.Dataset:
    """
    Build canopy-height variables on the target grid.

    LANDFIRE CH values are encoded as meters * 10. This function converts
    them to meters.

    Main full-cell outputs:
      ch_total_full_m
      ch_positive_full_fraction
      ch_nonforest_full_fraction
      ch_lt10m_full_fraction
      ch_10to30m_full_fraction
      ch_gt30m_full_fraction

    Conditional / valid-support outputs are only kept when output_mode="both"
    or output_mode="frac", depending on variable name.
    """
    official_codes = build_ch_codes(ch_df)

    # Fixed CH bins requested.
    nonforest_codes = np.asarray([0], dtype=np.int32)
    lt10_codes = np.asarray([30, 70], dtype=np.int32)
    m10_30_codes = np.asarray([110, 150, 190, 230, 270, 310], dtype=np.int32)
    gt30_codes = np.asarray([350, 390, 430, 470, 510], dtype=np.int32)

    out = {}

    with rasterio.open(ch_tif) as src:
        src_ma, src_transform = read_source_subset(src, target, band=1)

        raw_mask = np.ma.getmaskarray(src_ma)
        values = src_ma.data.astype(np.int32)

        is_filled = np.isin(values, EXPECTED_FILL_CODES)
        is_official = np.isin(values, official_codes)

        expected_invalid = raw_mask | is_filled
        valid = (~expected_invalid) & is_official
        unexpected = (~expected_invalid) & (~is_official)

        if raise_on_unexpected and np.any(unexpected):
            uniq, cnt = np.unique(values[unexpected], return_counts=True)
            raise ValueError(
                "Unexpected unmasked CH codes found: "
                + ", ".join(f"{u}: {c}" for u, c in zip(uniq.tolist(), cnt.tolist()))
            )

        # Convert CH from meters * 10 to meters.
        ch_m = values.astype("float32") / 10.0

        out["ch_valid_fraction"] = regrid_binary_fraction_full_cell(
            valid.astype("float32"),
            src_transform,
            src.crs,
            target,
            name="ch_valid_fraction",
        )

        positive = valid & (values > 0)

        out["ch_positive_full_fraction"] = regrid_binary_fraction_full_cell(
            positive.astype("float32"),
            src_transform,
            src.crs,
            target,
            name="ch_positive_full_fraction",
        )
        out["ch_positive_full_fraction"].attrs.update(
            {
                "long_name": "Whole-cell fraction with canopy height greater than zero",
                "units": "1",
            }
        )

        # Conditional mean CH where CH > 0.
        out["ch_forest_m"] = regrid_masked_array(
            np.ma.array(ch_m, mask=~positive),
            src_transform,
            src.crs,
            target,
            resampling=Resampling.average,
            name="ch_forest_m",
            units="m",
        )
        out["ch_forest_m"].attrs.update(
            {
                "long_name": "Mean canopy height over positive-CH pixels",
                "semantics": "conditional mean over CH > 0 support",
            }
        )

        # Valid-support mean where valid nonforest zeros are included.
        ch_total_valid = regrid_masked_array(
            np.ma.array(np.where(valid, ch_m, 0.0), mask=~valid),
            src_transform,
            src.crs,
            target,
            resampling=Resampling.average,
            name="ch_total_valid_m",
            units="m",
        )
        out["ch_total_valid_m"] = ch_total_valid
        out["ch_total_valid_m"].attrs.update(
            {
                "long_name": "Mean canopy height over valid support, including valid zeros",
                "units": "m",
            }
        )

        out["ch_total_full_m"] = ch_total_valid * out["ch_valid_fraction"]
        out["ch_total_full_m"].name = "ch_total_full_m"
        out["ch_total_full_m"].attrs.update(
            {
                "long_name": "Whole-cell canopy height burden",
                "units": "m",
                "semantics": (
                    "valid-support mean CH, including valid zeros, multiplied by valid support fraction"
                ),
            }
        )

        # Requested categorical fractions over the full 500 m cell.
        bin_masks = {
            "ch_nonforest_full_fraction": valid & np.isin(values, nonforest_codes),
            "ch_lt10m_full_fraction": valid & np.isin(values, lt10_codes),
            "ch_10to30m_full_fraction": valid & np.isin(values, m10_30_codes),
            "ch_gt30m_full_fraction": valid & np.isin(values, gt30_codes),
        }

        for name, mask in bin_masks.items():
            out[name] = regrid_binary_fraction_full_cell(
                mask.astype("float32"),
                src_transform,
                src.crs,
                target,
                name=name,
            )
            out[name].attrs.update(
                {
                    "long_name": name.replace("_", " "),
                    "units": "1",
                    "semantics": "fraction of full 500 m cell",
                }
            )

    ds = xr.Dataset(out)

    # QC: these four bins should approximately partition valid support.
    bin_vars = [
        "ch_nonforest_full_fraction",
        "ch_lt10m_full_fraction",
        "ch_10to30m_full_fraction",
        "ch_gt30m_full_fraction",
    ]

    ds["ch_bin_sum_full_fraction"] = xr.concat(
        [ds[v] for v in bin_vars],
        dim="ch_bin",
    ).sum("ch_bin", skipna=True, min_count=1)

    ds["ch_bin_sum_full_fraction"].attrs.update(
        {
            "long_name": "QC sum of CH full-cell bin fractions",
            "units": "1",
            "semantics": "Should approximately equal ch_valid_fraction",
        }
    )

    return select_output_vars(
        ds,
        output_mode=output_mode,
        keep_valid_fraction=keep_valid_fraction,
        keep_qc_vars=keep_qc_vars,
        extra_keep=["ch_forest_m"],
    )