# src/firelens/slow_context/outputs.py
from __future__ import annotations

import xarray as xr

def _is_qc_var(name: str) -> bool:
    """
    Identify diagnostic/QC variables that are useful for inspection
    but not usually needed as model input channels.
    """
    return (
        name.endswith("_family_sum")
        or name.endswith("_full_family_sum")
        or name.endswith("_bin_sum_full_fraction")
        or name.endswith("_bin_sum_fraction")
    )


def select_output_vars(
    ds: xr.Dataset,
    *,
    output_mode: str = "full",
    keep_valid_fraction: bool = True,
    keep_family_sum: bool = False,
    keep_qc_vars: bool = False,
    extra_keep: list[str] | None = None,
) -> xr.Dataset:
    """
    Select variables to keep before writing to Zarr.

    Parameters
    ----------
    output_mode
        "full"  -> keep whole-cell variables.
        "frac"  -> keep valid-support-normalized fraction variables.
        "both"  -> keep all non-QC variables.
    keep_valid_fraction
        Keep *_valid_fraction variables.
    keep_family_sum
        Keep *_family_sum variables. Mostly useful for FBFM40/FDIST QC.
    keep_qc_vars
        Keep other QC variables such as CH bin sums.
    extra_keep
        Always keep these variable names regardless of output_mode.
        Example: ["ch_forest_m"].
    """
    mode = output_mode.lower()

    if mode not in {"full", "frac", "fraction", "both"}:
        raise ValueError(
            f"output_mode must be one of 'full', 'frac', 'fraction', or 'both'; "
            f"got {output_mode!r}"
        )

    if mode == "both":
        names = list(ds.data_vars)

    elif mode == "full":
        names = [
            name
            for name in ds.data_vars
            if (
                "_full_" in name
                or name.endswith("_full_pct")
                or name.endswith("_full_m")
                or name.endswith("_full_fraction")
            )
        ]

    else:
        names = [
            name
            for name in ds.data_vars
            if (
                "_frac_" in name
                or name.endswith("_valid_pct")
                or name.endswith("_valid_m")
            )
        ]

    # Drop QC variables unless explicitly requested.
    if not keep_qc_vars:
        names = [name for name in names if not _is_qc_var(name)]

    if keep_valid_fraction:
        names.extend(
            name
            for name in ds.data_vars
            if name.endswith("_valid_fraction") and name not in names
        )

    if keep_family_sum:
        names.extend(
            name
            for name in ds.data_vars
            if name.endswith("_family_sum") and name not in names
        )

    if keep_qc_vars:
        names.extend(
            name
            for name in ds.data_vars
            if _is_qc_var(name) and name not in names
        )

    if extra_keep is not None:
        names.extend(
            name
            for name in extra_keep
            if name in ds.data_vars and name not in names
        )

    # Preserve original dataset order.
    keep = set(names)
    ordered = [name for name in ds.data_vars if name in keep]

    return ds[ordered]


def prefix_dataset_vars(ds: xr.Dataset, prefix: str) -> xr.Dataset:
    """
    Prefix data variables, e.g.

        fbfm40_full_gr -> landfire_fbfm40_full_gr
    """
    prefix = prefix.strip("_")

    rename = {
        name: name if name.startswith(prefix + "_") else f"{prefix}_{name}"
        for name in ds.data_vars
    }

    return ds.rename(rename)