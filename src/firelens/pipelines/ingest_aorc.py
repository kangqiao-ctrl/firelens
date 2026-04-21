# src/firelens/pipelines/ingest_aorc.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

from firelens.met.aorc import (
    AORC_RENAME_DEFAULT,
    AORC_VAR_ATTRS,
    open_aorc_year_subset_and_regridder,
    regrid_aorc_block_to_met_grid,
)
from firelens.met.writer import (
    ensure_time_yx_arrays,
    get_time_region_for_year,
    read_group_chunk_policy,
    write_time_yx_region,
)


DEFAULT_AORC_ROOT = "/scratch/users/qiaok/data/aorc/zarr"
DEFAULT_MET_ZARR = "/scratch/groups/dsivas/qiaok/firelens_zarr/met_500m_1h.zarr"
DEFAULT_WEIGHTS = "/home/groups/dsivas/qiaok/firelens/artifacts/aorc_to_ca500m_bilinear.nc"

# Set this to the AORC variables you want as model inputs.
DEFAULT_VARIABLES = [
    "TMP_2maboveground",
    "SPFH_2maboveground",
    "PRES_surface",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
    "APCP_surface",
    "DSWRF_surface",
    "DLWRF_surface",
]


def aorc_year_path(root: str | Path, year: int) -> Path:
    return Path(root) / f"{year}.zarr"


def iter_time_blocks(
    target_time: xr.DataArray,
    *,
    block_hours: int,
):
    """
    Yield contiguous slices and target-time blocks.
    """
    if block_hours <= 0:
        raise ValueError(f"block_hours must be positive, got {block_hours}")

    n = int(target_time.sizes["time"])

    for start in range(0, n, block_hours):
        stop = min(start + block_hours, n)
        yield slice(start, stop), target_time.isel(time=slice(start, stop))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regrid and ingest AORC variables into FireLens met_500m_1h.zarr /aorc."
    )

    parser.add_argument(
        "--aorc-root",
        default=DEFAULT_AORC_ROOT,
        help="Directory containing yearly AORC Zarr stores, e.g. 2025.zarr.",
    )

    parser.add_argument(
        "--met-zarr",
        default=DEFAULT_MET_ZARR,
        help="Initialized FireLens hourly met store.",
    )

    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        help="Years to ingest, e.g. --years 2024 2025.",
    )

    parser.add_argument(
        "--variables",
        nargs="+",
        default=DEFAULT_VARIABLES,
        help="AORC source variable names to ingest.",
    )

    parser.add_argument(
        "--weights-nc",
        default=DEFAULT_WEIGHTS,
        help="Path to xESMF weights NetCDF.",
    )

    parser.add_argument(
        "--method",
        default="bilinear",
        help="xESMF regridding method.",
    )

    parser.add_argument(
        "--buffer-deg",
        type=float,
        default=1.0,
        help="Lat/lon buffer around California target extent before regridding.",
    )

    parser.add_argument(
        "--time-block-hours",
        type=int,
        default=24 * 31,
        help="Number of hourly timesteps to regrid/write per block.",
    )

    parser.add_argument(
        "--overwrite-vars",
        action="store_true",
        help="Delete and recreate target AORC variables before writing.",
    )

    args = parser.parse_args()

    met_zarr = Path(args.met_zarr)
    weights_nc = Path(args.weights_nc)

    chunks = read_group_chunk_policy(
        met_zarr,
        group="aorc",
    )

    output_names = [
        AORC_RENAME_DEFAULT.get(v, v.lower())
        for v in args.variables
    ]

    var_specs = {
        out_name: {
            **AORC_VAR_ATTRS.get(out_name, {}),
            "source_group": "aorc",
            "source": "AORC",
        }
        for out_name in output_names
    }

    ensure_time_yx_arrays(
        store_zarr=met_zarr,
        group="aorc",
        var_specs=var_specs,
        dtype="float32",
        chunks=chunks,
        overwrite=args.overwrite_vars,
    )

    for year in args.years:
        src_path = aorc_year_path(args.aorc_root, year)

        if not src_path.exists():
            raise FileNotFoundError(f"Missing AORC year store: {src_path}")

        print(f"\n=== Ingesting AORC year {year} ===")
        print(f"Source: {src_path}")

        time_region_year, target_time_year = get_time_region_for_year(
            met_zarr,
            group="aorc",
            year=year,
        )

        src_sub, target_grid, regridder = open_aorc_year_subset_and_regridder(
            aorc_year_zarr=src_path,
            met_zarr=met_zarr,
            weights_nc=weights_nc,
            buffer_deg=args.buffer_deg,
            method=args.method,
        )

        year_region_start = int(time_region_year.start)

        for local_slice, target_time_block in iter_time_blocks(
            target_time_year,
            block_hours=args.time_block_hours,
        ):
            global_time_region = slice(
                year_region_start + int(local_slice.start),
                year_region_start + int(local_slice.stop),
            )

            print(
                f"\n  Block time region: {global_time_region}, "
                f"start={str(target_time_block.values[0])}, "
                f"end={str(target_time_block.values[-1])}"
            )

            ds_block = regrid_aorc_block_to_met_grid(
                src_sub=src_sub,
                target_grid=target_grid,
                regridder=regridder,
                target_time=target_time_block,
                variables=args.variables,
                time_chunk=int(chunks["time"]),
                output_chunks={
                    "y": int(chunks["y"]),
                    "x": int(chunks["x"]),
                },
            )

            ds_block = ds_block.chunk(
                {
                    "time": int(chunks["time"]),
                    "y": int(chunks["y"]),
                    "x": int(chunks["x"]),
                }
            )

            write_time_yx_region(
                store_zarr=met_zarr,
                group="aorc",
                ds_region=ds_block,
                time_region=global_time_region,
                var_names=output_names,
                chunks=chunks,
            )

            print(f"  Wrote AORC block to /aorc region {global_time_region}")

    zarr.consolidate_metadata(str(met_zarr))
    print(f"\nDone. Consolidated metadata for: {met_zarr}")


if __name__ == "__main__":
    main()