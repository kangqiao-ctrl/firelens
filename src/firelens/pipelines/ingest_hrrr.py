# src/firelens/pipelines/ingest_hrrr.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

from firelens.met.hrrr import (
    HRRR_VAR_ATTRS,
    build_or_load_hrrr_regridder,
    configure_eccodes_env,
    find_hrrr_files,
    read_hrrr_block,
    regrid_hrrr_block_to_met_grid,
)
from firelens.met.writer import (
    ensure_time_yx_arrays,
    get_time_region_for_year,
    read_group_chunk_policy,
    write_time_yx_region,
)


DEFAULT_HRRR_ROOT = "/scratch/users/qiaok/data/hrrr"
DEFAULT_MET_ZARR = "/scratch/groups/dsivas/qiaok/firelens_zarr/met_500m_1h.zarr"
DEFAULT_WEIGHTS = "/home/groups/dsivas/qiaok/firelens/artifacts/hrrr_to_ca500m_bilinear.nc"
DEFAULT_VARIABLES = ["gust_surface", "blh"]


def iter_time_blocks(
    target_time: xr.DataArray,
    *,
    block_hours: int,
):
    if block_hours <= 0:
        raise ValueError(f"block_hours must be positive, got {block_hours}")

    n = int(target_time.sizes["time"])

    for start in range(0, n, block_hours):
        stop = min(start + block_hours, n)
        yield slice(start, stop), target_time.isel(time=slice(start, stop))


def _time_key(value) -> np.datetime64:
    return np.datetime64(value, "ns")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read HRRR GRIB2 fields, regrid, and ingest into FireLens /hrrr."
    )

    parser.add_argument(
        "--hrrr-root",
        default=DEFAULT_HRRR_ROOT,
        help="Root directory containing HRRR files, e.g. /scratch/users/qiaok/data/hrrr.",
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
        help="Years to ingest, e.g. --years 2017 2018.",
    )

    parser.add_argument(
        "--variables",
        nargs="+",
        default=DEFAULT_VARIABLES,
        choices=DEFAULT_VARIABLES,
        help="HRRR variables to ingest.",
    )

    parser.add_argument(
        "--forecast-hour",
        type=int,
        default=0,
        help="Forecast hour to ingest. Current layout assumes sfc_f00 by default.",
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
        default=24,
        help="Number of hourly timesteps to read/regrid/write per block.",
    )

    parser.add_argument(
        "--eccodes-root",
        default=None,
        help="Optional ecCodes root, e.g. $HOME/local/eccodes.",
    )

    parser.add_argument(
        "--overwrite-vars",
        action="store_true",
        help="Delete and recreate target HRRR variables before writing.",
    )

    args = parser.parse_args()

    configure_eccodes_env(args.eccodes_root)

    met_zarr = Path(args.met_zarr)
    weights_nc = Path(args.weights_nc)

    chunks = read_group_chunk_policy(
        met_zarr,
        group="hrrr",
    )

    var_specs = {
        name: {
            **HRRR_VAR_ATTRS.get(name, {}),
            "source_group": "hrrr",
            "source": "HRRR",
        }
        for name in args.variables
    }

    ensure_time_yx_arrays(
        store_zarr=met_zarr,
        group="hrrr",
        var_specs=var_specs,
        dtype="float32",
        chunks=chunks,
        overwrite=args.overwrite_vars,
    )

    print("Discovering HRRR files...")
    file_map = find_hrrr_files(
        hrrr_root=args.hrrr_root,
        years=args.years,
        forecast_hour=args.forecast_hour,
    )

    if not file_map:
        raise FileNotFoundError(
            f"No HRRR files found under {args.hrrr_root} for years={args.years}, "
            f"forecast_hour={args.forecast_hour}"
        )

    sample_time = sorted(file_map)[0]
    sample_file = file_map[sample_time]

    print(f"Using sample file for grid/weights: {sample_file}")

    target_grid, subset, regridder = build_or_load_hrrr_regridder(
        sample_file=sample_file,
        met_zarr=met_zarr,
        weights_nc=weights_nc,
        buffer_deg=args.buffer_deg,
        method=args.method,
    )

    for year in args.years:
        print(f"\n=== Ingesting HRRR year {year} ===")

        time_region_year, target_time_year = get_time_region_for_year(
            met_zarr,
            group="hrrr",
            year=year,
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

            block_times = [_time_key(t) for t in target_time_block.values]
            block_paths = [file_map[t] for t in block_times if t in file_map]

            print(
                f"\n  Block region={global_time_region}, "
                f"start={str(target_time_block.values[0])}, "
                f"end={str(target_time_block.values[-1])}, "
                f"files={len(block_paths)}/{len(block_times)}"
            )

            if not block_paths:
                print("  No files in this block; leaving target values as NaN.")
                continue

            hrrr_block = read_hrrr_block(
                paths=block_paths,
                target_grid=target_grid,
                subset=subset,
                variables=args.variables,
            )

            ds_block = regrid_hrrr_block_to_met_grid(
                hrrr_block=hrrr_block,
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
                group="hrrr",
                ds_region=ds_block,
                time_region=global_time_region,
                var_names=args.variables,
                chunks=chunks,
            )

            print(f"  Wrote HRRR block to /hrrr region {global_time_region}")

    zarr.consolidate_metadata(str(met_zarr))
    print(f"\nDone. Consolidated metadata for: {met_zarr}")


if __name__ == "__main__":
    main()