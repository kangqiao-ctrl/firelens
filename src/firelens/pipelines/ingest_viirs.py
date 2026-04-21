# src/firelens/pipelines/ingest_viirs.py
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import xarray as xr
import zarr

from firelens.grid import build_target_grid_from_zarr
from firelens.satfire.viirs import (
    attach_grid_indices_from_lonlat,
    assign_time_bins,
    ensure_viirs_arrays,
    infer_time_frequency,
    log,
    month_interval,
    read_events_month,
    standardize_viirs_events,
    task_id_to_year_month,
    time_region_for_interval,
    write_viirs_events_by_chunk,
    zero_time_region,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rasterize sparse VIIRS active-fire events into satfire_500m_15min.zarr /viirs."
    )

    p.add_argument(
        "--events-root",
        type=Path,
        required=True,
        help="Root of partitioned VIIRS event parquet.",
    )

    p.add_argument(
        "--satfire-zarr",
        type=Path,
        required=True,
        help="Initialized satfire_500m_15min.zarr store.",
    )

    p.add_argument(
        "--group",
        type=str,
        default="viirs",
        help="Target group in satfire store.",
    )

    p.add_argument(
        "--grid-group",
        type=str,
        default="grid",
        help="Grid group in satfire store.",
    )

    p.add_argument("--start-year", type=int, required=True)
    p.add_argument("--end-year", type=int, required=True)

    p.add_argument(
        "--year",
        type=int,
        default=None,
        help="Optional explicit year. If omitted, use --task-id or SLURM_ARRAY_TASK_ID.",
    )

    p.add_argument(
        "--month",
        type=int,
        default=None,
        help="Optional explicit month. If omitted, use --task-id or SLURM_ARRAY_TASK_ID.",
    )

    p.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Array task id mapped to year/month.",
    )

    p.add_argument(
        "--overwrite-vars",
        action="store_true",
        help="Delete and recreate VIIRS variables before writing. Use only before a full rebuild.",
    )

    p.add_argument(
        "--overwrite-month",
        action="store_true",
        help="Reset this month time region to fill values before writing. Can be expensive.",
    )

    p.add_argument(
        "--bin-policy",
        choices=["floor", "ceil", "nearest"],
        default="floor",
        help=(
            "How to map obs_time to the 15-min store time. "
            "Use ceil if timestamps represent interval ends. "
            "floor preserves the previous script behavior."
        ),
    )

    p.add_argument("--obs-time-column", default=None)
    p.add_argument("--lat-column", default=None)
    p.add_argument("--lon-column", default=None)
    p.add_argument("--frp-column", default=None)
    p.add_argument("--confidence-column", default=None)

    p.add_argument(
        "--mask-var",
        default="domain_mask",
        help="Optional mask variable in /grid. Use '' to disable.",
    )

    return p.parse_args()


def resolve_year_month(args: argparse.Namespace) -> tuple[int, int]:
    if args.year is not None or args.month is not None:
        if args.year is None or args.month is None:
            raise ValueError("Must provide both --year and --month, or neither.")

        if args.month < 1 or args.month > 12:
            raise ValueError(f"month must be 1..12, got {args.month}")

        return int(args.year), int(args.month)

    task_id = args.task_id

    if task_id is None:
        if "SLURM_ARRAY_TASK_ID" not in os.environ:
            raise RuntimeError("No --year/--month, no --task-id, and SLURM_ARRAY_TASK_ID is not set.")
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    return task_id_to_year_month(
        task_id,
        start_year=args.start_year,
        end_year=args.end_year,
    )


def main() -> None:
    args = parse_args()

    year, month = resolve_year_month(args)

    mask_var = args.mask_var if args.mask_var else None

    log(f"Processing VIIRS month {year}-{month:02d}")
    log(f"Events root: {args.events_root}")
    log(f"Satfire store: {args.satfire_zarr}")
    log(f"Target group: /{args.group}")

    ensure_viirs_arrays(
        satfire_zarr=args.satfire_zarr,
        group=args.group,
        overwrite_vars=args.overwrite_vars,
    )

    viirs_ds = xr.open_zarr(
        args.satfire_zarr,
        group=args.group,
        consolidated=True,
    )

    zroot = zarr.open_group(str(args.satfire_zarr), mode="a")
    zgrp = zroot[args.group]

    time_index = pd.DatetimeIndex(viirs_ds["time"].values)
    dt = infer_time_frequency(time_index)

    month_start, month_end_exclusive = month_interval(year, month)
    time_region = time_region_for_interval(
        time_index,
        start=month_start,
        end_exclusive=month_end_exclusive,
    )

    log(f"Store frequency: {dt}")
    log(f"Month interval: {month_start} -> {month_end_exclusive} exclusive")
    log(f"Time region: {time_region}")

    if args.overwrite_month:
        log("Resetting this month region before writing...")
        zero_time_region(
            zgrp=zgrp,
            time_region=time_region,
        )

    events_raw = read_events_month(
        args.events_root,
        year=year,
        month=month,
    )

    log(f"Raw events loaded: {len(events_raw):,}")

    if len(events_raw) == 0:
        log("No VIIRS events this month.")
        zarr.consolidate_metadata(str(args.satfire_zarr))
        return

    events = standardize_viirs_events(
        events_raw,
        obs_time_col=args.obs_time_column,
        lat_col=args.lat_column,
        lon_col=args.lon_column,
        frp_col=args.frp_column,
        confidence_col=args.confidence_column,
    )

    log(f"Events after column standardization/dropna: {len(events):,}")

    events = assign_time_bins(
        events,
        time_index=time_index,
        dt=dt,
        bin_policy=args.bin_policy,
    )

    log(f"Events after time binning into store axis: {len(events):,}")

    i0 = int(time_region.start)
    i1 = int(time_region.stop)

    events = events.loc[
        (events["time_idx"] >= i0)
        & (events["time_idx"] < i1)
    ].copy()

    log(f"Events after month time filtering: {len(events):,}")

    if len(events) == 0:
        log("No VIIRS events remain after time filtering.")
        zarr.consolidate_metadata(str(args.satfire_zarr))
        return

    target = build_target_grid_from_zarr(
        args.satfire_zarr,
        group=args.grid_group,
        consolidated=True,
        mask_var=mask_var,
    )

    events = attach_grid_indices_from_lonlat(
        events,
        target=target,
    )

    log(f"Events after lon/lat -> grid row/col filtering: {len(events):,}")

    if len(events) == 0:
        log("No VIIRS events fall inside the target grid.")
        zarr.consolidate_metadata(str(args.satfire_zarr))
        return

    write_viirs_events_by_chunk(
        zgrp=zgrp,
        events=events,
        time_index=time_index,
        time_region=time_region,
        ny=int(viirs_ds.sizes["y"]),
        nx=int(viirs_ds.sizes["x"]),
    )

    zarr.consolidate_metadata(str(args.satfire_zarr))

    log("Done.")
    log(f"Wrote VIIRS month {year}-{month:02d} into {args.satfire_zarr}/{args.group}")


if __name__ == "__main__":
    main()