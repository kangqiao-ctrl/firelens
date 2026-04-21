# src/firelens/pipelines/init_met_500m_1h_zarr.py
from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

from firelens.store_init import ChunkPolicy, init_spatiotemporal_store
from firelens.time_axis import build_time_axis_dataset


START_UTC = "2017-01-01 00:00:00"
END_UTC = "2025-12-31 23:00:00"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize the hourly FireLens meteorology Zarr store."
    )

    parser.add_argument(
        "--context-zarr",
        default="/scratch/groups/dsivas/qiaok/firelens_zarr/context_500m.zarr",
        help="Context store containing the canonical /grid group.",
    )

    parser.add_argument(
        "--met-zarr",
        default="/scratch/groups/dsivas/qiaok/firelens_zarr/met_500m_1h.zarr",
        help="Output hourly met store.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate the store if it already exists.",
    )

    parser.add_argument(
        "--chunk-time",
        type=int,
        default=24,
        help="Time chunk size, usually one day for hourly data.",
    )

    parser.add_argument(
        "--chunk-y",
        type=int,
        default=512,
        help="Spatial y chunk size.",
    )

    parser.add_argument(
        "--chunk-x",
        type=int,
        default=512,
        help="Spatial x chunk size.",
    )

    args = parser.parse_args()

    grid = xr.open_zarr(
        args.context_zarr,
        group="grid",
        consolidated=True,
    )

    time_1h = build_time_axis_dataset(
        start_utc=START_UTC,
        end_utc=END_UTC,
        freq="1h",
        title="FireLens hourly UTC time coordinate",
        time_semantics="timestamp marks the end of the hourly feature availability/aggregation interval",
    )

    init_spatiotemporal_store(
        store_zarr=args.met_zarr,
        grid=grid,
        time_ds=time_1h,
        groups={
            "aorc": "Hourly AORC variables on California 500 m grid",
            "hrrr": "Hourly HRRR variables on California 500 m grid",
        },
        chunk_policy=ChunkPolicy(
            time=args.chunk_time,
            y=args.chunk_y,
            x=args.chunk_x,
        ),
        overwrite=args.overwrite,
    )

    print(f"Initialized: {args.met_zarr}")


if __name__ == "__main__":
    main()