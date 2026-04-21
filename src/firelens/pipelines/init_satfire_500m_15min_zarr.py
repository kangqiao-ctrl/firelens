# src/firelens/pipelines/init_satfire_500m_15min_zarr.py
from __future__ import annotations

import argparse

import xarray as xr

from firelens.store_init import ChunkPolicy, init_spatiotemporal_store
from firelens.time_axis import build_time_axis_dataset


START_UTC = "2017-01-01 00:15:00"
END_UTC = "2026-01-01 00:00:00"
FREQ = "15min"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize the 15-minute FireLens GOES/VIIRS Zarr store."
    )

    parser.add_argument(
        "--context-zarr",
        default="/scratch/groups/dsivas/qiaok/firelens_zarr/context_500m.zarr",
        help="Context store containing the canonical /grid group.",
    )

    parser.add_argument(
        "--satfire-zarr",
        default="/scratch/groups/dsivas/qiaok/firelens_zarr/satfire_500m_15min.zarr",
        help="Output 15-minute sat/fire store.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate the store if it already exists.",
    )

    parser.add_argument(
        "--chunk-time",
        type=int,
        default=96,
        help="Time chunk size, usually one day for 15-minute data.",
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

    time_15m = build_time_axis_dataset(
        start_utc=START_UTC,
        end_utc=END_UTC,
        freq=FREQ,
        title="FireLens 15-minute UTC interval-end time coordinate",
        time_semantics=(
            "timestamp marks the end of the 15-minute interval; "
            "interval represented by time t is [t - 15min, t)"
        ),
    )
    time_15m.attrs.update(
    {
        "interval_duration": "15min",
        "interval_label": "end",
        "coverage_start": "2017-01-01 00:00:00",
        "coverage_end": "2026-01-01 00:00:00",
    }
)

    init_spatiotemporal_store(
        store_zarr=args.satfire_zarr,
        grid=grid,
        time_ds=time_15m,
        groups={
            "goes": "15-minute corrected GOES variables on California 500 m grid",
            "viirs": "15-minute VIIRS rasterized detections on California 500 m grid",
        },
        chunk_policy=ChunkPolicy(
            time=args.chunk_time,
            y=args.chunk_y,
            x=args.chunk_x,
        ),
        overwrite=args.overwrite,
    )

    print(f"Initialized: {args.satfire_zarr}")


if __name__ == "__main__":
    main()