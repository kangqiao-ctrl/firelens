# src/firelens/pipelines/init_california_500m_context_zarr.py
from __future__ import annotations

import argparse
import os
from pathlib import Path

import xarray as xr

from firelens.grid import projected_bounds_from_polygon, build_grid_dataset
from firelens.store_init import ChunkPolicy, init_context_store


# ---------------------------------------------------------------------
# Canonical FireLens California 500 m grid settings
# ---------------------------------------------------------------------
TARGET_CRS = "EPSG:5070"

# Stable snap origin used in the earlier FireLens workflow.
ORIGIN_X = -2362425.0
ORIGIN_Y = 3177435.0

RESOLUTION_M = 500.0
BUFFER_M = 25_000.0


def expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser()


def build_california_grid(
    *,
    boundary_path: Path,
    add_latlon: bool = True,
) -> xr.Dataset:
    bounds = projected_bounds_from_polygon(
        boundary_path,
        target_crs=TARGET_CRS,
        buffer_m=BUFFER_M,
        resolution_m=RESOLUTION_M,
        origin_x=ORIGIN_X,
        origin_y=ORIGIN_Y,
    )

    return build_grid_dataset(
        bounds,
        target_crs=TARGET_CRS,
        origin_x=ORIGIN_X,
        origin_y=ORIGIN_Y,
        resolution_m=RESOLUTION_M,
        add_latlon=add_latlon,
        title="FireLens California 500 m EPSG:5070 grid",
        source_boundary_path=boundary_path,
        source_buffer_m=BUFFER_M,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize the FireLens California 500 m context Zarr store. "
            "Creates /grid, /static, and /slow_context. "
            "No dense time axis is created."
        )
    )

    parser.add_argument(
        "--boundary",
        default="/home/groups/dsivas/data/state_boundary/california_boundary.geojson",
        help="California boundary GeoJSON/Shapefile path.",
    )

    parser.add_argument(
        "--grid-nc",
        default="/home/groups/dsivas/qiaok/firelens/artifacts/california_500m_grid.nc",
        help="Optional output reference grid NetCDF path.",
    )

    parser.add_argument(
        "--context-zarr",
        default="/scratch/groups/dsivas/qiaok/firelens_zarr/context_500m.zarr",
        help="Output FireLens context Zarr store path.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate the context Zarr store if it already exists.",
    )

    parser.add_argument(
        "--no-latlon",
        action="store_true",
        help="Do not create 2-D lon/lat auxiliary coordinates in /grid.",
    )

    parser.add_argument(
        "--skip-grid-nc",
        action="store_true",
        help="Do not write the reference grid NetCDF.",
    )

    parser.add_argument(
        "--chunk-y",
        type=int,
        default=512,
        help="Default y chunk size for spatial variables.",
    )

    parser.add_argument(
        "--chunk-x",
        type=int,
        default=512,
        help="Default x chunk size for spatial variables.",
    )

    args = parser.parse_args()

    boundary_path = expand_path(args.boundary)
    grid_nc = expand_path(args.grid_nc)
    context_zarr = expand_path(args.context_zarr)

    if not boundary_path.exists():
        raise FileNotFoundError(f"Boundary file does not exist: {boundary_path}")

    if args.chunk_y <= 0 or args.chunk_x <= 0:
        raise ValueError("chunk-y and chunk-x must be positive.")

    if not args.skip_grid_nc:
        grid_nc.parent.mkdir(parents=True, exist_ok=True)

    context_zarr.parent.mkdir(parents=True, exist_ok=True)

    print("Building California 500 m grid...")
    grid = build_california_grid(
        boundary_path=boundary_path,
        add_latlon=not args.no_latlon,
    )

    print(grid)
    print("Grid sizes:", dict(grid.sizes))
    print("Number of cells:", grid.sizes["y"] * grid.sizes["x"])

    if not args.skip_grid_nc:
        print(f"Writing reference grid NetCDF: {grid_nc}")
        grid.to_netcdf(grid_nc)

    chunk_policy = ChunkPolicy(
        time=None,
        y=args.chunk_y,
        x=args.chunk_x,
        vintage=1,
    )

    print(f"Initializing context Zarr store: {context_zarr}")
    init_context_store(
        context_zarr=context_zarr,
        grid=grid,
        overwrite=args.overwrite,
        chunk_policy=chunk_policy,
    )

    print("\nInitialized groups:")
    for group in ["grid", "static", "slow_context"]:
        ds = xr.open_zarr(context_zarr, group=group, consolidated=True)
        print(f"\n[{group}]")
        print(ds)

    print("\nDone.")
    if not args.skip_grid_nc:
        print(f"Grid NetCDF:    {grid_nc}")
    print(f"Context Zarr:   {context_zarr}")


if __name__ == "__main__":
    main()