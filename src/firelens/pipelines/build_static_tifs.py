# src/firelens/pipelines/build_static_tifs.py
from __future__ import annotations

import argparse
from pathlib import Path

from firelens.static.aggregate import aggregate_tpi_to_target_grid
from firelens.static.terrain import write_annulus_tpi_10m


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build derived static terrain TIFFs for FireLens."
    )

    parser.add_argument(
        "--dem-10m",
        default="/scratch/groups/dsivas/qiaok/firelens_data/ca_dem_static/dem_10m_5070_cube_aligned.tif",
        help="Input 10 m DEM aligned to EPSG:5070.",
    )

    parser.add_argument(
        "--grid-nc",
        default="./grid/california_500m_grid.nc",
        help="Canonical FireLens 500 m grid NetCDF.",
    )

    parser.add_argument(
        "--out-dir",
        default="/scratch/groups/dsivas/qiaok/firelens_data/ca_dem_static",
        help="Output directory for derived static TIFFs.",
    )

    parser.add_argument("--inner-m", type=float, default=10.0)
    parser.add_argument("--outer-m", type=float, default=2000.0)
    parser.add_argument("--tile-pixels", type=int, default=1024)
    parser.add_argument("--coarse-tile", type=int, default=32)
    parser.add_argument("--nodata-out", type=float, default=-9999.0)
    parser.add_argument("--min-valid-frac", type=float, default=0.0)

    parser.add_argument(
        "--skip-tpi-10m",
        action="store_true",
        help="Skip writing the 10 m TPI intermediate and only aggregate existing file.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output TIFFs.",
    )

    args = parser.parse_args()

    dem_10m = Path(args.dem_10m)
    grid_nc = Path(args.grid_nc)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tpi_10m = out_dir / "tpi2k_10m.tif"
    tpi_mean_500m = out_dir / "tpi_mean_500m.tif"
    tpi_abs_p90_500m = out_dir / "tpi_abs_p90_500m.tif"
    tpi_support_frac_500m = out_dir / "tpi_support_frac_500m.tif"

    if not args.skip_tpi_10m:
        write_annulus_tpi_10m(
            src_path=dem_10m,
            dst_path=tpi_10m,
            inner_m=args.inner_m,
            outer_m=args.outer_m,
            tile_pixels=args.tile_pixels,
            nodata_out=args.nodata_out,
            overwrite=args.overwrite,
        )

    aggregate_tpi_to_target_grid(
        src_path=tpi_10m,
        template_nc=grid_nc,
        out_mean=tpi_mean_500m,
        out_abs_p90=tpi_abs_p90_500m,
        out_support_frac=tpi_support_frac_500m,
        coarse_tile=args.coarse_tile,
        nodata_out=args.nodata_out,
        min_valid_frac=args.min_valid_frac,
        strict_alignment=True,
        overwrite=args.overwrite,
    )

    print("\nStatic TIFF build complete.")
    print(f"TPI 10 m:          {tpi_10m}")
    print(f"TPI mean 500 m:    {tpi_mean_500m}")
    print(f"TPI abs p90 500 m: {tpi_abs_p90_500m}")
    print(f"TPI support frac:  {tpi_support_frac_500m}")


if __name__ == "__main__":
    main()