# src/firelens/pipelines/ingest_static_group.py
from __future__ import annotations

import argparse
from pathlib import Path

from firelens.static.writer import write_static_group


def build_var_specs(static_root: Path) -> dict:
    """
    Build the default static variable registry.

    These paths assume your existing static products live under static_root.
    """
    return {
        "dem": {
            "path": static_root / "dem_500m_5070_cube_aligned.tif",
            "long_name": "Mean elevation on 500 m cube grid",
            "units": "m",
        },
        "tpi": {
            "path": static_root / "tpi_mean_500m.tif",
            "long_name": "Mean topographic position index on 500 m cube grid",
            "units": "m",
        },
        "tpi_abs_p90": {
            "path": static_root / "tpi_abs_p90_500m.tif",
            "long_name": "90th percentile absolute topographic position index on 500 m cube grid",
            "units": "m",
        },
        "tpi_support_frac": {
            "path": static_root / "tpi_support_frac_500m.tif",
            "long_name": "Valid fine-pixel support fraction for 500 m TPI aggregation",
            "units": "1",
        },
        "northness": {
            "path": static_root / "northness_mean_500m.tif",
            "long_name": "Mean northness on 500 m cube grid",
            "units": "1",
        },
        "eastness": {
            "path": static_root / "eastness_mean_500m.tif",
            "long_name": "Mean eastness on 500 m cube grid",
            "units": "1",
        },
        "slope_pct": {
            "path": static_root / "slope_mean_pct_500m.tif",
            "long_name": "Mean slope percent on 500 m cube grid",
            "units": "percent",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject aligned static TIFFs into the FireLens /static Zarr group."
    )

    parser.add_argument(
        "--cube-zarr",
        default="/scratch/groups/dsivas/qiaok/firelens_zarr/cube_500m_15min.zarr",
        help="Initialized FireLens cube Zarr store.",
    )

    parser.add_argument(
        "--static-root",
        default="/scratch/groups/dsivas/qiaok/firelens_data/ca_dem_static",
        help="Directory containing aligned 500 m static TIFFs.",
    )

    parser.add_argument(
        "--expected-epsg",
        type=int,
        default=5070,
        help="Expected CRS EPSG code for static TIFFs.",
    )

    parser.add_argument(
        "--overwrite-vars",
        action="store_true",
        help="Overwrite variables if they already exist in /static.",
    )

    args = parser.parse_args()

    cube_zarr = Path(args.cube_zarr)
    static_root = Path(args.static_root)

    var_specs = build_var_specs(static_root)

    missing = [
        str(spec["path"])
        for spec in var_specs.values()
        if not Path(spec["path"]).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing static TIFFs:\n" + "\n".join(f"  {p}" for p in missing)
        )

    write_static_group(
        cube_zarr=cube_zarr,
        var_specs=var_specs,
        consolidated=True,
        expected_epsg=args.expected_epsg,
        overwrite_vars=args.overwrite_vars,
    )


if __name__ == "__main__":
    main()