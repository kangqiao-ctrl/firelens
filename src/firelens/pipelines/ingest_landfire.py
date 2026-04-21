# src/firelens/pipelines/ingest_landfire.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from firelens.grid import build_target_grid_from_zarr
from firelens.slow_context.landfire_catalog import (
    default_product_root_map,
    resolve_product_files,
)
from firelens.slow_context.landfire_fbfm40 import fbfm40_dataset
from firelens.slow_context.landfire_canopy import cc_dataset, ch_dataset
from firelens.slow_context.landfire_fdist import fdist_dataset
from firelens.slow_context.outputs import prefix_dataset_vars
from firelens.slow_context.vintage import YEAR_TO_VINTAGE, year_to_vintage_json
from firelens.slow_context.writer import write_slow_context_dataset


DEFAULT_VINTAGES = ["LF2016", "LF2020", "LF2022", "LF2023", "LF2024"]
DEFAULT_PRODUCTS = ["fbfm40", "cc", "ch", "fdist"]


def _require_product_root(
    product_roots: dict[str, Any],
    *,
    vintage: str,
    product: str,
) -> str:
    """
    Get LANDFIRE product search root from product_roots[vintage][product].
    """
    if vintage not in product_roots:
        raise KeyError(
            f"Vintage {vintage!r} missing from product root map. "
            f"Available vintages: {list(product_roots)}"
        )

    if product not in product_roots[vintage]:
        raise KeyError(
            f"Product {product!r} missing for vintage {vintage!r}. "
            f"Available products: {list(product_roots[vintage])}"
        )

    return product_roots[vintage][product]


def _build_one_product_dataset(
    *,
    product: str,
    vintage: str,
    product_root: str,
    target,
    output_mode: str,
    keep_valid_fraction: bool,
    keep_qc_vars: bool,
) -> xr.Dataset:
    """
    Build one LANDFIRE product dataset for one vintage.
    """
    print(f"  Building product={product}, vintage={vintage}")
    print(f"  Search root: {product_root}")

    files = resolve_product_files(product, product_root)

    tif_uri = files["tif_uri"]
    csv_path = files["csv_path"]

    print(f"  TIF: {tif_uri}")
    print(f"  CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    if product == "fbfm40":
        return fbfm40_dataset(
            fbfm40_tif=tif_uri,
            fbfm40_df=df,
            target=target,
            output_mode=output_mode,
            keep_valid_fraction=keep_valid_fraction,
            keep_family_sum=keep_qc_vars,
        )

    if product == "cc":
        return cc_dataset(
            cc_tif=tif_uri,
            cc_df=df,
            target=target,
            output_mode=output_mode,
            keep_valid_fraction=keep_valid_fraction,
            keep_qc_vars=keep_qc_vars,
        )

    if product == "ch":
        return ch_dataset(
            ch_tif=tif_uri,
            ch_df=df,
            target=target,
            output_mode=output_mode,
            keep_valid_fraction=keep_valid_fraction,
            keep_qc_vars=keep_qc_vars,
        )

    if product == "fdist":
        return fdist_dataset(
            fdist_tif=tif_uri,
            fdist_df=df,
            target=target,
            output_mode=output_mode,
            keep_valid_fraction=keep_valid_fraction,
            keep_family_sum=keep_qc_vars,
        )

    raise ValueError(f"Unsupported LANDFIRE product: {product!r}")


def build_landfire_vintage_dataset(
    *,
    context_zarr: str | Path,
    product_roots: dict[str, Any],
    vintages: list[str] = DEFAULT_VINTAGES,
    products: list[str] = DEFAULT_PRODUCTS,
    output_mode: str = "full",
    keep_valid_fraction: bool = True,
    keep_qc_vars: bool = False,
    year_to_vintage: dict[int, str] = YEAR_TO_VINTAGE,
    prefix: str = "landfire",
) -> xr.Dataset:
    """
    Build a LANDFIRE slow-context dataset with dimensions:

        vintage, y, x

    Physical storage uses product vintage, not dense context_time.
    """
    target = build_target_grid_from_zarr(
        context_zarr,
        group="grid",
        consolidated=True,
        x_name="x",
        y_name="y",
        crs=None,          # infer from /grid spatial_ref
        mask_var=None,
    )

    per_vintage: list[xr.Dataset] = []

    for vintage in vintages:
        print(f"\n=== LANDFIRE vintage {vintage} ===")

        product_datasets: list[xr.Dataset] = []

        for product in products:
            product_root = _require_product_root(
                product_roots,
                vintage=vintage,
                product=product,
            )

            ds_product = _build_one_product_dataset(
                product=product,
                vintage=vintage,
                product_root=product_root,
                target=target,
                output_mode=output_mode,
                keep_valid_fraction=keep_valid_fraction,
                keep_qc_vars=keep_qc_vars,
            )

            product_datasets.append(ds_product)

        if not product_datasets:
            raise ValueError("No LANDFIRE product datasets were built.")

        ds_v = xr.merge(product_datasets, compat="override", join="exact")

        if prefix:
            ds_v = prefix_dataset_vars(ds_v, prefix)

        ds_v = ds_v.expand_dims(
            vintage=np.asarray([vintage], dtype="U16")
        )

        per_vintage.append(ds_v)

    ds = xr.concat(per_vintage, dim="vintage")
    ds = ds.assign_coords(
        vintage=("vintage", np.asarray(vintages, dtype="U16"))
    )

    ds.attrs.update(
        {
            "group_role": "slow_context",
            "source": "LANDFIRE",
            "physical_time_axis": "vintage",
            "year_to_vintage_json": year_to_vintage_json(year_to_vintage),
            "output_mode": output_mode,
            "keep_valid_fraction": str(bool(keep_valid_fraction)),
            "keep_qc_vars": str(bool(keep_qc_vars)),
            "products": json.dumps(products),
            "vintages": json.dumps(vintages),
        }
    )

    return ds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and inject LANDFIRE slow-context variables into /slow_context."
    )

    parser.add_argument(
        "--context-zarr",
        "--cube-zarr",
        dest="context_zarr",
        default="/scratch/groups/dsivas/qiaok/firelens_zarr/context_500m.zarr",
        help="Initialized FireLens context Zarr store.",
    )

    parser.add_argument(
        "--landfire-root",
        default="/home/groups/dsivas/data/landfire",
        help="Root directory for unzipped LANDFIRE products.",
    )

    parser.add_argument(
        "--vintages",
        nargs="+",
        default=DEFAULT_VINTAGES,
        help="LANDFIRE vintages to ingest.",
    )

    parser.add_argument(
        "--products",
        nargs="+",
        default=DEFAULT_PRODUCTS,
        choices=DEFAULT_PRODUCTS,
        help="LANDFIRE products to ingest.",
    )

    parser.add_argument(
        "--output-mode",
        default="full",
        choices=["full", "frac", "both"],
        help=(
            "'full' writes whole-cell variables only; "
            "'frac' writes valid-support-normalized variables; "
            "'both' writes both."
        ),
    )

    parser.add_argument(
        "--drop-valid-fraction",
        action="store_true",
        help="Do not write *_valid_fraction variables.",
    )

    parser.add_argument(
        "--keep-qc-vars",
        action="store_true",
        help="Keep QC variables such as family sums or CH bin sums.",
    )

    parser.add_argument(
        "--overwrite-vars",
        action="store_true",
        help="Overwrite existing variables in /slow_context.",
    )

    args = parser.parse_args()

    product_roots = default_product_root_map(
        root=args.landfire_root,
        vintages=args.vintages,
        products=args.products,
        check_exists=True,
    )

    ds = build_landfire_vintage_dataset(
        context_zarr=args.context_zarr,
        product_roots=product_roots,
        vintages=args.vintages,
        products=args.products,
        output_mode=args.output_mode,
        keep_valid_fraction=not args.drop_valid_fraction,
        keep_qc_vars=args.keep_qc_vars,
    )

    print("\nBuilt LANDFIRE dataset:")
    print(ds)

    write_slow_context_dataset(
        cube_zarr=args.context_zarr,
        ds=ds,
        overwrite_vars=args.overwrite_vars,
    )


if __name__ == "__main__":
    main()