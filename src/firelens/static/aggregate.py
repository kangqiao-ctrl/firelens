# src/firelens/static/aggregate.py
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from firelens.grid import load_template_grid_from_nc
from firelens.static.terrain import get_valid_mask


def aggregate_fine_raster_to_template(
    *,
    src_path: str | Path,
    template_nc: str | Path,
    out_mean: str | Path | None = None,
    out_abs_p90: str | Path | None = None,
    out_support_frac: str | Path | None = None,
    x_name: str = "x",
    y_name: str = "y",
    coarse_tile: int = 32,
    nodata_out: float = -9999.0,
    min_valid_frac: float = 0.0,
    strict_alignment: bool = True,
    overwrite: bool = False,
) -> dict[str, Path]:
    """
    Aggregate a fine-resolution raster onto the exact grid in template_nc.

    Outputs can include:
    - mean
    - p90(abs(value))
    - valid support fraction

    This assumes the template grid is an integer multiple of the source grid
    and, when strict_alignment=True, exactly aligned to the source raster.
    """
    src_path = Path(src_path)
    template_nc = Path(template_nc)

    outputs_requested = {
        "mean": Path(out_mean) if out_mean is not None else None,
        "abs_p90": Path(out_abs_p90) if out_abs_p90 is not None else None,
        "support_frac": Path(out_support_frac) if out_support_frac is not None else None,
    }

    outputs_requested = {k: v for k, v in outputs_requested.items() if v is not None}

    if not outputs_requested:
        raise ValueError("At least one output path must be provided.")

    for out_path in outputs_requested.values():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"{out_path} already exists; pass overwrite=True")

    out_width, out_height, out_transform, dx, dy = load_template_grid_from_nc(
        template_nc,
        x_name=x_name,
        y_name=y_name,
    )

    with rasterio.open(src_path) as src:
        xres = abs(src.transform.a)
        yres = abs(src.transform.e)

        fx = dx / xres
        fy = dy / yres

        if not (np.isclose(fx, round(fx)) and np.isclose(fy, round(fy))):
            raise ValueError(
                "Template resolution must be an integer multiple of source resolution. "
                f"source=({xres}, {yres}), template=({dx}, {dy})"
            )

        fx = int(round(fx))
        fy = int(round(fy))

        col0_f, row0_f = (~src.transform) * (out_transform.c, out_transform.f)

        if strict_alignment:
            if not (
                np.isclose(col0_f, round(col0_f))
                and np.isclose(row0_f, round(row0_f))
            ):
                raise ValueError(
                    "Template grid is not aligned to the fine raster grid. "
                    f"Source offset is col={col0_f}, row={row0_f}."
                )

        profile = src.profile.copy()
        profile.update(
            dtype="float32",
            nodata=nodata_out,
            width=out_width,
            height=out_height,
            transform=out_transform,
            compress="LZW",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        open_outputs = {
            name: rasterio.open(path, "w", **profile)
            for name, path in outputs_requested.items()
        }

        src_nodata = src.nodata
        fill_value = src_nodata if src_nodata is not None else np.nan
        full_count = float(fx * fy)

        try:
            for out_row in range(0, out_height, coarse_tile):
                ch = min(coarse_tile, out_height - out_row)

                for out_col in range(0, out_width, coarse_tile):
                    cw = min(coarse_tile, out_width - out_col)

                    x_left = out_transform.c + out_col * dx
                    y_top = out_transform.f - out_row * dy

                    col_off_f, row_off_f = (~src.transform) * (x_left, y_top)
                    col_off = int(round(col_off_f))
                    row_off = int(round(row_off_f))

                    fine_win = Window(col_off, row_off, cw * fx, ch * fy)

                    arr = src.read(
                        1,
                        window=fine_win,
                        boundless=True,
                        fill_value=fill_value,
                    ).astype(np.float32)

                    valid = get_valid_mask(arr, src_nodata)
                    arr = np.where(valid, arr, np.nan)

                    arr4 = arr.reshape(ch, fy, cw, fx)
                    valid4 = np.isfinite(arr4)

                    support_frac = (
                        valid4.sum(axis=(1, 3)).astype(np.float32) / full_count
                    )

                    keep = support_frac >= min_valid_frac

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)

                        if "mean" in open_outputs:
                            mean_block = np.nanmean(arr4, axis=(1, 3)).astype(np.float32)

                        if "abs_p90" in open_outputs:
                            abs_p90_block = np.nanpercentile(
                                np.abs(arr4),
                                90,
                                axis=(1, 3),
                            ).astype(np.float32)

                    if "mean" in open_outputs:
                        block = np.where(
                            keep & np.isfinite(mean_block),
                            mean_block,
                            nodata_out,
                        )
                        open_outputs["mean"].write(
                            block.astype(np.float32),
                            1,
                            window=Window(out_col, out_row, cw, ch),
                        )

                    if "abs_p90" in open_outputs:
                        block = np.where(
                            keep & np.isfinite(abs_p90_block),
                            abs_p90_block,
                            nodata_out,
                        )
                        open_outputs["abs_p90"].write(
                            block.astype(np.float32),
                            1,
                            window=Window(out_col, out_row, cw, ch),
                        )

                    if "support_frac" in open_outputs:
                        block = np.where(
                            np.isfinite(support_frac),
                            support_frac,
                            nodata_out,
                        )
                        open_outputs["support_frac"].write(
                            block.astype(np.float32),
                            1,
                            window=Window(out_col, out_row, cw, ch),
                        )

        finally:
            for ds in open_outputs.values():
                ds.close()

    print(f"Wrote 500 m aggregate products from: {src_path}")
    return outputs_requested


def aggregate_tpi_to_target_grid(
    *,
    src_path: str | Path,
    template_nc: str | Path,
    out_mean: str | Path,
    out_abs_p90: str | Path,
    out_support_frac: str | Path | None = None,
    x_name: str = "x",
    y_name: str = "y",
    coarse_tile: int = 32,
    nodata_out: float = -9999.0,
    min_valid_frac: float = 0.0,
    strict_alignment: bool = True,
    overwrite: bool = False,
) -> dict[str, Path]:
    """
    Backward-compatible TPI-specific wrapper.
    """
    return aggregate_fine_raster_to_template(
        src_path=src_path,
        template_nc=template_nc,
        out_mean=out_mean,
        out_abs_p90=out_abs_p90,
        out_support_frac=out_support_frac,
        x_name=x_name,
        y_name=y_name,
        coarse_tile=coarse_tile,
        nodata_out=nodata_out,
        min_valid_frac=min_valid_frac,
        strict_alignment=strict_alignment,
        overwrite=overwrite,
    )