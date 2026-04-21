# src/firelens/static/terrain.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.signal import fftconvolve


def annulus_kernel(inner_px: int, outer_px: int) -> np.ndarray:
    """
    Binary annulus kernel centered on the focal cell.

    Includes cells whose centers fall in [inner_px, outer_px].
    """
    if inner_px < 0:
        raise ValueError(f"inner_px must be >= 0, got {inner_px}")

    if outer_px <= 0:
        raise ValueError(f"outer_px must be > 0, got {outer_px}")

    if outer_px < inner_px:
        raise ValueError(
            f"outer_px must be >= inner_px, got inner={inner_px}, outer={outer_px}"
        )

    y, x = np.ogrid[-outer_px : outer_px + 1, -outer_px : outer_px + 1]
    d = np.sqrt(x * x + y * y)

    kernel = ((d >= inner_px) & (d <= outer_px)).astype(np.float32)

    if kernel.sum() == 0:
        raise ValueError("Annulus kernel is empty. Check radii and pixel size.")

    return kernel


def get_valid_mask(arr: np.ndarray, nodata_value) -> np.ndarray:
    """Return True where arr contains valid finite data."""
    valid = np.isfinite(arr)

    if nodata_value is not None:
        try:
            nodata_is_nan = bool(np.isnan(nodata_value))
        except TypeError:
            nodata_is_nan = False

        if not nodata_is_nan:
            valid &= arr != nodata_value

    return valid


def write_annulus_tpi_10m(
    *,
    src_path: str | Path,
    dst_path: str | Path,
    inner_m: float,
    outer_m: float,
    tile_pixels: int = 1024,
    nodata_out: float = -9999.0,
    overwrite: bool = False,
    progress: bool = True,
) -> Path:
    """
    Compute annulus TPI at the source raster resolution using tiled processing.

    TPI = focal elevation - mean elevation in annulus [inner_m, outer_m]

    Parameters
    ----------
    src_path
        Source DEM path.
    dst_path
        Output TPI GeoTIFF path.
    inner_m, outer_m
        Annulus radii in meters.
    tile_pixels
        Tile size in source pixels.
    nodata_out
        Output nodata value.
    overwrite
        Whether to replace an existing output file.
    progress
        Whether to print tile progress.

    Returns
    -------
    Path
        Output path.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if dst_path.exists() and not overwrite:
        raise FileExistsError(f"{dst_path} already exists; pass overwrite=True")

    with rasterio.open(src_path) as src:
        xres = abs(src.transform.a)
        yres = abs(src.transform.e)

        if not np.isclose(xres, yres):
            raise ValueError(f"Pixels must be square. Got xres={xres}, yres={yres}")

        res = float(xres)

        inner_px = int(round(inner_m / res))
        outer_px = int(round(outer_m / res))

        if outer_px <= 0:
            raise ValueError(
                f"outer_m={outer_m} is too small for source resolution {res}"
            )

        halo = outer_px
        kernel = annulus_kernel(inner_px, outer_px)

        src_nodata = src.nodata
        fill_value = src_nodata if src_nodata is not None else np.nan

        profile = src.profile.copy()
        profile.update(
            dtype="float32",
            nodata=nodata_out,
            compress="LZW",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            for row in range(0, src.height, tile_pixels):
                h = min(tile_pixels, src.height - row)

                for col in range(0, src.width, tile_pixels):
                    w = min(tile_pixels, src.width - col)

                    if progress:
                        print(f"[TPI] row={row}, col={col}, size={h}x{w}")

                    halo_win = Window(
                        col - halo,
                        row - halo,
                        w + 2 * halo,
                        h + 2 * halo,
                    )

                    arr = src.read(
                        1,
                        window=halo_win,
                        boundless=True,
                        fill_value=fill_value,
                    ).astype(np.float32)

                    valid = get_valid_mask(arr, src_nodata)

                    zfill = np.where(valid, arr, 0.0).astype(np.float32)
                    vfill = valid.astype(np.float32)

                    sum_nb = fftconvolve(zfill, kernel, mode="same")
                    cnt_nb = fftconvolve(vfill, kernel, mode="same")

                    with np.errstate(divide="ignore", invalid="ignore"):
                        mean_nb = np.where(cnt_nb > 0, sum_nb / cnt_nb, np.nan)
                        tpi = np.where(valid, arr - mean_nb, np.nan)

                    core = tpi[halo : halo + h, halo : halo + w].astype(np.float32)
                    core = np.where(np.isfinite(core), core, nodata_out)

                    dst.write(core, 1, window=Window(col, row, w, h))

    print(f"Wrote TPI raster: {dst_path}")
    return dst_path