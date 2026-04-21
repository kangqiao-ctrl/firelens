# src/firelens/slow_context/raster.py
from __future__ import annotations

import numpy as np
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import reproject, transform_bounds

from firelens.grid import TargetGrid, normalize_crs


def as_target_dataarray(
    arr: np.ndarray,
    target: TargetGrid,
    *,
    name: str | None = None,
    units: str = "1",
) -> xr.DataArray:
    """
    Wrap a numpy array on the target grid as an xarray DataArray.
    """
    arr = np.asarray(arr, dtype=np.float32)

    if arr.shape != target.shape:
        raise ValueError(f"Array shape {arr.shape} does not match target shape {target.shape}")

    if target.mask is not None:
        arr = np.where(target.mask, arr, np.nan)

    return xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={
            "y": target.y,
            "x": target.x,
        },
        name=name,
        attrs={
            "units": units,
            "grid_mapping": "spatial_ref",
        },
    )


def read_source_subset(
    src: rasterio.io.DatasetReader,
    target: TargetGrid,
    *,
    band: int = 1,
    pad_pixels: int = 2,
):
    """
    Read only the part of the source raster needed to cover the target grid.

    Returns
    -------
    arr
        Masked source array.
    transform
        Raster transform for the returned window.
    """
    if pad_pixels < 0:
        raise ValueError(f"pad_pixels must be >= 0, got {pad_pixels}")

    left, bottom, right, top = target.bounds

    if src.crs is None:
        raise ValueError("Source raster has no CRS.")

    src_crs = normalize_crs(src.crs)

    if not src_crs.equals(target.crs):
        left, bottom, right, top = transform_bounds(
            target.crs_wkt,
            src_crs.to_wkt(),
            left,
            bottom,
            right,
            top,
            densify_pts=21,
        )

    win = src.window(left, bottom, right, top)

    col0 = max(0, int(np.floor(win.col_off)) - pad_pixels)
    row0 = max(0, int(np.floor(win.row_off)) - pad_pixels)
    col1 = min(src.width, int(np.ceil(win.col_off + win.width)) + pad_pixels)
    row1 = min(src.height, int(np.ceil(win.row_off + win.height)) + pad_pixels)

    if col1 <= col0 or row1 <= row0:
        raise ValueError(
            "Target grid does not overlap source raster after CRS transform. "
            f"window={win}, clipped=({col0}, {row0}, {col1}, {row1})"
        )

    win = Window(col0, row0, col1 - col0, row1 - row0)

    arr = src.read(band, window=win, masked=True)
    transform = src.window_transform(win)

    return arr, transform


def regrid_binary_fraction_full_cell(
    src_binary01,
    src_transform,
    src_crs,
    target: TargetGrid,
    *,
    name: str | None = None,
    dst_nodata: float = -9999.0,
) -> xr.DataArray:
    """
    Regrid a 0/1 array to the target grid as a full-cell fraction.

    Important:
    - src_binary01 already contains 0 for invalid / absent pixels.
    - zeros are valid numeric inputs, not nodata.
    - uncovered destination cells are set to 0.
    """
    src = np.asarray(src_binary01, dtype="float32")
    dst = np.full(target.shape, dst_nodata, dtype="float32")

    src_crs_wkt = normalize_crs(src_crs).to_wkt()

    reproject(
        source=src,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs_wkt,
        src_nodata=None,
        dst_transform=target.transform,
        dst_crs=target.crs_wkt,
        dst_nodata=dst_nodata,
        resampling=Resampling.average,
        init_dest_nodata=True,
    )

    dst[dst == dst_nodata] = 0.0

    return as_target_dataarray(
        dst,
        target,
        name=name,
        units="1",
    )


def regrid_masked_array(
    src_ma,
    src_transform,
    src_crs,
    target: TargetGrid,
    *,
    resampling: Resampling = Resampling.average,
    name: str | None = None,
    units: str = "1",
    nodata: float = -9999.0,
) -> xr.DataArray:
    """
    Regrid a masked source array to the target grid.

    Masked source pixels are ignored by average resampling.
    """
    src_filled = np.ma.array(src_ma, copy=False).astype("float32").filled(nodata)
    dst = np.full(target.shape, nodata, dtype="float32")

    src_crs_wkt = normalize_crs(src_crs).to_wkt()

    reproject(
        source=src_filled,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs_wkt,
        src_nodata=nodata,
        dst_transform=target.transform,
        dst_crs=target.crs_wkt,
        dst_nodata=nodata,
        resampling=resampling,
        init_dest_nodata=True,
    )

    dst[dst == nodata] = np.nan

    return as_target_dataarray(
        dst,
        target,
        name=name,
        units=units,
    )