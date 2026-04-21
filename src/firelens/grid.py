# firelens/grid.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.transform import Affine, from_origin

@dataclass(frozen=True)
class TargetGrid:
    x: np.ndarray
    y: np.ndarray
    crs: CRS
    transform: Affine
    width: int
    height: int
    mask: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left = float(self.transform.c)
        top = float(self.transform.f)
        right = left + self.width * float(self.transform.a)
        bottom = top + self.height * float(self.transform.e)
        return (left, bottom, right, top)

    @property
    def crs_wkt(self) -> str:
        """
        Rasterio/GDAL-safe CRS representation.
        """
        return self.crs.to_wkt()

def normalize_crs(crs_like: Any) -> CRS:
    """
    Normalize CRS-like input to pyproj.CRS.

    Accepts:
    - "EPSG:5070"
    - EPSG integers
    - WKT strings
    - pyproj.CRS
    - rasterio CRS or any object exposing .to_wkt()
    """
    if crs_like is None:
        raise ValueError("crs_like cannot be None")

    if isinstance(crs_like, CRS):
        return crs_like

    if hasattr(crs_like, "to_wkt"):
        return CRS.from_wkt(crs_like.to_wkt())

    return CRS.from_user_input(crs_like)

def _crs_attr_string(crs: CRS) -> str:
    """
    Compact CRS string for attrs when possible.
    """
    authority = crs.to_authority()
    if authority is not None:
        return f"{authority[0]}:{authority[1]}"
    return crs.to_string()


def _infer_crs_from_dataset(
    ds: xr.Dataset,
    *,
    fallback: str = "EPSG:5070",
) -> CRS:
    """
    Infer CRS from Dataset attrs or spatial_ref attrs.
    Prefer WKT/spatial_ref because they preserve full projection metadata.
    """
    spatial_ref = None

    if "spatial_ref" in ds.coords:
        spatial_ref = ds.coords["spatial_ref"]
    elif "spatial_ref" in ds:
        spatial_ref = ds["spatial_ref"]

    if spatial_ref is not None:
        if "spatial_ref" in spatial_ref.attrs:
            return normalize_crs(spatial_ref.attrs["spatial_ref"])
        if "crs_wkt" in spatial_ref.attrs:
            return normalize_crs(spatial_ref.attrs["crs_wkt"])

    if "crs_wkt" in ds.attrs:
        return normalize_crs(ds.attrs["crs_wkt"])

    if "crs" in ds.attrs:
        return normalize_crs(ds.attrs["crs"])

    return normalize_crs(fallback)

def snap_down(v: float, origin: float, res: float) -> float:
    """Snap coordinate downward to a grid defined by origin and resolution."""
    return float(origin + np.floor((v - origin) / res) * res)


def snap_up(v: float, origin: float, res: float) -> float:
    """Snap coordinate upward to a grid defined by origin and resolution."""
    return float(origin + np.ceil((v - origin) / res) * res)


def projected_bounds_from_polygon(
    polygon_path: str | Path,
    *,
    target_crs: str | CRS,
    buffer_m: float,
    resolution_m: float,
    origin_x: float,
    origin_y: float,
) -> tuple[float, float, float, float]:
    """
    Read a polygon, project it, buffer it, and return snapped projected bounds.

    Returns
    -------
    xmin, ymin, xmax, ymax
        Bounds in target_crs, snapped to the requested grid.
    """
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be positive, got {resolution_m}")

    polygon_path = Path(polygon_path)

    target_crs_obj = normalize_crs(target_crs)
    gdf = gpd.read_file(polygon_path).to_crs(target_crs_obj)

    # GeoPandas has changed preferred dissolve APIs across versions.
    if hasattr(gdf.geometry, "union_all"):
        geom = gdf.geometry.union_all()
    else:
        geom = gdf.geometry.unary_union

    geom = geom.buffer(buffer_m)
    xmin, ymin, xmax, ymax = geom.bounds

    xmin = snap_down(xmin, origin_x, resolution_m)
    xmax = snap_up(xmax, origin_x, resolution_m)
    ymin = snap_down(ymin, origin_y, resolution_m)
    ymax = snap_up(ymax, origin_y, resolution_m)

    return xmin, ymin, xmax, ymax


def _validate_projected_bounds(
    bounds_proj: tuple[float, float, float, float],
    *,
    resolution_m: float,
) -> tuple[int, int]:
    """Validate snapped bounds and return nx, ny."""
    xmin, ymin, xmax, ymax = map(float, bounds_proj)

    if not np.all(np.isfinite([xmin, ymin, xmax, ymax])):
        raise ValueError(f"Bounds must be finite, got {bounds_proj}")

    if xmax <= xmin:
        raise ValueError(f"xmax must be greater than xmin, got {bounds_proj}")

    if ymax <= ymin:
        raise ValueError(f"ymax must be greater than ymin, got {bounds_proj}")

    width = (xmax - xmin) / resolution_m
    height = (ymax - ymin) / resolution_m

    if not np.isclose(width, round(width), atol=1e-9):
        raise ValueError(f"x bounds are not aligned to resolution: width/res={width}")

    if not np.isclose(height, round(height), atol=1e-9):
        raise ValueError(f"y bounds are not aligned to resolution: height/res={height}")

    nx = int(round(width))
    ny = int(round(height))

    if nx <= 0 or ny <= 0:
        raise ValueError(f"Grid has non-positive shape: nx={nx}, ny={ny}")

    return nx, ny


def _gdal_geotransform(transform: Affine) -> tuple[float, float, float, float, float, float]:
    """
    Return GDAL GeoTransform order:
    xmin, xres, xskew, ymax, yskew, yres
    """
    return (
        float(transform.c),
        float(transform.a),
        float(transform.b),
        float(transform.f),
        float(transform.d),
        float(transform.e),
    )


def _affine_transform_coeffs(transform: Affine) -> tuple[float, float, float, float, float, float]:
    """
    Return Affine coefficients in rasterio/affine order:
    a, b, c, d, e, f
    """
    return (
        float(transform.a),
        float(transform.b),
        float(transform.c),
        float(transform.d),
        float(transform.e),
        float(transform.f),
    )


def _build_spatial_ref_attrs(
    *,
    target_crs: str | CRS,
    transform: Affine,
) -> dict[str, Any]:
    """
    Build CF/GDAL-style spatial_ref attributes.

    Uses pyproj CRS.to_cf() when available, then adds WKT and GeoTransform.
    """
    crs = normalize_crs(target_crs)
    crs_wkt = crs.to_wkt()

    attrs: dict[str, Any] = {}

    try:
        cf_attrs = crs.to_cf()
        attrs.update(cf_attrs)
    except Exception:
        # Keep this robust even if pyproj cannot export a full CF mapping.
        pass

    attrs.update(
        {
            "crs_wkt": crs_wkt,
            "spatial_ref": crs_wkt,
            "GeoTransform": " ".join(str(v) for v in _gdal_geotransform(transform)),
        }
    )

    return attrs


def build_grid_dataset(
    bounds_proj: tuple[float, float, float, float],
    *,
    target_crs: str | CRS,
    origin_x: float,
    origin_y: float,
    resolution_m: float,
    add_latlon: bool = True,
    title: str = "FireLens base grid",
    source_boundary_path: str | Path | None = None,
    source_buffer_m: float | None = None,
) -> xr.Dataset:
    """
    Build the canonical FireLens model grid from snapped projected bounds.

    Parameters
    ----------
    bounds_proj
        Tuple of xmin, ymin, xmax, ymax in target_crs.
    target_crs
        CRS string, e.g. "EPSG:5070".
    origin_x, origin_y
        Snap origin used to define the global grid alignment.
    resolution_m
        Grid resolution in projected units. For your current cube, 500.0.
    add_latlon
        If True, add 2-D lon/lat auxiliary coordinates.
    title
        Dataset title.
    source_boundary_path
        Optional source polygon path for provenance.
    source_buffer_m
        Optional buffer distance used around the source polygon.

    Returns
    -------
    xr.Dataset
        Dataset with x/y coordinates, x_edge/y_edge coordinates,
        spatial_ref, optional lon/lat, and a domain_mask.
    """
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be positive, got {resolution_m}")

    xmin, ymin, xmax, ymax = map(float, bounds_proj)
    nx, ny = _validate_projected_bounds(bounds_proj, resolution_m=resolution_m)

    # Cell centers
    x = xmin + resolution_m / 2.0 + np.arange(nx, dtype=np.float64) * resolution_m
    y = ymax - resolution_m / 2.0 - np.arange(ny, dtype=np.float64) * resolution_m

    # Cell edges
    x_edge = xmin + np.arange(nx + 1, dtype=np.float64) * resolution_m
    y_edge = ymax - np.arange(ny + 1, dtype=np.float64) * resolution_m

    transform = from_origin(xmin, ymax, resolution_m, resolution_m)

    crs = normalize_crs(target_crs)
    crs_wkt = crs.to_wkt()
    crs_attr = _crs_attr_string(crs)

    domain_mask = xr.DataArray(
        np.ones((ny, nx), dtype=np.uint8),
        dims=("y", "x"),
        coords={
            "y": ("y", y),
            "x": ("x", x),
        },
        name="domain_mask",
        attrs={
            "long_name": "Rectangular grid domain mask",
            "flag_values": [0, 1],
            "flag_meanings": "outside inside",
            "grid_mapping": "spatial_ref",
        },
    )

    attrs: dict[str, Any] = {
        "title": title,
        "crs": crs_attr,
        "crs_wkt": crs_wkt,
        "resolution_m": float(resolution_m),
        "xmin": float(xmin),
        "ymin": float(ymin),
        "xmax": float(xmax),
        "ymax": float(ymax),
        "origin_x": float(origin_x),
        "origin_y": float(origin_y),
        "input_bounds_proj": [float(xmin), float(ymin), float(xmax), float(ymax)],
        "affine_transform": _affine_transform_coeffs(transform),
        "gdal_geotransform": _gdal_geotransform(transform),
        "axis_order": "x,y",
    }

    if source_boundary_path is not None:
        attrs["source_boundary_path"] = str(source_boundary_path)

    if source_buffer_m is not None:
        attrs["source_buffer_m"] = float(source_buffer_m)

    ds = xr.Dataset(
        data_vars={
            "domain_mask": domain_mask,
        },
        coords={
            "x": ("x", x),
            "y": ("y", y),
            "x_edge": ("x_edge", x_edge),
            "y_edge": ("y_edge", y_edge),
        },
        attrs=attrs,
    )

    spatial_ref = xr.DataArray(
        0,
        name="spatial_ref",
        attrs=_build_spatial_ref_attrs(target_crs=crs, transform=transform),
    )
    ds = ds.assign_coords(spatial_ref=spatial_ref)

    if add_latlon:
        xx, yy = np.meshgrid(x, y)

        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(xx, yy)

        lon_da = xr.DataArray(
            lon.astype(np.float32),
            dims=("y", "x"),
            coords={"y": y, "x": x},
            name="lon",
            attrs={
                "standard_name": "longitude",
                "units": "degrees_east",
            },
        )

        lat_da = xr.DataArray(
            lat.astype(np.float32),
            dims=("y", "x"),
            coords={"y": y, "x": x},
            name="lat",
            attrs={
                "standard_name": "latitude",
                "units": "degrees_north",
            },
        )

        ds = ds.assign_coords(lon=lon_da, lat=lat_da)

    return ds


def build_grid_from_projected_bounds(
    bounds_proj: tuple[float, float, float, float],
    *,
    target_crs: str | CRS,
    origin_x: float,
    origin_y: float,
    resolution_m: float,
    add_latlon: bool = True,
    title: str = "FireLens base grid",
    source_boundary_path: str | Path | None = None,
    source_buffer_m: float | None = None,
) -> xr.Dataset:
    """
    Backward-compatible wrapper.

    Prefer build_grid_dataset() in new code.
    """
    return build_grid_dataset(
        bounds_proj,
        target_crs=target_crs,
        origin_x=origin_x,
        origin_y=origin_y,
        resolution_m=resolution_m,
        add_latlon=add_latlon,
        title=title,
        source_boundary_path=source_boundary_path,
        source_buffer_m=source_buffer_m,
    )


def transform_from_xy(
    x: np.ndarray,
    y: np.ndarray,
) -> Affine:
    """
    Build a rasterio Affine transform from 1-D x/y cell-center coordinates.

    Assumes:
    - x is increasing
    - y is decreasing
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Expected 1-D x and y coordinates.")

    if len(x) < 2 or len(y) < 2:
        raise ValueError("Need at least two x and y coordinates.")

    dx = float(x[1] - x[0])
    dy = float(y[0] - y[1])

    if dx <= 0:
        raise ValueError("Expected x coordinate to be increasing.")

    if dy <= 0:
        raise ValueError("Expected y coordinate to be decreasing.")

    return from_origin(
        float(x[0] - dx / 2.0),
        float(y[0] + dy / 2.0),
        dx,
        dy,
    )


def load_template_grid_from_nc(
    template_nc: str | Path,
    *,
    x_name: str = "x",
    y_name: str = "y",
) -> tuple[int, int, Affine, float, float]:
    """
    Load output width, height, transform, dx, dy from a template grid NetCDF.

    This is useful for raster aggregation / validation code.
    """
    with xr.open_dataset(template_nc, decode_coords="all") as ds:
        if x_name not in ds.coords:
            raise KeyError(f"Missing x coordinate {x_name!r} in {template_nc}")

        if y_name not in ds.coords:
            raise KeyError(f"Missing y coordinate {y_name!r} in {template_nc}")

        x = np.asarray(ds[x_name].values, dtype=float)
        y = np.asarray(ds[y_name].values, dtype=float)

    transform = transform_from_xy(x, y)

    dx = float(x[1] - x[0])
    dy = float(y[0] - y[1])

    width = int(len(x))
    height = int(len(y))

    return width, height, transform, dx, dy

def build_target_grid_from_dataset(
    ds: xr.Dataset,
    *,
    x_name: str = "x",
    y_name: str = "y",
    crs: str | CRS | None = None,
    mask_var: str | None = None,
) -> TargetGrid:
    if x_name not in ds.coords:
        raise KeyError(f"Missing x coordinate {x_name!r}")

    if y_name not in ds.coords:
        raise KeyError(f"Missing y coordinate {y_name!r}")

    x = np.asarray(ds[x_name].values, dtype=np.float64)
    y = np.asarray(ds[y_name].values, dtype=np.float64)

    transform = transform_from_xy(x, y)

    mask = None
    if mask_var is not None:
        if mask_var not in ds:
            raise KeyError(f"mask_var={mask_var!r} not found in dataset")
        mask = np.asarray(ds[mask_var].values).astype(bool)

    crs_obj = normalize_crs(crs) if crs is not None else _infer_crs_from_dataset(ds)

    return TargetGrid(
        x=x,
        y=y,
        crs=crs_obj,
        transform=transform,
        width=len(x),
        height=len(y),
        mask=mask,
    )


def build_target_grid_from_nc(
    grid_nc: str | Path,
    *,
    x_name: str = "x",
    y_name: str = "y",
    crs: str | CRS | None = None,
    mask_var: str | None = None,
) -> TargetGrid:
    with xr.open_dataset(grid_nc, decode_coords="all") as ds:
        return build_target_grid_from_dataset(
            ds,
            x_name=x_name,
            y_name=y_name,
            crs=crs,
            mask_var=mask_var,
        )


def build_target_grid_from_zarr(
    cube_zarr: str | Path,
    *,
    group: str = "static",
    consolidated: bool | None = True,
    x_name: str = "x",
    y_name: str = "y",
    crs: str | CRS | None = None,
    mask_var: str | None = None,
) -> TargetGrid:
    ds = xr.open_zarr(cube_zarr, group=group, consolidated=consolidated)
    return build_target_grid_from_dataset(
        ds,
        x_name=x_name,
        y_name=y_name,
        crs=crs,
        mask_var=mask_var,
    )