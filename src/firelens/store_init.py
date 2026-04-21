# firelens/store_init.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil

import xarray as xr
import zarr


@dataclass(frozen=True)
class ChunkPolicy:
    time: int | None = None
    y: int = 512
    x: int = 512
    vintage: int = 1

    def __post_init__(self) -> None:
        if self.time is not None and self.time <= 0:
            raise ValueError(f"time chunk must be positive or None, got {self.time}")

        if self.y <= 0:
            raise ValueError(f"y chunk must be positive, got {self.y}")

        if self.x <= 0:
            raise ValueError(f"x chunk must be positive, got {self.x}")

        if self.vintage <= 0:
            raise ValueError(f"vintage chunk must be positive, got {self.vintage}")


def chunk_policy_attrs(policy: ChunkPolicy) -> dict[str, str]:
    return {
        "chunk_policy_json": json.dumps(asdict(policy)),
    }


def _require_xy(grid: xr.Dataset) -> None:
    if "x" not in grid.coords:
        raise KeyError("Grid dataset must contain x coordinate.")

    if "y" not in grid.coords:
        raise KeyError("Grid dataset must contain y coordinate.")


def _with_spatial_ref(ds: xr.Dataset, grid: xr.Dataset) -> xr.Dataset:
    """
    Attach spatial_ref from grid as a coordinate when available.
    """
    if "spatial_ref" in ds.coords:
        return ds

    if "spatial_ref" in grid.coords:
        return ds.assign_coords(spatial_ref=grid["spatial_ref"])

    if "spatial_ref" in grid:
        return ds.assign_coords(spatial_ref=grid["spatial_ref"])

    return ds


def _chunks_for_dims(
    dims: tuple[str, ...],
    sizes: Mapping[str, int],
    *,
    chunk_policy: ChunkPolicy,
    time_coord: str = "time",
) -> tuple[int, ...] | None:
    chunks: list[int] = []

    for dim in dims:
        if dim == time_coord:
            if chunk_policy.time is None:
                return None
            chunks.append(min(int(chunk_policy.time), int(sizes[dim])))

        elif dim == "y":
            chunks.append(min(int(chunk_policy.y), int(sizes[dim])))

        elif dim == "x":
            chunks.append(min(int(chunk_policy.x), int(sizes[dim])))

        elif dim == "vintage":
            chunks.append(min(int(chunk_policy.vintage), int(sizes[dim])))

        else:
            # Unknown dimension, for example x_edge/y_edge.
            # Let xarray/zarr choose default chunking.
            return None

    return tuple(chunks)


def encoding_from_chunk_policy(
    ds: xr.Dataset,
    *,
    chunk_policy: ChunkPolicy,
    time_coord: str = "time",
) -> dict:
    """
    Build Zarr chunk encoding for variables/coordinates whose dimensions
    are covered by ChunkPolicy.

    This intentionally skips scalar variables such as spatial_ref and leaves
    unknown dimensions unencoded.
    """
    encoding = {}

    for name in ds.variables:
        var = ds[name]

        if len(var.dims) == 0:
            continue

        chunks = _chunks_for_dims(
            tuple(var.dims),
            ds.sizes,
            chunk_policy=chunk_policy,
            time_coord=time_coord,
        )

        if chunks is not None:
            encoding[name] = {"chunks": chunks}

    return encoding


def make_grid_backbone_group(
    grid: xr.Dataset,
    *,
    chunk_policy: ChunkPolicy | None = None,
    group_role: str = "grid",
    description: str = "Canonical spatial grid backbone",
) -> xr.Dataset:
    """
    Create a grid-only group.

    This intentionally keeps only the spatial backbone and avoids copying
    derived static variables such as dem_m, slope_pct, tpi, etc.
    """
    _require_xy(grid)

    data_vars = {}
    coords = {}

    for name in ["x", "y", "x_edge", "y_edge"]:
        if name in grid.coords:
            coords[name] = grid.coords[name]

    if "domain_mask" in grid:
        data_vars["domain_mask"] = grid["domain_mask"]

    for name in ["lon", "lat", "spatial_ref"]:
        if name in grid.coords:
            coords[name] = grid.coords[name]
        elif name in grid:
            coords[name] = grid[name]

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            **grid.attrs,
            "group_role": group_role,
            "description": description,
        },
    )

    if chunk_policy is not None:
        ds.attrs.update(chunk_policy_attrs(chunk_policy))

    return ds


def make_static_group(
    grid: xr.Dataset,
    *,
    chunk_policy: ChunkPolicy | None = None,
) -> xr.Dataset:
    """
    Initialize the /static group.

    /static contains immutable static predictor variables.
    The canonical grid backbone lives in /grid.
    """
    _require_xy(grid)

    ds = xr.Dataset(
        coords={
            "y": grid["y"],
            "x": grid["x"],
        },
        attrs={
            "group_role": "static",
            "description": "Immutable static predictor layers on the canonical grid",
        },
    )

    ds = _with_spatial_ref(ds, grid)

    if chunk_policy is not None:
        ds.attrs.update(chunk_policy_attrs(chunk_policy))

    return ds


def make_slow_context_group(
    grid: xr.Dataset,
    *,
    chunk_policy: ChunkPolicy | None = None,
) -> xr.Dataset:
    _require_xy(grid)

    ds = xr.Dataset(
        coords={
            "y": grid["y"],
            "x": grid["x"],
        },
        attrs={
            "group_role": "slow_context",
            "description": "Low-frequency contextual layers such as versioned LANDFIRE products",
            "time_like_dimension_policy": "Use product-specific dimensions such as vintage or year, not dense time.",
        },
    )

    ds = _with_spatial_ref(ds, grid)

    if chunk_policy is not None:
        ds.attrs.update(chunk_policy_attrs(chunk_policy))

    return ds


def make_dynamic_group(
    grid: xr.Dataset,
    time_ds: xr.Dataset,
    *,
    chunk_policy: ChunkPolicy | None = None,
) -> xr.Dataset:
    _require_xy(grid)

    if "time" not in time_ds.coords:
        raise KeyError("time_ds must contain a 'time' coordinate")

    ds = xr.Dataset(
        coords={
            "time": time_ds["time"],
            "y": grid["y"],
            "x": grid["x"],
        },
        attrs={
            "group_role": "dynamic",
            "time_frequency": time_ds.attrs.get("frequency", ""),
            "time_semantics": time_ds.attrs.get("time_semantics", ""),
        },
    )

    ds = _with_spatial_ref(ds, grid)

    if chunk_policy is not None:
        ds.attrs.update(chunk_policy_attrs(chunk_policy))

    return ds


def make_labels_group(
    grid: xr.Dataset,
    time_ds: xr.Dataset,
    *,
    chunk_policy: ChunkPolicy | None = None,
) -> xr.Dataset:
    _require_xy(grid)

    if "time" not in time_ds.coords:
        raise KeyError("time_ds must contain a 'time' coordinate")

    ds = xr.Dataset(
        coords={
            "time": time_ds["time"],
            "y": grid["y"],
            "x": grid["x"],
        },
        attrs={
            "group_role": "labels",
            "description": "Prediction targets and label masks",
            "time_frequency": time_ds.attrs.get("frequency", ""),
            "time_semantics": time_ds.attrs.get("time_semantics", ""),
        },
    )

    ds = _with_spatial_ref(ds, grid)

    if chunk_policy is not None:
        ds.attrs.update(chunk_policy_attrs(chunk_policy))

    return ds


def make_spatiotemporal_group(
    *,
    grid: xr.Dataset,
    time_ds: xr.Dataset,
    group_role: str,
    description: str = "",
    time_coord: str = "time",
    chunk_policy: ChunkPolicy | None = None,
) -> xr.Dataset:
    _require_xy(grid)

    if time_coord not in time_ds.coords:
        raise KeyError(f"time_ds must contain {time_coord!r} coordinate")

    ds = xr.Dataset(
        coords={
            time_coord: time_ds[time_coord],
            "y": grid["y"],
            "x": grid["x"],
        },
        attrs={
            "group_role": group_role,
            "description": description,
            "time_frequency": time_ds.attrs.get("frequency", ""),
            "time_semantics": time_ds.attrs.get("time_semantics", ""),
            "grid_source": grid.attrs.get("title", ""),
        },
    )

    ds = _with_spatial_ref(ds, grid)

    if chunk_policy is not None:
        ds.attrs.update(chunk_policy_attrs(chunk_policy))

    return ds


def _write_group(
    *,
    store: str | Path,
    group: str,
    ds: xr.Dataset,
    mode: str,
    zarr_format: int,
    chunk_policy: ChunkPolicy,
    time_coord: str = "time",
) -> None:
    ds.to_zarr(
        store,
        group=group,
        mode=mode,
        zarr_format=zarr_format,
        consolidated=False,
        encoding=encoding_from_chunk_policy(
            ds,
            chunk_policy=chunk_policy,
            time_coord=time_coord,
        ),
    )

def init_context_store(
    *,
    context_zarr: str | Path,
    grid: xr.Dataset,
    overwrite: bool = False,
    zarr_format: int = 2,
    chunk_policy: ChunkPolicy = ChunkPolicy(time=None, y=512, x=512, vintage=1),
) -> None:
    """
    Initialize the FireLens context Zarr store.

    Creates:
    - /grid
    - /static
    - /slow_context

    This store intentionally has no dense time axis.
    """
    context_zarr = Path(context_zarr)

    if context_zarr.exists():
        if overwrite:
            shutil.rmtree(context_zarr)
        else:
            raise FileExistsError(f"{context_zarr} already exists")

    context_zarr.parent.mkdir(parents=True, exist_ok=True)

    grid_ds = make_grid_backbone_group(
        grid,
        chunk_policy=chunk_policy,
        group_role="grid",
        description="Canonical California 500 m EPSG:5070 spatial grid",
    )

    static_ds = make_static_group(
        grid,
        chunk_policy=chunk_policy,
    )

    slow_ds = make_slow_context_group(
        grid,
        chunk_policy=chunk_policy,
    )

    _write_group(
        store=context_zarr,
        group="grid",
        ds=grid_ds,
        mode="w",
        zarr_format=zarr_format,
        chunk_policy=chunk_policy,
    )

    _write_group(
        store=context_zarr,
        group="static",
        ds=static_ds,
        mode="a",
        zarr_format=zarr_format,
        chunk_policy=chunk_policy,
    )

    _write_group(
        store=context_zarr,
        group="slow_context",
        ds=slow_ds,
        mode="a",
        zarr_format=zarr_format,
        chunk_policy=chunk_policy,
    )

    zarr.consolidate_metadata(str(context_zarr))


def init_spatiotemporal_store(
    *,
    store_zarr: str | Path,
    grid: xr.Dataset,
    time_ds: xr.Dataset,
    groups: Mapping[str, str],
    chunk_policy: ChunkPolicy,
    overwrite: bool = False,
    zarr_format: int = 2,
    include_grid_group: bool = True,
) -> None:
    """
    Initialize a generic FireLens spatiotemporal Zarr store.

    Example groups:
      {"aorc": "Hourly AORC on CA 500 m grid",
       "hrrr": "Hourly HRRR on CA 500 m grid"}

    or:
      {"goes": "15-minute GOES on CA 500 m grid",
       "viirs": "15-minute VIIRS rasterized fire detections"}
    """
    if not groups and not include_grid_group:
        raise ValueError("Nothing to initialize: groups is empty and include_grid_group=False")

    store_zarr = Path(store_zarr)

    if store_zarr.exists():
        if overwrite:
            shutil.rmtree(store_zarr)
        else:
            raise FileExistsError(f"{store_zarr} already exists")

    store_zarr.parent.mkdir(parents=True, exist_ok=True)

    wrote_any = False

    if include_grid_group:
        grid_ds = make_grid_backbone_group(
            grid,
            chunk_policy=chunk_policy,
            group_role="grid",
            description="Canonical spatial grid copied from FireLens context store",
        )

        _write_group(
            store=store_zarr,
            group="grid",
            ds=grid_ds,
            mode="w",
            zarr_format=zarr_format,
            chunk_policy=chunk_policy,
        )

        wrote_any = True

    for group_name, description in groups.items():
        ds = make_spatiotemporal_group(
            grid=grid,
            time_ds=time_ds,
            group_role=group_name,
            description=description,
            chunk_policy=chunk_policy,
        )

        _write_group(
            store=store_zarr,
            group=group_name,
            ds=ds,
            mode="a" if wrote_any else "w",
            zarr_format=zarr_format,
            chunk_policy=chunk_policy,
        )

        wrote_any = True

    zarr.consolidate_metadata(str(store_zarr))