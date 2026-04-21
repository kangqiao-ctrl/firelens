# FireLens Datacube Build Workflow

FireLens stores California wildfire modeling inputs as grouped Zarr stores on a common California 500 m EPSG:5070 grid.

The current workflow uses three stores:

```text
context_500m.zarr
  /grid
  /static
  /slow_context

met_500m_1h.zarr
  /grid
  /aorc
  /hrrr

satfire_500m_15min.zarr
  /grid
  /goes
  /viirs


Context zarr building:
python3 -m firelens.pipelines.init_california_500m_context_zarr   --context-zarr /path/to/zarr/context_500m.zarr
python3 -m firelens.pipelines.ingest_static_group --cube-zarr /scratch/groups/dsivas/qiaok/firelens_zarr/context_500m.zarr 
python3 -m firelens.pipelines.ingest_landfire --cube-zarr /scratch/groups/dsivas/qiaok/firelens_zarr/context_500m.zarr

python3 -m firelens.pipelines.init_met_500m_1h_zarr
python3 -m firelens.pipelines.ingest_aorc --met-zarr /scratch/groups/dsivas/qiaok/firelens_zarr/met_500m_1h.zarr --aorc-root /scratch/users/qiaok/data/aorc/zarr --years 2017 2018 2019 2020 2021 2022 2023 2024 2025 --weights-nc /home/groups/dsivas/qiaok/firelens/artifacts/aorc_to_ca500m_bilinear.nc
python3 -m firelens.pipelines.ingest_hrrr --met-zarr /scratch/groups/dsivas/qiaok/firelens_zarr/met_500m_1h.zarr --hrrr-root /scratch/users/qiaok/data/hrrr --years 2017 --variables gust_surface blh --time-block-hours 24

python3 -m firelens.pipelines.init_satfire_500m_15min_zarr