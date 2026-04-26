# slam-evals

Ingest VSLAM-LAB benchmark sequences into a Rerun catalog and browse them in
the viewer. Each sequence becomes one segment composed of multiple `.rrd`
layer files (one per source data stream); the catalog stitches them together
at view time. See [`docs/schema.md`](docs/schema.md) for the schema, COLMAP-
aligned entity tree, and design rationale.

## Setup (one-time)

```bash
# From repo root. Solves the env + writes lockfile entries.
pixi install -e slam-evals
```

`VSLAM_LAB_BENCHMARK` is pinned in the env to
`/home/pablo/0Dev/work/VSLAM-LAB-Benchmark` (override with `--benchmark-root`
on `discover.py` if it lives elsewhere).

## End-to-end workflow

```bash
# 1. Discover + ingest in one shot. The ingest task depends-on discover, so
#    pixi runs the manifest scan first automatically.
pixi run -e slam-evals --frozen slam-evals-ingest

# 2. Mount the catalog server (prints a viewer-connectable URL; blocks)
pixi run -e slam-evals --frozen slam-evals-catalog
```

(If you want to refresh just the manifest without re-ingesting — e.g. to
spot-check what discovery picked up — `pixi run … slam-evals-discover`
runs the scan in isolation.)

Step 2 prints something like:

```
Catalog URL:  rerun+http://127.0.0.1:9987

In the Rerun viewer:  + → Open Data Source → paste the URL
Or from a terminal:   rerun rerun+http://127.0.0.1:9987
```

In your already-running Rerun viewer: click **+** next to **Sources** → **Open
Data Source** → paste the catalog URL. The 109-segment catalog appears in the
sources panel; click any segment to load it (the slam-evals blueprint is
applied automatically).

Stop the catalog with `Ctrl-C`.

## Selective re-ingestion

Each layer file is independent. To re-emit only one stream after a change
(e.g. a new dataset entry, an NVENC flake, a tweaked codec):

```bash
# Re-emit only the rgb_0 layer for one sequence
pixi run -e slam-evals --frozen python tools/ingest.py \
    --manifest ../../data/slam-evals/manifest.json \
    --out ../../data/slam-evals/rrd \
    --only EUROC/MH_01_easy --layers rgb_0 --force

# Re-emit view_coordinates for every sequence (after adding a new DatasetSpec)
pixi run -e slam-evals --frozen python tools/ingest.py \
    --manifest ../../data/slam-evals/manifest.json \
    --out ../../data/slam-evals/rrd \
    --layers view_coordinates --force
```

Available layer names: `calibration`, `groundtruth`, `view_coordinates`,
`rgb_0`, `rgb_1`, `depth_0`, `depth_1`, `imu_0` (only the ones applicable to a
sequence's modality are emitted). The catalog server caches registered URIs
at startup, so any re-ingest while the server is running requires a server
restart to take effect.

## Adding a new dataset's world-frame convention

Different datasets express their GT trajectory in different world frames
(EUROC: Z-up gravity-aligned; KITTI: cam_0 frame; REPLICA: OpenGL Y-up; …).
Without a `DatasetSpec` the viewer shows the rig at whatever default it
picks, which often looks sideways. To fix:

1. Add an entry to `slam_evals/data/datasets.py`:

   ```python
   MYDATASET = DatasetSpec(name="MYDATASET", world_view_coordinates=rr.ViewCoordinates.FLU)
   ```

   then add it to the `DATASETS` tuple. The name must match the on-disk
   directory name exactly. Pick the right convention from the
   [Rerun ViewCoordinates docs](https://rerun.io/docs/reference/types/archetypes/view_coordinates).

2. Re-emit the `view_coordinates` layer:

   ```bash
   pixi run -e slam-evals --frozen python tools/ingest.py \
       --manifest ../../data/slam-evals/manifest.json \
       --out ../../data/slam-evals/rrd \
       --only MYDATASET --layers view_coordinates --force
   ```

3. Restart the catalog server, click into a `MYDATASET` segment in the
   viewer, confirm the rig is upright.

## Optional: web viewer auto-popup

Pass `--open-browser` to `tools/catalog.py` to host a local web viewer at
`http://127.0.0.1:9090` and auto-open the system browser pointed at it (the
catalog auto-loads via the `connect_to` hint). Use this when you don't have a
desktop viewer running. Default is off — point your existing viewer at the
printed URL instead.

## Filtering

`tools/catalog.py --filter-modality stereo-vi` or `--filter-dataset EUROC`
restrict the printed segment summary; the catalog itself still serves all
segments (the filters are pandas-side, after `segment_table()`).

## Where things live

- `slam_evals/data/` — discovery, parsing, types
- `slam_evals/data/datasets.py` — per-dataset `DatasetSpec` registry
- `slam_evals/ingest/layer_*.py` — one writer per source stream
- `slam_evals/catalog/mount.py` — `rr.server.Server` wrapper that registers
  every layer file with `dataset.register([uri], layer_name=stem)`
- `slam_evals/blueprint.py` — default 3D + per-camera 2D + IMU timeseries
  layout; registered at the catalog level so it applies to every segment
- `tools/{discover,ingest,catalog}.py` — CLIs
- `docs/schema.md` — full schema description

## Re-ingest from scratch

Wipe and rebuild from source data:

```bash
mv ../../data/slam-evals/rrd ../../data/slam-evals/rrd.bak  # safety
mkdir -p ../../data/slam-evals/rrd
pixi run -e slam-evals --frozen slam-evals-ingest  # ~3-4 min on 4-worker NVENC pool
```
