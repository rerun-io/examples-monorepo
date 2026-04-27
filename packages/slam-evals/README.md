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
on `tools/ingest.py` if it lives elsewhere).

> The `pixi -q` flag below silences pixi's workspace-level warnings about
> `linux-64` target selectors on features (`wilor`, `gsplat-rust-renderer`)
> while the workspace is pinned to `linux-aarch64`. Drop `-q` if you want to
> see those warnings.

## End-to-end workflow

```bash
# 1. Walk the benchmark root + ingest every sequence into per-layer .rrd files
pixi -q run -e slam-evals --frozen slam-evals-ingest

# 2. Mount the catalog server (prints a viewer-connectable URL; blocks)
pixi -q run -e slam-evals --frozen slam-evals-catalog
```

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

The cold mount parses ~25 GB of RRD content (depth_0 alone is 20+ GB on the
local benchmark) and takes ~30-40 s. That cost is server-side and unavoidable;
**don't pay it on every iteration** — see *Dev loop* below.

## Viewer conventions (decoder ring)

Things you'll see in the 3D view that aren't obvious from the entity tree:

- **Green camera frustum at `/world/rig_0/cam_0`.** This is the rig's
  *reference sensor* — the primary camera the body pose is anchored to,
  and the one stereo / multi-cam baselines treat as the "left" / "main"
  view. Other cameras (`cam_1`, …) keep the viewer's default frustum
  colour. cam_0 is the reference by convention across every dataset
  spec; it's also implicit in the schema (the time-varying GT pose
  composes onto cam_0's static extrinsic). The convention is also
  surfaced as `property:calibration:cam0_name` for programmatic
  consumers.
- **GT trajectory polyline at `/world/rig_0_path`** — green→red
  gradient. Green is the *first frame*, red is the *final frame*.
  Endpoint marker dots labeled `start` / `end` reinforce the gradient
  direction. This is purely a viewer aid; the underlying GT timestamps
  are unchanged.
- **`property:groundtruth:has_rotation`** in `segment_table()` — `False`
  for sequences whose source GT is position-only (ROVER-* mostly).
  These segments have their rig translate along the path with identity
  rotation; expect static-orientation cameras since the dataset doesn't
  publish rotation GT. See `slam_evals.ingest.columns` for how zero /
  non-unit-norm quaternions get sanitised to identity.
- **First-frame snap-to-trajectory.** When GT starts later than the
  first RGB frame (most EUROC / HILTI / ARIEL sequences — Vicon
  warm-up), we prepend a synthetic Transform3D at `t=0` holding the
  first GT pose so cameras start at the right place instead of
  teleporting from the world origin once GT begins.

## Dev loop: edit → re-ingest → refresh (no restart)

Once the catalog is mounted, push edits into the running server with
`tools/catalog.py --refresh` instead of restarting it. The refresh path uses
`OnDuplicateSegmentLayer.REPLACE` over the gRPC catalog client, so only the
files you actually changed get parsed — typical refresh is **sub-second**, vs.
30-40 s for a full restart.

Two-terminal workflow:

```bash
# Terminal A — leave running
pixi -q run -e slam-evals --frozen slam-evals-catalog

# Terminal B — every time you change something
pixi -q run -e slam-evals --frozen slam-evals-ingest --layers <stem> --datasets <NAME> --force
pixi -q run -e slam-evals --frozen slam-evals-refresh --datasets <NAME> --layers <stem>

# In the viewer: reload the segment to see the change
```

`slam-evals-ingest` and `slam-evals-refresh` are pixi tasks that wrap the
underlying CLIs (so they work from any directory and honour `--datasets /
--layers / --only` directly as extra args). Common patterns:

```bash
# Re-emit + push only one layer for one segment
pixi -q run -e slam-evals --frozen slam-evals-ingest  --only EUROC/MH_01_easy --layers rgb_0 --force
pixi -q run -e slam-evals --frozen slam-evals-refresh --only EUROC/MH_01_easy --layers rgb_0

# Re-emit + push view_coordinates for every sequence (after adding a DatasetSpec)
pixi -q run -e slam-evals --frozen slam-evals-ingest  --layers view_coordinates --force
pixi -q run -e slam-evals --frozen slam-evals-refresh --layers view_coordinates
```

Available layer names: `calibration`, `groundtruth`, `view_coordinates`,
`rgb_0`, `rgb_1`, `depth_0`, `depth_1`, `imu_0` (only the ones applicable to a
sequence's modality are emitted).

Restart the catalog only when you change `tools/catalog.py` itself, the
blueprint, or want a clean slate. Routine spec/code changes go through
`--refresh`.

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

2. Re-emit the `view_coordinates` layer for that dataset, then push it
   into the running catalog:

   ```bash
   pixi -q run -e slam-evals --frozen slam-evals-ingest  --datasets MYDATASET --layers view_coordinates --force
   pixi -q run -e slam-evals --frozen slam-evals-refresh --datasets MYDATASET --layers view_coordinates
   ```

3. Reload the segment in the viewer, confirm the rig is upright. If it
   isn't, swap the convention in `datasets.py` and repeat — the whole
   loop is a few seconds.

## Optional: web viewer auto-popup

Pass `--open-browser` to `tools/catalog.py` to host a local web viewer at
`http://127.0.0.1:9090` and auto-open the system browser pointed at it (the
catalog auto-loads via the `connect_to` hint). Use this when you don't have a
desktop viewer running. Default is off — point your existing viewer at the
printed URL instead.

## Where things live

- `slam_evals/data/` — discovery, parsing, types
- `slam_evals/data/discovery.py` — `discover_sequences(benchmark_root)` walks
  the disk + classifies modality (called from `tools/ingest.py`; no separate CLI)
- `slam_evals/data/datasets.py` — per-dataset `DatasetSpec` registry
- `slam_evals/ingest/layer_*.py` — one writer per source stream
- `slam_evals/catalog/mount.py` — `mount_catalog` (in-process server,
  registers all layer files at startup) and `refresh_catalog` (connects to
  a running server via `CatalogClient` and pushes a subset with
  `OnDuplicateSegmentLayer.REPLACE`). Both share a per-layer-name batched
  registration helper that issues one `dataset.register(uris,
  layer_name=stem)` call per stem rather than per file — bulk path on the
  server side.
- `slam_evals/blueprint.py` — default 3D + per-camera 2D + IMU timeseries
  layout; registered at the catalog level so it applies to every segment
- `tools/{ingest,catalog}.py` — CLIs
- `docs/schema.md` — full schema description

## Re-ingest from scratch

Wipe and rebuild from source data:

```bash
mv ../../data/slam-evals/rrd ../../data/slam-evals/rrd.bak  # safety
mkdir -p ../../data/slam-evals/rrd
pixi -q run -e slam-evals --frozen slam-evals-ingest  # ~3-4 min on 4-worker NVENC pool
```
