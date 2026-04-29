# slam-evals architecture

End-to-end view of how a directory of VSLAM-LAB sequences becomes a Rerun catalog you can browse in the viewer. Read [`schema.md`](schema.md) first if you haven't — this doc focuses on *how* data flows; the schema doc explains *what* lives at which entity path and why.

---

## 1. The whole pipeline

```
                                            offline ingest                                              live serving
       ┌────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐    ┌──────────────────────┐
       │ VSLAM-LAB-Benchmark │   │ slam_evals.data     │    │ slam_evals.ingest   │    │ slam_evals.catalog   │
       │ (~25 GB on disk)   │   │  discovery + parse  │    │  per-layer writers  │    │  mount + refresh     │
       │                    │ → │  (rgb.csv, gt.csv,  │ →  │  (1 .rrd per layer  │ →  │  (rr.server.Server +  │
       │  EUROC, KITTI, …   │   │   imu.csv, calib.   │    │   per sequence)     │    │   gRPC catalog)      │
       │  109 seqs, 18 sets │   │   yaml, ts only)    │    │                     │    │                      │
       └────────────────────┘    └─────────────────────┘    └─────────────────────┘    └──────────────────────┘
                                                                       │                          │
                                                                       ▼                          ▼
                                                        data/slam-evals/rrd/         rerun+http://127.0.0.1:9987
                                                          <dataset>/<seq>/                 (or refresh into a
                                                          <layer>.rrd                       running catalog)
                                                          (~510 files, 24 GB)                       │
                                                                                                    ▼
                                                                                         Rerun viewer
                                                                                         (109 segments, one
                                                                                         per source sequence)
```

Two distinct CLIs sit on top of these stages:

| CLI | Stage it drives | What it does |
|---|---|---|
| `tools/ingest.py` (`pixi run slam-evals-ingest`) | **discovery → parse → write** | Walks the benchmark, parses CSV/YAML, writes `.rrd` layer files |
| `tools/catalog.py` (`pixi run slam-evals-catalog`) | **serve** | Starts an in-process gRPC catalog server, registers every `.rrd` under its layer name, blocks until Ctrl-C |
| `tools/catalog.py --refresh` (`pixi run slam-evals-refresh`) | **partial re-serve** | Connects to a *running* catalog and pushes only changed files via `OnDuplicateSegmentLayer.REPLACE` |

The split lets you re-emit one layer for one dataset (~5s) and push the result into the live catalog (~1s) without paying the ~35s cold-mount cost.

---

## 2. Stage 1 — Discovery (`slam_evals.data.discovery`)

### Input

The benchmark root: a directory tree with this shape under each `<dataset>/<sequence>/`:

```
EUROC/MH_01_easy/
├── rgb_0/*.png             (required)
├── rgb.csv                 (required — timestamps + paths for rgb_0/1, depth_0/1)
├── groundtruth.csv         (required — ts, tx,ty,tz, qx,qy,qz,qw)
├── calibration.yaml        (optional — intrinsics + extrinsics + IMU noise)
├── rgb_1/*.png             (optional — stereo)
├── depth_0/*.png           (optional — rgbd, uint16 PNG)
├── depth_1/*.png           (optional — stereo-rgbd, e.g. OPENLORIS-D400)
└── imu_0.csv               (optional — *-vi modalities)
```

### Output

`list[Sequence]`. `Sequence` is a frozen dataclass carrying the on-disk root, the dataset name, the sequence name, and a derived `Modality` enum (`mono`, `mono-vi`, `stereo`, `stereo-vi`, `rgbd`, `rgbd-vi`).

### Key code

- `discover_sequences(benchmark_root: Path) -> list[Sequence]` — walks two levels deep, applies the required-files triad, classifies modality from optional file presence.
- `_classify(...)` — defensive: catches `ValueError`/`KeyError`/`OSError`/`SerdeError`/`csv.Error` from a malformed `rgb.csv` and skips the sequence rather than aborting the walk.
- `Modality` derives `has_imu` / `has_stereo` / `has_depth` properties from the enum value (string-suffix checks).

`Sequence.dataset_spec` looks up `slam_evals.data.datasets.lookup(name)` to fetch the per-dataset world-frame convention. Returns `None` for unclassified datasets — the `view_coordinates` layer is skipped in that case.

---

## 3. Stage 2 — Parse (`slam_evals.data.parse`)

Pyserde-driven CSV + YAML parsing. Source files map 1:1 to typed dataclasses, then row-stack into numpy aggregates.

| Source | Per-row schema | Aggregate |
|---|---|---|
| `rgb.csv` | `_RgbRow` (ts/path × {rgb_0, rgb_1, depth_0, depth_1}) | `RgbCsv` (numpy int64 timestamps, tuples of paths) |
| `groundtruth.csv` | `_GtRow` (ts, tx, ty, tz, qx, qy, qz, qw) | `GroundTruth` (numpy `(n,3)` translation, `(n,4)` xyzw quat) |
| `imu_0.csv` | `_ImuRow` (ts, wx/wy/wz, ax/ay/az) | `ImuSamples` (numpy `(n,3)` gyro/accel) |
| `calibration.yaml` | `_CameraYaml`, `_ImuYaml`, `_CalibrationYaml` | `Calibration` |

### Calibration is the only stage where simplecv types are constructed

`_to_camera_spec` reads the YAML row and emits a `CameraSpec`:

```python
CameraSpec(
    parameters: PinholeParameters | Fisheye62Parameters,  # from simplecv
    fps: float | None,                                     # VSLAM-LAB-only
    depth_factor: float | None,                            # VSLAM-LAB-only
)
```

`PinholeParameters` carries `Intrinsics` (focal/principal + image size + RDF/RUB convention), `Extrinsics` (rig→cam transform stored as `world_R_cam` / `world_t_cam`, where "world" = our rig frame), and the `BrownConradyDistortion` for radtan4/5. Fisheye datasets (Kannala-Brandt) get a `Fisheye62Parameters` with `KannalaBrandtDistortion`.

The YAML field name `T_BS` only appears at this parse boundary — internally we use `rig_T_cam` / `rig_T_imu`.

### IMU parameters

Stored as an opaque `dict[str, float | list[float]]` on `Calibration.imu_params` rather than a typed dataclass, because the noise terms (`sigma_g_c`, `sigma_a_c`, `a_max`, …) vary between datasets and we want them to flow through as-is into the `imu_0` property bag.

---

## 4. Stage 3 — Per-layer write (`slam_evals.ingest`)

The orchestrator (`ingest_sequence` in `sequence.py`) takes one `Sequence` and writes 3-7 `.rrd` files into `<out_dir>/<dataset>/<seq>/`. Each file is one **layer** — the smallest unit of selective re-emit.

### Per-sequence state machine

```
Sequence
   │
   ▼
applicable_layers(sequence)               ← returns subset of:
   │                                        {calibration, groundtruth, view_coordinates,
   │                                         video_0, video_1, depth_0, depth_1, imu_0}
   ▼
intersect with `--layers` filter (if any) ← caller-driven subset
   │
   ▼
parse minimum needed                      ← rgb.csv (always; carries t0_ns)
   │                                        calibration.yaml if has_calibration
   │                                        groundtruth.csv lazily (only if GT layer selected)
   │                                        imu_0.csv      lazily
   ▼
for each selected layer in fixed order:
    open RecordingStream(
        application_id="slam-evals",
        recording_id=f"{dataset}__{name}",  ← shared across all layers for the same segment
        send_properties=True,
    )
    write entities + send_property
    rec.save(<seq_dir>/<layer>.rrd)
   │
   ▼
return tuple of written paths
```

### Layer writers (one per source stream)

| Layer file | Module | What it writes |
|---|---|---|
| `calibration.rrd` | `layer_calibration.py` | static `Transform3D` + `Pinhole` per cam at `/world/rig_0/cam_<i>`, IMU `Transform3D` if present, plus segment-level `info` and `calibration` property bags. cam_0 frustum gets a green wireframe override. |
| `groundtruth.rrd` | `layer_groundtruth.py` | time-varying `world_T_rig_0` `Transform3D` at `/world/rig_0`; static `LineStrips3D` GT path with green→red gradient at `/world/rig_0_path`; static start/end `Points3D` markers at `/world/rig_0_path/endpoints`. Property bag: `num_poses`, `trajectory_len_m`, `duration_s`, `has_rotation`. |
| `view_coordinates.rrd` | `layer_view_coordinates.py` | static `ViewCoordinates` at `/world` from the dataset's `DatasetSpec` (FLU, RDF, RUB, …). Skipped for unclassified datasets. |
| `video_<i>.rrd` | `layer_video.py` (+ `video.py`) | HEVC `VideoStream` at `/world/rig_0/cam_<i>/pinhole/video`. Threaded PNG decode + NVENC encoder. NVENC session-contention failures are detected by error string and retried serially after the parallel pool drains. |
| `depth_<i>.rrd` | `layer_depth.py` | per-frame `EncodedDepthImage` (PNG passthrough, no re-encode) at `/world/rig_0/cam_<i>/pinhole/depth`. `meter` parameter passed through directly from VSLAM-LAB's `depth_factor` (both are native-units-per-metre). |
| `imu_<i>.rrd` | `layer_imu.py` (+ `columns.py`) | 3-component `Scalars` for gyro and accel at `/world/rig_0/imu_<i>/{gyro,accel}`. Property bag carries the YAML's noise terms verbatim. |

### Defensive normalisation in the GT writer

Two issues are handled inside `log_groundtruth_columns`:

1. **Degenerate quaternions** (e.g. ROVER's all-zero quat from position-only GT): rows with non-unit-norm quaternions get sanitised to identity. Translation still applies; rotation stays fixed at identity. `has_rotation` flag flips to `False`.
2. **Late-starting GT** (e.g. EUROC's Vicon warm-up where the first GT row is at `t=+1.08s`): a synthetic row at `t=0` is prepended holding the first GT pose so `latest_at` lookups find a valid pose for the entire visible timeline, instead of the rig teleporting from the world origin once GT starts.

### Concurrency

`tools/ingest.py` runs sequences through a `ProcessPoolExecutor(max_workers=4)`. NVENC saturates the GPU at ~4 workers; over-subscribing rejects with `avcodec_open2(hevc_nvenc) ExternalError`. Failures matching that signature are buffered and retried serially after the parallel pass.

---

## 5. Stage 4 — Catalog mount (`slam_evals.catalog.mount`)

### Everything funnels into one dataset

```python
server = rr.server.Server(datasets={"vslam": []}, port=9987)
dataset = server.client().get_dataset("vslam")
```

A single catalog dataset called `vslam` holds all 109 segments. The source benchmark name (EUROC, KITTI, …) is a queryable property column (`property:info:dataset`), not a separate catalog dataset — see the rationale in [`schema.md`](schema.md).

### Layer-grouped registration

Per advice from rerun engineers, the server has a fast path for `register(uris, layer_name=stem)` when many URIs share one layer name:

```python
layers_by_stem: dict[str, list[Path]] = group rrd files by filename stem
for stem, files in layers_by_stem.items():        # 8 groups, not 510 files
    uris = [f.resolve().as_uri() for f in files]
    dataset.register(uris, layer_name=stem).wait()
```

8 `register` calls cover the whole 510-file corpus instead of 510 individual round-trips. Cold mount takes ~35 s — dominated by server-side `.rrd` parsing (especially the ~21 GB depth_0 layer), not client roundtrips.

### Recording-id collapse

All layer files for one sequence share the same `recording_id = "<DATASET>__<seq>"`. The catalog server stitches them into one **segment** at query/view time:

```
EUROC/MH_01_easy/calibration.rrd        ─┐
EUROC/MH_01_easy/groundtruth.rrd         │
EUROC/MH_01_easy/view_coordinates.rrd    │   recording_id =
EUROC/MH_01_easy/video_0.rrd             ├─  "EUROC__MH_01_easy"   ───►  ONE segment
EUROC/MH_01_easy/video_1.rrd             │   (8 layers merged
EUROC/MH_01_easy/imu_0.rrd              ─┘    by the catalog)
```

This is what makes selective re-emit cheap: editing one dataset spec → re-emit `view_coordinates` for that dataset → 8 small files change → push into running catalog → seconds.

### Blueprint

`slam_evals.blueprint.build_blueprint()` returns an `rrb.Blueprint` describing the default 3D + per-camera 2D + IMU timeseries layout. Saved to a `TemporaryDirectory` (lifetime tied to the server via `weakref.finalize` + `atexit`) and registered at the dataset level so every segment picks it up automatically.

---

## 6. Stage 5 — Refresh (the dev loop)

```
                   long-running tools/catalog.py (port 9987)
                           ▲                       ▲
                           │ initial mount         │ register(REPLACE)
                           │ (35 s, once)          │ (sub-second)
                           │                       │
                  ┌────────┴────────┐    ┌─────────┴─────────────────┐
                  │ slam-evals-     │    │ slam-evals-refresh        │
                  │ catalog         │    │  --datasets X --layers Y  │
                  │ (terminal A)    │    │ (terminal B, after edit)  │
                  └─────────────────┘    └───────────────────────────┘
                                                    ▲
                                                    │
                                         slam-evals-ingest
                                          --layers Y --datasets X --force
                                         (re-emits the changed layer files)
```

`refresh_catalog` is a thin client-only path: opens a `CatalogClient(catalog_url)` against the running server, glob-filters layer files using the same `--datasets` / `--layers` / `--only` predicates as ingest, and re-registers with `OnDuplicateSegmentLayer.REPLACE`. Existing segment+layer entries are swapped in place; new entries are inserted. Sub-second turnaround on small change sets.

The mount and refresh paths share `_register_layer_groups` — same layer-grouped batching, just a different starting context (in-process server vs. remote client) and a different `OnDuplicateSegmentLayer` policy (ERROR for mount → fail loud on collision; REPLACE for refresh → upsert).

---

## 7. Stage 6 — Query (`slam_evals.catalog.query`)

`segment_summary(server)` materialises a `pandas.DataFrame` with one row per segment and one column per known property (`property:info:dataset`, `property:rgb_0:codec`, `property:groundtruth:trajectory_len_m`, …). Used by tests and ad-hoc scripts.

This module exists to wrap a known SDK 0.31 FFI lifetime issue: calling `select(*cols).to_pandas()` *inline* in a caller raises `TaskContextProvider went out of scope`, but wrapping the same code in `segment_summary` works (the function call frame keeps the right Rust-side context alive long enough). Treat it as load-bearing infrastructure, not a stylistic helper — the docstring explains the empirical evidence.

When the SDK fixes the underlying lifetime bug, this module can shrink to a one-liner or be deleted.

---

## 8. Tests

`packages/slam-evals/tests/`:

- `test_import.py` — package imports cleanly (smoke).
- `test_datasets.py` — `DatasetSpec` registry invariants (every entry has a unique name, lookup returns the right object, etc.).
- `test_ingest_modalities.py` — round-trip tests parametrised over every modality. For each: build a synthetic fixture (`slam_evals.data.synthetic.build_fixture`), run `ingest_sequence`, mount the result, verify the segment_table has the right per-layer property bags. Also tests selective re-emit (only the requested layer's mtime changes) and degenerate cases (empty groundtruth.csv).

Tests run in `slam-evals-dev` env (`pixi run -e slam-evals-dev pytest -q`). `slam-evals-dev` adds beartype, ruff, pyrefly, pytest on top of the prod env.

---

## 9. Code map

```
packages/slam-evals/
├── pyproject.toml              # standard packaging metadata
├── README.md                   # user-facing workflow docs
├── docs/
│   ├── schema.md               # what lives at which entity path, layer model, transform notation
│   └── architecture.md         # this file
├── slam_evals/
│   ├── __init__.py             # PIXI_DEV_MODE beartype activation
│   ├── data/
│   │   ├── datasets.py         # DatasetSpec registry (one entry per on-disk dataset)
│   │   ├── discovery.py        # walk + classify modality
│   │   ├── parse.py            # pyserde CSV/YAML parsers + simplecv adapter
│   │   ├── synthetic.py        # fixture builder for tests + smoke
│   │   └── types.py            # Sequence, Modality, Calibration, CameraSpec
│   ├── ingest/
│   │   ├── sequence.py         # ingest_sequence (orchestrator) + LayerName Literal
│   │   ├── columns.py          # send_columns helpers for GT + IMU
│   │   ├── calibration.py      # log_camera_static — thin wrapper over simplecv.log_pinhole
│   │   ├── video.py            # NVENC HEVC encode + threaded PNG decode
│   │   └── layer_*.py          # one writer per layer name (calibration, groundtruth,
│   │                             view_coordinates, video, depth, imu)
│   ├── catalog/
│   │   ├── mount.py            # mount_catalog (server) + refresh_catalog (client)
│   │   └── query.py            # segment_summary (FFI-safe materialisation)
│   └── blueprint.py            # default 3D + 2D + timeseries view layout
├── tools/
│   ├── ingest.py               # discover + write CLI (tyro)
│   └── catalog.py              # mount or refresh CLI (tyro)
└── tests/
    ├── conftest.py             # fixture_factory + segment_table_df helper
    ├── test_import.py
    ├── test_datasets.py
    └── test_ingest_modalities.py
```

---

## 10. Mental model in one sentence

> **Each sequence becomes one catalog segment composed of 3-8 small `.rrd` layer files, all sharing the same `recording_id`. The mount step registers each file by layer name; the catalog server collapses files-with-the-same-recording-id into one segment automatically. Selective re-emit + refresh is the operational story: you change one layer for one dataset, push it in, see it.**

Everything else — modality classification, simplecv types, NVENC encoding, GT sanitisation, the dev-loop refresh — is detail in service of that core composition pattern.
