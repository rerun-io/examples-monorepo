# arkitscenes-download

Download [ARKitScenes](https://github.com/apple/ARKitScenes) (Apple, 5,071 indoor
iPhone/iPad captures) and ingest it into **layered Rerun recordings**: full-rate
60 fps video, true per-frame camera poses, raw IMU, ARKit depth,
lens distortion, ARKit mesh, and 3D ground-truth boxes — most of which never appears in the
dataset's published PNG/traj assets because it lives inside the `.mov` files.
The catalog also recognizes optional CA-1M laser-GT pose and depth layers when a
separate CA-1M tool has produced them.

## Quickstart

```bash
# 1. Download a 5-sequence sample (~7 GB: mov + depth + calibration, no point clouds)
pixi run -e arkitscenes-download arkitscenes-download-sample

# 2. Ingest everything downloaded into layered .rrd files under data/rrd/
pixi run -e arkitscenes-download arkitscenes-download-ingest

# 3. Serve a Rerun catalog, register the recordings, open the viewer
pixi run -e arkitscenes-download arkitscenes-download-serve      # terminal 1
pixi run -e arkitscenes-download arkitscenes-download-register   # terminal 2
pixi run -e arkitscenes-download arkitscenes-download-view
```

Ingestion wants an NVIDIA GPU (`av1_nvenc` transcodes the video track); without
one it falls back to CPU SVT-AV1, just slower.

### Registering multiple datasets

The OSS Rerun server keeps one file descriptor open for every registered `.rrd`
so it can load chunks on demand. A 5,015-segment ARKitScenes dataset with eight
required layers therefore uses 40,120 descriptors; two such datasets need 80,240.
With a 65,536 descriptor limit, the second registration fails during a later layer
because the combined file count exceeds the limit—not because that layer is faulty.

The `arkitscenes-download-serve` task raises the server limit to 524,288. Restart
an already-running server with that task before registering multiple datasets;
the limit is inherited when the server starts. This is a capacity workaround
for the current Rerun server, which does not yet cap open files with an
open-on-demand or LRU policy.

## What one sequence becomes

Eight small required `.rrd` files sharing one `recording_id`, so the catalog stacks them
into a single recording — and any one aspect can be regenerated and re-registered
without touching the rest (no re-transcode to fix box math):

| layer | contents |
|---|---|
| `base` | recording properties (clock offsets, orientation, pose provenance, …) — queryable segment-table columns |
| `calibration` | 60 Hz `world_T_rig`, per-camera pinholes, **lens distortion** (8-coefficient polynomials via simplecv components), stereo extrinsics |
| `video_wide` / `video_ultrawide` | AV1 `VideoStream` at native resolution and framerate |
| `arkit_depth` | low-resolution ARKit depth + confidence |
| `imu` | 100 Hz accelerometer / gyroscope / fused attitude |
| `arkit_mesh` | reconstructed ARKit mesh |
| `gt_boxes` | oriented 3D ground-truth boxes |

Covered captures may also carry `gt_poses` and `gt_depth`, both produced by the
separate CA-1M tool. Neither is required for ingest completeness; absence means no
laser GT is available for that capture.

## Laser ground truth (CA-1M) and provenance

The layer names state *what* the data is; recording properties state *where it
came from*. What is actually ground truth:

| layer | provenance | ground truth? |
|---|---|---|
| `arkit_depth` | on-device ARKit depth + confidence | no — device estimate |
| `arkit_mesh` | reconstructed from device depth | no — device estimate |
| `gt_boxes` | human-annotated 3DOD boxes (on the ARKit mesh) | labels yes, geometry approximate |
| `gt_poses` | per-frame camera poses registered to the FARO laser scan | **yes** |
| `gt_depth` | 512×384 depth rendered from the FARO scan, per-frame `K` | **yes** |

`gt_poses`/`gt_depth` currently come from Apple's
[CA-1M release](https://github.com/apple/ml-cubifyanything) via:

```bash
# --output is required (the layer-major dataset root the new dirs land beside)
pixi run -e arkitscenes-download arkitscenes-download-ca1m --output /path/to/dataset-root
```

Facts that shape consumption:

- **Coverage is partial by construction.** CA-1M covers ~61% of the original
  captures (only those whose laser registration succeeded), and within a capture
  the GT frames can start late, end early, or have interior holes (e.g.
  `42898570` has no GT for its first 16.4 s). Per-capture coverage and quality
  ship as typed `property:gt:*` segment-table columns (`start_s`, `end_s`,
  `max_interior_gap_s`, `umeyama_rms_m`, `provenance`). A capture without the
  `gt_*` layers has no laser GT — that absence is the intended marker.
- **GT is ~10 Hz** (the hi-res frame grid) on the shared `video_time` timeline;
  the 60 Hz device streams simply coexist with it.
- **Frames are already upright** in CA-1M; the tool applies no rotation.
- **Coordinate frames:** CA-1M poses live in the FARO venue frame. A per-capture
  rigid Umeyama fit (residual in `property:gt:umeyama_rms_m`) connects them to
  the ARKit `/world` via a static transform at `/world/gt`.
- **License:** CA-1M data is **CC BY-NC-ND 4.0** — internal research use only;
  do **not** publicly redistribute RRDs derived from it. The original
  ARKitScenes-derived layers keep the far more permissive
  [ARKitScenes license](https://github.com/apple/ARKitScenes/blob/main/LICENSE).

### TODO

- **FARO-based regeneration of `gt_poses`/`gt_depth`** (unblocks public
  redistribution): the released FARO scans registered scanner-to-scanner only;
  the camera→FARO transform was never published
  ([ARKitScenes#41](https://github.com/apple/ARKitScenes/issues/41)), so this
  means reimplementing Apple's registration pipeline (synthetic laser views,
  feature matching, PnP/RANSAC, photometric refinement). Same layer names,
  different `gt_provenance`.
- **Better mesh:** TSDF-fuse the laser `gt_depth` into a true GT mesh
  (`arkit_mesh` is a device product).
- **Mesh in the `world GT` tab** once wanted (excluded for now).
- **RGB under the GT camera:** blocked on
  [rerun#10422](https://github.com/rerun-io/rerun/issues/10422)
  (`VideoFrameReference` cannot reference a `VideoStream`); until then the
  `world GT` view includes the wide-camera frustum for RGB context. Duplicating
  pixels instead would cost ~545 GB.

## Where the good data hides

The `.mov` is the master, not just video. Its `mebx` metadata streams carry what
the published assets drop, decoded here via NSKeyedArchiver parsing:

- **True 60 Hz camera poses** (`ARImageData.visionTransform`) — 6× denser than
  the published 10 Hz trajectory, with automatic fallback when the alignment fit
  is poor (recorded in `pose_source`).
- **Raw IMU** — accel (in g), gyro (**in deg/s**, converted), fused attitude.
- **Per-frame lens distortion** — real 8-coefficient forward+inverse polynomials.
- **Wide↔ultrawide stereo extrinsics** — ~1.2 cm baseline, millimetre units.
- **Per-camera clocks** — each camera has its own PTS origin (they differ by up
  to ~100 ms in one file); offsets are recovered per camera and gated on
  dispersion and drift.

Orientation is *measured* from gravity (the metadata `sky_direction` label is
wrong for ~60% of sequences) and baked into pixels, intrinsics, and poses.

## Specific sequences

The quickstart downloads a random sample; for explicit sequences pass
`--video-ids` to the download tool (the other steps are unchanged — ingest
skips already-converted sequences):

```bash
python tools/apps/download.py --download-dir data --video-ids 40776203 40776204 \
  --no-include-point-clouds --assets mov annotation mesh lowres_wide.traj \
  confidence lowres_depth lowres_wide_intrinsics ultrawide_intrinsics
python tools/apps/ingest_batch.py --workers 2
python tools/apps/register_catalog.py --rrd-dir data/rrd
```

## Scaling up

The full corpus (5,047 sequences) is a Modal job, not a local one — 32 GPU
workers into a staging volume, one batch uploader to HuggingFace (see
`arkitscenes_download/modal_jobs/` and
[`docs/full-run-runbook.md`](docs/full-run-runbook.md)):

```bash
pixi run -e arkitscenes-download modal run --detach \
  -m arkitscenes_download.modal_jobs.convert_sequences::full_run --encoder gpu --confirm
```

Full per-sequence design: [`docs/architecture.md`](docs/architecture.md).

## Dataset notes

The default raw subset contains movs, low-resolution depth and confidence,
calibration, annotation, and mesh assets. High-resolution upsampling assets, RGB
PNG derivatives, and laser point clouds are deliberately excluded. The downloader
can still fetch any of them when explicitly requested. The 5-sequence sample stays
to a few GB.
Data comes from Apple's CDN under the
[ARKitScenes license](https://github.com/apple/ARKitScenes/blob/main/LICENSE);
24 sequences ship without mesh/annotation/trajectory and are skipped or
recorded as expected failures.
