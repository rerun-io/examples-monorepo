# arkitscenes-download

Download [ARKitScenes](https://github.com/apple/ARKitScenes) (Apple, 5,071 indoor
iPhone/iPad captures) and ingest it into **layered Rerun recordings**: full-rate
60 fps video, true per-frame camera poses, raw IMU, depth (ARKit + laser GT),
lens distortion, and 3D ground-truth boxes — most of which never appears in the
dataset's published PNG/traj assets because it lives inside the `.mov` files.

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

## What one sequence becomes

Seven small `.rrd` files sharing one `recording_id`, so the catalog stacks them
into a single recording — and any one aspect can be regenerated and re-registered
without touching the rest (no re-transcode to fix box math):

| layer | contents |
|---|---|
| `base` | recording properties (clock offsets, orientation, pose provenance, …) — queryable segment-table columns |
| `calibration` | 60 Hz `world_T_rig`, per-camera pinholes, **lens distortion** (8-coefficient polynomials via simplecv components), stereo extrinsics |
| `video_wide` / `video_ultrawide` | AV1 `VideoStream` at native resolution and framerate |
| `depth` | ARKit depth + laser GT depth + confidence |
| `imu` | 100 Hz accelerometer / gyroscope / fused attitude |
| `gt` | annotated mesh + oriented 3D bounding boxes |

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

## Scaling up

The chunked pipeline downloads → ingests → verifies → ships → registers →
cleans staging, resumably (kill it anywhere; rerun continues from
`data/pipeline-state/`):

```bash
# everything, published to a local directory
pixi run -e arkitscenes-download arkitscenes-download-pipeline

# or to a remote destination over ssh (transport + sha256 verification per scheme)
python -m arkitscenes_download.pipeline --destination user@host:/srv/arkitscenes/rrd --read-mount /mnt/arkitscenes/rrd
```

Failures record to `failed.txt` and never block; `--retry-failed` re-queues them
after transient outages. Full design and measured performance history:
[`docs/architecture.md`](docs/architecture.md).

## Dataset notes

Sizes are real: the full raw subset this pipeline needs is ~6.5 TB downloaded
(movs + depth + calibration; PNG image folders and laser point clouds are
deliberately excluded — the PNGs are downsampled derivatives of the movs), and
the resulting rrds total ~2 TB. The 5-sequence sample keeps that to a few GB.
Data comes from Apple's CDN under the
[ARKitScenes license](https://github.com/apple/ARKitScenes/blob/main/LICENSE);
24 sequences ship without mesh/annotation/trajectory and are skipped or
recorded as expected failures.
