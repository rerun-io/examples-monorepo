# live-rerun

Realtime, **zero-transcode** logging of a sensor's hardware-encoded H.264/H.265
stream straight into a Rerun [`VideoStream`](https://rerun.io/docs/reference/types/archetypes/video_stream).
The camera's on-chip encoder bytes pass through untouched — no host-side decode
or re-encode — so a full multi-camera rig streams live at framerate over USB2.

Backend #1 is **DepthAI / OAK**. The logging core (`rerun_video_logger.py`,
`blueprint.py`) is sensor-agnostic; new sensors slot in under `sources/`.

Runs on `linux-64`, `linux-aarch64`, and `osx-arm64`.

## Usage

```bash
# Live viewer only:
pixi run -e live-rerun --frozen live-rerun-oak

# Live viewer AND save a .rrd at the same time (dual-sink):
pixi run -e live-rerun --frozen live-rerun-oak -- --rr-config.save out.rrd

# Headless shell (no DISPLAY): always pass --rr-config.headless or the viewer
# spawn wedges logging.
pixi run -e live-rerun --frozen live-rerun-oak -- --rr-config.save out.rrd --rr-config.headless
```

Key flags: `--source.codec {h265,h264}`, `--source.fps`, `--source.rgb-resolution
{720p,1080p,4k}`, `--source.mono-resolution {720p,800p}`, `--source.usb2`, `--seconds N`
(default: run until Ctrl-C). All three cameras default to **720p (1280x720)** so the
streams share one resolution; RGB 720p is the 1080p sensor ISP-downscaled by 2/3.

## What it logs

A generic, system-agnostic **rig** schema (COLMAP-style), so a different
multicam system maps onto the same layout and opens identically. Entity paths
use `rig_*` / `cam_*` (no `oak`); the OAK specifics stay in `sources/`. The
contract is documented in [`docs/rig_schema.md`](docs/rig_schema.md).

```
/world                       ViewCoordinates.RDF (static)
  /rig_00                    schema metadata (static); no transform = implicit
                             identity now — a SLAM pass logs world_T_rig here
    /cam_00                  left  — reference sensor, identity, frustum tinted green
      /pinhole/video         VideoStream (encoded H.265/H.264 samples)
    /cam_01                  rgb
    /cam_02                  right
```

(Entity ids are zero-padded to two digits — `cam_00`, `rig_00` — so they sort
lexicographically; see [`docs/rig_schema.md`](docs/rig_schema.md).)

- Three encoded streams — left (`CAM_B`) → `cam_00`, RGB (`CAM_A`) → `cam_01`,
  right (`CAM_C`) → `cam_02` — each a `VideoStream` under `…/cam_<NN>/pinhole/video`.
- The **left** camera is the rig **reference sensor** (identity pose, the rig
  origin); rgb/right are expressed relative to it. SLAM is run around the left
  camera, so the rig moves rigidly with it.
- Camera **intrinsics + extrinsics** as static Rerun pinholes (the rig is assumed
  stationary), so the three cameras appear as frusta in a 3D view.
- Per-sensor metadata as static `rr.AnyValues` (`name`, `kind`); a schema block
  (`schema_version`, `reference`, `num_cameras`) on `/world/rig_00`.
- A blueprint with the 3D rig above the three video panels.

Keyframe flags come from the device (`EncodedFrame.getFrameType()`), so scrubbing
anchors to real IDR frames rather than guessing.

## Known limitation: mid-stream seeking / late-join (v1)

The OAK encoder emits the codec parameter sets (SPS/PPS/VPS — the decoder's
"setup header") **once**, on the first keyframe only. v1 logs the encoded bytes
as-is, so:

- ✅ Watching from the **start** decodes correctly (live or replaying a saved `.rrd`).
- ⚠️ **Scrubbing/seeking to a later keyframe** in a saved `.rrd` may not decode.
- ⚠️ A viewer that **connects mid-stream** (a second/reconnecting viewer) may not decode.

The fix (cache the parameter sets and staple them onto every keyframe) is
deferred. Track it if scrubbing/late-join becomes important.

## Not supported

- **IMU is never enabled.** On the target OAK-D-W unit even a minimal IMU stream
  crashes the device (`INTERNAL_ERROR_CORE` / `IMUHalTask`). Do not add it without
  testing the failure deliberately.
- **DepthAI 2.x only** (`depthai==2.27.0.0`). 3.x fails to open this unit.

## Tests

```bash
pixi run -e live-rerun-dev --frozen tests   # import + calibration unit tests
pytest -m hardware                          # end-to-end capture (needs an OAK attached)
```

## Maintaining dependencies (lockfile must be regenerated on Linux)

This package composes the shared `common` feature, and to support Apple Silicon
it adds `osx-arm64` to `common` and `dev` (plus an osx-scoped `pytorch-cpu`).
Because those are *shared* features, **any** change to this package's deps — or to
`common`/`dev` — forces pixi to re-solve the whole workspace lock. Several
linux-only environments build from source (`wilor-nano`'s git `rtmlib`, the
`no-build-isolation` deps `gsplat`/`dpvo`/`sam2`/`moge`), and those **cannot be
cross-built from macOS** (`osx-arm64` → `linux-64` build-dispatch fails).

So: **regenerate `pixi.lock` on the Linux host, not on macOS.** On `dl-server`:

```bash
cd /path/to/examples-monorepo && pixi lock        # or: pixi install -e live-rerun-dev
```

then commit the updated `pixi.lock`. On macOS you can only `pixi install -e
live-rerun-dev --frozen` against an already-current lock.
