# slam-rs

Visual-inertial odometry with a Rust core. The estimator is a port of the
basalt VIO fork: pure Rust, CPU first, N-camera from the start. Python owns the
plumbing — catalog feed, decode, evaluation and Rerun logging — and talks to the
core through a PyO3 extension module.

The core is still a stub. The state machine, the error taxonomy and the value
types across the boundary are real, and so is everything on the Python side —
manifest, feed, metrics and replay — but the estimator itself lands in later PRs,
so `track()` never reports `Tracking` yet.

## Layout

| Path | What it is |
|---|---|
| `crates/slam-rs` | The core (`slam_rs` lib). No Python, no Rerun, no GPU. |
| `crates/slam-rs-py` | PyO3 `cdylib` built in place as `slam_rs/_core.so`. |
| `crates/slam-rs-cli` | `slam-rs` binary: runs the core with no Python at all. |
| `slam_rs/` | The Python package: stubs, Tyro entry points under `apis/`. |
| `tools/` | Thin CLI shims over `slam_rs/apis/`. |
| `reference_segments.toml` | The frozen reference set (below). |
| `tests/reference/` | Checked-in basalt C++ trajectories the gate tests reproduce. |

`Cargo.lock` is committed. `cargo` never runs during `pixi lock` or
`pixi install`: the build is an explicit, cached pixi task.

## Build and gates

```bash
pixi run -e slam-rs-dev --frozen slam-rs-build      # cargo build + install _core.so in place
pixi run -e slam-rs-dev --frozen tests              # pytest (depends on the build)
pixi run -e slam-rs-dev --frozen lint               # ruff
pixi run -e slam-rs-dev --frozen typecheck          # pyrefly
pixi run -e slam-rs-dev --frozen deadcode           # vulture
pixi run -e slam-rs-dev --frozen slam-rs-clippy     # cargo clippy -D warnings
pixi run -e slam-rs-dev --frozen slam-rs-rust-test  # cargo test --workspace
pixi run -e slam-rs-dev --frozen slam-rs-version    # print the core version
```

## Python API

```python
import numpy as np
from slam_rs import _core

vio = _core.Vio(camera_count=2, min_imu_samples=1)
vio.push_imu_batch(t_ns, gyro, accel)  # int64[n], float64[n, 3], float64[n, 3]
result = vio.track(t_ns, [left, right])  # uint8[h, w] per camera
result.status, result.world_from_rig  # VioStatus, [tx ty tz qx qy qz qw]
```

Images are copied in and the GIL is released around the core call, so a decoder
thread keeps running. Wrong dtype, rank, shape or memory layout raises
`ValueError`; IMU samples must be strictly increasing in time.

## The reference set

`reference_segments.toml` freezes ten Monado SLAM Dataset segments — five
two-camera `msd-index` (KB4 fisheye, 54 Hz) and five four-camera `msd-g2`
(radtan8, 30 Hz) — in three tiers: **smoke** on every commit, **accuracy** per
pull request, **long** nightly. Each entry carries the catalog entry id, the NAS
storage URLs of the `base` and `gt` layers with their registered size and schema
digest, the `gt.csv` sidecar, the capture and ground-truth properties the catalog
reports, the frozen decode path and the frozen IMU noise model. A `[[dataset]]`
block per catalog dataset pins the rig geometry — per-camera resolution and image
rotation — and a `[robocap]` section adds session 15, which has no ground truth
and is gated against basalt's own output instead.

Three things are frozen because the catalog cannot carry them and each one moves
the numbers: the IMU noise densities and update rate (basalt's `msd*_calib.json`),
the camera-to-IMU time offset (0 for MSD, 14,902,432 ns for RoboCap), and the
decode path (`cpu_gray8_dav1d_1thread`, worth about 5 cm of ATE against NVDEC RGB).

```python
from slam_rs.reference import load_manifest

manifest = load_manifest()
segment = manifest.in_tier("smoke")[0]
```

### Two clocks, converted once

The catalog indexes a segment on `video_time`, which is **relative** to
`capture.start_time_ns`. Every basalt CSV, including the `gt.csv` sidecars, is on
the **absolute** device clock. On the Index smoke segment the two differ by
10,433,867,587,166 ns, so a trajectory exported on the wrong clock associates with
nothing at all. The feed works in `video_time` throughout and
`trajectory.shift_clock` converts once, at the CSV boundary — the same discipline
the w-first-versus-XYZW quaternion ordering follows.

## The feed, the metrics and the replay tool

`slam_rs.catalog_feed` turns one segment into calibration and grayscale
framesets. It reads a catalog URL or local `.rrd` files served in process (no
catalog server needed) and decodes AV1 with single-threaded dav1d to `gray8`.
Video, inertial samples and ground truth are all fetched one `video_time` window
at a time, with window edges on frames that are keyframes in every camera, so
`window_s` really does bound memory on the 410 s and 586 s segments. Each
frameset carries a sha256 of its pixels, the inertial samples since the previous
frameset (running one past its own timestamp, so a blocking backend cannot
deadlock), and the nearest ground-truth pose within the association tolerance.

`slam_rs.trajectory` reads and writes basalt's CSV form (`#timestamp [ns], p_x,
…, q_w, …`, w-first, integer nanoseconds), associates two trajectories with a
5 ms tolerance, and reports rigid-aligned ATE plus the fork's PASS criteria. The
alignment is `golden_compare.py`'s inline arithmetic rather than the shared
`simplecv` helper, whose variance floor would reject a stationary rig that the
fork passes.

```bash
pixi run -e slam-rs-dev --frozen python tools/apps/replay.py \
    --rr-config.headless --rr-config.save data/replay-smoke.rrd
```

Both `--rr-config.headless` and `--rr-config.save` are honoured; in a shell
without `DISPLAY`, pass `--rr-config.headless` or the spawned viewer wedges the
recording stream.

## Tests

`pytest -q` deselects the `slow` marker and runs in under a second on synthetic
inputs. The slow tests read a reference `.rrd` from the NAS or query the
catalog, and skip when neither is reachable:

```bash
pixi run -e slam-rs-dev --frozen tests   # fast
cd packages/slam-rs && pytest -m slow -q # NAS + catalog
```
