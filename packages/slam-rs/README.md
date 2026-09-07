# slam-rs

Visual-inertial odometry with a Rust core. The estimator is a port of the
basalt VIO fork: pure Rust, CPU first, N-camera from the start. Python owns the
plumbing — catalog feed, decode, evaluation and Rerun logging — and talks to the
core through a PyO3 extension module.

The core is still a stub. The state machine, the error taxonomy and the value
types across the boundary are real, and so is everything on the Python side —
manifest, feed, metrics and replay — but the estimator itself lands in later PRs,
so `track()` never reports `Tracking` yet.

## Core modules

The core is being filled in stage by stage, bottom up. What is in it today:

| Module | What it is |
|---|---|
| `lie` | `So3`/`Se3` over any `f32`/`f64` scalar: Sophus's `exp`/`log`, the adjoint, basalt's four SO(3) Jacobians and their inverses, the decoupled SE(3) pair, and the left-multiplied pose increment the estimator runs on. |
| `types` | `TimeCamId`, `KeypointId`/`LandmarkId`, `AbsOrderMap`, `PoseState`/`PoseVelState`/`PoseVelBiasState` and the two fixed-linearization wrappers. |
| `config` | basalt's `VioConfig`, read straight from `data/**/*_config.json`. |
| `calib` | basalt's `Calibration`: extrinsics, the six shipped camera models, the 9- and 12-parameter IMU bias calibrations, plus a constructor that takes what the Python catalog feed reports. |
| `camera` | `pinhole`, `kb4` and `pinhole-radtan8` with basalt's 4-D homogeneous `project`/`unproject` and their analytic Jacobians (2x4 point, 2xN parameter, 4x2 and 4xN for unprojection), the `rpmax` and `z >= epsilonSqrt` domain checks, and a `CameraEnum` that dispatches without a vtable. `ds`, `eucm` and `ucm` parse but are rejected here. |
| `image` | `ImageU16`: an owned flat 16-bit frame with an explicit row stride, the stride-aware `u8 << 8` widening basalt's readers do, and `interp`/`interp_grad`/`in_bounds` reproduced from `image.h` in the same arithmetic order. |
| `pyramid` | The `PyramidBuilder` stage seam with an associated `Pyramid` type that lends nothing (geometry plus a copy into the caller's buffer), `PyramidU16` (one flat buffer per level, not basalt's packed mipmap) and `CpuPyramidBuilder`, whose `subsample` is bit-exact with `image_pyr.h:99-140`. |
| `imu` | Preintegration: `IntegratedImuMeasurement<S>` with basalt's midpoint propagation, covariance and bias-Jacobian recurrences, the 9-vector residual and its Jacobians, the LDLT square-root inverse covariance, the between-frames accumulation loop, gravity initialisation, and the 15-row IMU block the estimator whitens. |
| `frontend` | The optical-flow frontend: `patterns` (Pattern24/52/51/50 from `patterns.h`), `se2` (`AffineCompact2` and `Sophus::SE2::exp`), `ldlt` (Eigen's pivoted LDLT at 3x3), `patch` (the streaming inverse-compositional patch build), `tracker` (`PatchSoA`, `FlowTransforms`, the `SourcePatches`/`PatchTracker` stage traits and `CpuPatchTracker`), `detect` (basalt's centred cell grid over kornia-rs's FAST plus OpenCV's suppression), `flow` (`FrameToFrameOpticalFlow`, generic over the builder and tracker) and `parallel` (the explicit thread budget). |

Every convention is quoted against the C++ it comes from, file and line, in the
doc comments. `crates/slam-rs/tests/fixtures/` holds the shipped basalt config
and calibration JSON the parsers are tested against, unmodified, plus four
fixtures produced by the C++ fork itself: `pyramid/`, the first frame of the
smoke reference segment as a PGM next to the four pyramid levels the fork builds
from it, which the pyramid is checked against byte for byte;
`camera_oracle.json`, what basalt's camera headers return for ten cameras and
thirty points each in **both** precisions - pixel, bearing, and in double also
both projection Jacobians and the unprojection Jacobian - plus six probe pixels
handed straight to `unproject`, one of them singular; and `imu/imu_oracle.json`,
the delta state, covariance, bias Jacobians, Eigen LDLT and square-root inverse
covariance of seven preintegration runs, plus what
`Quaternion::FromTwoVectors` returns for ten accelerometer readings; and
`flow/`, three 960x960 frameset pairs as PGMs beside the keypoints the C++
frontend produced from eight of them. The camera port reproduces every double to
1e-15 relative (1e-12 for unprojections, which run a Newton iteration) and every
float **exactly**; the IMU port reproduces every double to 1e-14, and to 1e-7
through the whitening, which inverts the covariance. All four generators live on
the fork's `slam-rs-reference` branch, as `tools/dump_pyramid.cpp`,
`tools/camera_oracle.cpp`, `tools/imu_oracle.cpp` and `tools/dump_flow.cpp`; the
monorepo never compiles C++.

The IMU fixture earns its keep on one run: the covariance after a single sample
with a still gyroscope and accelerometer is rank deficient, and what basalt does
with it is decided entirely by `Eigen::LDLT`. Eigen pivots on the *un-updated*
diagonal, eliminates velocity first and leaves the position pivots at `-1.6e-27`,
which basalt's `vectorD()[i] < numeric_limits::min()` test zeroes. A textbook
pivoted LDLT eliminates position first, leaves a tiny *positive* pivot, and puts
an information weight of `6.2e26` on a direction the measurement says nothing
about.

One thing the camera port inherits and the frontend does live with:
`unproject` runs a fixed three (kb4) or five (radtan8) Newton steps, and on wide
calibrations that is not always enough. Inside basalt's own
`optical_flow_image_safe_radius` nine of the ten shipped cameras invert to 1e-11;
msd-g2 cam2 is off by 0.12 in bearing *inside* that radius, and RoboCap cam1
outside it returns a bearing pointing backwards. basalt's C++ returns the same
numbers to the last figure, so this is a property of the algorithm, not of the
port; `crates/slam-rs/tests/camera_jacobians.rs` pins all three cases.

### The frontend, and the one thing that is not bit-parity

Everything on the tracking path is the C++'s arithmetic in the C++'s order, and
it shows: seeded with basalt's own keypoints on the smoke segment's first eight
framesets, the port's tracker puts **697 of 697** of them within half a pixel of
where the C++ put the same id one frame later, the worst of them 0.0003 px away.
That gate lives in `crates/slam-rs/tests/flow_parity.rs` and runs off two small
fixtures: three 960x960 frameset pairs as PGMs (the exact bytes the Python feed
decodes) and the eight per-frameset JSON dumps the fork's `tools/dump_flow.cpp`
produced from them.

The detector is the one place the port cannot be bit-identical (decision D09,
trap 2), but it is close. basalt runs `cv::FAST` on each 8-bit cell; the port
runs kornia-rs's FAST over basalt's own centred cell geometry, and at arc length
9 kornia returns the same canonical FAST-9 `cornerScore` OpenCV computes. The
grid, the threshold ladder, the per-cell budget, the safe radius, the masks, the
edge margin **and OpenCV's non-maximum suppression** — strictly greater than all
eight neighbours, so a tie kills both sides — are all reproduced. What is left is
the cell walk itself and an unstable `std::sort`. Measured against the C++ dump:
**95.5 to 96.9% of the C++'s keypoints have a port keypoint within one pixel**,
the two agree on camera 0's keypoint count on **seven of the eight** framesets,
their occupancy grids agree on **87 of 87** occupied cells, and every one of the
171 corners both sides picked at the same pixel carries the **same integer
`cornerScore`** — OpenCV's `max(a0, -b0) - 1`, which is one less than the value
kornia returns. The gate still seeds the tracker rather than diffing keypoint
sets, because "close" is not "equal".

The frontend is generic over its two stages. `PyramidBuilder` and `PatchTracker`
(with `SourcePatches` beside it) carry associated pyramid and patch types, and
`FrameToFrameOpticalFlow<P, B, T>` defaults them to the CPU pair — so a CubeCL
backend arrives through `with_backends` and no public signature here names a
concrete pyramid, and `FlowResult` has a public writing surface (`reset`,
`set_track`, `parts_mut`, `finish`) that the CPU tracker itself publishes
through, so a second backend can too. Every per-patch buffer is
structure-of-arrays with the patch index fast-varying, including the 2x3 warps,
which live as six flat coefficient arrays; the patch build streams straight into
those arrays, and at `-O3` its largest local allocation measures 36 bytes against
a 344-byte stack frame.

A frame the frontend refuses is as if it never happened: the whole call runs
against a staging pyramid set over a snapshot of the keypoint state, and the two
pyramid sets swap and the clock advances only after tracking, the cell counts and
the add/match/filter passes have all succeeded. Snapshot and restore are
**allocation-free** — every type in the chain writes `clone_from` by hand, since
the derived one replaces the buffers instead of overwriting them — and
`crates/slam-rs/tests/frame_allocations.rs` counts that with a global allocator
rather than claiming it. A whole frame is not allocation-free: on a 200x200
scene it reaches the allocator a few hundred times, all of it inside kornia's
FAST, which allocates one `Vec` per image row per cell per rung of the threshold
ladder. A frame that finds no keypoints at all costs more, which is what pins the
cost there rather than on anything the port owns.

Three deviations are recorded in the source. `E[i]` is computed from `T_c0_ci`
per camera rather than reusing the cam0-cam1 matrix everywhere, which is
upstream's bug (`optical_flow.h:207-213`); `FrontendOptions::epipolar_per_camera`
switches it back for a C++-parity run and makes no difference at all on a
two-camera rig. Image bounds come from each camera's own resolution rather than
camera 0's, because the msd-g2 recordings are stored rotated — while the
detection grid is per camera, as the C++'s is, and the occupancy matrix keeps
camera 0's shape, as the C++'s does. And `FrontendOptions::max_keypoints` is a
budget basalt has no equivalent of: the port's buffers are preallocated, so
detection and matching stop adding once a camera is full rather than producing a
frame the tracker cannot carry.

Still to come: the square-root estimator.

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
| `slam_rs/reference_bundle.py` | Resolves the two long-tier artifacts kept out of git. |

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

### The basalt C++ reference and the gate policy

`tests/reference/msd/<segment>/` holds what the basalt C++ fork produced on each
segment: `run.json` for all ten (fork commit, deterministic settings, the VIO
config and calibration actually pushed, timings and the ATE against `gt.csv`),
`basalt_traj.csv` for the eight smoke and accuracy segments, and `frames.sha256`
plus a copy of `gt.csv` for the smoke pair so its gate runs with no NAS and no
catalog. The two long-tier trajectories are 3.4 MB and 4.8 MB and stay out of
git; `slam_rs.reference_bundle` resolves them from `SLAM_RS_REFERENCE_DIR` (or
`data/reference/`) and the tests skip with a message naming the variable.

Each segment carries a `gate_policy`, because basalt is not equally good
everywhere:

| policy | segments | why |
|---|---|---|
| `tight` | MIO10, MGO09, MIO07, MGO07 | basalt scores 0.8-2.4 cm; a regression is unambiguous |
| `standard` | MIO04, MGO14, MIO14, MIPT03 | 8-38 cm, stable; gate relative to basalt's own number, not an absolute threshold |
| `no_divergence` | MGO01, MGO13 | basalt is near failure: 43 cm and 78 cm here, 68 cm for the C++ binary on the raw files, and **18-32 cm of spread between two legitimate decode paths of the same estimator** — on MGO01 the ordering even flips. Only "kept tracking, did not diverge" is measurable. |

`slam_rs.trajectory.ate` reproduces all ten published C++ figures exactly, and a
re-decode through `catalog_feed` reproduces the C++ run's per-camera pixel
digests frame for frame (824 of 824 on MIO10, 428 of 428 on MGO09). That second
result is the load-bearing one: it means an A/B between the two estimators
measures the estimator, not the decoder.

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
