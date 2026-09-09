# slam-rs

Visual-inertial odometry with a Rust core. The estimator is a port of the
basalt VIO fork: pure Rust, CPU first, N-camera from the start. Python owns the
plumbing — catalog feed, decode, evaluation and Rerun logging — and talks to the
core through a PyO3 extension module.

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
| `frontend` | The optical-flow frontend: `patterns` (Pattern52/51 from `patterns.h`; the other two are unreachable on every shipped config), `se2` (`AffineCompact2` and `Sophus::SE2::exp`), `ldlt` (Eigen's pivoted LDLT at 3x3), `patch` (the streaming inverse-compositional patch build), `tracker` (`PatchSoA`, `FlowTransforms`, the `SourcePatches`/`PatchTracker` stage traits and `CpuPatchTracker`), `detect` (basalt's centred cell grid over kornia-rs's FAST plus OpenCV's suppression), `flow` (`FrameToFrameOpticalFlow`, generic over the builder and tracker) and `parallel` (the explicit thread budget). |

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

## Layout

| Path | What it is |
|---|---|
| `crates/slam-rs` | The core (`slam_rs` lib). No Python, no Rerun, no GPU. |
| `crates/slam-rs-py` | PyO3 `cdylib` built in place as `slam_rs/_core.so`. |
| `crates/slam-rs-cli` | `slam-rs` binary: a placeholder. `version` is the only subcommand that does anything; a replay runs through the Python tools. |
| `slam_rs/` | The Python package: stubs, Tyro entry points under `apis/`. |
| `tools/` | Thin CLI shims over `slam_rs/apis/`. |

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

The frontend alone is driven the same way, for a run that needs no backend:

```python
flow = _core.OpticalFlow(calibration, config, threads=1)
frame = flow.process(t_ns, [left, right])
frame.ids(0)         # int64[n], ascending
frame.positions(0)   # float32[n, 2] pixels
frame.transforms(0)  # float32[n, 2, 3], [[m00, m01, tx], [m10, m11, ty]]
frame.occupancy(0)   # int32[rows, columns] over camera 0's detection grid
frame.num_new(0), frame.num_tracks(0)
flow.t_ns              # int | None: the last accepted frameset, None before the first
```

Any `int64` is a timestamp, negative ones included: the frontend's clock is an
`Option<i64>` rather than basalt's `t_ns = -1` sentinel (`optical_flow.h:172`),
which read every negative timestamp as "no previous frame" and so restarted
tracking on each one. Framesets must still arrive strictly in order, and a
refused frameset leaves the frontend exactly as the last accepted one did.
