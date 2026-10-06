# slam-rs design notes

A reference for the code; paths are relative to `packages/slam-rs/`.
Decisions are one line each. The full record behind them, with measurements and
rejected experiments, is the version of this file before the cut:
`git show b05efc00:packages/slam-rs/docs/design-notes.md` (2,022 lines).

## Core modules

### Layout

| Path | What it is |
|---|---|
| `crates/slam-rs` | The core (`slam_rs` lib). No Python, no Rerun; the GPU frontend is its `gpu-wgpu` feature. |
| `crates/slam-rs-py` | PyO3 `cdylib` built in place as `slam_rs/_core.so`. |
| `crates/slam-rs-cli` | Native `slam-rs` commands, including replay and catalog input. |
| `slam_rs/` | The Python package: stubs, Tyro entry points under `apis/`. |
| `tools/` | Thin CLI shims over `slam_rs/apis/`. |
| `slam.toml` | Runtime settings: estimator files and RoboCap camera selection/reader rules. Sensor calibration comes from the catalog. |
| `benchmarks.toml` | Regression cases, tiers, hold-outs, frozen decode paths and lane baselines. |
| `configs/` | Dataset VIO configurations and the `profiles/` overlays. |

### Modules

| Module | What it is |
|---|---|
| `lie` | Scalar literals and NaN-preserving maximum; `So3`/`Se3` use `kornia-staging-algebra::lie`. |
| `types` | `TimeCamId`, `KeypointId`/`LandmarkId`, `AbsOrderMap`, `PoseState`/`PoseVelState`/`PoseVelBiasState` and the two fixed-linearization wrappers. |
| `config` | `VioConfig`, read from the package's `configs/*_config.json`, plus the `port.*` overlay keys the profiles set. |
| `calib` | `Calibration`: extrinsics, the six shipped camera models, the 9- and 12-parameter IMU bias calibrations, plus a constructor that takes what the Python catalog feed reports. |
| `camera` | `SlamCamera` adapts the validated `kornia-staging-3d::camera` models to homogeneous estimator inputs. Calibration formats live in staging. |
| `image` | `kornia_image::Image<u16, 1>`: an owned flat 16-bit frame with an explicit row stride, the stride-aware `u8 << 8` widening basalt's readers do, and `interp`/`interp_grad`/`in_bounds` reproduced from `image.h` in the same arithmetic order. |
| `pyramid` | Application pyramid scheduling around `kornia-staging-imgproc::pyramid::PyramidPlanU16`; integer reduction and geometry validation live in staging. |
| `landmark` | `StereographicParam` (`project`/`unproject` and both Jacobians), the three-parameter `Landmark` with its backup pair, and `LandmarkDatabase`: the host->target->landmark adjacency, the `min_num_obs = 2` sweep and `remove_keyframes`. Landmarks live in one id-sorted `Vec` behind a `BTreeMap` index rather than a per-landmark hash map, and every map is a `BTreeMap`, so iteration order is reproducible (D31). |
| `ba_base` | Window states, error evaluation and backup/restore; hosted reprojection and relative pose factors use `kornia-staging-slam::factors`; bearing triangulation uses `kornia-staging-3d::pose`. |
| `imu` | Frame-boundary accumulation, gravity initialization and inertial factors. `kornia-staging-sensors::imu` owns midpoint preintegration, covariance and bias propagation. |
| `frontend` | Application frame stages, camera selection, track identities and worker policy. `kornia-staging-imgproc` owns cell detection and patch arithmetic; `kornia-staging-slam::tracking::optical_flow` owns batch tracking and template caches. |
| `gpu` | Application scheduling around `kornia-staging-gpu` pyramids, FAST selection, fused KLT, camera kernels and readback. One tracker collection path also carries detector and stereo results. |
| `linearize` | Application landmark/state indexing and linearization scheduling; `kornia-staging-slam::sqrt_ba` owns landmark QR, back-substitution and deterministic dense reduction. |
| `marg` | Application keep/marginalize scheduling around the rank-aware QR operations in `kornia-staging-slam::sqrt_ba`. |
| `qr` | Householder and scaled Givens kernels now live in `kornia-staging-algebra::linalg`. |
| `estimator` | The Offline sliding-window driver: `process_frame` (cover, initialise, measure, optimise, marginalize), `schedule` (basalt's keyframe vote, the lazy keyframe budget and the keep/marginalize sets `marg` is given) and `optimize` (the Levenberg-Marquardt loop, with the per-frame `lambda` reset, the `lambda · diag(H)` damping and the shared 7-iteration budget over reused scratch), and `frame_update`, the fast profile's between-keyframes solve. |

## Python API

```python
from pathlib import Path

from slam_rs import _core
from slam_rs.reference import profiled_config_text

calibration = _core.Calibration.from_catalog(feed.cameras, feed.imu)  # the feed's dataclasses
config = _core.VioConfig.from_json(Path("configs/msdmi_config.json").read_text())  # the dataset configuration
config = _core.VioConfig.from_json(profiled_config_text(Path("configs/msdmi_config.json"), "fast"))  # or with the overlay
config.optical_flow_image_safe_radius = 472.0    # settable per device, though the shipped file carries it

vio = _core.Vio(calibration, config, threads=1, gpu=False)   # gpu=True runs the frontend on this host's GPU
vio.push_imu_batch(t_ns, gyro, accel)     # int64[n], float64[n, 3], float64[n, 3], uncalibrated
result = vio.track(t_ns, [left, right])   # uint8[h, w] per camera
result.status, result.world_from_rig      # VioStatus, [tx ty tz qx qy qz qw]
```

`track` returns a status and world-from-rig pose: `Tracking`, or `NeedMoreImu` with no state change; retry the same frameset.
`snapshot()` copies the last measured window and landmarks; it stays on the previous measurement during `NeedMoreImu`.
`flow_frame()` copies the last accepted frameset’s keypoints; both accessors return `None` before their first result.

```python
snapshot = vio.snapshot()       # None until a frameset has measured
snapshot.window_t_ns            # int64[n], the 15-dof states then the pose blocks
snapshot.window_poses           # float64[n, 7], [tx ty tz qx qy qz qw]
snapshot.window_keyframe        # bool[n]: a keyframe, and window_long_term for a long-term one
snapshot.kf_ids, snapshot.marginalized                   # the keyframes as ids, plus what left
snapshot.landmark_ids, snapshot.landmark_positions       # int64[p], float64[p, 3] world
snapshot.landmark_hosts         # int64[p], the hosting keyframe's timestamp
snapshot.lm_iterations, snapshot.lm_lambda, snapshot.num_observations
snapshot.lm_error_before, snapshot.lm_error_after
snapshot.timings_ms             # the six estimator stages, milliseconds
snapshot.frame_update           # whether the between-keyframes update ran

frame = vio.flow_frame()        # the keypoints of the last accepted frameset, or None
```

## The gate

- Tiers: `smoke` (MIO10, MGO09), `release` (MIO07, MGO07, MIO14),
  and `listed` (the other five); the two hold-outs stay excluded from tuning.
- Inputs and ground truth come from the catalog; an unavailable catalog fails
  integration tests. RoboCap has no ground truth and receives no ATE verdict.
- Prefer the baseline for the host, lane and profile; otherwise use the first
  row for that lane/profile. RMSE must be at most `1.10 * baseline.gt_rmse_cm`.
- Median accepted tracker-call time must be at most
  `1.10 * baseline.median_tracker_ms` on the baseline host only.
  Elsewhere report the time with “speed not gated on this host”.
- A missing lane/profile baseline reports “no baseline” and leaves accuracy
  ungated; tracking and validity clauses still apply.
- Require enough tracked poses, zero lost framesets, at least
  `MIN_ASSOCIATED_POSES` estimate poses associated with ground truth, finite
  poses and measurements, and estimated extent at most `DIVERGENCE_FACTOR`
  times ground-truth extent.

MSD `video_time` is relative to `capture.start_time_ns`; trajectory CSV uses
absolute device time. Add the capture epoch only at export (`trajectory.shift_clock`).
RoboCap `video_time` is already an absolute boot-relative camera clock.
DataForge subtracts the camera-to-IMU offset when logging IMU samples.
The feed adds the offset to **all three streams: cameras, IMU and ground truth**,
subtracting it from query bounds first; shifting cameras alone misaligns them.
The common shift is the negative of static IMU `applied_time_shift_ns`.
Do not apply factory per-camera offsets again. A catalog pose layer subtracts
both the export epoch and camera offset to return to base `video_time`.

## Register a SLAM layer

Run a registered RoboCap, msd-index, msd-g2 or msd-odyssey segment and replace
its single `slam_rs` layer. The output directory must be visible at the same
absolute path to both the worker and the catalog server:

```bash
pixi install -e slam-rs-cuda
pixi run -e slam-rs-cuda --frozen slam-rs-wgpu-build
pixi run -e slam-rs-cuda --frozen slam-rs-catalog-layer \
  --catalog rerun+http://dgx-spark.ilish-ruler.ts.net:9988 \
  --segment robocap__f408193e6447b3b0__s00000059 \
  --output-dir /path/to/shared/slam_rs
```

Defaults are `fast`, automatic GPU frontend selection, and CUDA/NVDEC decoding
when available. `--decode-device cpu` selects the reference PyAV pixel
conversion; `--backend cpu` selects the CPU estimator frontend. On hosts without
CUDA, use the existing `slam-rs` or `slam-rs-osx` environment.

The offline command fetches the selected cameras' compressed packets together
once, builds their timestamp index from that result, and keeps decoders alive
for the whole catalog segment. CUDA uses SimpleCV's TorchCodec reader, GPU
resize and grayscale conversion, then transfers small grayscale batches to the
Rust API. After indexing and muxing, the feed releases the packet buffers and
retains only muxed streams. Allow several times the encoded size for transient
Arrow, muxing and decoder buffers. The existing bounded-window feed
remains the default for other tools and cap use.

The layer animates the existing rig and adds a full trajectory, a recent trail,
start/end markers and run metadata, following the Basalt layout. It preserves
the base videos, sensor data and calibration. One estimator spans the session's
file rolls. A run must finish with a finite pose for every supplied frameset
before replacing the result. The DataForge blueprint includes both old Basalt
and new slam-rs paths. Repeated runs replace `slam_rs`; they do not create named
run versions. This command registers only the derived `slam_rs` data layer.
The base recording and its blueprint must already be registered; the command
does not ingest raw data or register/change blueprints.
The shared layout has no RoboCap-specific eye orientation. RoboCap ingestion
continues to supply its calibrated follow-eye settings.

Layer generation does not load ground truth. When separate scoring tools need
ground truth, the feed isolates the catalog's registered `gt` RRD in a
temporary local catalog. This prevents estimated rig poses from entering later
ground-truth queries through merged layers. The worker must be able to read
that registered URI; a `file://` URI requires the input storage mounted at the
same path. An inaccessible source fails explicitly, without using merged poses
as ground truth. No raw dataset files are parsed or copied.

NVDEC's RGB-to-gray conversion can differ from PyAV's direct YUV-to-gray
conversion. The decoder is recorded in layer metadata; changing it is a change
to the estimator's pixels, not only its speed.

## Patched dependencies

`kornia-staging-patch-deps` verifies the CubeCL archive SHA256, applies the checked-in
patches into `packages/kornia-staging/target/patch/`, and checks the prepared files on reuse.
A process lock makes concurrent preparation safe. Cargo consumes that tree
through `[patch.crates-io]`; the Pixi Cargo tasks prepare it first.

Bare Cargo and rust-analyzer need this once per fresh checkout:

```bash
pixi run -e kornia-staging --frozen kornia-staging-patch-deps
# Same command on macOS
```

The CubeCL park and wgpu poll tests run offline on standalone prepared crates.
From `packages/kornia-staging`, fill each Cargo home's cache once:

```bash
pixi run -e kornia-staging --frozen cargo fetch --locked --manifest-path target/patch/cubecl-common-0.11.0-pre.3/Cargo.toml
pixi run -e kornia-staging --frozen cargo fetch --locked --manifest-path target/patch/cubecl-wgpu-0.11.0-pre.3/Cargo.toml
pixi run -e kornia-staging --frozen kornia-staging-patch-test
```

Use the same environment on macOS. A missing `test-log` offline error means the
cache is incomplete. The version-bump runbook is beside `[patch.crates-io]`
in `packages/kornia-staging/Cargo.toml`.

```bash
pixi run -e slam-rs-dev --frozen slam-rs-build
pixi run -e slam-rs-dev --frozen tests
pixi run -e slam-rs-dev --frozen lint
pixi run -e slam-rs-dev --frozen typecheck
pixi run -e slam-rs-dev --frozen deadcode
pixi run -e slam-rs-dev --frozen slam-rs-clippy
pixi run -e slam-rs-dev --frozen slam-rs-rust-test
pixi run -e slam-rs-dev --frozen slam-rs-wgpu-build
pixi run -e slam-rs-dev --frozen slam-rs-wgpu-clippy
pixi run -e slam-rs-dev --frozen slam-rs-wgpu-test
pixi run -e slam-rs-dev --frozen slam-rs-wgpu-doc
# macOS: use -e slam-rs-osx-dev for the same tasks.
```

## Decision references

- **D03** — The core logs nothing; Python consumes copied arrays and scalars for visualization.
- **D05** — Use a scalar-generic estimator and preintegration, f64 calibration input, and an f32 frontend.
- **D08** — Widen grayscale input with `u8 << 8` into owned u16 images for interpolation.
- **D09** — FAST detection reuses kornia's grid-cell detector behind a thin wrapper
- **D10** — Damp the reduced system with `lambda · diag(H)` and a diagonal floor.
- **D11** — Reset LM lambda each frame; the tag is also used for allocation-free camera Jacobians and rejection of unsupported camera models.
- **D12** — Share the LM iteration budget with rejected trials; the tag also names midpoint IMU integration and translation/rotation/velocity state order.
- **D13** — Support only ABS_QR with square-root marginalization; the tag also names negating the increment for residuals `π(x) − z`.
- **D14** — Superseded by S34: catalog ground truth, per-lane/profile manifest baselines, and same-host speed checks.
- **D16** — Expose the estimator to Python one frameset at a time.
- **D17** — Threading: Offline (lockstep) mode first; Realtime mode is an enum value reserved for later
- **D18** — Construct VIO from configuration and calibration, with square-root priors on initial translation, yaw and biases; leave roll and pitch free.
- **D20** — The marginalization prior cost omits the constant `½rᵀr` and may be negative.
- **D21** — Choose CPU/CubeCL stages at construction with kernel tolerance checks; the tag also names camera 0 as the sole, rate-limited keyframe voter.
- **D22** — Evict keyframes by tracked-feature ratio, then the DSO-derived distance score whose self-distance favors removing a candidate near the newest keyframe.
- **D24** — Feed the same IMU samples to separate frontend and estimator preintegrators; frontend prediction seeds KLT.
- **D28** — Freeze MSD decoding to single-threaded dav1d `gray8` and honor padded image row strides.
- **D29** — Apply the camera-to-IMU clock correction once to all streams; read the common shift from catalog `applied_time_shift_ns`.
- **D30** — Keep each camera’s own resolution, including rigs with different stored orientations.
- **D31** — Deterministic reductions and the thread budget have an explicit Rust mapping
- **D32** — Panic policy: the core never panics on data; non-finite handling follows each numerical contract
- **D34** — The shipped VIO path has the landmark and pose damping machinery disabled; the port mirrors that
- **D35** — Superseded by S34: catalog ground truth, per-lane/profile manifest baselines, and same-host speed checks.
- **D36** — Superseded by S34: catalog ground truth, per-lane/profile manifest baselines, and same-host speed checks.
- **D41** — Originally port Eigen's pivoted LDLT semantics; S34 replaces the dynamic solve with scaled f64 nalgebra LU.
- **D44** — Originally preserve Eigen's rotation and threshold-sensitive operation order; D79 and S34 supersede that requirement.
- **D47** — Triangulation acceptance depends on reduction order at the inverse-distance boundary; D79 replaces the ordered SVD with nalgebra in f64.
- **D49** — Reuse host staging buffers so GPU corner scanning does not allocate on the host between frames.
- **D51** — Expose copied window, landmark and marginalized-frame snapshots for Python visualization.
- **D58** — Superseded by S34: catalog ground truth, per-lane/profile manifest baselines, and same-host speed checks.
- **D59** — Superseded by S34: catalog ground truth, per-lane/profile manifest baselines, and same-host speed checks.
- **D60** — Superseded by S34: catalog ground truth, per-lane/profile manifest baselines, and same-host speed checks.
- **D64** — Reaffirmed for the GPU lanes: no bit-accuracy; the bar is accuracy inside the band and faster than the CPU lane on the same machine
- **D65** — Merge reprojection-error partial sums in host-frame index order for deterministic accumulation.
- **D68** — The three unreachable blocks go: squared-form marginalization, nullspace diagnostics, the D34 damping stack
- **D70** — One GPU runtime: the CUDA lane is removed; wgpu is the GPU lane (2026-09-09)
- **D71** — Exponent-bit finite classification and bounded small-angle trig; the MIO14 replay passes its unchanged accuracy limit (2026-09-09)
- **D72** — The GPU detector picks one corner per grid cell on the device; the candidate image never comes back (2026-09-09)
- **D73** — The estimator's LM buffers live on the estimator and its hot loops walk columns; no arithmetic changes (2026-09-10)
- **D74** — Dataset JSON is the base input; profiles overlay validated keys. S34 makes fast the default and retains reference as the unchanged input.
- **D75** — Redetect on demand: the fast profile detects when camera 0 holds fewer than 85 % of the last detecting frameset's keypoints (2026-09-10)
- **D76** — The fast profile solves the window at keyframes and the newest 15-dof state alone between them, falling back to the joint solve when that update declines (2026-09-10)
- **D77** — The GPU frontend waits once per phase and reserves its queue budget before it enqueues (2026-09-10)
- **D78** — One stage's download carries another's buffers: camera 0's cell selection rides the temporal tracks' read (2026-09-10)
- **D79** — Eigen's operation order is no longer a requirement: nalgebra and kornia-algebra do the arithmetic where they can; the gate is ATE on the catalog (2026-09-10)
- **D80** — D64 remains the GPU rule: accuracy inside the band and faster than the CPU lane on the same machine (2026-09-11)
- **D81** — Prefer a host/lane/profile baseline; fall back to the first lane/profile row, and gate speed only on the selected row's host (2026-09-11)
- **D82** — Ship the CubeCL worker fix as a checksummed archive plus local patch, prepared under a process lock and selected by Cargo; no maintained fork or upstream worker PR (2026-09-11)
- **D83** — Require at least 1.2x GPU/CPU speedup on the same machine with accuracy in band; 1.5x is a research goal, separate from the 1.10x regression limits (2026-09-11)
- **D84** — `port.keyframe_solve_deferred`: a keyframe frameset with visual support returns the frame-update pose; its triangulation, joint solve and marginalization run on a second thread beside the next frameset's frontend and finish before the estimator touches the window again (deterministic; gated on ATE, 2026-10-01)
- **S34** — Remove external comparison fixtures and fallbacks; use catalog ground truth, lane/profile baselines, and library numerics.
- **S36** — Remove Metal wait and worker sleeps through the wgpu upgrade and local CubeCL park/unpark patch; use matched host baselines.
