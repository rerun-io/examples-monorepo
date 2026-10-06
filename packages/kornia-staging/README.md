# kornia-staging

Rust code prepared for the Kornia organization lives here until it lands upstream.
Consumers use Cargo path dependencies into this workspace. This package never
imports slam-rs, robocap-live, handfit, or gsplat-rust-renderer.

## Rules

- Each crate and module has a named upstream destination. Use Kornia types,
  typed errors, destination image buffers, public API documentation, doctests,
  unit tests, and Criterion benches for performance-critical kernels on entry.
- Rewire consumers in the same change. Pure moves must preserve output bytes;
  numerical changes must pass the agreed accuracy bounds and golden tests.
- When an item lands upstream, bump the upstream dependency, swap consumer
  imports, and delete the staged item in one change. If upstream rejects an
  item, keep it here and mark it `not going upstream` with the reason.
- Track every staged item in the table below, including its upstream issue or PR.
  Add crates when their first item lands.
- Land prerequisites before their consumers: Scalar, algebra kernels, cameras/pose,
  image operators/tracking, then SLAM, sensors, and GPU backends.

## Destinations

CPU crates are default workspace members;
GPU, IO and sensor-IIO are opt-in members for bare Cargo commands.
Module paths mirror the destination module paths.

| Crate | Destination repository / crate |
| --- | --- |
| kornia-staging-imgproc | kornia-rs / kornia-imgproc |
| kornia-staging-3d | kornia-rs / kornia-3d |
| kornia-staging-algebra | kornia-rs / kornia-algebra |
| kornia-staging-io | kornia-rs / kornia-io |
| kornia-staging-sensors | kornia-slam / kornia-sensors |
| kornia-staging-slam | kornia-slam |
| kornia-staging-sensor-iio | sensor-rt / proposed sensor-iio |
| kornia-staging-gpu | proposed kornia-gpu |

## Checks

From the repository root:

```sh
pixi run -e kornia-staging-dev --frozen gate
```

The gate checks Rust formatting, Clippy with warnings denied, workspace tests,
and doctests for every workspace member except GPU, in both feature modes,
plus Python patch-preparation tests.
Run `pixi run -e kornia-staging-dev --frozen kornia-staging-gpu-test`
for the opt-in GPU tests and Clippy on the host device.
CPU-only bare Cargo commands also require these workspace patches.
Prepare the pinned CubeCL trees with `kornia-staging-patch-deps` before bare Cargo;
consumer workspace roots must repeat the three `[patch.crates-io]` overrides. Run each affected consumer's gate
and its required numeric checks before committing a staged item.

The GPU gate also tests the patched wgpu poll scheduler. On a fresh Cargo home,
fetch its standalone test dependencies once with `cargo fetch --locked --manifest-path
packages/kornia-staging/target/patch/cubecl-wgpu-0.11.0-pre.3/Cargo.toml` after patch preparation.

## Numerical layouts

Camera Jacobian arrays use `[row][column]`: two pixel rows, with three point
columns or the model's intrinsic-parameter columns. Other matrix arrays use
`[column][row]`, matching nalgebra/glam storage. Lie and factor boundaries use
nalgebra types directly. Conversions at camera boundaries change storage only.

## Tracking

| Item | From | Destination repo/crate/module | Status | Upstream issue/PR |
| --- | --- | --- | --- | --- |
| Sealed f32/f64 Scalar and Sophus constants | slam-rs LieScalar; W1 camera precision | kornia-rs / kornia-algebra / scalar | staged | - |
| Camera trait, pinhole, robust KB4; ProjectionReject extends upstream with OutsideDomain/NonFinite | slam-rs camera.rs; robocap-live kornia_ext/fisheye.rs | kornia-rs / kornia-3d / camera | staged | - |
| Brown4/5/8/12/14, analytic derivatives and robust inverse | slam-rs radtan8; simplecv Brown–Conrady | kornia-rs / kornia-3d / camera | staged | - |
| Fisheye624 / Fisheye62, analytic derivatives and robust inverse | simplecv fisheye624.py; handfit SymForce test oracles | kornia-rs / kornia-3d / camera | staged | - |
| Canonical serde tags; COLMAP and Basalt converters | camera crosswalk research; slam-rs calibration schemas | kornia-rs / kornia-3d / camera::formats | staged | - |
| Virtual pinhole maps and opt-in approximate KB4 f32/NEON kernel | robocap-live kornia_ext/virtual_camera | kornia-rs / kornia-3d / camera::virtual_camera | staged | - |
| Integer area resize | slam-rs `area.rs`, `area/kernels.rs` | kornia-rs / kornia-imgproc / resize | staged | - |
| 4x4 pooling | robocap-live `kornia_ext/pool.rs` | kornia-rs / kornia-imgproc / resize | staged | - |
| Scaled zero-border bilinear remap | robocap-live `kornia_ext/remap.rs` | kornia-rs / kornia-imgproc / interpolation::remap | staged | - |
| Heatmap peaks | robocap-live `kornia_ext/heatmap.rs` | kornia-rs / kornia-imgproc / features | staged | - |
| Minimum enclosing circle | robocap-live `hands/circles.rs` | kornia-rs / kornia-imgproc / contours::min_enclosing_circle | staged | - |
| Pivoted fixed-size LDLT | slam-rs `ldlt.rs` | kornia-rs / kornia-algebra / linalg::ldlt | staged | - |
| Lie precision and update extensions (proposes the documented Sophus Taylor branches upstream) | slam-rs `lie.rs` | kornia-rs / kornia-algebra / lie | staged | - |
| Bearing triangulation and stereographic chart | slam-rs `landmark.rs`, `ba_base.rs` | kornia-rs / kornia-3d / pose | staged | - |
| Strided u8-shift8 ingestion and sparse u16 bilinear values/gradients (dense conversion uses upstream cast_and_scale) | slam-rs `image.rs` | kornia-rs / kornia-imgproc / color, interpolation | staged | - |
| Floor-halved integer u16 Gaussian downsampling and reusable PyramidPlanU16 | slam-rs `pyramid.rs` | kornia-rs / kornia-imgproc / `pyramid` | staged | - |
| Centered FAST cells, band scans, masks and deterministic selection | slam-rs `frontend/detect*`, `frontend/cell.rs` | kornia-rs / kornia-imgproc / `features` (private cells) | staged | - |
| Mean-normalized SE(2) patches and sealed sampling patterns (private Sophus-style exponential preserves normalization and cubic small-angle term) | slam-rs `frontend/{patch,patterns,se2,simd,ldlt}.rs` | kornia-rs / kornia-imgproc / `optical_flow::patch_se2` | staged | - |
| Stateless forward/backward CPU patch tracking and reusable storage | slam-rs `frontend/tracker/cpu.rs`, `patch_soa.rs`, `storage.rs` | kornia-rs / kornia-imgproc / `optical_flow::patch_tracker` | staged | - |
| Batched tracking protocol and identity-keyed template caches | slam-rs `frontend/tracker.rs`, `tracker/cpu.rs` | kornia-slam / kornia-slam / `tracking::optical_flow` | staged | - |
| Wgpu polling wakeup | cubecl-wgpu 0.11.0-pre.3 compute/{poll,stream,timings} | cubecl-wgpu | staged (temporary patch) | - |
| CubeCL runtime, storage/subgroup probes and typed failures; pinned patch preparation | slam-rs `gpu/runtime.rs`, patches and prepare helper | proposed kornia-gpu / runtime; CubeCL patches | staged | - |
| Persistent u16 GPU pyramids | slam-rs `gpu/pyramid.rs`, `gpu/kernels/pyramid.rs` | proposed kornia-gpu / `pyramid` | staged | - |
| GPU FAST scan and packed cell selection | slam-rs `gpu/detect`, FAST and cell kernels | proposed kornia-gpu / `features` | staged | - |
| Fused forward/backward SE(2) KLT, persistent dispatch storage and finite/trig helpers | slam-rs `gpu/kernels/klt_fused.rs`, reusable parts of `gpu/track*` | proposed kornia-gpu / `optical_flow` | staged | - |
| Checked GPU transfers, readback lookahead and exclusive execution | slam-rs `gpu/submission.rs` (generic part) | proposed kornia-gpu / `transfer` | staged | - |
| Brown8 GPU projection and robust damped inverse with CPU validity rules | slam-rs `gpu/kernels/onewait.rs`; staged CPU camera | proposed kornia-gpu / `camera` | staged | - |
| IIO scan decoder and configurable Linux RAII buffer owner | robocap-recorder `iio.rs`, `device.rs`; robocap-live `capture/iio.rs`, `capture/device.rs` | sensor-rt / sensor-iio | staged | - |
| Rust MPLANE capture (no C), clock flags, leased MMAP planes and copied/borrowed luma | robocap-live `capture/camera.rs`, `v4l2_mplane.c`; robocap-recorder `camera.rs`, `native/camera.c` (C cores ported to Rust) | kornia-rs / kornia-io / v4l::mplane | staged | - |
| Annex-B splitter and subprocess H.264 encoder | robocap-live log/video.rs | kornia-rs / kornia-io::video | staged; upstream fit open: overlaps kornia_io gstreamer::VideoWriter; Pablo decides at landing (D6) | - |
| H.264/AV1 packets to borrowed luma planes with explicit limited/full/unknown range | slam-rs-cli decode.rs + native/dav1d.c | kornia-rs / kornia-io::video | staged | - |
| N-camera body rig and IMU corrections | slam-rs calib.rs, robocap-live frame.rs | kornia-slam / kornia-sensors::rig | deferred: no consumer reads it | - |
| Runtime frames, IMU combiner and timestamp matcher | `robocap-types`, `robocap-live/{source,capture}` + `robocap-recorder/live_slam.rs` | kornia-slam `kornia-sensors` (`imu` module and frame exports) | staged | - |
| Householder QR and scaled Givens with slice storage and scratch | slam-rs `qr.rs` | kornia-rs / kornia-algebra / linalg::qr | staged | - |
