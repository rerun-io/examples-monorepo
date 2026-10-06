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
GPU, IO, and sensor-IIO crates will be opt-in, with Linux gates where required.
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
and doctests. It does not run Python tooling. Run each affected consumer's gate
and its required numeric checks before committing a staged item.

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
