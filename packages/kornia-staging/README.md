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
  Add crates when their first item lands. The empty 3d crate bootstraps the workspace.

## Destinations

Only `kornia-staging-3d` exists so far. CPU crates are default workspace members;
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

## Tracking

| Item | From | Destination repo/crate/module | Status | Upstream issue/PR |
| --- | --- | --- | --- | --- |
