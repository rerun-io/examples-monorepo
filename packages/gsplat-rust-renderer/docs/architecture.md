# Architecture

The Cargo workspace has six crates:

| Crate | Responsibility |
| --- | --- |
| `gsplat-core` | Raw-wgpu projection, sorting, rasterization, and per-view feedback |
| `gsplat-render` | Brush PLY input, camera files, and standalone image output |
| `gsplat-bench` | Brush/native/core comparison, orbit cameras, and synchronized timing |
| `gsplat-eval` | Image metrics and report provenance |
| `gsplat-train` | Brush training with native Rerun observations |
| `gsplat-viewer` | Native archetype queries, compute visualization, and composition |

## Source and compatibility

Brush is pinned to `1388f74c`. The `brush-src` Pixi-build package fetches that
revision and applies `patches/brush-1388f74c-process-observer.patch`. This patch
exposes existing train-step and refinement statistics; it changes no training or
renderer math. All 16 Brush workspace packages resolve from one installed tree.
An activation script maintains a stable `target/brush-src` link across prod/dev
environments. The source recipe and the Cargo patch comment describe upgrades.
The separately built `brush-cli` reference uses unpatched upstream source.

Rerun is pinned to 0.38.1, Burn to `faec3983`, CubeCL to `46752244`, and wgpu to
registry 30.0.0. Rerun's wasm-bindgen pin prevents use of Brush's wgpu 30.0.1 fork.
The native path is tested; a web build still needs compatible bindings and
WebGPU subgroup support. Kernels stay within WebGPU workgroup, storage-buffer,
and dispatch limits. Devices without subgroups fail with a capability error.

## Core and image boundaries

The core depends only on wgpu and small math/layout libraries. It contains no
Rerun, Brush, Burn, or async runtime types. `Scene` owns shared GPU input;
`ViewState` owns scratch buffers, bindings, and bounded feedback slots. Several
views can share one upload and encode before one queue submission.

The forward path projects, sorts visible depths, scans intersection counts,
projects visible splats, maps and sorts tiles, computes tile offsets, and
rasterizes. Counts and dispatch plans stay on the GPU. An eight-byte count
readback detects overflow; the output remains intact until capacity grows and
the view rerenders. Pinhole, KB4, RT8, and thin-prism camera models share this path.
Render mode is a per-render option: default blur or mip antialiasing.

Targets own their GPU handles and provide f32 RGBA, packed RGBA8, or storage
textures. The viewer requests color and alpha-weighted camera depth for
composition with opaque scene content. Standalone parity uses raw f32 splats;
native `GaussianSplats3D` adds RGBA8/f16 quantization, which is reported separately.

## Viewer

The custom visualizer reads native components and respects explicit visualizer
instructions. Automatic selection changes only the active custom viewer's
blueprint. Recordings without overrides remain portable to stock Rerun.

A store-owned cache shares uploads across views and tracks each instance's target
with dirty, pending, or complete state. Optional-attribute changes invalidate the
upload even when centers retain their row ID. Blueprint values need a content
hash because their source row IDs are cleared by Rerun. Camera and mode changes
reuse the scene. Unused entries are evicted when frames advance.

The integration currently submits once and allocates two full-viewport targets
per instance per view. Submission count and target memory therefore grow with
both instance and view count. An upstream integration should batch encodes and
pool targets. Splats in separate entities are not jointly sorted. Picking,
hover, and outlines are not implemented. The eye comes from public view state
and can lag one UI frame. Bounds support automatic framing.

`gsplat-viewer-probe` is an optional binary (`probe` feature). It reads the camera
through a view context system and measures a full headless UI/compute/composite
frame with GPU completion. Production viewer code contains no timing hooks.

## Training and evaluation

The trainer consumes typed Brush process events. Recording is disabled unless a
sink is set. A bounded channel retains at most eight pending observations;
intermediate step metrics can be dropped and counted, while snapshots,
evaluations, and final metrics use backpressure. Intermediate snapshots are
DC-only; the final snapshot includes higher-order SH. Brush's seeded synthetic
initialization is replaced by the shared seed-42 100k-point cube to avoid its
pinned random-frustum initialization defect.

Training scores use in-memory, unclipped float renders through `gsplat-bench score`.
PNG directory scores include clipping and rounding from image export. Both call the
same evaluator. Image scores use Brush's black-background, byte-premultiplied GT and SSIM
convention. The white-background published-checkpoint guard is explicit and
separate. Render parity compares unclipped RGBA floats, including alpha and
white-composited PSNR. Reports record runtime source SHA, Cargo.lock SHA-256,
and crate version; exported binaries can set `GSPLAT_SOURCE_SHA` explicitly.

Speed measures render plus GPU completion with pixels left on the GPU. Admission
checks load and available GPU utilization, then warmup and repeated timed loops
produce the median of repeat medians and p95. Separate stage timestamps diagnose
costs. Every repeat lasts at least ten seconds and covers complete orbits, so
frame counts can differ but each camera has equal weight. Equal counts would
shorten fast runs below ten seconds or greatly extend the native runs.
Camera checks read the target format and render settings used during timing.
The Rerun native renderer path re-sorts and re-uploads every splat on the CPU
each frame; these renderer-level comparisons do not measure the complete viewer.
Readback and image encoding are outside the timed loop. Missing telemetry
is recorded as unknown. See the [comparison report](https://pablos-4800gt.ilish-ruler.ts.net:8768/gsplat-modern/report.html)
for measurements and host conditions.
