# Architecture

The Cargo workspace has four crates:

| Crate | Responsibility |
| --- | --- |
| `gsplat-core` | Raw-wgpu projection, sorting, rasterization, and per-view feedback |
| `gsplat-cli` | PLY/camera input, image output, float/image metrics, parity, and timing |
| `gsplat-train` | Brush training with native Rerun observations |
| `gsplat-viewer` | Native archetype queries, compute visualization, and composition |

## Source and compatibility

The Brush observer patch changes no training or renderer math.

Rerun is pinned to 0.38.1, Burn to `faec3983`, CubeCL to `46752244`, and wgpu to
registry 30.0.0. Rerun's wasm-bindgen pin prevents use of Brush's wgpu 30.0.1 fork.
The native path is tested; a web build still needs compatible bindings and
WebGPU subgroup support. Kernels stay within WebGPU workgroup, storage-buffer,
and dispatch limits. The core rejects devices without subgroups; the custom viewer uses stock
Gaussian quads when compute capabilities or buffer capacity are insufficient.

## Core and image boundaries

The core depends only on wgpu and small math/layout libraries. It contains no
Rerun, Brush, Burn, or async runtime types. `Scene` owns shared GPU input;
`ViewState` owns scratch buffers, bindings, and bounded feedback slots. Several
views can share one upload and encode before one queue submission. `Renderer`
holds the device and pipelines; the caller supplies the queue and encoder.
The 256-byte uniform block follows renderer upload alignment. Raw allocation,
binding, upload, and readback operations are isolated in `gpu.rs`. Shader files
use relative `#import <...>` directives through a small local import resolver.

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
with dirty, pending, complete, or capacity-failed state. Optional-attribute changes invalidate the
upload even when centers retain their row ID. Blueprint values need a content
hash because their source row IDs are cleared by Rerun. Camera and mode changes
reuse the scene. Unused entries are evicted when frames advance.

`collect_drawables` records compute through the shared
`before_view_builder_encoder`; instances and views add no separate queue submission.
The Transparent composite writes `frag_depth` and uses the renderer's 4x MSAA
state. Each instance/view still owns two full-viewport targets (color and depth);
an upstream integration should pool them. A capacity failure uses stock quads
and keeps that fallback until the scene changes. Both paths provide transformed
bounds for automatic framing.

Splats in separate entities are not jointly sorted. Picking, hover, and outlines
are not implemented. The public 0.38.1 view state supplies `last_eye`, which can lag
one UI frame. Upstream integration needs the current full pose, projection, and
pixel resolution in `DrawableCollectionViewInfo`; camera position alone cannot
replace those fields.

`gsplat probe` is enabled by the `probe` feature of `gsplat-cli`. It reads the camera
through a view context system and measures a full headless UI/compute/composite
frame with GPU completion. Its moving-camera guard rejects held views, and warmup
and measurement end on whole 300-frame orbit boundaries. Production viewer code
contains no timing hooks.

## Training and evaluation

The trainer consumes typed Brush process events. Recording is disabled unless a
sink is set. A bounded channel retains at most eight pending observations;
intermediate step metrics can be dropped and counted, while snapshots,
evaluations, and final metrics use backpressure. Intermediate snapshots are
DC-only; the final snapshot includes higher-order SH. Brush's seeded synthetic
initialization is replaced by the shared seed-42 100k-point cube to avoid its
pinned random-frustum initialization defect.

The [CLI reference](../crates/gsplat-cli/README.md) defines scoring conventions.
Reports record runtime source SHA, Cargo.lock SHA-256, and crate version;
exported binaries can set `GSPLAT_SOURCE_SHA` explicitly.

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
is recorded as unknown. Saved camera fixtures make the comparison path explicit;
resolution changes scale focal lengths and principal points with the image.

## Upstream subset

The portable core is `crates/gsplat-core`: GPU setup, sort/scan/dispatch primitives,
camera and scene types, native conversion, per-view state, and forward-rendering shaders.
Viewer integration lives in `crates/gsplat-viewer/src/{renderer,cache,visualizer}.rs`
and `composite.wgsl`. Resource handles must adapt to Rerun's pools and view context.

Local support stays here: core `lens.rs`/`lens.wgsl` for non-pinhole cameras,
`output.rs`/`raster_outputs.wgsl` for float and packed outputs, `timing.rs`, and the
`shader.rs` import resolver. Texture targets in `output.rs` need pooled handles upstream.
CLI, training, viewer startup and compatibility selection, the frame probe,
Python orchestration, and the patched-source package are also local support.
