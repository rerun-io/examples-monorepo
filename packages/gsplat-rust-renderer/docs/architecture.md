# Architecture

The Cargo workspace has four crates:

| Crate | Responsibility |
| --- | --- |
| `gsplat-core` | Raw-wgpu projection, sorting, rasterization, and per-view feedback |
| `gsplat-cli` | PLY/camera input, image output, float/image metrics, parity, and timing |
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

`gsplat-viewer-probe` is an optional binary (`probe` feature). It reads the camera
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

Training scores use in-memory, unclipped float renders through `gsplat score`.
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
is recorded as unknown. Saved camera fixtures make the comparison path explicit;
resolution changes scale focal lengths and principal points with the image.

## Upstream subset

These files are the inputs to five proposed Rerun changes. `core/` and `viewer/`
refer to `crates/gsplat-core/` and `crates/gsplat-viewer/`; `cli/` is
`crates/gsplat-cli/`. Counts are physical
source lines, including inline tests. Partial-file entries count only the named
functions and give the whole-file size for context; those sizes are not added twice.
The map identifies adaptation work as well as files that can move.

| Future PR | Current files and physical line counts |
| --- | --- |
| U1 infrastructure | `core/src/gpu.rs` (109); `core/src/lib.rs`: `check_adapter`, `compute_limits` (36 selected / 148 file lines); `viewer/src/application.rs`: `compute_wgpu_setup` (27 selected / 365 file lines) |
| U2 sort, scan, dispatch | `core/src/primitives.rs` (6); `core/src/primitives/dispatch.rs` (187); `core/src/primitives/scan.rs` (109); `core/src/primitives/sort.rs` (154); `core/shader/counts.wgsl` (8); `core/shader/dispatch.wgsl` (25); `core/shader/scan.wgsl` (55); `core/shader/scan_common.wgsl` (61); `core/shader/sort.wgsl` (205) |
| U3 forward renderer | `core/src/camera.rs` (49); `core/src/scene.rs` (68); `core/src/types.rs` (95); `core/src/renderer.rs` (66); `core/src/kernels.rs` (106); `core/src/view/mod.rs` (431); `core/src/view/encode.rs` (202); `core/shader/common.wgsl` (80); `core/shader/project.wgsl` (196); `core/shader/map.wgsl` (84); `core/shader/raster.wgsl` (30); `core/shader/raster_common.wgsl` (74); `viewer/src/renderer.rs` (392); `viewer/src/composite.wgsl` (19) |
| U4 GaussianSplats3D wiring | `core/src/native.rs` (90); `viewer/src/cache.rs` (229); `viewer/src/visualizer.rs` (388) |
| U5 precision fields | Future SDK/blueprint schema change; no implementation claimed here. Existing native conversion and archetype comparison tests supply evidence. |

U1 replaces `gpu.rs` with renderer buffer/shader pools, a compute-pipeline pool
with hot reload, and buffer readback. The named capability functions map to
`DeviceCaps` and the viewer device descriptor; the rest of the application stays
local. U2 introduces the primitives with only their pipeline initialization.
U3 adds the forward pipelines, per-view scratch, texture targets, and transparent
composite. Its resource handles must use the new pools. Uniforms already occupy
256 bytes, and the WGSL import syntax is compatible with the renderer resolver.

U4 adapts `cache.rs` to Rerun's retained-cache facilities and integrates native
conversion into the existing Gaussian visualizer. It uses the current full view
camera once U1 extends the collection context. The custom visualizer identifier,
selection system, and fallback-bound providers are local compatibility code;
upstream uses the existing visualizer and its normal bounds. U5 adds typed render
mode and precision fields through SDK/blueprint schemas, with compatibility tests.
It is separate from the compute renderer and is not implemented by this package.

Primitive evidence lives in `core/src/primitive_tests.rs`, the inline dispatch
contracts, and `core/tests/common/`. Forward evidence is in
`core/tests/{render,views,indirect}.rs`; native precision contracts are in
`core/tests/native.rs` and `cli/tests/archetype.rs`. The CLI's float parity suite
and the Python viewer pixel tests exercise the integration boundaries.

Local support remains outside this subset: `core/src/lens.rs` and
`core/shader/lens.wgsl` for KB4/RT8/thin-prism; `core/src/output.rs` and
`core/shader/raster_outputs.wgsl` for float/packed outputs; `core/src/timing.rs`;
the local import resolver in `core/src/shader.rs`; CLI, training, application
startup, frame probe, Python orchestration, and the patched-source package.
The texture target definitions in `output.rs` must be adapted to pooled handles
when moving the forward renderer. Core module wiring and error types likewise
adapt to the renderer module rather than creating a new upstream crate.
