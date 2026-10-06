# Architecture

This page describes the current renderer and recording paths. For copy-paste usage, start with the [README](../README.md).

## System shape

The native training path is `gsplat-train` → Brush tensors → Rerun 0.38.1
`GaussianSplats3D`. The bench crate owns float rendering and evaluation.
The older custom viewer and standalone `gsplat-render` still share `gsplat_core`
and its WGSL shaders in this branch; the native viewer migration replaces that
viewer path separately.

## Native training wire contract

The trainer uses the generated Rerun `GaussianSplats3D` archetype. Centers and
scales are float32 triples, rotations are XYZW quaternions, and colors are RGBA8.
Higher SH uses `SphericalHarmonics3Rgb` (45 float16 values), with an explicit
`spherical_harmonics_degree`. The legacy Python `Gaussians3D` adapter belongs to
the older viewer path and is not used by training.

## Viewer frame lifecycle

For every matching entity and frame, `GaussianSplatVisualizer::execute`:

1. Resolves the real eye committed by the `Spatial3DView`.
2. Skips the first camera-less frame and requests a repaint after 100 ms. It does not invent a fallback camera, avoiding the old tiny/misplaced splats that snapped into place after startup.
3. Queries required centers plus every optional component and the entity transform.
4. Hashes Rerun's resolved query rows, the SH toggle, splat count, and transform.
5. Reuses or rebuilds a store memoized `RenderGaussianCloud`, assigning every rebuild a globally unique generation.
6. Submits the full cloud, generation, and camera to the GPU renderer.

The renderer keeps per-entity GPU buffers across camera motion. A generation change reuploads attributes; capacity grows geometrically; entities unused for 600 frames are evicted. On a steady frame the CPU mainly updates the camera uniform and encodes commands.

## GPU compute pipeline

The GPU receives the full cloud; there is no CPU cull or depth sort.

| Stage | Shader / operation | Result |
|---|---|---|
| 1. Cull + compact | `gaussian_project::project_forward_main` | Visible `(global_gid, depth_bits)` pairs via `atomicAdd(num_visible)` |
| 2. Depth argsort | `gaussian_dynamic_sort` | GID canonicalization, then front-to-back radix sort |
| 3. Project | `gaussian_project::project_visible_main` | 2D covariance, SH color, tile bounds, hit counts |
| 4. Scan | `gaussian_project::scan_*` | Three-level prefix sum of tile-hit counts |
| 5. Map | `gaussian_map_intersections::map_main` | One `(tile_id, compact_gid)` per covered tile |
| 6. Clamp + dispatch | `clamp_count_main` | Exact live intersection count and indirect sort dispatch arguments |
| 7. Tile sort | `gaussian_dynamic_sort` | Tile-contiguous intersections; count/scatter dispatch only over the live count |
| 8. Tile offsets | `gaussian_tile_offsets` | `[start, end)` range per tile |
| 9. Raster + composite | `gaussian_raster_tiles`, then viewer composite | Premultiplied raster texture, then a fullscreen triangle |

Rasterization uses one 256-thread workgroup per 16×16 tile, Morton-order pixel assignment, and **64-entry shared-memory splat batches**. Pixels blend front-to-back and stop below transmittance `1e-4`; a cooperative counter stops the whole workgroup once all pixels finish.

The clamp stage stores `DrawIndirectArgs` beside the live count. Viewer tile-radix `sort_count` and `sort_scatter` consume those arguments with `dispatch_workgroups_indirect`, avoiding capacity-sized work on sparse frames. The standalone renderer shares the buffer layout and shader but keeps direct dispatches.

## Shared GPU resources

`gsplat_core/gpu_types.rs` owns 12 bind-group layouts and 13 compute pipelines. Important constants are:

| Constant | Value | Meaning |
|---|---:|---|
| `TILE_WIDTH` | 16 px | Raster tile width and height |
| `PROJECT_WORKGROUP_SIZE` | 128 | Projection threads per workgroup |
| `SORT_WORKGROUP_SIZE` | 256 | Radix-sort threads per workgroup |
| `SORT_BITS_PER_PASS` | 4 | 16 radix bins per pass |
| `INTERSECTION_CAPACITY_MULTIPLIER` | 32 | Initial per-splat intersection capacity |
| `MIN_RADIUS_PX` | 0.35 px | Sub-pixel culling threshold |
| `SIGMA_COVERAGE` | 3.0 | Screen-space bounding radius |
| `BRUSH_COVARIANCE_BLUR_PX` | 0.3 | Brush-matching antialias blur |

`TileProjectedSplat` is 64 bytes. The intersection counter/readback uses a small ring so later frames can grow dense-scene buffers without synchronously stalling the render path.

## Training recordings

`crates/gsplat-train` drives Brush's pinned `create_process_with_device` stream.
The local observer adapter exposes the existing loss/LR/refine values without
changing training math, loading, config merge, evaluation or exports. It emits
step events every five steps and at the final step. Refine statistics come
from refine events, not sampled step events. The temporary adapter and complete
patch live in `vendor/brush-process`; remove it when Brush exposes these stats.

The trainer captures tensor handles from the current splat slot before advancing
the stream. A separate logging thread reads them asynchronously, folds Brush's
minimum-scale filter as its exporter does, and logs native `GaussianSplats3D`.
Scales use exp, WXYZ rotations become normalized XYZW, DC/opacity become RGBA8,
and higher SH uses coefficient-major 15-by-3 f16 values. Degree 4 is truncated
only at this logging boundary. Snapshots retain step 50, every 1,000 steps, and
the final step by default; only the final snapshot carries higher SH.
`--snapshot-first` changes the first retained step. Brush's
`--rerun-log-splats-every` changes the periodic cadence. These and
`--rerun-log-train-stats-every` accept positive multiples of five; the latter
keeps Brush's 50-step default. Console progress is independent at 100 steps.
`--rerun-max-img-size` controls eval thumbnails (default 512); camera thumbnails
are capped at 256. Images use JPEG quality 85. The legacy `--rerun-enabled` and
unsupported distribution flag are rejected explicitly, including merged config.
Brush's own Rerun logger stays disabled.

All dynamic data uses `iterations`. Scalars include loss, step milliseconds,
splat counts, learning rates, refine statistics, sampled GPU memory and Brush's
EvalResult PSNR/SSIM. Cameras use Pinhole + Transform3D and small JPEG thumbnails;
distorted camera models are explicitly documented as pinhole approximations.
Four fixed eval views are rendered only at Brush's eval steps, at thumbnail
resolution on the logging thread. They use a black background; GT is
premultiplied with Brush's packed-image conversion. All snapshot and scalar
readbacks run asynchronously on that thread. A logging failure emits a warning
and does not abort training; sink flushing has a two-second timeout. A live
sink must connect during a bounded check before training; an unavailable sink
is disabled with a warning to avoid the SDK's unconnected queue backpressure.

The Rust blueprint places a 3D scene and four eval pairs above metric tabs.
`--video` adds a spinning eye and a flat Loss/PSNR/SSIM/Splats row. Brush's
estimated axis is a model-rotation target from -Y. Apply that rotation's inverse
to -Y for the eye up and nearest world ViewCoordinates, so the spin follows the
unrotated scene. The default has no visualizer override and opens in
stock Rerun. `--compute-visualizer` explicitly selects `ComputeGaussianSplats3D`
for a compatible custom viewer; it cannot be combined with stock `--spawn`.
`--connect` and `--save`
fan out through Rerun's sinks, so live and saved recordings contain the same data.

## Evaluation path

`gsplat-render` loads one PLY and `transforms_test.json`, creates Metal/Vulkan resources once, then reuses them for all cameras. The Python harness pairs outputs by strict relative path and averages per-image PSNR/SSIM after the same 8-bit roundtrip used for checkpoint validation. The checkpoint-only gate first proves the metric implementation against each downloaded prediction/ground-truth split.

## File map

```text
packages/gsplat-rust-renderer/
├── src/
│   ├── main.rs                       custom viewer, headless loop, positional RRD loading
│   ├── gaussian_visualizer.rs        Rerun query, camera lifecycle, CPU cloud cache
│   ├── gaussian_renderer.rs          viewer GPU cache, compute encoding, composite
│   ├── render_cli.rs                 standalone CLI
│   ├── ply_loader.rs                 Rust INRIA PLY parser
│   ├── nerf_camera.rs                NeRF transform parser
│   └── gsplat_core/                  Rerun-free shared renderer
├── shader/                           six WGSL shaders
├── gsplat_rust_renderer/
│   ├── gaussians3d.py                PLY parser and wire batches
│   ├── evaluation.py                 full-split render/eval harness
│   └── apis/                         typed CLI implementations
├── tools/                            thin Tyro entrypoints
├── tests/                            Python tests
└── docs/media/                       real checkpoint and dense-run media
```
