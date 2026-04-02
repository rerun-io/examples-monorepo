# Architecture

Detailed internals of the gsplat-rust-renderer for developers who want to understand or modify the Rust/WGSL code. For usage, see the [README](../README.md).

## Two-Process Design

```
  ┌──────────┐  gRPC    ┌─────────────────────────────────────────────────────┐
  │  Python   │────────► │                 Rust Viewer Process                  │
  │  Client   │  :9876   │                                                     │
  └──────────┘          │  ┌───────────┐   ┌──────────────┐   ┌───────────┐  │
       │                 │  │ Rerun Data│──►│  Gaussian    │──►│ Gaussian  │  │
  ┌────▼─────┐          │  │   Store   │   │  Visualizer  │   │ Renderer  │  │
  │ .ply file │          │  │           │   │(VisualizerSys│   │  (wgpu)   │  │
  └──────────┘          │  └───────────┘   └──────────────┘   └─────┬─────┘  │
                         │                                           │        │
                         │                                      ┌────▼────┐   │
                         │                                      │ Display │   │
                         └──────────────────────────────────────┴─────────┘───┘
```

**Python side** (`gsplat_rust_renderer/gaussians3d.py`):
- Parses PLY files into a `Gaussians3D` dataclass
- Implements `rr.AsComponents` so it can be passed directly to `rr.log()`
- Sends component batches over gRPC to the Rust viewer

**Rust side** (`src/`):
- `main.rs` — Launches a stock Rerun viewer + gRPC server, registers the custom visualizer
- `gaussian_visualizer.rs` — Queries the data store, builds render clouds, culls/sorts splats
- `gaussian_renderer.rs` — Manages GPU resources and dispatches the compute pipeline

## Component Contract

Python and Rust agree on a custom `GaussianSplats3D` archetype. Both sides must produce/consume the exact same component descriptors:

| Component | Rerun Type | Shape | Description |
|---|---|---|---|
| `centers` | `Translation3D` | `[N, 3]` | World-space Gaussian positions |
| `quaternions` | `RotationQuat` | `[N, 4]` | Rotation quaternions (xyzw) |
| `scales` | `Scale3D` | `[N, 3]` | Per-axis scale factors |
| `opacities` | `Opacity` | `[N]` | Per-splat opacity [0, 1] |
| `colors` | `Color` | `[N]` | Base RGB from SH DC coefficient |
| `sh_coefficients` | `TensorData` | `[N, C, 3]` | Optional higher-order SH (degree 0-4) |

If you add a new component, you must update both:
- Python: `gsplat_rust_renderer/gaussians3d.py` → `as_component_batches()`
- Rust: `src/gaussian_visualizer.rs` → `GaussianSplats3D` archetype impl + `build_render_cloud()`

## Per-Frame Visualizer Pipeline

Each frame, the Rerun viewer calls `GaussianSplatVisualizer::execute()`. Here's what happens:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Per-Frame Execution                             │
│                                                                     │
│  ┌───────────┐     ┌──────────────┐     ┌──────────────────────┐   │
│  │  Query     │────►│  Cache Check │────►│  Build/Reuse Cloud   │   │
│  │  Archetype │     │  (signature) │     │  (RenderGaussianCloud│   │
│  └───────────┘     └──────────────┘     └──────────┬───────────┘   │
│                                                     │               │
│                                          ┌──────────▼───────────┐   │
│                                          │  Extract Camera      │   │
│                                          │  (view state or      │   │
│                                          │   fallback bounds)   │   │
│                                          └──────────┬───────────┘   │
│                                                     │               │
│                                          ┌──────────▼───────────┐   │
│                                          │  Visibility Culling  │   │
│                                          │  (frustum + depth    │   │
│                                          │   + opacity filter)  │   │
│                                          └──────────┬───────────┘   │
│                                                     │               │
│                                          ┌──────────▼───────────┐   │
│                                          │  Depth Sort          │   │
│                                          │  (back-to-front,     │   │
│                                          │   cap at 200K)       │   │
│                                          └──────────┬───────────┘   │
│                                                     │               │
│                                          ┌──────────▼───────────┐   │
│                                          │  Submit to Renderer  │   │
│                                          │  (GaussianDrawData)  │   │
│                                          └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Cache Signature

The cloud is only rebuilt when the "signature" changes:
- `expected_splats` — total number of Gaussians
- `sh_coeffs_per_channel` — SH degree (None, 1, 4, 9, 16, or 25)
- `transform_bits` — bit-exact 3×4 entity transform

This means if you log the same PLY twice without changing anything, the cloud is reused.

### Visibility Culling (CPU)

Before sending splats to the GPU, the visualizer does a fast approximate cull on the CPU:
1. Reject splats with zero opacity
2. Reject splats behind the near plane
3. Reject splats outside 1.5× the NDC bounds (generous margin)
4. If more than 200K survive, keep only the closest ones
5. Sort back-to-front for correct alpha blending

For clouds with >16K splats, culling and sorting use `rayon` for parallel execution.

## GPU Renderer: Compute Tile Path

The preferred render path runs a 7-stage compute pipeline entirely on the GPU. This is inspired by the [Brush](https://github.com/ArthurBrussee/brush) renderer.

```
Sorted splat indices + cloud data (from CPU)
          │
          ▼
┌──────────────────┐   gaussian_project.wgsl
│  1. Project      │   Exact 3D→2D Gaussian projection on GPU.
│     Splats       │   Builds 3D covariance (R·diag(s²)·Rᵀ), projects to 2D
│                  │   via camera Jacobian, evaluates SH for view-dependent
│                  │   color, computes tile bounding box.
└────────┬─────────┘
         │
         ▼
┌──────────────────┐   gaussian_project.wgsl (scan_blocks + scan_block_sums + scatter)
│  2. Compact      │   Parallel prefix sum (Blelloch scan) to remove invisible
│     Visible      │   splats and pack the visible ones into a dense array.
│     Splats       │   Three sub-dispatches: scan within blocks → scan block
│                  │   sums → scatter to compacted output.
└────────┬─────────┘
         │
         ▼
┌──────────────────┐   gaussian_map_intersections.wgsl
│  3. Map          │   For each visible splat, scatter one (tile_id, splat_id)
│     Intersections│   pair for every screen tile it overlaps. A single splat
│                  │   covering 4 tiles produces 4 entries. Also runs a tiny
│                  │   1-thread "clamp" dispatch to cap the total count.
└────────┬─────────┘
         │
         ▼
┌──────────────────┐   gaussian_dynamic_sort.wgsl
│  4. Radix Sort   │   Multi-pass 4-bit radix sort on (tile_id, splat_id)
│     by Tile ID   │   pairs. Groups all splats for the same tile together.
│                  │   6 sub-dispatches per pass: count → reduce → scan →
│                  │   compose → add → scatter.
└────────┬─────────┘
         │
         ▼
┌──────────────────┐   gaussian_tile_offsets.wgsl
│  5. Tile Offsets │   Scan the sorted tile IDs to find the [start, end)
│                  │   range for each tile. Like finding boundaries in a
│                  │   sorted list.
└────────┬─────────┘
         │
         ▼
┌──────────────────┐   gaussian_raster_tiles.wgsl
│  6. Tile Raster  │   One workgroup (256 threads) per 16×16 pixel tile.
│                  │   Loads splats in batches into shared memory, evaluates
│                  │   the Gaussian function per pixel, accumulates color
│                  │   front-to-back with early termination when a pixel
│                  │   reaches full opacity.
└────────┬─────────┘
         │
         ▼
┌──────────────────┐   gaussian_composite.wgsl
│  7. Composite    │   Fullscreen triangle that samples the off-screen raster
│                  │   texture and writes it to the Rerun viewport framebuffer.
└──────────────────┘
```

### CPU Fallback Path

When the compute tile path isn't available, the renderer falls back to:
1. Project each visible splat on the CPU using `project_gaussian_to_ndc()`
2. Upload per-splat instance data (center, radius, inverse covariance, color, opacity)
3. Draw instanced quads using `gaussian_splat.wgsl` (6 vertices per splat)
4. The fragment shader evaluates the Gaussian falloff using the Mahalanobis distance

### Buffer Management

GPU buffers are cached per-entity path and grow as needed (never shrink):
- Splat data buffers: means, quaternions, scales+opacity, colors, SH coefficients
- Intermediate buffers: visibility flags, tile hit counts/offsets, intersection pairs, sort scratch
- The intersection count is read back from the GPU (with a 2-frame delay via readback slots) to right-size tile buffers for the next frame

## Python Client Flow

```
┌──────────┐     ┌────────────────┐     ┌─────────────┐     ┌──────────┐
│  Load    │────►│  Parse PLY     │────►│  Normalize   │────►│  Log via │
│  .ply    │     │  (plyfile)     │     │  & Validate  │     │  gRPC    │
│  file    │     │                │     │              │     │          │
└──────────┘     │  Extract:      │     │  - unit quat │     │  rr.log( │
                 │  - positions   │     │  - clamp     │     │    entity│
                 │  - rotations   │     │    scales    │     │    splats│
                 │  - scales      │     │  - sigmoid   │     │  )       │
                 │  - opacity     │     │    opacity   │     │          │
                 │  - SH coeffs   │     │  - SH→RGB    │     │  + Send  │
                 │  - f_rest_*    │     │  - shape     │     │  blueprint
                 └────────────────┘     │    checks    │     └──────────┘
                                        └──────────────┘
```

The `Gaussians3D.from_ply()` class method handles all standard 3DGS PLY formats:
- `x/y/z` → centers
- `rot_0/1/2/3` → quaternions (converted from wxyz to xyzw)
- `scale_0/1/2` → scales (exponentiated from log-space)
- `opacity` → sigmoid activation
- `f_dc_0/1/2` → DC SH coefficients → base RGB
- `f_rest_*` → higher-order SH (channel-major layout)
- `red/green/blue` or `r/g/b` → direct vertex colors (fallback)

## Key Constants

| Constant | Value | Where | Purpose |
|---|---|---|---|
| `MAX_SPLATS_RENDERED` | 200,000 | visualizer + renderer | Hard cap per frame |
| `TILE_WIDTH` | 16 px | shaders | Tile size for compute raster |
| `PARALLEL_SPLAT_THRESHOLD` | 16,384 | visualizer | Switch to rayon parallel |
| `SIGMA_COVERAGE` | 3.0 | visualizer | Standard deviations for bbox |
| `BRUSH_COVARIANCE_BLUR_PX` | 0.3 | visualizer + shaders | Anti-aliasing blur |
| `SH_C0` | 0.28209 | visualizer + shaders | Zeroth SH coefficient |
| `GRPC_PORT` | 9876 | main.rs | Default gRPC listen port |
