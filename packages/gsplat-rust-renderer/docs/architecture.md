# Architecture

Detailed internals of the gsplat-rust-renderer for developers who want to understand or modify the code. For usage, see the [README](../README.md).

## Design Principles

1. **`gsplat_core` is the single source of truth** — all algorithm code, GPU types, pipeline definitions, and math live in the Rerun-free `gsplat_core/` module
2. **GPU-only rendering, no CPU fallback** — follows the [Brush](https://github.com/ArthurBrussee/brush) approach
3. **Two rendering paths, shared pipeline** — the Rerun viewer and the standalone CLI both use the same WGSL shaders, bind group layouts, compute pipelines, and GPU buffer types
4. **Clean dependency boundary** — `gsplat_core/` depends only on `glam`, `rayon`, `wgpu`, `bytemuck`; the viewer adds `re_*` crates behind a feature flag

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         gsplat_core/ (Rerun-free)                          │
│                                                                             │
│  types.rs          Data structures: RenderGaussianCloud, CameraApproximation│
│  constants.rs      SH_C0, MAX_SPLATS_RENDERED, SIGMA_COVERAGE, etc.        │
│  projection.rs     3D→2D Gaussian projection (camera Jacobian)             │
│  sh.rs             Spherical harmonics evaluation (degrees 0-4)            │
│  covariance.rs     2D covariance from 3D Gaussian + view transform         │
│  culling.rs        CPU frustum culling + depth sorting (pre-GPU pass)      │
│  camera.rs         Camera constructors (look-at, NeRF transform, fallback) │
│  gpu_types.rs      GPU buffer structs, bind group layouts, compute         │
│                    pipelines, helpers — SINGLE SOURCE OF TRUTH              │
│  gpu_context.rs    Headless wgpu device/queue initialization               │
│  gpu_renderer.rs   Standalone 7-stage GPU compute pipeline + readback      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  ply_loader.rs     Rust PLY parser (mirrors Python Gaussians3D.from_ply)   │
│  nerf_camera.rs    NeRF transforms_*.json camera loader                    │
└─────────────────────────────────────────────────────────────────────────────┘
          │                                    │
          ▼                                    ▼
┌──────────────────────┐        ┌──────────────────────────────┐
│  gsplat-render CLI   │        │  gsplat-rust-renderer viewer │
│  (render_cli.rs)     │        │  (main.rs)                   │
│                      │        │                              │
│  No Rerun deps       │        │  gaussian_visualizer.rs      │
│  Raw wgpu            │        │  gaussian_renderer.rs        │
│  Headless rendering  │        │  re_renderer integration     │
│  PNG output          │        │  Rerun viewport composite    │
└──────────────────────┘        └──────────────────────────────┘
```

## Module Dependency Graph

```
render_cli.rs ──► gsplat_core/gpu_renderer ──► gsplat_core/gpu_types ◄── gaussian_renderer.rs
                  gsplat_core/gpu_context       gsplat_core/culling
                  gsplat_core/types             gsplat_core/constants
                  ply_loader                    gsplat_core/sh
                  nerf_camera                   gsplat_core/projection
                                                gsplat_core/covariance
                                                gsplat_core/camera
```

Both rendering paths share:
- **5 WGSL shaders** (`shader/*.wgsl`) — unchanged, no Rerun-specific code
- **12 bind group layouts** (`GpuBindGroupLayouts` in `gpu_types.rs`)
- **13 compute pipelines** (`GpuComputePipelines` in `gpu_types.rs`)
- **7 GPU buffer structs** (`ProjectUniformBuffer`, `ScanUniformBuffer`, etc.)
- **All helper functions** (buffer creation, data packing, dispatch sizing)

## Two Rendering Paths

### Path A: Standalone CLI (`gsplat-render`)

```
PLY file + NeRF JSON ──► ply_loader + nerf_camera
                              │
                              ▼
                    RenderGaussianCloud + CameraApproximation
                              │
                              ▼
                    gpu_render() in gpu_renderer.rs
                      ├─ CPU cull + sort (culling.rs)
                      ├─ Upload to GPU buffers
                      ├─ 7-stage compute pipeline
                      ├─ Readback raster texture
                      └─► RenderOutput → PNG
```

Built with `--no-default-features` (zero `re_*` crates). Uses raw `wgpu` via `GpuContext`.

### Path B: Rerun Viewer (`gsplat-rust-renderer`)

```
Python rr.log() ──gRPC──► Rerun Data Store
                              │
                              ▼
                    GaussianSplatVisualizer::execute()
                      ├─ Query archetype components
                      ├─ Build RenderGaussianCloud
                      ├─ Extract camera from view state
                      ├─ CPU cull + sort (gsplat_core/culling.rs)
                      └─► GaussianDrawData::add_batch()
                              │
                              ▼
                    gaussian_renderer.rs prepare_compute_batch()
                      ├─ Upload to GPU buffers
                      ├─ 7-stage compute pipeline (same shaders!)
                      ├─ Composite to Rerun viewport
                      └─► Rerun draw phase
```

Built with default features (all `re_*` crates). Uses `re_renderer::RenderContext` for wgpu access.

## GPU Compute Pipeline (7 Stages)

Both paths execute the same 7-stage pipeline. The shaders are the single source of truth — defined once in `shader/`, loaded via `include_str!()`.

```
CPU Pre-pass: frustum cull + depth sort (gsplat_core/culling.rs)
          │
          ▼ Upload sorted indices + splat data to GPU
          │
Stage 1:  PROJECT              gaussian_project.wgsl :: project_main
          │  Build 3D covariance (R·diag(s²)·Rᵀ)
          │  Project to 2D via camera Jacobian
          │  Evaluate SH for view-dependent color
          │  Compute tile bounding box + hit count
          │  Write visibility flags
          ▼
Stage 2:  COMPACT              gaussian_project.wgsl :: scan_blocks_main
          │                                          :: scan_block_sums_main
          │  Parallel prefix sum (Blelloch scan)
          │  Remove invisible splats
          │  Pack visible into dense array
          ▼
Stage 3:  MAP INTERSECTIONS    gaussian_map_intersections.wgsl :: map_main
          │                                                    :: clamp_count_main
          │  Scatter (tile_id, splat_id) pairs
          │  One entry per overlapped tile per splat
          │  Clamp total to intersection capacity
          ▼
Stage 4:  RADIX SORT           gaussian_dynamic_sort.wgsl :: sort_count_main
          │                                               :: sort_reduce_main
          │                                               :: sort_scan_main
          │                                               :: sort_scan_compose_main
          │                                               :: sort_scan_add_main
          │                                               :: sort_scatter_main
          │  4-bit radix sort by tile_id
          │  Groups all splats for the same tile together
          │  Multiple passes: ceil(log2(tile_count)) / 4
          ▼
Stage 5:  TILE OFFSETS         gaussian_tile_offsets.wgsl :: main
          │  Find [start, end) range per tile
          │  in the sorted intersection list
          ▼
Stage 6:  RASTERIZE            gaussian_raster_tiles.wgsl :: main
          │  One workgroup (256 threads) per 16×16 tile
          │  Load splats in batches into shared memory
          │  Per-pixel Gaussian evaluation + alpha blend
          │  Early termination when transmittance < 1e-4
          ▼
Stage 7:  COMPOSITE (viewer only)  gaussian_composite.wgsl
          │  Fullscreen triangle blit to Rerun viewport
          │  (Standalone path skips this — reads back texture directly)
          ▼
Output:   Raster texture (Rgba8Unorm) → PNG (CLI) or viewport (viewer)
```

## Shared GPU Types (`gsplat_core/gpu_types.rs`)

This is the **single source of truth** for all GPU pipeline definitions. Both the standalone renderer and the Rerun viewer import from here.

### Structs

| Struct | Size | Purpose |
|--------|------|---------|
| `ProjectUniformBuffer` | 160 B | Camera + viewport + SH config |
| `ScanUniformBuffer` | 16 B | Prefix sum params |
| `SortUniformBuffer` | 16 B | Radix sort shift/pass |
| `MapUniformBuffer` | 16 B | Tile mapping params |
| `RasterUniformBuffer` | 16 B | Tile bounds + image size |
| `TileProjectedSplat` | 64 B | Per-splat projected data |
| `DrawIndirectArgs` | 16 B | Indirect draw/count buffer |

### Shared Resources

| Resource | Function |
|----------|----------|
| `GpuBindGroupLayouts` | 12 bind group layouts for all pipeline stages |
| `GpuComputePipelines` | 13 compute pipelines from 5 WGSL shaders |
| `create_compute_bind_group_layouts()` | Creates all layouts from a `wgpu::Device` |
| `create_compute_pipelines()` | Creates all pipelines given layouts |
| `create_raster_texture()` | Shared texture creation with `extra_usage` param |
| `fill_project_uniform()` | Fills camera/SH uniforms |
| `fill_scan_uniform()` | Fills prefix sum uniforms |
| `fill_map_uniform()` | Fills tile mapping uniforms |

### Helper Functions

Buffer creation: `create_filled_buffer()`, `create_sized_buffer()`
Data packing: `pack_vec3s()`, `pack_quats()`, `pack_scales_opacity()`, `pack_rgb()`, `pack_sh_coefficients()`
Dispatch sizing: `dispatch_grid_1d()`, `dispatch_grid_for_workgroups()`, `calc_tile_bounds()`, `tile_count()`, `calc_raster_extent()`
Capacity: `next_capacity()`, `intersection_capacity_for_instances()`, `compaction_block_count()`, `next_block_capacity()`

## Component Contract (Python ↔ Rust)

Python and Rust agree on a custom `GaussianSplats3D` archetype:

| Component | Rerun Type | Shape | Description |
|---|---|---|---|
| `centers` | `Translation3D` | `[N, 3]` | World-space Gaussian positions |
| `quaternions` | `RotationQuat` | `[N, 4]` | Rotation quaternions (xyzw) |
| `scales` | `Scale3D` | `[N, 3]` | Per-axis scale factors |
| `opacities` | `Opacity` | `[N]` | Per-splat opacity [0, 1] |
| `colors` | `Color` | `[N]` | Base RGB from SH DC coefficient |
| `sh_coefficients` | `TensorData` | `[N, C, 3]` | Optional higher-order SH (degree 0-4) |

## Python Metrics (`gsplat_rust_renderer/metrics.py`)

| Metric | Implementation | Range | Direction |
|--------|---------------|-------|-----------|
| PSNR | NumPy MSE → dB | 0-100 dB | Higher = better |
| SSIM | Gaussian-windowed (11×11, σ=1.5) | 0-1 | Higher = better |
| LPIPS | VGG-based via PyTorch `lpips` package | 0-1 | Lower = better |

All metrics apply 8-bit roundtrip quantization to match Brush's eval convention.

## Key Constants (`gsplat_core/constants.rs`)

| Constant | Value | Purpose |
|---|---|---|
| `MAX_SPLATS_RENDERED` | 200,000 | Hard cap per frame |
| `TILE_WIDTH` | 16 px | Tile size for compute raster |
| `PARALLEL_SPLAT_THRESHOLD` | 16,384 | Switch to rayon parallel |
| `SIGMA_COVERAGE` | 3.0 | Standard deviations for bbox |
| `BRUSH_COVARIANCE_BLUR_PX` | 0.3 | Anti-aliasing blur (matches Brush) |
| `SH_C0` | 0.28209 | Zeroth SH coefficient |
| `PROJECT_WORKGROUP_SIZE` | 128 | Threads per project dispatch |
| `SORT_WORKGROUP_SIZE` | 256 | Threads per sort dispatch |
| `SORT_BIN_COUNT` | 16 | Radix sort bins (4-bit) |

## File Map

```
packages/gsplat-rust-renderer/
├── Cargo.toml                     # Lib + 2 binaries, feature-gated deps
├── src/
│   ├── lib.rs                     # Shared library root
│   ├── render_cli.rs              # gsplat-render binary (no Rerun)
│   ├── main.rs                    # gsplat-rust-renderer binary (Rerun viewer)
│   ├── gaussian_visualizer.rs     # Rerun VisualizerSystem (imports from gsplat_core)
│   ├── gaussian_renderer.rs       # Rerun Renderer trait (imports from gsplat_core)
│   ├── ply_loader.rs              # Rust PLY parser
│   ├── nerf_camera.rs             # NeRF transforms JSON parser
│   └── gsplat_core/               # ★ Core algorithm — zero Rerun deps ★
│       ├── mod.rs                 # Public API + re-exports
│       ├── types.rs               # Data structures + RenderOutput
│       ├── constants.rs           # Shared constants
│       ├── projection.rs          # 3D→2D Gaussian projection
│       ├── sh.rs                  # Spherical harmonics evaluation
│       ├── covariance.rs          # 2D covariance math
│       ├── culling.rs             # CPU frustum culling + depth sort
│       ├── camera.rs              # Camera constructors
│       ├── gpu_types.rs           # ★ GPU single source of truth ★
│       ├── gpu_context.rs         # Headless wgpu init
│       └── gpu_renderer.rs        # Standalone GPU compute pipeline
├── shader/                        # WGSL compute shaders (shared)
│   ├── gaussian_project.wgsl      # Stages 1-2: project + compact
│   ├── gaussian_map_intersections.wgsl  # Stage 3: tile assignment
│   ├── gaussian_dynamic_sort.wgsl      # Stage 4: radix sort
│   ├── gaussian_tile_offsets.wgsl      # Stage 5: tile ranges
│   ├── gaussian_raster_tiles.wgsl      # Stage 6: per-pixel rasterize
│   └── gaussian_composite.wgsl         # Stage 7: viewport blit (viewer only)
├── gsplat_rust_renderer/          # Python module
│   ├── __init__.py                # Beartype activation
│   ├── gaussians3d.py             # PLY loader + rr.AsComponents
│   └── metrics.py                 # PSNR + SSIM + LPIPS
├── tools/
│   └── log_gaussian_ply.py        # CLI: load PLY → log to viewer
├── tests/                         # Python tests
└── docs/
    ├── architecture.md            # This file
    └── testing-plan.md            # Eval pipeline docs
```

## Baseline Quality Metrics

GPU renderer, chair + hotdog (Brush-trained 30K steps, 4 test views each):

| Scene | PSNR (dB) | SSIM | LPIPS | Render time (RTX 5090) |
|-------|-----------|------|-------|----------------------|
| Chair (40K splats) | 29.05 | 0.9479 | 0.0612 | 7.4ms (135 FPS) |
| Hotdog (19K splats) | 26.65 | 0.9387 | 0.1175 | 7.5ms |

Reference: Brush MipNeRF-360 averages: ~29.7 PSNR, ~0.886 SSIM, ~0.196 LPIPS.
