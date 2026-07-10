# Architecture

Detailed internals of the gsplat-rust-renderer for developers who want to understand or modify the code. For usage, see the [README](../README.md).

## Design Principles

1. **`gsplat_core` is the single source of truth** — all algorithm code, GPU types, pipeline definitions, and math live in the Rerun-free `gsplat_core/` module
2. **GPU-only rendering, no CPU fallback** — follows the [Brush](https://github.com/ArthurBrussee/brush) approach
3. **Two rendering paths, shared pipeline** — the Rerun viewer and the standalone CLI both use the same WGSL shaders, bind group layouts, compute pipelines, and GPU buffer types
4. **Clean dependency boundary** — `gsplat_core/` depends only on `glam`, `wgpu`, `bytemuck`; the viewer adds `re_*` crates behind a feature flag

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         gsplat_core/ (Rerun-free)                          │
│                                                                             │
│  types.rs          Data structures: RenderGaussianCloud, CameraApproximation│
│  constants.rs      SH_C0, SIGMA_COVERAGE, BRUSH_COVARIANCE_BLUR_PX, etc.   │
│  projection.rs     Quaternion helpers                                      │
│  sh.rs             SH metadata (degree from coefficient count)            │
│  camera.rs         Camera constructors (look-at, NeRF transform, fallback) │
│  gpu_types.rs      GPU buffer structs, bind group layouts, compute         │
│                    pipelines, helpers — SINGLE SOURCE OF TRUTH              │
│  gpu_context.rs    Headless wgpu device/queue initialization               │
│  gpu_renderer.rs   Standalone GPU-only compute pipeline + readback         │
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
                  gsplat_core/gpu_context       gsplat_core/constants
                  gsplat_core/types             gsplat_core/sh
                  ply_loader                    gsplat_core/projection
                  nerf_camera
                                                gsplat_core/camera
```

Both rendering paths share:
- **5 shared WGSL compute shaders** (`gaussian_project`, `gaussian_dynamic_sort`, `gaussian_map_intersections`, `gaussian_tile_offsets`, `gaussian_raster_tiles`) embedded in `gsplat_core` — no Rerun-specific code. A 6th, `gaussian_composite.wgsl`, is viewer-only (blits to the Rerun viewport) and is **not** part of the shared core
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
                      ├─ Upload to GPU buffers
                      ├─ GPU-only compute pipeline (cull + sort on GPU)
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
                      ├─ Build or reuse RenderGaussianCloud (cached per entity)
                      ├─ Extract camera from view state
                      └─► GaussianDrawData::add_batch()  (full cloud — GPU culls + sorts)
                              │
                              ▼
                    gaussian_renderer.rs prepare_compute_batch()
                      ├─ Reuse/grow cached GPU buffers, write uniforms
                      ├─ GPU-only compute pipeline (same shaders!)
                      ├─ Composite to Rerun viewport
                      └─► Rerun draw phase
```

Built with default features (all `re_*` crates). Uses `re_renderer::RenderContext` for wgpu access.

## GPU Compute Pipeline (9 Stages)

Both paths execute the same GPU-only pipeline (the standalone path skips the final composite). There is no CPU pre-pass — the GPU is handed the full cloud each frame and culls and depth-sorts it itself. The shaders are the single source of truth — defined once in `shader/`, loaded via `include_str!()`.

```
Upload full splat cloud + uniforms to GPU (no CPU cull/sort)
          │
          ▼
Stage 1:  CULL + COMPACT       gaussian_project.wgsl :: project_forward_main
          │  Full-cloud GPU cull (near plane, opacity, finite
          │  projection, on-screen bbox)
          │  Survivors append (global_gid, depth bits) via
          │  atomicAdd(num_visible) — no visibility flags, no
          │  separate compaction pass
          ▼
Stage 2:  DEPTH ARGSORT        gaussian_dynamic_sort.wgsl :: sort_*_main
          │  Gid canonicalization sort first (atomicAdd compaction is
          │  racy; ties must resolve deterministically), then radix
          │  argsort ascending by f32 depth bits → front-to-back order
          ▼
Stage 3:  PROJECT              gaussian_project.wgsl :: project_visible_main
          │  Build 3D covariance (R·diag(s²)·Rᵀ)
          │  Project to 2D via camera Jacobian
          │  Evaluate SH for view-dependent color
          │  Compute tile bounding box + hit count
          ▼
Stage 4:  SCAN                 gaussian_project.wgsl :: scan_blocks_main
          │                                          :: scan_block_sums_main
          │  3-level prefix sum over per-splat tile hit counts
          │  (supports clouds beyond 262,144 splats)
          ▼
Stage 5:  MAP INTERSECTIONS    gaussian_map_intersections.wgsl :: map_main
          │                                                    :: clamp_count_main
          │  Scatter (tile_id, compact_gid) pairs
          │  One entry per overlapped tile per splat
          │  Clamp total to intersection capacity
          ▼
Stage 6:  TILE SORT            gaussian_dynamic_sort.wgsl :: sort_count_main
          │                                               :: sort_reduce_main
          │                    gaussian_project.wgsl      :: scan_blocks_main
          │                                               :: scan_block_sums_main
          │                    gaussian_dynamic_sort.wgsl :: sort_scan_compose_main
          │                                               :: sort_scan_add_main
          │                                               :: sort_scatter_main
          │  4-bit radix sort by tile_id; the prefix scan reuses
          │  scan_blocks_main/scan_block_sums_main from
          │  gaussian_project.wgsl (shared with Stage 4)
          │  Groups all splats for the same tile together
          │  Pass count: ceil(bits(tile_count) / 4)
          ▼
Stage 7:  TILE OFFSETS         gaussian_tile_offsets.wgsl :: main
          │  Find [start, end) range per tile
          │  in the sorted intersection list
          ▼
Stage 8:  RASTERIZE            gaussian_raster_tiles.wgsl :: main
          │  One workgroup (256 threads) per 16×16 tile
          │  Load splats in batches into shared memory
          │  Per-pixel Gaussian evaluation + alpha blend
          │  Early termination when transmittance < 1e-4
          ▼
Stage 9:  COMPOSITE (viewer only)  gaussian_composite.wgsl
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
| `ProjectUniformBuffer` | 192 B | Camera + viewport + SH config |
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

Python and Rust agree on the upstream `rerun.archetypes.Gaussians3D` schema
(logged as custom components on released rerun-sdk; the Rust visualizer is named
`Gaussians3D`). Descriptor `component` strings are `Gaussians3D:<field>`:

| Component | `component_type` | Arrow datatype | Description |
|---|---|---|---|
| `centers` | `Position3D` | `FixedSizeList<Float32,3>` | World-space Gaussian positions (required) |
| `scales` | `Scale3D` | `FixedSizeList<Float32,3>` | Per-axis scale factors |
| `quaternions` | `RotationQuat` | `FixedSizeList<Float32,4>` | Rotation quaternions (xyzw) |
| `colors` | `Color` | `UInt32` (0xRRGGBBAA) | RGB from SH DC term, opacity in alpha |
| `sh_coefficients` | `SphericalHarmonics3` | `FixedSizeList<Float16,45>` | Optional SH degrees 1-3, coefficient-major |
| `show_spherical_harmonics` | `ShowSphericalHarmonics` | `Bool` (single) | Optional; evaluate higher-order SH |

Every FixedSizeList child field is named `item` and non-nullable (the Rust
deserializer checks datatype equality including these).

## Python Metrics (`gsplat_rust_renderer/metrics.py`)

| Metric | Implementation | Range | Direction |
|--------|---------------|-------|-----------|
| PSNR | NumPy MSE → dB | 0-100 dB | Higher = better |
| SSIM | Gaussian-windowed (11×11, σ=1.5) | 0-1 | Higher = better |
| LPIPS | VGG-based via PyTorch `lpips` package | 0-1 | Lower = better |

All metrics apply 8-bit roundtrip quantization to match Brush's eval convention.

## Key Constants

Algorithm constants live in `gsplat_core/constants.rs`:

| Constant | Value | Purpose |
|---|---|---|
| `MIN_RADIUS_PX` | 0.35 px | Cull splats with sub-pixel projected radius |
| `OPACITY_SCALE` | 1.0 | Global opacity multiplier (1.0 = no change) |
| `SIGMA_COVERAGE` | 3.0 | Standard deviations for screen-space bbox (3σ ≈ 99.7%) |
| `SH_C0` | 0.28209 | Zeroth SH coefficient — DC color conversion |
| `BRUSH_COVARIANCE_BLUR_PX` | 0.3 | Anti-aliasing blur (matches Brush) |
| `BRUSH_VISIBILITY_ALPHA_THRESHOLD` | 1/255 | Min alpha for a splat to be visible |

GPU pipeline tuning constants live in `gsplat_core/gpu_types.rs`:

| Constant | Value | Purpose |
|---|---|---|
| `TILE_WIDTH` | 16 px | Tile size for compute raster (mirrored as a `const` in the WGSL shaders) |
| `PROJECT_WORKGROUP_SIZE` | 128 | Threads per project dispatch |
| `SORT_WORKGROUP_SIZE` | 256 | Threads per sort dispatch |
| `SORT_BIN_COUNT` | 16 | Radix sort bins (4-bit) |
| `INTERSECTION_CAPACITY_MULTIPLIER` | 32 | Per-instance tile-intersection buffer capacity (reverted from a brief 4× experiment) |

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
│       ├── projection.rs          # Quaternion helpers
│       ├── sh.rs                  # SH metadata (degree from coeff count)
│       ├── camera.rs              # Camera constructors
│       ├── gpu_types.rs           # ★ GPU single source of truth ★
│       ├── gpu_context.rs         # Headless wgpu init
│       └── gpu_renderer.rs        # Standalone GPU compute pipeline
├── shader/                        # WGSL compute shaders (shared)
│   ├── gaussian_project.wgsl      # Stages 1, 3, 4: cull, project, scan
│   ├── gaussian_map_intersections.wgsl  # Stage 5: tile assignment
│   ├── gaussian_dynamic_sort.wgsl      # Stages 2 + 6: depth / tile radix sort
│   ├── gaussian_tile_offsets.wgsl      # Stage 7: tile ranges
│   ├── gaussian_raster_tiles.wgsl      # Stage 8: per-pixel rasterize
│   └── gaussian_composite.wgsl         # Stage 9: viewport blit (viewer only)
├── gsplat_rust_renderer/          # Python module
│   ├── __init__.py                # Beartype activation
│   ├── gaussians3d.py             # PLY loader + rr.AsComponents
│   └── metrics.py                 # PSNR + SSIM + LPIPS
├── tools/
│   ├── log_gaussian_ply.py         # CLI: load PLY → log to viewer
│   ├── log_splats_with_cameras.py  # CLI: splats + cameras + GT images (tabs/pages)
│   ├── calibration_scene.py        # CLI: cross-renderer calibration scene
│   ├── relog_check.py              # Re-log staleness check
│   ├── visualize_brush_training.py # Overlay GPU splats on Brush training
│   └── run_brush_native_demo.sh    # Brush comparison demo driver
├── tests/                         # Python tests
└── docs/
    └── architecture.md            # This file (the only doc; see the PR for perf/accuracy evidence)
```
