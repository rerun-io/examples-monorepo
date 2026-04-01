# gsplat-rust-renderer

GPU-accelerated Gaussian Splatting viewer built as a custom Rerun visualizer. A Rust binary extends the stock Rerun viewer with a tile-based compute renderer (wgpu/WGSL), while a Python module handles data loading and logging via gRPC.

## Architecture

The system is a two-process design: a **Rust viewer** with a custom GPU pipeline and a **Python client** that parses PLY files and logs component batches over gRPC.

```
                        ┌─────────────────────────────────────────────────────────────┐
                        │                    Rust Viewer Process                       │
                        │                                                             │
  ┌──────────┐  gRPC   │  ┌─────────────┐    ┌──────────────────┐    ┌────────────┐  │
  │  Python   │────────►│  │ Rerun Data  │───►│  Gaussian Splat  │───►│  Gaussian  │  │
  │  Client   │  :9876  │  │   Store     │    │   Visualizer     │    │  Renderer  │  │
  │           │         │  │             │    │ (VisualizerSystem)│    │   (wgpu)   │  │
  └──────────┘         │  └─────────────┘    └──────────────────┘    └─────┬──────┘  │
       │                │                                                   │         │
       │                │                                              ┌────▼─────┐   │
  ┌────▼─────┐         │                                              │ Framebuf  │   │
  │ .ply file │         │                                              │ (display) │   │
  └──────────┘         │                                              └──────────┘   │
                        └─────────────────────────────────────────────────────────────┘
```

### Component Contract

Python and Rust agree on a custom `GaussianSplats3D` archetype with these components:

| Component | Type | Shape | Description |
|---|---|---|---|
| `centers` | `Translation3D` | `[N, 3]` | World-space Gaussian positions |
| `quaternions` | `RotationQuat` | `[N, 4]` | Rotation quaternions (xyzw) |
| `scales` | `Scale3D` | `[N, 3]` | Per-axis scale factors |
| `opacities` | `Opacity` | `[N]` | Per-splat opacity [0, 1] |
| `colors` | `Color` | `[N]` | Base RGB from SH DC coefficient |
| `sh_coefficients` | `TensorData` | `[N, C, 3]` | Optional higher-order SH (degree 0-4) |

### State Machine: Per-Frame Render Pipeline

Each frame, the visualizer queries the data store and drives the GPU renderer through these states:

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

### GPU Renderer Pipeline (Compute / Tile Path)

The renderer has two paths. The **compute tile path** (preferred) runs entirely on the GPU via WGSL compute shaders:

```
Sorted splat indices + cloud data
          │
          ▼
┌──────────────────┐   gaussian_project.wgsl
│  1. Project      │   Exact 3D→2D Gaussian projection
│     Splats       │   Covariance linearization, SH evaluation
└────────┬─────────┘
         │
         ▼
┌──────────────────┐   (prefix sum - scan blocks + block sums)
│  2. Compact      │   Parallel prefix scan on per-splat tile counts
│     Tile Counts  │   Produces tile-hit offsets for scatter
└────────┬─────────┘
         │
         ▼
┌──────────────────┐   gaussian_map_intersections.wgsl
│  3. Map          │   Scatter (splat, tile) pairs into flat buffer
│     Intersections│   Each visible splat × overlapped tiles
└────────┬─────────┘
         │
         ▼
┌──────────────────┐   gaussian_dynamic_sort.wgsl
│  4. Radix Sort   │   16-bin radix sort by depth within each tile
│     by Depth     │   Multi-pass: count → reduce → scan → scatter
└────────┬─────────┘
         │
         ▼
┌──────────────────┐   gaussian_tile_offsets.wgsl
│  5. Tile Offsets │   Binary search for per-tile start/end in
│                  │   the sorted intersection array
└────────┬─────────┘
         │
         ▼
┌──────────────────┐   gaussian_splat.wgsl
│  6. Tile Raster  │   Per-pixel alpha blending within each tile
│                  │   Front-to-back accumulation, early termination
└────────┬─────────┘
         │
         ▼
┌──────────────────┐   gaussian_composite.wgsl
│  7. Composite    │   Blit raster texture onto the Rerun viewport
│                  │   with depth and alpha compositing
└────────┘─────────┘
```

When compute shaders are unavailable, a **CPU fallback** projects splats into instanced quads rendered via the standard rasterization pipeline.

### Python Client Flow

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

## File Structure

```
gsplat-rust-renderer/
├── Cargo.toml                         # Rust crate: rerun 0.30.2 + re_* crates
├── Cargo.lock                         # Pinned Rust deps (committed for binary)
├── pyproject.toml                     # Python package metadata (hatchling)
├── src/
│   ├── main.rs                        # Viewer binary: gRPC listener + visualizer registration
│   ├── gaussian_visualizer.rs         # VisualizerSystem: query → cloud → cull → sort → submit
│   └── gaussian_renderer.rs           # GPU renderer: compute pipelines + CPU fallback
├── shader/
│   ├── gaussian_project.wgsl          # 3D→2D projection + SH evaluation
│   ├── gaussian_map_intersections.wgsl # Scatter (splat, tile) pairs
│   ├── gaussian_dynamic_sort.wgsl     # 16-bin radix sort by depth per tile
│   ├── gaussian_tile_offsets.wgsl     # Binary search for tile ranges
│   ├── gaussian_splat.wgsl            # Per-pixel tile rasterization
│   ├── gaussian_composite.wgsl        # Final composite blit
│   └── gaussian_raster_tiles.wgsl     # Tile grid utilities
├── gsplat_rust_renderer/
│   ├── __init__.py                    # Beartype activation
│   └── gaussians3d.py                 # Gaussians3D dataclass + PLY parser
├── tools/
│   └── log_gaussian_ply.py            # CLI: load PLY → log to viewer via gRPC
├── tests/
│   ├── test_import.py                 # Smoke test
│   └── test_gaussians3d.py            # Unit tests for PLY parsing + validation
└── examples/
    └── .gitkeep                       # PLY downloaded at runtime from HuggingFace
```

## Usage

```bash
# Build and launch the viewer (first build ~5-10 min, compiles Rerun from source)
pixi run -e gsplat-rust-renderer viewer

# In a second terminal, download example PLY and log it to the viewer
pixi run -e gsplat-rust-renderer log-ply

# Or log your own PLY file
pixi run -e gsplat-rust-renderer -- python tools/log_gaussian_ply.py /path/to/scene.ply

# Dev tasks
pixi run -e gsplat-rust-renderer-dev lint        # ruff
pixi run -e gsplat-rust-renderer-dev typecheck   # pyrefly
pixi run -e gsplat-rust-renderer-dev tests       # pytest
pixi run -e gsplat-rust-renderer fmt             # cargo fmt
pixi run -e gsplat-rust-renderer clippy          # cargo clippy
pixi run -e gsplat-rust-renderer rust-test       # cargo test
```

## Key Design Decisions

- **Two-process model**: The Rust viewer runs as a standalone native window. Python sends data over gRPC, keeping the Python side lightweight and the render loop free from GIL contention.
- **Custom Rerun visualizer**: Registered on the built-in `Spatial3DView` via `extend_view_class`. No fork of Rerun required.
- **Tile-based compute rendering**: Inspired by [Brush](https://github.com/ArthurBrussee/brush), the GPU path uses a multi-stage compute pipeline (project → compact → sort → raster → composite) instead of instanced quads, enabling higher splat counts at interactive framerates.
- **Cloud caching**: The visualizer caches `RenderGaussianCloud` per entity path, keyed by a signature of splat count + SH shape + transform. Clouds are only rebuilt when the data or transform changes.
- **SH evaluation on GPU**: Spherical harmonics up to degree 4 (25 coefficients/channel) are evaluated per-splat in the projection shader, enabling view-dependent color without CPU recomputation.
- **No CUDA dependency**: All GPU work uses wgpu (Vulkan/Metal backend), so the package does not require CUDA or the monorepo's `cuda` feature.
