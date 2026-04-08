# gsplat-rust-renderer

GPU-accelerated [Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) viewer built as a custom [Rerun](https://rerun.io) visualizer. A Rust binary extends the stock Rerun viewer with a tile-based compute renderer (wgpu/WGSL), while a Python module handles data loading and logging via gRPC. Uses **tyro** for the CLI and **Pixi** for one-command setup.

<p align="center">
  <a title="Rerun" href="https://rerun.io" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Rerun-0.30%2B-0b82f9" alt="Rerun badge">
  </a>
  <a title="Pixi" href="https://pixi.sh/latest/" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Install%20with-Pixi-16A34A" alt="Pixi badge">
  </a>
  <a title="Rust" href="https://www.rust-lang.org/" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Rust-1.93-dea584" alt="Rust badge">
  </a>
</p>

## Installation

Make sure you have the [Pixi](https://pixi.sh/latest/#installation) package manager installed.

TL;DR install Pixi:
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```
Restart your shell so the new `pixi` binary is on `PATH`.

This is Linux only (Vulkan GPU required, no CUDA needed).

## Quick Start

Two terminals — one for the viewer, one for logging data:

```bash
# Terminal 1: Build and launch the viewer
pixi run --frozen -e gsplat-rust-renderer viewer

# Terminal 2: Download example chair PLY and log it to the viewer
pixi run --frozen -e gsplat-rust-renderer log-ply
```

The first build compiles Rerun from source and takes a few minutes. Subsequent runs are instant.

On the first `log-ply` run, an example PLY (~36 MB) is automatically downloaded from [HuggingFace](https://huggingface.co/datasets/pablovela5620/splat-dataset).

## Usage

### Log your own PLY file

```bash
pixi run --frozen -e gsplat-rust-renderer -- python tools/log_gaussian_ply.py \
    --rr-config.connect \
    --rr-config.application-id gsplat-rust-renderer \
    --ply-path /path/to/your/scene.ply
```

The `--rr-config` flags come from [simplecv's `RerunTyroConfig`](https://github.com/pablovela5620/simplecv) and support all standard Rerun output modes:

| Flag | Description |
|------|-------------|
| `--rr-config.connect` | Send to the running Rust viewer on `:9876` (default for `log-ply`) |
| `--rr-config.save output.rrd` | Save to an RRD file instead of viewing |
| `--rr-config.serve` | Launch a web viewer + gRPC server |
| (no flag) | Spawn the stock Rerun viewer (no custom Gaussian rendering) |

### Log from your own Python code

```python
import rerun as rr
from gsplat_rust_renderer.gaussians3d import Gaussians3D

# Load a PLY file into the Gaussians3D dataclass
gaussians = Gaussians3D.from_ply("scene.ply")

# Connect to the running Rust viewer
rr.init("my-app", spawn=False)
rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")

# Log the splats — Gaussians3D implements rr.AsComponents
rr.log("world/splats", gaussians, static=True)
```

### Available tasks

All commands can be listed with `pixi task list -e gsplat-rust-renderer`.

| Task | Description |
|------|-------------|
| `viewer` | Launch the Rerun viewer with Gaussian splat visualization |
| `log-ply` | Download example PLY and log it to the running viewer |
| `fmt` | Format Rust code (`cargo fmt`) |
| `clippy` | Lint Rust code (`cargo clippy`) |
| `rust-test` | Run Rust test suite (`cargo test`) |

Dev tasks (use `-e gsplat-rust-renderer-dev`):

| Task | Description |
|------|-------------|
| `lint` | Lint Python code (`ruff check`) |
| `typecheck` | Typecheck Python code (`pyrefly`) |
| `tests` | Run Python tests (`pytest`) |

## Project Structure

```
gsplat-rust-renderer/
├── Cargo.toml                          # Rust crate: rerun 0.31.1 + re_* crates
├── Cargo.lock                          # Pinned Rust deps (committed for binary)
├── pyproject.toml                      # Python package metadata (hatchling)
├── src/
│   ├── main.rs                         # Viewer binary: gRPC listener + visualizer registration
│   ├── gaussian_visualizer.rs          # VisualizerSystem: query → cloud → cull → sort → submit
│   └── gaussian_renderer.rs            # GPU renderer: compute pipelines + viewport composite
├── shader/                             # WGSL compute shaders (7-stage pipeline)
│   ├── gaussian_project.wgsl           # Stage 1: 3D→2D projection + SH evaluation
│   ├── gaussian_map_intersections.wgsl # Stage 3: scatter (splat, tile) pairs
│   ├── gaussian_dynamic_sort.wgsl      # Stage 4: radix sort by tile ID
│   ├── gaussian_tile_offsets.wgsl      # Stage 5: find per-tile [start, end) ranges
│   ├── gaussian_raster_tiles.wgsl      # Stage 6: per-pixel alpha blending per tile
│   └── gaussian_composite.wgsl         # Stage 7: blit raster texture to viewport
├── gsplat_rust_renderer/               # Python package
│   ├── __init__.py                     # Beartype activation (dev env only)
│   └── gaussians3d.py                  # Gaussians3D dataclass + PLY parser
├── tools/
│   └── log_gaussian_ply.py             # CLI: load PLY → log to viewer (tyro + RerunTyroConfig)
├── tests/
│   ├── test_import.py                  # Smoke test
│   └── test_gaussians3d.py             # Unit tests for PLY parsing + validation
└── examples/
    └── .gitkeep                        # PLY downloaded at runtime from HuggingFace
```

## Architecture

Two-process design: a **Rust viewer** with a custom GPU pipeline and a **Python client** that parses PLY files and logs Rerun component batches over gRPC. The GPU renderer uses a 7-stage compute pipeline inspired by [Brush](https://github.com/ArthurBrussee/brush) — project, compact, map intersections, radix sort, tile offsets, tile raster, composite. No CUDA required (uses wgpu/Vulkan).

For detailed internals (per-frame pipeline, GPU stages, component contract, buffer management, constants), see **[docs/architecture.md](docs/architecture.md)**.

## Acknowledgements

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — Kerbl et al., SIGGRAPH 2023
- [Brush](https://github.com/ArthurBrussee/brush) — tile-based compute renderer that inspired the GPU pipeline
- [Rerun](https://rerun.io) — visualization framework and custom visualizer API
