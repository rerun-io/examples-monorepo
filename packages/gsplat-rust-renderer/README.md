# gsplat-rust-renderer

GPU-accelerated [Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) viewer built as a custom [Rerun](https://rerun.io) visualizer. A Rust binary extends the stock Rerun viewer with a tile-based GPU compute renderer (wgpu/WGSL/Vulkan), while a Python module parses PLY files and logs Rerun component batches to the viewer over gRPC. The CLIs use **tyro**, and **Pixi** provides one-command setup.

<p align="center">
  <a title="Rerun" href="https://rerun.io" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Rerun-0.33-0b82f9" alt="Rerun badge">
  </a>
  <a title="Pixi" href="https://pixi.sh/latest/" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Install%20with-Pixi-16A34A" alt="Pixi badge">
  </a>
  <a title="Rust" href="https://www.rust-lang.org/" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Rust-1.93-dea584" alt="Rust badge">
  </a>
</p>

## Installation

Install the [Pixi](https://pixi.sh/latest/#installation) package manager:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Restart your shell so the new `pixi` binary is on `PATH`.

Linux only. A Vulkan-capable GPU is required; **no CUDA is needed** (rendering runs on wgpu/Vulkan).

The viewer builds against the **published Rerun `=0.33.0` crates** from crates.io, so a clean checkout builds anywhere. The first build compiles those crates from source and takes a few minutes; subsequent runs are instant. Pixi also pins the `rerun-sdk==0.33.0` Python wheel and the system libraries the viewer links against (`libudev`, X11/Wayland), so no manual `PKG_CONFIG_PATH` setup is needed inside a Pixi shell.

## Quick Start

Two terminals — one for the viewer, one for logging data. Run from the repo root.

```bash
# Terminal 1: build and launch the Rust viewer (gRPC listener on 127.0.0.1:9876)
pixi run --frozen -e gsplat-rust-renderer gsplat-rust-renderer-viewer

# Terminal 2: download the example chair PLY (~36 MB, from HuggingFace) and log it
pixi run --frozen -e gsplat-rust-renderer gsplat-rust-renderer-log-ply
```

The `log-ply` task downloads `chair.ply` from
[pablovela5620/splat-dataset](https://huggingface.co/datasets/pablovela5620/splat-dataset)
on first run, then logs it to the running viewer.

### Available Pixi tasks

List them with `pixi task list -e gsplat-rust-renderer`. The public tasks (all prefixed `gsplat-rust-renderer-`):

| Task | Description |
|------|-------------|
| `gsplat-rust-renderer-viewer` | Launch the Rust viewer (`cargo run --release --bin gsplat-rust-renderer`). |
| `gsplat-rust-renderer-log-ply` | Download the example PLY and log it to the running viewer. |
| `gsplat-rust-renderer-render` | Render a PLY at a NeRF camera pose to PNG with the standalone `gsplat-render` binary — no Rerun (`--no-default-features`). |
| `gsplat-rust-renderer-brush-train-{chair,hotdog,lego,train,truck}` | Train splats on a scene with `brush_app` for 30K steps and export a PLY to `data/trained/`. Each depends on its dataset-download task. |
| `gsplat-rust-renderer-fmt` / `-clippy` / `-rust-test` | `cargo fmt --all` / `cargo clippy --all-targets -- -D warnings` / `cargo test`. |

Dev-env tasks live under the monorepo-wide Python tooling; from a `gsplat-rust-renderer-dev` shell use `ruff check .`, `pyrefly check .`, and `pytest -q` directly.

## Training with live splats

`tools/run_brush_native_demo.sh` trains a scene with [brush](https://github.com/ArthurBrussee/brush) and streams **real GPU splats** into brush's own rich training blueprint (loss/lr/psnr/ssim/splat-count/memory time-series and eval-view tabs).

```bash
tools/run_brush_native_demo.sh DATA_DIR [TOTAL_ITERS] [EXPORT_DIR]
# e.g.
tools/run_brush_native_demo.sh data/nerf-synthetic/lego 30000
```

`TOTAL_ITERS` defaults to `30000`; `EXPORT_DIR` defaults to `/tmp/brush-runs/<scene>`.

How it works:

1. **Start the viewer headless first** so it owns the gRPC store on `:9876`:
   ```bash
   packages/gsplat-rust-renderer/target/release/gsplat-rust-renderer --headless &
   ```
   The script warns if nothing is listening on `127.0.0.1:9876`.
2. The script picks a fixed, shared recording id and launches `brush-cli` (path set via `BRUSH_CLI`, default `/home/pablo/0Dev/work/brush/target/release/brush-cli`) with `--rerun-enabled`. brush trains, logs its full dashboard, and sends its blueprint, pinned to that recording id via `BRUSH_RERUN_RECORDING_ID`.
3. Once brush starts training, the sidecar `tools/visualize_brush_training.py --brush-native` joins the **same** recording id and does the two things brush can't: it overlays a `GaussianSplats3D` snapshot at `world/splats` per exported `export_NNNNN.ply` (on brush's `iterations` timeline) and re-sends brush's blueprint with a visualizer override pinning `world/splats` to the custom `GaussianSplats3D` visualizer.

Result: brush's exact training dashboard, but with sharp GPU splats in the Scene view instead of fuzzy ellipsoids. Optional env knobs: `EXPORT_EVERY` (200), `EVAL_EVERY` (500), `EVAL_SPLIT_EVERY` (0 = off).

## Logging your own PLY from Python

The `log-ply` task wraps `tools/log_gaussian_ply.py`. Point it at your own file:

```bash
pixi run --frozen -e gsplat-rust-renderer -- python tools/log_gaussian_ply.py \
    --rr-config.connect \
    --rr-config.application-id gsplat-rust-renderer \
    --ply-path /path/to/your/scene.ply
```

The `--rr-config` flags come from [simplecv's `RerunTyroConfig`](https://github.com/pablovela5620/simplecv) and support the standard Rerun output modes:

| Flag | Description |
|------|-------------|
| `--rr-config.connect` | Send to the running Rust viewer on `:9876` (used by the `log-ply` task). |
| `--rr-config.save output.rrd` | Save to an RRD file instead of viewing. |
| `--rr-config.serve` | Launch a web viewer + gRPC server. |
| `--rr-config.recording-id <str>` | Join an existing recording (string id, e.g. to share a store with brush). |
| (no flag) | Spawn the stock Rerun viewer (no custom Gaussian rendering). |

Or do it directly from your own code:

```python
import rerun as rr
from gsplat_rust_renderer.gaussians3d import Gaussians3D

# Parse a PLY into the Gaussians3D dataclass (implements rr.AsComponents).
gaussians = Gaussians3D.from_ply("scene.ply")

# Connect to the running Rust viewer and log the splats.
rr.init("my-app", spawn=False)
rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")
rr.log("world/splats", gaussians, static=True)
```

## MCP viewport control (debug-only)

The shipped viewer uses the published Rerun `0.33` crates. The MCP viewport-control demo (driving the viewer's camera/blueprint over MCP) was developed against a local `0.34.0-alpha` Rerun checkout plus the [`lucasmerlin/egui` `kittest-mcp-patch`](https://github.com/lucasmerlin/egui) fork. That setup is **not** shipped because absolute path deps and git-branch patches are not reproducible. To reproduce it locally, repoint the viewer crates in `Cargo.toml` at your rerun checkout (path deps) and mirror its egui-family `[patch.crates-io]` entries (use branch form, not `rev=`; pin revisions with `cargo update -p`). The forked `eframe` moves present-mode/frame-latency into `WgpuConfiguration.surface: SurfaceConfig` (egui 0.35), so `native_options()` in `src/main.rs` needs the `SurfaceConfig::HIGH_THROUGHPUT` form instead of the 0.33 `present_mode`/`on_surface_status` fields, and `re_viewer` pulls `re_gamepad`/`gilrs` (needs `libudev`).

## Project Structure

```
gsplat-rust-renderer/
├── Cargo.toml                          # Rust crate: published rerun =0.33.0 crates (viewer feature) + wgpu core
├── Cargo.lock                          # Pinned Rust deps (committed)
├── pyproject.toml                      # Python package metadata (hatchling)
├── src/
│   ├── main.rs                         # Viewer binary: gRPC listener on :9876 + visualizer registration (--headless)
│   ├── render_cli.rs                   # `gsplat-render` binary: render a PLY to PNG, no Rerun
│   ├── lib.rs                          # `gsplat_lib` shared library crate
│   ├── gaussian_visualizer.rs          # VisualizerSystem: query → build/reuse splat cloud → submit
│   ├── gaussian_renderer.rs            # Viewer-side GPU renderer: compute pipelines + viewport composite
│   ├── nerf_camera.rs                  # NeRF transforms.json camera parsing (render CLI)
│   ├── ply_loader.rs                   # Rust PLY parsing
│   └── gsplat_core/                    # Shared GPU core (gpu_renderer, gpu_types, projection, sh, camera, constants)
├── shader/                             # WGSL compute shaders for the tile-based pipeline
│   ├── gaussian_project.wgsl           #   GPU cull/compact, projection + SH, prefix scan
│   ├── gaussian_dynamic_sort.wgsl      #   radix sort (depth argsort, tile-id sort)
│   ├── gaussian_map_intersections.wgsl #   scatter (tile, splat) pairs
│   ├── gaussian_tile_offsets.wgsl      #   per-tile [start, end) ranges
│   ├── gaussian_raster_tiles.wgsl      #   per-pixel alpha blending per tile
│   └── gaussian_composite.wgsl         #   blit raster texture to the viewport
├── gsplat_rust_renderer/               # Python package
│   ├── __init__.py                     # Beartype activation (dev env only)
│   ├── gaussians3d.py                  # Gaussians3D dataclass + PLY parser (rr.AsComponents)
│   └── metrics.py                      # LPIPS / PSNR / SSIM helpers
├── tools/
│   ├── log_gaussian_ply.py             # CLI: load a PLY → log to viewer (tyro + RerunTyroConfig)
│   ├── run_brush_native_demo.sh        # Train with brush + overlay live GPU splats on its blueprint
│   ├── visualize_brush_training.py     # Sidecar joined by the demo script (--brush-native)
│   ├── log_splats_with_cameras.py      # Log splats alongside dataset cameras
│   ├── calibration_scene.py            # Calibration / debug scene logger
│   └── relog_check.py                  # Re-log / round-trip sanity check
├── tests/                              # test_gaussians3d.py, test_metrics.py, test_import.py
├── docs/architecture.md                # Per-frame pipeline, GPU stages, component contract
└── examples/                           # Example PLY downloaded here at runtime
```

## Architecture

Two-process design: a **Rust viewer** with a custom GPU pipeline and a **Python client** that parses PLY files and logs Rerun component batches over gRPC. The GPU renderer uses a tile-based, GPU-only compute pipeline inspired by [Brush](https://github.com/ArthurBrussee/brush) — cull + compact, depth argsort, projection + SH evaluation, prefix scan, intersection mapping, tile radix sort, tile offsets, per-tile raster, and composite. No CUDA is required.

For detailed internals (per-frame pipeline, GPU stages, component contract, buffer management, constants), see **[docs/architecture.md](docs/architecture.md)**.

## Acknowledgements

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — Kerbl et al., SIGGRAPH 2023
- [Brush](https://github.com/ArthurBrussee/brush) — tile-based compute renderer that inspired the GPU pipeline
- [Rerun](https://rerun.io) — visualization framework and custom visualizer API