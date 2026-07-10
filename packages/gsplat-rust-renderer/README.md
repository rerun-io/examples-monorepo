# gsplat-rust-renderer

GPU-accelerated [Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) viewer built as a custom [Rerun](https://rerun.io) visualizer. A Rust binary extends the stock Rerun viewer with a tile-based GPU compute renderer (wgpu/WGSL/Metal or Vulkan), while a Python module parses PLY files and logs Rerun component batches to the viewer over gRPC. The CLIs use **tyro**, and **Pixi** provides one-command setup.

<p align="center">
  <a title="Rerun" href="https://rerun.io" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Rerun-0.34.1-0b82f9" alt="Rerun badge">
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

An Apple Silicon Metal GPU or Linux Vulkan GPU is required; **no CUDA is needed**.

The viewer builds against the **published Rerun `=0.34.1` crates** from crates.io. Pixi pins the matching `rerun-sdk==0.34.1` Python wheel; the wire contract, Rust crates, Python SDK, and viewer must remain on that exact version.

## Quick Start

Two terminals — one for the viewer, one for logging data. Run from the repo root.

```bash
# Terminal 1: build and launch the Rust viewer (gRPC listener on 127.0.0.1:9876)
pixi run --frozen -e gsplat-rust-renderer gsplat-rust-renderer-viewer

# Terminal 2: download the pretrained lego PLY (nerfbaselines) and log it
pixi run --frozen -e gsplat-rust-renderer gsplat-rust-renderer-log-ply
```

The `log-ply` task downloads the pretrained `lego` 3dgs-mcmc model from
[nerfbaselines/nerfbaselines](https://huggingface.co/nerfbaselines/nerfbaselines)
(`3dgs-mcmc/blender/lego.zip`) on first run, extracts its INRIA-layout
`point_cloud.ply`, and logs it to the running viewer under `/world/splats` at
`frame=0`. `gsplat-rust-renderer-log-scene` additionally logs the dataset
cameras (from [nerfbaselines/nerfbaselines-data](https://huggingface.co/datasets/nerfbaselines/nerfbaselines-data))
as Pinhole frustums with their GT images.

### Available Pixi tasks

List them with `pixi task list -e gsplat-rust-renderer`. The public tasks (all prefixed `gsplat-rust-renderer-`):

| Task | Description |
|------|-------------|
| `gsplat-rust-renderer-viewer` | Launch the Rust viewer (`cargo run --release --bin gsplat-rust-renderer`). |
| `gsplat-rust-renderer-log-ply` | Download the pretrained lego PLY and log it to the running viewer. |
| `gsplat-rust-renderer-log-scene` | Download the pretrained lego PLY + dataset and log the splat alongside its cameras. |
| `gsplat-rust-renderer-render` | Render a PLY at a NeRF camera pose to PNG with the standalone `gsplat-render` binary — no Rerun (`--no-default-features`). |
| `gsplat-rust-renderer-evaluate-checkpoints` | Recompute PSNR/SSIM over all eight bundled 200-image checkpoint prediction splits and compare with each `results.json`. |
| `gsplat-rust-renderer-evaluate` | Render all eight 200-image splits with one persistent standalone process per scene, then write `data/evaluation/metrics.json`. |
| `gsplat-rust-renderer-brush-train-lego-mac` | Install checksummed Brush v0.3.0 locally and run the raw Lego Metal trainer, exporting PLY every 50 iterations. This train-only task retains every PLY. |
| `gsplat-rust-renderer-brush-replay-lego` | Build the pinned pure-CLI metrics variant of Brush, then train Lego while saving a disk-bounded rich Rerun 0.34.1 replay with cameras, metrics, at most 31 splat snapshots, and only the final PLY retained. |
| `gsplat-rust-renderer-brush-train-{chair,hotdog,lego,train,truck}` | Train splats on a scene with `brush_app` for 30K steps and export a PLY to `data/trained/`. Each depends on its dataset-download task. |
| `gsplat-rust-renderer-fmt` / `-clippy` / `-rust-test` | `cargo fmt --all` / `cargo clippy --all-targets -- -D warnings` / `cargo test`. |

Dev-env tasks live under the monorepo-wide Python tooling; from a `gsplat-rust-renderer-dev` shell use `ruff check .`, `pyrefly check .`, and `pytest -q` directly.

## Full-split quality guard

The standalone renderer can render one frame with `--output`, or every frame in a NeRF transforms file with `--output-dir`. Full-split mode initializes Metal/Vulkan, uploads the PLY, and allocates renderer scratch buffers once, then reuses them across all 200 cameras. The Python harness first validates its metric implementation against each checkpoint's own `predictions/color` and `predictions/gt-color`, then applies the same strict relative-path pairing and per-image PSNR/SSIM averaging to standalone output.

```bash
pixi run -e gsplat-rust-renderer-dev --frozen gsplat-rust-renderer-evaluate-checkpoints
pixi run -e gsplat-rust-renderer-dev --frozen gsplat-rust-renderer-evaluate
```

## Brush training replay on macOS

Brush v0.3.0 has no osx-arm64 conda package in the configured channels. The raw-trainer task uses upstream's official Apple Silicon archive: `_gsplat-rust-renderer-install-brush-mac` downloads it into this package (never globally) and verifies SHA-256 `65b2631398c839be3c1d4d7160fe2326389dec87830aac0710985e6690a1048c`. The release corresponds to commit `3edecbb2fe79d3e2c87eeab85b15e0b1dd10d486`.

Brush v0.3.0's pure CLI emits eval PSNR/SSIM and refine counts, but its `TrainStep` arm discards loss. The replay task therefore builds the exact same pinned revision locally with `cargo --release --locked` and applies [`patches/brush-v0.3.0-cli-loss.patch`](patches/brush-v0.3.0-cli-loss.patch). That one-file patch only prints the already-computed loss at Brush's 50-step statistics cadence; it does not change training. Source and Cargo target trees are deleted after the project-local binary is copied, avoiding a persistent ~2.3 GB build tree.

Brush is still used only as a trainer: neither `--with-viewer` nor `--rerun-enabled` is passed, so its embedded Rerun 0.24 never creates a recording. It exports a PLY every 50 iterations and evaluates every 1,000. Our separate logger, running with this workspace's Rerun 0.34.1, writes `world/splats` plus loss/eval/splat-count scalars on the plural `iterations` timeline. It also logs all 100 Lego training cameras as static `rr.Pinhole` + `rr.Transform3D` frusta under `world/cameras/train_###`, with 160-pixel, JPEG-quality-80 GT image planes. The collapsed-panel blueprint keeps a 0.2-rad/s spinning 3D view dominant and shows one GT image plus the metric curves in a side column.

```bash
pixi run -e gsplat-rust-renderer-dev --frozen gsplat-rust-renderer-brush-replay-lego
```

This task starts Brush and the 0.34.1 logger concurrently. Retention is deliberately independent of export frequency: keep iteration 50, exact 1,000-step boundaries, and the final iteration; retain full geometry in every saved snapshot, but higher-order SH only in the final snapshot. A 30k run therefore has at most 31 RRD snapshots. Conservatively assuming one million splats at every snapshot, aligned geometry/color payloads, final float16 SH, and 15% RRD framing overhead gives 1.82 GB; Lego's 325k cap is estimated below 600 MB. The measured 30k Lego replay is 48,646,512 bytes (46.4 MiB), only 764,816 bytes (0.729 MiB) larger than the earlier splat-only full run after adding all 100 JPEG cameras and four metric series, and just 2.43% of the 2 GB cap. After each stable PLY has been logged or intentionally skipped, the logger deletes that intermediate; only the final PLY remains beside `training.rrd`. This prevents 600 uncompressed 50-step exports from accumulating on disk. A trainer-done sentinel makes the logger wait until final stdout is flushed, preventing the final metrics from racing the last PLY.

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
│   ├── nerfbaselines.py                # nerfbaselines data layout + zip extraction
│   ├── scene_io.py                     # NeRF-synthetic / COLMAP camera loaders
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
└── data/nerfbaselines/                 # Datasets + pretrained PLYs downloaded here at runtime
```

## Architecture

Two-process design: a **Rust viewer** with a custom GPU pipeline and a **Python client** that parses PLY files and logs Rerun component batches over gRPC. The GPU renderer uses a tile-based, GPU-only compute pipeline inspired by [Brush](https://github.com/ArthurBrussee/brush) — cull + compact, depth argsort, projection + SH evaluation, prefix scan, intersection mapping, tile radix sort, tile offsets, per-tile raster, and composite. No CUDA is required.

For detailed internals (per-frame pipeline, GPU stages, component contract, buffer management, constants), see **[docs/architecture.md](docs/architecture.md)**.

## Acknowledgements

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — Kerbl et al., SIGGRAPH 2023
- [Brush](https://github.com/ArthurBrussee/brush) — tile-based compute renderer that inspired the GPU pipeline
- [Rerun](https://rerun.io) — visualization framework and custom visualizer API
