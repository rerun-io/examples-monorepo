# gsplat-rust-renderer

`gsplat-rust-renderer` adds a tile-based, GPU compute Gaussian-splat visualizer to the [Rerun](https://rerun.io) desktop viewer. Python logs Rerun 0.38.1's native `GaussianSplats3D` archetype; Rust renders it with wgpu on Metal or Vulkan. The same GPU core also powers a standalone PNG renderer, so no CUDA is required.

<p align="center">
  <a title="Rerun" href="https://rerun.io" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Rerun-0.38.1-0b82f9" alt="Rerun badge">
  </a>
  <a title="Pixi" href="https://pixi.sh/latest/" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Install%20with-Pixi-16A34A" alt="Pixi badge">
  </a>
  <a title="Rust" href="https://www.rust-lang.org/" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Rust-1.98-dea584" alt="Rust badge">
  </a>
</p>

<p align="center">
</p>


## Requirements

- [Pixi](https://pixi.sh/latest/#installation)
- Apple Silicon with Metal, or Linux with Vulkan
- Enough local storage for build artifacts, datasets, and recordings

Run every command below from the repository root. Rust and Python Rerun packages are pinned together at `0.38.1`; do not update one side independently.

## Install and build

Install the development environment and build the custom viewer:

```bash
pixi install -e gsplat-rust-renderer-dev --frozen
pixi run -e gsplat-rust-renderer-dev --frozen cargo build --release --bin gsplat-rust-renderer --manifest-path packages/gsplat-rust-renderer/Cargo.toml
```

## Download the Lego example

Download the NeRF-synthetic dataset and pretrained 3DGS-MCMC checkpoint. Both commands are idempotent.

```bash
pixi run -e gsplat-rust-renderer-dev --frozen python -m gsplat_rust_renderer.nerfbaselines data lego
pixi run -e gsplat-rust-renderer-dev --frozen python -m gsplat_rust_renderer.nerfbaselines pretrained lego
```

## Quickstart: view a pretrained splat

Start the custom viewer in one terminal:

```bash
pixi run -e gsplat-rust-renderer-dev --frozen gsplat-rust-renderer-viewer
```

Then log the pretrained PLY, all train/test cameras, and their ground-truth image planes from a second terminal:

```bash
pixi run -e gsplat-rust-renderer-dev --frozen gsplat-rust-renderer-log-scene
```

The viewer listens on `127.0.0.1:9876`. Recordings need no visualizer override: this viewer selects `ComputeGaussianSplats3D` automatically, while stock Rerun 0.38.1 draws native splats. Use `--compute` only for an explicit compute selection or render-mode override. See the [viewer registration notes](crates/gsplat-viewer/README.md).

<p align="center">
  <img src="docs/media/pretrained-lego.png" width="560" alt="Pretrained Lego checkpoint prediction">
</p>

The image above is a real prediction bundled with the downloaded checkpoint.

## Brush training

The native trainer runs pinned Brush in process. See [Training recordings](docs/architecture.md#training-recordings) for the logging contract and dashboard layout.

```bash
# Prepare the local seed-42 initialization (training also runs this automatically).
pixi run -e gsplat-rust-renderer --frozen gsplat-rust-renderer-prepare-nerf-init lego
# Scene, step count, and mode; modes are record, live, and video.
pixi run -e gsplat-rust-renderer --frozen gsplat-rust-renderer-train lego 30000 record
pixi run -e gsplat-rust-renderer --frozen gsplat-rust-renderer-train lego 7000 video
```

Scenes include the eight Blender scenes, Truck, and Train. Outputs go to
`data/brush-runs/<scene>/<iterations>/<mode>/` inside the package. Each 30k run
exports a 7k checkpoint and its final model. Live mode connects to a viewer
already listening on port 9876. Pass extra Brush flags after `--`.

Blender inputs are derived under `data/nerf-synthetic-init/`; set
`GSPLAT_NERF_INIT_ROOT` only to use an existing derived dataset elsewhere.
Both trainers must read the same initialized dataset for comparisons.

```bash
# Build the pinned upstream reference locally in .brush/bin/brush-cli.
pixi run -e gsplat-rust-renderer --frozen gsplat-brush-cli-build
# Score all test cameras from unclipped float renders using the bench library.
pixi run -e gsplat-rust-renderer --frozen gsplat-train-score \
  --ply /path/to/export_7000.ply --dataset /path/to/lego --out /path/to/score.json
```

`gsplat-train-build` builds the native binary; `target/release/gsplat-train --help`
lists Brush options. With no `--save`, `--connect`, or `--spawn`, all logging is off.

## Full-split PSNR/SSIM evaluation

First validate the metric implementation against the eight downloaded checkpoints; then render and score every 200-image Blender test split:

```bash
pixi run -e gsplat-rust-renderer-dev --frozen gsplat-rust-renderer-evaluate-checkpoints
pixi run -e gsplat-rust-renderer-dev --frozen gsplat-rust-renderer-evaluate
```

The second task reuses one standalone GPU process per scene and writes `packages/gsplat-rust-renderer/data/evaluation/metrics.json`. For reference, the downloaded Lego checkpoint reports PSNR `35.74852`, SSIM `0.98415`, and LPIPS `0.01062` across 200 test images.

## Development

Run the package gates from the repository root:

```bash
CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0 \
  pixi run -e gsplat-rust-renderer-dev --frozen gate
```

The integration lane includes a 200-step training recording check. `GSPLAT_LEGO`
can override its default initialized dataset at
`packages/gsplat-rust-renderer/data/nerf-synthetic-init/lego`. The golden lane compares the
pretrained Lego conversion against Rerun’s PLY loader; `GSPLAT_TEST_PLY` selects
the checkpoint. Missing assets produce explicit skips.

## Architecture

The system has two front ends—Rerun viewer and standalone renderer—over one GPU pipeline. See [docs/architecture.md](docs/architecture.md) for the wire contract, camera/cache lifecycle, compute stages, training-recording layouts, and file map.

## Acknowledgements

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — Kerbl et al., SIGGRAPH 2023
- [Brush](https://github.com/ArthurBrussee/brush) — the tile-based compute renderer and trainer that inspired this pipeline
- [Rerun](https://rerun.io) — the data model, viewer, blueprints, and custom-visualizer API
